# -*- coding: utf-8 -*-

"""
This file contains a dummy hardware module for sampling data at a constant rate, such as for
continuous wave ODMR for example.

Copyright (c) 2021, the qudi developers. See the AUTHORS.md file at the top-level directory of this
distribution and on <https://github.com/Ulm-IQO/qudi-iqo-modules/>

This file is part of qudi.

Qudi is free software: you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

Qudi is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with qudi.
If not, see <https://www.gnu.org/licenses/>.
"""

import time
import numpy as np
from enum import Enum
from qudi.interface.finite_sampling_input_interface import FiniteSamplingInputInterface
from qudi.interface.finite_sampling_input_interface import FiniteSamplingInputConstraints
from qudi.util.mutex import RecursiveMutex
from qudi.core.configoption import ConfigOption


class SimulationMode(Enum):
    RANDOM = 0
    ODMR = 1


class FiniteSamplingInputDummy(FiniteSamplingInputInterface):
    """
    This file contains a dummy hardware module for sampling data at a constant rate, such as for
    continuous wave ODMR for example.

    example config for copy-paste:

    finite_sampling_input_dummy:
        module.Class: 'dummy.finite_sampling_input_dummy.FiniteSamplingInputDummy'
        options:
            simulation_mode: 'ODMR'
            sample_rate_limits: [1, 1e6]  # optional, default [1, 1e6]
            frame_size_limits: [1, 1e9]  # optional, default [1, 1e9]
            channel_units:
                'APD counts': 'c/s'
                'Photodiode': 'V'
            use_nv_simulator: True  # optional, whether to use the NV simulator
            magnetic_field: [0, 0, 500]  # optional, magnetic field in Gauss [x, y, z]
            temperature: 300  # optional, temperature in Kelvin
    """

    _sample_rate_limits = ConfigOption(name='sample_rate_limits', default=(1, 1e6))
    _frame_size_limits = ConfigOption(name='frame_size_limits', default=(1, 1e9))
    _channel_units = ConfigOption(name='channel_units',
                                  default={'APD counts': 'c/s', 'Photodiode': 'V'})
    _simulation_mode = ConfigOption(name='simulation_mode',
                                    default='ODMR',
                                    constructor=lambda x: SimulationMode[x.upper()])
    _use_nv_simulator = ConfigOption(name='use_nv_simulator', default=False)
    _magnetic_field = ConfigOption(name='magnetic_field', default=[0, 0, 500])  # Gauss
    _temperature = ConfigOption(name='temperature', default=300)  # Kelvin

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._thread_lock = RecursiveMutex()
        self._sample_rate = -1
        self._frame_size = -1
        self._active_channels = frozenset()
        self._constraints = None

        self.__start_time = 0.0
        self.__returned_samples = 0
        self.__simulated_samples = None

    def on_activate(self):
        # Create constraints object and perform sanity/type checking
        self._constraints = FiniteSamplingInputConstraints(
            channel_units=self._channel_units,
            frame_size_limits=self._frame_size_limits,
            sample_rate_limits=self._sample_rate_limits
        )
        # Make sure the ConfigOptions have correct values and types
        # (ensured by FiniteSamplingInputConstraints)
        self._sample_rate_limits = self._constraints.sample_rate_limits
        self._frame_size_limits = self._constraints.frame_size_limits
        self._channel_units = self._constraints.channel_units

        # initialize default settings
        self._sample_rate = self._constraints.max_sample_rate
        self._frame_size = 0
        self._active_channels = frozenset(self._constraints.channel_names)

        # process parameters
        self.__start_time = 0.0
        self.__returned_samples = 0
        self.__simulated_samples = None
        self.__simulated_odmr_params = dict()
        
        # Try to initialize the NV simulator manager
        self._nv_sim = None
        if self._use_nv_simulator:
            try:
                # Use relative import for nv_simulator_manager
                from . import nv_simulator_manager
                self._nv_sim = nv_simulator_manager.NVSimulatorManager(
                    magnetic_field=self._magnetic_field,
                    temperature=self._temperature,
                    use_simulator=True
                )
                self.log.info("NV simulator integration enabled for finite_sampling_input_dummy")
                self.log.debug(f"NV simulator initialized with magnetic field {self._magnetic_field} G, "
                              f"temperature {self._temperature} K")
            except ImportError as e:
                self.log.warning(f"Could not import NV simulator manager: {e}")
                self.log.warning("Trying fallback import...")
                # Fallback direct import attempt
                try:
                    import os
                    import sys
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    if current_dir not in sys.path:
                        sys.path.insert(0, current_dir)
                    from nv_simulator_manager import NVSimulatorManager
                    self._nv_sim = NVSimulatorManager(
                        magnetic_field=self._magnetic_field,
                        temperature=self._temperature,
                        use_simulator=True
                    )
                    self.log.info("NV simulator integration enabled for finite_sampling_input_dummy (fallback import)")
                except ImportError:
                    self.log.warning("Fallback import also failed - Using fallback simulation model instead")
            except Exception as e:
                self.log.warning(f"Failed to initialize NV simulator: {e}")
                self.log.warning("Using fallback simulation model instead")

    def on_deactivate(self):
        self.__simulated_samples = None

    @property
    def constraints(self):
        return self._constraints

    @property
    def active_channels(self):
        return self._active_channels

    @property
    def sample_rate(self):
        return self._sample_rate

    @property
    def frame_size(self):
        return self._frame_size

    @property
    def samples_in_buffer(self):
        with self._thread_lock:
            if self.module_state() == 'locked':
                elapsed_time = time.time() - self.__start_time
                acquired_samples = min(self._frame_size,
                                       int(elapsed_time * self._sample_rate))
                return max(0, acquired_samples - self.__returned_samples)
            return 0

    def set_sample_rate(self, rate):
        sample_rate = float(rate)
        assert self._constraints.sample_rate_in_range(sample_rate)[0], \
            f'Sample rate "{sample_rate}Hz" to set is out of ' \
            f'bounds {self._constraints.sample_rate_limits}'
        with self._thread_lock:
            assert self.module_state() == 'idle', \
                'Unable to set sample rate. Data acquisition in progress.'
            self._sample_rate = sample_rate

    def set_active_channels(self, channels):
        chnl_set = frozenset(channels)
        assert chnl_set.issubset(self._constraints.channel_names), \
            'Invalid channels encountered to set active'
        with self._thread_lock:
            assert self.module_state() == 'idle', \
                'Unable to set active channels. Data acquisition in progress.'
            self._active_channels = chnl_set

    def set_frame_size(self, size):
        samples = int(round(size))
        assert self._constraints.frame_size_in_range(samples)[0], \
            f'frame size "{samples}" to set is out of bounds {self._constraints.frame_size_limits}'
        with self._thread_lock:
            assert self.module_state() == 'idle', \
                'Unable to set frame size. Data acquisition in progress.'
            self._frame_size = samples

    def start_buffered_acquisition(self):
        with self._thread_lock:
            assert self.module_state() == 'idle', \
                'Unable to start data acquisition. Data acquisition already in progress.'
            assert isinstance(self._simulation_mode, SimulationMode), 'Invalid simulation mode'
            self.module_state.lock()

            # ToDo: discriminate between different types of data
            if self._simulation_mode is SimulationMode.ODMR:
                self.__simulate_odmr(self._frame_size)
            elif self._simulation_mode is SimulationMode.RANDOM:
                self.__simulate_random(self._frame_size)

            self.__returned_samples = 0
            self.__start_time = time.time()

    def stop_buffered_acquisition(self):
        with self._thread_lock:
            if self.module_state() == 'locked':
                remaining_samples = self._frame_size - self.__returned_samples
                if remaining_samples > 0:
                    self.log.warning(
                        f'Buffered sample acquisition stopped before all samples have '
                        f'been read. {remaining_samples} remaining samples will be lost.'
                    )
                self.module_state.unlock()

    def get_buffered_samples(self, number_of_samples=None):
        with self._thread_lock:
            available_samples = self.samples_in_buffer
            if number_of_samples is None:
                number_of_samples = available_samples
            else:
                remaining_samples = self._frame_size - self.__returned_samples
                assert number_of_samples <= remaining_samples, \
                    f'Number of samples to read ({number_of_samples}) exceeds remaining samples ' \
                    f'in this frame ({remaining_samples})'

            # Return early if no samples are requested
            if number_of_samples < 1:
                return dict()

            # Wait until samples have been acquired if requesting more samples than in the buffer
            pending_samples = number_of_samples - available_samples
            if pending_samples > 0:
                time.sleep(pending_samples / self._sample_rate)
            # return data and increment sample counter
            data = {ch: samples[self.__returned_samples:self.__returned_samples + number_of_samples]
                    for ch, samples in self.__simulated_samples.items()}
            self.__returned_samples += number_of_samples
            return data

    def acquire_frame(self, frame_size=None):
        with self._thread_lock:
            if frame_size is None:
                buffered_frame_size = None
            else:
                buffered_frame_size = self._frame_size
                self.set_frame_size(frame_size)

            self.start_buffered_acquisition()
            data = self.get_buffered_samples(self._frame_size)
            self.stop_buffered_acquisition()

            if buffered_frame_size is not None:
                self._frame_size = buffered_frame_size
            return data

    def __simulate_random(self, length):
        channels = self._constraints.channel_names
        self.__simulated_samples = {
            ch: np.random.rand(length) for ch in channels if ch in self._active_channels
        }

    def __simulate_odmr(self, length):
        if length < 3:
            self.__simulate_random(length)
            return
        
        # Generate data for each active channel
        data = dict()
        
        # Get current microwave settings from scan configuration
        zero_field = 2.87e9  # Zero-field splitting in Hz (Default)
        
        # For 500G B-field, Zeeman splitting is ~1.4 GHz (2.8 MHz/G * 500G)
        # So we need a wide range to capture both resonances
        freq_range = 3.0e9  # Widened range to ensure we capture both dips (3 GHz)
        freq_min = 1.8e9    # Start well below the lower dip (~1.47 GHz)
        freq_max = 4.8e9    # End well above the upper dip (~4.27 GHz)
        
        self.log.debug(f"ODMR simulation requested with {length} frequency points from "
                     f"{freq_min/1e9:.3f} GHz to {freq_max/1e9:.3f} GHz")
        
        try:
            # Get simulator instance
            if not hasattr(self, '_nv_sim') or self._nv_sim is None:
                # Create the simulator if it doesn't exist yet
                try:
                    # Try relative import first
                    try:
                        from .nv_simulator_manager import NVSimulatorManager
                        self.log.debug("Imported NVSimulatorManager from relative import")
                    except (ImportError, ValueError):
                        # If that fails, try direct import
                        from nv_simulator_manager import NVSimulatorManager
                        self.log.debug("Imported NVSimulatorManager from direct import")
                    
                    self._nv_sim = NVSimulatorManager(
                        magnetic_field=self._magnetic_field,
                        temperature=self._temperature,
                        use_simulator=True
                    )
                    self.log.info("NV simulator initialized on-demand for ODMR simulation")
                except Exception as e:
                    self.log.error(f"Could not initialize NV simulator: {e}")
                    # If simulator fails, fall back to random data
                    self.__simulate_random(length)
                    self.log.warning("Using random data instead of ODMR simulation")
                    return
            
            # Simulate ODMR using the NV simulator
            try:
                self.log.debug("Requesting ODMR simulation from NV simulator")
                odmr_result = self._nv_sim.simulate_odmr(freq_min, freq_max, length)
                
                # Verify we have valid data
                if 'signal' not in odmr_result or len(odmr_result['signal']) != length:
                    self.log.warning(f"Invalid ODMR result: expected {length} points, got " + 
                                   f"{len(odmr_result.get('signal', []))} points")
                    # Fall back to random data
                    self.__simulate_random(length)
                    return
                
                # Use the simulation data
                for ch in self._active_channels:
                    # Add some random variation to base level for each channel
                    base_level = ((np.random.rand() - 0.5) * 0.05 + 1) * 200000
                    signal_scale = base_level / 100000.0  # Scale factor
                    
                    # Scale the signal to the base level for this channel
                    signal = odmr_result['signal'] * signal_scale
                    
                    # Add some extra random noise
                    noise_level = base_level * 0.01  # 1% noise
                    noise = np.random.normal(0, noise_level, length)
                    
                    # Store the final signal with noise
                    data[ch] = signal + noise
                
                # Store the simulated data
                self.__simulated_samples = data
                
                # Log success
                self.log.debug(f"Generated ODMR data using NV simulator model: min={np.min(signal):.1f}, max={np.max(signal):.1f}")
                
                # If simulation shows flat line with no dips, log a warning
                if np.max(odmr_result['signal']) - np.min(odmr_result['signal']) < 1000:
                    self.log.warning("ODMR signal shows minimal contrast - check magnetic field and simulator settings")
                
            except Exception as e:
                self.log.error(f"Failed to simulate ODMR: {e}")
                # Fall back to random data
                self.__simulate_random(length)
                self.log.warning("Using random data instead of ODMR simulation due to error")
        
        except Exception as e:
            self.log.error(f"Global error in ODMR simulation: {e}")
            # Fall back to random data as a last resort
            self.__simulate_random(length)
