# -*- coding: utf-8 -*-

"""
This file contains the Qudi hardware file to control the microwave dummy.

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

from qudi.interface.microwave_interface import MicrowaveInterface, MicrowaveConstraints
from qudi.util.enums import SamplingOutputMode
from qudi.util.mutex import Mutex
from qudi.core.configoption import ConfigOption


class MicrowaveDummy(MicrowaveInterface):
    """A qudi dummy hardware module to emulate a microwave source.

    Example config for copy-paste:

    mw_source_dummy:
        module.Class: 'microwave.mw_source_dummy.MicrowaveDummy'
        options:
            use_nv_simulator: True  # optional, whether to use the NV simulator
            magnetic_field: [0, 0, 500]  # optional, magnetic field in Gauss [x, y, z]
            temperature: 300  # optional, temperature in Kelvin
    """
    
    _use_nv_simulator = ConfigOption(name='use_nv_simulator', default=False)
    _magnetic_field = ConfigOption(name='magnetic_field', default=[0, 0, 500])  # Gauss
    _temperature = ConfigOption(name='temperature', default=300)  # Kelvin

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._thread_lock = Mutex()
        self._constraints = None

        self._cw_power = 0.
        self._cw_frequency = 2.87e9
        self._scan_power = 0.
        self._scan_frequencies = None
        self._scan_sample_rate = -1.
        self._scan_mode = SamplingOutputMode.JUMP_LIST
        self._is_scanning = False

    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        self._constraints = MicrowaveConstraints(
            power_limits=(-60.0, 30),
            frequency_limits=(100e3, 20e9),
            scan_size_limits=(2, 1001),
            sample_rate_limits=(0.1, 200),
            scan_modes=(SamplingOutputMode.JUMP_LIST, SamplingOutputMode.EQUIDISTANT_SWEEP)
        )

        self._cw_power = self._constraints.min_power + (
                    self._constraints.max_power - self._constraints.min_power) / 2
        self._cw_frequency = 2.87e9
        self._scan_power = self._cw_power
        self._scan_frequencies = None
        self._scan_mode = SamplingOutputMode.JUMP_LIST
        self._scan_sample_rate = 100
        self._is_scanning = False
        
        # Try to initialize the NV simulator manager
        self._nv_sim = None
        if self._use_nv_simulator:
            try:
                from nv_simulator_manager import NVSimulatorManager
                self._nv_sim = NVSimulatorManager(
                    magnetic_field=self._magnetic_field,
                    temperature=self._temperature,
                    use_simulator=True
                )
                self.log.info("NV simulator integration enabled for microwave_dummy")
                self.log.debug(f"NV simulator initialized with magnetic field {self._magnetic_field} G, "
                              f"temperature {self._temperature} K")
            except ImportError as e:
                self.log.warning(f"Could not import NV simulator manager: {e}")
                self.log.warning("NV simulator integration disabled")
            except Exception as e:
                self.log.warning(f"Failed to initialize NV simulator: {e}")
                self.log.warning("NV simulator integration disabled")

    def on_deactivate(self):
        """ Cleanup performed during deactivation of the module.
        """
        pass

    @property
    def constraints(self):
        """The microwave constraints object for this device.

        @return MicrowaveConstraints:
        """
        return self._constraints

    @property
    def is_scanning(self):
        """Read-Only boolean flag indicating if a scan is running at the moment. Can be used
        together with module_state() to determine if the currently running microwave output is a
        scan or CW.
        Should return False if module_state() is 'idle'.

        @return bool: Flag indicating if a scan is running (True) or not (False)
        """
        with self._thread_lock:
            return self._is_scanning

    @property
    def cw_power(self):
        """Read-only property returning the currently configured CW microwave power in dBm.

        @return float: The currently set CW microwave power in dBm.
        """
        with self._thread_lock:
            return self._cw_power

    @property
    def cw_frequency(self):
        """Read-only property returning the currently set CW microwave frequency in Hz.

        @return float: The currently set CW microwave frequency in Hz.
        """
        with self._thread_lock:
            return self._cw_frequency

    @property
    def scan_power(self):
        """Read-only property returning the currently configured microwave power in dBm used for
        scanning.

        @return float: The currently set scanning microwave power in dBm
        """
        with self._thread_lock:
            return self._scan_power

    @property
    def scan_frequencies(self):
        """Read-only property returning the currently configured microwave frequencies used for
        scanning.

        In case of self.scan_mode == SamplingOutputMode.JUMP_LIST, this will be a 1D numpy array.
        In case of self.scan_mode == SamplingOutputMode.EQUIDISTANT_SWEEP, this will be a tuple
        containing 3 values (freq_begin, freq_end, number_of_samples).
        If no frequency scan has been configured, return None.

        @return float[]: The currently set scanning frequencies. None if not set.
        """
        with self._thread_lock:
            return self._scan_frequencies

    @property
    def scan_mode(self):
        """Read-only property returning the currently configured scan mode Enum.

        @return SamplingOutputMode: The currently set scan mode Enum
        """
        with self._thread_lock:
            return self._scan_mode

    @property
    def scan_sample_rate(self):
        """Read-only property returning the currently configured scan sample rate in Hz.

        @return float: The currently set scan sample rate in Hz
        """
        with self._thread_lock:
            return self._scan_sample_rate

    def off(self):
        """Switches off any microwave output (both scan and CW).
        Must return AFTER the device has actually stopped.
        """
        with self._thread_lock:
            if self.module_state() == 'idle':
                self.log.debug('Microwave output was not active')
                return
            self.log.debug('Stopping microwave output')
            
            # Update NV simulator if available
            if hasattr(self, '_nv_sim') and self._nv_sim is not None:
                try:
                    # Turn off microwave in simulator
                    self._nv_sim.set_microwave(self._cw_frequency, self._cw_power, False)
                    self.log.debug("NV simulator microwave turned off")
                except Exception as e:
                    self.log.warning(f"Failed to update NV simulator: {e}")
            
            time.sleep(1)
            self._is_scanning = False
            self.module_state.unlock()

    def set_cw(self, frequency, power):
        """Configure the CW microwave output. Does not start physical signal output, see also
        "cw_on".

        @param float frequency: frequency to set in Hz
        @param float power: power to set in dBm
        """
        with self._thread_lock:
            # Check if CW parameters can be set.
            if self.module_state() != 'idle':
                raise RuntimeError(
                    'Unable to set CW power and frequency. Microwave output is active.'
                )
            self._assert_cw_parameters_args(frequency, power)

            # Set power and frequency
            self.log.debug(f'Setting CW power to {power} dBm and frequency to {frequency:.9e} Hz')
            self._cw_power = power
            self._cw_frequency = frequency
            
            # Update NV simulator if available
            if hasattr(self, '_nv_sim') and self._nv_sim is not None:
                try:
                    # Only update frequency and power, but don't turn on
                    # (since this method doesn't turn on the microwave)
                    if self.module_state() == 'locked' and not self._is_scanning:
                        # Only update if we're in CW mode and running
                        self._nv_sim.set_microwave(self._cw_frequency, self._cw_power, True)
                    else:
                        # Store values but don't turn on
                        self._nv_sim.set_microwave(self._cw_frequency, self._cw_power, False)
                except Exception as e:
                    self.log.warning(f"Failed to update NV simulator: {e}")

    def cw_on(self):
        """ Switches on cw microwave output.

        Must return AFTER the output is actually active.
        """
        with self._thread_lock:
            if self.module_state() == 'idle':
                self.log.debug(f'Starting CW microwave output with {self._cw_frequency:.6e} Hz '
                               f'and {self._cw_power:.6f} dBm')
                time.sleep(1)
                self._is_scanning = False
                
                # Update NV simulator if available
                if hasattr(self, '_nv_sim') and self._nv_sim is not None:
                    try:
                        # Turn on MW with current frequency and power
                        self._nv_sim.set_microwave(self._cw_frequency, self._cw_power, True)
                        self.log.debug("NV simulator microwave turned on")
                    except Exception as e:
                        self.log.warning(f"Failed to update NV simulator: {e}")
                
                # Lock the module state
                self.module_state.lock()
            elif self._is_scanning:
                raise RuntimeError(
                    'Unable to start microwave CW output. Frequency scanning in progress.'
                )
            else:
                self.log.debug('CW microwave output already running')

    def configure_scan(self, power, frequencies, mode, sample_rate):
        """
        """
        with self._thread_lock:
            # Sanity checking
            if self.module_state() != 'idle':
                raise RuntimeError('Unable to configure scan. Microwave output is active.')
            self._assert_scan_configuration_args(power, frequencies, mode, sample_rate)

            # Actually change settings
            time.sleep(1)
            if mode == SamplingOutputMode.EQUIDISTANT_SWEEP:
                self._scan_frequencies = tuple(frequencies)
            else:
                self._scan_frequencies = np.asarray(frequencies, dtype=np.float64)
            self._scan_power = power
            self._scan_mode = mode
            self._scan_sample_rate = sample_rate
            self.log.debug(
                f'Scan configured in mode "{mode.name}" with {sample_rate:.9e} Hz sample rate, '
                f'{power} dBm power and frequencies:\n{self._scan_frequencies}.'
            )

    def start_scan(self):
        """Switches on the microwave scanning.

        Must return AFTER the output is actually active (and can receive triggers for example).
        """
        with self._thread_lock:
            if self.module_state() != 'idle':
                raise RuntimeError(
                    'Unable to start microwave frequency scan. Microwave output is active.'
                )
            self.module_state.lock()
            self._is_scanning = True
            
            # Update NV simulator if available
            # For frequency scans, we don't actually control the NV simulator's microwave
            # since the ODMR module is going to get data from finite_sampling_input_dummy,
            # not from the NV simulator directly
            if hasattr(self, '_nv_sim') and self._nv_sim is not None:
                try:
                    # Just notify about scan starting
                    self.log.debug("Notifying NV simulator about frequency scan")
                    
                    # Set microwave power but not frequency (will be scanned)
                    if self._scan_mode == SamplingOutputMode.EQUIDISTANT_SWEEP:
                        freq_min, freq_max, num_points = self._scan_frequencies
                        self.log.debug(f"Frequency scan range: {freq_min/1e9:.6f}-{freq_max/1e9:.6f} GHz")
                    else:
                        # Jump list mode
                        freq_min = np.min(self._scan_frequencies)
                        freq_max = np.max(self._scan_frequencies)
                        self.log.debug(f"Frequency scan list: {len(self._scan_frequencies)} points")
                except Exception as e:
                    self.log.warning(f"Failed to update NV simulator about scan: {e}")
            
            time.sleep(1)
            self.log.debug(f'Starting frequency scan in "{self._scan_mode.name}" mode')

    def reset_scan(self):
        """Reset currently running scan and return to start frequency.
        Does not need to stop and restart the microwave output if the device allows soft scan reset.
        """
        with self._thread_lock:
            if self._is_scanning:
                self.log.debug('Frequency scan soft reset')
                time.sleep(0.5)
                
    def scan_next(self):
        """Move to the next frequency in the scan.
        This method is added for compatibility with NVSimMicrowaveDevice.
        It advances to the next frequency in the frequency list rather than
        resetting to the beginning.
        
        @return bool: False if end of sequence reached, True otherwise
        """
        with self._thread_lock:
            if not self._is_scanning:
                self.log.warning("scan_next called but not scanning")
                return False
            
            # Simple implementation - just advance an internal counter for now
            # In a real implementation, this would change the actual frequency
            if not hasattr(self, "_scan_index"):
                self._scan_index = 0
                
            self._scan_index += 1
            
            # Check if we've reached the end of the scan
            if self._scan_mode == SamplingOutputMode.EQUIDISTANT_SWEEP:
                _, _, num_points = self._scan_frequencies
                if self._scan_index >= num_points:
                    self._scan_index = 0  # Reset for next scan
                    return False
            else:  # JUMP_LIST
                if self._scan_index >= len(self._scan_frequencies):
                    self._scan_index = 0  # Reset for next scan
                    return False
                    
            # Return True if there are more frequencies
            return True
