# -*- coding: utf-8 -*-

"""
This file contains the Qudi hardware file to control the NV simulator as a microwave source.

Copyright (c) 2023, IQO

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
import os
import sys
from pathlib import Path

from qudi.interface.microwave_interface import MicrowaveInterface, MicrowaveConstraints
from qudi.util.enums import SamplingOutputMode
from qudi.util.mutex import Mutex
from qudi.core.configoption import ConfigOption
from qudi.core.connector import Connector

# Import QudiFacade directly from current directory to avoid circular imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from qudi_facade import QudiFacade

class NVSimMicrowaveDevice(MicrowaveInterface):
    """A qudi hardware module that uses the NV simulator as a microwave source.
    
    This module translates the microwave interface commands to operations on the NV model.
    
    Example config for copy-paste:

    nv_sim_microwave:
        module.Class: 'nv_simulator.microwave_device.NVSimMicrowaveDevice'
        magnetic_field: [0, 0, 100]  # Gauss, optional
        temperature: 300  # Kelvin, optional
        fixed_startup_time: 0.2  # seconds, time to simulate hardware startup delay
    """

    # Configuration options
    _magnetic_field = ConfigOption('magnetic_field', default=[0, 0, 0], missing='warn')
    _temperature = ConfigOption('temperature', default=300, missing='warn')
    _fixed_startup_time = ConfigOption('fixed_startup_time', default=0.1, missing='warn')
    
    # Connector for simulator
    simulator = Connector(interface='MicrowaveInterface')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._thread_lock = Mutex()
        self._constraints = None
        
        # Get startup time from config
        self._startup_time = self._fixed_startup_time
        
        # Internal state variables
        self._cw_power = 0.
        self._cw_frequency = 2.87e9
        self._scan_power = 0.
        self._scan_frequencies = None
        self._scan_sample_rate = -1.
        self._scan_mode = SamplingOutputMode.JUMP_LIST
        self._is_scanning = False
        self._current_scan_index = 0

    def on_activate(self):
        """Initialisation performed during activation of the module."""
        # Define hardware constraints
        self._constraints = MicrowaveConstraints(
            power_limits=(-60.0, 30),
            frequency_limits=(100e3, 20e9),
            scan_size_limits=(2, 1001),
            sample_rate_limits=(0.1, 200),
            scan_modes=(SamplingOutputMode.JUMP_LIST, SamplingOutputMode.EQUIDISTANT_SWEEP)
        )
        
        # Initialize parameters
        self._cw_power = self._constraints.min_power + (
                    self._constraints.max_power - self._constraints.min_power) / 2
        self._cw_frequency = 2.87e9
        self._scan_power = self._cw_power
        self._scan_frequencies = None
        self._scan_mode = SamplingOutputMode.JUMP_LIST
        self._scan_sample_rate = 100
        self._is_scanning = False
        
        # Get QudiFacade from connector
        self._qudi_facade = self.simulator()
        
        # Reset the NV simulator state
        self._qudi_facade.reset()
        
        self.log.info('NV Simulator Microwave Device initialized')

    def on_deactivate(self):
        """Cleanup performed during deactivation of the module."""
        # Turn off microwave
        self._qudi_facade.microwave_controller.off()

    @property
    def constraints(self):
        """The microwave constraints object for this device.

        @return MicrowaveConstraints:
        """
        return self._constraints

    @property
    def is_scanning(self):
        """Read-Only boolean flag indicating if a scan is running at the moment.

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
            
            # Turn off microwave in simulator
            self._qudi_facade.microwave_controller.off()
            
            # Simulate hardware delay
            time.sleep(self._startup_time)
            
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

    def cw_on(self):
        """Switches on cw microwave output.

        Must return AFTER the output is actually active.
        """
        with self._thread_lock:
            if self.module_state() == 'idle':
                self.log.debug(f'Starting CW microwave output with {self._cw_frequency:.6e} Hz '
                               f'and {self._cw_power:.6f} dBm')
                               
                # Set the microwave parameters in the simulator
                self._qudi_facade.microwave_controller.set_frequency(self._cw_frequency)
                self._qudi_facade.microwave_controller.set_power(self._cw_power)
                self._qudi_facade.microwave_controller.on()
                
                # Simulate hardware delay
                time.sleep(self._startup_time)
                
                self._is_scanning = False
                self.module_state.lock()
                
            elif self._is_scanning:
                raise RuntimeError(
                    'Unable to start microwave CW output. Frequency scanning in progress.'
                )
            else:
                self.log.debug('CW microwave output already running')

    def configure_scan(self, power, frequencies, mode, sample_rate):
        """Configure a frequency scan for the microwave output.
        
        @param float power: Power in dBm
        @param frequencies: Frequencies in Hz
        @param mode: Scan mode enum
        @param float sample_rate: Sample rate in Hz
        """
        with self._thread_lock:
            # Sanity checking
            if self.module_state() != 'idle':
                raise RuntimeError('Unable to configure scan. Microwave output is active.')
            self._assert_scan_configuration_args(power, frequencies, mode, sample_rate)

            # Simulate hardware delay for configuration
            time.sleep(self._startup_time)
            
            # Store scan parameters
            if mode == SamplingOutputMode.EQUIDISTANT_SWEEP:
                self._scan_frequencies = tuple(frequencies)
            else:
                self._scan_frequencies = np.asarray(frequencies, dtype=np.float64)
                
            self._scan_power = power
            self._scan_mode = mode
            self._scan_sample_rate = sample_rate
            self._current_scan_index = 0
            
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
                
            # Lock module state and set scanning flag
            self.module_state.lock()
            self._is_scanning = True
            self._current_scan_index = 0
            
            # Get the first frequency and apply it to the simulator
            if self._scan_mode == SamplingOutputMode.JUMP_LIST:
                first_freq = self._scan_frequencies[0]
            else:  # EQUIDISTANT_SWEEP
                first_freq = self._scan_frequencies[0]  # Start frequency
                
            # Add extensive logging
            self.log.info(f"Starting scan with frequencies: {self._scan_frequencies}")
            self.log.info(f"Setting initial frequency to {first_freq/1e9:.6f} GHz with power {self._scan_power} dBm")
                
            # Set the microwave parameters in the simulator
            self._qudi_facade.microwave_controller.set_frequency(first_freq)
            self._qudi_facade.microwave_controller.set_power(self._scan_power)
            self._qudi_facade.microwave_controller.on()
            
            # Simulate hardware delay
            time.sleep(self._startup_time)
            
            self.log.info(f'Started frequency scan in "{self._scan_mode.name}" mode')

    def reset_scan(self):
        """Reset currently running scan and return to start frequency.
        Does not need to stop and restart the microwave output if the device allows soft scan reset.
        """
        with self._thread_lock:
            if self._is_scanning:
                self.log.debug('Frequency scan soft reset')
                
                # Reset to the first frequency
                self._current_scan_index = 0
                
                if self._scan_mode == SamplingOutputMode.JUMP_LIST:
                    first_freq = self._scan_frequencies[0]
                else:  # EQUIDISTANT_SWEEP
                    first_freq = self._scan_frequencies[0]  # Start frequency
                    
                # Apply the frequency to the simulator
                self._qudi_facade.microwave_controller.set_frequency(first_freq)
                
                # Simulate hardware delay
                time.sleep(self._startup_time / 2)  # Less delay for a reset

    def scan_next(self):
        """Move to the next frequency in the scan.
        This is a custom method not in the MicrowaveInterface but useful for integration.
        """
        with self._thread_lock:
            if not self._is_scanning:
                self.log.warning("scan_next called but not scanning")
                return False
                
            self._current_scan_index += 1
            
            # Check if we reached the end of the scan
            if self._scan_mode == SamplingOutputMode.JUMP_LIST:
                if self._current_scan_index >= len(self._scan_frequencies):
                    self.log.info(f"Scan completed: reached end of jump list ({len(self._scan_frequencies)} frequencies)")
                    return False
                next_freq = self._scan_frequencies[self._current_scan_index]
            else:  # EQUIDISTANT_SWEEP
                start_freq, stop_freq, num_steps = self._scan_frequencies
                if self._current_scan_index >= num_steps:
                    self.log.info(f"Scan completed: reached end of sweep ({num_steps} steps)")
                    return False
                next_freq = start_freq + (stop_freq - start_freq) * (self._current_scan_index / (num_steps - 1))
                
            # Apply the frequency to the simulator
            self.log.info(f"Scanning to next frequency: {next_freq/1e9:.6f} GHz (index {self._current_scan_index})")
            self._qudi_facade.microwave_controller.set_frequency(next_freq)
            
            return True

    def get_current_frequency(self):
        """Get the current frequency being output.
        This is a custom method not in the MicrowaveInterface but useful for integration.
        """
        with self._thread_lock:
            if not self._is_scanning:
                return self._cw_frequency
                
            if self._scan_mode == SamplingOutputMode.JUMP_LIST:
                return self._scan_frequencies[self._current_scan_index]
            else:  # EQUIDISTANT_SWEEP
                start_freq, stop_freq, num_steps = self._scan_frequencies
                return start_freq + (stop_freq - start_freq) * (self._current_scan_index / (num_steps - 1))