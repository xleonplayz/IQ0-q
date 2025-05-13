# -*- coding: utf-8 -*-

"""
Qudi hardware interface adapter for NV simulator microwave control.

Copyright (c) 2023
"""

import numpy as np
import time
from typing import Union, Tuple, Optional

from qudi.interface.microwave_interface import MicrowaveInterface, MicrowaveConstraints
from qudi.util.enums import SamplingOutputMode


class NVSimulatorMicrowave(MicrowaveInterface):
    """
    Hardware adapter that implements the MicrowaveInterface for the NV center simulator.
    This interface enables ODMR experiments through Qudi.
    """

    def __init__(self, nv_simulator, name='nvmw'):
        """
        Initialize the microwave adapter for the NV simulator.
        
        @param nv_simulator: Instance of the PhysicalNVModel simulator
        @param name: Unique name for this hardware module
        """
        # Initialize the module base class
        super().__init__(name=name)

        # Store simulator reference
        self._simulator = nv_simulator
        
        # Create microwave constraints
        self._constraints = MicrowaveConstraints(
            power_limits=(-60, 10),
            frequency_limits=(2.7e9, 3.1e9),  # Range around NV zero-field splitting
            scan_size_limits=(2, 1001),
            sample_rate_limits=(0.1, 1000),
            scan_modes=(SamplingOutputMode.JUMP_LIST, SamplingOutputMode.EQUIDISTANT_SWEEP)
        )
        
        # Initialize state variables
        self._is_scanning = False
        self._cw_power = 0.0
        self._cw_frequency = 2.87e9  # Default to NV zero-field splitting
        self._scan_power = 0.0
        self._scan_frequencies = None
        self._scan_mode = SamplingOutputMode.JUMP_LIST
        self._scan_sample_rate = 100.0
        self._current_frequency_index = 0
        self._scanner_running = False
        
        # Thread lock for thread safety
        self._thread_lock = self.module_state.lock_access()

    @property
    def constraints(self) -> MicrowaveConstraints:
        """The microwave constraints object for this device."""
        return self._constraints

    @property
    def is_scanning(self) -> bool:
        """
        Read-Only boolean flag indicating if a scan is running at the moment.
        """
        return self._is_scanning

    @property
    def cw_power(self) -> float:
        """
        Read-only property returning the currently configured CW microwave power in dBm.
        """
        return self._cw_power

    @property
    def cw_frequency(self) -> float:
        """
        Read-only property returning the currently set CW microwave frequency in Hz.
        """
        return self._cw_frequency

    @property
    def scan_power(self) -> float:
        """
        Read-only property returning the currently configured microwave power in dBm used for scanning.
        """
        return self._scan_power

    @property
    def scan_frequencies(self) -> Union[np.ndarray, Tuple[float, float, float], None]:
        """
        Read-only property returning the currently configured microwave frequencies used for scanning.
        """
        return self._scan_frequencies

    @property
    def scan_mode(self) -> SamplingOutputMode:
        """
        Read-only property returning the currently configured scan mode Enum.
        """
        return self._scan_mode

    @property
    def scan_sample_rate(self) -> float:
        """
        Read-only property returning the currently configured scan sample rate in Hz.
        """
        return self._scan_sample_rate

    def on_activate(self):
        """
        Called when module is activated
        """
        self.module_state.lock()
        try:
            self.log.info('NV Simulator microwave interface activated')
            # Reset the simulator state and initialize microwave settings
            self._simulator.reset_state()
            self._simulator.apply_microwave(self._cw_frequency, self._cw_power, on=False)
        finally:
            self.module_state.unlock()

    def on_deactivate(self):
        """
        Called when module is deactivated
        """
        self.module_state.lock()
        try:
            # Turn off microwave before deactivation
            self._simulator.apply_microwave(self._cw_frequency, self._cw_power, on=False)
            self.log.info('NV Simulator microwave interface deactivated')
        finally:
            self.module_state.unlock()

    def off(self) -> None:
        """
        Switches off any microwave output (both scan and CW).
        """
        with self._thread_lock:
            if self.module_state() == 'idle':
                return
                
            # Turn off the microwave in the simulator
            self._simulator.apply_microwave(self._cw_frequency, self._cw_power, on=False)
            
            # Update state flags
            self._is_scanning = False
            self._scanner_running = False
            self._current_frequency_index = 0
            
            # Set module to idle
            self.module_state.unlock()

    def set_cw(self, frequency: float, power: float) -> None:
        """
        Configure the CW microwave output.
        
        @param float frequency: frequency to set in Hz
        @param float power: power to set in dBm
        """
        # Validate parameters against constraints
        self._assert_cw_parameters_args(frequency, power)
        
        with self._thread_lock:
            # Update internal state
            self._cw_frequency = frequency
            self._cw_power = power
            
            # Update the simulator if we're in CW mode and output is on
            if self.module_state() == 'running' and not self._is_scanning:
                self._simulator.apply_microwave(frequency, power, on=True)

    def cw_on(self) -> None:
        """
        Switches on the CW microwave output.
        """
        with self._thread_lock:
            # Check if we're idle
            if self.module_state() != 'idle':
                self.log.error('Cannot turn on CW microwave. Microwave output already active.')
                return
                
            # Turn on microwave in the simulator
            self._simulator.apply_microwave(self._cw_frequency, self._cw_power, on=True)
            
            # Update state
            self._is_scanning = False
            self.module_state.lock()

    def configure_scan(self, power: float, frequencies: Union[np.ndarray, Tuple[float, float, float]],
                       mode: SamplingOutputMode, sample_rate: float) -> None:
        """
        Configure a frequency scan.
        
        @param float power: the power in dBm to be used during the scan
        @param frequencies: an array of frequencies (jump list) or a tuple (start, stop, points)
        @param SamplingOutputMode mode: enum stating how the frequencies are defined
        @param float sample_rate: external scan trigger rate
        """
        # Validate the parameters
        self._assert_scan_configuration_args(power, frequencies, mode, sample_rate)
        
        with self._thread_lock:
            # Store the configuration
            self._scan_power = power
            self._scan_mode = mode
            self._scan_sample_rate = sample_rate
            
            # Convert frequencies to the correct format if needed
            if mode == SamplingOutputMode.EQUIDISTANT_SWEEP:
                start, stop, num_points = frequencies
                self._scan_frequencies = np.linspace(start, stop, int(num_points))
            else:  # JUMP_LIST
                self._scan_frequencies = np.array(frequencies)

    def start_scan(self) -> None:
        """
        Switches on the preconfigured microwave scanning.
        """
        with self._thread_lock:
            # Check if we're idle and have configured frequencies
            if self.module_state() != 'idle':
                self.log.error('Cannot start frequency scan. Microwave output already active.')
                return
                
            if self._scan_frequencies is None:
                self.log.error('No scan frequencies configured. Cannot start scan.')
                return
                
            # Setup for scanning
            self._current_frequency_index = 0
            self._is_scanning = True
            self._scanner_running = True
            
            # Start with the first frequency in the scan
            current_freq = self._scan_frequencies[0]
            self._simulator.apply_microwave(current_freq, self._scan_power, on=True)
            
            # Lock the module state
            self.module_state.lock()

    def reset_scan(self) -> None:
        """
        Reset currently running scan and return to start frequency.
        """
        with self._thread_lock:
            if not self._is_scanning:
                return
                
            # Reset scan index
            self._current_frequency_index = 0
            
            # Apply the first frequency again
            if len(self._scan_frequencies) > 0:
                current_freq = self._scan_frequencies[0]
                self._simulator.apply_microwave(current_freq, self._scan_power, on=True)

    def scan_next(self) -> None:
        """
        Internal helper method to step to the next frequency in a scan.
        This would typically be called in response to external triggers.
        """
        if not self._is_scanning or not self._scanner_running:
            return
            
        with self._thread_lock:
            # Advance to next frequency if available
            self._current_frequency_index += 1
            
            # Check if we've reached the end of the scan
            if self._current_frequency_index >= len(self._scan_frequencies):
                self._current_frequency_index = 0
                
            # Apply the new frequency
            current_freq = self._scan_frequencies[self._current_frequency_index]
            self._simulator.apply_microwave(current_freq, self._scan_power, on=True)

    def get_current_frequency(self) -> float:
        """
        Get the currently active frequency (useful for synchronizing with scanner).
        
        @return float: Currently active frequency in Hz
        """
        if self._is_scanning and self._scanner_running:
            return self._scan_frequencies[self._current_frequency_index]
        elif self.module_state() == 'running' and not self._is_scanning:
            return self._cw_frequency
        else:
            return 0.0