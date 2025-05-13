# -*- coding: utf-8 -*-

"""
ODMR experiment mode for the NV center simulator.

Copyright (c) 2023
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np

from .base import ExperimentMode
from .utils import convert_to_qudi_format, add_noise


class ODMRMode(ExperimentMode):
    """
    Optically Detected Magnetic Resonance experiment mode.
    Implements both continuous wave (CW) and pulsed ODMR experiments.
    """
    
    def __init__(self, simulator):
        """
        Initialize the ODMR experiment mode.
        
        @param simulator: PhysicalNVModel instance to run the experiment on
        """
        super().__init__(simulator)
        self._default_params = {
            'freq_start': 2.82e9,    # Hz
            'freq_stop': 2.92e9,     # Hz
            'num_points': 101,       # Number of frequency points
            'power': -10.0,          # dBm
            'avg_count': 3,          # Number of averages
            'pulsed': False,         # CW or pulsed ODMR
            'laser_power': 1.0,      # Relative laser power (0 to 1)
            'contrast_enhancement': False,  # Apply contrast enhancement techniques
            'noise_level': 0.01,     # Relative noise level
            'pi_pulse_duration': 50e-9,  # Duration of π pulse for pulsed ODMR (s)
            'readout_time': 300e-9,  # Readout window duration (s)
            'repetition_time': 2e-6  # Time between sequence repetitions (s)
        }
        self._params = self._default_params.copy()
    
    def run(self) -> Dict[str, Any]:
        """
        Run the ODMR experiment and return results.
        
        @return: ODMR experiment results in Qudi-compatible format
        """
        # Extract parameters
        f_min = self._params['freq_start']
        f_max = self._params['freq_stop']
        num_points = self._params['num_points']
        power = self._params['power']
        avg_count = self._params['avg_count']
        pulsed = self._params['pulsed']
        laser_power = self._params['laser_power']
        noise_level = self._params['noise_level']
        
        # Set laser power in simulator
        self._simulator.apply_laser(laser_power, True)
        
        # Run the appropriate simulation based on mode
        if not pulsed:
            # CW ODMR - use simulator's ODMR function
            # Run first simulation
            result = self._simulator.simulate_odmr(f_min, f_max, num_points, power)
            signal = result.signal
            
            # Average multiple runs if requested
            if avg_count > 1:
                for _ in range(avg_count - 1):
                    result_new = self._simulator.simulate_odmr(f_min, f_max, num_points, power)
                    signal += result_new.signal
                
                # Compute average
                signal = signal / avg_count
            
            # Update result with averaged signal
            result.signal = signal
            
            # Add realistic noise if specified
            if noise_level > 0:
                result.signal = add_noise(result.signal, noise_level)
                
        else:
            # Pulsed ODMR - implement using pulse sequence
            pi_pulse_duration = self._params['pi_pulse_duration']
            readout_time = self._params['readout_time']
            repetition_time = self._params['repetition_time']
            
            # Frequencies to sweep
            frequencies = np.linspace(f_min, f_max, num_points)
            signal = np.zeros(num_points)
            
            # For each frequency, run the pulsed sequence
            for i, freq in enumerate(frequencies):
                # Configure microwave for this frequency
                self._simulator.set_microwave(freq, power, True)
                
                # Simulate a π pulse followed by readout
                # This is simplified - actual implementation would use pulse sequences
                this_signal = 0
                
                for _ in range(avg_count):
                    # Apply π pulse and then readout
                    self._simulator.apply_microwave_pulse(pi_pulse_duration, freq, power)
                    readout = self._simulator.read_state(readout_time)
                    this_signal += readout.signal
                    
                    # Wait for system to reset
                    self._simulator.wait(repetition_time)
                
                signal[i] = this_signal / avg_count
            
            # Add realistic noise if specified
            if noise_level > 0:
                signal = add_noise(signal, noise_level)
            
            # Create a result object similar to CW ODMR
            result = type('ODMRResult', (), {
                'frequencies': frequencies,
                'signal': signal,
                'experiment_type': 'pulsed_odmr',
                'parameters': self._params.copy()
            })
            
        # Apply contrast enhancement if requested
        if self._params['contrast_enhancement']:
            # Normalize to [0, 1]
            min_val = np.min(result.signal)
            max_val = np.max(result.signal)
            if max_val > min_val:
                result.signal = (result.signal - min_val) / (max_val - min_val)
                # Invert if needed so dips become peaks
                if np.median(result.signal) > 0.5:
                    result.signal = 1 - result.signal
            
        # Convert to Qudi-compatible format
        qudi_result = convert_to_qudi_format(result, 'odmr')
        qudi_result['parameters'] = self._params.copy()
        
        return qudi_result