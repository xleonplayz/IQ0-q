# -*- coding: utf-8 -*-

"""
Custom pulse sequence experiment mode for the NV center simulator.

Copyright (c) 2023
"""

from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import numpy as np

from .base import ExperimentMode
from .utils import convert_to_qudi_format, add_noise


class CustomSequenceMode(ExperimentMode):
    """
    Custom pulse sequence experiment mode for the NV center simulator.
    Allows defining arbitrary pulse sequences for advanced experiments.
    """
    
    def __init__(self, simulator):
        """
        Initialize the custom sequence experiment mode.
        
        @param simulator: PhysicalNVModel instance to run the experiment on
        """
        super().__init__(simulator)
        self._default_params = {
            'sequence_name': 'custom',                 # Name of the sequence
            'tau_times': np.linspace(50e-9, 10e-6, 51),  # s
            'mw_frequency': 2.87e9,                      # Hz
            'mw_power': -10.0,                           # dBm
            'avg_count': 3,                              # Number of averages
            'laser_power': 1.0,                          # Relative laser power (0 to 1)
            'noise_level': 0.01,                         # Relative noise level
            'pi_half_time': 25e-9,                       # Duration of π/2 pulse (s)
            'pi_time': 50e-9,                            # Duration of π pulse (s)
            'readout_time': 300e-9,                      # Readout window duration (s)
            'initialization_time': 3e-6,                 # Laser initialization time (s)
            'repetition_time': 50e-6,                    # Time between sequence repetitions (s)
            'sequence_params': {}                        # Additional sequence-specific parameters
        }
        self._params = self._default_params.copy()
        self._sequence_func = None
    
    def set_sequence_function(self, func: Callable):
        """
        Set the function that implements the custom pulse sequence.
        
        @param func: Function that implements the pulse sequence
                    Should take (simulator, tau, params) as arguments
                    and return a measurement result.
        """
        self._sequence_func = func
        return self
    
    def run(self) -> Dict[str, Any]:
        """
        Run the custom sequence experiment and return results.
        
        @return: Custom sequence experiment results in Qudi-compatible format
        """
        # Extract parameters
        sequence_name = self._params['sequence_name']
        tau_times = self._params['tau_times']
        mw_freq = self._params['mw_frequency']
        mw_power = self._params['mw_power']
        avg_count = self._params['avg_count']
        laser_power = self._params['laser_power']
        noise_level = self._params['noise_level']
        pi_time = self._params['pi_time']
        pi_half_time = self._params['pi_half_time']
        seq_params = self._params['sequence_params']
        
        # Set laser power in simulator
        self._simulator.apply_laser(laser_power, True)
        
        # Configure microwave
        self._simulator.set_microwave(mw_freq, mw_power, True)
        
        # Check for built-in dynamical decoupling sequences
        if sequence_name.lower() in ['cpmg', 'xy4', 'xy8', 'xy16'] and hasattr(self._simulator, 'simulate_dynamical_decoupling'):
            n_pulses = seq_params.get('n_pulses', 4)
            result = self._simulator.simulate_dynamical_decoupling(
                sequence_type=sequence_name.lower(),
                t_max=np.max(tau_times),
                n_points=len(tau_times),
                n_pulses=n_pulses,
                mw_frequency=mw_freq,
                mw_power=mw_power
            )
            
            # Average multiple runs if requested
            if avg_count > 1:
                coherence = result.coherence.copy()
                for _ in range(avg_count - 1):
                    new_result = self._simulator.simulate_dynamical_decoupling(
                        sequence_type=sequence_name.lower(),
                        t_max=np.max(tau_times),
                        n_points=len(tau_times),
                        n_pulses=n_pulses,
                        mw_frequency=mw_freq,
                        mw_power=mw_power
                    )
                    coherence += new_result.coherence
                
                # Update with averaged signal
                result.coherence = coherence / avg_count
            
            # Add noise if requested
            if noise_level > 0:
                result.coherence = add_noise(result.coherence, noise_level)
            
            # Prepare result
            qudi_result = convert_to_qudi_format(result, 'custom_sequence')
            qudi_result['parameters'] = self._params.copy()
            return qudi_result
        
        # Execute custom sequence function if provided
        elif self._sequence_func is not None:
            # Create storage for results
            signal = np.zeros(len(tau_times))
            
            # Run the sequence for each tau value
            for i, tau in enumerate(tau_times):
                this_signal = 0
                
                for _ in range(avg_count):
                    # Call the custom sequence function
                    result = self._sequence_func(self._simulator, tau, self._params)
                    this_signal += result
                    
                    # Wait for system to reset
                    self._simulator.wait(self._params['repetition_time'])
                
                signal[i] = this_signal / avg_count
            
            # Add noise if requested
            if noise_level > 0:
                signal = add_noise(signal, noise_level)
            
            # Create a result object
            result = type('CustomSequenceResult', (), {
                'times': tau_times,
                'signal': signal,
                'experiment_type': 'custom_sequence',
                'sequence_name': sequence_name,
                'parameters': self._params.copy()
            })
        
        # Fall back to CPMG implementation for dynamical decoupling
        elif sequence_name.lower() in ['cpmg', 'xy4', 'xy8', 'xy16']:
            # Get number of pulses/cycles
            n_pulses = seq_params.get('n_pulses', 4)
            signal = np.zeros(len(tau_times))
            
            for i, tau in enumerate(tau_times):
                this_signal = 0
                
                for _ in range(avg_count):
                    # Initialize with laser
                    self._simulator.apply_laser(laser_power, True, self._params['initialization_time'])
                    
                    # First π/2 pulse (X axis)
                    self._simulator.apply_microwave_pulse(pi_half_time, mw_freq, mw_power)
                    
                    # Apply DD sequence (implementation based on sequence type)
                    if sequence_name.lower() == 'cpmg':
                        # CPMG sequence - all pulses along same axis
                        pulse_interval = tau / n_pulses
                        for j in range(n_pulses):
                            self._simulator.wait(pulse_interval/2)
                            self._simulator.apply_microwave_pulse(pi_time, mw_freq, mw_power)  # π pulse (X axis)
                            self._simulator.wait(pulse_interval/2)
                    
                    elif sequence_name.lower() in ['xy4', 'xy8', 'xy16']:
                        # XY-family sequences - alternating X and Y pulses
                        # Simplified implementation - actual should consider phase properly
                        pulse_interval = tau / n_pulses
                        for j in range(n_pulses):
                            self._simulator.wait(pulse_interval/2)
                            # Alternate between X and Y pulses (simplified)
                            phase = 0 if j % 2 == 0 else 90  # 0° for X, 90° for Y
                            self._simulator.apply_microwave_pulse(pi_time, mw_freq, mw_power, phase)
                            self._simulator.wait(pulse_interval/2)
                    
                    # Final π/2 pulse (X axis)
                    self._simulator.apply_microwave_pulse(pi_half_time, mw_freq, mw_power)
                    
                    # Readout
                    readout = self._simulator.read_state(self._params['readout_time'])
                    this_signal += readout.signal
                    
                    # Wait for system to reset
                    self._simulator.wait(self._params['repetition_time'])
                
                signal[i] = this_signal / avg_count
            
            # Add noise if requested
            if noise_level > 0:
                signal = add_noise(signal, noise_level)
            
            # Create a result object
            result = type('CustomSequenceResult', (), {
                'times': tau_times,
                'signal': signal,
                'coherence': signal,  # For compatibility with DD results
                'experiment_type': 'custom_sequence',
                'sequence_name': sequence_name,
                'parameters': self._params.copy()
            })
        
        # Generic custom sequence - not implemented
        else:
            raise ValueError(f"Custom sequence '{sequence_name}' not implemented and no custom function provided.")
        
        # Convert to Qudi-compatible format
        qudi_result = convert_to_qudi_format(result, 'custom_sequence')
        qudi_result['parameters'] = self._params.copy()
        
        return qudi_result
    
    def register_predefined_sequence(self, name: str, func: Callable):
        """
        Register a predefined sequence by name.
        
        @param name: Name of the sequence
        @param func: Function implementing the sequence
        """
        # In a real implementation, this would store the function in a dictionary
        # and set it when the sequence_name is selected
        if name == self._params['sequence_name']:
            self._sequence_func = func
        return self