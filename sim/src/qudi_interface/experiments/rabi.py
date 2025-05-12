# -*- coding: utf-8 -*-

"""
Rabi experiment mode for the NV center simulator.

Copyright (c) 2023
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np

from .base import ExperimentMode
from .utils import convert_to_qudi_format, add_noise


class RabiMode(ExperimentMode):
    """
    Rabi oscillation experiment mode for the NV center simulator.
    Simulates Rabi oscillations by varying the microwave pulse duration.
    """
    
    def __init__(self, simulator):
        """
        Initialize the Rabi experiment mode.
        
        @param simulator: PhysicalNVModel instance to run the experiment on
        """
        super().__init__(simulator)
        self._default_params = {
            'rabi_times': np.linspace(0, 500e-9, 51),  # s
            'mw_frequency': 2.87e9,                    # Hz
            'mw_power': -10.0,                         # dBm
            'avg_count': 3,                            # Number of averages
            'laser_power': 1.0,                        # Relative laser power (0 to 1)
            'noise_level': 0.01,                       # Relative noise level
            'readout_time': 300e-9,                    # Readout window duration (s)
            'initialization_time': 3e-6,               # Laser initialization time (s)
            'repetition_time': 5e-6                    # Time between sequence repetitions (s)
        }
        self._params = self._default_params.copy()
    
    def run(self) -> Dict[str, Any]:
        """
        Run the Rabi experiment and return results.
        
        @return: Rabi experiment results in Qudi-compatible format
        """
        # Extract parameters
        rabi_times = self._params['rabi_times']
        mw_freq = self._params['mw_frequency']
        mw_power = self._params['mw_power']
        avg_count = self._params['avg_count']
        laser_power = self._params['laser_power']
        noise_level = self._params['noise_level']
        
        # Set laser power in simulator
        self._simulator.apply_laser(laser_power, True)
        
        # Use built-in simulator function if available
        if hasattr(self._simulator, 'simulate_rabi'):
            result = self._simulator.simulate_rabi(
                t_max=np.max(rabi_times),
                n_points=len(rabi_times),
                mw_frequency=mw_freq,
                mw_power=mw_power
            )
            
            # Average multiple runs if requested
            if avg_count > 1:
                signal = result.populations[:, 0]  # ms=0 population
                for _ in range(avg_count - 1):
                    new_result = self._simulator.simulate_rabi(
                        t_max=np.max(rabi_times),
                        n_points=len(rabi_times),
                        mw_frequency=mw_freq,
                        mw_power=mw_power
                    )
                    signal += new_result.populations[:, 0]
                
                # Update with averaged signal
                result.populations[:, 0] = signal / avg_count
        
        # Or implement using pulse sequences
        else:
            # Create storage for results
            signal = np.zeros(len(rabi_times))
            
            # Set up initial state
            self._simulator.initialize_state()
            
            # Configure microwave for this frequency
            self._simulator.set_microwave(mw_freq, mw_power, True)
            
            for i, pulse_time in enumerate(rabi_times):
                this_signal = 0
                
                for _ in range(avg_count):
                    # Initialize with laser
                    self._simulator.apply_laser(laser_power, True, self._params['initialization_time'])
                    
                    # Apply microwave pulse
                    if pulse_time > 0:  # Skip if time is zero
                        self._simulator.apply_microwave_pulse(pulse_time, mw_freq, mw_power)
                    
                    # Readout
                    readout = self._simulator.read_state(self._params['readout_time'])
                    this_signal += readout.signal
                    
                    # Wait for system to reset
                    self._simulator.wait(self._params['repetition_time'])
                
                signal[i] = this_signal / avg_count
            
            # Create a result object
            result = type('RabiResult', (), {
                'times': rabi_times,
                'populations': np.zeros((len(rabi_times), 3)),
                'experiment_type': 'rabi',
                'parameters': self._params.copy()
            })
            result.populations[:, 0] = signal
        
        # Add realistic noise if specified
        if noise_level > 0:
            result.populations[:, 0] = add_noise(result.populations[:, 0], noise_level)
        
        # Convert to Qudi-compatible format
        qudi_result = convert_to_qudi_format(result, 'rabi')
        qudi_result['parameters'] = self._params.copy()
        
        return qudi_result