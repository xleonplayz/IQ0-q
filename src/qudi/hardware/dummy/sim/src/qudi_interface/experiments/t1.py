# -*- coding: utf-8 -*-

"""
T1 relaxation experiment mode for the NV center simulator.

Copyright (c) 2023
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np

from .base import ExperimentMode
from .utils import convert_to_qudi_format, add_noise


class T1Mode(ExperimentMode):
    """
    T1 relaxation experiment mode for the NV center simulator.
    Simulates spin relaxation over varying wait times after initialization.
    """
    
    def __init__(self, simulator):
        """
        Initialize the T1 experiment mode.
        
        @param simulator: PhysicalNVModel instance to run the experiment on
        """
        super().__init__(simulator)
        self._default_params = {
            'tau_times': np.logspace(-7, -3, 51),       # s, log spaced from 100ns to 1ms
            'avg_count': 3,                            # Number of averages
            'laser_power': 1.0,                        # Relative laser power (0 to 1)
            'noise_level': 0.01,                       # Relative noise level
            'readout_time': 300e-9,                    # Readout window duration (s)
            'initialization_time': 3e-6,               # Laser initialization time (s)
            'repetition_time': 100e-6                  # Time between sequence repetitions (s)
        }
        self._params = self._default_params.copy()
    
    def run(self) -> Dict[str, Any]:
        """
        Run the T1 experiment and return results.
        
        @return: T1 experiment results in Qudi-compatible format
        """
        # Extract parameters
        tau_times = self._params['tau_times']
        avg_count = self._params['avg_count']
        laser_power = self._params['laser_power']
        noise_level = self._params['noise_level']
        
        # Set laser power in simulator
        self._simulator.apply_laser(laser_power, True)
        
        # Use built-in simulator function if available
        if hasattr(self._simulator, 'simulate_t1'):
            result = self._simulator.simulate_t1(
                t_max=np.max(tau_times),
                n_points=len(tau_times)
            )
            
            # Average multiple runs if requested
            if avg_count > 1:
                signal = result.populations[:, 0]  # ms=0 population
                for _ in range(avg_count - 1):
                    new_result = self._simulator.simulate_t1(
                        t_max=np.max(tau_times),
                        n_points=len(tau_times)
                    )
                    signal += new_result.populations[:, 0]
                
                # Update with averaged signal
                result.populations[:, 0] = signal / avg_count
        
        # Or implement using basic operations
        else:
            # Create storage for results
            signal = np.zeros(len(tau_times))
            
            for i, tau in enumerate(tau_times):
                this_signal = 0
                
                for _ in range(avg_count):
                    # Initialize with laser to ms=0 state
                    self._simulator.apply_laser(laser_power, True, self._params['initialization_time'])
                    
                    # Wait for varying relaxation time
                    if tau > 0:  # Skip if time is zero
                        self._simulator.wait(tau)
                    
                    # Readout
                    readout = self._simulator.read_state(self._params['readout_time'])
                    this_signal += readout.signal
                    
                    # Wait for system to reset
                    self._simulator.wait(self._params['repetition_time'])
                
                signal[i] = this_signal / avg_count
            
            # Create a result object
            result = type('T1Result', (), {
                'times': tau_times,
                'populations': np.zeros((len(tau_times), 3)),
                'experiment_type': 't1',
                'parameters': self._params.copy()
            })
            result.populations[:, 0] = signal
        
        # Add realistic noise if specified
        if noise_level > 0:
            result.populations[:, 0] = add_noise(result.populations[:, 0], noise_level)
        
        # Convert to Qudi-compatible format
        qudi_result = convert_to_qudi_format(result, 't1')
        qudi_result['parameters'] = self._params.copy()
        
        return qudi_result