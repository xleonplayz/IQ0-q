# -*- coding: utf-8 -*-

"""
Spin Echo experiment mode for the NV center simulator.

Copyright (c) 2023
"""

from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np

from .base import ExperimentMode
from .utils import convert_to_qudi_format, add_noise


class SpinEchoMode(ExperimentMode):
    """
    Spin Echo (Hahn Echo) experiment mode for the NV center simulator.
    Simulates the Hahn echo sequence: π/2 - τ - π - τ - π/2.
    """
    
    def __init__(self, simulator):
        """
        Initialize the Spin Echo experiment mode.
        
        @param simulator: PhysicalNVModel instance to run the experiment on
        """
        super().__init__(simulator)
        self._default_params = {
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
            'repetition_time': 10e-6                     # Time between sequence repetitions (s)
        }
        self._params = self._default_params.copy()
    
    def run(self) -> Dict[str, Any]:
        """
        Run the Spin Echo experiment and return results.
        
        @return: Spin Echo experiment results in Qudi-compatible format
        """
        # Extract parameters
        tau_times = self._params['tau_times']
        mw_freq = self._params['mw_frequency']
        mw_power = self._params['mw_power']
        avg_count = self._params['avg_count']
        laser_power = self._params['laser_power']
        noise_level = self._params['noise_level']
        pi_half_time = self._params['pi_half_time']
        pi_time = self._params['pi_time']
        
        # Set laser power in simulator
        self._simulator.apply_laser(laser_power, True)
        
        # Use built-in simulator function if available (might be named t2_echo)
        if hasattr(self._simulator, 'simulate_t2_echo'):
            result = self._simulator.simulate_t2_echo(
                t_max=2*np.max(tau_times),  # t_max is the full sequence time (2*tau)
                n_points=len(tau_times),
                mw_frequency=mw_freq,
                mw_power=mw_power
            )
            
            # Average multiple runs if requested
            if avg_count > 1:
                coherence = result.coherence.copy()
                for _ in range(avg_count - 1):
                    new_result = self._simulator.simulate_t2_echo(
                        t_max=2*np.max(tau_times),
                        n_points=len(tau_times),
                        mw_frequency=mw_freq,
                        mw_power=mw_power
                    )
                    coherence += new_result.coherence
                
                # Update with averaged signal
                result.coherence = coherence / avg_count
        
        # Or implement using pulse sequences
        else:
            # Create storage for results
            signal = np.zeros(len(tau_times))
            
            # Configure microwave
            self._simulator.set_microwave(mw_freq, mw_power, True)
            
            for i, tau in enumerate(tau_times):
                this_signal = 0
                
                for _ in range(avg_count):
                    # Initialize with laser
                    self._simulator.apply_laser(laser_power, True, self._params['initialization_time'])
                    
                    # First π/2 pulse (X axis)
                    self._simulator.apply_microwave_pulse(pi_half_time, mw_freq, mw_power)
                    
                    # First free evolution period
                    self._simulator.wait(tau)
                    
                    # π pulse (X axis) - refocusing pulse
                    self._simulator.apply_microwave_pulse(pi_time, mw_freq, mw_power)
                    
                    # Second free evolution period
                    self._simulator.wait(tau)
                    
                    # Final π/2 pulse (X axis)
                    self._simulator.apply_microwave_pulse(pi_half_time, mw_freq, mw_power)
                    
                    # Readout
                    readout = self._simulator.read_state(self._params['readout_time'])
                    this_signal += readout.signal
                    
                    # Wait for system to reset
                    self._simulator.wait(self._params['repetition_time'])
                
                signal[i] = this_signal / avg_count
            
            # Create a result object
            result = type('SpinEchoResult', (), {
                'times': tau_times,
                'coherence': signal,
                'experiment_type': 'spin_echo',
                'parameters': self._params.copy()
            })
        
        # Add realistic noise if specified
        if noise_level > 0:
            result.coherence = add_noise(result.coherence, noise_level)
        
        # Convert to Qudi-compatible format
        qudi_result = convert_to_qudi_format(result, 'spin_echo')
        qudi_result['parameters'] = self._params.copy()
        
        return qudi_result