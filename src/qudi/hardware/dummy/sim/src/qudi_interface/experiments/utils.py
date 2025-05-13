# -*- coding: utf-8 -*-

"""
Utility functions for experiment modes in the NV center simulator.

Copyright (c) 2023
"""

from typing import Dict, Any, Union, List, Tuple
import numpy as np


def convert_to_qudi_format(simulator_result, experiment_type: str) -> Dict[str, Any]:
    """
    Convert simulator results to Qudi-compatible format.
    
    @param simulator_result: Result object from simulator
    @param experiment_type: Type of experiment ('odmr', 'rabi', etc.)
    
    @return: Dictionary with Qudi-compatible data structure
    """
    if experiment_type == 'odmr':
        return {
            'frequencies': simulator_result.frequencies,
            'odmr_signal': simulator_result.signal,
            'raw_data': simulator_result.__dict__
        }
    
    elif experiment_type == 'rabi':
        return {
            'times': simulator_result.times,
            'signal': simulator_result.populations[:, 0],  # ms=0 population
            'raw_data': simulator_result.__dict__
        }
    
    elif experiment_type == 'ramsey':
        return {
            'times': simulator_result.times,
            'signal': simulator_result.coherence,
            'raw_data': simulator_result.__dict__
        }
    
    elif experiment_type == 'spin_echo':
        return {
            'times': simulator_result.times,
            'signal': simulator_result.coherence,
            'raw_data': simulator_result.__dict__
        }
    
    elif experiment_type == 't1':
        return {
            'times': simulator_result.times,
            'signal': simulator_result.populations[:, 0],  # ms=0 population
            'raw_data': simulator_result.__dict__
        }
    
    elif experiment_type == 'custom_sequence':
        return {
            'times': simulator_result.times,
            'signal': simulator_result.signal,
            'raw_data': simulator_result.__dict__
        }
    
    # Default case - return the entire result dictionary
    if hasattr(simulator_result, '__dict__'):
        return simulator_result.__dict__
    return {'result': simulator_result}


def add_noise(signal: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
    """
    Add Gaussian noise to a signal.
    
    @param signal: Input signal array
    @param noise_level: Standard deviation of the noise relative to signal amplitude
    
    @return: Noisy signal
    """
    amplitude = np.max(signal) - np.min(signal)
    if amplitude == 0:
        amplitude = 1.0
    
    noise = np.random.normal(0, noise_level * amplitude, signal.shape)
    return signal + noise