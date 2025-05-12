# -*- coding: utf-8 -*-

"""
Standard dynamical decoupling sequence generators.

This module provides functions to create standard dynamical decoupling
sequences used in NV center quantum experiments.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union

from .base_sequence import DynamicalDecouplingSequence, PulseParameters

def create_hahn_echo(nv_system: Any,
                    tau: float,
                    pulse_params: Optional[PulseParameters] = None) -> DynamicalDecouplingSequence:
    """
    Create a Hahn Echo sequence (π/2 - τ - π - τ - π/2).
    
    Parameters:
    -----------
    nv_system : Any
        NV system object (compatible with SimOS)
    tau : float
        Delay time between pulses in seconds
    pulse_params : PulseParameters, optional
        Pulse parameters configuration
        
    Returns:
    --------
    DynamicalDecouplingSequence
        Hahn Echo sequence
    """
    # Create sequence
    seq = DynamicalDecouplingSequence(nv_system, pulse_params)
    seq.name = "Hahn Echo"
    
    # Add initial π/2 pulse
    seq.add_pulse('x', np.pi/2)
    
    # Add delay
    seq.add_delay(tau)
    
    # Add π pulse
    seq.add_pulse('x', np.pi)
    
    # Add delay
    seq.add_delay(tau)
    
    # Add final π/2 pulse for readout
    seq.add_pulse('x', np.pi/2)
    
    # Add measurement
    seq.add_measurement()
    
    return seq

def create_cpmg(nv_system: Any,
               tau: float,
               n_pulses: int,
               pulse_params: Optional[PulseParameters] = None) -> DynamicalDecouplingSequence:
    """
    Create a Carr-Purcell-Meiboom-Gill (CPMG) sequence.
    
    The CPMG-n sequence consists of:
    (π/2)x - [τ - (π)y - τ]^n - (π/2)x
    
    Parameters:
    -----------
    nv_system : Any
        NV system object (compatible with SimOS)
    tau : float
        Delay time between pulses in seconds
    n_pulses : int
        Number of π pulses in the sequence
    pulse_params : PulseParameters, optional
        Pulse parameters configuration
        
    Returns:
    --------
    DynamicalDecouplingSequence
        CPMG sequence
    """
    # Create sequence
    seq = DynamicalDecouplingSequence(nv_system, pulse_params)
    seq.name = f"CPMG-{n_pulses}"
    
    # Add initial π/2 pulse along x
    seq.add_pulse('x', np.pi/2)
    
    # Add n π pulses along y with delays
    for i in range(n_pulses):
        seq.add_delay(tau)
        seq.add_pulse('y', np.pi)
        if i < n_pulses - 1 or n_pulses % 2 == 0:
            seq.add_delay(tau)
    
    # Add final π/2 pulse for readout (phase depends on n)
    seq.add_pulse('x', np.pi/2)
    
    # Add measurement
    seq.add_measurement()
    
    return seq

def create_xy4(nv_system: Any,
              tau: float,
              repetitions: int = 1,
              pulse_params: Optional[PulseParameters] = None) -> DynamicalDecouplingSequence:
    """
    Create an XY4 dynamical decoupling sequence.
    
    The XY4 sequence consists of:
    (π/2)x - [τ - (π)x - τ - (π)y - τ - (π)x - τ - (π)y]^n - (π/2)x
    
    Parameters:
    -----------
    nv_system : Any
        NV system object (compatible with SimOS)
    tau : float
        Delay time between pulses in seconds
    repetitions : int
        Number of repetitions of the basic XY4 block
    pulse_params : PulseParameters, optional
        Pulse parameters configuration
        
    Returns:
    --------
    DynamicalDecouplingSequence
        XY4 sequence
    """
    # Create sequence
    seq = DynamicalDecouplingSequence(nv_system, pulse_params)
    seq.name = f"XY4-{repetitions}"
    
    # Add initial π/2 pulse
    seq.add_pulse('x', np.pi/2)
    
    # Add repetitions of XY4 block
    for _ in range(repetitions):
        # XY4 block: X-Y-X-Y
        for axis in ['x', 'y', 'x', 'y']:
            seq.add_delay(tau)
            seq.add_pulse(axis, np.pi)
            # No delay after last pulse if it's the final repetition
            if not (_ == repetitions - 1 and axis == 'y'):
                seq.add_delay(tau)
    
    # Add final π/2 pulse for readout
    seq.add_pulse('x', np.pi/2)
    
    # Add measurement
    seq.add_measurement()
    
    return seq

def create_xy8(nv_system: Any,
              tau: float,
              repetitions: int = 1,
              pulse_params: Optional[PulseParameters] = None) -> DynamicalDecouplingSequence:
    """
    Create an XY8 dynamical decoupling sequence.
    
    The XY8 sequence consists of:
    (π/2)x - [τ - (π)x - τ - (π)y - τ - (π)x - τ - (π)y - τ - (π)y - τ - (π)x - τ - (π)y - τ - (π)x]^n - (π/2)x
    
    Parameters:
    -----------
    nv_system : Any
        NV system object (compatible with SimOS)
    tau : float
        Delay time between pulses in seconds
    repetitions : int
        Number of repetitions of the basic XY8 block
    pulse_params : PulseParameters, optional
        Pulse parameters configuration
        
    Returns:
    --------
    DynamicalDecouplingSequence
        XY8 sequence
    """
    # Create sequence
    seq = DynamicalDecouplingSequence(nv_system, pulse_params)
    seq.name = f"XY8-{repetitions}"
    
    # Add initial π/2 pulse
    seq.add_pulse('x', np.pi/2)
    
    # Add repetitions of XY8 block
    for _ in range(repetitions):
        # XY8 block: X-Y-X-Y-Y-X-Y-X
        for axis in ['x', 'y', 'x', 'y', 'y', 'x', 'y', 'x']:
            seq.add_delay(tau)
            seq.add_pulse(axis, np.pi)
            # No delay after last pulse if it's the final repetition
            if not (_ == repetitions - 1 and axis == 'x'):
                seq.add_delay(tau)
    
    # Add final π/2 pulse for readout
    seq.add_pulse('x', np.pi/2)
    
    # Add measurement
    seq.add_measurement()
    
    return seq

def create_xy16(nv_system: Any,
               tau: float,
               repetitions: int = 1,
               pulse_params: Optional[PulseParameters] = None) -> DynamicalDecouplingSequence:
    """
    Create an XY16 dynamical decoupling sequence.
    
    The XY16 sequence consists of:
    (π/2)x - [τ - (π)x - τ - (π)y - τ - (π)x - τ - (π)y - τ - (π)y - τ - (π)x - τ - (π)y - τ - (π)x 
              - τ - (π)x - τ - (π)y - τ - (π)x - τ - (π)y - τ - (π)y - τ - (π)x - τ - (π)y - τ - (π)x]^n - (π/2)x
    
    Parameters:
    -----------
    nv_system : Any
        NV system object (compatible with SimOS)
    tau : float
        Delay time between pulses in seconds
    repetitions : int
        Number of repetitions of the basic XY16 block
    pulse_params : PulseParameters, optional
        Pulse parameters configuration
        
    Returns:
    --------
    DynamicalDecouplingSequence
        XY16 sequence
    """
    # Create sequence
    seq = DynamicalDecouplingSequence(nv_system, pulse_params)
    seq.name = f"XY16-{repetitions}"
    
    # Add initial π/2 pulse
    seq.add_pulse('x', np.pi/2)
    
    # Add repetitions of XY16 block
    for _ in range(repetitions):
        # XY16 is two XY8 blocks with the second one phase-shifted by π
        # XY8 block 1: X-Y-X-Y-Y-X-Y-X
        for axis in ['x', 'y', 'x', 'y', 'y', 'x', 'y', 'x']:
            seq.add_delay(tau)
            seq.add_pulse(axis, np.pi)
            seq.add_delay(tau)
            
        # XY8 block 2: X-Y-X-Y-Y-X-Y-X
        for axis in ['x', 'y', 'x', 'y', 'y', 'x', 'y', 'x']:
            seq.add_delay(tau)
            seq.add_pulse(axis, np.pi)
            # No delay after last pulse if it's the final repetition
            if not (_ == repetitions - 1 and axis == 'x'):
                seq.add_delay(tau)
    
    # Add final π/2 pulse for readout
    seq.add_pulse('x', np.pi/2)
    
    # Add measurement
    seq.add_measurement()
    
    return seq

def create_kdd(nv_system: Any,
              tau: float,
              repetitions: int = 1,
              pulse_params: Optional[PulseParameters] = None) -> DynamicalDecouplingSequence:
    """
    Create a KDD (Knill Dynamical Decoupling) sequence.
    
    The KDD sequence uses composite π pulses to improve robustness against pulse errors.
    Each π pulse is replaced by a composite pulse: (π/2)x - (π)y - (π/2)x
    
    Parameters:
    -----------
    nv_system : Any
        NV system object (compatible with SimOS)
    tau : float
        Delay time between pulses in seconds
    repetitions : int
        Number of repetitions of the basic KDD block
    pulse_params : PulseParameters, optional
        Pulse parameters configuration
        
    Returns:
    --------
    DynamicalDecouplingSequence
        KDD sequence
    """
    # Create sequence
    seq = DynamicalDecouplingSequence(nv_system, pulse_params)
    seq.name = f"KDD-{repetitions}"
    
    # Add initial π/2 pulse
    seq.add_pulse('x', np.pi/2)
    
    # Define KDD block phases (x, y, x, y, x)
    phases = ['x', 'y', 'x', 'y', 'x']
    
    # Add repetitions of KDD block
    for _ in range(repetitions):
        for phase in phases:
            # Add delay
            seq.add_delay(tau)
            
            # Add KDD composite pulse
            # For X-phase: (π/2)y - (π)x - (π/2)y
            # For Y-phase: (π/2)x - (π)y - (π/2)x
            if phase == 'x':
                seq.add_pulse('y', np.pi/2)
                seq.add_pulse('x', np.pi)
                seq.add_pulse('y', np.pi/2)
            else:  # phase == 'y'
                seq.add_pulse('x', np.pi/2)
                seq.add_pulse('y', np.pi)
                seq.add_pulse('x', np.pi/2)
                
            # Add delay
            if not (_ == repetitions - 1 and phase == phases[-1]):
                seq.add_delay(tau)
    
    # Add final π/2 pulse for readout
    seq.add_pulse('x', np.pi/2)
    
    # Add measurement
    seq.add_measurement()
    
    return seq

def create_concatenated_dd(nv_system: Any,
                         tau: float,
                         level: int,
                         pulse_params: Optional[PulseParameters] = None) -> DynamicalDecouplingSequence:
    """
    Create a Concatenated Dynamical Decoupling (CDD) sequence.
    
    The CDD sequence is defined recursively:
    CDD_1 = τ - (π)x - τ
    CDD_n = CDD_{n-1} - (π)y - CDD_{n-1}
    
    Parameters:
    -----------
    nv_system : Any
        NV system object (compatible with SimOS)
    tau : float
        Delay time between pulses in seconds
    level : int
        Concatenation level (1 = Hahn echo, 2 = XY4, etc.)
    pulse_params : PulseParameters, optional
        Pulse parameters configuration
        
    Returns:
    --------
    DynamicalDecouplingSequence
        CDD sequence
    """
    # Create sequence
    seq = DynamicalDecouplingSequence(nv_system, pulse_params)
    seq.name = f"CDD-{level}"
    
    # Add initial π/2 pulse
    seq.add_pulse('x', np.pi/2)
    
    # Generate CDD sequence recursively
    def add_cdd_level(level, axis):
        if level == 1:
            # Base case: Hahn echo
            seq.add_delay(tau)
            seq.add_pulse(axis, np.pi)
            seq.add_delay(tau)
        else:
            # Recursive case: alternate axes
            next_axis = 'y' if axis == 'x' else 'x'
            add_cdd_level(level-1, next_axis)
            seq.add_pulse(axis, np.pi)
            add_cdd_level(level-1, next_axis)
    
    # Add CDD sequence
    add_cdd_level(level, 'x')
    
    # Add final π/2 pulse for readout
    seq.add_pulse('x', np.pi/2)
    
    # Add measurement
    seq.add_measurement()
    
    return seq

def create_custom_sequence(nv_system: Any,
                          operations: List[Dict[str, Any]],
                          pulse_params: Optional[PulseParameters] = None,
                          name: str = "Custom Sequence") -> DynamicalDecouplingSequence:
    """
    Create a custom dynamical decoupling sequence.
    
    Parameters:
    -----------
    nv_system : Any
        NV system object (compatible with SimOS)
    operations : List[Dict[str, Any]]
        List of operations, each specified as a dictionary
        - Pulse: {'type': 'pulse', 'axis': 'x', 'angle': np.pi, 'duration': 50e-9}
        - Delay: {'type': 'delay', 'duration': 1e-6}
        - Measurement: {'type': 'measurement', 'axis': 'z'}
    pulse_params : PulseParameters, optional
        Pulse parameters configuration
    name : str
        Name of the custom sequence
        
    Returns:
    --------
    DynamicalDecouplingSequence
        Custom sequence
    """
    # Create sequence
    seq = DynamicalDecouplingSequence(nv_system, pulse_params)
    seq.name = name
    
    # Add operations
    for op in operations:
        op_type = op['type']
        
        if op_type == 'pulse':
            # Add pulse
            seq.add_pulse(
                axis=op['axis'],
                angle=op['angle'],
                duration=op.get('duration'),
                shape=op.get('shape')
            )
        elif op_type == 'delay':
            # Add delay
            seq.add_delay(op['duration'])
        elif op_type == 'measurement':
            # Add measurement
            seq.add_measurement(
                axis=op.get('axis'),
                duration=op.get('duration', 0.0)
            )
        else:
            raise ValueError(f"Unknown operation type: {op_type}")
    
    # Ensure there's a measurement at the end
    if not any(op['type'] == 'measurement' for op in operations):
        seq.add_measurement()
    
    return seq

def calculate_sequence_filter_function(sequence: DynamicalDecouplingSequence,
                                     omega: np.ndarray) -> np.ndarray:
    """
    Calculate the filter function for a dynamical decoupling sequence.
    
    The filter function describes how the sequence filters environmental noise
    at different frequencies.
    
    Parameters:
    -----------
    sequence : DynamicalDecouplingSequence
        Dynamical decoupling sequence
    omega : np.ndarray
        Angular frequencies at which to calculate the filter function
        
    Returns:
    --------
    np.ndarray
        Filter function values at the specified frequencies
    """
    # Get sequence elements
    elements = sequence.sequence
    
    # Extract pulse times and phases
    pulse_times = []
    pulse_phases = []
    
    current_time = 0.0
    for element in elements:
        if element['type'] == 'pulse' and abs(abs(element['angle']) - np.pi) < 1e-6:
            # Add π pulse
            pulse_time = current_time + element['duration'] / 2  # Use middle of pulse
            
            # Determine phase based on axis
            if element['axis'] == 'x':
                phase = 0.0
            elif element['axis'] == 'y':
                phase = np.pi/2
            elif element['axis'] == '-x':
                phase = np.pi
            elif element['axis'] == '-y':
                phase = 3*np.pi/2
            else:
                # Skip pulses with other axes
                current_time += element['duration']
                continue
                
            pulse_times.append(pulse_time)
            pulse_phases.append(phase)
            
        # Update current time
        current_time += element.get('duration', 0.0)
    
    # Total sequence time
    T = sequence.total_duration
    
    # Calculate filter function
    F = np.zeros_like(omega)
    
    # For each frequency
    for i, w in enumerate(omega):
        if abs(w) < 1e-10:
            # Filter function is zero at DC
            F[i] = 0.0
            continue
            
        # Calculate modulation function
        sum_cos = 0.0
        sum_sin = 0.0
        
        for t, phi in zip(pulse_times, pulse_phases):
            # Effect of each π pulse
            sum_cos += np.cos(w*t) * np.cos(phi) - np.sin(w*t) * np.sin(phi)
            sum_sin += np.sin(w*t) * np.cos(phi) + np.cos(w*t) * np.sin(phi)
        
        # Add final readout pulse effect
        sum_cos += np.cos(w*T)
        sum_sin += np.sin(w*T)
        
        # Calculate filter function
        F[i] = 2.0 * (sum_cos**2 + sum_sin**2) / (w**2)
    
    return F