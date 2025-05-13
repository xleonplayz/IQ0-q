# -*- coding: utf-8 -*-

"""
Dynamical decoupling sequence module for NV center simulator.

This module provides a framework for creating and simulating quantum-accurate
dynamical decoupling pulse sequences for NV center spin manipulation.
"""

from .base_sequence import DynamicalDecouplingSequence, PulseParameters, PulseError
from .standard_sequences import (
    create_hahn_echo,
    create_cpmg,
    create_xy4,
    create_xy8,
    create_xy16,
    create_kdd,
    create_concatenated_dd
)
from .pulse_shapes import PulseShape, create_pulse_shape
from .sequence_analyzer import (
    analyze_sequence_fidelity,
    calculate_filter_function,
    error_susceptibility
)

__all__ = [
    'DynamicalDecouplingSequence',
    'PulseParameters',
    'PulseError',
    'create_hahn_echo',
    'create_cpmg',
    'create_xy4',
    'create_xy8',
    'create_xy16',
    'create_kdd',
    'create_concatenated_dd',
    'PulseShape',
    'create_pulse_shape',
    'analyze_sequence_fidelity',
    'calculate_filter_function',
    'error_susceptibility'
]