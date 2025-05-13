"""
Nuclear spin environment module for the NV center simulator.

This module provides classes and functions for simulating nuclear spin
environments around an NV center, including hyperfine interactions,
decoherence effects, and advanced sensing protocols.
"""

# Import main classes for easy access
from .spin_bath import NuclearSpinBath, SpinConfig
from .hyperfine import HyperfineCalculator
from .nuclear_control import NuclearControl
from .decoherence_models import SpinBathDecoherence

__all__ = [
    'NuclearSpinBath',
    'SpinConfig', 
    'HyperfineCalculator',
    'NuclearControl',
    'SpinBathDecoherence'
]