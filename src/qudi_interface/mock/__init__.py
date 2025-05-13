# -*- coding: utf-8 -*-
"""
Mock module providing stubs for Qudi interfaces to enable standalone testing.
This allows the simulator to be tested without having a full Qudi installation.

Copyright (c) 2023
"""

from .core import Base, Module
from .interfaces import (
    MicrowaveInterface, MicrowaveConstraints,
    FastCounterInterface,
    PulserInterface, PulserConstraints,
    ScanningProbeInterface, ScannerAxis, ScannerChannel, ScanConstraints, ScannerSettings, BackScanCapability,
    SimpleLaserInterface, ControlMode, ShutterState, LaserState
)
from .enums import SamplingOutputMode

__all__ = [
    'Base',
    'Module',
    'MicrowaveInterface',
    'MicrowaveConstraints',
    'FastCounterInterface',
    'PulserInterface',
    'PulserConstraints',
    'ScanningProbeInterface',
    'ScannerAxis',
    'ScannerChannel',
    'ScanConstraints',
    'ScannerSettings',
    'BackScanCapability',
    'SimpleLaserInterface',
    'ControlMode',
    'ShutterState',
    'LaserState',
    'SamplingOutputMode'
]