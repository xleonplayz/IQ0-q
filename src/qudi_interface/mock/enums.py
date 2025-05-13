# -*- coding: utf-8 -*-
"""
Mock implementation of Qudi enums for standalone testing.

Copyright (c) 2023
"""

from enum import Enum, auto, IntEnum


class SamplingOutputMode(IntEnum):
    """
    Defines the type of frequency scan performed by a microwave source.
    """
    JUMP_LIST = 0
    EQUIDISTANT_SWEEP = 1