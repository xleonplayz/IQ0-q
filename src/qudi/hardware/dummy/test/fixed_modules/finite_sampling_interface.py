#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mock finite sampling interface for testing.

Copyright (c) 2023, IQO

This file is part of qudi.

Qudi is free software: you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

Qudi is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with qudi.
If not, see <https://www.gnu.org/licenses/>.
"""

from fixed_modules.qudi_core import Base


class FiniteSamplingInputConstraints:
    """
    Container class to hold constraints for a finite sampling input device.
    """
    
    def __init__(self, channel_units=None, frame_size_limits=None, sample_rate_limits=None):
        """
        Initialize finite sampling constraints with given parameters.
        
        @param dict channel_units: Dictionary of {channel_name: unit}
        @param tuple frame_size_limits: (min, max) number of samples per frame
        @param tuple sample_rate_limits: (min, max) sample rate in Hz
        """
        self.channel_units = channel_units if channel_units is not None else {}
        self.frame_size_limits = frame_size_limits if frame_size_limits is not None else (1, 1e6)
        self.sample_rate_limits = sample_rate_limits if sample_rate_limits is not None else (1, 1e6)
        
    @property
    def min_frame_size(self):
        """Minimum frame size."""
        return self.frame_size_limits[0]
        
    @property
    def max_frame_size(self):
        """Maximum frame size."""
        return self.frame_size_limits[1]
        
    @property
    def min_sample_rate(self):
        """Minimum sample rate."""
        return self.sample_rate_limits[0]
        
    @property
    def max_sample_rate(self):
        """Maximum sample rate."""
        return self.sample_rate_limits[1]
        
    @property
    def channels(self):
        """Available channels."""
        return list(self.channel_units.keys())
        
    def test_configuration(self, active_channels, sample_rate, frame_size):
        """
        Test if a configuration is valid.
        
        @param list active_channels: List of channel names
        @param float sample_rate: Sample rate in Hz
        @param int frame_size: Number of samples per frame
        
        @raises ValueError: If configuration is invalid
        """
        # Check channels
        for channel in active_channels:
            if channel not in self.channel_units:
                raise ValueError(f"Channel '{channel}' not supported")
                
        # Check sample rate
        if not self.min_sample_rate <= sample_rate <= self.max_sample_rate:
            raise ValueError(f"Sample rate {sample_rate:.4f} Hz out of bounds ({self.min_sample_rate:.4f}, {self.max_sample_rate:.4f})")
            
        # Check frame size
        if not self.min_frame_size <= frame_size <= self.max_frame_size:
            raise ValueError(f"Frame size {frame_size} out of bounds ({self.min_frame_size}, {self.max_frame_size})")


class FiniteSamplingInputInterface(Base):
    """
    Mock interface for finite sampling input devices.
    """
    
    def __init__(self, qudi_main_weakref=None, name=None, **kwargs):
        """
        Initialize finite sampling input interface.
        
        @param qudi_main_weakref: Weakref to Qudi main object
        @param name: Unique name for this module instance
        """
        super().__init__(qudi_main_weakref=qudi_main_weakref, name=name, **kwargs)