#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mock microwave interface for testing.

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

from enum import Enum, auto


class SamplingOutputMode(Enum):
    """Enum containing sample output modes."""
    JUMP_LIST = auto()
    EQUIDISTANT_SWEEP = auto()


class MicrowaveConstraints:
    """
    Container class to hold constraints for a microwave device.
    """
    
    def __init__(self, power_limits=None, frequency_limits=None, scan_size_limits=None,
                 sample_rate_limits=None, scan_modes=None):
        """
        Initialize microwave constraints with given parameters.
        
        @param tuple power_limits: (min, max) power in dBm
        @param tuple frequency_limits: (min, max) frequency in Hz
        @param tuple scan_size_limits: (min, max) number of scan points
        @param tuple sample_rate_limits: (min, max) sample rate in Hz
        @param tuple scan_modes: supported scan modes
        """
        self.power_limits = power_limits if power_limits is not None else (-100, 30)
        self.frequency_limits = frequency_limits if frequency_limits is not None else (1e6, 20e9)
        self.scan_size_limits = scan_size_limits if scan_size_limits is not None else (2, 1000)
        self.sample_rate_limits = sample_rate_limits if sample_rate_limits is not None else (1, 1000)
        self.scan_modes = scan_modes if scan_modes is not None else (
            SamplingOutputMode.JUMP_LIST,
            SamplingOutputMode.EQUIDISTANT_SWEEP
        )
        
    @property
    def min_power(self):
        """Minimum output power."""
        return self.power_limits[0]
        
    @property
    def max_power(self):
        """Maximum output power."""
        return self.power_limits[1]
        
    @property
    def min_frequency(self):
        """Minimum output frequency."""
        return self.frequency_limits[0]
        
    @property
    def max_frequency(self):
        """Maximum output frequency."""
        return self.frequency_limits[1]
        
    @property
    def min_scansize(self):
        """Minimum scan size."""
        return self.scan_size_limits[0]
        
    @property
    def max_scansize(self):
        """Maximum scan size."""
        return self.scan_size_limits[1]


from fixed_modules.qudi_core import Base


class MicrowaveInterface(Base):
    """
    Mock interface for controlling a microwave source.
    """
    
    def __init__(self, qudi_main_weakref=None, name=None, **kwargs):
        """
        Initialize microwave interface.
        
        @param qudi_main_weakref: Weakref to Qudi main object
        @param name: Unique name for this module instance
        """
        super().__init__(qudi_main_weakref=qudi_main_weakref, name=name, **kwargs)
    
    def _assert_cw_parameters_args(self, frequency, power):
        """
        Helper to check CW parameters.
        
        @param float frequency: frequency in Hz
        @param float power: power in dBm
        """
        constraints = self.constraints
        if not constraints.min_frequency <= frequency <= constraints.max_frequency:
            raise ValueError(f"Frequency {frequency:.4e} Hz out of bounds ({constraints.min_frequency:.4e}, {constraints.max_frequency:.4e})")
            
        if not constraints.min_power <= power <= constraints.max_power:
            raise ValueError(f"Power {power:.4f} dBm out of bounds ({constraints.min_power:.4f}, {constraints.max_power:.4f})")
            
    def _assert_scan_configuration_args(self, power, frequencies, mode, sample_rate):
        """
        Helper to check scan parameters.
        
        @param float power: power in dBm
        @param frequencies: frequencies in Hz
        @param SamplingOutputMode mode: scan mode
        @param float sample_rate: sample rate in Hz
        """
        constraints = self.constraints
        
        # Check power
        if not constraints.min_power <= power <= constraints.max_power:
            raise ValueError(f"Power {power:.4f} dBm out of bounds ({constraints.min_power:.4f}, {constraints.max_power:.4f})")
            
        # Check mode
        if mode not in constraints.scan_modes:
            raise ValueError(f"Mode {mode} not supported")
            
        # Check sample rate
        if not constraints.min_samplerate <= sample_rate <= constraints.max_samplerate:
            raise ValueError(f"Sample rate {sample_rate:.4f} Hz out of bounds ({constraints.min_samplerate:.4f}, {constraints.max_samplerate:.4f})")
            
        # Check frequencies depending on mode
        if mode == SamplingOutputMode.EQUIDISTANT_SWEEP:
            # frequencies should be (start, stop, num_points)
            if not isinstance(frequencies, (list, tuple)) or len(frequencies) != 3:
                raise ValueError(f"Equidistant sweep frequencies must be (start, stop, num_points)")
                
            start, stop, num_points = frequencies
            
            if not constraints.min_frequency <= start <= constraints.max_frequency:
                raise ValueError(f"Start frequency {start:.4e} Hz out of bounds ({constraints.min_frequency:.4e}, {constraints.max_frequency:.4e})")
                
            if not constraints.min_frequency <= stop <= constraints.max_frequency:
                raise ValueError(f"Stop frequency {stop:.4e} Hz out of bounds ({constraints.min_frequency:.4e}, {constraints.max_frequency:.4e})")
                
            if not constraints.min_scansize <= num_points <= constraints.max_scansize:
                raise ValueError(f"Number of points {num_points} out of bounds ({constraints.min_scansize}, {constraints.max_scansize})")
                
        else:  # JUMP_LIST
            # frequencies should be a list of frequencies
            if not hasattr(frequencies, '__iter__'):
                raise ValueError(f"Jump list frequencies must be iterable")
                
            num_points = len(frequencies)
            
            if not constraints.min_scansize <= num_points <= constraints.max_scansize:
                raise ValueError(f"Number of points {num_points} out of bounds ({constraints.min_scansize}, {constraints.max_scansize})")
                
            for freq in frequencies:
                if not constraints.min_frequency <= freq <= constraints.max_frequency:
                    raise ValueError(f"Frequency {freq:.4e} Hz out of bounds ({constraints.min_frequency:.4e}, {constraints.max_frequency:.4e})")