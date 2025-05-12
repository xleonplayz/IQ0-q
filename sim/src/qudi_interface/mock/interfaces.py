# -*- coding: utf-8 -*-
"""
Mock implementations of Qudi interfaces for standalone testing.
These are simplified versions of the Qudi interfaces that can be used
for testing the simulator without having Qudi installed.

Copyright (c) 2023
"""

from abc import abstractmethod
from enum import Enum, auto, Flag, IntEnum
from dataclasses import dataclass, field, asdict, replace
from typing import Tuple, Dict, Any, List, Set, Optional, Union, FrozenSet

import numpy as np

from .core import Base


class MicrowaveConstraints:
    """
    Mock of Qudi MicrowaveConstraints for testing without Qudi.
    """
    
    def __init__(self, power_limits=None, frequency_limits=None, scan_size_limits=None,
                sample_rate_limits=None, scan_modes=None):
        """
        Initialize with the given constraints.
        
        @param power_limits: Tuple of (min, max) power in dBm
        @param frequency_limits: Tuple of (min, max) frequency in Hz
        @param scan_size_limits: Tuple of (min, max) scan size in points
        @param sample_rate_limits: Tuple of (min, max) sample rate in Hz
        @param scan_modes: Tuple of supported scan modes
        """
        self.power_limits = power_limits if power_limits is not None else (-100, 30)
        self.frequency_limits = frequency_limits if frequency_limits is not None else (1e6, 20e9)
        self.scan_size_limits = scan_size_limits if scan_size_limits is not None else (2, 10000)
        self.sample_rate_limits = sample_rate_limits if sample_rate_limits is not None else (0.1, 1000)
        self.scan_modes = scan_modes if scan_modes is not None else (0, 1)


class MicrowaveInterface(Base):
    """
    Mock of Qudi MicrowaveInterface for testing without Qudi.
    """
    
    @property
    @abstractmethod
    def constraints(self):
        """The hardware constraints."""
        pass
    
    @property
    @abstractmethod
    def is_scanning(self):
        """Boolean flag indicating if a scan is running."""
        pass
    
    @property
    @abstractmethod
    def cw_power(self):
        """The current CW microwave power in dBm."""
        pass
    
    @property
    @abstractmethod
    def cw_frequency(self):
        """The current CW microwave frequency in Hz."""
        pass
    
    @property
    @abstractmethod
    def scan_power(self):
        """The current scan power in dBm."""
        pass
    
    @property
    @abstractmethod
    def scan_frequencies(self):
        """The current scan frequencies."""
        pass
    
    @property
    @abstractmethod
    def scan_mode(self):
        """The current scan mode."""
        pass
    
    @property
    @abstractmethod
    def scan_sample_rate(self):
        """The current scan sample rate in Hz."""
        pass
    
    @abstractmethod
    def off(self):
        """Turn off any microwave output."""
        pass
    
    @abstractmethod
    def set_cw(self, frequency, power):
        """Set the CW parameters."""
        pass
    
    @abstractmethod
    def cw_on(self):
        """Turn on the CW output."""
        pass
    
    @abstractmethod
    def configure_scan(self, power, frequencies, mode, sample_rate):
        """Configure a frequency scan."""
        pass
    
    @abstractmethod
    def start_scan(self):
        """Start the frequency scan."""
        pass
    
    @abstractmethod
    def reset_scan(self):
        """Reset the scan to the start frequency."""
        pass
    
    def _assert_cw_parameters_args(self, frequency, power):
        """Validation helper for CW parameters."""
        constraints = self.constraints
        if not constraints.frequency_limits[0] <= frequency <= constraints.frequency_limits[1]:
            raise ValueError(f"Frequency {frequency} Hz out of allowed range "
                           f"{constraints.frequency_limits}")
        if not constraints.power_limits[0] <= power <= constraints.power_limits[1]:
            raise ValueError(f"Power {power} dBm out of allowed range "
                           f"{constraints.power_limits}")
    
    def _assert_scan_configuration_args(self, power, frequencies, mode, sample_rate):
        """Validation helper for scan parameters."""
        constraints = self.constraints
        # Power check
        if not constraints.power_limits[0] <= power <= constraints.power_limits[1]:
            raise ValueError(f"Power {power} dBm out of allowed range "
                           f"{constraints.power_limits}")
        
        # Mode check
        if mode not in constraints.scan_modes:
            raise ValueError(f"Mode {mode} not in allowed modes {constraints.scan_modes}")
        
        # Sample rate check
        if not constraints.sample_rate_limits[0] <= sample_rate <= constraints.sample_rate_limits[1]:
            raise ValueError(f"Sample rate {sample_rate} Hz out of allowed range "
                           f"{constraints.sample_rate_limits}")
        
        # Frequencies check
        if mode == 0:  # JUMP_LIST
            if len(frequencies) < constraints.scan_size_limits[0] or len(frequencies) > constraints.scan_size_limits[1]:
                raise ValueError(f"Number of frequencies {len(frequencies)} out of allowed range "
                               f"{constraints.scan_size_limits}")
            for freq in frequencies:
                if not constraints.frequency_limits[0] <= freq <= constraints.frequency_limits[1]:
                    raise ValueError(f"Frequency {freq} Hz out of allowed range "
                                   f"{constraints.frequency_limits}")
        else:  # EQUIDISTANT_SWEEP
            if not len(frequencies) == 3:
                raise ValueError("For equidistant sweep, frequencies must be a tuple of 3 values: "
                               "(start, stop, points)")
            start, stop, points = frequencies
            if not constraints.frequency_limits[0] <= start <= constraints.frequency_limits[1]:
                raise ValueError(f"Start frequency {start} Hz out of allowed range "
                               f"{constraints.frequency_limits}")
            if not constraints.frequency_limits[0] <= stop <= constraints.frequency_limits[1]:
                raise ValueError(f"Stop frequency {stop} Hz out of allowed range "
                               f"{constraints.frequency_limits}")
            if not constraints.scan_size_limits[0] <= points <= constraints.scan_size_limits[1]:
                raise ValueError(f"Number of points {points} out of allowed range "
                               f"{constraints.scan_size_limits}")


class FastCounterInterface(Base):
    """
    Mock of Qudi FastCounterInterface for testing without Qudi.
    """
    
    @abstractmethod
    def get_constraints(self):
        """Get hardware constraints."""
        pass
    
    @abstractmethod
    def configure(self, bin_width_s, record_length_s, number_of_gates=0):
        """Configure the fast counter."""
        pass
    
    @abstractmethod
    def start_measure(self):
        """Start the fast counter measurement."""
        pass
    
    @abstractmethod
    def stop_measure(self):
        """Stop the fast counter measurement."""
        pass
    
    @abstractmethod
    def pause_measure(self):
        """Pause the fast counter measurement."""
        pass
    
    @abstractmethod
    def continue_measure(self):
        """Continue a paused measurement."""
        pass
    
    @abstractmethod
    def is_gated(self):
        """Check if the counter is in gated mode."""
        pass
    
    @abstractmethod
    def get_data_trace(self):
        """Get the current data trace."""
        pass


class PulserConstraints:
    """
    Mock of Qudi PulserConstraints for testing without Qudi.
    """
    
    def __init__(self):
        """Initialize with default constraints."""
        self.waveform_num = 0
        self.sequence_num = 0
        self.sequence_steps = 0
        self.waveform_length = None
        self.sample_rate = None
        self.a_ch_amplitude = None
        self.a_ch_offset = None
        self.d_ch_low = None
        self.d_ch_high = None
        self.activation_config = {}
        self.analog_channels = frozenset()
        self.digital_channels = frozenset()


class PulserInterface(Base):
    """
    Mock of Qudi PulserInterface for testing without Qudi.
    """
    
    @abstractmethod
    def get_constraints(self):
        """Get hardware constraints."""
        pass
    
    @abstractmethod
    def pulser_on(self):
        """Switches the pulsing device on."""
        pass
    
    @abstractmethod
    def pulser_off(self):
        """Switches the pulsing device off."""
        pass
    
    @abstractmethod
    def write_waveform(self, name, analog_samples, digital_samples, is_first_chunk, is_last_chunk):
        """Write a waveform to the device."""
        pass
    
    @abstractmethod
    def write_sequence(self, name, sequence_parameters):
        """Write a sequence to the device."""
        pass
    
    @abstractmethod
    def get_waveform_names(self):
        """Get all waveform names."""
        pass
    
    @abstractmethod
    def get_sequence_names(self):
        """Get all sequence names."""
        pass
    
    @abstractmethod
    def delete_waveform(self, waveform_name):
        """Delete a waveform."""
        pass
    
    @abstractmethod
    def delete_sequence(self, sequence_name):
        """Delete a sequence."""
        pass
    
    @abstractmethod
    def load_waveform(self, waveform_name, to_ch=None):
        """Load a waveform to channels."""
        pass
    
    @abstractmethod
    def load_sequence(self, sequence_name, to_ch=None):
        """Load a sequence to channels."""
        pass
    
    @abstractmethod
    def get_loaded_assets(self):
        """Get the currently loaded assets."""
        pass
    
    @abstractmethod
    def clear_all(self):
        """Clear all stored waveforms and sequences."""
        pass
    
    @abstractmethod
    def get_errors(self):
        """Get hardware errors."""
        pass
    
    @abstractmethod
    def get_sample_rate(self):
        """Get the sample rate."""
        pass
    
    @abstractmethod
    def set_sample_rate(self, sample_rate):
        """Set the sample rate."""
        pass
    
    @abstractmethod
    def get_analog_level(self, amplitude=None, offset=None):
        """Get the analog levels."""
        pass
    
    @abstractmethod
    def set_analog_level(self, amplitude=None, offset=None):
        """Set the analog levels."""
        pass
    
    @abstractmethod
    def get_digital_level(self, low=None, high=None):
        """Get the digital levels."""
        pass
    
    @abstractmethod
    def set_digital_level(self, low=None, high=None):
        """Set the digital levels."""
        pass
    
    @abstractmethod
    def get_active_channels(self, ch=None):
        """Get the active channels."""
        pass
    
    @abstractmethod
    def set_active_channels(self, ch=None):
        """Set the active channels."""
        pass


class BackScanCapability(Flag):
    """
    Availability and configurability of the back scan.
    """
    AVAILABLE = auto()
    FREQUENCY_CONFIGURABLE = auto()
    RESOLUTION_CONFIGURABLE = auto()
    FULLY_CONFIGURABLE = auto()


@dataclass(frozen=True)
class ScannerAxis:
    """
    Data class representing a scanner axis and its constraints.
    """
    name: str
    unit: str = ''
    value_range: Tuple[float, float] = field(default_factory=lambda: (-1.0, 1.0))
    step_range: Tuple[float, float] = field(default_factory=lambda: (0.0, 1.0))
    resolution_range: Tuple[int, int] = field(default_factory=lambda: (1, 1000))


@dataclass(frozen=True)
class ScannerChannel:
    """
    Data class representing a scanner channel and its constraints.
    """
    name: str
    unit: str = ''
    dtype: str = 'float64'


@dataclass(frozen=True)
class ScanConstraints:
    """
    Data class representing scanner constraints.
    """
    axes: Dict[str, ScannerAxis]
    channels: Dict[str, ScannerChannel]
    backscan_capability: BackScanCapability = BackScanCapability.AVAILABLE
    has_position_feedback: bool = False
    maximum_frequency: float = 0.0


@dataclass(frozen=True)
class ScannerSettings:
    """
    Data class representing scanner settings.
    """
    resolution: int = 1
    forward_range: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    backward_range: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    forward_axes: FrozenSet[str] = field(default_factory=frozenset)
    backward_axes: FrozenSet[str] = field(default_factory=frozenset)
    static_axes: FrozenSet[str] = field(default_factory=frozenset)
    backscan_frequency_factor: float = 1.0
    backscan_resolution_factor: float = 1.0
    analog_channels: FrozenSet[str] = field(default_factory=frozenset)
    backscan_analog_channels: FrozenSet[str] = field(default_factory=frozenset)


class ScanningProbeInterface(Base):
    """
    Mock of Qudi ScanningProbeInterface for testing without Qudi.
    """
    
    @abstractmethod
    def get_constraints(self):
        """
        Get hardware constraints/limitations of the scanner.
        
        @return ScanConstraints: Scanner constraints object
        """
        pass
    
    @abstractmethod
    def reset(self):
        """
        Reset the hardware settings to the default state.
        """
        pass
    
    @abstractmethod
    def configure_scan(self, settings):
        """
        Configure the scan settings.
        
        @param ScannerSettings settings: ScannerSettings instance with scan settings
        @return ScannerSettings: The actual configured settings
        """
        pass
    
    @abstractmethod
    def get_scan_settings(self):
        """
        Get the currently configured scan settings.
        
        @return ScannerSettings: The current scan settings
        """
        pass
    
    @abstractmethod
    def start_scan(self):
        """
        Start a scan based on the current settings.
        
        @return int: error code (0:OK, -1:error)
        """
        pass
    
    @abstractmethod
    def stop_scan(self):
        """
        Stop the current scan.
        
        @return int: error code (0:OK, -1:error)
        """
        pass
    
    @abstractmethod
    def get_scanner_axes(self):
        """
        Get the available scanner axes.
        
        @return frozenset(str): Set of axis names
        """
        pass
    
    @abstractmethod
    def get_scanner_channels(self):
        """
        Get the available scanner channels.
        
        @return frozenset(str): Set of channel names
        """
        pass
    
    @abstractmethod
    def get_scanner_position(self):
        """
        Get the current scanner position.
        
        @return dict: Axis name -> position value
        """
        pass
    
    @abstractmethod
    def set_position(self, **kwargs):
        """
        Set the scanner position.
        
        @param kwargs: Axis name -> position value
        @return dict: Actual position after move
        """
        pass


class ControlMode(IntEnum):
    """
    Control mode for laser interface.
    """
    POWER = 0
    CURRENT = 1
    UNKNOWN = 2


class ShutterState(IntEnum):
    """
    Shutter state for laser interface.
    """
    CLOSED = 0
    OPEN = 1
    NO_SHUTTER = 2
    UNKNOWN = 3


class LaserState(IntEnum):
    """
    Laser state for laser interface.
    """
    OFF = 0
    ON = 1
    LOCKED = 2
    UNKNOWN = 3


class SimpleLaserInterface(Base):
    """
    Mock of Qudi SimpleLaserInterface for testing without Qudi.
    """
    
    @abstractmethod
    def get_power_range(self):
        """Get the laser power range."""
        pass
    
    @abstractmethod
    def get_power(self):
        """Get the current laser power."""
        pass
    
    @abstractmethod
    def set_power(self, power):
        """Set the laser power."""
        pass
    
    @abstractmethod
    def get_power_setpoint(self):
        """Get the laser power setpoint."""
        pass
    
    @abstractmethod
    def get_current_unit(self):
        """Get the current unit."""
        pass
    
    @abstractmethod
    def get_current_range(self):
        """Get the laser current range."""
        pass
    
    @abstractmethod
    def get_current(self):
        """Get the current laser current."""
        pass
    
    @abstractmethod
    def get_current_setpoint(self):
        """Get the laser current setpoint."""
        pass
    
    @abstractmethod
    def set_current(self, current):
        """Set the laser current."""
        pass
    
    @abstractmethod
    def get_shutter_state(self):
        """Get the shutter state."""
        pass
    
    @abstractmethod
    def set_shutter_state(self, state):
        """Set the shutter state."""
        pass
    
    @abstractmethod
    def get_laser_state(self):
        """Get the laser state."""
        pass
    
    @abstractmethod
    def set_laser_state(self, state):
        """Set the laser state."""
        pass
    
    @abstractmethod
    def get_temperatures(self):
        """Get laser temperatures."""
        pass
    
    @abstractmethod
    def on(self):
        """Turn on the laser."""
        pass
    
    @abstractmethod
    def off(self):
        """Turn off the laser."""
        pass