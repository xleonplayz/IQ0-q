# -*- coding: utf-8 -*-

"""
Qudi scanning probe interface for the confocal simulator.

Copyright (c) 2023
"""

import numpy as np
import time
from typing import Dict, Tuple, List, Any, Optional, Union

try:
    # Import Qudi interface classes if available
    from qudi.interface.scanning_probe_interface import ScanningProbeInterface
    from qudi.core.module import RemoteContextMutex as RecursiveMutex
    from qudi.core.configoption import ConfigOption
    from qudi.core.connector import Connector
    from qudi.util.helpers import in_range
    from qudi.util.constraints import ScalarConstraint
    from qudi.core.statusvariable import StatusVar
    from qudi.interface.scanning_probe_interface import ScanAxis, ScanConstraints, \
        ScannerChannel, ScanData, ScannerAxis, BackScanCapability
except ImportError:
    # Mock classes for standalone testing
    class ScanningProbeInterface:
        pass
    
    class RecursiveMutex:
        def __enter__(self):
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    
    class ConfigOption:
        def __init__(self, default):
            self.default = default
    
    class Connector:
        def __init__(self, interface_type):
            self.interface_type = interface_type
    
    class StatusVar:
        def __init__(self, default):
            self.default = default
    
    # Mock ScannerAxis, etc.
    class ScanAxis:
        def __init__(self, name, unit):
            self.name = name
            self.unit = unit
    
    class ScannerAxis:
        def __init__(self, name, unit, position, step, resolution, frequency):
            self.name = name
            self.unit = unit
            self.position = position
            self.step = step
            self.resolution = resolution
            self.frequency = frequency
    
    class ScannerChannel:
        def __init__(self, name, unit, dtype):
            self.name = name
            self.unit = unit
            self.dtype = dtype
    
    class BackScanCapability:
        AVAILABLE = "Available"
        UNAVAILABLE = "Unavailable"
    
    class ScalarConstraint:
        def __init__(self, default, bounds, enforce_int=False):
            self.default = default
            self.bounds = bounds
            self.enforce_int = enforce_int
        
        def check(self, value):
            return value >= self.bounds[0] and value <= self.bounds[1]
    
    class ScanConstraints:
        def __init__(self, axis_objects, channel_objects, back_scan_capability, 
                   has_position_feedback, square_px_only):
            self.axes = {ax.name: ax for ax in axis_objects}
            self.channels = {ch.name: ch for ch in channel_objects}
            self.back_scan_capability = back_scan_capability
            self.has_position_feedback = has_position_feedback
            self.square_px_only = square_px_only
        
        def check_settings(self, settings):
            pass
        
        def check_back_scan_settings(self, backward_settings, forward_settings):
            pass
    
    class ScanData:
        @classmethod
        def from_constraints(cls, settings, constraints, scanner_target_at_start):
            instance = cls()
            instance.settings = settings
            instance.data = {ch: np.zeros((10, 10)) for ch in constraints.channels}
            return instance
        
        def new_scan(self):
            pass
        
        def copy(self):
            return self

# Import from confocal simulator
from .confocal_simulator import ConfocalSimulator


class ConfocalSimulatorScanner(ScanningProbeInterface):
    """
    A Qudi scanning probe interface implementation for the confocal simulator.
    This class provides a standard interface for Qudi to control the confocal simulator.
    """
    
    # Config options
    _sample_dimensions = ConfigOption('dimensions', default=(20e-6, 20e-6, 5e-6))
    _nv_density = ConfigOption('nv_density', default=1e14)
    _random_seed = ConfigOption('random_seed', default=None)
    
    # Status variables
    _position = StatusVar('position', default={})
    _scan_settings = StatusVar('scan_settings', default=None)
    _back_scan_settings = StatusVar('back_scan_settings', default=None)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread_lock = RecursiveMutex()
        self._scan_data = None
        self._back_scan_data = None
        self._confocal_simulator = None
        self._constraints = None
    
    def on_activate(self):
        """Initialize the confocal simulator."""
        # Create confocal simulator
        self._confocal_simulator = ConfocalSimulator(
            dimensions=self._sample_dimensions,
            nv_density=self._nv_density,
            random_seed=self._random_seed
        )
        
        # Set up constraints for scanner axes
        axes = []
        x_range = (0, self._confocal_simulator.diamond.dimensions[0])
        y_range = (0, self._confocal_simulator.diamond.dimensions[1])
        z_range = (0, self._confocal_simulator.diamond.dimensions[2])
        
        # Position, step, resolution, and frequency constraints for each axis
        axes.append(
            ScannerAxis(
                name='x', 
                unit='m', 
                position=ScalarConstraint(default=x_range[0], bounds=x_range),
                step=ScalarConstraint(default=0.1e-6, bounds=(0.01e-6, 1e-6)),
                resolution=ScalarConstraint(default=100, bounds=(2, 1000), enforce_int=True),
                frequency=ScalarConstraint(default=10, bounds=(0.1, 100))
            )
        )
        axes.append(
            ScannerAxis(
                name='y', 
                unit='m', 
                position=ScalarConstraint(default=y_range[0], bounds=y_range),
                step=ScalarConstraint(default=0.1e-6, bounds=(0.01e-6, 1e-6)),
                resolution=ScalarConstraint(default=100, bounds=(2, 1000), enforce_int=True),
                frequency=ScalarConstraint(default=10, bounds=(0.1, 100))
            )
        )
        axes.append(
            ScannerAxis(
                name='z', 
                unit='m', 
                position=ScalarConstraint(default=z_range[0], bounds=z_range),
                step=ScalarConstraint(default=0.1e-6, bounds=(0.01e-6, 1e-6)),
                resolution=ScalarConstraint(default=100, bounds=(2, 1000), enforce_int=True),
                frequency=ScalarConstraint(default=1, bounds=(0.1, 10))
            )
        )
        
        # Define scanner channels
        channels = [
            ScannerChannel(name='fluorescence', unit='c/s', dtype='float64')
        ]
        
        # Create constraints
        self._constraints = ScanConstraints(
            axis_objects=tuple(axes),
            channel_objects=tuple(channels),
            back_scan_capability=BackScanCapability.AVAILABLE,
            has_position_feedback=False,
            square_px_only=False
        )
        
        # Initialize position
        if not self._position:
            self._position = {
                'x': self._confocal_simulator.diamond.dimensions[0]/2,
                'y': self._confocal_simulator.diamond.dimensions[1]/2,
                'z': 0
            }
    
    def on_deactivate(self):
        """Deinitialize hardware."""
        self.reset()
    
    def reset(self):
        """Hard reset of the hardware."""
        with self._thread_lock:
            if self.module_state() == 'locked':
                self.stop_scan()
                self.module_state.unlock()
    
    @property
    def constraints(self):
        """Return scanner constraints."""
        return self._constraints
    
    @property
    def scan_settings(self):
        """Return current scan settings."""
        return self._scan_settings
    
    @property
    def back_scan_settings(self):
        """Return back scan settings."""
        return self._back_scan_settings
    
    def configure_scan(self, settings):
        """Configure the scan."""
        with self._thread_lock:
            if self.module_state() != 'idle':
                raise RuntimeError('Cannot configure scan while scanning is in progress.')
            
            # Check settings against constraints
            if hasattr(self.constraints, 'check_settings'):
                self.constraints.check_settings(settings)
            
            # Store settings
            self._scan_settings = settings
            self._back_scan_settings = None
    
    def configure_back_scan(self, settings):
        """Configure the back scan."""
        with self._thread_lock:
            if self.module_state() != 'idle':
                raise RuntimeError('Cannot configure back scan while scanning is in progress.')
            if self._scan_settings is None:
                raise RuntimeError('Configure forward scan first.')
            
            # Check settings
            if hasattr(self.constraints, 'check_back_scan_settings'):
                self.constraints.check_back_scan_settings(
                    backward_settings=settings, 
                    forward_settings=self._scan_settings
                )
            
            # Store settings
            self._back_scan_settings = settings
    
    def move_absolute(self, position, velocity=None, blocking=False):
        """Move to absolute position."""
        with self._thread_lock:
            if self.module_state() != 'idle':
                raise RuntimeError('Cannot move while scanning is in progress.')
            
            # Check position is valid
            for axis, pos in position.items():
                if axis in self.constraints.axes:
                    self.constraints.axes[axis].position.check(pos)
            
            # Update position
            self._position.update(position)
            
            return self._position.copy()
    
    def move_relative(self, distance, velocity=None, blocking=False):
        """Move by relative distance."""
        with self._thread_lock:
            if self.module_state() != 'idle':
                raise RuntimeError('Cannot move while scanning is in progress.')
            
            # Calculate new position
            new_pos = {}
            for axis, dist in distance.items():
                if axis in self._position:
                    new_pos[axis] = self._position[axis] + dist
                    
                    # Check position is valid
                    if axis in self.constraints.axes:
                        self.constraints.axes[axis].position.check(new_pos[axis])
            
            # Update position
            self._position.update(new_pos)
            
            return self._position.copy()
    
    def get_target(self):
        """Get current target position."""
        return self._position.copy()
    
    def get_position(self):
        """Get current actual position."""
        return self._position.copy()
    
    def start_scan(self):
        """Start a configured scan."""
        with self._thread_lock:
            if self.module_state() != 'idle':
                raise RuntimeError('Cannot start scan while scanning is in progress.')
            if self._scan_settings is None:
                raise RuntimeError('No scan configured.')
            
            # Lock module state
            self.module_state.lock()
            
            try:
                # Initialize scan data
                if hasattr(ScanData, 'from_constraints'):
                    self._scan_data = ScanData.from_constraints(
                        settings=self._scan_settings,
                        constraints=self.constraints,
                        scanner_target_at_start=self.get_target()
                    )
                    self._scan_data.new_scan()
                    
                    # Initialize back scan data if configured
                    if self._back_scan_settings is not None:
                        self._back_scan_data = ScanData.from_constraints(
                            settings=self._back_scan_settings,
                            constraints=self.constraints,
                            scanner_target_at_start=self.get_target()
                        )
                        self._back_scan_data.new_scan()
                else:
                    # For testing without full Qudi
                    self._scan_data = {}
                    self._back_scan_data = {} if self._back_scan_settings is not None else None
                
                # Calculate scan parameters from settings
                axes = self._scan_settings.axes
                ranges = self._scan_settings.range
                resolutions = self._scan_settings.resolution
                
                # Perform the scan
                if len(axes) == 1:
                    # 1D scan
                    axis = axes[0]
                    start = ranges[0][0]
                    end = ranges[0][1]
                    steps = resolutions[0]
                    
                    # Other position components
                    position = self._position.copy()
                    
                    # Create start and end point
                    start_pos = position.copy()
                    start_pos[axis] = start
                    
                    end_pos = position.copy()
                    end_pos[axis] = end
                    
                    # Convert to tuples for simulator
                    start_tuple = (start_pos['x'], start_pos['y'], start_pos['z'])
                    end_tuple = (end_pos['x'], end_pos['y'], end_pos['z'])
                    
                    # Perform 1D scan
                    counts = self._confocal_simulator.scan_line(start_tuple, end_tuple, steps)
                    
                    # Update scan data
                    if hasattr(self._scan_data, 'data'):
                        self._scan_data.data['fluorescence'] = counts.reshape(-1, 1)
                    else:
                        self._scan_data = {'fluorescence': counts}
                    
                elif len(axes) == 2:
                    # 2D scan
                    x_axis, y_axis = axes
                    x_start, x_end = ranges[0]
                    y_start, y_end = ranges[1]
                    x_steps, y_steps = resolutions
                    
                    # Other position components
                    position = self._position.copy()
                    
                    # Z position for all scan
                    z = position.get('z', 0)
                    
                    # Calculate center and size for simulator
                    center = (
                        (x_start + x_end) / 2,
                        (y_start + y_end) / 2,
                        z
                    )
                    
                    size = (
                        abs(x_end - x_start),
                        abs(y_end - y_start)
                    )
                    
                    # Perform 2D scan
                    image = self._confocal_simulator.scan_plane(center, size, (x_steps, y_steps))
                    
                    # Update scan data
                    if hasattr(self._scan_data, 'data'):
                        self._scan_data.data['fluorescence'] = image
                    else:
                        self._scan_data = {'fluorescence': image}
                
                elif len(axes) == 3:
                    # 3D scan
                    x_axis, y_axis, z_axis = axes
                    x_start, x_end = ranges[0]
                    y_start, y_end = ranges[1]
                    z_start, z_end = ranges[2]
                    x_steps, y_steps, z_steps = resolutions
                    
                    # Calculate center and size for simulator
                    center = (
                        (x_start + x_end) / 2,
                        (y_start + y_end) / 2,
                        (z_start + z_end) / 2
                    )
                    
                    size = (
                        abs(x_end - x_start),
                        abs(y_end - y_start),
                        abs(z_end - z_start)
                    )
                    
                    # Perform 3D scan
                    volume = self._confocal_simulator.scan_volume(
                        center, size, (x_steps, y_steps, z_steps))
                    
                    # Update scan data
                    if hasattr(self._scan_data, 'data'):
                        self._scan_data.data['fluorescence'] = volume
                    else:
                        self._scan_data = {'fluorescence': volume}
                
                # Back scan if configured - similar implementation would go here
                
                # Update position to end of scan
                if len(axes) >= 1:
                    self._position[axes[0]] = ranges[0][1]
                if len(axes) >= 2:
                    self._position[axes[1]] = ranges[1][1]
                if len(axes) >= 3:
                    self._position[axes[2]] = ranges[2][1]
                
            finally:
                # Unlock module state
                self.module_state.unlock()
    
    def stop_scan(self):
        """Stop the current scan."""
        with self._thread_lock:
            if self.module_state() != 'locked':
                raise RuntimeError('No scan in progress.')
            
            self.module_state.unlock()
    
    def get_scan_data(self):
        """Return the scan data."""
        with self._thread_lock:
            if self._scan_data is None:
                return None
                
            if hasattr(self._scan_data, 'copy'):
                return self._scan_data.copy()
            else:
                return self._scan_data
    
    def get_back_scan_data(self):
        """Return the back scan data."""
        with self._thread_lock:
            if self._back_scan_data is None:
                return None
                
            if hasattr(self._back_scan_data, 'copy'):
                return self._back_scan_data.copy()
            else:
                return self._back_scan_data
    
    def emergency_stop(self):
        """Emergency stop the scanner."""
        with self._thread_lock:
            if self.module_state() == 'locked':
                self.module_state.unlock()