# -*- coding: utf-8 -*-

"""
This file contains the Qudi hardware module for the NV simulator scanning probe.

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

import time
from typing import Optional, Dict, Tuple, Any, List
import numpy as np
import os
import sys
from PySide2 import QtCore
from fysom import FysomError
from dataclasses import dataclass

from qudi.util.mutex import RecursiveMutex
from qudi.util.constraints import ScalarConstraint
from qudi.core.configoption import ConfigOption
from qudi.core.connector import Connector
from qudi.interface.scanning_probe_interface import (
    ScanningProbeInterface,
    ScanData,
    ScanConstraints,
    ScannerAxis,
    ScannerChannel,
    ScanSettings,
    BackScanCapability,
    CoordinateTransformMixin
)

# Import QudiFacade directly from current directory to avoid circular imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from qudi_facade import QudiFacade


@dataclass(frozen=True)
class NVSimScanConstraints(ScanConstraints):
    """Extended ScanConstraints for NV simulator"""
    spot_number: ScalarConstraint
    
    def __init__(self, axis_objects, channel_objects, back_scan_capability, has_position_feedback, square_px_only, spot_number):
        super().__init__(
            axis_objects=axis_objects,
            channel_objects=channel_objects,
            back_scan_capability=back_scan_capability,
            has_position_feedback=has_position_feedback,
            square_px_only=square_px_only
        )
        # Add the spot_number constraint as an attribute
        object.__setattr__(self, 'spot_number', spot_number)


class NVSimScanningProbe(CoordinateTransformMixin, ScanningProbeInterface):
    """NV simulator scanning probe for confocal microscopy.
    
    Example config for copy-paste:
    
    nv_sim_scanner:
        module.Class: 'nv_simulator.scanning_probe.NVSimScanningProbe'
        options:
            position_ranges:
                x: [0, 100e-6]
                y: [0, 100e-6]
                z: [-50e-6, 50e-6]
            frequency_ranges:
                x: [1, 1000]
                y: [1, 1000]
                z: [1, 500]
            resolution_ranges:
                x: [1, 1000]
                y: [1, 1000]
                z: [2, 500]
            position_accuracy:
                x: 10e-9
                y: 10e-9
                z: 50e-9
            nv_density: 1e15  # NV density in 1/m^3
    """
    
    _threaded = True
    
    # Configuration options
    _position_ranges = ConfigOption('position_ranges', default={}, missing='error')
    _frequency_ranges = ConfigOption('frequency_ranges', default={}, missing='error')
    _resolution_ranges = ConfigOption('resolution_ranges', default={}, missing='error')
    _position_accuracy = ConfigOption('position_accuracy', default={}, missing='error')
    _max_spot_number = ConfigOption('max_spot_number', default=int(80e3), missing='warn')
    _require_square_pixels = ConfigOption('require_square_pixels', default=False, missing='warn')
    _nv_density = ConfigOption('nv_density', default=1e15, missing='warn')
    _back_scan_available = ConfigOption('back_scan_available', default=True, missing='warn')
    _back_scan_frequency_configurable = ConfigOption('back_scan_frequency_configurable', default=True, missing='warn')
    _back_scan_resolution_configurable = ConfigOption('back_scan_resolution_configurable', default=True, missing='warn')
    
    # Connector declarations
    simulator = Connector(interface='QudiFacade')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Scan process parameters
        self._scan_settings: Optional[ScanSettings] = None
        self._back_scan_settings: Optional[ScanSettings] = None
        self._current_position = dict()
        self._scan_data = None
        self._back_scan_data = None

        # "Hardware" constraints
        self._constraints: Optional[NVSimScanConstraints] = None
        
        # Mutex for access serialization
        self._thread_lock = RecursiveMutex()
        
        # Simulation parameters
        self._simulated_diamond = None
        self._scan_start_time = 0
        self._last_forward_pixel = 0
        self._last_backward_pixel = 0
        self._update_timer = None

    def on_activate(self):
        """Initialisation performed during activation of the module."""
        # Get QudiFacade from connector
        self._qudi_facade = self.simulator()
        
        # Generate static constraints
        axes = list()
        for axis, ax_range in self._position_ranges.items():
            dist = max(ax_range) - min(ax_range)
            freq_range = tuple(self._frequency_ranges.get(axis, (0.1, 1.0)))
            resolution_range = tuple(self._resolution_ranges.get(axis, (2, 10000)))
            pos_accuracy = float(self._position_accuracy.get(axis, 1e-9))
            # Create the required ScalarConstraint objects
            position_constraint = ScalarConstraint(
                default=(min(ax_range) + max(ax_range)) / 2,
                bounds=(min(ax_range), max(ax_range))
            )
            step_constraint = ScalarConstraint(
                default=pos_accuracy * 10,
                bounds=(pos_accuracy, dist)
            )
            resolution_constraint = ScalarConstraint(
                default=(resolution_range[0] + resolution_range[1]) // 2,
                bounds=(resolution_range[0], resolution_range[1]),
                enforce_int=True
            )
            frequency_constraint = ScalarConstraint(
                default=(freq_range[0] + freq_range[1]) / 2,
                bounds=(freq_range[0], freq_range[1])
            )
            
            scanner_axis = ScannerAxis(
                name=axis,
                unit='m',
                position=position_constraint,
                step=step_constraint,
                resolution=resolution_constraint,
                frequency=frequency_constraint
            )
            axes.append(scanner_axis)
        
        # Spot number constraint is the maximum number of spots to scan
        spot_number_constraint = ScalarConstraint(
            default=self._max_spot_number // 2,
            bounds=(1, self._max_spot_number),
            enforce_int=True
        )
        
        # Prepare back scan capability flags
        back_scan_capability = BackScanCapability(0)
        if self._back_scan_available:
            back_scan_capability |= BackScanCapability.AVAILABLE
        if self._back_scan_frequency_configurable:
            back_scan_capability |= BackScanCapability.FREQUENCY_CONFIGURABLE
        if self._back_scan_resolution_configurable:
            back_scan_capability |= BackScanCapability.RESOLUTION_CONFIGURABLE
            
        # Create basic constraints object
        self._constraints = NVSimScanConstraints(
            axis_objects=tuple(axes),
            channel_objects=tuple([ScannerChannel(name='NV Fluorescence', unit='c/s')]),
            back_scan_capability=back_scan_capability,
            has_position_feedback=False,
            square_px_only=self._require_square_pixels,
            spot_number=spot_number_constraint
        )
        
        # Reset position of the scanner
        for axis in self._constraints.axes:
            pos_constraint = self._constraints.axes[axis].position
            self._current_position[axis] = pos_constraint.default
        
        # Initialize simulated diamond with random NV centers
        volume = np.prod([(max(ax_range) - min(ax_range)) for ax_range in self._position_ranges.values()])
        n_nv = int(volume * self._nv_density)  # Number of NV centers in volume
        self._simulated_diamond = self._generate_nv_positions(n_nv)
        
        self.log.info('NV Simulator Scanner initiated')

    def on_deactivate(self):
        """Cleanup performed during deactivation of the module."""
        if self.module_state() != 'idle':
            self.stop_scan()
            
        if self._update_timer is not None and self._update_timer.isActive():
            self._update_timer.stop()
            
        # Reset internal data
        self._scan_data = None
        self._back_scan_data = None
        self._scan_settings = None
        self._back_scan_settings = None
        self._simulated_diamond = None
        
    @property
    def constraints(self):
        """Get hardware constraints/limitations.
        
        @return ScanConstraints: Scanner constraints object
        """
        return self._constraints
        
    def get_constraints(self):
        """Legacy method for backward compatibility.
        
        @return ScanConstraints: Scanner constraints object
        """
        return self.constraints
        
    def reset(self):
        """Reset the hardware settings to the default state."""
        with self._thread_lock:
            # Stop any running scans
            if self.module_state() != 'idle':
                self.stop_scan()
            
            # Reset position of the scanner
            for axis in self._constraints.axes:
                pos_constraint = self._constraints.axes[axis].position
                self._current_position[axis] = pos_constraint.default
                
            # Reset simulated scan data
            self._scan_data = None
            self._back_scan_data = None
            self._scan_settings = None
            self._back_scan_settings = None
            
    def validate_scan_settings(self, settings):
        """Validate the scan settings against the constraints.
        
        @param ScanSettings settings: Settings object to validate
        
        @return bool: True if settings are valid
        """
        # Check if settings axes are valid
        for axis in settings.axes:
            if axis not in self._constraints.axes:
                raise ValueError(f"Invalid axis: {axis}")
                
        # Check if settings channels are valid
        for channel in settings.channels:
            if channel not in [ch.name for ch in self._constraints.channels]:
                raise ValueError(f"Invalid channel: {channel}")
                
        # Check if resolution is within constraints
        for i, axis in enumerate(settings.axes):
            axis_obj = self._constraints.axes[axis]
            if settings.resolution[i] < axis_obj.resolution.bounds[0] or settings.resolution[i] > axis_obj.resolution.bounds[1]:
                raise ValueError(f"Resolution out of range for axis {axis}: {settings.resolution[i]}")
                
        # Check if frequency is within constraints
        for axis, freq in settings.frequency.items():
            axis_obj = self._constraints.axes[axis]
            if freq < axis_obj.frequency.bounds[0] or freq > axis_obj.frequency.bounds[1]:
                raise ValueError(f"Frequency out of range for axis {axis}: {freq}")
                
        # Check if number of spots exceeds maximum
        total_spots = 1
        for res in settings.resolution:
            total_spots *= res
        if total_spots > self._constraints.spot_number.bounds[1]:
            raise ValueError(f"Total number of spots ({total_spots}) exceeds maximum ({self._constraints.spot_number.bounds[1]})")
            
        return True
            
    def calculate_backwards_scan_settings(self, settings):
        """Calculate the back scan settings based on the forward scan settings.
        
        @param ScanSettings settings: Settings object for the forward scan
        
        @return ScanSettings: Settings object for the backward scan
        """
        # For simplicity, back scan uses the same settings as forward scan
        return settings
    
    def configure_scan(self, settings):
        """Configure the scanner with scan settings for the next scan to execute.
        
        @param ScanSettings settings: Settings object holding desired scan settings
        
        @return ScanSettings: Actual applied scan settings
        """
        with self._thread_lock:
            if self.module_state() != 'idle':
                self.log.error('Cannot configure scanner while scan is running. Stop scan first.')
                return self._scan_settings
                
            # Check if scan settings are valid
            self.validate_scan_settings(settings)
            
            # Apply settings
            self._scan_settings = settings
            if hasattr(settings, 'backward_scan_enabled') and settings.backward_scan_enabled:
                self._back_scan_settings = self.calculate_backwards_scan_settings(settings)
            else:
                self._back_scan_settings = None
                
            # Reset scan data
            self._scan_data = None
            self._back_scan_data = None
            
            return self._scan_settings
            
    def start_scan(self):
        """Start the configured scan.
        
        @return bool: Scan started successfully
        """
        with self._thread_lock:
            if self.module_state() != 'idle':
                self.log.error('Cannot start scan. Scanner is already scanning.')
                return False
                
            if self._scan_settings is None:
                self.log.error('Cannot start scan. No scan settings configured.')
                return False
                
            # Initialize scan data
            self._scan_data = self._init_scan_data(self._scan_settings)
            if self._back_scan_settings is not None:
                self._back_scan_data = self._init_scan_data(self._back_scan_settings)
                
            # Reset pixel counters
            self._last_forward_pixel = 0
            self._last_backward_pixel = 0
            
            # Record start time
            self._scan_start_time = time.time()
            
            # Set module state to running
            self.module_state.lock()
            
            # Set up timer for simulating scanning
            if self._update_timer is None:
                self._update_timer = QtCore.QTimer()
                self._update_timer.setSingleShot(False)
                self._update_timer.timeout.connect(self._simulate_scan_progress, QtCore.Qt.QueuedConnection)
                
            # Start timer for simulating scan progress
            scan_time = self._calculate_scan_time(self._scan_settings)
            update_interval = scan_time / (self._scan_settings.resolution[0] * self._scan_settings.resolution[1]) * 1000  # in ms
            update_interval = max(10, min(500, update_interval))  # Keep between 10-500ms for responsiveness
            self._update_timer.start(int(update_interval))
            
            self.log.info('Scan started')
            return True
            
    def stop_scan(self):
        """Stop a running scan.
        
        @return bool: Scan stopped successfully
        """
        with self._thread_lock:
            if self.module_state() == 'idle':
                self.log.warning('No scan is running. Cannot stop.')
                return False
                
            # Stop scan timer
            if self._update_timer is not None and self._update_timer.isActive():
                self._update_timer.stop()
                
            # Unlock module state
            self.module_state.unlock()
            
            self.log.info('Scan stopped')
            return True
            
    def get_scan_data(self):
        """Get the scan data for the currently running or last finished scan.
        
        @return ScanData: Scan data object containing data and scan settings
        """
        with self._thread_lock:
            if self._scan_data is None:
                self.log.warning('No scan data available')
                return None
                
            return self._scan_data

    def get_position(self):
        """Get the current position of the scanning probe.
        
        @return Dict[str, float]: Current position of the scan probe in meters for each axis
        """
        with self._thread_lock:
            return self._current_position.copy()
            
    def set_position(self, position):
        """Set the current position of the scanning probe.
        
        @param Dict[str, float] position: Position to set in meters for each axis
        
        @return Dict[str, float]: Actual position set for each axis
        """
        with self._thread_lock:
            if self.module_state() != 'idle':
                self.log.error('Cannot set position while scan is running')
                return self._current_position.copy()
                
            # Validate position against constraints
            for axis, pos in position.items():
                if axis not in self._current_position:
                    self.log.error(f'Invalid axis: {axis}')
                    continue
                    
                # Check if position is within range
                pos_constraint = self._constraints.axes[axis].position
                if not pos_constraint.bounds[0] <= pos <= pos_constraint.bounds[1]:
                    self.log.warning(f'Position out of range for axis {axis}: {pos}')
                    # Clip to valid range
                    pos = max(pos_constraint.bounds[0], min(pos_constraint.bounds[1], pos))
                    
                # Apply position
                self._current_position[axis] = pos
                
            # Update the simulated fluorescence signal based on the current position
            self._update_simulated_signal()
                
            return self._current_position.copy()
            
    def move_absolute(self, position, velocity=None, blocking=False):
        """Move the scanning probe to an absolute position.
        
        @param Dict[str, float] position: Position to move to for each axis
        @param float velocity: Optional velocity to use
        @param bool blocking: Whether to wait for the movement to complete
        
        @return Dict[str, float]: Actual position reached
        """
        with self._thread_lock:
            if self.module_state() != 'idle':
                self.log.error('Cannot move while scan is running')
                return self._current_position.copy()
                
            # This is equivalent to set_position for the simulator
            return self.set_position(position)
    
    def move_relative(self, distance, velocity=None, blocking=False):
        """Move the scanning probe by a relative distance.
        
        @param Dict[str, float] distance: Distance to move for each axis
        @param float velocity: Optional velocity to use
        @param bool blocking: Whether to wait for the movement to complete
        
        @return Dict[str, float]: Actual position reached
        """
        with self._thread_lock:
            if self.module_state() != 'idle':
                self.log.error('Cannot move while scan is running')
                return self._current_position.copy()
                
            # Calculate new position
            new_position = self._current_position.copy()
            for axis, dist in distance.items():
                if axis in new_position:
                    new_position[axis] += dist
                    
            # Set new position
            return self.set_position(new_position)
            
    def get_target(self):
        """Get the current target position of the scanning probe.
        
        @return Dict[str, float]: Current target position for each axis
        """
        # For the simulator, target position is the same as current position
        return self._current_position.copy()
            
    def get_target_position(self):
        """Get the target position of the scanning probe defined by the current scanner settings.
        
        @return Dict[str, float]: Target position for each axis
        """
        with self._thread_lock:
            # For NV simulator, the target is the current position
            return self._current_position.copy()
            
    def signal_scan_next_line(self):
        """Signal that the position should be moved to the start of the next line in the scan.
        This is a no-op for the NV simulator.
        
        @return bool: Next line preparation successful
        """
        return True
        
    def signal_image_updated(self, data=None):
        """Signal that the image is updated to potentially call slave logic module methods.
        This is a no-op for the NV simulator.
        
        @param data: Generic scan data object
        
        @return bool: Image update signal sending successful
        """
        return True
        
    # Helper methods
    def _init_scan_data(self, settings):
        """Initialize a scan data object with empty data arrays.
        
        @param ScanSettings settings: Scan settings to use
        
        @return ScanData: Initialized scan data object
        """
        data_arrays = dict()
        for channel in self._constraints.channels:
            data_arrays[channel.name] = np.zeros(tuple(settings.resolution), dtype=np.float64)
            
        return ScanData(
            settings=settings,
            data=data_arrays,
            scanner_position=self._current_position.copy(),
            forward=True,
            scan_time=self._calculate_scan_time(settings)
        )
        
    def _calculate_scan_time(self, settings):
        """Calculate the expected scan time based on settings.
        
        @param ScanSettings settings: Scan settings to use
        
        @return float: Expected scan time in seconds
        """
        # Calculate scan time based on the slowest axis frequency
        slowest_freq = min(settings.frequency.values())
        pixels_per_line = settings.resolution[0]
        num_lines = settings.resolution[1]
        
        # Calculate time per line and total scan time
        time_per_line = 1.0 / slowest_freq
        total_time = time_per_line * num_lines
        
        return total_time
        
    def _generate_nv_positions(self, n_nv):
        """Generate random NV positions within the scan volume.
        
        @param int n_nv: Number of NV centers to generate
        
        @return np.ndarray: Array of NV center positions with shape (n_nv, 3)
        """
        # Get scan volume dimensions
        x_range = self._position_ranges.get('x', [0, 100e-6])
        y_range = self._position_ranges.get('y', [0, 100e-6])
        z_range = self._position_ranges.get('z', [-50e-6, 50e-6])
        
        # Generate random positions
        x_pos = np.random.uniform(min(x_range), max(x_range), n_nv)
        y_pos = np.random.uniform(min(y_range), max(y_range), n_nv)
        z_pos = np.random.uniform(min(z_range), max(z_range), n_nv)
        
        # Return as array
        return np.column_stack((x_pos, y_pos, z_pos))
        
    def _update_simulated_signal(self):
        """Update the simulated fluorescence signal based on the current position.
        
        @return float: Fluorescence signal value
        """
        # Simulated PSF parameters
        psf_width_xy = 0.3e-6  # 300 nm lateral resolution
        psf_width_z = 0.7e-6   # 700 nm axial resolution
        
        # Calculate distance to each NV center
        if self._simulated_diamond is None or len(self._simulated_diamond) == 0:
            return 0.0
            
        # Current position as array
        pos = np.array([
            self._current_position.get('x', 0),
            self._current_position.get('y', 0),
            self._current_position.get('z', 0)
        ])
        
        # Calculate distances
        distances_sq = np.sum((self._simulated_diamond - pos)**2, axis=1)
        
        # Calculate PSF weights
        xy_dist_sq = np.sum((self._simulated_diamond[:, :2] - pos[:2])**2, axis=1)
        z_dist_sq = (self._simulated_diamond[:, 2] - pos[2])**2
        
        # Apply PSF
        psf_weights = np.exp(-2 * xy_dist_sq / psf_width_xy**2) * np.exp(-2 * z_dist_sq / psf_width_z**2)
        
        # Sum contributions (each NV contributes proportionally to PSF weight)
        signal = np.sum(psf_weights)
        
        # Scale and add noise
        base_level = 1000  # counts per second background
        max_signal = 100000  # max counts per second
        
        signal = base_level + signal * max_signal
        
        # Add random noise (Poisson)
        noise = np.random.poisson(signal * 0.01)  # 1% noise
        signal += noise
        
        # Update the QudiFacade collection efficiency (0-1 scale)
        if self._qudi_facade is not None:
            # Scale to 0-1 range for collection efficiency
            self._qudi_facade.set_collection_efficiency(signal / (max_signal + base_level))
        
        return signal
        
    def _simulate_scan_progress(self):
        """Simulate scan progress by updating pixels.
        
        This method is called by the timer to simulate scanning.
        """
        with self._thread_lock:
            if self.module_state() == 'idle' or self._scan_settings is None:
                if self._update_timer is not None and self._update_timer.isActive():
                    self._update_timer.stop()
                return
                
            # Calculate how many pixels to update this time
            pixels_per_call = max(1, self._scan_settings.resolution[0] // 100)
            total_pixels = self._scan_settings.resolution[0] * self._scan_settings.resolution[1]
            
            # Update forward scan
            for _ in range(pixels_per_call):
                if self._last_forward_pixel >= total_pixels:
                    break
                    
                # Calculate current x, y position in the scan grid
                y_idx = self._last_forward_pixel // self._scan_settings.resolution[0]
                x_idx = self._last_forward_pixel % self._scan_settings.resolution[0]
                
                # Calculate the actual position in space
                x_range = (self._scan_settings.range[0]['x'], self._scan_settings.range[1]['x'])
                y_range = (self._scan_settings.range[0]['y'], self._scan_settings.range[1]['y'])
                
                x_pos = x_range[0] + (x_range[1] - x_range[0]) * (x_idx / (self._scan_settings.resolution[0] - 1))
                y_pos = y_range[0] + (y_range[1] - y_range[0]) * (y_idx / (self._scan_settings.resolution[1] - 1))
                
                # Set the current position
                self._current_position['x'] = x_pos
                self._current_position['y'] = y_pos
                
                # Calculate the simulated signal
                signal = self._update_simulated_signal()
                
                # Store the signal in the scan data
                self._scan_data.data['NV Fluorescence'][y_idx, x_idx] = signal
                
                # Increment pixel counter
                self._last_forward_pixel += 1
                
            # Update backward scan if enabled
            if self._back_scan_settings is not None and self._back_scan_data is not None:
                for _ in range(pixels_per_call):
                    if self._last_backward_pixel >= total_pixels:
                        break
                        
                    # For backward scan, x index runs backwards
                    y_idx = self._last_backward_pixel // self._back_scan_settings.resolution[0]
                    x_idx = self._back_scan_settings.resolution[0] - 1 - (self._last_backward_pixel % self._back_scan_settings.resolution[0])
                    
                    # The position is the same as for the forward scan
                    x_range = (self._back_scan_settings.range[0]['x'], self._back_scan_settings.range[1]['x'])
                    y_range = (self._back_scan_settings.range[0]['y'], self._back_scan_settings.range[1]['y'])
                    
                    x_pos = x_range[0] + (x_range[1] - x_range[0]) * (x_idx / (self._back_scan_settings.resolution[0] - 1))
                    y_pos = y_range[0] + (y_range[1] - y_range[0]) * (y_idx / (self._back_scan_settings.resolution[1] - 1))
                    
                    # Set the current position (already set by forward scan, so no need to update)
                    
                    # Calculate the simulated signal (add some different noise)
                    signal = self._scan_data.data['NV Fluorescence'][y_idx, x_idx]
                    signal = signal + np.random.normal(0, signal * 0.05)  # 5% noise for backward scan
                    
                    # Store the signal in the back scan data
                    self._back_scan_data.data['NV Fluorescence'][y_idx, x_idx] = signal
                    
                    # Increment pixel counter
                    self._last_backward_pixel += 1
                    
            # Check if scan is complete
            if self._last_forward_pixel >= total_pixels:
                if self._back_scan_settings is None or self._last_backward_pixel >= total_pixels:
                    # Stop the timer and scan
                    if self._update_timer is not None and self._update_timer.isActive():
                        self._update_timer.stop()
                        
                    # Unlock module state
                    self.module_state.unlock()
                    
                    self.log.info('Scan completed')
                    
                    # Emit scan finished signal
                    self.scan_finished.emit(self._scan_data)
                    
    def get_back_scan_data(self):
        """Get the backward scan data for the currently running or last finished scan.
        
        @return ScanData: Backward scan data object containing data and scan settings
        """
        with self._thread_lock:
            return self._back_scan_data
            
    @property
    def scan_settings(self):
        """Property returning all parameters needed for a 1D or 2D scan. Returns None if not configured.
        
        @return ScanSettings: Current scan settings or None
        """
        with self._thread_lock:
            return self._scan_settings
            
    @property
    def back_scan_settings(self):
        """Property returning all parameters of the backwards scan. Returns None if not configured or not available.
        
        @return ScanSettings: Current back scan settings or None
        """
        with self._thread_lock:
            return self._back_scan_settings
            
    def configure_back_scan(self, settings):
        """Configure the hardware with all parameters of the backwards scan.
        Raise an exception if the settings are invalid and do not comply with the hardware constraints.
        
        @param ScanSettings settings: ScanSettings instance holding all parameters for the back scan
        
        @return ScanSettings: Actual applied back scan settings
        """
        with self._thread_lock:
            if self.module_state() != 'idle':
                self.log.error('Cannot configure scanner while scan is running. Stop scan first.')
                return self._back_scan_settings
                
            # Check if back scan is available
            if not self._back_scan_available:
                self.log.error('Back scan is not available on this hardware.')
                return None
                
            # Check if scan settings are valid
            if self._scan_settings is None:
                self.log.error('Cannot configure back scan before configuring the forward scan.')
                return None
                
            # Validate settings against constraints
            self.validate_scan_settings(settings)
            
            # Check back scan settings against forward scan
            if settings.axes != self._scan_settings.axes:
                self.log.error('Back scan must use the same axes as the forward scan.')
                return self._back_scan_settings
                
            if settings.range != self._scan_settings.range:
                self.log.error('Back scan must use the same range as the forward scan.')
                return self._back_scan_settings
                
            # Apply settings
            self._back_scan_settings = settings
            
            # Reset scan data
            self._back_scan_data = None
            
            return self._back_scan_settings
            
    def emergency_stop(self):
        """Immediately stop all hardware movement.
        
        @return bool: Emergency stop successful
        """
        with self._thread_lock:
            # For the simulator, this is the same as stop_scan
            if self.module_state() == 'idle':
                self.log.warning('No scan is running. Emergency stop not needed.')
                return True
                
            # Stop scan timer
            if self._update_timer is not None and self._update_timer.isActive():
                self._update_timer.stop()
                
            # Unlock module state
            self.module_state.unlock()
            
            self.log.info('Emergency stop executed')
            return True