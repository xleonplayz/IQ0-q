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


class NVSimScanningProbe(CoordinateTransformMixin, ScanningProbeInterface):
    """NV simulator scanning probe for confocal microscopy.
    
    Example config for copy-paste:

    nv_sim_scanner:
        module.Class: 'nv_simulator.scanning_probe.NVSimScanningProbe'
        options:
            position_ranges:
                x: [0, 200e-6]
                y: [0, 200e-6]
                z: [-100e-6, 100e-6]
            frequency_ranges:
                x: [1, 5000]
                y: [1, 5000]
                z: [1, 1000]
            resolution_ranges:
                x: [1, 10000]
                y: [1, 10000]
                z: [2, 1000]
            position_accuracy:
                x: 10e-9
                y: 10e-9
                z: 50e-9
            # max_spot_number: 80e3 # optional
            # require_square_pixels: False # optional
            # nv_density: 1e15 # optional, NV density in 1/m^3
            # back_scan_available: True # optional
            # back_scan_frequency_configurable: True # optional
            # back_scan_resolution_configurable: True # optional
    """

    _threaded = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Get configuration parameters
        config = self.get_connector_config()
        
        # Required configuration options
        self._position_ranges = config.get('position_ranges')
        self._frequency_ranges = config.get('frequency_ranges')
        self._resolution_ranges = config.get('resolution_ranges')
        self._position_accuracy = config.get('position_accuracy')
        
        # Optional configuration options
        self._max_spot_number = config.get('max_spot_number', int(80e3))
        self._require_square_pixels = config.get('require_square_pixels', False)
        self._nv_density = config.get('nv_density', 1e15)  # NV density in 1/m^3
        self._back_scan_available = config.get('back_scan_available', True)
        self._back_scan_frequency_configurable = config.get('back_scan_frequency_configurable', True)
        self._back_scan_resolution_configurable = config.get('back_scan_resolution_configurable', True)

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
        # Initialize the simulator
        self._qudi_facade = QudiFacade()
        
        # Generate static constraints
        axes = list()
        for axis, ax_range in self._position_ranges.items():
            dist = max(ax_range) - min(ax_range)
            resolution_range = tuple(self._resolution_ranges[axis])
            res_default = min(resolution_range[1], 100)
            frequency_range = tuple(self._frequency_ranges[axis])
            f_default = min(frequency_range[1], 1e3)

            position = ScalarConstraint(default=min(ax_range), bounds=tuple(ax_range))
            resolution = ScalarConstraint(default=res_default, bounds=resolution_range, enforce_int=True)
            frequency = ScalarConstraint(default=f_default, bounds=frequency_range)
            step = ScalarConstraint(default=0, bounds=(0, dist))
            spot_number = ScalarConstraint(default=self._max_spot_number, bounds=(0, self._max_spot_number))
            axes.append(
                ScannerAxis(
                    name=axis, unit='m', position=position, step=step, resolution=resolution, frequency=frequency
                )
            )
        channels = [
            ScannerChannel(name='fluorescence', unit='c/s', dtype='float64'),
            ScannerChannel(name='APD events', unit='count', dtype='float64'),
        ]

        if not self._back_scan_available:
            back_scan_capability = BackScanCapability(0)
        else:
            back_scan_capability = BackScanCapability.AVAILABLE
            if self._back_scan_resolution_configurable:
                back_scan_capability = back_scan_capability | BackScanCapability.RESOLUTION_CONFIGURABLE
            if self._back_scan_frequency_configurable:
                back_scan_capability = back_scan_capability | BackScanCapability.FREQUENCY_CONFIGURABLE
                
        self._constraints = NVSimScanConstraints(
            axis_objects=tuple(axes),
            channel_objects=tuple(channels),
            back_scan_capability=back_scan_capability,
            has_position_feedback=False,
            square_px_only=False,
            spot_number=spot_number,
        )

        # Set default process values
        self._current_position = {ax.name: np.mean(ax.position.bounds) for ax in self.constraints.axes.values()}
        self._scan_data = None
        self._back_scan_data = None

        # Initialize the scan timer
        self._scan_start_time = 0
        self._last_forward_pixel = 0
        self._last_backward_pixel = 0
        self._update_timer = QtCore.QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._handle_timer, QtCore.Qt.QueuedConnection)
        
        # Set the initial position of the confocal simulator
        self._qudi_facade.confocal_simulator.set_position(
            self._current_position.get('x', 0),
            self._current_position.get('y', 0),
            self._current_position.get('z', 0)
        )
        
        self.log.info('NV Simulator Scanning Probe initialized')

    def on_deactivate(self):
        """Deactivate the scanning probe module."""
        self.reset()
        try:
            self._update_timer.stop()
        except:
            pass
        self._update_timer.timeout.disconnect()

    @property
    def scan_settings(self) -> Optional[ScanSettings]:
        """Property returning all parameters needed for a 1D or 2D scan. Returns None if not configured."""
        with self._thread_lock:
            return self._scan_settings

    @property
    def back_scan_settings(self) -> Optional[ScanSettings]:
        """Property returning all parameters needed for the backward scan. Returns None if not configured."""
        with self._thread_lock:
            return self._back_scan_settings

    def reset(self):
        """Hard reset of the hardware."""
        with self._thread_lock:
            if self.module_state() == 'locked':
                self.module_state.unlock()
            self.log.debug('NV simulator scanning probe has been reset.')

    @property
    def constraints(self) -> ScanConstraints:
        """Read-only property returning the constraints of this scanning probe hardware."""
        return self._constraints

    def configure_scan(self, settings: ScanSettings) -> None:
        """
        Configure the hardware with all parameters required for a 1D or 2D scan.

        Raises an exception if the provided settings are invalid or do not comply with hardware constraints.

        Parameters
        ----------
        settings : ScanSettings
            An instance of `ScanSettings` containing all necessary scan parameters.

        Raises
        ------
        ValueError
            If the settings are invalid or incompatible with hardware constraints.
        """
        with self._thread_lock:
            self.log.debug('NV simulator scanning probe "configure_scan" called.')
            # Sanity checking
            if self.module_state() != 'idle':
                raise RuntimeError(
                    'Unable to configure scan parameters while scan is running. ' 'Stop scanning and try again.'
                )

            # check settings - will raise appropriate exceptions if something is not right
            self.constraints.check_settings(settings)

            self._scan_settings = settings
            # reset back scan configuration
            self._back_scan_settings = None

    def configure_back_scan(self, settings: ScanSettings) -> None:
        """Configure the hardware with all parameters of the backwards scan.
        Raise an exception if the settings are invalid and do not comply with the hardware constraints.

        @param ScanSettings settings: ScanSettings instance holding all parameters for the back scan
        """
        with self._thread_lock:
            if self.module_state() != 'idle':
                raise RuntimeError(
                    'Unable to configure scan parameters while scan is running. ' 'Stop scanning and try again.'
                )
            if self._scan_settings is None:
                raise RuntimeError('Configure forward scan settings first.')

            # check settings - will raise appropriate exceptions if something is not right
            self.constraints.check_back_scan_settings(backward_settings=settings, forward_settings=self._scan_settings)
            self._back_scan_settings = settings

    def move_absolute(self, position, velocity=None, blocking=False):
        """Move the scanning probe to an absolute position as fast as possible or with a defined
        velocity.

        Log error and return current target position if something fails or a 1D/2D scan is in
        progress.
        """
        with self._thread_lock:
            if self.module_state() != 'idle':
                raise RuntimeError('Scanning in progress. Unable to move to position.')
            elif not set(position).issubset(self._position_ranges):
                raise ValueError(
                    f'Invalid axes encountered in position dict. ' f'Valid axes are: {set(self._position_ranges)}'
                )
            else:
                move_distance = {ax: np.abs(pos - self._current_position[ax]) for ax, pos in position.items()}
                if velocity is None:
                    move_time = 0.01
                else:
                    move_time = max(0.01, np.sqrt(np.sum(dist**2 for dist in move_distance.values())) / velocity)
                    
                # Simulate movement time
                time.sleep(move_time)
                
                # Update the position
                self._current_position.update(position)
                
                # Update the simulator position
                self._qudi_facade.confocal_simulator.set_position(
                    self._current_position.get('x', None),
                    self._current_position.get('y', None),
                    self._current_position.get('z', None)
                )
                
            return self._current_position

    def move_relative(self, distance, velocity=None, blocking=False):
        """Move the scanning probe by a relative distance from the current target position as fast
        as possible or with a defined velocity.

        Log error and return current target position if something fails or a 1D/2D scan is in
        progress.
        """
        with self._thread_lock:
            self.log.debug('NV simulator scanning probe "move_relative" called.')
            if self.module_state() != 'idle':
                raise RuntimeError('Scanning in progress. Unable to move relative.')
            elif not set(distance).issubset(self._position_ranges):
                raise ValueError(
                    'Invalid axes encountered in distance dict. ' f'Valid axes are: {set(self._position_ranges)}'
                )
            else:
                new_pos = {ax: self._current_position[ax] + dist for ax, dist in distance.items()}
                if velocity is None:
                    move_time = 0.01
                else:
                    move_time = max(0.01, np.sqrt(np.sum(dist**2 for dist in distance.values())) / velocity)
                    
                # Simulate movement time
                time.sleep(move_time)
                
                # Update the position
                self._current_position.update(new_pos)
                
                # Update the simulator position
                self._qudi_facade.confocal_simulator.set_position(
                    self._current_position.get('x', None),
                    self._current_position.get('y', None),
                    self._current_position.get('z', None)
                )
                
            return self._current_position

    def get_target(self):
        """
        Retrieve the current target position of the scanner hardware.

        Returns
        -------
        dict
            A dictionary representing the current target position for each axis.
        """
        with self._thread_lock:
            return self._current_position.copy()

    def get_position(self):
        """
        Retrieve a snapshot of the actual scanner position from position feedback sensors.

        Returns
        -------
        dict
            A dictionary representing the current actual position for each axis.
        """
        with self._thread_lock:
            self.log.debug('NV simulator scanning probe "get_position" called.')
            # Add some random noise to simulate position inaccuracy
            position = {
                ax: pos + np.random.normal(0, self._position_accuracy[ax]) for ax, pos in self._current_position.items()
            }
            return position

    def start_scan(self):
        """Start a scan as configured beforehand.
        Log an error if something fails or a 1D/2D scan is in progress.
        """
        with self._thread_lock:
            self.log.debug('NV simulator scanning probe "start_scan" called.')
            if self.module_state() != 'idle':
                raise RuntimeError('Cannot start scan. Scan already in progress.')
            if not self.scan_settings:
                raise RuntimeError('No scan settings configured. Cannot start scan.')
                
            self.module_state.lock()

            # Initialize scan data
            self._scan_data = ScanData.from_constraints(
                settings=self.scan_settings, 
                constraints=self.constraints, 
                scanner_target_at_start=self.get_target()
            )
            self._scan_data.new_scan()

            if self._back_scan_settings is not None:
                self._back_scan_data = ScanData.from_constraints(
                    settings=self.back_scan_settings,
                    constraints=self.constraints,
                    scanner_target_at_start=self.get_target(),
                )
                self._back_scan_data.new_scan()

            self._scan_start_time = time.time()
            self._last_forward_pixel = 0
            self._last_backward_pixel = 0
            
            # Calculate timer interval - update twice per line
            line_time = self.scan_settings.resolution[0] / self.scan_settings.frequency
            timer_interval_ms = int(0.5 * line_time * 1000)
            self._update_timer.setInterval(timer_interval_ms)

        self._start_timer()

    def stop_scan(self):
        """Stop the currently running scan.
        Log an error if something fails or no 1D/2D scan is in progress.
        """
        self.log.debug('NV simulator scanning probe "stop_scan" called.')
        if self.module_state() == 'locked':
            self._stop_timer()
            self.module_state.unlock()
        else:
            raise RuntimeError('No scan in progress. Cannot stop scan.')

    def emergency_stop(self):
        """Emergency stop the scan process."""
        try:
            self.module_state.unlock()
        except FysomError:
            pass
        self.log.warning('NV simulator scanner has been emergency stopped.')

    def _handle_timer(self):
        """Update during a running scan."""
        try:
            with self._thread_lock:
                self._update_scan_data()
        except Exception as e:
            self.log.error("Could not update scan data.", exc_info=e)

    def _update_scan_data(self) -> None:
        """Update scan data by simulating fluorescence at each scan position."""
        if self.module_state() == 'idle':
            raise RuntimeError("Scan is not running.")

        # Calculate how far the scan has progressed based on time elapsed
        t_elapsed = time.time() - self._scan_start_time
        t_forward = self.scan_settings.resolution[0] / self.scan_settings.frequency
        
        if self.back_scan_settings is not None:
            back_resolution = self.back_scan_settings.resolution[0]
            t_backward = back_resolution / self.back_scan_settings.frequency
        else:
            back_resolution = 0
            t_backward = 0
            
        t_complete_line = t_forward + t_backward

        # Calculate how many lines have been completed and current position in the line
        aq_lines = int(t_elapsed / t_complete_line)
        t_current_line = t_elapsed % t_complete_line
        
        if t_current_line < t_forward:
            # Currently in forwards scan
            aq_px_backward = back_resolution * aq_lines
            aq_lines_forward = aq_lines + (t_current_line / t_forward)
            aq_px_forward = int(self.scan_settings.resolution[0] * aq_lines_forward)
        else:
            # Currently in backwards scan
            aq_px_forward = self.scan_settings.resolution[0] * (aq_lines + 1)
            aq_lines_backward = aq_lines + (t_current_line - t_forward) / t_backward
            aq_px_backward = int(back_resolution * aq_lines_backward)

        # Calculate the scan vectors for the newly acquired pixels
        scan_vectors = self._calculate_scan_vectors(self._last_forward_pixel, aq_px_forward)
        
        # Generate fluorescence data for each new position
        new_forward_data = self._generate_fluorescence_data(scan_vectors)
        
        # Update scan data with the newly computed values
        # transposing is necessary to fill along the fast axis first
        for ch in self.constraints.channels:
            self._scan_data.data[ch].T.flat[self._last_forward_pixel : aq_px_forward] = new_forward_data
            
        self._last_forward_pixel = aq_px_forward

        # Handle back scan if configured
        if self._back_scan_settings is not None:
            back_scan_vectors = self._calculate_back_scan_vectors(self._last_backward_pixel, aq_px_backward)
            new_backward_data = self._generate_fluorescence_data(back_scan_vectors)
            
            for ch in self.constraints.channels:
                self._back_scan_data.data[ch].T.flat[self._last_backward_pixel : aq_px_backward] = new_backward_data
                
            self._last_backward_pixel = aq_px_backward

        # Check if scan is finished
        if self.scan_settings.scan_dimension == 1:
            is_finished = aq_lines >= 1
        else:
            is_finished = aq_lines >= self.scan_settings.resolution[1]
            
        if is_finished:
            self.module_state.unlock()
            self.log.debug("Scan finished.")
        else:
            self._start_timer()

    def get_scan_data(self) -> Optional[ScanData]:
        """Retrieve the ScanData instance used in the scan."""
        with self._thread_lock:
            if self._scan_data is None:
                return None
            else:
                return self._scan_data.copy()

    def get_back_scan_data(self) -> Optional[ScanData]:
        """Retrieve the ScanData instance used in the backwards scan."""
        with self._thread_lock:
            if self._back_scan_data is None:
                return None
            return self._back_scan_data.copy()

    def _start_timer(self):
        """Start the update timer."""
        if self.thread() is not QtCore.QThread.currentThread():
            QtCore.QMetaObject.invokeMethod(self._update_timer, 'start', QtCore.Qt.BlockingQueuedConnection)
        else:
            self._update_timer.start()

    def _stop_timer(self):
        """Stop the update timer."""
        if self.thread() is not QtCore.QThread.currentThread():
            QtCore.QMetaObject.invokeMethod(self._update_timer, 'stop', QtCore.Qt.BlockingQueuedConnection)
        else:
            self._update_timer.stop()

    def _calculate_scan_vectors(self, start_pixel, end_pixel):
        """Calculate the scan positions for a range of pixels."""
        # Get the axes being scanned
        axes = self.scan_settings.axes
        
        # Get the total number of pixels
        total_pixels = np.prod(self.scan_settings.resolution)
        
        # Get the scan ranges for each axis
        ranges = self.scan_settings.range
        
        # Calculate the position for each pixel in the range
        positions = {}
        for i, axis in enumerate(axes):
            if i == 0:  # Fast axis
                # Calculate the fast axis position for each pixel
                axis_range = np.linspace(ranges[0][0], ranges[0][1], self.scan_settings.resolution[0])
                pos_indices = np.arange(start_pixel, end_pixel) % self.scan_settings.resolution[0]
                positions[axis] = axis_range[pos_indices]
            elif i == 1 and self.scan_settings.scan_dimension > 1:  # Slow axis
                # Calculate the slow axis position for each pixel
                axis_range = np.linspace(ranges[1][0], ranges[1][1], self.scan_settings.resolution[1])
                pos_indices = np.arange(start_pixel, end_pixel) // self.scan_settings.resolution[0]
                positions[axis] = axis_range[pos_indices]
            else:
                # For other axes, use the current position
                positions[axis] = np.ones(end_pixel - start_pixel) * self._current_position[axis]
                
        return positions

    def _calculate_back_scan_vectors(self, start_pixel, end_pixel):
        """Calculate the scan positions for a range of pixels in the back scan."""
        # Similar to _calculate_scan_vectors but using back_scan_settings
        if self._back_scan_settings is None:
            return {}
            
        axes = self._back_scan_settings.axes
        ranges = self._back_scan_settings.range
        
        positions = {}
        for i, axis in enumerate(axes):
            if i == 0:  # Fast axis (reversed for back scan)
                axis_range = np.linspace(ranges[0][1], ranges[0][0], self._back_scan_settings.resolution[0])
                pos_indices = np.arange(start_pixel, end_pixel) % self._back_scan_settings.resolution[0]
                positions[axis] = axis_range[pos_indices]
            elif i == 1 and self._back_scan_settings.scan_dimension > 1:  # Slow axis
                axis_range = np.linspace(ranges[1][0], ranges[1][1], self._back_scan_settings.resolution[1])
                pos_indices = np.arange(start_pixel, end_pixel) // self._back_scan_settings.resolution[0]
                positions[axis] = axis_range[pos_indices]
            else:
                positions[axis] = np.ones(end_pixel - start_pixel) * self._current_position[axis]
                
        return positions

    def _generate_fluorescence_data(self, scan_vectors):
        """Generate fluorescence data for the given scan positions."""
        # Check if we have positions to process
        if not scan_vectors or all(len(pos) == 0 for pos in scan_vectors.values()):
            return np.array([])
            
        # Number of pixels to generate data for
        num_pixels = len(next(iter(scan_vectors.values())))
        
        # For each position, set the confocal position and get the fluorescence
        fluorescence_data = np.zeros(num_pixels)
        
        for i in range(num_pixels):
            # Get the position for this pixel
            position = {axis: values[i] for axis, values in scan_vectors.items()}
            
            # Set the position in the simulator
            self._qudi_facade.confocal_simulator.set_position(
                position.get('x', None),
                position.get('y', None),
                position.get('z', None)
            )
            
            # Get the fluorescence rate from the simulator
            # Apply laser excitation
            self._qudi_facade.laser_controller.on()
            
            # Get the fluorescence rate (counts per second)
            count_rate = self._qudi_facade.nv_model.get_fluorescence_rate()
            
            # Add some randomness to the count rate to simulate photon statistics
            # Use Poisson distribution for realistic noise
            integration_time = 1.0 / self.scan_settings.frequency  # seconds per pixel
            photon_count = np.random.poisson(count_rate * integration_time)
            fluorescence_data[i] = photon_count / integration_time  # Convert back to rate
            
            # Turn off laser
            self._qudi_facade.laser_controller.off()
            
        return fluorescence_data