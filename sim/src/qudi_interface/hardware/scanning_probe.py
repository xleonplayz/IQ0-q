# -*- coding: utf-8 -*-

"""
Qudi hardware interface adapter for NV simulator scanning probe.
This module implements the ScanningProbeInterface for the NV center simulator,
enabling confocal microscopy and other scanning measurements.

Copyright (c) 2023
"""

import numpy as np
import time
import threading
from typing import Dict, Any, List, Tuple, Union, Optional

from qudi.interface.scanning_probe_interface import ScanningProbeInterface, ScannerAxis
from qudi.interface.scanning_probe_interface import ScannerChannel, BackScanCapability
from qudi.interface.scanning_probe_interface import ScanConstraints, ScannerSettings

from .qudi_facade import QudiFacade


class NVSimScanningProbe(ScanningProbeInterface):
    """
    Hardware adapter that implements the ScanningProbeInterface for the NV center simulator.
    This interface enables confocal microscopy and other scanning measurements.
    """

    def __init__(self, config=None, **kwargs):
        """
        Initialize the scanning probe adapter for the NV simulator.
        
        @param config: Configuration dictionary
        @param **kwargs: Additional keyword arguments for the base class
        """
        # Initialize the module base class
        super().__init__(config=config, **kwargs)
        
        # Get the Qudi facade instance
        self._qudi_facade = QudiFacade(config)
        self._simulator = self._qudi_facade.get_nv_model()
        self._confocal_simulator = self._qudi_facade.get_confocal_simulator()
        
        # Check if confocal simulator is available
        if self._confocal_simulator is None:
            self.log.warning("Confocal simulator not available, scanning functionality will be limited")
        
        # Parse configuration
        self._config = self.config
        
        # Initialize scanner settings
        self._position = {'x': 0.0, 'y': 0.0, 'z': 0.0}  # Current position in meters
        self._target_position = self._position.copy()  # Target position for movement
        self._resolution = 100  # Resolution for scanning
        self._range = {'x': (-50e-6, 50e-6), 
                      'y': (-50e-6, 50e-6), 
                      'z': (-50e-6, 50e-6)}  # Scanner range in meters
        self._backscan_ratio = 0.1  # Fraction of the forward scan dedicated to the backward scan
        
        # Apply configuration if provided
        if 'initial_position' in self._config:
            pos = self._config['initial_position']
            for axis in pos:
                if axis in self._position:
                    self._position[axis] = pos[axis]
                    self._target_position[axis] = pos[axis]
        
        if 'scanner_range' in self._config:
            ranges = self._config['scanner_range']
            for axis in ranges:
                if axis in self._range:
                    self._range[axis] = ranges[axis]
        
        # Initialize scanning state
        self._is_scanning = False
        self._scan_thread = None
        self._stop_scan = threading.Event()
        
        # Scanner channels
        self._scanner_channels = {
            'fluorescence': ScannerChannel(name='fluorescence', unit='counts/s'),
            'reflected': ScannerChannel(name='reflected', unit='norm')
        }
        
        # Thread lock for thread safety
        self._thread_lock = self.module_state.lock_access()
        
        self.log.info("NV Simulator scanning probe initialized")

    def on_activate(self):
        """
        Called when the module is activated
        """
        self.log.info("NV Simulator scanning probe activated")
        
        # Ensure simulator is initialized
        if self._confocal_simulator is not None:
            # Set initial position in confocal simulator
            self._confocal_simulator.set_position(
                self._position['x'],
                self._position['y'],
                self._position['z']
            )

    def on_deactivate(self):
        """
        Called when the module is deactivated
        """
        # Stop any running scan
        self.stop_scan()
        self.log.info("NV Simulator scanning probe deactivated")

    def get_constraints(self):
        """
        Get hardware constraints/limitations of the scanner.
        
        @return ScanConstraints: Scanner constraints object
        """
        axes = {
            'x': ScannerAxis(
                name='x',
                unit='m',
                value_range=self._range['x'],
                step_range=(1e-9, 1e-3),  # 1 nm to 1 mm steps
                resolution_range=(1, 10000),  # 1 to 10000 pixels
            ),
            'y': ScannerAxis(
                name='y',
                unit='m',
                value_range=self._range['y'],
                step_range=(1e-9, 1e-3),
                resolution_range=(1, 10000),
            ),
            'z': ScannerAxis(
                name='z',
                unit='m',
                value_range=self._range['z'],
                step_range=(1e-9, 1e-3),
                resolution_range=(1, 10000),
            )
        }
        
        channels = {ch.name: ch for ch in self._scanner_channels.values()}
        
        backscan_capabilities = BackScanCapability.AVAILABLE | BackScanCapability.FULLY_CONFIGURABLE
        
        return ScanConstraints(
            axes=axes,
            channels=channels,
            backscan_capability=backscan_capabilities,
            has_position_feedback=True,
            maximum_frequency=1000.0  # 1 kHz maximum scan frequency
        )

    def reset(self):
        """
        Reset the hardware settings to the default state.
        """
        with self._thread_lock:
            # Stop any running scan
            self.stop_scan()
            
            # Reset position to center
            self._position = {'x': 0.0, 'y': 0.0, 'z': 0.0}
            self._target_position = self._position.copy()
            
            # Apply to confocal simulator if available
            if self._confocal_simulator is not None:
                self._confocal_simulator.set_position(
                    self._position['x'],
                    self._position['y'],
                    self._position['z']
                )
            
            self.log.info("Scanner reset to default state")
            
    def configure_scan(self, settings):
        """
        Configure the scan settings. These are independent from the scanner settings
        (like position, voltage ranges etc.) and represent scan specific configurations
        like the scan range, scan resolution, scan mode, etc.
        
        @param ScannerSettings settings: ScannerSettings instance holding all scan settings
        
        @return ScannerSettings: The actual (i.e. updated/applied/valid) settings
        """
        # Validate and update scan settings
        with self._thread_lock:
            if self._is_scanning:
                self.log.error("Cannot configure scan while scanner is running")
                return self.get_scan_settings()
            
            # Apply resolution
            # Note: This is just for demonstration, in a real scanner you'd apply
            # more of the scan settings here
            self._resolution = settings.resolution
            
            # Return the updated settings
            return self.get_scan_settings()

    def get_scan_settings(self):
        """
        Get the currently configured scan settings.
        This method returns the actual scan settings that will be used for the next scan.
        It should return the default settings if no scan has been configured yet.
        
        @return ScannerSettings: The current scan settings
        """
        # Create a ScannerSettings object with current settings
        return ScannerSettings(
            resolution=self._resolution,
            forward_range={'x': self._range['x'], 'y': self._range['y'], 'z': self._range['z']},
            backward_range={'x': self._range['x'], 'y': self._range['y'], 'z': self._range['z']},
            forward_axes=frozenset({'x', 'y'}),  # Default: scan in xy plane
            backward_axes=frozenset({'x'}),  # Default: backscan in x
            static_axes=frozenset({'z'}),  # Default: static z
            backscan_frequency_factor=self._backscan_ratio,
            backscan_resolution_factor=self._backscan_ratio,
            analog_channels=frozenset(self._scanner_channels.keys()),
            backscan_analog_channels=frozenset()  # No backscan channels by default
        )

    def start_scan(self):
        """
        Start a new scan based on the current scan settings.
        
        @return int: error code (0: OK, -1: error)
        """
        with self._thread_lock:
            if self._is_scanning:
                self.log.error("Scanner is already running")
                return -1
            
            # Check if confocal simulator is available for scanning
            if self._confocal_simulator is None:
                self.log.error("Confocal simulator not available, cannot start scan")
                return -1
            
            # Reset stop flag
            self._stop_scan.clear()
            
            # Set status and start thread
            self._is_scanning = True
            
            # Start scan thread
            self._scan_thread = threading.Thread(
                target=self._scan_thread_target,
                args=()
            )
            self._scan_thread.daemon = True
            self._scan_thread.start()
            
            self.log.info("Scan started")
            return 0

    def stop_scan(self):
        """
        Stop a currently running scan.
        
        @return int: error code (0: OK, -1: error)
        """
        with self._thread_lock:
            if not self._is_scanning:
                return 0
            
            # Set stop flag
            self._stop_scan.set()
            
            # Wait for scan thread to finish (with timeout)
            if self._scan_thread is not None and self._scan_thread.is_alive():
                self._scan_thread.join(timeout=1.0)
            
            # Update status
            self._is_scanning = False
            self.log.info("Scan stopped")
            return 0

    def scan_line(self):
        """
        Perform a single line scan in the currently configured forward and/or backward scan axes.
        
        @return dict: {'forward': {axis_name: array of positions}, 
                      'backward': {axis_name: array of positions},
                      'forward_analog': {channel_name: array of values},
                      'backward_analog': {channel_name: array of values}}
        """
        raise NotImplementedError("This method is not implemented for the NV simulator scanner")

    def get_scanner_axes(self):
        """
        Get the available axes of this scanner.
        
        @return frozenset(str): Set of available axis names
        """
        return frozenset(self._position.keys())

    def get_scanner_channels(self):
        """
        Get the available data channels of this scanner.
        
        @return frozenset(str): Set of available scanner channel names
        """
        return frozenset(self._scanner_channels.keys())

    def get_scanner_count_channels(self):
        """
        Get the available counter channels to be used as data channels of this scanner.
        
        @return frozenset(str): Set of available counter channel names
        """
        # For the simulator, we just use the same channels as scanner channels
        return self.get_scanner_channels()

    def get_scanner_position(self):
        """
        Get the current scanner position in absolute coordinates.
        
        @return dict: Axis names as keys, absolute position in axis unit as values
        """
        # Return a copy of the current position
        return self._position.copy()

    def get_scanner_target(self):
        """
        Get the target scanner position in absolute coordinates.
        
        @return dict: Axis names as keys, absolute target position in axis unit as values
        """
        # Return a copy of the target position
        return self._target_position.copy()

    def set_position(self, x=None, y=None, z=None):
        """
        Move the scanner to a specific position in absolute coordinates.
        This method expects multiple keyword arguments with axis names as keys and absolute
        target position coordinates as values.
        For valid axis name keys see the constraints.axes attribute returned by get_constraints().
        
        @param float x: position in x-direction (default: None)
        @param float y: position in y-direction (default: None)
        @param float z: position in z-direction (default: None)
        
        @return dict: Actual position of the scanner after movement as dict with all
                      axis names as keys.
        """
        with self._thread_lock:
            # Update target position with provided values
            if x is not None:
                self._target_position['x'] = x
            
            if y is not None:
                self._target_position['y'] = y
            
            if z is not None:
                self._target_position['z'] = z
            
            # Update current position (immediate move for simulator)
            self._position = self._target_position.copy()
            
            # Apply to confocal simulator if available
            if self._confocal_simulator is not None:
                self._confocal_simulator.set_position(
                    self._position['x'],
                    self._position['y'],
                    self._position['z']
                )
            
            self.log.debug(f"Scanner position set to x={self._position['x']}, y={self._position['y']}, z={self._position['z']}")
            
            # Return the current position
            return self._position.copy()

    def move_position(self, dx=None, dy=None, dz=None):
        """
        Move the scanner by the given differential amounts from its current position.
        This method expects multiple keyword arguments with axis names as keys and relative
        position changes as values.
        For valid axis name keys see the constraints.axes attribute returned by get_constraints().
        
        @param float dx: position relative movement in x-direction (default: None)
        @param float dy: position relative movement in y-direction (default: None)
        @param float dz: position relative movement in z-direction (default: None)
        
        @return dict: Actual position of the scanner after movement as dict with all
                      axis names as keys.
        """
        with self._thread_lock:
            # Calculate new target position based on current position and relative movement
            if dx is not None:
                self._target_position['x'] = self._position['x'] + dx
            
            if dy is not None:
                self._target_position['y'] = self._position['y'] + dy
            
            if dz is not None:
                self._target_position['z'] = self._position['z'] + dz
            
            # Update current position (immediate move for simulator)
            self._position = self._target_position.copy()
            
            # Apply to confocal simulator if available
            if self._confocal_simulator is not None:
                self._confocal_simulator.set_position(
                    self._position['x'],
                    self._position['y'],
                    self._position['z']
                )
            
            self.log.debug(f"Scanner moved by dx={dx}, dy={dy}, dz={dz} to " +
                          f"x={self._position['x']}, y={self._position['y']}, z={self._position['z']}")
            
            # Return the current position
            return self._position.copy()

    def get_position_feedback(self, all_axes=None):
        """
        Get a list of position feedbacks for the specified axes. Position feedback data can 
        either be directly measured from the scanner hardware or generated from the target 
        position set by set_position(). The position feedback information is critical to obtain
        value pairs from encoder and scanning sequences.

        @param set all_axes: Set of strings for the axes to get the position feedback for
                                if None, feedback is given for all scanner axes
        
        @return dict: Dictionary containing the position feedback for the specified axes
        """
        # For the simulator, we just return the current position
        if all_axes is None:
            all_axes = self.get_scanner_axes()
            
        feedback = {}
        for axis in all_axes:
            if axis in self._position:
                feedback[axis] = self._position[axis]
                
        return feedback

    def _scan_thread_target(self):
        """
        Thread target for simulated scanning.
        This simulates a full scan process by generating a raster scan pattern
        and collecting data at each point.
        """
        try:
            # Get current scan settings
            settings = self.get_scan_settings()
            
            # Get the scan axes
            forward_axes = list(settings.forward_axes)
            if len(forward_axes) < 1:
                self.log.error("No forward scan axes defined")
                return
                
            # For simplicity, we'll implement a basic 2D scan (assuming x,y are the forward axes)
            if 'x' in forward_axes and 'y' in forward_axes:
                # Create a 2D scan grid
                x_range = settings.forward_range['x']
                y_range = settings.forward_range['y']
                
                # Set z position (assuming z is a static axis)
                z_pos = self._position['z']
                
                # Calculate step sizes
                x_step = (x_range[1] - x_range[0]) / (settings.resolution - 1)
                y_step = (y_range[1] - y_range[0]) / (settings.resolution - 1)
                
                # Perform raster scan
                for y_idx in range(settings.resolution):
                    # Check if scan should be stopped
                    if self._stop_scan.is_set():
                        break
                        
                    # Calculate y position
                    y_pos = y_range[0] + y_idx * y_step
                    
                    # Scan a line in x direction
                    for x_idx in range(settings.resolution):
                        # Check if scan should be stopped
                        if self._stop_scan.is_set():
                            break
                            
                        # Calculate x position
                        x_pos = x_range[0] + x_idx * x_step
                        
                        # Move to position
                        self.set_position(x=x_pos, y=y_pos, z=z_pos)
                        
                        # Simulate data acquisition time
                        time.sleep(0.001)  # 1 ms per point
                        
                        # Signal point acquisition (done through the Qudi Logic module)
                        # The actual data collection is handled by Qudi
            
            elif 'x' in forward_axes:
                # 1D scan along x axis
                x_range = settings.forward_range['x']
                
                # Keep current y and z positions
                y_pos = self._position['y']
                z_pos = self._position['z']
                
                # Calculate step size
                x_step = (x_range[1] - x_range[0]) / (settings.resolution - 1)
                
                # Perform line scan
                for x_idx in range(settings.resolution):
                    # Check if scan should be stopped
                    if self._stop_scan.is_set():
                        break
                        
                    # Calculate x position
                    x_pos = x_range[0] + x_idx * x_step
                    
                    # Move to position
                    self.set_position(x=x_pos, y=y_pos, z=z_pos)
                    
                    # Simulate data acquisition time
                    time.sleep(0.001)  # 1 ms per point
                    
                    # Signal point acquisition (done through the Qudi Logic module)
                    
            elif 'y' in forward_axes:
                # 1D scan along y axis
                y_range = settings.forward_range['y']
                
                # Keep current x and z positions
                x_pos = self._position['x']
                z_pos = self._position['z']
                
                # Calculate step size
                y_step = (y_range[1] - y_range[0]) / (settings.resolution - 1)
                
                # Perform line scan
                for y_idx in range(settings.resolution):
                    # Check if scan should be stopped
                    if self._stop_scan.is_set():
                        break
                        
                    # Calculate y position
                    y_pos = y_range[0] + y_idx * y_step
                    
                    # Move to position
                    self.set_position(x=x_pos, y=y_pos, z=z_pos)
                    
                    # Simulate data acquisition time
                    time.sleep(0.001)  # 1 ms per point
                    
                    # Signal point acquisition (done through the Qudi Logic module)
            
            else:
                self.log.error("Unsupported scan axes configuration")
            
        except Exception as e:
            self.log.error(f"Error in scan thread: {str(e)}")
        finally:
            self._is_scanning = False
            self.log.info("Scan completed")