# -*- coding: utf-8 -*-

"""
This file contains the Qudi NV center simulator manager.

Copyright (c) 2021, the qudi developers. See the AUTHORS.md file at the top-level directory of this
distribution and on <https://github.com/Ulm-IQO/qudi-iqo-modules/>

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

import os
import time
import copy
import threading
import logging
import numpy as np
from qudi.core.configoption import ConfigOption
from qudi.core.module import Base
from qudi.util.mutex import RecursiveMutex
from qudi.hardware.nv_simulator.model import PhysicalNVModel

class SimulatorManager(Base):
    """
    Central manager for NV center simulator integration with Qudi dummy modules.
    
    This class implements a singleton pattern to ensure a single simulator instance
    is shared across all dummy hardware modules. It provides a simplified access layer
    to the underlying NV simulator functionality with robust error handling.
    """
    
    # Config options
    _zero_field_splitting = ConfigOption('zero_field_splitting', 2.87e9)
    _gyromagnetic_ratio = ConfigOption('gyromagnetic_ratio', 28.025e9)
    _t1 = ConfigOption('t1', 5.0e-3)
    _t2 = ConfigOption('t2', 1.0e-5)
    _temperature = ConfigOption('temperature', 300.0)
    _magnetic_field = ConfigOption('magnetic_field', [0, 0, 0])
    _c13_concentration = ConfigOption('c13_concentration', 0.011)
    _optics = ConfigOption('optics', True)
    _nitrogen = ConfigOption('nitrogen', False)
    
    # Singleton instance
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton implementation."""
        if cls._instance is None:
            cls._instance = super(SimulatorManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, *args, **kwargs):
        """Initialize the simulator manager."""
        # Skip re-initialization if already done
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        # Initialize base class first
        super().__init__(*args, **kwargs)
        
        # Mark as not fully initialized yet to prevent recursion issues
        self._initialized = False
        
        # Set up thread lock early for thread-safe initialization
        self._thread_lock = RecursiveMutex()
        
        # This will be filled in on_activate
        self.nv_model = None
        self.confocal_simulator = None
        self.active_modules = set()
        self._module_watchdogs = {}
        self._last_health_check = 0
        self._health_timer = None
        self._is_running = False
    
    def on_activate(self):
        """Called when module is activated."""
        with self._thread_lock:
            try:
                # Create simulator instance with parameters from config
                simulator_params = {
                    'zero_field_splitting': self._zero_field_splitting,
                    'gyromagnetic_ratio': self._gyromagnetic_ratio,
                    't1': self._t1,
                    't2': self._t2,
                    'temperature': self._temperature,
                    'c13_concentration': self._c13_concentration,
                    'optics': self._optics,
                    'nitrogen': self._nitrogen,
                    'thread_safe': True
                }
                
                self.nv_model = PhysicalNVModel(**simulator_params)
                self.log.info("NV center simulator model initialized")
                
                # Apply magnetic field if specified
                if self._magnetic_field is not None:
                    self.nv_model.set_magnetic_field(self._magnetic_field)
                    self.log.info(f"Applied magnetic field: {self._magnetic_field}")
                
                # Initialize additional components
                self._init_confocal_simulator()
                
                # Set up health monitoring
                self._last_health_check = time.time()
                self._is_running = True
                self._init_health_monitoring()
                
                # Mark as successfully initialized
                self._initialized = True
                self.log.info("NV Simulator Manager initialized successfully")
                
            except Exception as e:
                self.log.error(f"Failed to initialize simulator manager: {str(e)}")
                # Set up fallback simulator for resilience
                self._setup_fallback_simulator()
                # We still consider it initialized to avoid repeated errors
                self._initialized = True
    
    def on_deactivate(self):
        """Called when module is deactivated."""
        # Stop health monitoring
        if hasattr(self, '_health_timer') and self._health_timer is not None:
            try:
                self._health_timer.stop()
            except:
                pass
                
        # Set state
        self._is_running = False
        
        # Clear registrations
        self.active_modules.clear()
        
        # Individual cleanup
        self.log.info("Simulator manager deactivated")
    
    def _init_confocal_simulator(self):
        """Initialize confocal simulator if needed."""
        # Confocal simulator would be implemented here
        # For now, we'll leave it as a placeholder
        self.confocal_simulator = None
    
    def _init_health_monitoring(self):
        """Initialize health monitoring according to environment."""
        try:
            # Use Qt timer for periodic checks
            from qtpy import QtCore
            
            self._health_timer = QtCore.QTimer()
            self._health_timer.timeout.connect(self._health_check)
            self._health_timer.start(5000)  # Check every 5 seconds
            self.log.debug("Health monitoring initialized with Qt timer")
            
        except ImportError:
            # Fall back to thread-based monitoring (for non-Qt environments)
            self._health_thread = threading.Thread(
                target=self._health_monitor_loop,
                daemon=True
            )
            self._health_thread.start()
            self.log.debug("Health monitoring initialized with background thread")
    
    def _health_monitor_loop(self):
        """Background thread for health monitoring."""
        while self._is_running:
            try:
                self._health_check()
            except Exception as e:
                self.log.error(f"Error in health monitor: {str(e)}")
                
            # Sleep between checks
            time.sleep(5)
    
    def _health_check(self):
        """Check simulator health and attempt recovery if needed."""
        with self._thread_lock:
            try:
                # Simple test operation to verify simulator is responsive
                if hasattr(self, 'nv_model') and self.nv_model is not None:
                    test = self.nv_model.get_fluorescence() is not None
                    
                    # If test fails, attempt recovery
                    if not test:
                        self.log.warning("Simulator health check failed, attempting recovery")
                        self._reinitialize_simulator()
                        
                # Check for stale module registrations
                self._check_module_watchdogs()
                
                # Update last health check timestamp
                self._last_health_check = time.time()
                
            except Exception as e:
                self.log.error(f"Health check failed: {str(e)}")
                self._reinitialize_simulator()
    
    def _check_module_watchdogs(self):
        """Check for stale module registrations."""
        current_time = time.time()
        stale_modules = []
        
        # Identify modules that haven't been active for 30 seconds
        for module, last_seen in self._module_watchdogs.items():
            if current_time - last_seen > 30:
                stale_modules.append(module)
        
        # Remove stale modules
        for module in stale_modules:
            if module in self.active_modules:
                self.active_modules.remove(module)
            self._module_watchdogs.pop(module, None)
            self.log.warning(f"Module {module} appears to be stale, unregistered")
    
    def _reinitialize_simulator(self):
        """Attempt to recover a damaged simulator instance."""
        try:
            # Create a fresh simulator instance
            simulator_params = {
                'zero_field_splitting': self._zero_field_splitting,
                'gyromagnetic_ratio': self._gyromagnetic_ratio,
                't1': self._t1,
                't2': self._t2,
                'temperature': self._temperature,
                'c13_concentration': self._c13_concentration,
                'optics': self._optics,
                'nitrogen': self._nitrogen,
                'thread_safe': True
            }
            
            self.nv_model = PhysicalNVModel(**simulator_params)
            
            # Apply magnetic field if specified
            if self._magnetic_field is not None:
                self.nv_model.set_magnetic_field(self._magnetic_field)
            
            self.log.info("Successfully reinitialized simulator")
            return True
        except Exception as e:
            self.log.error(f"Failed to reinitialize simulator: {str(e)}")
            self._setup_fallback_simulator()
            return False
    
    def _setup_fallback_simulator(self):
        """Set up a minimal fallback simulator for graceful degradation."""
        # Create a minimal stand-in with the same API but simplified behavior
        class MinimalNVModel:
            def get_fluorescence(self):
                return 1e5
                
            def apply_microwave(self, *args, **kwargs):
                pass
                
            def apply_laser(self, *args, **kwargs):
                pass
                
            def set_magnetic_field(self, *args, **kwargs):
                pass
                
            def reset_state(self):
                pass
                
            def evolve(self, duration):
                pass
                
            def simulate_odmr(self, f_min, f_max, n_points, mw_power):
                # Return a simple ODMR spectrum with two dips
                frequencies = np.linspace(f_min, f_max, n_points)
                signal = np.ones(n_points)
                
                # Create dips at typical NV frequencies
                center = 2.87e9
                width = 5e6
                depth = 0.3
                signal -= depth * width**2 / ((frequencies - center - 5e6)**2 + width**2)
                signal -= depth * width**2 / ((frequencies - center + 5e6)**2 + width**2)
                
                # Add noise
                signal += np.random.normal(0, 0.01, n_points)
                signal *= 1e5  # Scale to counts/s
                
                class Result:
                    def __init__(self):
                        self.frequencies = frequencies
                        self.signal = signal
                return Result()
                
            def simulate_rabi(self, t_max, n_points, mw_power, mw_frequency):
                # Return a simple damped oscillation
                times = np.linspace(0, t_max, n_points)
                signal = 1 - 0.3 * np.sin(2 * np.pi * 5e6 * times)**2 * np.exp(-times/5e-6)
                signal *= 1e5  # Scale to counts/s
                
                class Result:
                    def __init__(self):
                        self.times = times
                        self.signal = signal
                return Result()
                
            def simulate_t1(self, t_max, n_points):
                # Return a simple T1 relaxation curve
                times = np.linspace(0, t_max, n_points)
                signal = (1 - np.exp(-times/5e-3)) * 1e5  # Assuming 5ms T1
                
                class Result:
                    def __init__(self):
                        self.times = times
                        self.signal = signal
                return Result()
        
        self.nv_model = MinimalNVModel()
        self.log.warning("Using minimal fallback simulator")
    
    def register_module(self, module_name):
        """Register a module as active with the simulator."""
        with self._thread_lock:
            # Ensure initialization
            if not self._initialized:
                self.log.warning("Simulator not fully initialized during module registration")
                
            self.active_modules.add(module_name)
            self._module_watchdogs[module_name] = time.time()
            self.log.debug(f"Module {module_name} registered with simulator")
    
    def unregister_module(self, module_name):
        """Unregister a module from the simulator."""
        with self._thread_lock:
            if module_name in self.active_modules:
                self.active_modules.remove(module_name)
                self._module_watchdogs.pop(module_name, None)
                self.log.debug(f"Module {module_name} unregistered from simulator")
            
            # If no more active modules, consider cleaning up resources
            if len(self.active_modules) == 0:
                self.log.info("No active modules, simulator resources can be released")
    
    def ping(self, module_name=None):
        """
        Update the watchdog timer for a module, or check simulator health.
        
        @param str module_name: Optional name of module to update watchdog
        @return bool: True if simulator is healthy
        """
        with self._thread_lock:
            # Update module watchdog if provided
            if module_name is not None and module_name in self.active_modules:
                self._module_watchdogs[module_name] = time.time()
            
            # Return health status
            if hasattr(self, 'nv_model') and self.nv_model is not None:
                return True
            return False
    
    # ===== Safe method wrapper =====
    
    def _safe_call(self, method_name, *args, **kwargs):
        """
        Safely call a method with standardized error handling.
        
        @param str method_name: Name of method to call
        @param *args, **kwargs: Arguments to pass to method
        @return: Result of method call or None if failed
        """
        with self._thread_lock:
            try:
                # Get the method and call it
                if not hasattr(self.nv_model, method_name):
                    self.log.error(f"Method {method_name} not found in simulator")
                    return None
                    
                method = getattr(self.nv_model, method_name)
                return method(*args, **kwargs)
            except Exception as e:
                self.log.error(f"Error calling {method_name}: {str(e)}")
                return None
    
    # ===== Core simulator access methods =====
    
    def reset_state(self):
        """Reset the NV center state."""
        with self._thread_lock:
            try:
                if hasattr(self, 'nv_model') and self.nv_model is not None:
                    self.nv_model.reset_state()
                    self.log.debug("NV state reset")
                    return True
            except Exception as e:
                self.log.error(f"Failed to reset NV state: {str(e)}")
            return False
    
    def apply_magnetic_field(self, field_vector):
        """
        Set the magnetic field vector.
        
        @param field_vector: [Bx, By, Bz] in Gauss
        @return bool: Success or failure
        """
        return self._safe_call('set_magnetic_field', field_vector)
    
    def apply_laser(self, power, on=True):
        """
        Control the laser for optical excitation.
        
        @param power: Laser power in normalized units (0.0-1.0)
        @param on: Bool whether laser is on/off
        @return bool: Success or failure
        """
        with self._thread_lock:
            try:
                self.nv_model.apply_laser(power, on)
                self.log.debug(f"Laser {'on' if on else 'off'} with power {power}")
                return True
            except Exception as e:
                self.log.error(f"Failed to apply laser: {str(e)}")
                return False
    
    def apply_microwave(self, frequency, power_dbm, on=True):
        """
        Control the microwave excitation.
        
        @param frequency: Microwave frequency in Hz
        @param power_dbm: Microwave power in dBm
        @param on: Bool whether microwave is on/off
        @return bool: Success or failure
        """
        with self._thread_lock:
            try:
                self.nv_model.apply_microwave(frequency, power_dbm, on)
                self.log.debug(f"Microwave {'on' if on else 'off'} at {frequency/1e6:.3f} MHz, {power_dbm} dBm")
                return True
            except Exception as e:
                self.log.error(f"Failed to apply microwave: {str(e)}")
                return False
    
    def get_fluorescence(self):
        """
        Get the current fluorescence signal.
        
        @return float: Fluorescence count rate in counts/s
        """
        with self._thread_lock:
            try:
                value = self.nv_model.get_fluorescence()
                return value
            except Exception as e:
                self.log.error(f"Failed to get fluorescence: {str(e)}")
                return 1e5  # Default fluorescence level
    
    def evolve(self, duration):
        """
        Evolve the quantum state for specified duration.
        
        @param duration: Time to evolve in seconds
        @return bool: Success or failure
        """
        return self._safe_call('evolve', duration)
    
    # ===== Fast Counter Interface Methods =====
    
    def generate_time_trace(self, bin_width_s, record_length_s, number_of_gates=0):
        """
        Generate a time-resolved fluorescence trace.
        
        @param bin_width_s: Bin width in seconds
        @param record_length_s: Total record length in seconds
        @param number_of_gates: Number of gates (0 for ungated mode)
        
        @return: Simulated time trace data
        """
        with self._thread_lock:
            try:
                # Calculate number of bins
                num_bins = int(record_length_s / bin_width_s)
                
                # Check if we're in gated mode
                if number_of_gates > 0:
                    # Gated mode (2D array)
                    trace = np.zeros((number_of_gates, num_bins))
                    
                    # For each gate, generate a fluorescence trace
                    for gate in range(number_of_gates):
                        # Get the current NV state fluorescence
                        fluorescence_level = self.get_fluorescence()
                        
                        # Generate decay pattern
                        time_bins = np.arange(num_bins) * bin_width_s
                        decay_trace = fluorescence_level * np.exp(-time_bins / 12e-9)  # 12 ns decay time
                        
                        # Add Poisson noise
                        background = 0.001 * np.ones(num_bins)
                        mean_counts = (decay_trace + background) * bin_width_s * 0.1  # 10% detection efficiency
                        noisy_trace = np.random.poisson(mean_counts)
                        
                        # Store in the trace array
                        trace[gate, :] = noisy_trace
                else:
                    # Ungated mode (1D array)
                    # Get fluorescence level
                    fluorescence_level = self.get_fluorescence()
                    
                    # Generate decay pattern
                    time_bins = np.arange(num_bins) * bin_width_s
                    decay_trace = fluorescence_level * np.exp(-time_bins / 12e-9)
                    
                    # Add Poisson noise
                    background = 0.001 * np.ones(num_bins)
                    mean_counts = (decay_trace + background) * bin_width_s * 0.1
                    trace = np.random.poisson(mean_counts)
                    
                return trace
                
            except Exception as e:
                self.log.error(f"Failed to generate time trace: {str(e)}")
                # Return zeros as fallback
                if number_of_gates > 0:
                    return np.zeros((number_of_gates, num_bins))
                else:
                    return np.zeros(num_bins)
    
    # ===== Microwave Interface Methods =====
    
    def simulate_odmr(self, f_min, f_max, n_points, mw_power=-10.0):
        """
        Simulate an ODMR experiment.
        
        @param f_min: Start frequency in Hz
        @param f_max: End frequency in Hz
        @param n_points: Number of frequency points
        @param mw_power: Microwave power in dBm
        
        @return: Dictionary with frequencies and signal
        """
        with self._thread_lock:
            try:
                # Use the simulator's ODMR simulation
                result = self.nv_model.simulate_odmr(f_min, f_max, n_points, mw_power)
                return {
                    'frequencies': result.frequencies,
                    'signal': result.signal
                }
            except Exception as e:
                self.log.error(f"Failed to simulate ODMR: {str(e)}")
                
                # Fallback to synthetic data
                frequencies = np.linspace(f_min, f_max, n_points)
                center = 2.87e9  # Zero-field splitting
                width = 5e6  # 5 MHz linewidth
                depth = 0.3  # 30% contrast
                signal = np.ones(n_points)
                signal -= depth * width**2 / ((frequencies - center)**2 + width**2)
                signal *= 1e5  # Scale to counts/s
                signal += np.random.normal(0, 0.01 * 1e5, n_points)  # Add noise
                
                return {
                    'frequencies': frequencies,
                    'signal': signal
                }
    
    def simulate_rabi(self, t_max, n_points, mw_power=0.0, mw_frequency=None):
        """
        Simulate a Rabi oscillation experiment.
        
        @param t_max: Maximum time in seconds
        @param n_points: Number of time points
        @param mw_power: Microwave power in dBm
        @param mw_frequency: Microwave frequency in Hz (or None for resonance)
        
        @return: Dictionary with times and signal
        """
        with self._thread_lock:
            try:
                # Use the simulator's Rabi simulation
                result = self.nv_model.simulate_rabi(t_max, n_points, mw_power, mw_frequency)
                return {
                    'times': result.times,
                    'signal': result.signal
                }
            except Exception as e:
                self.log.error(f"Failed to simulate Rabi: {str(e)}")
                
                # Fallback to synthetic data
                times = np.linspace(0, t_max, n_points)
                # Approx 5 MHz Rabi frequency
                signal = 1 - 0.3 * np.sin(2 * np.pi * 5e6 * times)**2 * np.exp(-times/5e-6)
                signal *= 1e5  # Scale to counts/s
                signal += np.random.normal(0, 0.01 * 1e5, n_points)  # Add noise
                
                return {
                    'times': times,
                    'signal': signal
                }
    
    # ===== Scanning Probe Interface Methods =====
    
    def set_position(self, x, y, z):
        """
        Set the position for scanning probe.
        
        @param x: X position in meters
        @param y: Y position in meters
        @param z: Z position in meters
        @return bool: Success or failure
        """
        with self._thread_lock:
            try:
                if self.confocal_simulator is not None:
                    self.confocal_simulator.set_position(x, y, z)
                    return True
                return False
            except Exception as e:
                self.log.error(f"Failed to set position: {str(e)}")
                return False
    
    def get_confocal_image(self, x_range, y_range, z_position, resolution):
        """
        Generate a confocal image based on current settings.
        
        @param x_range: (min, max) for x axis in meters
        @param y_range: (min, max) for y axis in meters
        @param z_position: Z position in meters
        @param resolution: Number of pixels per dimension
        
        @return: 2D array of confocal image data
        """
        with self._thread_lock:
            try:
                if self.confocal_simulator is not None:
                    # Use the actual confocal simulator
                    x_vals = np.linspace(x_range[0], x_range[1], resolution)
                    y_vals = np.linspace(y_range[0], y_range[1], resolution)
                    
                    # Create a grid and scan it
                    image = np.zeros((resolution, resolution))
                    
                    for i, y in enumerate(y_vals):
                        for j, x in enumerate(x_vals):
                            self.confocal_simulator.set_position(x, y, z_position)
                            image[i, j] = self.confocal_simulator.get_intensity()
                    
                    return image
                else:
                    # Generate synthetic data without confocal simulator
                    return self._generate_synthetic_confocal_image(x_range, y_range, resolution)
            except Exception as e:
                self.log.error(f"Failed to generate confocal image: {str(e)}")
                # Return empty image as fallback
                return self._generate_synthetic_confocal_image(x_range, y_range, resolution)
    
    def _generate_synthetic_confocal_image(self, x_range, y_range, resolution):
        """Generate a synthetic confocal image with random spots as fallback."""
        x_vals = np.linspace(x_range[0], x_range[1], resolution)
        y_vals = np.linspace(y_range[0], y_range[1], resolution)
        
        # Create a grid
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        
        # Generate some random Gaussian spots
        num_spots = 10
        image = np.zeros((resolution, resolution))
        
        for _ in range(num_spots):
            # Random position
            x_pos = np.random.uniform(x_range[0], x_range[1])
            y_pos = np.random.uniform(y_range[0], y_range[1])
            
            # Random intensity and size
            intensity = np.random.uniform(0.5, 1.0)
            sigma = np.random.uniform(0.5e-6, 2e-6)
            
            # Add Gaussian spot
            image += intensity * np.exp(-((x_grid - x_pos)**2 + (y_grid - y_pos)**2) / (2 * sigma**2))
        
        # Add noise
        image += np.random.normal(0, 0.05, (resolution, resolution))
        image = np.clip(image, 0, None)
        
        return image