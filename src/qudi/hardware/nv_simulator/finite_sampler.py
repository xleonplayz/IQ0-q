# -*- coding: utf-8 -*-

"""
This file contains the Qudi hardware module for the NV simulator finite sampler.

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
import numpy as np
import os
import sys
from typing import Tuple, List, Dict, Optional, Union

from qudi.interface.finite_sampling_input_interface import FiniteSamplingInputInterface, FiniteSamplingInputConstraints
from qudi.util.mutex import Mutex
from qudi.core.configoption import ConfigOption
from qudi.core.connector import Connector

# Import QudiFacade directly from current directory to avoid circular imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from qudi_facade import QudiFacade


class NVSimFiniteSampler(FiniteSamplingInputInterface):
    """A qudi hardware module that simulates finite sampling input for the NV simulator.
    
    This module is used primarily for ODMR measurements, where it samples the NV fluorescence
    signal as a function of microwave frequency.
    
    Example config for copy-paste:
    
    nv_sim_finite_sampler:
        module.Class: 'nv_simulator.finite_sampler.NVSimFiniteSampler'
        options:
            simulation_mode: 'ODMR'
            sample_rate_limits: [1, 1e6]
            frame_size_limits: [1, 1e8]
            channel_units:
                'APD counts': 'c/s'
                'Photodiode': 'V'
    """
    
    # Configuration options
    _simulation_mode = ConfigOption('simulation_mode', default='ODMR', missing='warn')
    _sample_rate_limits = ConfigOption('sample_rate_limits', default=[1, 1e6], missing='warn')
    _frame_size_limits = ConfigOption('frame_size_limits', default=[1, 1e8], missing='warn')
    _channel_units = ConfigOption('channel_units', default={'APD counts': 'c/s', 'Photodiode': 'V'}, missing='warn')
    
    # Connectors
    simulator = Connector(interface='MicrowaveInterface')  # Connect to QudiFacade which implements MicrowaveInterface
    
    def __init__(self, qudi_main_weakref=None, name=None, *args, **kwargs):
        super().__init__(qudi_main_weakref=qudi_main_weakref, name=name, *args, **kwargs)
        
        self._thread_lock = Mutex()
        self._constraints = None
        
        # Internal state variables
        self._active_channels = ['APD counts']
        self._current_sample_rate = 1000.0  # Hz
        self._current_frame_size = 1000
        self._is_running = False
        self._data_buffer = None
        
    def on_activate(self):
        """Initialization performed during activation of the module."""
        # Define hardware constraints
        self._constraints = FiniteSamplingInputConstraints(
            channel_units=self._channel_units,
            frame_size_limits=self._frame_size_limits,
            sample_rate_limits=self._sample_rate_limits
        )
        
        try:
            # Get QudiFacade from connector
            self.log.info('Trying to get QudiFacade from connector...')
            self._qudi_facade = self.simulator()
            self.log.info('Successfully retrieved QudiFacade from connector')
            
            # Reset the NV simulator state
            self._qudi_facade.reset()
            
            # Initialize data buffer
            self._data_buffer = np.zeros((len(self._active_channels), self._current_frame_size))
            
            self.log.info('NV Simulator Finite Sampler initialized successfully')
        except Exception as e:
            self.log.error(f"Error activating NVSimFiniteSampler: {str(e)}")
            # Try direct import as fallback
            try:
                from qudi_facade import QudiFacade
                self._qudi_facade = QudiFacade()
                self.log.info('Using direct QudiFacade instantiation as fallback')
                
                # Initialize data buffer
                self._data_buffer = np.zeros((len(self._active_channels), self._current_frame_size))
                
                self.log.info('NV Simulator Finite Sampler initialized with fallback')
            except Exception as e2:
                self.log.error(f"Fallback also failed: {str(e2)}")
                raise
        
    def on_deactivate(self):
        """Cleanup performed during deactivation of the module."""
        # Stop sampling if running
        if self._is_running:
            try:
                self.stop_buffering()
            except:
                self.log.exception("Failed to properly stop buffering")
        
    @property
    def constraints(self):
        """Finite sampling constraints for the device.
        
        @return FiniteSamplingInputConstraints: input sampling constraints
        """
        return self._constraints
    
    @property
    def sample_rate(self):
        """The currently set sample rate with which new samples are acquired from the hardware.
        This is a property instead of a method in order to keep compatiblity with the 
        slow counter interface.

        @return float: sample rate in Hz
        """
        with self._thread_lock:
            return self._current_sample_rate
    
    @property
    def frame_size(self):
        """The currently set frame size, i.e. the number of samples to acquire per channel.
        This is a property instead of a method in order to keep compatiblity with the 
        slow counter interface.

        @return int: Number of samples per frame
        """
        with self._thread_lock:
            return self._current_frame_size
    
    @property
    def active_channels(self):
        """Names of active input channels. 
        This is a property instead of a method in order to keep compatiblity with the 
        slow counter interface.

        @return list(str): List of channel names corresponding to the FiniteSamplingInputConstraints.channels property.
        """
        with self._thread_lock:
            return self._active_channels.copy()
    
    def configure(self, active_channels, sample_rate, frame_size):
        """
        Configure finite sampling input parameters.
        
        @param list(str) active_channels: Channel names to use for acquisition
        @param float sample_rate: Sampling rate in Hz
        @param int frame_size: Number of samples to acquire per channel
        
        @return dict: Actual set configuration
        """
        with self._thread_lock:
            # Check if sampling is running
            if self._is_running:
                raise RuntimeError("Unable to configure while sampling is running. Call stop_buffering() first.")
                
            # Check parameters for sanity
            self._constraints.test_configuration(active_channels, sample_rate, frame_size)
            
            # Set new configuration
            self._active_channels = active_channels.copy()
            self._current_sample_rate = sample_rate
            self._current_frame_size = frame_size
            
            # Update data buffer
            self._data_buffer = None
            
            self.log.debug(f"Configured finite sampler with {active_channels}, "
                          f"rate {sample_rate:.1e} Hz, frame size {frame_size}")
            
            # Return actual set configuration
            return {
                'active_channels': self._active_channels,
                'sample_rate': self._current_sample_rate,
                'frame_size': self._current_frame_size
            }
            
    def start_buffering(self):
        """
        Start acquiring samples and buffer them or directly write them to file.
        
        Must raise exception if no channels are configured or if start fails.
        
        @return bool: Whether the start was successful
        """
        with self._thread_lock:
            # Check if already running
            if self._is_running:
                self.log.warning("Finite sampler is already running")
                return True
                
            # Check if channels are configured
            if not self._active_channels:
                raise RuntimeError("No active channels configured")
                
            # Initialize data buffer
            self._data_buffer = np.zeros((len(self._active_channels), self._current_frame_size), dtype=np.float64)
            
            # Setup simulator for acquisition
            if self._simulation_mode == 'ODMR':
                # Get current microwave parameters from the facade for simulation
                self.log.debug("Setting up NV simulator for ODMR sampling")
                
            # Mark as running
            self._is_running = True
            
            self.log.debug("Started buffering")
            return True
            
    def stop_buffering(self):
        """
        Stop acquiring samples.
        
        Must return after hardware stopped recording.
        
        @return bool: Whether the stop was successful
        """
        with self._thread_lock:
            # Check if running
            if not self._is_running:
                self.log.warning("Finite sampler is not running")
                return True
                
            # Mark as not running
            self._is_running = False
            
            self.log.debug("Stopped buffering")
            return True
            
    def set_active_channels(self, channels):
        """Set the active channels for acquisition.

        @param list(str) channels: List of channel names
        """
        with self._thread_lock:
            if self._is_running:
                self.log.error("Cannot set active channels while acquisition is running")
                return False
            
            # Check if all channels are valid
            for channel in channels:
                if channel not in self._channel_units:
                    self.log.error(f"Invalid channel: {channel}")
                    return False
                    
            self._active_channels = channels.copy()
            return True
            
    def set_sample_rate(self, rate):
        """Set the sample rate for acquisition.

        @param float rate: Sample rate in Hz
        """
        with self._thread_lock:
            if self._is_running:
                self.log.error("Cannot set sample rate while acquisition is running")
                return False
                
            if not self._sample_rate_limits[0] <= rate <= self._sample_rate_limits[1]:
                self.log.error(f"Sample rate {rate} out of range: {self._sample_rate_limits}")
                return False
                
            self._current_sample_rate = rate
            return True
            
    def set_frame_size(self, size):
        """Set the number of samples to acquire per channel.

        @param int frame_size: Number of samples
        """
        with self._thread_lock:
            if self._is_running:
                self.log.error("Cannot set frame size while acquisition is running")
                return False
                
            if not self._frame_size_limits[0] <= size <= self._frame_size_limits[1]:
                self.log.error(f"Frame size {size} out of range: {self._frame_size_limits}")
                return False
                
            self._current_frame_size = size
            return True
            
    def samples_in_buffer(self):
        """Get the number of samples currently in the buffer.
        This is an abstract method required by the interface.

        @return int: Number of samples
        """
        with self._thread_lock:
            return self._current_frame_size if self._is_running else 0
            
    def get_samples_in_buffer(self):
        """Get the number of samples currently in the buffer.
        Legacy method for compatibility.

        @return int: Number of samples
        """
        return self.samples_in_buffer()
            
    def start_buffered_acquisition(self):
        """Start buffered acquisition.

        @return bool: Success
        """
        with self._thread_lock:
            if self._is_running:
                self.log.warning("Acquisition already running")
                return True
                
            # Create data buffer
            self._data_buffer = np.zeros((len(self._active_channels), self._current_frame_size))
            self._is_running = True
            return True
            
    def stop_buffered_acquisition(self):
        """Stop buffered acquisition.

        @return bool: Success
        """
        with self._thread_lock:
            if not self._is_running:
                self.log.warning("Acquisition not running")
                return True
                
            self._is_running = False
            return True
            
    def acquire_frame(self):
        """Acquire a single frame.

        This is a blocking call that returns when the frame is acquired.

        @return dict: Dictionary with keys being the channel names and values being numpy.ndarrays
                      of shape (frame_size,).
        """
        with self._thread_lock:
            # Start buffered acquisition if not already running
            if not self._is_running:
                self.start_buffered_acquisition()
                
            # Simulate acquisition by just getting the buffered data
            data = self.get_buffered_data()
            
            # Stop acquisition
            self._is_running = False
            
            return data
            
    def get_buffered_samples(self):
        """
        Get the most recently acquired samples.
        This is an abstract method required by the interface.
        
        @return dict: Dictionary with keys being the channel names and values being numpy.ndarrays
                      of shape (frame_size,).
        """
        return self.get_buffered_data()
        
    def get_buffered_data(self):
        """
        Return most recently acquired data.
        
        For NV simulator, we simulate the data based on the current microwave parameters.
        
        @return dict: Dictionary with keys being the channel names and values being numpy.ndarrays
                      of shape (frame_size,).
        """
        with self._thread_lock:
            # Check if running
            if not self._is_running:
                raise RuntimeError("Finite sampler is not running")
                
            # Get the current frequency from the shared state - This is key for ODMR synchronization
            current_freq = self._qudi_facade.get_current_frequency()
            current_power = self._qudi_facade.get_current_power()
            is_mw_on = self._qudi_facade.is_microwave_on()
            is_scanning = self._qudi_facade.is_scanning()
            scan_index = self._qudi_facade.get_current_scan_index()
            
            # Add extensive debug logging
            self.log.info(f"[SAMPLE DEBUG] ODMR sampling at frequency: {current_freq/1e9:.6f} GHz, power: {current_power} dBm")
            self.log.info(f"[SAMPLE DEBUG] MW on: {is_mw_on}, Scanning: {is_scanning}, Scan index: {scan_index}")
            
            # If microwave is off, return baseline signal with noise
            if not is_mw_on:
                self.log.warning("[SAMPLE DEBUG] Microwave is off, returning baseline signal only")
                baseline = 100000.0  # 100k counts/s
                result = {}
                
                for i, channel in enumerate(self._active_channels):
                    # Create baseline signal with 2% noise
                    noise = np.random.normal(0, 0.02 * baseline, self._current_frame_size)
                    self._data_buffer[i, :] = baseline + noise
                    result[channel] = self._data_buffer[i, :].copy()
                
                return result
            
            # Generate simulated fluorescence for ODMR
            # For simplicity, we use a Lorentzian dip for the resonance
            resonance_freq = 2.87e9  # Zero-field splitting (Hz)
            
            # Access b_field attribute from model (in Tesla)
            b_field = self._qudi_facade.nv_model.b_field
            
            # Convert Tesla to Gauss (1 T = 10,000 G)
            field_strength_gauss = np.linalg.norm(b_field) * 10000.0
            
            # Zeeman splitting (~2.8 MHz/G)
            zeeman_shift = 2.8e6 * field_strength_gauss  # field in G, shift in Hz
            
            # Log the current frequency and computed Zeeman shift for debugging
            self.log.info(f"[SAMPLE DEBUG] Current MW frequency: {current_freq/1e9:.6f} GHz")
            self.log.info(f"[SAMPLE DEBUG] Magnetic field: {field_strength_gauss:.2f} G, Zeeman shift: {zeeman_shift/1e6:.2f} MHz")
            
            # Create two dips for the ms=±1 states
            dip1_center = resonance_freq - zeeman_shift  # ms=0 to ms=-1 transition
            dip2_center = resonance_freq + zeeman_shift  # ms=0 to ms=+1 transition
            
            self.log.info(f"[SAMPLE DEBUG] Resonances at: {dip1_center/1e9:.6f} GHz and {dip2_center/1e9:.6f} GHz")
            
            linewidth = 20e6  # 20 MHz linewidth (realistic for ODMR in diamond)
            contrast = 0.3  # 30% contrast (typical for NV centers)
            baseline = 1.0
            
            # Compare current frequency to resonances
            close_to_resonance = False
            if abs(current_freq - dip1_center) < 100e6 or abs(current_freq - dip2_center) < 100e6:
                self.log.info(f"[SAMPLE DEBUG] Close to resonance at {current_freq/1e9:.6f} GHz")
                close_to_resonance = True
            
            # Lorentzian function for each dip
            dip1 = contrast * linewidth**2 / ((current_freq - dip1_center)**2 + linewidth**2)
            dip2 = contrast * linewidth**2 / ((current_freq - dip2_center)**2 + linewidth**2)
            
            # Calculate and log resonance depths at current frequency
            dip_strength = dip1 + dip2  # Combined dip strength
            self.log.info(f"[SAMPLE DEBUG] Dip1 value: {dip1:.4f}, Dip2 value: {dip2:.4f} at freq: {current_freq/1e9:.6f} GHz")
            
            # Power scaling - stronger MW power means deeper dips
            # Convert dBm to mW for power scaling
            power_mw = 10**(current_power/10) 
            power_factor = min(1.0, power_mw / 10.0)  # Scale up to a maximum of 1.0
            
            # Apply power scaling to contrast
            scaled_dip1 = dip1 * power_factor
            scaled_dip2 = dip2 * power_factor
            
            # Combine dips and scale to photon counts (typical rates for NV)
            signal = (baseline - scaled_dip1 - scaled_dip2) * 100000.0  # ~100k counts/s
            
            # Add some noise (Poisson noise)
            sampling_time = 1.0 / self._current_sample_rate
            expected_counts = signal * sampling_time
            noise = np.random.poisson(expected_counts, self._current_frame_size)
            simulated_signal = noise / sampling_time
            
            # For debug purposes
            if close_to_resonance:
                self.log.info(f"[SAMPLE DEBUG] Signal at resonance: {np.mean(simulated_signal):.1f} counts/s, contrast: {(scaled_dip1 + scaled_dip2):.4f}")
            
            # Calculate and log the signal contrast percentage
            contrast_percent = (scaled_dip1 + scaled_dip2) * 100
            self.log.info(f"[SAMPLE DEBUG] ODMR contrast at {current_freq/1e9:.6f} GHz: {contrast_percent:.2f}%")
            
            # Fill the buffer with the simulated data
            self._data_buffer[0, :] = simulated_signal
            
            # For photodiode channel, just use constant value with small noise
            if len(self._active_channels) > 1 and 'Photodiode' in self._active_channels:
                pd_idx = self._active_channels.index('Photodiode')
                pd_value = 2.5 + 0.02 * np.random.randn(self._current_frame_size)
                self._data_buffer[pd_idx, :] = pd_value
            
            # Prepare result
            result = {}
            for i, channel in enumerate(self._active_channels):
                result[channel] = self._data_buffer[i, :].copy()
                
            return result