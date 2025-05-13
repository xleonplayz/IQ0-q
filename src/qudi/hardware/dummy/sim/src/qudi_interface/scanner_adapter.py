# -*- coding: utf-8 -*-

"""
Qudi hardware interface adapter for NV simulator fluorescence scanning.

Copyright (c) 2023
"""

import numpy as np
import time
import threading
from typing import Dict, List, Optional, Tuple, Union

from qudi.interface.finite_sampling_input_interface import FiniteSamplingInputInterface, FiniteSamplingInputConstraints


class NVSimulatorScanner(FiniteSamplingInputInterface):
    """
    Hardware adapter that implements the FiniteSamplingInputInterface for the NV center simulator.
    This interface provides fluorescence data acquisition for ODMR and other experiments.
    """

    def __init__(self, nv_simulator, microwave_adapter, name='nvscan'):
        """
        Initialize the scanner adapter for the NV simulator.
        
        @param nv_simulator: Instance of the PhysicalNVModel simulator
        @param microwave_adapter: The microwave interface adapter for synchronization
        @param name: Unique name for this hardware module
        """
        # Initialize the module base class
        super().__init__(name=name)
        
        # Store references to simulator and microwave adapter
        self._simulator = nv_simulator
        self._microwave = microwave_adapter
        
        # Define channel configuration with units
        self._channel_units = {'default': 'counts/s'}
        
        # Create constraints
        self._constraints = FiniteSamplingInputConstraints(
            channel_units=self._channel_units,
            frame_size_limits=(1, 1000),
            sample_rate_limits=(0.1, 1000)
        )
        
        # Initialize operational settings
        self._active_channels = frozenset(('default',))  # Default is the only channel
        self._sample_rate = 100.0  # Hz
        self._frame_size = 100  # Samples per frame
        
        # Acquisition state variables
        self._buffer = []  # Buffer to store samples
        self._buffer_lock = threading.RLock()
        self._acquisition_thread = None
        self._stop_acquisition = threading.Event()
        
        # Thread lock for thread safety
        self._thread_lock = self.module_state.lock_access()

    @property
    def constraints(self):
        """
        Constraints of the sampling input as specified in the FiniteSamplingInputConstraints class.
        """
        return self._constraints

    @property
    def active_channels(self):
        """
        Names of all currently active channels.
        
        @return frozenset: The active channel name strings as set
        """
        return self._active_channels

    @property
    def sample_rate(self):
        """
        The sample rate (in Hz) at which the samples will be acquired.
        
        @return float: The current sample rate in Hz
        """
        return self._sample_rate

    @property
    def frame_size(self):
        """
        Currently set number of samples per channel to acquire for each data frame.
        
        @return int: Number of samples per frame
        """
        return self._frame_size

    @property
    def samples_in_buffer(self):
        """
        Currently available samples per channel being held in the input buffer.
        
        @return int: Number of unread samples per channel
        """
        with self._buffer_lock:
            return len(self._buffer)

    def on_activate(self):
        """
        Called when module is activated
        """
        self.log.info('NV Simulator scanner interface activated')
        # No specific activation needed, buffer is already initialized

    def on_deactivate(self):
        """
        Called when module is deactivated
        """
        # Stop any running acquisition
        self.stop_buffered_acquisition()
        self.log.info('NV Simulator scanner interface deactivated')

    def set_sample_rate(self, rate):
        """
        Will set the sample rate to a new value.
        
        @param float rate: The sample rate to set
        """
        if not self.constraints.sample_rate_in_range(rate):
            self.log.error(f'Sample rate {rate} Hz out of allowed range {self.constraints.sample_rate_limits}')
            return
            
        with self._thread_lock:
            if self.module_state() != 'idle':
                self.log.error('Cannot set sample rate while acquisition is running')
                return
                
            self._sample_rate = float(rate)
            self.log.debug(f'Sample rate set to {self._sample_rate} Hz')

    def set_active_channels(self, channels):
        """
        Will set the currently active channels. All other channels will be deactivated.
        
        @param iterable(str) channels: Iterable of channel names to set active.
        """
        # Validate channel names
        valid_channels = set(self._channel_units.keys())
        channels_set = set(channels)
        
        if not channels_set.issubset(valid_channels):
            invalid_channels = channels_set - valid_channels
            self.log.error(f'Cannot set invalid channels: {invalid_channels}')
            return
            
        with self._thread_lock:
            if self.module_state() != 'idle':
                self.log.error('Cannot set active channels while acquisition is running')
                return
                
            self._active_channels = frozenset(channels_set)
            self.log.debug(f'Active channels set to: {self._active_channels}')

    def set_frame_size(self, size):
        """
        Will set the number of samples per channel to acquire within one frame.
        
        @param int size: The frame size to set
        """
        size = int(size)
        if not self.constraints.frame_size_in_range(size):
            self.log.error(f'Frame size {size} out of allowed range {self.constraints.frame_size_limits}')
            return
            
        with self._thread_lock:
            if self.module_state() != 'idle':
                self.log.error('Cannot set frame size while acquisition is running')
                return
                
            self._frame_size = size
            self.log.debug(f'Frame size set to {self._frame_size}')

    def start_buffered_acquisition(self):
        """
        Will start the acquisition of a data frame in a non-blocking way.
        """
        with self._thread_lock:
            if self.module_state() != 'idle':
                self.log.error('Cannot start acquisition. Device already running.')
                raise RuntimeError('Cannot start acquisition. Device already running.')
                
            # Clear buffer
            with self._buffer_lock:
                self._buffer = []
                
            # Reset stop flag
            self._stop_acquisition.clear()
            
            # Create and start acquisition thread
            self._acquisition_thread = threading.Thread(
                target=self._acquisition_loop, 
                args=()
            )
            self._acquisition_thread.daemon = True
            self._acquisition_thread.start()
            
            # Set module state to running
            self.module_state.lock()
            self.log.debug('Started buffered acquisition')

    def _cleanup_resources(self):
        """
        Clean up resources to prevent memory leaks.
        """
        # Release large arrays
        self._scan_data = None
        
        # Clear buffer
        with self._buffer_lock:
            self._buffer = []
        
        # Force garbage collection
        import gc
        gc.collect()
        
        self.log.debug('Resources cleaned up')
    
    def stop_buffered_acquisition(self):
        """
        Will abort the currently running data frame acquisition and clean up resources.
        """
        with self._thread_lock:
            if self.module_state() == 'idle':
                return
                
            # Set stop flag
            self._stop_acquisition.set()
            
            # Wait for acquisition thread to finish (with timeout)
            if self._acquisition_thread is not None and self._acquisition_thread.is_alive():
                self._acquisition_thread.join(timeout=1.0)
                
            # Clean up resources
            self._cleanup_resources()
            
            # Set module to idle
            self.module_state.unlock()
            self.log.debug('Stopped buffered acquisition')

    def get_buffered_samples(self, number_of_samples=None):
        """
        Returns a chunk of the current data frame for all active channels read from the frame buffer.
        
        @param int number_of_samples: optional, the number of samples to read from buffer
        
        @return dict: Sample arrays (values) for each active channel (keys)
        """
        with self._buffer_lock:
            # Determine how many samples to get
            if number_of_samples is None:
                number_of_samples = len(self._buffer)
            else:
                # Check if we have enough samples or can get them
                samples_pending = self._frame_size - len(self._buffer)
                if number_of_samples > (len(self._buffer) + samples_pending) and self.module_state() == 'running':
                    self.log.error(f'Requested {number_of_samples} samples but only {len(self._buffer) + samples_pending} '
                                   f'samples are available in this frame')
                    raise ValueError(f'Requested {number_of_samples} samples but only {len(self._buffer) + samples_pending} '
                                    f'samples are available in this frame')
                    
            # Wait for samples if needed
            while len(self._buffer) < number_of_samples and self.module_state() == 'running':
                # Release lock during waiting
                self._buffer_lock.release()
                time.sleep(0.01)  # Short sleep to avoid busy waiting
                self._buffer_lock.acquire()
                
            # Get samples from buffer
            if len(self._buffer) < number_of_samples:
                number_of_samples = len(self._buffer)
                
            # Prepare result dictionary with samples for each active channel
            result = {channel: np.array([]) for channel in self._active_channels}
            if number_of_samples > 0:
                samples = self._buffer[:number_of_samples]
                self._buffer = self._buffer[number_of_samples:]
                
                # Convert list of dict samples to dict of arrays
                for channel in self._active_channels:
                    if channel in samples[0]:  # Ensure channel exists in samples
                        result[channel] = np.array([sample[channel] for sample in samples])
                        
            return result

    def acquire_frame(self, frame_size=None):
        """
        Acquire a single data frame for all active channels.
        This method call is blocking until the entire data frame has been acquired.
        
        @param int frame_size: optional, the number of samples to acquire in this frame
        
        @return dict: Sample arrays (values) for each active channel (keys)
        """
        # Remember original frame size to restore it later if needed
        original_frame_size = self._frame_size
        
        try:
            # Set temporary frame size if provided
            if frame_size is not None:
                self.set_frame_size(frame_size)
                
            # Start acquisition
            self.start_buffered_acquisition()
            
            # Wait for all samples or acquisition to stop
            start_time = time.time()
            timeout = (self._frame_size / self._sample_rate) * 2.0  # Twice the expected acquisition time
            
            while self.samples_in_buffer < self._frame_size and self.module_state() == 'running':
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    self.log.warning(f'Acquisition timeout after {elapsed:.2f}s')
                    break
                time.sleep(0.01)  # Short sleep to avoid busy waiting
                
            # Get all available samples
            samples = self.get_buffered_samples()
            
            # Stop acquisition
            self.stop_buffered_acquisition()
            
            return samples
            
        finally:
            # Restore original frame size if needed
            if frame_size is not None and frame_size != original_frame_size:
                self._frame_size = original_frame_size

    def _acquisition_loop(self):
        """
        Thread target for continuous data acquisition with improved thread safety.
        """
        try:
            # Calculate time between samples
            sample_period = 1.0 / self._sample_rate
            next_sample_time = time.time()
            
            # Acquire samples until stopped or frame complete
            sample_count = 0
            
            while not self._stop_acquisition.is_set() and sample_count < self._frame_size:
                # Wait until next sample time
                current_time = time.time()
                if current_time < next_sample_time:
                    # Sleep until next sample time
                    time.sleep(max(0, next_sample_time - current_time))
                
                try:
                    # Get sample from simulator - use thread lock for simulator access
                    with self._thread_lock:
                        sample = self._acquire_sample()
                    
                    # Add to buffer with proper locking
                    with self._buffer_lock:
                        self._buffer.append(sample)
                    
                    # Update counters and timing
                    sample_count += 1
                    next_sample_time += sample_period
                    
                    # If microwave is scanning, step to next frequency with proper locking
                    if hasattr(self._microwave, 'is_scanning') and self._microwave.is_scanning:
                        # Use a try block to handle potential exceptions
                        try:
                            self._microwave.scan_next()
                        except Exception as mw_error:
                            self.log.warning(f"Error stepping microwave frequency: {str(mw_error)}")
                except Exception as sample_error:
                    self.log.warning(f"Error acquiring sample: {str(sample_error)}")
                    # Continue with next sample rather than terminating the loop
                    next_sample_time += sample_period
                    
        except Exception as e:
            self.log.error(f'Error in acquisition loop: {str(e)}', exc_info=True)
        finally:
            # Set is_running flag to false on completion (even if exception occurred)
            with self._buffer_lock:
                self._is_running = False
                
            if not self._stop_acquisition.is_set():
                # Acquisition completed normally
                self.log.debug(f'Acquisition completed: {sample_count} samples acquired')
                
    def _acquire_sample(self):
        """
        Acquire a single sample from the simulator with physical shot noise.
        
        @return dict: Sample data for each channel
        """
        # Get current fluorescence rate from simulator (counts per second)
        count_rate = self._simulator.get_fluorescence()
        
        # Calculate collection duration based on sample rate
        collection_time = 1.0 / self._sample_rate  # seconds
        
        # Calculate expected number of photons during this collection window
        expected_counts = count_rate * collection_time
        
        # Generate actual photon counts using Poisson distribution (photon shot noise)
        actual_counts = np.random.poisson(expected_counts)
        
        # Convert back to counts per second
        fluorescence = actual_counts / collection_time
        
        # Add detector noise (e.g. dark counts, readout noise)
        # APD dark count rate is typically 100-500 counts/sec
        dark_count_rate = self._config.get('dark_count_rate', 200)  # counts/sec
        dark_counts = np.random.poisson(dark_count_rate * collection_time)
        
        # Add electronic noise (typically small for photon counting, more relevant for analog detectors)
        electronic_noise_std = self._config.get('electronic_noise', 10)  # counts/sec
        electronic_noise = np.random.normal(0, electronic_noise_std)
        
        # Calculate final measured count rate with noise
        fluorescence = (actual_counts + dark_counts) / collection_time + electronic_noise
        
        # Ensure non-negative value
        fluorescence = max(0, fluorescence)
        
        # Create sample dictionary
        sample = {'default': fluorescence}
        
        return sample