# -*- coding: utf-8 -*-

"""
Qudi hardware interface adapter for NV simulator fast counter.
This module implements the FastCounterInterface for the NV center simulator,
enabling time-resolved measurements of NV fluorescence.

Copyright (c) 2023
"""

import numpy as np
import time
import threading
from typing import Dict, Any, List, Tuple, Union, Optional

from qudi.interface.fast_counter_interface import FastCounterInterface

from .qudi_facade import QudiFacade


class NVSimFastCounter(FastCounterInterface):
    """
    Hardware adapter that implements the FastCounterInterface for the NV center simulator.
    This interface enables time-resolved measurements for pulse experiments.
    """

    def __init__(self, config=None, **kwargs):
        """
        Initialize the fast counter adapter for the NV simulator.
        
        @param config: Configuration dictionary
        @param **kwargs: Additional keyword arguments for the base class
        """
        # Initialize the module base class
        super().__init__(config=config, **kwargs)
        
        # Get the Qudi facade instance
        self._qudi_facade = QudiFacade(config)
        self._simulator = self._qudi_facade.get_nv_model()
        
        # Parse configuration
        self._config = self.config
        
        # Initialize operational parameters
        self._bin_width_s = 1e-9  # Default bin width: 1 ns
        self._record_length_s = 1e-6  # Default record length: 1 μs
        self._number_of_gates = 0  # Default: ungated mode
        
        # Simulation parameters
        self._fluorescence_decay_time = self._config.get('fluorescence_decay_time', 12e-9)  # Default: 12 ns
        self._background_counts = self._config.get('background_counts', 0.001)  # Background counts per bin
        self._detection_efficiency = self._config.get('detection_efficiency', 0.1)  # Detection efficiency (10%)
        
        # Trace generation parameters
        self._current_trace = None
        
        # State variables
        self._acquisition_thread = None
        self._stop_acquire = threading.Event()
        self._is_running = False
        
        # Thread lock for thread safety
        self._thread_lock = self.module_state.lock_access()
        
        self.log.info("NV Simulator fast counter initialized")

    def on_activate(self):
        """
        Called when module is activated
        """
        self.log.info('NV Simulator fast counter activated')
        
        # Apply initial configuration if provided
        if 'bin_width_s' in self._config:
            self._bin_width_s = self._config['bin_width_s']
            
        if 'record_length_s' in self._config:
            self._record_length_s = self._config['record_length_s']
            
        if 'number_of_gates' in self._config:
            self._number_of_gates = self._config['number_of_gates']

    def on_deactivate(self):
        """
        Called when module is deactivated
        """
        # Stop any running acquisition
        self.stop_measure()
        self.log.info('NV Simulator fast counter deactivated')

    def get_constraints(self):
        """
        Retrieve the hardware constraints of the fast counter.
        
        @return dict: constraints dictionary with keys:
                      - 'hardware_binwidth_list': list of possible binwidths in seconds
                      - 'max_sweep_len': maximum sweep length in seconds
                      - 'min_sweep_len': minimum sweep length in seconds
                      - 'max_gate_len': maximum gate length in seconds
                      - 'min_gate_len': minimum gate length in seconds
                      - 'max_gates': maximum number of gates
                      - 'min_gates': minimum number of gates
        """
        constraints = dict()
        
        # Define bin width options (log scale from 0.1 ns to 10 μs)
        bin_widths = [1e-10 * 10**i for i in range(3)]  # 0.1 ns, 1 ns, 10 ns
        bin_widths.extend([1e-9 * 10**i for i in range(1, 4)])  # 10 ns, 100 ns, 1 μs
        bin_widths.extend([1e-6 * 10**i for i in range(1, 2)])  # 10 μs
        
        constraints['hardware_binwidth_list'] = bin_widths
        
        # Define sweep length constraints
        constraints['max_sweep_len'] = 1.0  # 1 second max
        constraints['min_sweep_len'] = 100e-9  # 100 ns min
        
        # Define gate constraints
        constraints['max_gates'] = 1000  # Maximum number of gates
        constraints['min_gates'] = 0  # 0 for ungated operation
        constraints['max_gate_len'] = 0.1  # 100 ms max gate length 
        constraints['min_gate_len'] = 1e-9  # 1 ns min gate length
        
        return constraints

    def configure(self, bin_width_s, record_length_s, number_of_gates=0):
        """
        Configure the fast counter with a new bin width, record length and number of gates.
        
        @param float bin_width_s: Length of a single time bin in the time trace in seconds
        @param float record_length_s: Total length of the timetrace/each single gate in seconds
        @param int number_of_gates: Number of gates in the pulse sequence (0 means just one trace)
        
        @return tuple(bin_width_s, record_length_s, number_of_gates):
                    configured values
        """
        with self._thread_lock:
            self.log.info(f"Configuring fast counter: bin_width={bin_width_s}s, "
                         f"record_length={record_length_s}s, gates={number_of_gates}")
            
            # Store configuration
            self._bin_width_s = bin_width_s
            self._record_length_s = record_length_s
            self._number_of_gates = number_of_gates
            
            # Reset trace
            self._current_trace = None
            
            return bin_width_s, record_length_s, number_of_gates

    def start_measure(self):
        """
        Start the fast counter measurement.
        """
        with self._thread_lock:
            if self._is_running:
                self.log.warning("Fast counter measurement already running")
                return
                
            # Reset stop flag and status
            self._stop_acquire.clear()
            self._is_running = True
            
            # Start acquisition thread
            self._acquisition_thread = threading.Thread(
                target=self._acquisition_loop,
                args=()
            )
            self._acquisition_thread.daemon = True
            self._acquisition_thread.start()
            
            self.log.debug("Fast counter measurement started")

    def stop_measure(self):
        """
        Stop the fast counter measurement.
        """
        with self._thread_lock:
            if not self._is_running:
                return
                
            # Set stop flag
            self._stop_acquire.set()
            
            # Wait for acquisition thread to finish (with timeout)
            if self._acquisition_thread is not None and self._acquisition_thread.is_alive():
                self._acquisition_thread.join(timeout=1.0)
                
            # Update status
            self._is_running = False
            self.log.debug("Fast counter measurement stopped")

    def pause_measure(self):
        """
        Pause the fast counter measurement.
        Not implemented for the NV simulator (stopping is the same as pausing).
        """
        self.stop_measure()

    def continue_measure(self):
        """
        Continue a paused fast counter measurement.
        Not implemented for the NV simulator (just restart the measurement).
        """
        self.start_measure()

    def is_gated(self):
        """
        Check if the fast counter is in gated mode.
        
        @return bool: True if the counter is in gated mode, False otherwise
        """
        return self._number_of_gates > 0

    def get_data_trace(self):
        """
        Get the current timetrace of the fast counter.
        
        @return numpy.ndarray: The current time trace (1D or 2D array, depending on gated mode)
        """
        with self._thread_lock:
            # If no trace exists, generate a new one
            if self._current_trace is None:
                self._generate_time_trace()
                
            return self._current_trace.copy()

    def _acquisition_loop(self):
        """
        Thread target for simulated data acquisition.
        """
        try:
            # Simply generate a trace initially
            self._generate_time_trace()
            
            # Wait until stopped
            while not self._stop_acquire.is_set():
                # In a real device we'd be continuously acquiring here
                # For the simulator, we just sleep and occasionally update the trace
                if self._stop_acquire.wait(timeout=0.2):  # 200 ms update interval
                    break
                
                # Update the trace periodically to simulate continuous acquisition
                self._generate_time_trace()
                
        except Exception as e:
            self.log.error(f"Error in acquisition loop: {str(e)}")
        finally:
            self._is_running = False

    def _generate_time_trace(self):
        """
        Generate a simulated time trace based on the NV model state.
        """
        # Calculate number of bins
        num_bins = int(self._record_length_s / self._bin_width_s)
        
        # Check if we're in gated mode
        if self._number_of_gates > 0:
            # Gated mode (2D array)
            self._current_trace = np.zeros((self._number_of_gates, num_bins))
            
            # For each gate, generate a fluorescence trace
            for gate in range(self._number_of_gates):
                # Get the current NV state (this would be more sophisticated in a real simulation)
                fluorescence_level = self._simulator.get_fluorescence()
                
                # Generate a trace with exponential decay from the fluorescence level
                time_bins = np.arange(num_bins) * self._bin_width_s
                decay_trace = fluorescence_level * np.exp(-time_bins / self._fluorescence_decay_time)
                
                # Add Poisson noise and background
                background = self._background_counts * np.ones(num_bins)
                mean_counts = (decay_trace + background) * self._bin_width_s * self._detection_efficiency
                noisy_trace = np.random.poisson(mean_counts)
                
                # Store in the trace array
                self._current_trace[gate, :] = noisy_trace
        else:
            # Ungated mode (1D array)
            # Get the current NV state
            fluorescence_level = self._simulator.get_fluorescence()
            
            # Generate a trace with exponential decay from the fluorescence level
            time_bins = np.arange(num_bins) * self._bin_width_s
            decay_trace = fluorescence_level * np.exp(-time_bins / self._fluorescence_decay_time)
            
            # Add Poisson noise and background
            background = self._background_counts * np.ones(num_bins)
            mean_counts = (decay_trace + background) * self._bin_width_s * self._detection_efficiency
            self._current_trace = np.random.poisson(mean_counts)
            
        return self._current_trace