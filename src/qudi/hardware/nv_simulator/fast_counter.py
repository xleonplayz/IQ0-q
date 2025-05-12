# -*- coding: utf-8 -*-

"""
This file contains the Qudi hardware module for the NV simulator fast counter.

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
from qudi.core.configoption import ConfigOption
from qudi.interface.fast_counter_interface import FastCounterInterface
from qudi.util.mutex import Mutex

from .qudi_facade import QudiFacade


class NVSimFastCounter(FastCounterInterface):
    """Implementation of the FastCounter interface for the NV simulator.
    
    This module simulates a time-resolved photon counter based on the NV center simulator.
    It can produce realistic fluorescence time traces for various pulse sequences.
    
    Example config for copy-paste:

    nv_sim_fastcounter:
        module.Class: 'nv_simulator.fast_counter.NVSimFastCounter'
        options:
            gated: False
            photon_rate: 100000  # Simulated maximum photon count rate in cps
            noise_factor: 0.1    # Relative amplitude of Poisson noise
            dark_counts: 200     # Dark counts per second
            time_jitter: 0.5e-9  # Timing jitter in seconds
            t1: 5.5e-6           # T1 relaxation time in seconds
            t2: 2.0e-6           # T2 coherence time in seconds
    """

    # Config options
    _gated = ConfigOption('gated', False, missing='warn')
    _photon_rate = ConfigOption('photon_rate', 100000, missing='warn')  # counts per second
    _noise_factor = ConfigOption('noise_factor', 0.1, missing='warn')
    _dark_counts = ConfigOption('dark_counts', 200, missing='warn')  # counts per second
    _time_jitter = ConfigOption('time_jitter', 0.5e-9, missing='warn')  # in seconds
    _t1 = ConfigOption('t1', 5.5e-6, missing='warn')  # in seconds
    _t2 = ConfigOption('t2', 2.0e-6, missing='warn')  # in seconds

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread_lock = Mutex()
        self._binwidth = 1
        self._gate_length_bins = 8192
        self._number_of_gates = 0
        self._count_data = None
        self._statusvar = 0
        self._start_time = 0
        self._measurement_running = False

    def on_activate(self):
        """Initialisation performed during activation of the module."""
        # Initialize the simulator facade
        self._qudi_facade = QudiFacade()
        
        # Set status to idle
        self._statusvar = 0
        self._binwidth = 1
        self._gate_length_bins = 8192
        self._count_data = None
        self._measurement_running = False
        
        self.log.info('NV Simulator Fast Counter initialized')

    def on_deactivate(self):
        """Deinitialisation performed during deactivation of the module."""
        # Set status to deactivated
        self._statusvar = -1
        self._measurement_running = False

    def get_constraints(self):
        """Retrieve the hardware constraints from the Fast counting device.

        @return dict: dict with hardware constraints
        """
        constraints = dict()
        # The unit of those entries are seconds per bin
        constraints['hardware_binwidth_list'] = [1e-9, 2e-9, 5e-9, 10e-9, 20e-9, 50e-9, 100e-9, 200e-9, 500e-9, 1e-6]
        return constraints

    def configure(self, bin_width_s, record_length_s, number_of_gates=0):
        """Configuration of the fast counter.

        @param float bin_width_s: Length of a single time bin in seconds
        @param float record_length_s: Total length of the timetrace/each single gate in seconds
        @param int number_of_gates: Number of gates in the pulse sequence (ignored for non-gated counter)

        @return tuple(binwidth_s, gate_length_s, number_of_gates):
                    binwidth_s: actual set binwidth in seconds
                    gate_length_s: actual set gate length in seconds
                    number_of_gates: number of gates accepted
        """
        with self._thread_lock:
            # Find the closest available binwidth from the constraints
            available_binwidths = self.get_constraints()['hardware_binwidth_list']
            closest_binwidth = min(available_binwidths, key=lambda x: abs(x - bin_width_s))
            
            self._binwidth = closest_binwidth
            self._gate_length_bins = int(np.rint(record_length_s / closest_binwidth))
            self._number_of_gates = number_of_gates if self._gated else 0
            
            actual_length = self._gate_length_bins * closest_binwidth
            
            # Set status to configured/idle
            self._statusvar = 1
            
            return closest_binwidth, actual_length, self._number_of_gates

    def get_status(self):
        """Receives the current status of the Fast Counter and outputs it.
        
        0 = unconfigured
        1 = idle
        2 = running
        3 = paused
        -1 = error state
        """
        return self._statusvar

    def start_measure(self):
        """Start the fast counter."""
        with self._thread_lock:
            # Check if device is configured
            if self._statusvar == 0:
                self.log.error('Cannot start fast counter, device not configured!')
                return -1
                
            # Generate the count data based on simulator
            if self._gated:
                self._generate_gated_count_data()
            else:
                self._generate_single_count_data()
                
            # Set status to running
            self._statusvar = 2
            self._measurement_running = True
            self._start_time = time.time()
            
            return 0

    def pause_measure(self):
        """Pauses the current measurement."""
        with self._thread_lock:
            if self._statusvar == 2:
                self._statusvar = 3
                return 0
            else:
                self.log.error('Cannot pause fast counter, not running!')
                return -1

    def stop_measure(self):
        """Stop the fast counter."""
        with self._thread_lock:
            self._statusvar = 1
            self._measurement_running = False
            return 0

    def continue_measure(self):
        """Continues the current measurement."""
        with self._thread_lock:
            if self._statusvar == 3:
                self._statusvar = 2
                return 0
            else:
                self.log.error('Cannot continue fast counter, not paused!')
                return -1

    def is_gated(self):
        """Check the gated counting possibility.

        @return bool: indicates if counter is gated (True) or not (False)
        """
        return self._gated

    def get_binwidth(self):
        """Returns the width of a single timebin in seconds.

        @return float: current length of a single bin in seconds
        """
        return self._binwidth

    def get_data_trace(self):
        """Polls the current timetrace data from the fast counter.

        @return tuple (numpy.ndarray, dict): timetrace data and info dict
        """
        with self._thread_lock:
            # Check if measurement is running
            if not self._measurement_running and self._statusvar != 2 and self._statusvar != 3:
                self.log.warning('Cannot get trace, measurement not running or paused!')
                
            # Create info dict with elapsed time and sweep info
            elapsed_time = time.time() - self._start_time if self._start_time > 0 else 0
            info_dict = {
                'elapsed_sweeps': int(elapsed_time / (self._binwidth * self._gate_length_bins)),
                'elapsed_time': elapsed_time
            }
            
            # Return the generated data
            return self._count_data, info_dict

    def get_frequency(self):
        """Gets the sample frequency of the counter in Hz.
        
        @return float: sample frequency in Hz
        """
        return 1.0 / self._binwidth

    def _generate_single_count_data(self):
        """Generate a simulated count trace using the NV model."""
        # Get the number of bins from the configuration
        num_bins = self._gate_length_bins
        
        # Create time axis
        time_axis = np.arange(num_bins) * self._binwidth
        
        # Get fluorescence from the simulator
        self._qudi_facade.laser_controller.on()
        base_fluorescence = self._qudi_facade.nv_model.get_fluorescence_rate()
        self._qudi_facade.laser_controller.off()
        
        # Create the count trace
        count_trace = np.zeros(num_bins)
        
        # Apply microwave at a typical NV resonance frequency
        mw_on = time_axis > (max(time_axis) / 2)  # Microwave on for the second half
        
        # Simulate the fluorescence drop due to microwave
        resonant_fluorescence = base_fluorescence * 0.7  # 30% drop in fluorescence when on resonance
        
        # Set fluorescence rates
        count_trace[~mw_on] = base_fluorescence
        count_trace[mw_on] = resonant_fluorescence
        
        # Convert rates to counts per bin
        counts_per_bin = count_trace * self._binwidth
        
        # Add Poisson noise
        noisy_counts = np.random.poisson(counts_per_bin)
        
        # Add dark counts
        dark_counts_per_bin = self._dark_counts * self._binwidth
        noisy_counts += np.random.poisson(dark_counts_per_bin, size=num_bins)
        
        # Store the simulated data
        self._count_data = noisy_counts.astype(np.int64)

    def _generate_gated_count_data(self):
        """Generate simulated gated count traces."""
        # Get parameters from configuration
        num_bins = self._gate_length_bins
        num_gates = self._number_of_gates
        
        # Create time axis
        time_axis = np.arange(num_bins) * self._binwidth
        
        # Initialize count data array
        count_data = np.zeros((num_gates, num_bins), dtype=np.int64)
        
        # Get base fluorescence from the simulator
        self._qudi_facade.laser_controller.on()
        base_fluorescence = self._qudi_facade.nv_model.get_fluorescence_rate()
        self._qudi_facade.laser_controller.off()
        
        # Define pulse sequence parameters
        # This simulates a simple Rabi oscillation with increasing microwave duration per gate
        rabi_period = 1e-6  # 1 µs Rabi period
        
        for gate_idx in range(num_gates):
            # Calculate microwave duration for this gate
            mw_duration = gate_idx * rabi_period / num_gates
            
            # Create a pulse sequence for this gate
            # First part: microwave pulse of varying duration
            mw_on = time_axis < mw_duration
            
            # Second part: laser readout pulse
            laser_on = (time_axis >= mw_duration) & (time_axis < mw_duration + 1e-6)
            
            # Calculate fluorescence for each time bin
            count_trace = np.zeros(num_bins)
            
            # During microwave pulse: no fluorescence
            count_trace[mw_on] = 0
            
            # During laser pulse: modulated fluorescence based on Rabi oscillation
            rabi_contrast = 0.3  # 30% contrast
            rabi_value = 1.0 - rabi_contrast * np.sin(2 * np.pi * mw_duration / rabi_period) ** 2
            count_trace[laser_on] = base_fluorescence * rabi_value
            
            # Convert rates to counts per bin
            counts_per_bin = count_trace * self._binwidth
            
            # Add Poisson noise
            noisy_counts = np.random.poisson(counts_per_bin)
            
            # Add dark counts
            dark_counts_per_bin = self._dark_counts * self._binwidth
            noisy_counts += np.random.poisson(dark_counts_per_bin, size=num_bins)
            
            # Store in the gate data
            count_data[gate_idx] = noisy_counts
            
        # Store the simulated data
        self._count_data = count_data.astype(np.int64)