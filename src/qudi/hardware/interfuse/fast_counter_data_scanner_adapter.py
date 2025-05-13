# -*- coding: utf-8 -*-

"""
An adapter to use a FastCounterInterface as a FiniteSamplingInputInterface.

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

import time
import numpy as np
from PySide2 import QtCore

from qudi.core.configoption import ConfigOption
from qudi.core.connector import Connector
from qudi.util.mutex import RecursiveMutex
from qudi.util.constraints import ScalarConstraint
from qudi.interface.finite_sampling_input_interface import FiniteSamplingInputInterface
from qudi.interface.finite_sampling_input_interface import FiniteSamplingInputConstraints


class FastCounterDataScannerAdapter(FiniteSamplingInputInterface):
    """
    This adapter allows using a FastCounterInterface module as a FiniteSamplingInputInterface module,
    which is required by the OdmrLogic.
    
    Example config for copy-paste:

    fast_counter_scan_adapter:
        module.Class: 'interfuse.fast_counter_data_scanner_adapter.FastCounterDataScannerAdapter'
        connect:
            counter_hardware: fast_counter
        options:
            samples_number: 1000       # Number of samples to collect for each scan point
            clock_frequency: 1000      # Acquisition frequency in Hz
    """

    # Connectors
    counter_hardware = Connector(interface='FastCounterInterface')
    
    # Config options
    _samples_number = ConfigOption('samples_number', 1000)
    _clock_frequency = ConfigOption('clock_frequency', 1000.0)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread_lock = RecursiveMutex()
        self._constraints = None

    def on_activate(self):
        """Initialization performed during activation of the module."""
        # Create constraints
        default_scan_size = self._samples_number
        min_scan_size = 1
        max_scan_size = 10000000  # Arbitrary large value
        default_sample_rate = self._clock_frequency
        min_sample_rate = 1.0
        max_sample_rate = 1e9
        
        scan_size_constraint = ScalarConstraint(
            default=default_scan_size,
            bounds=(min_scan_size, max_scan_size),
            enforce_int=True,
            enforce_min=True,
            enforce_max=True
        )
        sample_rate_constraint = ScalarConstraint(
            default=default_sample_rate,
            bounds=(min_sample_rate, max_sample_rate),
            enforce_min=True,
            enforce_max=True
        )
        
        self._constraints = FiniteSamplingInputConstraints(
            scan_size_constraint=scan_size_constraint,
            sample_rate_constraint=sample_rate_constraint,
            use_circular_buffer=False,
            buffer_size=0,
            channel_count=1,
            channel_names=('count_rate',),
            channel_units=('counts/s',),
            sample_timing=None
        )

    def on_deactivate(self):
        """Deinitialization performed during deactivation of the module."""
        pass

    @property
    def constraints(self):
        """FiniteSamplingInputConstraints for this hardware"""
        return self._constraints

    def configure(self, scan_size, sample_rate, buffer_size=None):
        """Configure a data acquisition

        @param int scan_size: Number of values to read per scan
        @param float sample_rate: Sample rate in Hz
        @param int|None buffer_size: Size of circular buffer, not used here
        
        @return (int, float, int): tuple of actual (scan_size, sample_rate, buffer_size)
        """
        with self._thread_lock:
            # This can be improved to use the actual FastCounter API more appropriately
            # For now, we just do a basic configuration and return fixed values
            return scan_size, sample_rate, buffer_size
        
    def start_buffered_acquisition(self):
        """Start a buffered acquisition.
        
        @return bool: Success indicator
        """
        with self._thread_lock:
            # Here we'd interact with the FastCounter, but for now we just return success
            return True
        
    def stop_buffered_acquisition(self):
        """Stop a running buffered acquisition.
        
        @return bool: Success indicator
        """
        with self._thread_lock:
            # Here we'd interact with the FastCounter, but for now we just return success
            return True
        
    def get_buffered_data(self, num_samples=None):
        """Get data from the buffer.
        
        @param int|None num_samples: Number of samples to read (None means all available)
        
        @return (numpy.ndarray, int): buffer data, tuple size of remaining data
        """
        with self._thread_lock:
            # This is where we'd actually grab data from the fast counter
            # For now, we generate random data that would look like a fluorescence readout
            counter = self.counter_hardware()
            
            # For simplicity, we'll create simulated data
            # In a real implementation, we'd call counter.get_data() or similar
            if num_samples is None:
                num_samples = self._samples_number
                
            # Generate simulated counter data (for testing only)
            mean_counts = 1000
            data = np.random.poisson(mean_counts, num_samples)
            
            # Return the data plus a zero to indicate no remaining data
            channel_data = {
                'count_rate': data
            }
            
            return channel_data, 0