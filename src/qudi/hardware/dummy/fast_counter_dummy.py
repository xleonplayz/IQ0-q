# -*- coding: utf-8 -*-

"""
This file contains the Qudi hardware dummy for fast counting devices.

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
import os
import numpy as np

from qudi.core.configoption import ConfigOption
from qudi.interface.fast_counter_interface import FastCounterInterface
from qudi.util.mutex import RecursiveMutex


class FastCounterDummy(FastCounterInterface):
    """ Implementation of the FastCounter interface methods for a dummy usage.

    This module integrates with the NV center simulator to provide physically
    accurate time traces based on the state of a simulated NV center.

    Example config for copy-paste:

    fastcounter_dummy:
        module.Class: 'fast_counter_dummy.FastCounterDummy'
        options:
            gated: False
            #load_trace: None # path to the saved dummy trace
            use_simulator: True  # Whether to use the NV center simulator
    """

    # config options
    _gated = ConfigOption('gated', False, missing='warn')
    trace_path = ConfigOption('load_trace', None)
    _use_simulator = ConfigOption('use_simulator', True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.trace_path is None:
            self.trace_path = os.path.abspath(os.path.join(__file__,
                                                           '..',
                                                           'FastComTec_demo_timetrace.asc'))
            self.log.debug(f"Loading dummy fastcounter trace: {self.trace_path}")
        
        # State variables
        self._thread_lock = RecursiveMutex()
        self._simulator_available = False
        self._simulator_manager = None
        self._binwidth = 1
        self._gate_length_bins = 8192
        self._count_data = None

    def on_activate(self):
        """ Initialisation performed during activation of the module.
        """
        self.statusvar = 0
        self._binwidth = 1
        self._gate_length_bins = 8192
        
        # Try to get the simulator manager if configured to use it
        if self._use_simulator:
            try:
                from qudi.hardware.nv_simulator.simulator_manager import SimulatorManager
                self._simulator_manager = SimulatorManager()
                self._simulator_manager.register_module('fast_counter_dummy')
                self._simulator_available = True
                self.log.info("Successfully connected to NV simulator")
            except Exception as e:
                self.log.warning(f"Could not connect to NV simulator: {str(e)}. "
                                f"Using fallback dummy behavior instead.")
                self._simulator_available = False
        else:
            self._simulator_available = False

    def on_deactivate(self):
        """ Deinitialisation performed during deactivation of the module.
        """
        # Unregister from simulator if we were using it
        if self._simulator_available and self._simulator_manager is not None:
            try:
                self._simulator_manager.unregister_module('fast_counter_dummy')
            except:
                pass
        
        self.statusvar = -1
        return

    def get_constraints(self):
        """ Retrieve the hardware constrains from the Fast counting device.

        @return dict: dict with keys being the constraint names as string and
                      items are the definition for the constaints.
        """
        constraints = dict()

        # the unit of those entries are seconds per bin. In order to get the
        # current binwidth in seonds use the get_binwidth method.
        constraints['hardware_binwidth_list'] = [1/950e6, 2/950e6, 4/950e6, 8/950e6]

        return constraints

    def configure(self, bin_width_s, record_length_s, number_of_gates = 0):
        """ Configuration of the fast counter.

        @param float bin_width_s: Length of a single time bin in the time trace
                                  histogram in seconds.
        @param float record_length_s: Total length of the timetrace/each single
                                      gate in seconds.
        @param int number_of_gates: optional, number of gates in the pulse
                                    sequence. Ignore for not gated counter.

        @return tuple(binwidth_s, gate_length_s, number_of_gates):
                    binwidth_s: float the actual set binwidth in seconds
                    gate_length_s: the actual set gate length in seconds
                    number_of_gates: the number of gated, which are accepted
        """
        with self._thread_lock:
            self._binwidth = int(np.rint(bin_width_s * 1e9 * 950 / 1000))
            self._gate_length_bins = int(np.rint(record_length_s / bin_width_s))
            actual_binwidth = self._binwidth * 1000 / 950e9
            actual_length = self._gate_length_bins * actual_binwidth
            self.statusvar = 1
            return actual_binwidth, actual_length, number_of_gates

    def get_status(self):
        """ Receives the current status of the Fast Counter and outputs it as
            return value.

        0 = unconfigured
        1 = idle
        2 = running
        3 = paused
        -1 = error state
        """
        return self.statusvar

    def start_measure(self):
        """Start fast counter measurement."""
        with self._thread_lock:
            # Use simulator when available for physically accurate traces
            if self._simulator_available and self._simulator_manager is not None:
                try:
                    # Ping simulator to update connection status
                    if self._simulator_manager.ping('fast_counter_dummy'):
                        # Get the binwidth and record length
                        bin_width_s = self.get_binwidth()
                        record_length_s = bin_width_s * self._gate_length_bins
                        
                        # Generate a time trace based on the current NV state
                        self._count_data = self._simulator_manager.generate_time_trace(
                            bin_width_s, 
                            record_length_s,
                            number_of_gates=self._gated and 1 or 0
                        )
                        
                        # Set status to running
                        self.statusvar = 2
                        return 0
                except Exception as e:
                    self.log.warning(f"Error using simulator for time trace: {str(e)}. "
                                    f"Falling back to dummy behavior.")
            
            # Fallback to dummy behavior if simulator is not available or fails
            time.sleep(1)
            self.statusvar = 2
            try:
                self._count_data = np.loadtxt(self.trace_path, dtype='int64')
                if self._gated:
                    self._count_data = self._count_data.transpose()
                return 0
            except:
                return -1

    def pause_measure(self):
        """ Pauses the current measurement.

        Fast counter must be initially in the run state to make it pause.
        """
        time.sleep(1)
        self.statusvar = 3
        return 0

    def stop_measure(self):
        """ Stop the fast counter. """
        time.sleep(1)
        self.statusvar = 1
        return 0

    def continue_measure(self):
        """ Continues the current measurement.

        If fast counter is in pause state, then fast counter will be continued.
        """
        self.statusvar = 2
        return 0

    def is_gated(self):
        """ Check the gated counting possibility.

        @return bool: Boolean value indicates if the fast counter is a gated
                      counter (TRUE) or not (FALSE).
        """
        return self._gated

    def get_binwidth(self):
        """ Returns the width of a single timebin in the timetrace in seconds.

        @return float: current length of a single bin in seconds (seconds/bin)
        """
        width_in_seconds = self._binwidth * 1/950e6
        return width_in_seconds

    def get_data_trace(self):
        """ Polls the current timetrace data from the fast counter.

        Return value is a numpy array (dtype = int64).
        The binning, specified by calling configure() in forehand, must be
        taken care of in this hardware class. A possible overflow of the
        histogram bins must be caught here and taken care of.
        If the counter is NOT GATED it will return a tuple (1D-numpy-array, info_dict) with
            returnarray[timebin_index]
        If the counter is GATED it will return a tuple (2D-numpy-array, info_dict) with
            returnarray[gate_index, timebin_index]

        info_dict is a dictionary with keys :
            - 'elapsed_sweeps' : the elapsed number of sweeps
            - 'elapsed_time' : the elapsed time in seconds

        If the hardware does not support these features, the values should be None
        """
        # Update simulator connection if available
        if self._simulator_available and self._simulator_manager is not None:
            try:
                self._simulator_manager.ping('fast_counter_dummy')
            except:
                pass
        
        # Include an artificial waiting time
        time.sleep(0.5)
        info_dict = {'elapsed_sweeps': None, 'elapsed_time': None}
        return self._count_data, info_dict

    def get_frequency(self):
        """Get the sample clock frequency."""
        freq = 950.
        time.sleep(0.5)
        return freq
