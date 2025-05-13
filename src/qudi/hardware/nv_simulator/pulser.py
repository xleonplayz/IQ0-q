# -*- coding: utf-8 -*-

"""
This file contains the Qudi hardware module for the NV simulator pulser.

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
import datetime
import os
import sys

from qudi.core.statusvariable import StatusVar
from qudi.core.configoption import ConfigOption
from qudi.core.connector import Connector
from qudi.util.datastorage import get_timestamp_filename, create_dir_for_file
from qudi.util.helpers import natural_sort
from qudi.util.yaml import yaml_dump
from qudi.interface.pulser_interface import PulserInterface, PulserConstraints, SequenceOption
from qudi.util.mutex import Mutex

# Import QudiFacade directly from current directory to avoid circular imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from qudi_facade import QudiFacade


class NVSimPulser(PulserInterface):
    """Pulser interface implementation for the NV simulator.
    
    This module simulates a pulse generator that controls the NV center via microwave
    and laser pulses. It supports both waveform and sequence mode operation.
    
    Example config for copy-paste:

    nv_sim_pulser:
        module.Class: 'nv_simulator.pulser.NVSimPulser'
        options:
            force_sequence_option: False
            save_samples: False
            laser_channel: 'd_ch1'
            microwave_channel: 'd_ch2'
            default_sample_rate: 1.0e9
        connect:
            simulator: nv_simulator
    """

    # Connectors
    simulator = Connector(interface='MicrowaveInterface')
    
    # Status variables that will be saved when closing qudi and restored when starting qudi
    activation_config = StatusVar(default=None)
    
    # Config options
    force_sequence_option = ConfigOption('force_sequence_option', default=False)
    save_samples = ConfigOption('save_samples', default=False)
    laser_channel = ConfigOption('laser_channel', default='d_ch1')
    microwave_channel = ConfigOption('microwave_channel', default='d_ch2')
    default_sample_rate = ConfigOption('default_sample_rate', default=1.0e9)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.log.info('NV Simulator Pulser: Initializing...')
        
        # Thread lock
        self._thread_lock = Mutex()

        # Device properties
        self.connected = False
        self.sample_rate = self.default_sample_rate
        
        # Channel states
        self.channel_states = {
            'a_ch1': False, 'a_ch2': False, 'a_ch3': False,
            'd_ch1': False, 'd_ch2': False, 'd_ch3': False, 'd_ch4': False,
            'd_ch5': False, 'd_ch6': False, 'd_ch7': False, 'd_ch8': False
        }

        # Analog channel parameters
        self.amplitude_dict = {'a_ch1': 1.0, 'a_ch2': 1.0, 'a_ch3': 1.0}
        self.offset_dict = {'a_ch1': 0.0, 'a_ch2': 0.0, 'a_ch3': 0.0}

        # Digital channel parameters
        self.digital_high_dict = {
            'd_ch1': 5.0, 'd_ch2': 5.0, 'd_ch3': 5.0, 'd_ch4': 5.0,
            'd_ch5': 5.0, 'd_ch6': 5.0, 'd_ch7': 5.0, 'd_ch8': 5.0
        }
        self.digital_low_dict = {
            'd_ch1': 0.0, 'd_ch2': 0.0, 'd_ch3': 0.0, 'd_ch4': 0.0,
            'd_ch5': 0.0, 'd_ch6': 0.0, 'd_ch7': 0.0, 'd_ch8': 0.0
        }

        # Internal storage for pulse patterns
        self.waveform_set = set()
        self.waveform_dict = dict()
        self.sequence_dict = dict()
        self.current_loaded_assets = dict()

        # Pulser settings
        self.use_sequencer = True
        self.interleave = False
        self.current_status = 0  # 0: Stopped, 1: Running
        
        # Variables for sequence execution
        self._current_waveform = None
        self._current_sequence = None
        self._current_sequence_step = 0

    def on_activate(self):
        """Initialisation performed during activation of the module."""
        try:
            # Get the QudiFacade instance from the simulator connector
            self._qudi_facade = self.simulator()
        except Exception as e:
            self.log.error(f"Failed to get QudiFacade from connector: {str(e)}")
            # Fallback to direct instantiation
            from qudi_facade import QudiFacade
            self._qudi_facade = QudiFacade()
            self.log.info("Using directly instantiated QudiFacade as fallback")
        
        self.connected = True
        
        # Reset channel states
        self.channel_states = {
            'a_ch1': False, 'a_ch2': False, 'a_ch3': False,
            'd_ch1': False, 'd_ch2': False, 'd_ch3': False, 'd_ch4': False,
            'd_ch5': False, 'd_ch6': False, 'd_ch7': False, 'd_ch8': False
        }
        
        # Set activation config
        if self.activation_config is None:
            self.activation_config = self.get_constraints().activation_config['config0']
        elif self.activation_config not in self.get_constraints().activation_config.values():
            self.activation_config = self.get_constraints().activation_config['config0']
        
        # Activate the channels specified in the activation config
        for chnl in self.activation_config:
            self.channel_states[chnl] = True
            
        self.log.info('NV Simulator Pulser initialized')

    def on_deactivate(self):
        """Deinitialisation performed during deactivation of the module."""
        self.connected = False
        
        # Turn off any active pulses
        if self.current_status == 1:
            self._qudi_facade.laser_controller.off()
            self._qudi_facade.microwave_controller.off()
            self.current_status = 0

    def get_constraints(self):
        """Retrieve the hardware constraints from the Pulsing device.

        @return PulserConstraints: object with pulser constraints
        """
        constraints = PulserConstraints()
        
        # Sample rate constraints (in Hz)
        constraints.sample_rate.min = 10.0e6
        constraints.sample_rate.max = 12.0e9
        constraints.sample_rate.step = 10.0e6
        constraints.sample_rate.default = 1.0e9
        
        # Analog channel amplitude constraints (in V)
        constraints.a_ch_amplitude.min = 0.0
        constraints.a_ch_amplitude.max = 2.0
        constraints.a_ch_amplitude.step = 0.001
        constraints.a_ch_amplitude.default = 1.0
        
        # Analog channel offset constraints (in V)
        constraints.a_ch_offset.min = -1.0
        constraints.a_ch_offset.max = 1.0
        constraints.a_ch_offset.step = 0.001
        constraints.a_ch_offset.default = 0.0
        
        # Digital channel high constraints (in V)
        constraints.d_ch_high.min = 0.0
        constraints.d_ch_high.max = 5.0
        constraints.d_ch_high.step = 0.1
        constraints.d_ch_high.default = 5.0
        
        # Digital channel low constraints (in V)
        constraints.d_ch_low.min = 0.0
        constraints.d_ch_low.max = 5.0
        constraints.d_ch_low.step = 0.1
        constraints.d_ch_low.default = 0.0
        
        # Waveform length constraints
        constraints.waveform_length.min = 1
        constraints.waveform_length.max = 10**9
        constraints.waveform_length.step = 1
        constraints.waveform_length.default = 1000
        
        # Waveform granularity
        constraints.waveform_granularity = 1
        
        # Sequence length constraints
        constraints.sequence_length.min = 1
        constraints.sequence_length.max = 8000
        constraints.sequence_length.step = 1
        constraints.sequence_length.default = 1
        
        # Sequence granularity
        constraints.sequence_granularity = 1
        
        # Sequence option constraints
        if self.force_sequence_option:
            constraints.sequence_option = SequenceOption.FORCED
        else:
            constraints.sequence_option = SequenceOption.OPTIONAL
        
        # Supported channel configurations
        constraints.activation_config = {
            'config0': {'a_ch1', 'd_ch1', 'd_ch2'},
            'config1': {'a_ch1', 'a_ch2', 'd_ch1', 'd_ch2'},
            'config2': {'a_ch1', 'a_ch2', 'a_ch3', 'd_ch1', 'd_ch2', 'd_ch3', 'd_ch4'},
            'config3': {'a_ch1', 'a_ch2', 'a_ch3', 'd_ch1', 'd_ch2', 'd_ch3', 'd_ch4',
                       'd_ch5', 'd_ch6', 'd_ch7', 'd_ch8'},
        }
        
        return constraints

    def pulser_on(self):
        """Switches the pulsing device on."""
        with self._thread_lock:
            if self.current_status == 0:
                self.current_status = 1
                self.log.info('NV Simulator Pulser: Switched on')
                
                # Start pulse sequence if loaded
                if 'waveform' in self.current_loaded_assets:
                    self._execute_waveform(self.current_loaded_assets['waveform'])
                elif 'sequence' in self.current_loaded_assets:
                    self._execute_sequence(self.current_loaded_assets['sequence'])
                    
            return 0

    def pulser_off(self):
        """Switches the pulsing device off."""
        with self._thread_lock:
            if self.current_status == 1:
                # Stop pulse execution
                self._qudi_facade.laser_controller.off()
                self._qudi_facade.microwave_controller.off()
                
                self.current_status = 0
                self.log.info('NV Simulator Pulser: Switched off')
                
            return 0

    def load_waveform(self, load_dict):
        """Load a waveform to the pulser device.

        @param dict load_dict: dictionary containing waveform parameters
            dictionary keys include:
                'waveform_name': str
                'analog_samples': numpy.ndarray of analog voltage values (optional)
                'digital_samples': numpy.ndarray of digital voltage values (optional)
                'is_first_chunk': bool (optional)
                'is_last_chunk': bool (optional)
                'total_number_of_samples': int (optional)
        """
        with self._thread_lock:
            waveform_name = load_dict['waveform_name']
            
            # Check if waveform exists
            if waveform_name not in self.waveform_set:
                self.log.error(f'Waveform "{waveform_name}" not found in waveform list')
                return -1
                
            self.current_loaded_assets = dict()
            self.current_loaded_assets['waveform'] = waveform_name
            self.log.info(f'NV Simulator Pulser: Waveform "{waveform_name}" loaded')
            
            return 0

    def load_sequence(self, sequence_name):
        """Load a sequence to the pulser device.

        @param str sequence_name: Name of the sequence to be loaded
        
        @return int: error code (0: OK, -1: error)
        """
        with self._thread_lock:
            if sequence_name not in self.sequence_dict:
                self.log.error(f'Sequence "{sequence_name}" not found in sequence list')
                return -1
                
            self.current_loaded_assets = dict()
            self.current_loaded_assets['sequence'] = sequence_name
            self.log.info(f'NV Simulator Pulser: Sequence "{sequence_name}" loaded')
            
            return 0

    def get_loaded_assets(self):
        """Retrieve the currently loaded assets from the device memory.

        @return dict: Dictionary containing all loaded assets
        """
        return self.current_loaded_assets.copy()

    def clear_all(self):
        """Clear all loaded waveforms and sequences from the device memory."""
        with self._thread_lock:
            self.waveform_set = set()
            self.waveform_dict = dict()
            self.sequence_dict = dict()
            self.current_loaded_assets = dict()
            self.log.info('NV Simulator Pulser: All waveforms and sequences cleared')
            return 0

    def get_status(self):
        """Retrieve the status of the pulser hardware.

        @return (int, dict): status code (0: idle, 1: running)
                             dictionary with status messages
        """
        status_dict = {'status': self.current_status}
        return self.current_status, status_dict

    def get_sample_rate(self):
        """Get the sample rate of the pulser device.

        @return float: current sample rate
        """
        return self.sample_rate

    def set_sample_rate(self, sample_rate):
        """Set the sample rate of the pulser device.

        @param float sample_rate: The sample rate to be set (in Hz)

        @return float: the sample rate actually set
        """
        with self._thread_lock:
            constraints = self.get_constraints()
            
            # Check if sample rate is within bounds
            if sample_rate < constraints.sample_rate.min:
                self.sample_rate = constraints.sample_rate.min
            elif sample_rate > constraints.sample_rate.max:
                self.sample_rate = constraints.sample_rate.max
            else:
                # Round to nearest multiple of step size
                steps = round(sample_rate / constraints.sample_rate.step)
                self.sample_rate = steps * constraints.sample_rate.step
                
            self.log.info(f'NV Simulator Pulser: Sample rate set to {self.sample_rate:.1f} Hz')
            return self.sample_rate

    def get_analog_level(self, amplitude=None, offset=None):
        """Retrieve the analog amplitude and offset of the provided channels.

        @param list(str) amplitude: Names of the amplitude channels
        @param list(str) offset: Names of the offset channels

        @return dict: with keys being the channel string and items being the values
        """
        result = dict()
        
        if amplitude is not None:
            for chnl in amplitude:
                if chnl in self.amplitude_dict:
                    result[chnl] = self.amplitude_dict[chnl]
                    
        if offset is not None:
            for chnl in offset:
                if chnl in self.offset_dict:
                    result[chnl] = self.offset_dict[chnl]
                    
        return result

    def set_analog_level(self, amplitude=None, offset=None):
        """Set amplitude and/or offset value of the provided analog channels.

        @param dict amplitude: with channel names as keys and amplitude values as items
        @param dict offset: with channel names as keys and offset values as items

        @return (dict, dict): with channel names as keys and error messages as items
        """
        with self._thread_lock:
            amplitude_result = dict()
            offset_result = dict()
            
            if amplitude is not None:
                for chnl, amp in amplitude.items():
                    if chnl in self.amplitude_dict:
                        constraints = self.get_constraints()
                        if amp < constraints.a_ch_amplitude.min:
                            amp = constraints.a_ch_amplitude.min
                        elif amp > constraints.a_ch_amplitude.max:
                            amp = constraints.a_ch_amplitude.max
                            
                        self.amplitude_dict[chnl] = amp
                        amplitude_result[chnl] = None  # No error
                    else:
                        amplitude_result[chnl] = f'Channel {chnl} not present in analog channel list'
                        
            if offset is not None:
                for chnl, off in offset.items():
                    if chnl in self.offset_dict:
                        constraints = self.get_constraints()
                        if off < constraints.a_ch_offset.min:
                            off = constraints.a_ch_offset.min
                        elif off > constraints.a_ch_offset.max:
                            off = constraints.a_ch_offset.max
                            
                        self.offset_dict[chnl] = off
                        offset_result[chnl] = None  # No error
                    else:
                        offset_result[chnl] = f'Channel {chnl} not present in analog channel list'
                        
            return amplitude_result, offset_result

    def get_digital_level(self, low=None, high=None):
        """Retrieve the digital low and high level of the provided channels.

        @param list(str) low: Names of the low level channels
        @param list(str) high: Names of the high level channels

        @return dict: with keys being the channel string and items being the values
        """
        result = dict()
        
        if low is not None:
            for chnl in low:
                if chnl in self.digital_low_dict:
                    result[chnl] = self.digital_low_dict[chnl]
                    
        if high is not None:
            for chnl in high:
                if chnl in self.digital_high_dict:
                    result[chnl] = self.digital_high_dict[chnl]
                    
        return result

    def set_digital_level(self, low=None, high=None):
        """Set low and/or high value of the provided digital channels.

        @param dict low: with channel names as keys and low values as items
        @param dict high: with channel names as keys and high values as items

        @return (dict, dict): with channel names as keys and error messages as items
        """
        with self._thread_lock:
            low_result = dict()
            high_result = dict()
            
            if low is not None:
                for chnl, value in low.items():
                    if chnl in self.digital_low_dict:
                        constraints = self.get_constraints()
                        if value < constraints.d_ch_low.min:
                            value = constraints.d_ch_low.min
                        elif value > constraints.d_ch_low.max:
                            value = constraints.d_ch_low.max
                            
                        self.digital_low_dict[chnl] = value
                        low_result[chnl] = None  # No error
                    else:
                        low_result[chnl] = f'Channel {chnl} not present in digital channel list'
                        
            if high is not None:
                for chnl, value in high.items():
                    if chnl in self.digital_high_dict:
                        constraints = self.get_constraints()
                        if value < constraints.d_ch_high.min:
                            value = constraints.d_ch_high.min
                        elif value > constraints.d_ch_high.max:
                            value = constraints.d_ch_high.max
                            
                        self.digital_high_dict[chnl] = value
                        high_result[chnl] = None  # No error
                    else:
                        high_result[chnl] = f'Channel {chnl} not present in digital channel list'
                        
            return low_result, high_result

    def get_active_channels(self, ch=None):
        """Get the active channels of the pulser hardware.

        @param list ch: optional, a list of channels to check
        @return dict: with channel string as key and bool as item
        """
        if ch is None:
            return self.channel_states.copy()
        else:
            result = dict()
            for chnl in ch:
                if chnl in self.channel_states:
                    result[chnl] = self.channel_states[chnl]
                else:
                    result[chnl] = False
            return result

    def set_active_channels(self, ch=None):
        """Set the active channels for the pulser hardware.

        @param dict ch: dictionary with channel string as key and bool as item

        @return dict: with channel string as key and error as item
        """
        with self._thread_lock:
            result = dict()
            
            if ch is None:
                result['error'] = 'No channels provided'
                return result
                
            for chnl, state in ch.items():
                if chnl in self.channel_states:
                    self.channel_states[chnl] = state
                    result[chnl] = None  # No error
                else:
                    result[chnl] = f'Channel {chnl} not present in channel list'
                    
            return result

    def write_waveform(self, name, analog_samples, digital_samples, is_first_chunk=True,
                      is_last_chunk=True, total_number_of_samples=None):
        """Write a waveform to the pulser device.

        @param str name: unique name to identify the waveform
        @param numpy.ndarray analog_samples: array of analog samples
        @param numpy.ndarray digital_samples: array of digital samples
        @param bool is_first_chunk: flag indicating if this is the first chunk
        @param bool is_last_chunk: flag indicating if this is the last chunk
        @param int total_number_of_samples: optional, total length of the waveform (if chunking)

        @return (int, list): number of samples written (-1 on error) and list of created waveform names
        """
        with self._thread_lock:
            if not is_first_chunk and name not in self.waveform_set:
                self.log.error(f'Write waveform failed: Waveform "{name}" not found but not first chunk.')
                return -1, list()
                
            if is_first_chunk:
                self.waveform_dict[name] = dict()
                self.waveform_dict[name]['analog'] = analog_samples
                self.waveform_dict[name]['digital'] = digital_samples
                self.waveform_set.add(name)
                
                if total_number_of_samples is not None and not is_last_chunk:
                    # Initialize for chunking
                    self.waveform_dict[name]['total_samples'] = total_number_of_samples
                    self.waveform_dict[name]['current_position'] = len(analog_samples)
            else:
                # Append chunk to existing waveform
                self.waveform_dict[name]['analog'] = np.append(self.waveform_dict[name]['analog'], 
                                                             analog_samples)
                self.waveform_dict[name]['digital'] = np.append(self.waveform_dict[name]['digital'], 
                                                              digital_samples)
                
                if not is_last_chunk:
                    self.waveform_dict[name]['current_position'] += len(analog_samples)
                    
            # Save samples if configured and this is the last chunk
            if self.save_samples and is_last_chunk:
                self._save_samples_to_file(name)
                
            return len(self.waveform_dict[name]['analog']), [name]

    def write_sequence(self, name, sequence_parameters):
        """Write a sequence to the pulser device.

        @param str name: unique name to identify the sequence
        @param list sequence_parameters: list containing dictionaries for each sequence step

        @return: int: number of sequence steps written (-1 on error)
        """
        with self._thread_lock:
            # Check if all waveforms in the sequence exist
            for step_params in sequence_parameters:
                if 'waveform_name' not in step_params or step_params['waveform_name'] not in self.waveform_set:
                    self.log.error('Sequence contains non-existent waveform')
                    return -1
                    
            # Store the sequence
            self.sequence_dict[name] = sequence_parameters
            
            return len(sequence_parameters)

    def get_waveform_names(self):
        """Retrieve the names of all waveforms stored on the device.

        @return list: list of all waveform names
        """
        return list(self.waveform_set)

    def get_sequence_names(self):
        """Retrieve the names of all sequences stored on the device.

        @return list: list of all sequence names
        """
        return list(self.sequence_dict.keys())

    def delete_waveform(self, waveform_name):
        """Delete the waveform with the passed name from the device memory.

        @param str waveform_name: name of the waveform to be deleted

        @return int: error code (0: OK, -1: error)
        """
        with self._thread_lock:
            if waveform_name not in self.waveform_set:
                self.log.error(f'Waveform "{waveform_name}" not found in waveform list')
                return -1
                
            self.waveform_set.remove(waveform_name)
            if waveform_name in self.waveform_dict:
                del self.waveform_dict[waveform_name]
                
            # Also check if it's currently loaded
            if 'waveform' in self.current_loaded_assets and self.current_loaded_assets['waveform'] == waveform_name:
                self.current_loaded_assets = dict()
                
            return 0

    def delete_sequence(self, sequence_name):
        """Delete the sequence with the passed name from the device memory.

        @param str sequence_name: name of the sequence to be deleted

        @return int: error code (0: OK, -1: error)
        """
        with self._thread_lock:
            if sequence_name not in self.sequence_dict:
                self.log.error(f'Sequence "{sequence_name}" not found in sequence list')
                return -1
                
            del self.sequence_dict[sequence_name]
            
            # Also check if it's currently loaded
            if 'sequence' in self.current_loaded_assets and self.current_loaded_assets['sequence'] == sequence_name:
                self.current_loaded_assets = dict()
                
            return 0

    def get_interleave(self):
        """Check whether Interleave is active on the AWG.

        @return bool: True if the interleave is active, False otherwise
        """
        return self.interleave

    def set_interleave(self, state=False):
        """Turn the interleave of an AWG on or off.

        @param bool state: The state the interleave should be set to
                           (True: ON, False: OFF)

        @return bool: actual interleave status (True: ON, False: OFF)
        """
        self.interleave = state
        return self.interleave

    def reset(self):
        """Reset the device.

        @return int: error code (0: OK, -1: error)
        """
        with self._thread_lock:
            self.waveform_set = set()
            self.waveform_dict = dict()
            self.sequence_dict = dict()
            self.current_loaded_assets = dict()
            
            # Reset device state
            self.current_status = 0
            
            # Turn off all pulses
            self._qudi_facade.laser_controller.off()
            self._qudi_facade.microwave_controller.off()
            
            return 0

    def has_sequence_mode(self):
        """Check whether the device has a sequence mode.

        @return bool: True if the device has a sequence mode, False otherwise
        """
        return True
        
    def _execute_waveform(self, waveform_name):
        """Execute a loaded waveform by sending appropriate pulses to the NV simulator.
        
        @param str waveform_name: Name of the waveform to execute
        """
        if waveform_name not in self.waveform_dict:
            self.log.error(f'Waveform "{waveform_name}" not found in waveform dictionary')
            return
            
        self.log.debug(f'Executing waveform "{waveform_name}"')
        
        # Get the waveform data
        waveform = self.waveform_dict[waveform_name]
        digital_samples = waveform['digital']
        
        # Get the channel indices for laser and microwave
        laser_index = int(self.laser_channel.split('ch')[1]) - 1
        mw_index = int(self.microwave_channel.split('ch')[1]) - 1
        
        # Process the pulse sequence
        self._process_pulse_pattern(digital_samples, laser_index, mw_index)
        
    def _execute_sequence(self, sequence_name):
        """Execute a loaded sequence by iterating through its steps.
        
        @param str sequence_name: Name of the sequence to execute
        """
        if sequence_name not in self.sequence_dict:
            self.log.error(f'Sequence "{sequence_name}" not found in sequence dictionary')
            return
            
        self.log.debug(f'Executing sequence "{sequence_name}"')
        
        # Get the sequence data
        sequence = self.sequence_dict[sequence_name]
        
        # Process each waveform in the sequence
        for step_params in sequence:
            waveform_name = step_params['waveform_name']
            
            if waveform_name not in self.waveform_dict:
                self.log.error(f'Waveform "{waveform_name}" not found in waveform dictionary')
                continue
                
            # Execute the waveform for this step
            self._execute_waveform(waveform_name)
            
            # Wait for repetitions if specified
            if 'repetitions' in step_params:
                for _ in range(int(step_params['repetitions']) - 1):
                    self._execute_waveform(waveform_name)
                    
    def _process_pulse_pattern(self, digital_samples, laser_index, mw_index):
        """Process a digital pulse pattern and apply it to the NV simulator.
        
        @param numpy.ndarray digital_samples: Array of digital samples
        @param int laser_index: Index of the laser channel in the digital samples
        @param int mw_index: Index of the microwave channel in the digital samples
        """
        # Find where pulses start and end
        laser_pattern = (digital_samples[:, laser_index] > 0).astype(int)
        mw_pattern = (digital_samples[:, mw_index] > 0).astype(int)
        
        # Find transitions (0->1 or 1->0)
        laser_transitions = np.diff(np.append(0, laser_pattern))
        mw_transitions = np.diff(np.append(0, mw_pattern))
        
        # Process laser pulses
        laser_starts = np.where(laser_transitions > 0)[0]
        laser_ends = np.where(laser_transitions < 0)[0]
        
        # Process microwave pulses
        mw_starts = np.where(mw_transitions > 0)[0]
        mw_ends = np.where(mw_transitions < 0)[0]
        
        # Combine all events in chronological order
        events = []
        
        for start in laser_starts:
            events.append(('laser_on', start))
            
        for end in laser_ends:
            events.append(('laser_off', end))
            
        for start in mw_starts:
            events.append(('mw_on', start))
            
        for end in mw_ends:
            events.append(('mw_off', end))
            
        # Sort events by time
        events.sort(key=lambda x: x[1])
        
        # Process events sequentially
        laser_state = False
        mw_state = False
        
        for event, time_idx in events:
            # Calculate the actual time for this event
            actual_time = time_idx / self.sample_rate
            
            # Evolve the system to this time point
            time_diff = actual_time - self._qudi_facade.nv_model.get_current_time()
            if time_diff > 0:
                self._qudi_facade.nv_model.evolve(time_diff)
            
            # Apply the event
            if event == 'laser_on' and not laser_state:
                self._qudi_facade.laser_controller.on()
                laser_state = True
            elif event == 'laser_off' and laser_state:
                self._qudi_facade.laser_controller.off()
                laser_state = False
            elif event == 'mw_on' and not mw_state:
                self._qudi_facade.microwave_controller.on()
                mw_state = True
            elif event == 'mw_off' and mw_state:
                self._qudi_facade.microwave_controller.off()
                mw_state = False
                
        # Ensure all pulses are turned off at the end
        if laser_state:
            self._qudi_facade.laser_controller.off()
        if mw_state:
            self._qudi_facade.microwave_controller.off()
            
    def _save_samples_to_file(self, waveform_name):
        """Save the samples of a waveform to file for debugging.
        
        @param str waveform_name: Name of the waveform to save
        """
        if waveform_name not in self.waveform_dict:
            return
            
        # Get timestamp for filename
        timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        
        # Create directory if it doesn't exist
        directory = os.path.join(os.path.expanduser('~'), 'saved_waveforms')
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Create filename
        filename = os.path.join(directory, f'{timestamp}_{waveform_name}.npz')
        
        # Save waveform data
        np.savez(filename, 
                 analog=self.waveform_dict[waveform_name]['analog'],
                 digital=self.waveform_dict[waveform_name]['digital'])
                 
        self.log.info(f'Saved waveform "{waveform_name}" to {filename}')