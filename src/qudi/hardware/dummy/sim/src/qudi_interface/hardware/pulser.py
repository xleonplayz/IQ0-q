# -*- coding: utf-8 -*-

"""
Qudi hardware interface adapter for NV simulator pulse generator.
This module implements the PulserInterface for the NV center simulator,
enabling pulse sequence control for quantum experiments.

Copyright (c) 2023
"""

import numpy as np
import time
import threading
from typing import Dict, Any, List, Tuple, Union, Optional, Set

from qudi.interface.pulser_interface import PulserInterface, PulserConstraints
from qudi.util.constraints import ScalarConstraint
from qudi.util.helpers import natural_sort

from .qudi_facade import QudiFacade


class NVSimPulser(PulserInterface):
    """
    Hardware adapter that implements the PulserInterface for the NV center simulator.
    This interface enables pulse sequence control for quantum experiments.
    """

    def __init__(self, config=None, **kwargs):
        """
        Initialize the pulser adapter for the NV simulator.
        
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
        
        # Set up channels
        self._analog_channels = {'a_ch1': 'MW_amplitude', 'a_ch2': 'MW_phase'}
        self._digital_channels = {'d_ch1': 'MW_switch', 'd_ch2': 'Laser'}
        
        # Default active channels
        self._active_analog_channels = set()
        self._active_digital_channels = set()
        
        # Store waveforms and sequences
        self._waveforms = {}  # name -> (analog_samples, digital_samples)
        self._sequences = {}  # name -> [(wfm_name, repetitions), ...]
        self._loaded_waveform = None
        self._loaded_sequence = None
        self._current_loaded_assets = None  # Tuple (waveform, sequence)
        
        # State variables
        self._is_running = False
        self._current_waveform_pos = 0
        self._current_sequence_pos = 0
        self._sequence_loop_count = 0
        
        # Running thread
        self._pulse_thread = None
        self._stop_playing = threading.Event()
        
        # Thread lock for thread safety
        self._thread_lock = self.module_state.lock_access()
        
        self.log.info("NV Simulator pulser initialized")

    def on_activate(self):
        """
        Called when module is activated
        """
        self.log.info('NV Simulator pulser activated')
        
        # Apply initial configuration
        # Set active channels from config if provided
        if 'active_analog_channels' in self._config:
            self._active_analog_channels = set(self._config['active_analog_channels'])
            
        if 'active_digital_channels' in self._config:
            self._active_digital_channels = set(self._config['active_digital_channels'])
        
        # Reset waveform storage
        self._waveforms = {}
        self._sequences = {}
        self._loaded_waveform = None
        self._loaded_sequence = None
        self._current_loaded_assets = None
        
    def on_deactivate(self):
        """
        Called when module is deactivated
        """
        # Stop any running sequence
        self.stop()
        self.log.info('NV Simulator pulser deactivated')

    def get_constraints(self):
        """
        Retrieve the hardware constrains from the pulser device.
        
        @return PulserConstraints: object with pulser constraints
        """
        constraints = PulserConstraints()
        
        # The maximum number of different waveforms or sequences that can be stored simultaneously
        constraints.waveform_num = 128
        constraints.sequence_num = 64
        
        # The maximum number of steps within a sequence
        constraints.sequence_steps = 8192
        
        # The length constraints for waveforms and sequences in (samples)
        constraints.waveform_length = ScalarConstraint(min=16, max=8e9, step=1)
        
        # The sample rate for generating waveforms in Hz
        constraints.sample_rate = ScalarConstraint(min=1e6, max=1e9, step=1)
        
        # The analog amplitude for waveforms in V
        constraints.a_ch_amplitude = ScalarConstraint(min=-1.0, max=1.0, step=0.001)
        # The analog offset for waveforms in V
        constraints.a_ch_offset = ScalarConstraint(min=-1.0, max=1.0, step=0.001)
        
        # The names of the activation config settings for the analog channels
        constraints.activation_config = {
            'MW_control': frozenset({'a_ch1', 'd_ch1'}),
            'MW_control_phase': frozenset({'a_ch1', 'a_ch2', 'd_ch1'}),
            'Full': frozenset({'a_ch1', 'a_ch2', 'd_ch1', 'd_ch2'})
        }
        
        # The available analog and digital channels
        constraints.analog_channels = frozenset(self._analog_channels.keys())
        constraints.digital_channels = frozenset(self._digital_channels.keys())
        
        return constraints

    def pulser_on(self):
        """
        Switches the pulsing device on.
        
        @return int: error code (0:OK, -1:error)
        """
        with self._thread_lock:
            self.log.info("Pulser turned on")
            
            # If we have a loaded waveform or sequence, start playing it
            if self._current_loaded_assets is not None:
                self._start_pulse_sequence()
                
            return 0

    def pulser_off(self):
        """
        Switches the pulsing device off.
        
        @return int: error code (0:OK, -1:error)
        """
        with self._thread_lock:
            # Stop the sequence if running
            self.stop()
            self.log.info("Pulser turned off")
            return 0

    def reset(self):
        """
        Resets the device.
        
        @return int: error code (0:OK, -1:error)
        """
        with self._thread_lock:
            # Stop any running sequence
            self.stop()
            
            # Reset waveform storage
            self._waveforms = {}
            self._sequences = {}
            self._loaded_waveform = None
            self._loaded_sequence = None
            self._current_loaded_assets = None
            
            self.log.info("Pulser reset")
            return 0

    def get_status(self):
        """
        Gets the current status of the pulser hardware.
        
        @return (bool, str): tuple of (status, statustext)
                    - status: True if device is running, False otherwise
                    - statustext: plain text description of current device status
        """
        if self._is_running:
            return True, "Pulser is running"
        else:
            return False, "Pulser is idle"

    def get_sample_rate(self):
        """
        Get the sample rate of the pulse generator hardware.
        
        @return float: The current sample rate in Hz
        """
        # Default sample rate
        return 1.0e9  # 1 GHz

    def set_sample_rate(self, sample_rate):
        """
        Set the sample rate of the pulse generator hardware.
        (Not actually changing anything in the simulator, but maintaining the interface)
        
        @param float sample_rate: The sampling rate to be set in Hz
        
        @return float: the sample rate actually set
        """
        # We don't actually change the sample rate in the simulator, just return the specified rate
        constraints = self.get_constraints()
        if sample_rate < constraints.sample_rate.min:
            sample_rate = constraints.sample_rate.min
        elif sample_rate > constraints.sample_rate.max:
            sample_rate = constraints.sample_rate.max
        
        self.log.info(f"Sample rate set to {sample_rate} Hz")
        return sample_rate

    def get_analog_level(self, amplitude=None, offset=None):
        """
        Retrieve the analog amplitude and offset of the provided channels.
        
        @param list amplitude: optional, names of the analog channels to get the amplitude
        @param list offset: optional, names of the analog channels to get the offset
        
        @return dict: with keys being the channel names and items being the values
        """
        # Default values
        amp_dict = {ch: 1.0 for ch in self._analog_channels}
        off_dict = {ch: 0.0 for ch in self._analog_channels}
        
        ret_dict = {}
        if amplitude is not None:
            for a_ch in amplitude:
                ret_dict[a_ch] = amp_dict[a_ch]
                
        if offset is not None:
            for a_ch in offset:
                ret_dict[a_ch+'_offset'] = off_dict[a_ch]
                
        return ret_dict

    def set_analog_level(self, amplitude=None, offset=None):
        """
        Set the amplitude and/or offset of the provided analog channels.
        (Not actually changing anything in the simulator, but maintaining the interface)
        
        @param dict amplitude: dictionary with keys being the channel names and items
                               being the amplitude values in V
        @param dict offset: dictionary with keys being the channel names and items
                            being the offset values in V
        
        @return dict: with the actually set values for amplitude and offset for all channels
        """
        # Since we're simulating, we don't actually change any hardware settings
        # Just log the requested changes
        if amplitude is not None:
            self.log.info(f"Setting analog amplitudes: {amplitude}")
            
        if offset is not None:
            self.log.info(f"Setting analog offsets: {offset}")
            
        return self.get_analog_level(
            amplitude=list(self._analog_channels.keys()) if amplitude is not None else None,
            offset=list(self._analog_channels.keys()) if offset is not None else None
        )

    def get_digital_level(self, low=None, high=None):
        """
        Retrieve the digital low and high level of the provided channels.
        
        @param list low: optional, names of the digital channels to get the low level
        @param list high: optional, names of the digital channels to get the high level
        
        @return dict: with keys being the channel names and items being the values
        """
        # Default values - low is always 0V, high is always 5V (standard TTL)
        low_dict = {ch: 0.0 for ch in self._digital_channels}
        high_dict = {ch: 5.0 for ch in self._digital_channels}
        
        ret_dict = {}
        if low is not None:
            for d_ch in low:
                ret_dict[d_ch+'_low'] = low_dict[d_ch]
                
        if high is not None:
            for d_ch in high:
                ret_dict[d_ch+'_high'] = high_dict[d_ch]
                
        return ret_dict

    def set_digital_level(self, low=None, high=None):
        """
        Set the low and/or high level of the provided digital channels.
        (Not actually changing anything in the simulator, but maintaining the interface)
        
        @param dict low: dictionary with keys being the channel names and items
                         being the low values in V
        @param dict high: dictionary with keys being the channel names and items
                          being the high values in V
        
        @return dict: with the actually set values for low and high for all channels
        """
        # Since we're simulating, we don't actually change any hardware settings
        # Just log the requested changes
        if low is not None:
            self.log.info(f"Setting digital low levels: {low}")
            
        if high is not None:
            self.log.info(f"Setting digital high levels: {high}")
            
        return self.get_digital_level(
            low=list(self._digital_channels.keys()) if low is not None else None,
            high=list(self._digital_channels.keys()) if high is not None else None
        )

    def get_active_channels(self, ch=None):
        """
        Get the active channels of the pulse generator hardware.
        
        @param list ch: optional, names of the channels to check
        
        @return dict: dictionary with keys being the channel names and items being
                      boolean values indicating whether the channel is active
        """
        if ch is None:
            ch = []
            ch.extend(self._analog_channels)
            ch.extend(self._digital_channels)
            
        active_dict = {}
        for channel in ch:
            if channel in self._analog_channels:
                active_dict[channel] = channel in self._active_analog_channels
            elif channel in self._digital_channels:
                active_dict[channel] = channel in self._active_digital_channels
                
        return active_dict

    def set_active_channels(self, ch=None):
        """
        Set the active channels for the pulse generator hardware.
        
        @param dict ch: dictionary with keys being the channel names and items being
                        boolean values indicating whether the channel should be active
        
        @return dict: dictionary with the actual set values for active channels
        """
        if ch is None:
            return self.get_active_channels()
            
        for channel, active in ch.items():
            if active:
                if channel in self._analog_channels:
                    self._active_analog_channels.add(channel)
                elif channel in self._digital_channels:
                    self._active_digital_channels.add(channel)
            else:
                if channel in self._analog_channels and channel in self._active_analog_channels:
                    self._active_analog_channels.remove(channel)
                elif channel in self._digital_channels and channel in self._active_digital_channels:
                    self._active_digital_channels.remove(channel)
                    
        self.log.info(f"Active analog channels: {self._active_analog_channels}")
        self.log.info(f"Active digital channels: {self._active_digital_channels}")
        
        return self.get_active_channels(ch=list(ch.keys()))

    def write_waveform(self, name, analog_samples, digital_samples, is_first_chunk, is_last_chunk):
        """
        Write a new waveform or append to an existing waveform on the device.
        
        @param str name: name for the waveform to be created/append to
        @param numpy.ndarray analog_samples: array of analog samples to be written
        @param numpy.ndarray digital_samples: array of digital samples to be written
        @param bool is_first_chunk: flag indicating if this is the first chunk to write
        @param bool is_last_chunk: flag indicating if this is the last chunk to write
        
        @return int: error code (0:OK, -1:error)
        """
        with self._thread_lock:
            # Convert digital samples to absolute if needed
            if digital_samples.ndim == 2:
                # Each digital channel is one column
                digital_absolute = np.zeros_like(digital_samples)
                for i in range(digital_samples.shape[1]):
                    digital_absolute[:, i] = digital_samples[:, i]
            else:
                # Digital samples is already in absolute format
                digital_absolute = digital_samples
            
            # If first chunk, create new or overwrite existing waveform
            if is_first_chunk:
                self._waveforms[name] = (analog_samples, digital_absolute)
            else:
                # Append to existing waveform
                if name in self._waveforms:
                    old_analog, old_digital = self._waveforms[name]
                    self._waveforms[name] = (
                        np.concatenate([old_analog, analog_samples]),
                        np.concatenate([old_digital, digital_absolute])
                    )
                else:
                    self.log.error(f"Cannot append to non-existent waveform: {name}")
                    return -1
            
            self.log.info(f"Waveform '{name}' written/updated")
            
            # If this is the last chunk, log the waveform details
            if is_last_chunk:
                analog, digital = self._waveforms[name]
                self.log.debug(f"Waveform '{name}' complete: {analog.shape[0]} samples, " +
                              f"{analog.shape[1]} analog channels, {digital.shape[1]} digital channels")
            
            return 0

    def write_sequence(self, name, sequence_parameters):
        """
        Write a new sequence on the device.
        
        @param str name: name for the sequence to be created
        @param list sequence_parameters: List containing tuples of 
                       (waveform_name, repetitions, segname) for each step
        
        @return int: error code (0:OK, -1:error)
        """
        with self._thread_lock:
            # Validate that all waveforms in the sequence exist
            for wfm_tuple in sequence_parameters:
                wfm_name = wfm_tuple[0]
                if wfm_name not in self._waveforms:
                    self.log.error(f"Cannot create sequence with non-existent waveform: {wfm_name}")
                    return -1
            
            # Store the sequence
            # Convert to simplified format (waveform_name, repetitions) for our simulator
            sequence_steps = [(step[0], step[1]) for step in sequence_parameters]
            self._sequences[name] = sequence_steps
            
            self.log.info(f"Sequence '{name}' created with {len(sequence_steps)} steps")
            return 0

    def get_waveform_names(self):
        """
        Retrieve the names of all uploaded waveforms on the device.
        
        @return list: list of all uploaded waveform names
        """
        return natural_sort(list(self._waveforms.keys()))

    def get_sequence_names(self):
        """
        Retrieve the names of all uploaded sequences on the device.
        
        @return list: list of all uploaded sequence names
        """
        return natural_sort(list(self._sequences.keys()))

    def delete_waveform(self, waveform_name):
        """
        Delete the waveform with name "waveform_name" from the device memory.
        
        @param str waveform_name: The name of the waveform to be deleted
        
        @return int: error code (0:OK, -1:error)
        """
        with self._thread_lock:
            if waveform_name in self._waveforms:
                del self._waveforms[waveform_name]
                self.log.info(f"Waveform '{waveform_name}' deleted")
                
                # If this waveform was loaded, unload it
                if self._loaded_waveform == waveform_name:
                    self._loaded_waveform = None
                    self._current_loaded_assets = None
                
                return 0
            else:
                self.log.error(f"Cannot delete non-existent waveform: {waveform_name}")
                return -1

    def delete_sequence(self, sequence_name):
        """
        Delete the sequence with name "sequence_name" from the device memory.
        
        @param str sequence_name: The name of the sequence to be deleted
        
        @return int: error code (0:OK, -1:error)
        """
        with self._thread_lock:
            if sequence_name in self._sequences:
                del self._sequences[sequence_name]
                self.log.info(f"Sequence '{sequence_name}' deleted")
                
                # If this sequence was loaded, unload it
                if self._loaded_sequence == sequence_name:
                    self._loaded_sequence = None
                    self._current_loaded_assets = None
                
                return 0
            else:
                self.log.error(f"Cannot delete non-existent sequence: {sequence_name}")
                return -1

    def load_waveform(self, waveform_name, to_ch=None):
        """
        Load the waveform "waveform_name" to the specified channel.
        
        @param str waveform_name: name of the waveform to load
        @param list to_ch: optional, channel name(s) to load waveform to
        
        @return int: error code (0:OK, -1:error)
        """
        with self._thread_lock:
            if waveform_name in self._waveforms:
                # Store the loaded waveform name
                self._loaded_waveform = waveform_name
                self._loaded_sequence = None
                self._current_loaded_assets = (waveform_name, None)
                
                # If channels are specified, validate and set active
                if to_ch is not None:
                    for ch in to_ch:
                        if (ch in self._analog_channels and ch not in self._active_analog_channels) or \
                           (ch in self._digital_channels and ch not in self._active_digital_channels):
                            self.log.warning(f"Channel {ch} is being loaded but is not active")
                
                self.log.info(f"Waveform '{waveform_name}' loaded")
                return 0
            else:
                self.log.error(f"Cannot load non-existent waveform: {waveform_name}")
                return -1

    def load_sequence(self, sequence_name, to_ch=None):
        """
        Load the sequence "sequence_name" to the specified channel.
        
        @param str sequence_name: name of the sequence to load
        @param list to_ch: optional, channel name(s) to load sequence to
        
        @return int: error code (0:OK, -1:error)
        """
        with self._thread_lock:
            if sequence_name in self._sequences:
                # Store the loaded sequence name
                self._loaded_sequence = sequence_name
                self._loaded_waveform = None
                self._current_loaded_assets = (None, sequence_name)
                
                # If channels are specified, validate and set active
                if to_ch is not None:
                    for ch in to_ch:
                        if (ch in self._analog_channels and ch not in self._active_analog_channels) or \
                           (ch in self._digital_channels and ch not in self._active_digital_channels):
                            self.log.warning(f"Channel {ch} is being loaded but is not active")
                
                self.log.info(f"Sequence '{sequence_name}' loaded")
                return 0
            else:
                self.log.error(f"Cannot load non-existent sequence: {sequence_name}")
                return -1

    def get_loaded_assets(self):
        """
        Retrieve the currently loaded assets (waveform and/or sequence).
        
        @return dict: dictionary with asset names for all channels
        """
        asset_dict = dict()
        
        if self._current_loaded_assets is not None:
            waveform_name, sequence_name = self._current_loaded_assets
            
            # If a waveform is loaded
            if waveform_name is not None:
                for ch in self._active_analog_channels:
                    asset_dict[ch] = waveform_name
                for ch in self._active_digital_channels:
                    asset_dict[ch] = waveform_name
            
            # If a sequence is loaded
            elif sequence_name is not None:
                for ch in self._active_analog_channels:
                    asset_dict[ch] = sequence_name
                for ch in self._active_digital_channels:
                    asset_dict[ch] = sequence_name
        
        return asset_dict

    def clear_all(self):
        """
        Clear all waveforms and sequences from the device memory.
        
        @return int: error code (0:OK, -1:error)
        """
        with self._thread_lock:
            # Stop any running sequence
            self.stop()
            
            # Clear all waveforms and sequences
            self._waveforms = {}
            self._sequences = {}
            self._loaded_waveform = None
            self._loaded_sequence = None
            self._current_loaded_assets = None
            
            self.log.info("All waveforms and sequences cleared")
            return 0

    def get_errors(self):
        """
        Get a list of errors/warnings of the pulser hardware.
        
        @return list: list of error/warning strings
        """
        # Since this is a simulated pulser, there are no hardware errors
        return []

    def _start_pulse_sequence(self):
        """
        Start playing the loaded waveform or sequence in a separate thread.
        """
        if self._is_running:
            self.log.warning("Pulse sequence already running")
            return
        
        # Reset flags and counters
        self._stop_playing.clear()
        self._is_running = True
        self._current_waveform_pos = 0
        self._current_sequence_pos = 0
        self._sequence_loop_count = 0
        
        # Start the pulse thread
        self._pulse_thread = threading.Thread(
            target=self._pulse_thread_target,
            args=()
        )
        self._pulse_thread.daemon = True
        self._pulse_thread.start()
        
        self.log.debug("Pulse sequence playback started")

    def _pulse_thread_target(self):
        """
        Thread target for simulating pulse sequence execution.
        """
        try:
            # Determine what we're playing
            if self._loaded_waveform is not None:
                # Playing a waveform
                self._play_waveform(self._loaded_waveform)
            elif self._loaded_sequence is not None:
                # Playing a sequence
                self._play_sequence(self._loaded_sequence)
            else:
                self.log.error("No waveform or sequence loaded to play")
        except Exception as e:
            self.log.error(f"Error in pulse thread: {str(e)}")
        finally:
            self._is_running = False
            self.log.debug("Pulse sequence playback ended")

    def _play_waveform(self, waveform_name):
        """
        Simulate playing a waveform by applying its effects to the NV simulator.
        
        @param str waveform_name: name of the waveform to play
        """
        if waveform_name not in self._waveforms:
            self.log.error(f"Cannot play non-existent waveform: {waveform_name}")
            return
            
        # Get the waveform data
        analog_samples, digital_samples = self._waveforms[waveform_name]
        
        # Get sample rate and calculate time per sample
        sample_rate = self.get_sample_rate()
        time_per_sample = 1.0 / sample_rate
        
        # Process the waveform and apply effects to the simulator
        for i in range(analog_samples.shape[0]):
            # Check if stop flag is set
            if self._stop_playing.is_set():
                break
                
            # Apply the sample effects to the simulator
            self._apply_pulse_sample(analog_samples[i], digital_samples[i], time_per_sample)
            
            # Update position
            self._current_waveform_pos = i
            
            # Sleep for a fraction of the actual time to speed up simulation
            # but still maintain the relative timing
            time.sleep(time_per_sample * 0.001)  # 1/1000 of real time
        
        # Wait a moment after the waveform completes
        time.sleep(0.01)

    def _play_sequence(self, sequence_name):
        """
        Simulate playing a sequence by applying its effects to the NV simulator.
        
        @param str sequence_name: name of the sequence to play
        """
        if sequence_name not in self._sequences:
            self.log.error(f"Cannot play non-existent sequence: {sequence_name}")
            return
            
        # Get the sequence data
        sequence_steps = self._sequences[sequence_name]
        
        # Process each step in the sequence
        for step_idx, (wfm_name, repetitions) in enumerate(sequence_steps):
            # Check if stop flag is set
            if self._stop_playing.is_set():
                break
                
            # Update sequence position
            self._current_sequence_pos = step_idx
            
            # Play the waveform for the specified number of repetitions
            for rep in range(repetitions):
                # Check if stop flag is set
                if self._stop_playing.is_set():
                    break
                    
                # Update loop count
                self._sequence_loop_count = rep
                
                # Play the waveform
                self._play_waveform(wfm_name)
        
        # Wait a moment after the sequence completes
        time.sleep(0.01)

    def _apply_pulse_sample(self, analog_sample, digital_sample, duration):
        """
        Apply the effects of a single pulse sample to the NV simulator.
        
        @param analog_sample: Analog values for this sample
        @param digital_sample: Digital values for this sample
        @param duration: Duration of this sample in seconds
        """
        # Map active channels to their values
        analog_values = {}
        for i, ch_name in enumerate(self._active_analog_channels):
            if i < len(analog_sample):
                analog_values[self._analog_channels[ch_name]] = analog_sample[i]
        
        digital_values = {}
        for i, ch_name in enumerate(self._active_digital_channels):
            if i < len(digital_sample):
                digital_values[self._digital_channels[ch_name]] = digital_sample[i] > 0
        
        # Apply effects to simulator based on channel values
        # Microwave control
        if 'MW_amplitude' in analog_values and 'MW_switch' in digital_values:
            mw_on = digital_values['MW_switch']
            if mw_on:
                # Use amplitude as power in normalized units (0-1)
                amplitude = analog_values['MW_amplitude']
                # Use preset frequency
                frequency = 2.87e9  # Default to NV zero-field splitting
                
                # Apply phase if available
                phase = 0.0
                if 'MW_phase' in analog_values:
                    phase = analog_values['MW_phase'] * 360.0  # Convert to degrees
                
                # Apply microwave pulse to simulator
                self._simulator.apply_pulse(frequency, amplitude, phase, duration)
            
        # Laser control
        if 'Laser' in digital_values:
            laser_on = digital_values['Laser']
            power = 1.0  # Default power
            
            # Apply laser to simulator
            self._simulator.apply_laser(power, laser_on, duration)

    def stop(self):
        """
        Stop the current pulse sequence execution and release any resources.
        """
        if not self._is_running:
            return
            
        # Set stop flag
        self._stop_playing.set()
        
        # Wait for thread to finish (with timeout)
        if self._pulse_thread is not None and self._pulse_thread.is_alive():
            self._pulse_thread.join(timeout=1.0)
        
        # Reset state
        self._is_running = False
        self.log.info("Pulse sequence stopped")