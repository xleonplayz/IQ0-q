# -*- coding: utf-8 -*-

"""
This file contains the facade class that manages all NV simulator resources.

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

import os
import sys
import json
import numpy as np
import time
from pathlib import Path

from qudi.core.module import Base
from qudi.core.configoption import ConfigOption

# Direct import from the physical location
# This is a hard-coded approach that should work regardless of installation
try:
    # Get the base directory
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.parent.parent
    sim_src_dir = os.path.join(str(base_dir), 'sim', 'src')
    
    # Add to path
    if sim_src_dir not in sys.path:
        sys.path.insert(0, sim_src_dir)
    
    # Import directly from the module
    from model import PhysicalNVModel
except ImportError as e:
    # Provide a clear error message
    error_msg = f"""
    NV Simulator Import Error:
    {e}
    
    Please ensure the sim module is available by running:
    pip install -e C:\\Users\\qudi\\Desktop\\IQO\\IQO-q\\sim
    
    Alternative: Copy the model.py file to {os.path.dirname(os.path.abspath(__file__))}
    """
    raise ImportError(error_msg)

class LaserController:
    """Manages the laser excitation for NV centers."""
    
    def __init__(self, nv_model):
        """
        Initialize the laser controller.
        
        @param nv_model: PhysicalNVModel, the NV model to control
        """
        self.nv_model = nv_model
        self._is_on = False
        self._power = 1.0  # mW
        
    def on(self):
        """Turn on the laser."""
        self._is_on = True
        self.nv_model.set_laser_power(self._power)
        
    def off(self):
        """Turn off the laser."""
        self._is_on = False
        self.nv_model.set_laser_power(0.0)
        
    @property
    def is_on(self):
        """Return the laser state."""
        return self._is_on
    
    def set_power(self, power):
        """
        Set the laser power.
        
        @param power: float, laser power in mW
        """
        self._power = power
        if self._is_on:
            self.nv_model.set_laser_power(power)


class ConfocalSimulator:
    """Simulates a confocal microscope for NV centers."""
    
    def __init__(self, nv_model):
        """
        Initialize the confocal simulator.
        
        @param nv_model: PhysicalNVModel, the NV model to control
        """
        self.nv_model = nv_model
        self._position = np.array([0.0, 0.0, 0.0])  # (x, y, z) in µm
        
        # Define a 3D grid of NV centers (simplified for now)
        self.nv_positions = np.random.rand(10, 3) * 10 - 5  # 10 NVs in a 10x10x10 µm³ volume
        
        # Point spread function parameters (µm)
        self.psf_width_xy = 0.3  # lateral resolution
        self.psf_width_z = 0.7   # axial resolution
        
    def get_position(self):
        """
        Get the current scanner position.
        
        @return list: [x, y, z] position in µm
        """
        return self._position.copy()
        
    def set_position(self, x=None, y=None, z=None):
        """
        Set the scanner position.
        
        @param float x: x position in µm
        @param float y: y position in µm
        @param float z: z position in µm
        """
        if x is not None:
            self._position[0] = x
        if y is not None:
            self._position[1] = y
        if z is not None:
            self._position[2] = z
            
        # Update the NV model based on the closest NV center
        self._update_nv_model()
        
    def _update_nv_model(self):
        """Update the NV model based on the current position."""
        # Calculate distances to all NV centers
        distances = np.sqrt(np.sum((self.nv_positions - self._position)**2, axis=1))
        
        # Calculate PSF weights for each NV
        weights_xy = np.exp(-2 * ((self.nv_positions[:, 0] - self._position[0])**2 + 
                                   (self.nv_positions[:, 1] - self._position[1])**2) / 
                              (self.psf_width_xy**2))
        weights_z = np.exp(-2 * (self.nv_positions[:, 2] - self._position[2])**2 / 
                             (self.psf_width_z**2))
        psf_weights = weights_xy * weights_z
        
        # Set the total fluorescence scaling based on PSF weights
        # This will affect the photon count rates when reading out
        self.nv_model.set_collection_efficiency(np.sum(psf_weights))


class MicrowaveController:
    """Controls microwave interaction with the NV center."""
    
    def __init__(self, nv_model):
        """
        Initialize the microwave controller.
        
        @param nv_model: PhysicalNVModel, the NV model to control
        """
        self.nv_model = nv_model
        self._frequency = 2.87e9  # Hz
        self._power = 0.0  # dBm
        self._is_on = False
        
    def set_frequency(self, frequency):
        """
        Set the microwave frequency.
        
        @param frequency: float, frequency in Hz
        """
        self._frequency = frequency
        if self._is_on:
            self._apply_microwave()
    
    def set_power(self, power):
        """
        Set the microwave power.
        
        @param power: float, power in dBm
        """
        self._power = power
        if self._is_on:
            self._apply_microwave()
    
    def on(self):
        """Turn on the microwave."""
        self._is_on = True
        self._apply_microwave()
    
    def off(self):
        """Turn off the microwave."""
        self._is_on = False
        self.nv_model.set_microwave_amplitude(0.0)
    
    def _apply_microwave(self):
        """Apply the microwave with current settings to the NV model."""
        # Convert dBm to amplitude for the simulator
        # P(dBm) = 10 * log10(P(mW))
        # P(mW) = 10^(P(dBm)/10)
        power_mw = 10**(self._power/10)
        amplitude = np.sqrt(power_mw) * 0.01  # Scaling factor for the model
        
        self.nv_model.set_microwave_frequency(self._frequency)
        self.nv_model.set_microwave_amplitude(amplitude)


class PulseController:
    """Controls pulse sequences for the NV center."""
    
    def __init__(self, nv_model):
        """
        Initialize the pulse controller.
        
        @param nv_model: PhysicalNVModel, the NV model to control
        """
        self.nv_model = nv_model
        self.sequence = []
        self.current_step = 0
        
    def add_pulse(self, channel, start_time, duration, params=None):
        """
        Add a pulse to the sequence.
        
        @param channel: str, channel identifier ('laser', 'microwave', etc.)
        @param start_time: float, start time in seconds
        @param duration: float, duration in seconds
        @param params: dict, optional parameters (frequency, power, etc.)
        """
        self.sequence.append({
            'channel': channel,
            'start_time': start_time,
            'duration': duration,
            'params': params or {}
        })
        
    def reset_sequence(self):
        """Clear the pulse sequence."""
        self.sequence = []
        self.current_step = 0
        
    def run_sequence(self):
        """Run the pulse sequence and return the result."""
        # Sort sequence by start time
        self.sequence.sort(key=lambda x: x['start_time'])
        
        # Run the sequence
        results = []
        prev_time = 0
        
        for pulse in self.sequence:
            # Evolve to the start of this pulse
            wait_time = pulse['start_time'] - prev_time
            if wait_time > 0:
                self.nv_model.evolve(wait_time)
            
            # Apply the pulse
            channel = pulse['channel']
            duration = pulse['duration']
            params = pulse['params']
            
            if channel == 'laser':
                self.nv_model.set_laser_power(params.get('power', 1.0))
                self.nv_model.evolve(duration)
                self.nv_model.set_laser_power(0.0)
                
                # If this is a readout pulse, record the result
                if params.get('readout', False):
                    count_rate = self.nv_model.get_fluorescence_rate()
                    photon_count = np.random.poisson(count_rate * duration)
                    results.append(photon_count)
                    
            elif channel == 'microwave':
                freq = params.get('frequency', 2.87e9)
                power = params.get('power', 0.0)
                power_mw = 10**(power/10)
                amplitude = np.sqrt(power_mw) * 0.01
                
                self.nv_model.set_microwave_frequency(freq)
                self.nv_model.set_microwave_amplitude(amplitude)
                self.nv_model.evolve(duration)
                self.nv_model.set_microwave_amplitude(0.0)
                
            prev_time = pulse['start_time'] + duration
        
        return np.array(results)


class QudiFacade(Base):
    """Central manager for all simulator resources used by Qudi interfaces."""
    
    # Config options using proper Qudi format
    _magnetic_field = ConfigOption('magnetic_field', default=[0, 0, 0], missing='warn')
    _temperature = ConfigOption('temperature', default=300, missing='warn')
    _zero_field_splitting = ConfigOption('zero_field_splitting', default=2.87e9, missing='warn')
    _gyromagnetic_ratio = ConfigOption('gyromagnetic_ratio', default=2.8025e10, missing='warn')
    _t1 = ConfigOption('t1', default=5.0e-3, missing='warn')
    _t2 = ConfigOption('t2', default=1.0e-5, missing='warn')
    _thread_safe = ConfigOption('thread_safe', default=True, missing='warn')
    _memory_management = ConfigOption('memory_management', default=True, missing='warn')
    _optimize_performance = ConfigOption('optimize_performance', default=True, missing='warn')
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(QudiFacade, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, **kwargs):
        """Initialize the QudiFacade."""
        super().__init__(**kwargs)
        self._initialized = False
    
    def on_activate(self):
        """Initialization performed during activation of the module."""
        # Only initialize once
        if self._initialized:
            return
            
        self._initialized = True
        
        # Create the NV model with options from config
        model_params = {
            'zero_field_splitting': self._zero_field_splitting,
            'gyromagnetic_ratio': self._gyromagnetic_ratio,
            't1': self._t1,
            't2': self._t2,
            'thread_safe': self._thread_safe,
            'memory_management': self._memory_management,
            'optimize_performance': self._optimize_performance
        }
        
        self.nv_model = PhysicalNVModel(**model_params)
        
        # Set magnetic field from config
        self.nv_model.set_magnetic_field(self._magnetic_field)
        
        # Set temperature from config
        self.nv_model.set_temperature(self._temperature)
        
        # Create controllers
        self.laser_controller = LaserController(self.nv_model)
        self.microwave_controller = MicrowaveController(self.nv_model)
        self.pulse_controller = PulseController(self.nv_model)
        self.confocal_simulator = ConfocalSimulator(self.nv_model)
        
        # State variables
        self.is_running = False
        
        self.log.info("NV Simulator Facade activated")
    
    def on_deactivate(self):
        """Clean-up performed during deactivation of the module."""
        # Nothing specific to clean up
        self.is_running = False
        self.log.info("NV Simulator Facade deactivated")
        
    def configure_from_file(self, config_path):
        """
        Load configuration from file and apply settings.
        
        @param config_path: str, path to the configuration file
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Re-apply configuration
        if 'magnetic_field' in config:
            self.nv_model.set_magnetic_field(config['magnetic_field'])
        
        if 'temperature' in config:
            self.nv_model.set_temperature(config['temperature'])
            
        self.log.info(f"Configuration loaded from {config_path}")
    
    def reset(self):
        """Reset all simulator components."""
        # Reset the NV model to ground state
        self.nv_model.reset_state()
        
        # Turn off all controllers
        self.laser_controller.off()
        self.microwave_controller.off()
        
        # Reset the pulse sequence
        self.pulse_controller.reset_sequence()
        
        # Reset the confocal position
        self.confocal_simulator.set_position(0, 0, 0)
        
        self.log.info("NV Simulator reset to initial state")