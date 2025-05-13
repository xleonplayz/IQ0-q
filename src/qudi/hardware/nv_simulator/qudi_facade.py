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
import logging
from pathlib import Path

# Ensure current_dir is defined for use in import paths
current_dir = os.path.dirname(os.path.abspath(__file__))

# Try to import the real Qudi Base class first
try:
    from qudi.core.module import Base
except ImportError:
    # Fall back to using the mock implementation if real Qudi is not available
    sys.path.insert(0, os.path.abspath(os.path.join(current_dir, '../../../..', 'sim', 'src')))
    try:
        from qudi_interface.mock.core import Base
        print("Using mock Base class from qudi_interface.mock.core")
    except ImportError:
        print("Failed to import mock Base class, creating a minimal version")
        # Create a minimal Base class if even the mock is not available
        class Base:
            def __init__(self, qudi_main_weakref=None, name=None, **kwargs):
                self.log = logging.getLogger(name or self.__class__.__name__)
                self._module_state = type('ModuleState', (), {'__call__': lambda self: 'idle', 'lock': lambda: None, 'unlock': lambda: None})()
                self.module_state = self._module_state
from qudi.core.configoption import ConfigOption
from qudi.interface.microwave_interface import MicrowaveInterface, MicrowaveConstraints
from qudi.util.enums import SamplingOutputMode

# Direct import from the physical location
# This is a hard-coded approach that should work regardless of installation
try:
    # Check for our wrapper module first
    current_dir = os.path.dirname(os.path.abspath(__file__))
    wrapper_path = os.path.join(current_dir, '..', 'dummy', 'sim', 'model_wrapper.py')
    wrapper_path = os.path.abspath(wrapper_path)
    
    if os.path.exists(wrapper_path):
        # Add directory to path
        wrapper_dir = os.path.dirname(wrapper_path)
        if wrapper_dir not in sys.path:
            sys.path.insert(0, wrapper_dir)
        
        # Use the wrapper module instead
        from model_wrapper import PhysicalNVModel
        print(f"Successfully imported PhysicalNVModel using wrapper at {wrapper_path}")
    else:
        # Fallback to direct import checks
        print(f"Wrapper module not found at {wrapper_path}, falling back to direct imports")
        
        # 1. First check if it's in the dummy/sim/src directory
        dummy_sim_path = os.path.join(current_dir, '..', 'dummy', 'sim', 'src')
        dummy_sim_path = os.path.abspath(dummy_sim_path)
        if dummy_sim_path not in sys.path:
            sys.path.insert(0, dummy_sim_path)
        
        # 2. Check the base directory path (original approach)
        base_dir = Path(current_dir).parent.parent.parent.parent
        sim_src_dir = os.path.join(str(base_dir), 'sim', 'src')
        if sim_src_dir not in sys.path:
            sys.path.insert(0, sim_src_dir)
            
        # 3. Add sim directory for module imports within model.py
        sim_dir = os.path.join(str(base_dir), 'sim')
        if sim_dir not in sys.path:
            sys.path.insert(0, sim_dir)
        
        # 4. Look in the local directory
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Try to import the model now that we've added multiple potential paths
        from model import PhysicalNVModel
    
except ImportError as e:
    # Provide a clear error message with all the paths we tried
    try:
        dummy_sim_path
    except NameError:
        dummy_sim_path = "Not checked (using wrapper)"
    
    try:
        sim_src_dir
    except NameError:
        sim_src_dir = "Not checked (using wrapper)"
    
    # Get all paths that were tried
    paths_tried = [
        wrapper_path,
        dummy_sim_path,
        sim_src_dir,
        current_dir
    ]
    
    error_msg = f"""
    NV Simulator Import Error:
    {e}
    
    Tried to find model.py or model_wrapper.py in the following paths:
    {os.linesep.join(f"{i+1}. {path}" for i, path in enumerate(paths_tried))}
    
    Current sys.path:
    {os.linesep.join(sys.path)}
    
    Please ensure the sim module is available by either:
    1. Running: pip install -e /path/to/IQO-q/sim
    2. Copying model.py to {current_dir}
    3. Checking that the path to dummy/sim/src is correct
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
        
        # Update shared state
        QudiFacade._shared_state['current_laser_power'] = self._power
        QudiFacade._shared_state['current_laser_on'] = self._is_on
        
    def on(self):
        """Turn on the laser."""
        self._is_on = True
        
        # Update shared state
        QudiFacade._shared_state['current_laser_on'] = True
        
        self.nv_model.set_laser_power(self._power)
        
    def off(self):
        """Turn off the laser."""
        self._is_on = False
        
        # Update shared state
        QudiFacade._shared_state['current_laser_on'] = False
        
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
        
        # Update shared state
        QudiFacade._shared_state['current_laser_power'] = power
        
        if self._is_on:
            self.nv_model.set_laser_power(power)
    
    def get_power(self):
        """Get the current laser power in mW."""
        return self._power


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
        
        # Update shared state with initial values
        QudiFacade._shared_state['current_mw_frequency'] = self._frequency
        QudiFacade._shared_state['current_mw_power'] = self._power
        QudiFacade._shared_state['current_mw_on'] = self._is_on
        
    def set_frequency(self, frequency):
        """
        Set the microwave frequency.
        
        @param frequency: float, frequency in Hz
        """
        print(f"[CRITICAL DEBUG] Setting MW frequency to {frequency/1e9:.6f} GHz")
        self._frequency = frequency
        
        # Update shared state
        QudiFacade._shared_state['current_mw_frequency'] = frequency
        
        if self._is_on:
            self._apply_microwave()
    
    def set_power(self, power):
        """
        Set the microwave power.
        
        @param power: float, power in dBm
        """
        print(f"[CRITICAL DEBUG] Setting MW power to {power} dBm")
        self._power = power
        
        # Update shared state
        QudiFacade._shared_state['current_mw_power'] = power
        
        if self._is_on:
            self._apply_microwave()
    
    def on(self):
        """Turn on the microwave."""
        print(f"[CRITICAL DEBUG] Turning MW ON at {self._frequency/1e9:.6f} GHz, {self._power} dBm")
        self._is_on = True
        
        # Update shared state
        QudiFacade._shared_state['current_mw_on'] = True
        
        self._apply_microwave()
    
    def off(self):
        """Turn off the microwave."""
        print(f"[CRITICAL DEBUG] Turning MW OFF")
        self._is_on = False
        
        # Update shared state
        QudiFacade._shared_state['current_mw_on'] = False
        
        self.nv_model.set_microwave_amplitude(0.0)
    
    def _apply_microwave(self):
        """Apply the microwave with current settings to the NV model."""
        # Convert dBm to amplitude for the simulator
        # P(dBm) = 10 * log10(P(mW))
        # P(mW) = 10^(P(dBm)/10)
        power_mw = 10**(self._power/10)
        amplitude = np.sqrt(power_mw) * 0.01  # Scaling factor for the model
        
        print(f"[CRITICAL DEBUG] Applying MW to NV model: {self._frequency/1e9:.6f} GHz, amp={amplitude}")
        self.nv_model.set_microwave_frequency(self._frequency)
        self.nv_model.set_microwave_amplitude(amplitude)
        
    # Accessor methods to get current state
    def get_frequency(self):
        """Get the current microwave frequency in Hz."""
        return self._frequency
    
    def get_power(self):
        """Get the current microwave power in dBm."""
        return self._power
    
    def is_on(self):
        """Check if the microwave is on."""
        return self._is_on


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


class QudiFacade(MicrowaveInterface):
    """Central manager for all simulator resources used by Qudi interfaces.
    Also implements MicrowaveInterface for ODMR experiments."""
    
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
    
    # Shared state for inter-module communication
    _shared_state = {
        'current_mw_frequency': 2.87e9,  # Hz
        'current_mw_power': -20.0,       # dBm
        'current_mw_on': False,          # MW on/off state
        'current_laser_power': 0.0,      # mW
        'current_laser_on': False,       # Laser on/off state
        'scanning_active': False,        # Is scanning in progress
        'current_scan_index': 0,         # Current index in scan
        'scan_frequencies': None,        # List of frequencies for scanning
    }
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern with reset capability for tests.
        """
        # Check if the environment variable is set for testing
        import os
        run_as_test = os.environ.get('QUDI_NV_TEST_MODE', '0') == '1'
        
        # For testing: completely new instance to avoid singleton conflicts
        if run_as_test:
            return super(QudiFacade, cls).__new__(cls)
        
        # Normal singleton behavior
        if cls._instance is None:
            cls._instance = super(QudiFacade, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (for testing)."""
        cls._instance = None
    
    def __init__(self, qudi_main_weakref=None, name=None, **kwargs):
        """Initialize the QudiFacade with optional Qudi main reference and name.
        
        @param qudi_main_weakref: Optional weakref to Qudi main object
        @param name: Optional name for this module
        """
        super().__init__(qudi_main_weakref=qudi_main_weakref, name=name, **kwargs)
        self._initialized = False
    
    def on_activate(self):
        """Initialization performed during activation of the module."""
        # Only initialize once
        if self._initialized:
            self.log.info("QudiFacade already initialized, returning")
            return
            
        self._initialized = True
        self.log.info("Initializing QudiFacade with parameters")
        
        try:
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
            
            self.log.info(f"Creating PhysicalNVModel with magnetic field: {self._magnetic_field} Gauss")
            
            # Create the NV model with parameters
            try:
                self.nv_model = PhysicalNVModel(**model_params)
                
                # Set magnetic field from config (convert Gauss to Tesla)
                b_field_tesla = [b * 1e-4 for b in self._magnetic_field]  # 1 G = 1e-4 T
                self.log.info(f"Setting magnetic field to {b_field_tesla} Tesla (from {self._magnetic_field} Gauss)")
                self.nv_model.set_magnetic_field(b_field_tesla)
                
                # Set temperature from config
                self.log.info(f"Setting temperature to {self._temperature} K")
                self.nv_model.set_temperature(self._temperature)
                self.log.info("NV model initialized successfully")
            except Exception as e:
                self.log.error(f"Error initializing NV model: {e}")
                self.log.warning("Creating simplified fallback NV model")
                
                # Create a simplified NV model as fallback
                from ..dummy.nv_simple_model import SimpleNVModel
                self.nv_model = SimpleNVModel(
                    magnetic_field=self._magnetic_field,
                    temperature=self._temperature,
                    zero_field_splitting=self._zero_field_splitting
                )
                self.log.info("Simplified NV model created")
            
            # Create controllers
            self.laser_controller = LaserController(self.nv_model)
            self.microwave_controller = MicrowaveController(self.nv_model)
            self.pulse_controller = PulseController(self.nv_model)
            self.confocal_simulator = ConfocalSimulator(self.nv_model)
            
            self.log.info("QudiFacade initialization complete")
        except Exception as e:
            self.log.error(f"Error during QudiFacade initialization: {str(e)}")
            self._initialized = False
            raise
        
        # Microwave interface implementation state variables
        self._is_scanning = False
        self._cw_power = 0.0
        self._cw_frequency = 2.87e9  # Default to zero-field splitting
        self._scan_power = 0.0
        self._scan_frequencies = None
        self._scan_mode = SamplingOutputMode.JUMP_LIST
        self._scan_sample_rate = 100.0
        
        # Create microwave constraints
        self._constraints = MicrowaveConstraints(
            power_limits=(-60, 10),
            frequency_limits=(2.7e9, 3.1e9),  # Range around NV zero-field splitting
            scan_size_limits=(2, 1001),
            sample_rate_limits=(0.1, 1000),
            scan_modes=(SamplingOutputMode.JUMP_LIST, SamplingOutputMode.EQUIDISTANT_SWEEP)
        )
        
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
        
        # Reset shared state
        self._shared_state['current_scan_index'] = 0
        self._shared_state['scanning_active'] = False
        self._shared_state['current_mw_on'] = False
        
        self.log.info("NV Simulator reset to initial state")
        
    @classmethod
    def force_reset_shared_state(cls):
        """Force reset of shared state (for testing when things get stuck).
        This is a special method for emergency cleanup.
        """
        cls._shared_state = {
            'current_mw_frequency': 2.87e9,  # Hz
            'current_mw_power': -20.0,       # dBm
            'current_mw_on': False,          # MW on/off state
            'current_laser_power': 0.0,      # mW
            'current_laser_on': False,       # Laser on/off state
            'scanning_active': False,        # Is scanning in progress
            'current_scan_index': 0,         # Current index in scan
            'scan_frequencies': None,        # List of frequencies for scanning
        }
        print("[CRITICAL] Forced reset of QudiFacade shared state!")
        
    # Methods for accessing shared state - these are key for inter-module communication
    
    def get_current_frequency(self):
        """Get the current microwave frequency in Hz.
        This is used by other modules to synchronize with microwave device.
        
        @return float: Current frequency in Hz
        """
        return self._shared_state['current_mw_frequency']
    
    def get_current_power(self):
        """Get the current microwave power in dBm.
        
        @return float: Current power in dBm
        """
        return self._shared_state['current_mw_power']
    
    def is_microwave_on(self):
        """Check if the microwave is currently on.
        
        @return bool: True if on, False if off
        """
        return self._shared_state['current_mw_on']
    
    def get_laser_power(self):
        """Get the current laser power in mW.
        
        @return float: Current laser power in mW
        """
        return self._shared_state['current_laser_power']
    
    def is_laser_on(self):
        """Check if the laser is currently on.
        
        @return bool: True if on, False if off
        """
        return self._shared_state['current_laser_on']
    
    def is_scanning(self):
        """Check if a frequency scan is in progress.
        
        @return bool: True if scanning, False otherwise
        """
        return self._shared_state['scanning_active']
    
    def get_current_scan_index(self):
        """Get the current index in the frequency scan.
        
        @return int: Current scan index
        """
        return self._shared_state['current_scan_index']
    
    def set_scanning_status(self, active, scan_index=None, frequencies=None):
        """Update the scanning status.
        
        @param bool active: Whether scanning is active
        @param int scan_index: Current index in scan
        @param list frequencies: List of scan frequencies
        """
        self._shared_state['scanning_active'] = active
        
        if scan_index is not None:
            self._shared_state['current_scan_index'] = scan_index
            
        if frequencies is not None:
            self._shared_state['scan_frequencies'] = frequencies

    # MicrowaveInterface implementation
    
    @property
    def constraints(self) -> MicrowaveConstraints:
        """The microwave constraints object for this device."""
        return self._constraints

    @property
    def is_scanning(self) -> bool:
        """Flag indicating if a scan is running at the moment."""
        return self._is_scanning

    @property
    def cw_power(self) -> float:
        """The currently configured CW microwave power in dBm."""
        return self._cw_power

    @property
    def cw_frequency(self) -> float:
        """The currently set CW microwave frequency in Hz."""
        return self._cw_frequency

    @property
    def scan_power(self) -> float:
        """The currently configured microwave power in dBm used for scanning."""
        return self._scan_power

    @property
    def scan_frequencies(self):
        """The currently configured microwave frequencies used for scanning."""
        return self._scan_frequencies

    @property
    def scan_mode(self) -> SamplingOutputMode:
        """The currently configured scan mode Enum."""
        return self._scan_mode

    @property
    def scan_sample_rate(self) -> float:
        """The currently configured scan sample rate in Hz."""
        return self._scan_sample_rate

    def off(self) -> None:
        """Switches off any microwave output (both scan and CW)."""
        self.microwave_controller.off()
        self._is_scanning = False
        
        if self.module_state() == 'locked':
            self.module_state.unlock()

    def set_cw(self, frequency: float, power: float) -> None:
        """Configure the CW microwave output."""
        # Validate parameters against constraints
        self._assert_cw_parameters_args(frequency, power)
        
        # Update internal state
        self._cw_frequency = frequency
        self._cw_power = power
        
        # Update the controller if we're in CW mode and running
        if self.module_state() == 'locked' and not self._is_scanning:
            self.microwave_controller.set_frequency(frequency)
            self.microwave_controller.set_power(power)

    def cw_on(self) -> None:
        """Switches on the CW microwave output."""
        # Check if we're idle
        if self.module_state() != 'idle':
            self.log.error('Cannot turn on CW microwave. Microwave output already active.')
            return
        
        # Configure and turn on the microwave
        self.microwave_controller.set_frequency(self._cw_frequency)
        self.microwave_controller.set_power(self._cw_power)
        self.microwave_controller.on()
        
        # Update state
        self._is_scanning = False
        self.module_state.lock()

    def configure_scan(self, power: float, frequencies, mode: SamplingOutputMode, sample_rate: float) -> None:
        """Configure a frequency scan."""
        # Validate the parameters
        self._assert_scan_configuration_args(power, frequencies, mode, sample_rate)
        
        # Store the configuration
        self._scan_power = power
        self._scan_mode = mode
        self._scan_sample_rate = sample_rate
        
        # Convert frequencies to the correct format if needed
        if mode == SamplingOutputMode.EQUIDISTANT_SWEEP:
            start, stop, num_points = frequencies
            self._scan_frequencies = np.linspace(start, stop, int(num_points))
        else:  # JUMP_LIST
            self._scan_frequencies = np.array(frequencies)

    def start_scan(self) -> None:
        """Switches on the preconfigured microwave scanning."""
        # Check if we're idle and have configured frequencies
        if self.module_state() != 'idle':
            self.log.error('Cannot start frequency scan. Microwave output already active.')
            return
            
        if self._scan_frequencies is None or len(self._scan_frequencies) == 0:
            self.log.error('No scan frequencies configured. Cannot start scan.')
            return
            
        # Start with the first frequency in the scan
        first_freq = self._scan_frequencies[0]
        self.microwave_controller.set_frequency(first_freq)
        self.microwave_controller.set_power(self._scan_power)
        self.microwave_controller.on()
        
        # Update state
        self._is_scanning = True
        self.module_state.lock()

    def reset_scan(self) -> None:
        """Reset currently running scan and return to start frequency."""
        if not self._is_scanning:
            return
            
        # Reset to the first frequency
        if len(self._scan_frequencies) > 0:
            first_freq = self._scan_frequencies[0]
            self.microwave_controller.set_frequency(first_freq)
            self.microwave_controller.set_power(self._scan_power)