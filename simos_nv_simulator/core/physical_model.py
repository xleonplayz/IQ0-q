"""
Physical NV-Center Model for quantum simulation.

This module provides the core physical model for NV-center simulations,
implementing the quantum mechanical behavior and integrating with SimOS.
"""

import numpy as np
import threading
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

# Configure logging
logger = logging.getLogger(__name__)


class PhysicalNVModel:
    """
    Implements the physical model for NV-centers.
    
    This class forms the core of the quantum simulation, handling the
    quantum state, Hamiltonian, and time evolution of the NV-center system.
    It integrates with SimOS for accurate quantum mechanical calculations.
    
    Attributes:
        config (Dict[str, Any]): Configuration parameters for the model
        lock (threading.RLock): Thread lock for ensuring thread safety
        magnetic_field (np.ndarray): Current magnetic field vector [Bx, By, Bz] in Tesla
        mw_frequency (float): Microwave frequency in Hz
        mw_power (float): Microwave power in dBm
        mw_on (bool): Whether microwave excitation is on
        laser_power (float): Laser power in mW
        laser_on (bool): Whether laser excitation is on
        dt (float): Simulation time step in seconds
        nv_system: SimOS NV system object (placeholder until SimOS integration)
        current_state: Quantum state vector (placeholder until SimOS integration)
        simulation_thread: Thread for running simulations in background
        stop_simulation: Threading event for controlling simulation execution
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the NV-center physical model.
        
        Args:
            config: Optional configuration dictionary. If None, default configuration will be used.
                   When providing a partial config, any missing fields will be filled with defaults.
        """
        # Initialize with default configuration first
        default_config = self._default_config()
        
        # If config is provided, update defaults with custom values
        if config is not None:
            default_config.update(config)
            
        self.config = default_config
        
        # Initialize with a placeholder for the SimOS NV system
        # Actual initialization will be done when SimOS is integrated
        self.nv_system = None
        
        # Current quantum state (placeholder)
        self.current_state = None
        
        # Current magnetic field (initialize to zero field)
        self.magnetic_field = np.array([0.0, 0.0, 0.0])
        
        # Microwave parameters
        self.mw_frequency = self.config['zero_field_splitting']  # Default to ZFS
        self.mw_power = 0.0  # dBm
        self.mw_on = False
        
        # Laser parameters
        self.laser_power = 0.0  # mW
        self.laser_on = False
        
        # Thread safety lock
        self.lock = threading.RLock()
        
        # Simulation time step
        self.dt = self.config['simulation_timestep']  # Default time step
        
        # Simulation thread control
        self.simulation_thread = None
        self.stop_simulation = threading.Event()
        
        logger.debug("PhysicalNVModel initialized with configuration: %s", self.config)
    
    def _default_config(self) -> Dict[str, Any]:
        """
        Provide the default configuration for the NV model.
        
        Returns:
            Dictionary containing default configuration parameters
        
        Note:
            These parameters define the physical properties of the NV-center and the
            simulation environment. All frequency values are in Hz, time values in seconds,
            and magnetic field related values use Tesla.
        """
        return {
            # NV-center parameters
            'zero_field_splitting': 2.87e9,  # Hz (D) - Zero-field splitting parameter
            'strain': 5e6,  # Hz (E) - Strain splitting parameter
            'gyromagnetic_ratio': 2.8025e10,  # Hz/T - Gyromagnetic ratio (gamma)
            
            # Relaxation times
            'T1': 1e-3,  # s - Longitudinal relaxation time
            'T2': 1e-6,  # s - Transverse relaxation time
            
            # Optical properties
            'fluorescence_contrast': 0.3,  # Fluorescence contrast between ms=0 and ms=±1
            'optical_readout_fidelity': 0.93,  # Fidelity of optical state readout
            
            # Environment parameters
            'temperature': 300,  # K - Temperature of the environment
            
            # Hyperfine parameters (14N)
            'hyperfine_coupling_14n': 2.14e6,  # Hz - Hyperfine coupling to 14N
            'quadrupole_splitting_14n': -4.96e6,  # Hz - Nuclear quadrupole splitting
            
            # Simulation parameters
            'simulation_timestep': 1e-9,  # s (1 ns) - Time step for numerical integration
            'use_gpu': False  # Whether to use GPU acceleration for simulation
        }
    
    def set_magnetic_field(self, field_vector: Union[List[float], np.ndarray]) -> None:
        """
        Set the external magnetic field.
        
        Args:
            field_vector: 3D vector [Bx, By, Bz] representing the magnetic field in Tesla
            
        Note:
            The magnetic field affects the energy levels of the NV-center through Zeeman splitting.
            The field is stored as a numpy array and is thread-safe.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.set_magnetic_field([0, 0, 0.1])  # 0.1 Tesla along z-axis
        """
        with self.lock:
            self.magnetic_field = np.array(field_vector, dtype=float)
            logger.info(f"Magnetic field set to {self.magnetic_field} T")
    
    def get_magnetic_field(self) -> np.ndarray:
        """
        Get the current magnetic field.
        
        Returns:
            3D vector representing the magnetic field in Tesla
            
        Note:
            Returns a copy of the magnetic field vector to prevent external modification
            of the internal state. The method is thread-safe.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> field = model.get_magnetic_field()
            >>> print(field)
            [0. 0. 0.]
        """
        with self.lock:
            return self.magnetic_field.copy()
    
    def apply_microwave(self, frequency: float, power: float, on: bool = True) -> None:
        """
        Apply microwave excitation to the NV-center.
        
        Args:
            frequency: Microwave frequency in Hz
            power: Microwave power in dBm
            on: Whether to turn the microwave on (True) or off (False)
            
        Note:
            Microwave excitation is used to drive transitions between spin states.
            The frequency determines which transition is addressed, while the power
            controls the Rabi frequency (rate of oscillation between states).
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.apply_microwave(2.87e9, -10, True)  # Turn on MW at ZFS frequency
        """
        with self.lock:
            self.mw_frequency = frequency
            self.mw_power = power
            self.mw_on = on
            logger.info(f"Microwave set to {on}, frequency={frequency} Hz, power={power} dBm")
    
    def apply_laser(self, power: float, on: bool = True) -> None:
        """
        Apply laser excitation to the NV-center.
        
        Args:
            power: Laser power in mW
            on: Whether to turn the laser on (True) or off (False)
            
        Note:
            Laser excitation is used for state initialization and readout of the NV-center.
            Higher laser power generally leads to faster polarization but may cause heating.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.apply_laser(5.0, True)  # Turn on laser at 5mW
        """
        with self.lock:
            self.laser_power = power
            self.laser_on = on
            logger.info(f"Laser set to {on}, power={power} mW")
    
    def reset_state(self) -> None:
        """
        Reset the quantum state to the initial state.
        
        This resets both the quantum state and control parameters.
        
        Note:
            The initial state is typically the |ms=0⟩ state, which is achieved in real
            NV-centers through optical polarization. This method simulates the effect
            of applying a laser pulse and waiting for the system to polarize.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.apply_microwave(2.87e9, -10, True)  # Apply some excitation
            >>> model.reset_state()  # Reset to initial state
        """
        with self.lock:
            # Initial state will be properly implemented with SimOS integration
            # For now, just reset the control parameters
            self.mw_on = False
            self.laser_on = False
            logger.info("NV state reset")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Copy of the current configuration dictionary
            
        Note:
            Returns a copy of the configuration to prevent external modification
            of the internal configuration. The method is thread-safe.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> config = model.get_config()
            >>> print(config['zero_field_splitting'])
            2.87e+09
        """
        with self.lock:
            return self.config.copy()
    
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update configuration parameters.
        
        Args:
            config_updates: Dictionary containing parameters to update
            
        Note:
            This method allows partial updates to the configuration.
            Only the parameters specified in config_updates will be changed.
            The method is thread-safe, ensuring consistent configuration state.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.update_config({'zero_field_splitting': 2.88e9, 'T1': 1.5e-3})
        """
        with self.lock:
            self.config.update(config_updates)
            logger.info(f"Configuration updated with {config_updates}")
    
    def get_state_info(self) -> Dict[str, Any]:
        """
        Get information about the current state.
        
        This is a placeholder that will be expanded with SimOS integration.
        
        Returns:
            Dictionary with basic state information
            
        Note:
            This method provides a snapshot of the current simulator state,
            including control parameters and field values. It will be expanded
            in the future to include quantum state information when SimOS
            integration is complete.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.set_magnetic_field([0, 0, 0.1])
            >>> state_info = model.get_state_info()
            >>> print(state_info['magnetic_field'])
            [0.0, 0.0, 0.1]
        """
        with self.lock:
            return {
                'magnetic_field': self.magnetic_field.tolist(),
                'mw_frequency': self.mw_frequency,
                'mw_power': self.mw_power,
                'mw_on': self.mw_on,
                'laser_power': self.laser_power,
                'laser_on': self.laser_on
            }