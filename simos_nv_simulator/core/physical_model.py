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
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the NV-center physical model.
        
        Args:
            config: Optional configuration dictionary. If None, default configuration will be used.
        """
        self.config = config or self._default_config()
        
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
        """
        return {
            # NV-center parameters
            'zero_field_splitting': 2.87e9,  # Hz (D)
            'strain': 5e6,  # Hz (E)
            'gyromagnetic_ratio': 2.8025e10,  # Hz/T
            
            # Relaxation times
            'T1': 1e-3,  # s
            'T2': 1e-6,  # s
            
            # Optical properties
            'fluorescence_contrast': 0.3,
            'optical_readout_fidelity': 0.93,
            
            # Environment parameters
            'temperature': 300,  # K
            
            # Hyperfine parameters (14N)
            'hyperfine_coupling_14n': 2.14e6,  # Hz
            'quadrupole_splitting_14n': -4.96e6,  # Hz
            
            # Simulation parameters
            'simulation_timestep': 1e-9,  # s (1 ns)
            'use_gpu': False
        }
    
    def set_magnetic_field(self, field_vector: Union[List[float], np.ndarray]) -> None:
        """
        Set the external magnetic field.
        
        Args:
            field_vector: 3D vector [Bx, By, Bz] representing the magnetic field in Tesla
        """
        with self.lock:
            self.magnetic_field = np.array(field_vector, dtype=float)
            logger.info(f"Magnetic field set to {self.magnetic_field} T")
    
    def get_magnetic_field(self) -> np.ndarray:
        """
        Get the current magnetic field.
        
        Returns:
            3D vector representing the magnetic field in Tesla
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
        """
        with self.lock:
            self.laser_power = power
            self.laser_on = on
            logger.info(f"Laser set to {on}, power={power} mW")
    
    def reset_state(self) -> None:
        """
        Reset the quantum state to the initial state.
        
        This resets both the quantum state and control parameters.
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
        """
        with self.lock:
            return self.config.copy()
    
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update configuration parameters.
        
        Args:
            config_updates: Dictionary containing parameters to update
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