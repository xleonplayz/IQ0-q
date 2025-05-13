# -*- coding: utf-8 -*-

"""
Qudi facade for coordinating all simulator resources.
This class serves as the central manager for all simulator components used by Qudi hardware interfaces.

Copyright (c) 2023
"""

import os
import sys
import json
import logging
import importlib.util
from typing import Dict, Any, Optional, Union, List, Tuple

# Import the simulator model
script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(script_dir, 'model.py')
spec = importlib.util.spec_from_file_location("model", model_path)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
PhysicalNVModel = model.PhysicalNVModel


class QudiFacade:
    """
    Central manager for all simulator resources used by Qudi hardware interfaces.
    
    This class manages shared resources among the different hardware interfaces,
    ensuring consistent state and configuration across all simulator components.
    """
    
    _instance = None  # Singleton instance
    
    def __new__(cls, *args, **kwargs):
        """Ensure only one instance of QudiFacade exists."""
        if cls._instance is None:
            cls._instance = super(QudiFacade, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, config=None):
        """
        Initialize the facade with all simulator resources.
        
        @param config: Optional configuration dictionary or path to config file
        """
        # Avoid re-initialization if this is a singleton instance
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        # Set up logging
        self.log = logging.getLogger('QudiFacade')
        self.log.info("Initializing Qudi Facade for NV Simulator")
        
        # Load configuration if provided
        self._config = {}
        if config is not None:
            if isinstance(config, dict):
                self._config = config
            elif isinstance(config, str) and os.path.isfile(config):
                self.configure_from_file(config)
        
        # Initialize simulator components
        self._initialize_simulator()
        self._initialize_components()
        
        # Set up shared resources
        self._shared_resources = {}
        
        # Mark as initialized
        self._initialized = True
        
    def _initialize_simulator(self):
        """Initialize the core NV simulator with configuration."""
        # Extract simulator parameters from config
        simulator_params = self._config.get('simulator', {})
        
        # Create the simulator instance
        self.nv_model = PhysicalNVModel(**simulator_params)
        self.log.info("NV simulator model initialized")
        
    def _initialize_components(self):
        """Initialize additional simulator components."""
        # Initialize confocal simulator if available
        try:
            from ...confocal import ConfocalSimulator, DiamondLattice, FocusedLaserBeam
            
            # Get confocal configuration
            confocal_config = self._config.get('confocal', {})
            
            # Create diamond lattice with NV centers
            lattice_config = confocal_config.get('lattice', {})
            self.diamond_lattice = DiamondLattice(
                nv_density=lattice_config.get('nv_density', 1.0),
                size=lattice_config.get('size', (50e-6, 50e-6, 50e-6)),
                seed=lattice_config.get('seed', None)
            )
            
            # Configure laser beam
            laser_config = confocal_config.get('laser', {})
            self.laser_beam = FocusedLaserBeam(
                wavelength=laser_config.get('wavelength', 532e-9),
                numerical_aperture=laser_config.get('numerical_aperture', 0.8),
                power=laser_config.get('power', 1.0)
            )
            
            # Create confocal simulator
            self.confocal_simulator = ConfocalSimulator(
                diamond_lattice=self.diamond_lattice,
                laser_beam=self.laser_beam,
                nv_model=self.nv_model
            )
            self.log.info("Confocal simulator initialized")
            
        except ImportError:
            self.log.info("Confocal simulator module not available, skipping initialization")
            self.confocal_simulator = None
            self.diamond_lattice = None
            self.laser_beam = None
        
        # Initialize laser controller
        self.laser_controller = LaserController(self.nv_model)
        self.log.info("Laser controller initialized")
        
    def configure_from_file(self, config_path: str) -> None:
        """
        Load configuration from a file and apply settings.
        
        @param config_path: Path to the configuration file
        """
        if not os.path.isfile(config_path):
            self.log.error(f"Configuration file not found: {config_path}")
            return
            
        try:
            # Load configuration file
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Store configuration
            self._config = config
            self.log.info(f"Configuration loaded from {config_path}")
            
            # Re-initialize components with new config
            self._initialize_simulator()
            self._initialize_components()
            
        except Exception as e:
            self.log.error(f"Error loading configuration: {str(e)}")
    
    def get_nv_model(self):
        """
        Get the underlying NV simulator model.
        
        @return: PhysicalNVModel instance
        """
        return self.nv_model
    
    def get_confocal_simulator(self):
        """
        Get the confocal simulator if available.
        
        @return: ConfocalSimulator instance or None
        """
        return self.confocal_simulator
    
    def get_laser_controller(self):
        """
        Get the laser controller.
        
        @return: LaserController instance
        """
        return self.laser_controller
    
    def register_shared_resource(self, name: str, resource: Any) -> None:
        """
        Register a shared resource for use by multiple interfaces.
        
        @param name: Name of the resource
        @param resource: The resource object
        """
        self._shared_resources[name] = resource
        self.log.debug(f"Registered shared resource: {name}")
    
    def get_shared_resource(self, name: str) -> Any:
        """
        Get a shared resource by name.
        
        @param name: Name of the resource
        @return: The shared resource or None if not found
        """
        return self._shared_resources.get(name)
    
    def reset_simulator_state(self) -> None:
        """Reset the simulator to its initial state."""
        self.nv_model.reset_state()
        self.log.info("Simulator state reset")
        
    def apply_global_environment(self, 
                               magnetic_field: Optional[List[float]] = None,
                               temperature: Optional[float] = None,
                               strain: Optional[List[float]] = None) -> None:
        """
        Apply global environmental parameters to the simulator.
        
        @param magnetic_field: [Bx, By, Bz] magnetic field in Gauss
        @param temperature: Temperature in Kelvin
        @param strain: [Ex, Ey, Ez] strain components
        """
        if magnetic_field is not None:
            self.nv_model.set_magnetic_field(magnetic_field)
            self.log.info(f"Magnetic field set to {magnetic_field} Gauss")
            
        if temperature is not None:
            # Apply temperature effects if model supports it
            if hasattr(self.nv_model, 'set_temperature'):
                self.nv_model.set_temperature(temperature)
                self.log.info(f"Temperature set to {temperature} K")
            else:
                self.log.warning("Temperature setting not supported by this simulator version")
                
        if strain is not None:
            # Apply strain effects if model supports it
            if hasattr(self.nv_model, 'set_strain'):
                self.nv_model.set_strain(strain)
                self.log.info(f"Strain set to {strain}")
            else:
                self.log.warning("Strain setting not supported by this simulator version")


class LaserController:
    """
    Manages the laser control for the NV model.
    
    This class provides a common interface for laser operations used by multiple hardware interfaces.
    """
    
    def __init__(self, nv_model):
        """
        Initialize the laser controller.
        
        @param nv_model: The PhysicalNVModel instance
        """
        self.log = logging.getLogger('LaserController')
        self._nv_model = nv_model
        self._power = 1.0  # Default laser power (normalized units)
        self._is_on = False
        
    def on(self):
        """Turn the laser on and apply optical excitation to NV centers."""
        self._is_on = True
        self._nv_model.apply_laser(self._power, True)
        self.log.debug(f"Laser turned ON with power {self._power}")
        
    def off(self):
        """Turn the laser off."""
        self._is_on = False
        self._nv_model.apply_laser(self._power, False)
        self.log.debug("Laser turned OFF")
        
    def set_power(self, power: float):
        """
        Set the laser power.
        
        @param power: Laser power (normalized units)
        """
        self._power = power
        if self._is_on:
            self._nv_model.apply_laser(self._power, True)
        self.log.debug(f"Laser power set to {power}")
        
    def get_power(self) -> float:
        """
        Get the current laser power.
        
        @return: Current laser power (normalized units)
        """
        return self._power
        
    def is_on(self) -> bool:
        """
        Check if the laser is on.
        
        @return: True if the laser is on, False otherwise
        """
        return self._is_on