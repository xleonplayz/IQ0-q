"""
Physical parameters module for NV center simulation.

This module provides a centralized management class for physical parameters
used throughout the simulator.
"""

import logging
import copy
import json
import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union

# Configure logging
logger = logging.getLogger(__name__)

class PhysicalParameters:
    """
    Centralized management of physical parameters used in the simulator.
    
    All physical parameters are stored with units and can be configured
    either through a config file or at runtime.
    """
    
    def __init__(self, config=None):
        """
        Initialize physical parameters with default values.
        
        Parameters
        ----------
        config : dict, optional
            Configuration dictionary with parameter values
        """
        # Load default values
        self._params = {
            # Generic RF parameters
            'rf_impedance': {'value': 50.0, 'unit': 'Ohm', 'description': 'RF circuit impedance'},
            'rf_coil_factor': {'value': 1e-4, 'unit': 'T/A', 'description': 'RF coil field per unit current'},
            
            # Magnetic field parameters
            'b0_field': {'value': 0.05, 'unit': 'T', 'description': 'Static magnetic field strength'},
            'b0_direction': {'value': [0, 0, 1], 'unit': 'vector', 'description': 'Static field direction'},
            
            # NV center parameters
            'zero_field_splitting': {'value': 2.87e9, 'unit': 'Hz', 'description': 'NV zero-field splitting'},
            'gyromagnetic_ratio_e': {'value': 28.024e9, 'unit': 'Hz/T', 'description': 'Electron gyromagnetic ratio'},
            
            # Optical parameters
            'collection_efficiency': {'value': 0.05, 'unit': 'dimensionless', 'description': 'Photon collection efficiency'},
            'fluorescence_lifetime': {'value': 12e-9, 'unit': 's', 'description': 'Excited state lifetime'},
            
            # Noise parameters
            'dark_count_rate': {'value': 200, 'unit': 'counts/s', 'description': 'APD dark count rate'},
            'electronic_noise': {'value': 10, 'unit': 'counts/s', 'description': 'Electronic noise standard deviation'},
            
            # Relaxation parameters
            't1_electron': {'value': 5e-3, 'unit': 's', 'description': 'Electron T1 relaxation time'},
            't2_electron': {'value': 500e-6, 'unit': 's', 'description': 'Electron T2 coherence time'},
            't2_star_electron': {'value': 1e-6, 'unit': 's', 'description': 'Electron T2* dephasing time'},
            
            # Temperature parameters
            'temperature': {'value': 300.0, 'unit': 'K', 'description': 'Operating temperature'},
            
            # Gyromagnetic ratios for common nuclear species
            'gyromagnetic_ratios': {
                'value': {
                    '1H': 42.577e6,   # Hz/T
                    '13C': 10.708e6,  # Hz/T
                    '14N': 3.077e6,   # Hz/T
                    '15N': -4.316e6,  # Hz/T
                    '29Si': -8.465e6, # Hz/T
                    '31P': 17.235e6,  # Hz/T
                },
                'unit': 'Hz/T',
                'description': 'Gyromagnetic ratios for nuclear spins'
            },
            
            # Spin quantum numbers
            'spin_quantum': {
                'value': {
                    '1H': 0.5,
                    '13C': 0.5,
                    '14N': 1.0,
                    '15N': 0.5,
                    '29Si': 0.5,
                    '31P': 0.5,
                    'NV': 1.0,  # NV electron spin is S=1
                },
                'unit': 'dimensionless',
                'description': 'Spin quantum numbers for different species'
            },
        }
        
        # Update with provided config
        if config is not None:
            self.update_from_config(config)
    
    def update_from_config(self, config):
        """
        Update parameters from a configuration dictionary.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary with parameter values
        """
        for key, value in config.items():
            if key in self._params:
                if isinstance(value, dict) and 'value' in value:
                    # Full parameter specification with value and possibly unit
                    self._params[key].update(value)
                else:
                    # Just the parameter value
                    self._params[key]['value'] = value
            else:
                # Add new parameter
                if isinstance(value, dict) and ('value' in value or 'unit' in value):
                    self._params[key] = value
                else:
                    self._params[key] = {'value': value, 'unit': 'unknown', 'description': 'User-added parameter'}
                logger.info(f"Added new parameter: {key}")
    
    def get(self, name, default=None):
        """
        Get a parameter value.
        
        Parameters
        ----------
        name : str
            Parameter name
        default : any, optional
            Default value if parameter not found
            
        Returns
        -------
        any
            Parameter value
        """
        param = self._params.get(name)
        if param is None:
            return default
        return param['value']
    
    def get_with_unit(self, name):
        """
        Get a parameter value with its unit.
        
        Parameters
        ----------
        name : str
            Parameter name
            
        Returns
        -------
        tuple
            (value, unit)
        """
        param = self._params.get(name)
        if param is None:
            return None, None
        return param['value'], param['unit']
    
    def set(self, name, value, unit=None, description=None):
        """
        Set a parameter value.
        
        Parameters
        ----------
        name : str
            Parameter name
        value : any
            Parameter value
        unit : str, optional
            Parameter unit
        description : str, optional
            Parameter description
        """
        if name in self._params:
            self._params[name]['value'] = value
            if unit is not None:
                self._params[name]['unit'] = unit
            if description is not None:
                self._params[name]['description'] = description
        else:
            # Create new parameter entry
            param = {'value': value}
            if unit is not None:
                param['unit'] = unit
            else:
                param['unit'] = 'unknown'
            if description is not None:
                param['description'] = description
            else:
                param['description'] = ''
            self._params[name] = param
    
    def export_config(self):
        """
        Export all parameters as a configuration dictionary.
        
        Returns
        -------
        dict
            Configuration dictionary
        """
        return {key: param.copy() for key, param in self._params.items()}
    
    def export_values_only(self):
        """
        Export all parameter values (without units and descriptions).
        
        Returns
        -------
        dict
            Parameter values dictionary
        """
        return {key: param['value'] for key, param in self._params.items()}
    
    def save_to_file(self, filename):
        """
        Save parameters to a JSON file.
        
        Parameters
        ----------
        filename : str
            Path to the file
        """
        try:
            with open(filename, 'w') as f:
                json.dump(self.export_config(), f, indent=2)
            logger.info(f"Parameters saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save parameters: {e}")
    
    def load_from_file(self, filename):
        """
        Load parameters from a JSON file.
        
        Parameters
        ----------
        filename : str
            Path to the file
        """
        try:
            with open(filename, 'r') as f:
                config = json.load(f)
            self.update_from_config(config)
            logger.info(f"Parameters loaded from {filename}")
        except Exception as e:
            logger.error(f"Failed to load parameters: {e}")
    
    def get_gyromagnetic_ratio(self, species):
        """
        Get the gyromagnetic ratio for a specific nuclear species.
        
        Parameters
        ----------
        species : str
            Nuclear species (e.g., '13C', '1H')
            
        Returns
        -------
        float
            Gyromagnetic ratio in Hz/T
        """
        gyromagnetic_ratios = self.get('gyromagnetic_ratios', {})
        return gyromagnetic_ratios.get(species, 0.0)
    
    def get_spin_quantum(self, species):
        """
        Get the spin quantum number for a specific species.
        
        Parameters
        ----------
        species : str
            Nuclear species (e.g., '13C', '1H') or 'NV' for the NV electron
            
        Returns
        -------
        float
            Spin quantum number (dimensionless)
        """
        spin_quantum = self.get('spin_quantum', {})
        return spin_quantum.get(species, 0.0)