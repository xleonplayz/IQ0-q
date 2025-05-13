#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains a singleton manager class for integrating the SimOS NV center simulator
with the Qudi dummy hardware modules.

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

import os
import sys
import numpy as np
import threading
import logging
from typing import Optional, Dict, Any, Tuple, List, Union

# Configure logging
logger = logging.getLogger(__name__)

class NVSimulatorManager:
    """Singleton manager class for the NV center simulator.
    
    This class provides a central access point to the NV center simulator
    for all dummy modules. It ensures that only one instance of the simulator
    is created and that all modules access the same simulator instance.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        """Implement the singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(NVSimulatorManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, 
                 magnetic_field: List[float] = None, 
                 temperature: float = 300, 
                 zero_field_splitting: float = 2.87e9,
                 use_simulator: bool = True):
        """Initialize the NV simulator manager.
        
        Args:
            magnetic_field: Magnetic field vector in Gauss [Bx, By, Bz]
            temperature: Temperature in Kelvin
            zero_field_splitting: Zero-field splitting in Hz
            use_simulator: Whether to use the simulator or simple model
        """
        with self._lock:
            # Only initialize once
            if hasattr(self, '_initialized') and self._initialized:
                return
                
            # Try to import the NV simulator model
            sim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'sim', 'src'))
            if sim_path not in sys.path:
                sys.path.insert(0, sim_path)
                
            self._initialized = True
            
            # Default magnetic field (500 G along z-axis)
            if magnetic_field is None:
                magnetic_field = [0, 0, 500]  # Gauss
                
            self._magnetic_field = magnetic_field
            self._temperature = temperature
            self._zero_field_splitting = zero_field_splitting
            
            # Current MW parameters
            self._mw_frequency = 2.87e9  # Hz
            self._mw_power = 0.0  # dBm
            self._mw_on = False
            
            # Current laser parameters
            self._laser_power = 0.0  # mW
            self._laser_on = False
            
            # Initialize the NV model
            try:
                # Import the PhysicalNVModel class
                try:
                    from model import PhysicalNVModel
                    
                    # Convert Gauss to Tesla
                    b_field_tesla = [b * 1e-4 for b in magnetic_field]  # 1 G = 1e-4 T
                    
                    # Create the NV model
                    self.nv_model = PhysicalNVModel(optics=True, nitrogen=False)
                    
                    # Set parameters
                    self.nv_model.set_magnetic_field(b_field_tesla)
                    self.nv_model.set_temperature(temperature)
                    self.nv_model.config["d_gs"] = zero_field_splitting
                    
                    logger.info(f"NV simulator initialized with magnetic field {magnetic_field} G, " 
                               f"temperature {temperature} K, ZFS {zero_field_splitting/1e9} GHz")
                except Exception as e:
                    logger.warning(f"PhysicalNVModel initialization failed: {e}")
                    logger.info("Falling back to SimpleNVModel")
                    
                    # Import the simplified model
                    from nv_simple_model import SimpleNVModel
                    
                    # Create a simple NV model
                    self.nv_model = SimpleNVModel(
                        magnetic_field=magnetic_field,
                        temperature=temperature,
                        zero_field_splitting=zero_field_splitting
                    )
                    
                    logger.info(f"SimpleNVModel initialized with magnetic field {magnetic_field} G")
            except Exception as e:
                raise RuntimeError(f"Failed to initialize any NV model: {e}")
    
    def set_magnetic_field(self, field_gauss: List[float]):
        """Set the magnetic field vector.
        
        Args:
            field_gauss: Magnetic field vector in Gauss [Bx, By, Bz]
        """
        with self._lock:
            self._magnetic_field = field_gauss
            
            if self._use_simulator:
                try:
                    # Convert Gauss to Tesla
                    b_field_tesla = [b * 1e-4 for b in field_gauss]  # 1 G = 1e-4 T
                    self.nv_model.set_magnetic_field(b_field_tesla)
                    logger.debug(f"NV simulator magnetic field set to {field_gauss} G")
                except Exception as e:
                    logger.warning(f"Failed to set magnetic field: {e}")
    
    def set_microwave(self, frequency: float, power: float, on: bool = True):
        """Set the microwave parameters.
        
        Args:
            frequency: Microwave frequency in Hz
            power: Microwave power in dBm
            on: Whether the microwave is on
        """
        with self._lock:
            self._mw_frequency = frequency
            self._mw_power = power
            self._mw_on = on
            
            if self._use_simulator:
                try:
                    if on:
                        # Convert dBm to amplitude
                        power_mw = 10**(power/10)
                        amplitude = np.sqrt(power_mw) * 0.01
                        
                        # Set microwave parameters in NV model
                        self.nv_model.set_microwave_frequency(frequency)
                        self.nv_model.set_microwave_amplitude(amplitude)
                        logger.debug(f"NV simulator microwave set to {frequency/1e9} GHz, {power} dBm")
                    else:
                        self.nv_model.set_microwave_amplitude(0.0)
                        logger.debug("NV simulator microwave turned off")
                except Exception as e:
                    logger.warning(f"Failed to set microwave parameters: {e}")
    
    def set_laser(self, power: float, on: bool = True):
        """Set the laser parameters.
        
        Args:
            power: Laser power in mW
            on: Whether the laser is on
        """
        with self._lock:
            self._laser_power = power
            self._laser_on = on
            
            if self._use_simulator:
                try:
                    if on:
                        self.nv_model.set_laser_power(power)
                        logger.debug(f"NV simulator laser power set to {power} mW")
                    else:
                        self.nv_model.set_laser_power(0.0)
                        logger.debug("NV simulator laser turned off")
                except Exception as e:
                    logger.warning(f"Failed to set laser power: {e}")
    
    def get_odmr_signal(self, frequency: float) -> float:
        """Get the ODMR signal at a specific frequency.
        
        Args:
            frequency: Microwave frequency in Hz
            
        Returns:
            float: Fluorescence signal in counts/s
        """
        with self._lock:
            try:
                # Save current MW state
                current_mw_on = self._mw_on
                current_mw_freq = self._mw_frequency
                current_mw_power = self._mw_power
                
                # Set MW to the requested frequency temporarily
                self.set_microwave(frequency, -10.0, True)  # Standard ODMR power
                
                # Get fluorescence signal
                # Ensure laser is on for measurement
                if not self._laser_on:
                    self.set_laser(1.0, True)  # Turn on with default power
                    signal = self.nv_model.get_fluorescence_rate()
                    self.set_laser(0.0, False)  # Return to off state
                else:
                    signal = self.nv_model.get_fluorescence_rate()
                
                # Restore original MW state
                self.set_microwave(current_mw_freq, current_mw_power, current_mw_on)
                
                return signal
            except Exception as e:
                raise RuntimeError(f"Failed to get ODMR signal from simulator: {e}")
    
    def _calculate_odmr_signal(self, frequency: float) -> float:
        """Calculate ODMR signal using a simple model.
        
        Args:
            frequency: Microwave frequency in Hz
            
        Returns:
            float: Fluorescence signal in counts/s
        """
        # Calculate resonance frequencies
        resonance_freq = self._zero_field_splitting  # Zero-field splitting (Hz)
        
        # Calculate field strength
        field_strength_gauss = np.linalg.norm(self._magnetic_field)
        
        # Zeeman splitting (~2.8 MHz/G)
        zeeman_shift = 2.8e6 * field_strength_gauss  # field in G, shift in Hz
        
        # Resonance dips
        dip1_center = resonance_freq - zeeman_shift
        dip2_center = resonance_freq + zeeman_shift
        
        # Signal parameters
        linewidth = 5e6  # 5 MHz linewidth
        contrast = 0.3  # 30% contrast
        baseline = 1.0
        
        # Lorentzian dips
        dip1 = contrast * linewidth**2 / ((frequency - dip1_center)**2 + linewidth**2)
        dip2 = contrast * linewidth**2 / ((frequency - dip2_center)**2 + linewidth**2)
        
        # Combine dips and scale to counts/s
        base_rate = 100000.0  # 100k counts/s
        signal = base_rate * (baseline - dip1 - dip2)
        
        # Add noise
        noise = np.random.normal(0, 0.02 * base_rate)
        
        return signal + noise
    
    def simulate_odmr(self, freq_min: float, freq_max: float, n_points: int) -> Dict[str, np.ndarray]:
        """Simulate an ODMR scan across a frequency range.
        
        Args:
            freq_min: Minimum frequency in Hz
            freq_max: Maximum frequency in Hz
            n_points: Number of frequency points
            
        Returns:
            Dict: Dictionary with 'frequencies' and 'signal' arrays
        """
        with self._lock:
            frequencies = np.linspace(freq_min, freq_max, n_points)
            
            try:
                # Use the simulator's ODMR function
                result = self.nv_model.simulate_odmr(freq_min, freq_max, n_points)
                return {
                    'frequencies': result.frequencies,
                    'signal': result.signal
                }
            except Exception as e:
                raise RuntimeError(f"Failed to simulate ODMR with simulator: {e}")
    
    def get_fluorescence_rate(self) -> float:
        """Get the current fluorescence rate.
        
        Returns:
            float: Fluorescence rate in counts/s
        """
        with self._lock:
            try:
                if not self._laser_on:
                    return 0.0
                    
                return self.nv_model.get_fluorescence_rate()
            except Exception as e:
                raise RuntimeError(f"Failed to get fluorescence rate from simulator: {e}")
                
    def reset(self):
        """Reset the NV state."""
        with self._lock:
            try:
                self.nv_model.reset_state()
                logger.debug("NV simulator state reset")
            except Exception as e:
                raise RuntimeError(f"Failed to reset NV state: {e}")
            
    def evolve(self, duration: float):
        """Evolve the NV state for a specified duration.
        
        Args:
            duration: Evolution time in seconds
        """
        with self._lock:
            try:
                self.nv_model.evolve(duration)
                logger.debug(f"NV simulator state evolved for {duration} s")
            except Exception as e:
                raise RuntimeError(f"Failed to evolve NV state: {e}")