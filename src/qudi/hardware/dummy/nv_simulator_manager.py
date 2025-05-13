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
import traceback
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
            self._use_simulator = use_simulator
            
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
            
            logger.info(f"Initializing NV simulator with magnetic field {magnetic_field} G, " 
                       f"temperature {temperature} K, ZFS {zero_field_splitting/1e9} GHz")
            
            # Initialize the NV model
            try:
                # Try to import the model wrapper
                try:
                    # Import model from sim directory
                    logger.debug("Attempting to import PhysicalNVModel from sim wrapper")
                    try:
                        from sim.model_wrapper import import_model
                        PhysicalNVModel = import_model()
                        logger.info("Successfully imported PhysicalNVModel using wrapper at " + 
                                   os.path.join(os.path.dirname(__file__), 'sim', 'model_wrapper.py'))
                    except ImportError:
                        # Direct import attempt
                        logger.debug("Direct import of model.py")
                        from model import PhysicalNVModel
                    
                    # Convert Gauss to Tesla
                    b_field_tesla = [b * 1e-4 for b in magnetic_field]  # 1 G = 1e-4 T
                    
                    # Create the NV model
                    self.nv_model = PhysicalNVModel(optics=True, nitrogen=False)
                    
                    # Set parameters
                    self.nv_model.set_magnetic_field(b_field_tesla)
                    self.nv_model.set_temperature(temperature)
                    self.nv_model.config["d_gs"] = zero_field_splitting
                    
                    logger.info(f"NV simulator initialized with PhysicalNVModel")
                    self.nv_model.reset_state()
                    
                except Exception as e:
                    logger.warning(f"SimOS state reset failed: {e}, using fallback model")
                    
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
            try:
                # Generate frequency array
                frequencies = np.linspace(freq_min, freq_max, n_points)
                
                # Use the simulator's ODMR function
                logger.debug(f"Simulating ODMR from {freq_min/1e9:.4f} GHz to {freq_max/1e9:.4f} GHz with {n_points} points")
                
                # Check NV simulator type
                nv_type = type(self.nv_model).__name__
                logger.debug(f"Using NV model: {nv_type}")
                
                # Get ODMR data from simulator
                try:
                    result = self.nv_model.simulate_odmr(freq_min, freq_max, n_points, mw_power=-10.0)
                    
                    if hasattr(result, 'frequencies') and hasattr(result, 'signal'):
                        # Model returned a container object
                        logger.debug("ODMR simulation successful with container result")
                        logger.debug(f"Signal min: {np.min(result.signal):.1f}, max: {np.max(result.signal):.1f}")
                        
                        # Check for NaN or infinity values
                        if np.any(np.isnan(result.signal)) or np.any(np.isinf(result.signal)):
                            logger.warning("ODMR signal contains NaN or Inf values - replacing with zeros")
                            result.signal = np.nan_to_num(result.signal, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        return {
                            'frequencies': result.frequencies,
                            'signal': result.signal
                        }
                    else:
                        # Model returned a dictionary or other format
                        logger.debug("ODMR simulation successful with dictionary-like result")
                        if isinstance(result, dict):
                            logger.debug(f"Result keys: {list(result.keys())}")
                            if 'frequencies' in result and 'signal' in result:
                                return {
                                    'frequencies': result['frequencies'],
                                    'signal': result['signal']
                                }
                
                except Exception as e:
                    logger.warning(f"NV model's simulate_odmr failed, will calculate point by point: {e}")
                    # If the model doesn't have a simulate_odmr function, calculate point by point
                    
                # Fallback: Manual ODMR calculation
                logger.debug("Falling back to manual ODMR calculation")
                signal = np.zeros(n_points)
                
                # Save current microwave state
                current_mw_on = self._mw_on
                current_mw_freq = self._mw_frequency
                current_mw_power = self._mw_power
                
                # Turn on laser if needed for measurement
                if not self._laser_on:
                    self.set_laser(1.0, True)
                
                # Calculate point by point
                for i, freq in enumerate(frequencies):
                    # Set MW to the current frequency
                    self.set_microwave(freq, -10.0, True)  # Standard ODMR power
                    
                    # Get fluorescence signal
                    signal[i] = self.nv_model.get_fluorescence_rate()
                
                # Restore original MW and laser state
                self.set_microwave(current_mw_freq, current_mw_power, current_mw_on)
                if not self._laser_on:
                    self.set_laser(0.0, False)
                
                logger.debug(f"Manual ODMR calculation complete, signal min: {np.min(signal):.1f}, max: {np.max(signal):.1f}")
                
                return {
                    'frequencies': frequencies,
                    'signal': signal
                }
            except Exception as e:
                logger.error(f"Failed to simulate ODMR: {e}")
                logger.error(traceback.format_exc())
                
                # Last resort fallback - generate synthetic ODMR
                logger.warning("Using synthetic ODMR data generation as last resort")
                frequencies = np.linspace(freq_min, freq_max, n_points)
                
                # Calculate resonance frequencies
                resonance_freq = self._zero_field_splitting  # Zero-field splitting (Hz)
                
                # Calculate field strength
                field_strength_gauss = np.linalg.norm(self._magnetic_field)
                
                # Zeeman splitting (~2.8 MHz/G)
                zeeman_shift = 2.8e6 * field_strength_gauss  # field in G, shift in Hz
                
                logger.debug(f"Generating synthetic ODMR with ZFS={resonance_freq/1e9:.4f} GHz, "
                          f"field={field_strength_gauss:.1f} G, Zeeman={zeeman_shift/1e6:.1f} MHz")
                
                # Resonance dips
                dip1_center = resonance_freq - zeeman_shift
                dip2_center = resonance_freq + zeeman_shift
                
                # Signal parameters
                linewidth = 5e6  # 5 MHz linewidth
                contrast = 0.3  # 30% contrast
                baseline = 100000.0  # Base count rate
                
                # Generate signal with Lorentzian dips
                signal = np.ones(n_points) * baseline
                for i, freq in enumerate(frequencies):
                    # Lorentzian dips
                    dip1 = contrast * baseline * linewidth**2 / ((freq - dip1_center)**2 + linewidth**2)
                    dip2 = contrast * baseline * linewidth**2 / ((freq - dip2_center)**2 + linewidth**2)
                    
                    # Apply dips
                    signal[i] = baseline - dip1 - dip2
                    
                    # Add noise
                    signal[i] += np.random.normal(0, 0.01 * baseline)
                
                return {
                    'frequencies': frequencies,
                    'signal': signal
                }
    
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