#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains a simplified NV center model for use when the full
SimOS quantum simulator is not available.

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

import numpy as np
import time
import logging
import threading

# Configure logging
logger = logging.getLogger(__name__)

class SimpleNVModel:
    """A simplified NV center model for use when the full SimOS quantum simulator is not available."""
    
    def __init__(self, magnetic_field=None, temperature=300, zero_field_splitting=2.87e9):
        """Initialize the simplified NV model.
        
        Args:
            magnetic_field: Magnetic field in Gauss [Bx, By, Bz]
            temperature: Temperature in Kelvin
            zero_field_splitting: Zero-field splitting in Hz
        """
        # Store parameters
        if magnetic_field is None:
            magnetic_field = [0, 0, 500]  # Default to 500 G along z
        
        # Convert Gauss to Tesla
        self.b_field = np.array([b * 1e-4 for b in magnetic_field])  # 1 G = 1e-4 T
        self.temperature = temperature
        self.config = {
            'd_gs': zero_field_splitting,  # Zero-field splitting (Hz)
            'gyro_e': 28.025e9,  # Gyromagnetic ratio (Hz/T)
            't1': 5.0e-3,    # T1 relaxation time (s)
            't2': 1.0e-5     # T2 dephasing time (s)
        }
        
        # Initialize state
        self.state = np.zeros(3)
        self.state[0] = 1.0  # |ms=0⟩ state
        
        # Runtime parameters
        self._collection_efficiency = 1.0
        self._microwave_frequency = zero_field_splitting
        self._microwave_amplitude = 0.0
        self._laser_power = 0.0
        
        # Threading
        self.lock = threading.RLock()
        
        logger.info(f"Simplified NV Model initialized with magnetic field {magnetic_field} G")
        
    def reset_state(self):
        """Reset the NV state to the ground state |0⟩."""
        with self.lock:
            self.state = np.zeros(3)
            self.state[0] = 1.0  # |ms=0⟩ state
            logger.debug("NV state reset to |0⟩")
            
    def set_magnetic_field(self, field_tesla):
        """Set the magnetic field vector.
        
        Args:
            field_tesla: Magnetic field vector in Tesla [Bx, By, Bz]
        """
        with self.lock:
            self.b_field = np.array(field_tesla, dtype=float)
            logger.debug(f"Magnetic field set to {self.b_field} T")
            
    def set_temperature(self, temperature):
        """Set the temperature.
        
        Args:
            temperature: Temperature in Kelvin
        """
        with self.lock:
            self.temperature = float(temperature)
            logger.debug(f"Temperature set to {temperature} K")
            
    def set_laser_power(self, power):
        """Set the laser power for optical pumping.
        
        Args:
            power: Laser power in mW
        """
        with self.lock:
            self._laser_power = float(power)
            
            # Pumping to ms=0 state if laser is on
            if power > 0:
                # Simplified optical pumping: exponential approach to ms=0 state
                pump_rate = power * 1e6  # Adjust for reasonable timescale
                self.state[0] = 1.0 - (1.0 - self.state[0]) * np.exp(-pump_rate * 1e-6)
                self.state[1] = (1.0 - self.state[0]) / 2
                self.state[2] = (1.0 - self.state[0]) / 2
                
    def set_microwave_frequency(self, frequency):
        """Set the microwave drive frequency.
        
        Args:
            frequency: Microwave frequency in Hz
        """
        with self.lock:
            self._microwave_frequency = float(frequency)
            
    def set_microwave_amplitude(self, amplitude):
        """Set the microwave drive amplitude.
        
        Args:
            amplitude: Microwave amplitude (relative units)
        """
        with self.lock:
            self._microwave_amplitude = float(amplitude)
            
    def set_collection_efficiency(self, efficiency):
        """Set the fluorescence collection efficiency.
        
        Args:
            efficiency: Collection efficiency (0.0 to 1.0)
        """
        with self.lock:
            self._collection_efficiency = float(efficiency)
            
    def evolve(self, duration):
        """Evolve the quantum state for a specified duration.
        
        Args:
            duration: Time to evolve in seconds
        """
        with self.lock:
            if duration <= 0:
                return
                
            # Simplified evolution model
            if self._microwave_amplitude > 0:
                # Magnetic field magnitude
                b_magnitude = np.linalg.norm(self.b_field)
                
                # Calculate resonance frequencies
                gyro = self.config["gyro_e"]
                zeeman_shift = gyro * b_magnitude
                
                # Resonance frequencies
                dip1_center = self.config["d_gs"] - zeeman_shift  # ms=0 to ms=-1 transition
                dip2_center = self.config["d_gs"] + zeeman_shift  # ms=0 to ms=+1 transition
                
                # Check if microwave frequency is near resonance
                linewidth = 5e6  # 5 MHz linewidth
                detuning1 = self._microwave_frequency - dip1_center
                detuning2 = self._microwave_frequency - dip2_center
                
                # Rabi frequency (depends on amplitude)
                rabi_freq = 10e6 * self._microwave_amplitude  # Rabi frequency in Hz
                
                # Simplified Rabi oscillation model
                if abs(detuning1) < 10*linewidth:
                    # Near ms=0 to ms=-1 resonance
                    # Calculate effective Rabi frequency with detuning
                    effective_rabi = np.sqrt(rabi_freq**2 + detuning1**2)
                    
                    # Calculate oscillation between |0⟩ and |-1⟩
                    if effective_rabi > 0:
                        # Calculate probability to be in |-1⟩
                        p_transfer = (rabi_freq/effective_rabi)**2 * np.sin(np.pi*effective_rabi*duration)**2
                        
                        # Update populations (conserve probability)
                        prev_ms0 = self.state[0]
                        self.state[0] -= p_transfer * prev_ms0  # Decrease |0⟩ population
                        self.state[2] += p_transfer * prev_ms0  # Increase |-1⟩ population
                        
                elif abs(detuning2) < 10*linewidth:
                    # Near ms=0 to ms=+1 resonance
                    # Calculate effective Rabi frequency with detuning
                    effective_rabi = np.sqrt(rabi_freq**2 + detuning2**2)
                    
                    # Calculate oscillation between |0⟩ and |+1⟩
                    if effective_rabi > 0:
                        # Calculate probability to be in |+1⟩
                        p_transfer = (rabi_freq/effective_rabi)**2 * np.sin(np.pi*effective_rabi*duration)**2
                        
                        # Update populations (conserve probability)
                        prev_ms0 = self.state[0]
                        self.state[0] -= p_transfer * prev_ms0  # Decrease |0⟩ population
                        self.state[1] += p_transfer * prev_ms0  # Increase |+1⟩ population
                        
            # T1 and T2 relaxation effects
            t1 = self.config['t1']
            t2 = self.config['t2']
            
            # T1 relaxation towards thermal equilibrium (ms=0 at low temperature)
            if t1 > 0:
                # Exponential decay towards equilibrium
                decay_factor = np.exp(-duration / t1)
                
                # Equilibrium populations (simple approximation)
                equil_pop = np.array([1.0, 0.0, 0.0])  # All in ms=0 at low temperature
                
                # Apply T1 relaxation
                self.state = self.state * decay_factor + equil_pop * (1 - decay_factor)
            
            # Ensure valid probabilities
            self.state = np.clip(self.state, 0.0, 1.0)
            self.state = self.state / np.sum(self.state)  # Normalize
            
    def get_fluorescence_rate(self):
        """Get the current fluorescence rate.
        
        Returns:
            float: Fluorescence rate in counts/s
        """
        with self.lock:
            # Basic parameters
            base_rate = 100000.0  # Base count rate
            contrast = 0.3  # 30% contrast between ms=0 and ms=±1
            
            # Calculate fluorescence based on populations
            # ms=0 gives high fluorescence, ms=±1 gives low fluorescence
            ms0_pop = self.state[0]
            rate = base_rate * (1.0 - contrast * (1.0 - ms0_pop))
            
            # Apply collection efficiency
            rate *= self._collection_efficiency
            
            # Add some noise
            noise = np.random.normal(0, rate * 0.02)  # 2% noise
            
            return rate + noise
            
    def simulate_odmr(self, f_min, f_max, n_points, mw_power=-10.0):
        """Simulate an ODMR scan over a frequency range.
        
        Args:
            f_min: Start frequency in Hz
            f_max: End frequency in Hz
            n_points: Number of frequency points
            mw_power: Microwave power in dBm
            
        Returns:
            SimulationResult: Object containing frequencies and signal data
        """
        with self.lock:
            # Generate frequency points
            frequencies = np.linspace(f_min, f_max, n_points)
            signal = np.zeros(n_points)
            
            # Save original state
            original_state = self.state.copy()
            
            # Calculate resonance frequencies
            resonance_freq = self.config["d_gs"]  # Zero-field splitting
            
            # Get magnetic field magnitude (in Tesla)
            b_magnitude = np.linalg.norm(self.b_field)
            
            # Calculate Zeeman splitting
            gyro = self.config["gyro_e"]
            zeeman_shift = gyro * b_magnitude
            
            # Resonance dips
            dip1_center = resonance_freq - zeeman_shift
            dip2_center = resonance_freq + zeeman_shift
            
            # Convert dBm to amplitude
            power_factor = 10**(mw_power/20)  # Convert from dBm to amplitude
            
            # Calculate line shape parameters
            width = 5e6  # Linewidth in Hz
            depth = 0.3  # 30% contrast
            
            # Width increases with power due to power broadening
            width *= (1 + 0.5 * power_factor)
            
            # Depth depends on power
            depth *= (1 - np.exp(-power_factor))
            
            # Base signal level
            base_level = 100000.0  # counts/s
            
            # Generate ODMR signal from Lorentzian dips
            for i, freq in enumerate(frequencies):
                # Start with base signal
                signal_val = base_level
                
                # Add Lorentzian dips at resonance frequencies
                dip1 = depth * width**2 / ((freq - dip1_center)**2 + width**2)
                dip2 = depth * width**2 / ((freq - dip2_center)**2 + width**2)
                
                # Combine dips and scale
                signal_val *= (1.0 - dip1 - dip2)
                
                # Add noise
                noise = np.random.normal(0, 0.01 * base_level)
                signal[i] = signal_val + noise
            
            # Restore original state
            self.state = original_state
            
            # Return result
            result = SimpleSimResult(
                type="ODMR",
                frequencies=frequencies,
                signal=signal,
                resonances=[dip1_center, dip2_center]
            )
            
            return result

class SimpleSimResult:
    """Simple container for simulation results."""
    
    def __init__(self, **kwargs):
        """Initialize with arbitrary attributes.
        
        Args:
            **kwargs: Attributes to set on the object
        """
        for key, value in kwargs.items():
            setattr(self, key, value)