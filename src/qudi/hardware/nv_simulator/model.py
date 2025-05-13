# -*- coding: utf-8 -*-

"""
This file contains the physical model for NV center simulation.

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
import time
import numpy as np
import threading
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class SimulationResult:
    """Data container for simulation results"""
    type: str = ""
    frequencies: Optional[np.ndarray] = None
    signal: Optional[np.ndarray] = None
    times: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    
    def __init__(self, type="", **kwargs):
        self.type = type
        self.metadata = {}
        
        # Store all provided kwargs as attributes
        for key, value in kwargs.items():
            setattr(self, key, value)


class PhysicalNVModel:
    """
    Main physical model for the NV center simulator.
    
    This class provides a physically accurate model of NV center dynamics
    for quantum simulations.
    """
    
    def __init__(self, optics=True, nitrogen=False, method="matrix", **kwargs):
        """
        Initialize the physical model with configurable options.
        
        Parameters
        ----------
        optics : bool, optional
            Include optical levels (ground, excited, singlet states)
        nitrogen : bool, optional
            Include nitrogen nuclear spin
        method : str, optional
            Simulation method, one of: "matrix", "qutip"
        
        Additional Parameters
        --------------------
        zero_field_splitting : float, optional
            Zero-field splitting (D) in Hz, default: 2.87 GHz
        gyromagnetic_ratio : float, optional
            Electron gyromagnetic ratio in Hz/T, default: 28.025 GHz/T
        strain : float or array, optional
            Strain component or vector in Hz
        temperature : float, optional
            Temperature in Kelvin, default: 300K
        t1 : float, optional
            T1 relaxation time in seconds, default: 5ms
        t2 : float, optional
            T2 dephasing time in seconds, default: 10μs
        c13_concentration : float, optional
            13C concentration, default: 0.011 (natural abundance)
        thread_safe : bool, optional
            Whether to use thread locking, default: True
        """
        # Configuration
        self.config = {
            "optics": optics,
            "nitrogen": nitrogen,
            "method": method,
            "d_gs": kwargs.get("zero_field_splitting", 2.87e9),  # Zero-field splitting (Hz)
            "gyro_e": kwargs.get("gyromagnetic_ratio", 28.025e9),  # Gyromagnetic ratio (Hz/T)
            "t1": kwargs.get("t1", 5.0e-3),    # T1 relaxation time (s)
            "t2": kwargs.get("t2", 1.0e-5),     # T2 dephasing time (s)
            "strain": kwargs.get("strain", 0.0),  # Strain in Hz
            "temperature": kwargs.get("temperature", 300.0),  # Temperature in K
            "c13_concentration": kwargs.get("c13_concentration", 0.011),  # 13C concentration
        }
        
        # Update with any additional kwargs
        self.config.update({k: v for k, v in kwargs.items() if k not in self.config})
        
        # Thread safety
        self.lock = threading.RLock() if kwargs.get("thread_safe", True) else DummyLock()
        
        # Initialize magnetic field (Tesla)
        self.b_field = np.array([0.0, 0.0, 0.0])
        
        # Initialize state
        self.state = np.zeros(3)
        self.state[0] = 1.0  # Start in ms=0 state
        
        # Runtime parameters
        self._collection_efficiency = 1.0
        self._microwave_frequency = self.config["d_gs"]  # Default to zero-field splitting
        self._microwave_amplitude = 0.0
        self._microwave_on = False
        self._laser_power = 0.0
        self._laser_on = False
        
        logger.info(f"NV Model initialized with {method} method, optics={optics}")
        
    def reset_state(self):
        """Reset the NV state to the ground state |0⟩"""
        with self.lock:
            self.state = np.zeros(3)
            self.state[0] = 1.0  # |ms=0⟩ state
            logger.debug("Reset NV state to |0⟩")
            
    def set_magnetic_field(self, field):
        """
        Set the magnetic field vector.
        
        Parameters
        ----------
        field : list or ndarray
            Magnetic field vector [Bx, By, Bz] in Gauss (or Tesla)
        """
        with self.lock:
            # Convert field to Tesla if given in Gauss
            if isinstance(field, (list, np.ndarray)) and len(field) == 3:
                # Check if given in Gauss (typical values 100-1000)
                if np.max(np.abs(field)) > 0.1:
                    # Convert from Gauss to Tesla
                    field_tesla = np.array(field, dtype=float) * 1e-4
                else:
                    # Already in Tesla
                    field_tesla = np.array(field, dtype=float)
            else:
                # Create uniform field in z direction as scalar
                field_tesla = np.array([0, 0, field], dtype=float)
                
            self.b_field = field_tesla
            logger.debug(f"Magnetic field set to {self.b_field} T")
            
    def set_temperature(self, temperature):
        """
        Set the temperature for thermal effects.
        
        Parameters
        ----------
        temperature : float
            Temperature in Kelvin
        """
        with self.lock:
            self.config["temperature"] = float(temperature)
            logger.debug(f"Temperature set to {temperature} K")
            
    def apply_laser(self, power, on=True):
        """
        Control the laser for optical excitation.
        
        Parameters
        ----------
        power : float
            Laser power in normalized units (0.0-1.0)
        on : bool
            Whether to turn on the laser (True) or off (False)
        """
        with self.lock:
            self._laser_power = float(power)
            self._laser_on = on
            
            # In real NV systems, laser excitation triggers optical pumping
            if on and power > 0:
                # Apply optical pumping effect depending on power
                # At high powers, this drives the NV to the ms=0 state
                self._apply_optical_pumping(power)
    
    def _apply_optical_pumping(self, power):
        """Apply optical pumping from laser excitation."""
        # Optical pumping polarizes the NV center toward ms=0 state
        # The rate depends on laser power
        try:
            # Simplified phenomenological model
            # Higher power means faster polarization to ms=0
            polarization_factor = min(1.0, power / 0.5)  # Saturates at ~0.5 mW
            
            # Update state probabilities toward ms=0
            ms0_population = self.state[0]
            ms1_population = self.state[1]
            msm1_population = self.state[2]
            
            # Polarization through optical cycle and intersystem crossing
            self.state[0] += polarization_factor * (1 - ms0_population) * 0.2
            self.state[1] -= polarization_factor * ms1_population * 0.2
            self.state[2] -= polarization_factor * msm1_population * 0.2
            
            # Normalize probabilities
            self.state = self.state / np.sum(self.state)
        except Exception as e:
            logger.warning(f"Error in optical pumping simulation: {e}")
    
    def apply_microwave(self, frequency, power_dbm, on=True):
        """
        Apply microwave drive with specific frequency and power.
        
        Parameters
        ----------
        frequency : float
            Microwave frequency in Hz
        power_dbm : float
            Microwave power in dBm
        on : bool
            Whether to turn on the microwave (True) or off (False)
        """
        with self.lock:
            # Set frequency
            self._microwave_frequency = frequency
            
            # Convert dBm to amplitude (simplified conversion)
            # 0 dBm = 1 mW, -10 dBm = 0.1 mW, etc.
            if on:
                # P(mW) = 10^(P(dBm)/10)
                power_mw = 10**(power_dbm/10)
                
                # Convert to amplitude (simplified relationship)
                # Scaling factor is arbitrary and depends on hardware implementation
                self._microwave_amplitude = np.sqrt(power_mw) * 0.01
                self._microwave_on = True
            else:
                # Turn off microwave
                self._microwave_amplitude = 0.0
                self._microwave_on = False
    
    def get_fluorescence(self):
        """
        Get the current fluorescence signal.
        
        Returns
        -------
        float
            Fluorescence count rate in counts/s
        """
        with self.lock:
            # Get ms=0 population
            ms0_pop = self.state[0]
            
            # ms=0 has higher fluorescence than ms=±1
            base_fluorescence = 1e5  # counts/s
            contrast = 0.3  # 30% contrast
            
            # Scale by collection efficiency
            return base_fluorescence * self._collection_efficiency * (1.0 - contrast * (1.0 - ms0_pop))
    
    def get_populations(self):
        """
        Get the populations of different spin states.
        
        Returns
        -------
        dict
            Dictionary with keys 'ms0', 'ms_plus', 'ms_minus' and their probabilities
        """
        with self.lock:
            # Basic implementation for 3-level system
            return {
                'ms0': self.state[0],
                'ms+1': self.state[1],
                'ms-1': self.state[2]
            }
    
    def evolve(self, duration):
        """
        Evolve the quantum state for a specified duration.
        
        Parameters
        ----------
        duration : float
            Time to evolve in seconds
        """
        with self.lock:
            # Get current state probabilities
            ms0_population = self.state[0]
            ms1_population = self.state[1]
            msm1_population = self.state[2]
            
            # Rabi oscillations if microwave is on
            if self._microwave_on and self._microwave_amplitude > 0:
                # Calculate the detuning from resonance
                resonance_freq = self.config["d_gs"]  # Zero-field splitting
                
                # Apply magnetic field effect on resonance
                b_magnitude = np.linalg.norm(self.b_field)
                zeeman_shift = self.config["gyro_e"] * b_magnitude
                
                # Consider resonances for both ms=0 to ms=±1 transitions
                detuning_plus = self._microwave_frequency - (resonance_freq + zeeman_shift)
                detuning_minus = self._microwave_frequency - (resonance_freq - zeeman_shift)
                
                # Scale the Rabi frequency by the microwave amplitude
                rabi_freq = 10e6 * self._microwave_amplitude  # 10 MHz at amplitude 1.0
                
                # Calculate effective Rabi frequencies with detuning
                omega_plus = np.sqrt(rabi_freq**2 + detuning_plus**2)
                omega_minus = np.sqrt(rabi_freq**2 + detuning_minus**2)
                
                # Calculate flip probabilities 
                if omega_plus > 0:
                    flip_prob_plus = (rabi_freq / omega_plus)**2 * np.sin(np.pi * omega_plus * duration)**2
                else:
                    flip_prob_plus = 0
                    
                if omega_minus > 0:
                    flip_prob_minus = (rabi_freq / omega_minus)**2 * np.sin(np.pi * omega_minus * duration)**2
                else:
                    flip_prob_minus = 0
                
                # Apply the state changes
                # Transitions between ms=0 and ms=+1
                ms0_to_plus = ms0_population * flip_prob_plus
                plus_to_ms0 = ms1_population * flip_prob_plus
                
                # Transitions between ms=0 and ms=-1
                ms0_to_minus = ms0_population * flip_prob_minus
                minus_to_ms0 = msm1_population * flip_prob_minus
                
                # Update populations
                new_ms0 = ms0_population - ms0_to_plus - ms0_to_minus + plus_to_ms0 + minus_to_ms0
                new_plus = ms1_population - plus_to_ms0 + ms0_to_plus
                new_minus = msm1_population - minus_to_ms0 + ms0_to_minus
                
                # Apply changes
                self.state[0] = new_ms0
                self.state[1] = new_plus
                self.state[2] = new_minus
            
            # Apply relaxation effects (T1, T2)
            t1 = self.config["t1"]
            if t1 > 0 and duration > 0:
                # T1 relaxation - equilibration to thermal state
                t1_factor = 1 - np.exp(-duration / t1)
                
                # At room temperature, thermal equilibrium is approximately equal populations
                thermal_population = 1/3
                
                # Apply T1 relaxation toward thermal equilibrium
                self.state[0] = self.state[0] * (1 - t1_factor) + thermal_population * t1_factor
                self.state[1] = self.state[1] * (1 - t1_factor) + thermal_population * t1_factor
                self.state[2] = self.state[2] * (1 - t1_factor) + thermal_population * t1_factor
            
            # Normalize state probabilities
            total = np.sum(self.state)
            if total > 0:
                self.state = self.state / total
                
            # Apply laser effects if active
            if self._laser_on and self._laser_power > 0:
                # Apply optical pumping effect from laser illumination
                self._apply_optical_pumping(self._laser_power)
    
    def simulate_odmr(self, f_min, f_max, n_points, mw_power=-10.0):
        """
        Run an ODMR experiment.
        
        Parameters
        ----------
        f_min : float
            Start frequency in Hz
        f_max : float
            End frequency in Hz
        n_points : int
            Number of frequency points
        mw_power : float, optional
            Microwave power in dBm
            
        Returns
        -------
        SimulationResult
            Object containing frequencies and signals
        """
        with self.lock:
            # Save original state and settings to restore later
            original_state = self.state.copy()
            original_mw_freq = self._microwave_frequency
            original_mw_amp = self._microwave_amplitude
            original_mw_on = self._microwave_on
            
            # Generate frequency points
            frequencies = np.linspace(f_min, f_max, n_points)
            
            # Generate ODMR spectrum using analytical model
            # Convert dBm to amplitude
            power_factor = 10**(mw_power/20)  # Convert from dBm to amplitude
            
            # Initialize signal array
            signal = np.ones(n_points)
            
            # Zero-field splitting
            d_gs = self.config["d_gs"]
            
            # Calculate Zeeman splitting based on magnetic field
            b_magnitude = np.linalg.norm(self.b_field)
            gyro = self.config["gyro_e"]
            zeeman_shift = gyro * b_magnitude
            
            # Create resonance dips
            f1 = d_gs - zeeman_shift  # ms=0 to ms=-1 transition
            f2 = d_gs + zeeman_shift  # ms=0 to ms=+1 transition
            
            # ODMR linewidth depends on microwave power (power broadening)
            width = 5e6  # 5 MHz base linewidth 
            width *= (1 + 0.5 * power_factor)  # Power broadening
            
            # ODMR contrast also depends on microwave power
            depth = 0.3  # 30% base contrast
            depth *= (1 - np.exp(-power_factor))  # Power-dependent contrast
            
            # Create Lorentzian dips
            for f in [f1, f2]:
                if frequencies[0] <= f <= frequencies[-1]:  # Only if resonance is in range
                    signal -= depth * width**2 / ((frequencies - f)**2 + width**2)
            
            # Scale to typical fluorescence rate and add noise
            base_rate = 100000.0  # counts/s
            signal *= base_rate * self._collection_efficiency
            
            # Add some noise
            noise_level = 0.01  # 1% noise
            signal += np.random.normal(0, noise_level * base_rate, len(frequencies))
            
            # Restore original state and settings
            self.state = original_state
            self._microwave_frequency = original_mw_freq
            self._microwave_amplitude = original_mw_amp
            self._microwave_on = original_mw_on
            
            # Return result object
            return SimulationResult(
                type="ODMR",
                frequencies=frequencies,
                signal=signal,
                mw_power=mw_power,
                resonances=[f1, f2],
                zeeman_shift=zeeman_shift,
                collection_efficiency=self._collection_efficiency
            )
                
    def simulate_rabi(self, t_max, n_points, mw_power=0.0, mw_frequency=None):
        """
        Run a Rabi oscillation experiment.
        
        Parameters
        ----------
        t_max : float
            Maximum Rabi time in seconds
        n_points : int
            Number of time points
        mw_power : float, optional
            Microwave power in dBm
        mw_frequency : float, optional
            Microwave frequency in Hz. If None, use resonance frequency.
            
        Returns
        -------
        SimulationResult
            Object containing times and signals
        """
        with self.lock:
            # Save original state and settings
            original_state = self.state.copy()
            original_mw_freq = self._microwave_frequency
            original_mw_amp = self._microwave_amplitude
            original_mw_on = self._microwave_on
            
            # Generate time points
            times = np.linspace(0, t_max, n_points)
            
            # Use analytical model for Rabi oscillations
            # Convert dBm to Rabi frequency (simplified model)
            # 0 dBm → ~10 MHz Rabi frequency for typical setup
            power_factor = 10**(mw_power/20)  # Convert from dBm to amplitude
            rabi_freq = 10e6 * power_factor  # Rabi frequency in Hz
            
            # Use resonance frequency if not specified
            if mw_frequency is None:
                # Calculate resonance based on magnetic field
                b_magnitude = np.linalg.norm(self.b_field)
                gyro = self.config["gyro_e"]
                zeeman_shift = gyro * b_magnitude
                
                # Use the ms=0 to ms=+1 transition
                mw_frequency = self.config["d_gs"] + zeeman_shift
            
            # Calculate detuning from resonance
            resonance_freq = self.config["d_gs"]  # Zero-field splitting
            b_magnitude = np.linalg.norm(self.b_field)
            zeeman_shift = self.config["gyro_e"] * b_magnitude
            
            # Pick the closest resonance
            if abs(mw_frequency - (resonance_freq + zeeman_shift)) < abs(mw_frequency - (resonance_freq - zeeman_shift)):
                # Closer to ms=0 to ms=+1 transition
                detuning = mw_frequency - (resonance_freq + zeeman_shift)
            else:
                # Closer to ms=0 to ms=-1 transition
                detuning = mw_frequency - (resonance_freq - zeeman_shift)
            
            # Effective Rabi frequency including detuning
            effective_rabi = np.sqrt(rabi_freq**2 + detuning**2)
            
            # Generate Rabi oscillation with detuning
            if detuning == 0:
                # On resonance: full contrast oscillation
                oscillation = 1 - np.sin(np.pi * rabi_freq * times)**2
            else:
                # Off resonance: reduced contrast oscillation
                contrast_factor = rabi_freq**2 / effective_rabi**2
                oscillation = 1 - contrast_factor * np.sin(np.pi * effective_rabi * times)**2
            
            # Add damping from T2 effects
            t2 = self.config["t2"]  # T2 time
            damping = np.exp(-times/t2)
            signal = 1 - (1 - oscillation) * damping
            
            # Scale to typical fluorescence rate and add noise
            base_rate = 100000.0  # counts/s
            contrast = 0.3  # 30% contrast
            signal = base_rate * (1 - contrast * (1 - signal))
            
            # Add some noise
            noise_level = 0.02  # 2% noise
            signal += np.random.normal(0, noise_level * base_rate, len(times))
            
            # Restore original state and settings
            self.state = original_state
            self._microwave_frequency = original_mw_freq
            self._microwave_amplitude = original_mw_amp
            self._microwave_on = original_mw_on
            
            # Return result object
            return SimulationResult(
                type="Rabi",
                times=times,
                signal=signal,
                rabi_frequency=rabi_freq,
                effective_rabi=effective_rabi,
                detuning=detuning,
                t2=t2,
                mw_power=mw_power,
                mw_frequency=mw_frequency
            )
    
    def simulate_t1(self, t_max, n_points):
        """
        Run a T1 relaxation experiment.
        
        Parameters
        ----------
        t_max : float
            Maximum time in seconds
        n_points : int
            Number of time points
            
        Returns
        -------
        SimulationResult
            Object containing times and signals
        """
        with self.lock:
            # Generate time points
            times = np.linspace(0, t_max, n_points)
            
            # T1 relaxation time
            t1 = self.config["t1"]
            
            # Generate T1 relaxation curve
            # NV starts in ms=±1 and relaxes to ms=0
            relaxation = 1 - np.exp(-times/t1)
            
            # Scale to typical fluorescence rate and add noise
            base_rate = 100000.0  # counts/s
            contrast = 0.3  # 30% contrast
            signal = base_rate * (1 - contrast * (1 - relaxation))
            
            # Add some noise
            noise_level = 0.02  # 2% noise
            signal += np.random.normal(0, noise_level * base_rate, len(times))
            
            # Return result object
            return SimulationResult(
                type="T1",
                times=times,
                signal=signal,
                t1=t1
            )


# Dummy lock for non-thread-safe operation
class DummyLock:
    def __enter__(self):
        return self
        
    def __exit__(self, *args):
        pass