# -*- coding: utf-8 -*-

"""
NV simulator manager for qudi - provides a simple interface to the simulator for dummy hardware modules.

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
import logging
import numpy as np
import time

# Try to import the simulator from IQO-q/sim/src if installed
try:
    from model import PhysicalNVModel
    _simulator_available = True
except ImportError:
    try:
        # Look for model in the path IQO-q/src/qudi/hardware/nv_simulator/ 
        current_dir = os.path.dirname(os.path.abspath(__file__))
        nv_simulator_dir = os.path.abspath(os.path.join(current_dir, '..', 'nv_simulator'))
        
        if os.path.exists(nv_simulator_dir) and nv_simulator_dir not in sys.path:
            sys.path.insert(0, nv_simulator_dir)
            
        # Try to import model from this path
        from model import PhysicalNVModel
        _simulator_available = True
    except ImportError:
        _simulator_available = False


class NVSimulatorManager:
    """
    Singleton class to manage the NV simulator for multiple dummy modules.
    This provides a simplified interface to the simulator that can be used by the
    microwave_dummy, finite_sampler_dummy, and counter_dummy modules.
    """
    
    _instance = None
    _logger = logging.getLogger(__name__)
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(NVSimulatorManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, magnetic_field=None, temperature=None, use_simulator=True):
        """Initialize the NV simulator.
        
        @param magnetic_field: magnetic field vector in Gauss [x, y, z]
        @param temperature: temperature in Kelvin
        @param use_simulator: whether to use the actual simulator or the fallback model
        """
        # Only initialize once
        if self._initialized:
            return
            
        self._initialized = True
        self._logger.info("Initializing NV simulator manager")
        
        # Default parameters
        self._magnetic_field = magnetic_field or [0, 0, 500]  # Default to 500 G along z
        self._temperature = temperature or 300  # Default to room temperature
        self._use_simulator = use_simulator and _simulator_available
        
        # Current microwave state
        self._mw_frequency = 2.87e9  # Default to zero-field splitting
        self._mw_power = -20.0  # Default to -20 dBm
        self._mw_on = False  # Default to off
        
        # Current laser state
        self._laser_power = 0.0  # Default to 0 mW (off)
        self._laser_on = False  # Default to off
        
        # Current scan state
        self._scanning = False
        self._scan_index = 0
        self._scan_frequencies = None
        
        # Create the NV model
        try:
            if self._use_simulator:
                self._logger.info("Creating PhysicalNVModel")
                model_params = {
                    'zero_field_splitting': 2.87e9,  # Hz
                    'gyromagnetic_ratio': 2.8025e10,  # Hz/T
                    't1': 5.0e-3,  # ms
                    't2': 1.0e-5,  # ms
                    'thread_safe': True,
                    'memory_management': True,
                    'optimize_performance': True
                }
                
                # Create the actual simulator model
                self._model = PhysicalNVModel(**model_params)
                
                # Set magnetic field (convert from Gauss to Tesla)
                b_field_tesla = [b * 1e-4 for b in self._magnetic_field]  # 1 G = 1e-4 T
                self._model.set_magnetic_field(b_field_tesla)
                
                # Set temperature
                self._model.set_temperature(self._temperature)
                
                self._logger.info(f"Using real simulator with magnetic field {self._magnetic_field} G")
                self._logger.info(f"  Zero-field splitting: 2.87 GHz")
                self._logger.info(f"  Gyromagnetic ratio: 2.8025e10 Hz/T")
                self._logger.info(f"  Resonances at:")
                
                # Calculate resonance frequencies
                zfs = 2.87e9  # Zero-field splitting (Hz)
                gyro = 2.8e6  # Gyromagnetic ratio (Hz/G)
                field = np.linalg.norm(self._magnetic_field)
                
                zeeman_shift = gyro * field
                dip1 = zfs - zeeman_shift
                dip2 = zfs + zeeman_shift
                
                self._logger.info(f"  - {dip1/1e9:.6f} GHz (ms=0 -> ms=-1)")
                self._logger.info(f"  - {dip2/1e9:.6f} GHz (ms=0 -> ms=+1)")
                
            else:
                # Use a simplified model
                self._logger.info("Using simplified NV model (simulator not available)")
                self._model = SimplifiedNVModel(self._magnetic_field, self._temperature)
                
        except Exception as e:
            self._logger.error(f"Failed to create NV model: {e}")
            self._logger.info("Using simplified NV model after error")
            self._model = SimplifiedNVModel(self._magnetic_field, self._temperature)
    
    @property
    def model(self):
        """Get the NV model."""
        return self._model
    
    @property
    def magnetic_field(self):
        """Get the magnetic field vector in Gauss."""
        return self._magnetic_field
    
    @property
    def temperature(self):
        """Get the temperature in Kelvin."""
        return self._temperature
    
    @property
    def mw_frequency(self):
        """Get the current microwave frequency in Hz."""
        return self._mw_frequency
    
    @property
    def mw_power(self):
        """Get the current microwave power in dBm."""
        return self._mw_power
    
    @property
    def mw_on(self):
        """Check if the microwave is on."""
        return self._mw_on
    
    @property
    def laser_power(self):
        """Get the current laser power in mW."""
        return self._laser_power
    
    @property
    def laser_on(self):
        """Check if the laser is on."""
        return self._laser_on
    
    @property
    def is_scanning(self):
        """Check if scanning is active."""
        return self._scanning
    
    @property
    def scan_index(self):
        """Get the current scan index."""
        return self._scan_index
    
    def set_magnetic_field(self, field_vector):
        """Set the magnetic field vector.
        
        @param field_vector: magnetic field vector in Gauss [x, y, z]
        """
        self._magnetic_field = field_vector
        
        try:
            # Update the model (convert from Gauss to Tesla)
            b_field_tesla = [b * 1e-4 for b in field_vector]  # 1 G = 1e-4 T
            self._model.set_magnetic_field(b_field_tesla)
            
            self._logger.info(f"Magnetic field updated to {field_vector} G")
        except Exception as e:
            self._logger.error(f"Failed to update magnetic field: {e}")
    
    def set_temperature(self, temperature):
        """Set the temperature.
        
        @param temperature: temperature in Kelvin
        """
        self._temperature = temperature
        
        try:
            # Update the model
            self._model.set_temperature(temperature)
            
            self._logger.info(f"Temperature updated to {temperature} K")
        except Exception as e:
            self._logger.error(f"Failed to update temperature: {e}")
    
    def set_microwave(self, frequency, power, on=True):
        """Set microwave parameters.
        
        @param frequency: frequency in Hz
        @param power: power in dBm
        @param on: whether the microwave is on
        """
        self._mw_frequency = frequency
        self._mw_power = power
        old_on_state = self._mw_on
        self._mw_on = on
        
        try:
            # Update the model
            if self._use_simulator:
                # Convert dBm to amplitude for the simulator
                # P(dBm) = 10 * log10(P(mW))
                # P(mW) = 10^(P(dBm)/10)
                power_mw = 10**(power/10)
                amplitude = np.sqrt(power_mw) * 0.01  # Scaling factor for the model
                
                self._model.set_microwave_frequency(frequency)
                
                # Only set amplitude if state changed to avoid unnecessary updates
                if on != old_on_state:
                    if on:
                        self._model.set_microwave_amplitude(amplitude)
                    else:
                        self._model.set_microwave_amplitude(0.0)
                
            self._logger.info(f"Microwave updated: freq={frequency/1e9:.6f} GHz, power={power} dBm, on={on}")
        except Exception as e:
            self._logger.error(f"Failed to update microwave: {e}")
    
    def set_laser(self, power, on=True):
        """Set laser parameters.
        
        @param power: power in mW
        @param on: whether the laser is on
        """
        self._laser_power = power
        old_on_state = self._laser_on
        self._laser_on = on
        
        try:
            # Update the model
            if self._use_simulator:
                if on != old_on_state:
                    if on:
                        self._model.set_laser_power(power)
                    else:
                        self._model.set_laser_power(0.0)
                
            self._logger.info(f"Laser updated: power={power} mW, on={on}")
        except Exception as e:
            self._logger.error(f"Failed to update laser: {e}")
    
    def set_scanning(self, scanning, scan_index=0):
        """Set scanning state.
        
        @param scanning: whether scanning is active
        @param scan_index: current scan index
        """
        self._scanning = scanning
        self._scan_index = scan_index
        
        self._logger.info(f"Scanning state updated: scanning={scanning}, scan_index={scan_index}")
    
    def get_signal(self, channel='APD counts', sample_count=1000):
        """Get simulated signal.
        
        @param channel: channel name
        @param sample_count: number of samples to generate
        
        @return numpy.ndarray: signal data
        """
        if channel == 'APD counts':
            # Calculate expected ODMR signal
            if not self._mw_on:
                # Laser off or MW off - just return baseline
                baseline = 100000.0  # 100k counts/s
                noise = np.random.normal(0, 0.02 * baseline, sample_count)
                return baseline + noise
            
            # Calculate resonance frequencies
            zfs = 2.87e9  # Zero-field splitting (Hz)
            gyro = 2.8e6  # Gyromagnetic ratio (Hz/G)
            field = np.linalg.norm(self._magnetic_field)
            
            zeeman_shift = gyro * field
            dip1_center = zfs - zeeman_shift
            dip2_center = zfs + zeeman_shift
            
            # Lorentzian function for each dip
            linewidth = 20e6  # 20 MHz
            contrast = 0.3  # 30% contrast
            baseline = 1.0
            
            # Power scaling
            power_mw = 10**(self._mw_power/10)
            power_factor = min(1.0, power_mw / 10.0)
            
            dip1 = contrast * power_factor * linewidth**2 / ((self._mw_frequency - dip1_center)**2 + linewidth**2)
            dip2 = contrast * power_factor * linewidth**2 / ((self._mw_frequency - dip2_center)**2 + linewidth**2)
            
            # Combine dips and scale to photon counts
            signal = (baseline - dip1 - dip2) * 100000.0
            
            # Add Poisson noise
            sampling_time = 0.001  # 1 ms per sample
            expected_counts = signal * sampling_time
            noise = np.random.poisson(expected_counts, sample_count)
            return noise / sampling_time
            
        elif channel == 'Photodiode':
            # Simple voltage signal with noise
            return 2.5 + 0.02 * np.random.randn(sample_count)
        
        else:
            # Unknown channel - return zeros
            return np.zeros(sample_count)
            
    def simulate_odmr(self, freq_min, freq_max, num_points):
        """Simulate ODMR spectrum.
        
        @param freq_min: minimum frequency in Hz
        @param freq_max: maximum frequency in Hz
        @param num_points: number of points in the spectrum
        
        @return dict: dictionary with keys 'frequencies' and 'signal'
        """
        self._logger.info(f"Simulating ODMR spectrum from {freq_min/1e9:.6f} to {freq_max/1e9:.6f} GHz with {num_points} points")
        
        # Generate frequency array
        frequencies = np.linspace(freq_min, freq_max, num_points)
        
        # Calculate resonance frequencies
        zfs = 2.87e9  # Zero-field splitting (Hz)
        gyro = 2.8e6  # Gyromagnetic ratio (Hz/G)
        field = np.linalg.norm(self._magnetic_field)
        
        zeeman_shift = gyro * field
        dip1_center = zfs - zeeman_shift
        dip2_center = zfs + zeeman_shift
        
        self._logger.info(f"Magnetic field: {field:.1f} G, Zeeman shift: {zeeman_shift/1e6:.2f} MHz")
        self._logger.info(f"Resonances at: {dip1_center/1e9:.6f} GHz and {dip2_center/1e9:.6f} GHz")
        
        # Generate signal at each frequency
        linewidth = 20e6  # 20 MHz
        contrast = 0.3  # 30% contrast
        baseline = 1.0
        
        # Power scaling - use current power setting
        power_mw = 10**(self._mw_power/10)
        power_factor = min(1.0, power_mw / 10.0)
        
        # Calculate Lorentzian dips
        dip1 = np.zeros(num_points)
        dip2 = np.zeros(num_points)
        
        for i, freq in enumerate(frequencies):
            dip1[i] = contrast * power_factor * linewidth**2 / ((freq - dip1_center)**2 + linewidth**2)
            dip2[i] = contrast * power_factor * linewidth**2 / ((freq - dip2_center)**2 + linewidth**2)
        
        # Combine dips and scale to photon counts
        signal = (baseline - dip1 - dip2) * 100000.0
        
        # Add some noise (1% of baseline)
        noise_level = 0.01 * 100000.0
        noise = np.random.normal(0, noise_level, num_points)
        signal += noise
        
        # Check for flat line - add warning if no contrast
        if np.max(signal) - np.min(signal) < 1000:
            self._logger.warning("ODMR spectrum simulation shows minimal contrast")
            self._logger.warning("Check if resonances are within frequency range")
            self._logger.warning(f"Frequency range: {freq_min/1e9:.6f}-{freq_max/1e9:.6f} GHz")
            self._logger.warning(f"Resonances: {dip1_center/1e9:.6f} GHz and {dip2_center/1e9:.6f} GHz")
        
        # Return results
        return {
            'frequencies': frequencies, 
            'signal': signal,
            'dip1_center': dip1_center,
            'dip2_center': dip2_center,
            'contrast': contrast * power_factor
        }
    
    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance."""
        cls._instance = None
        cls._logger.info("NV simulator manager instance reset")


class SimplifiedNVModel:
    """
    Simplified model for when the real simulator is not available.
    """
    
    def __init__(self, magnetic_field=None, temperature=None):
        """Initialize the simplified model.
        
        @param magnetic_field: magnetic field vector in Gauss [x, y, z]
        @param temperature: temperature in Kelvin
        """
        self._logger = logging.getLogger(__name__)
        self._logger.info("Creating simplified NV model")
        
        self._magnetic_field = magnetic_field or [0, 0, 500]  # Default to 500 G along z
        self._temperature = temperature or 300  # Default to room temperature
        
        # Convert Gauss to Tesla for internal use
        self.b_field = [b * 1e-4 for b in self._magnetic_field]  # 1 G = 1e-4 T
    
    def set_magnetic_field(self, field_vector_tesla):
        """Set the magnetic field vector.
        
        @param field_vector_tesla: magnetic field vector in Tesla [x, y, z]
        """
        self.b_field = field_vector_tesla
        # Convert Tesla to Gauss for storage
        self._magnetic_field = [b * 1e4 for b in field_vector_tesla]  # 1 T = 10,000 G
    
    def set_temperature(self, temperature):
        """Set the temperature.
        
        @param temperature: temperature in Kelvin
        """
        self._temperature = temperature
    
    def set_microwave_frequency(self, frequency):
        """Set the microwave frequency.
        
        @param frequency: frequency in Hz
        """
        self._frequency = frequency
    
    def set_microwave_amplitude(self, amplitude):
        """Set the microwave amplitude.
        
        @param amplitude: amplitude (arb. units)
        """
        self._amplitude = amplitude
    
    def set_laser_power(self, power):
        """Set the laser power.
        
        @param power: power in mW
        """
        self._laser_power = power
    
    def get_fluorescence_rate(self):
        """Get the fluorescence rate.
        
        @return float: rate in counts/s
        """
        # Simple model - return baseline rate
        return 100000.0
    
    def reset_state(self):
        """Reset the model state."""
        pass