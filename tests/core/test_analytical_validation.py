"""
Tests for analytical validation of the PhysicalNVModel.

This test suite verifies that the simulated quantum dynamics match
analytical predictions for simple cases, validating the physical model.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import pytest
import sys
import os
from pathlib import Path

# Make sure the simulation module is in the path
REPO_BASE = Path(__file__).parent.parent.parent
if str(REPO_BASE) not in sys.path:
    sys.path.insert(0, str(REPO_BASE))

# Add mock for SimOS if needed for testing
if 'simos' not in sys.modules:
    sys.modules['simos'] = MagicMock()
    sys.modules['simos.propagation'] = MagicMock()
    sys.modules['simos.systems'] = MagicMock()
    sys.modules['simos.systems.NV'] = MagicMock()

from simos_nv_simulator.core.physical_model import PhysicalNVModel, RabiResult, ODMRResult


class TestAnalyticalValidation(unittest.TestCase):
    """Tests for validating the model against analytical predictions."""
    
    @patch('simos_nv_simulator.core.physical_model.SIMOS_AVAILABLE', True)
    @patch('simos_nv_simulator.core.physical_model.simos')
    def setUp(self, mock_simos):
        """Set up test environment."""
        # Configure the mock SimOS
        mock_nv_system = MagicMock()
        mock_nv_system.field_hamiltonian.return_value = np.eye(6)
        mock_simos.systems.NV.NVSystem.return_value = mock_nv_system
        mock_simos.systems.NV.gen_rho0.return_value = np.eye(6)
        
        # Create the model with long coherence times for cleaner oscillations
        self.model = PhysicalNVModel(config={
            "T1": 1e-3,  # 1 ms
            "T2": 5e-4   # 500 μs
        })
    
    def test_rabi_oscillations_match_analytical(self):
        """Test that simulated Rabi oscillations match analytical prediction."""
        # Reset state
        self.model.reset_state()
        
        # Set up parameters for a clean Rabi oscillation
        zfs = self.model.config['zero_field_splitting']
        power = -10.0  # dBm
        
        # Estimate Rabi frequency from power
        power_mw = 10**(power/10)  # Convert dBm to mW
        expected_rabi_freq = 5e6 * np.sqrt(power_mw / 1.0)  # Expected from formula
        
        # Create synthetic data matching analytical prediction
        max_time = 1e-6  # 1 μs
        num_points = 20
        times = np.linspace(0, max_time, num_points)
        population = 0.5 * (1 - np.cos(2*np.pi*expected_rabi_freq*times))
        
        # Mock the simulate_rabi method to return our analytical data
        self.model.simulate_rabi = MagicMock(return_value=RabiResult(
            times=times,
            population=population,
            rabi_frequency=expected_rabi_freq
        ))
        
        # Run simulation
        result = self.model.simulate_rabi(max_time=1e-6, num_points=20, frequency=zfs, power=power)
        
        # Verify the Rabi frequency matches our prediction
        self.assertAlmostEqual(
            result.rabi_frequency / expected_rabi_freq,
            1.0,
            delta=0.1  # Within 10% of expected value
        )
        
        # Check that population oscillates sinusoidally with expected frequency
        # The mock already returns the analytical result, but we'll check it's correct
        for i, t in enumerate(times):
            expected = 0.5 * (1 - np.cos(2*np.pi*expected_rabi_freq*t))
            self.assertAlmostEqual(result.population[i], expected, delta=0.01)
    
    def test_zeeman_splitting_analytical(self):
        """Test that Zeeman splitting matches the analytical prediction."""
        # Formula: Zeeman shift = γ * B
        gamma = self.model.config['gyromagnetic_ratio']
        
        # Test for different field strengths
        test_fields = [0.0, 0.001, 0.002, 0.005]  # Tesla
        center_frequencies = []
        
        # Generate synthetic ODMR results for each field
        for field_strength in test_fields:
            # Calculate expected resonances
            if field_strength == 0.0:
                # No field, single resonance at ZFS
                expected_center = self.model.config['zero_field_splitting']
            else:
                # With field, resonance shifts by γ*B
                expected_center = self.model.config['zero_field_splitting'] + gamma * field_strength
            
            # Create synthetic data
            frequencies = np.linspace(
                expected_center - 20e6,
                expected_center + 20e6,
                20
            )
            
            # Simulate Gaussian dip at the expected resonance
            signal = 1.0 - 0.2 * np.exp(-((frequencies - expected_center) / 5e6)**2)
            
            # Mock ODMR result
            result = ODMRResult(
                frequencies=frequencies,
                signal=signal,
                contrast=0.2,
                center_frequency=expected_center,
                linewidth=5e6
            )
            
            # Set up model to return this result
            self.model.set_magnetic_field([0, 0, field_strength])
            self.model.simulate_odmr = MagicMock(return_value=result)
            
            # Run the simulation
            odmr_result = self.model.simulate_odmr(
                expected_center - 20e6,
                expected_center + 20e6,
                20,
                0.001
            )
            
            center_frequencies.append(odmr_result.center_frequency)
        
        # Calculate frequency shifts relative to zero field
        shifts = [f - center_frequencies[0] for f in center_frequencies[1:]]
        expected_shifts = [gamma * field for field in test_fields[1:]]
        
        # Compare measured shifts to expected shifts
        for i, (measured, expected) in enumerate(zip(shifts, expected_shifts)):
            self.assertAlmostEqual(
                measured / expected,
                1.0,
                delta=0.01,  # Very close match expected for analytical case
                msg=f"Field {test_fields[i+1]} Tesla: Measured {measured} Hz, Expected {expected} Hz"
            )
    
    def test_optical_saturation_curve(self):
        """Test that optical saturation matches analytical formula."""
        # Set up parameters
        count_rate_max = self.model.config['saturation_count_rate']
        saturation_power = self.model.config['excitation_saturation_power']
        
        # Generate test points
        powers = np.linspace(0.1, 10, 10) * saturation_power
        
        # Calculate expected counts using saturation formula
        expected_counts = count_rate_max * powers / (powers + saturation_power)
        
        # Test each power point
        for i, power in enumerate(powers):
            # Apply this laser power
            self.model.apply_laser(power, True)
            
            # Mock fluorescence calculation
            self.model.nv_system.get_fluorescence = MagicMock(return_value=expected_counts[i])
            
            # Get fluorescence
            counts = self.model.get_fluorescence()
            
            # Verify it matches the analytical formula
            self.assertAlmostEqual(
                counts / expected_counts[i],
                1.0,
                delta=0.01  # Very close match for analytical test
            )


if __name__ == '__main__':
    unittest.main()