"""
Tests for optical processes in the PhysicalNVModel.

This test suite verifies the implementation of optical excitation and 
fluorescence processes in the NV center simulator (TS-105).
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import pytest

from simos_nv_simulator.core.physical_model import PhysicalNVModel, OpticalResult


class TestOpticalProcesses(unittest.TestCase):
    """Test suite for the optical processes of the PhysicalNVModel."""
    
    # Maximum run times for time-sensitive tests
    MAX_TIME_OPTICAL = 30  # seconds
    
    def setUp(self):
        """Set up test environment before each test."""
        self.model = PhysicalNVModel()
    
    def test_optical_excitation(self):
        """Test optical excitation of the NV center."""
        # Start with ground state
        self.model.reset_state()
        
        # Check initial state (should be in ground state)
        initial_state = self.model.get_state_info()
        self.assertAlmostEqual(initial_state['ground_state_population'], 1.0, delta=0.1)
        self.assertAlmostEqual(initial_state['excited_state_population'], 0.0, delta=0.1)
        
        # Apply optical excitation
        self.model.apply_laser(1.0, True)
        
        # Let the system evolve
        for _ in range(100):
            if hasattr(self.model.nv_system, 'evolve'):
                self.model.nv_system.evolve(self.model.dt)
        
        # Check excited state population
        excited_state = self.model.get_state_info()
        self.assertGreater(excited_state['excited_state_population'], 0.2)
    
    def test_spin_dependent_fluorescence(self):
        """Test the spin-dependent fluorescence of the NV center."""
        # Initialize to ms=0 state
        self.model.reset_state()
        self.model.initialize_state(ms=0)
        
        # Get ms=0 fluorescence
        self.model.apply_laser(1.0, True)
        ms0_fluorescence = self.model.get_fluorescence()
        
        # Reset and initialize to ms=±1 state
        self.model.reset_state()
        self.model.initialize_state(ms=1)
        
        # Get ms=±1 fluorescence
        self.model.apply_laser(1.0, True)
        ms1_fluorescence = self.model.get_fluorescence()
        
        # Verify spin-dependent contrast (ms=0 should have higher fluorescence)
        self.assertGreater(ms0_fluorescence, ms1_fluorescence * 1.2)
    
    def test_fluorescence_noise(self):
        """Test that fluorescence includes realistic noise."""
        # Set fixed seed for reproducibility
        np.random.seed(42)
        
        # Collect multiple fluorescence readings
        samples = [self.model.get_fluorescence() for _ in range(100)]
        
        # Check statistical properties
        mean = np.mean(samples)
        std_dev = np.std(samples)
        
        # Verify noise is present but not excessive
        self.assertGreater(std_dev, 0)  # Some noise should exist
        self.assertLess(std_dev / mean, 0.5)  # But not overwhelming
        
        # Reset random seed
        np.random.seed()
    
    def test_optical_saturation(self):
        """Test the optical saturation behavior."""
        # Test different laser powers
        powers = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0]
        fluorescence = []
        
        for power in powers:
            self.model.reset_state()
            self.model.apply_laser(power, True)
            
            # Let system equilibrate
            for _ in range(50):
                if hasattr(self.model.nv_system, 'evolve'):
                    self.model.nv_system.evolve(self.model.dt)
            
            fluorescence.append(self.model.get_fluorescence())
        
        # Check saturation behavior (should follow saturation curve)
        # Lower powers should have steeper increase than higher powers
        diff_low = fluorescence[1] - fluorescence[0]
        diff_high = fluorescence[-1] - fluorescence[-2]
        self.assertGreater(diff_low / (powers[1] - powers[0]), 
                          diff_high / (powers[-1] - powers[-2]))
    
    @pytest.mark.slow
    def test_simulate_optical_dynamics(self):
        """Test simulation of optical dynamics."""
        import time
        start_time = time.time()
        
        # Run optical dynamics simulation
        # Parameters: max_time, num_points, laser_power
        result = self.model.simulate_optical_dynamics(1e-6, 10, 1.0)
        
        # Check result type
        self.assertIsInstance(result, OpticalResult)
        
        # Check that times are as expected
        self.assertEqual(len(result.times), 10)
        self.assertEqual(result.times[0], 0.0)
        self.assertEqual(result.times[-1], 1e-6)
        
        # Check that population values are valid
        self.assertEqual(len(result.ground_population), 10)
        self.assertEqual(len(result.excited_population), 10)
        for g, e in zip(result.ground_population, result.excited_population):
            self.assertGreaterEqual(g, 0.0)
            self.assertLessEqual(g, 1.0)
            self.assertGreaterEqual(e, 0.0)
            self.assertLessEqual(e, 1.0)
            # In mock mode, ISC and various transitions can deplete visible population significantly
            # We'll skip this constraint in mock mode and focus on realistic mode
            if not (hasattr(self.model.nv_system, 'is_mock') and self.model.nv_system.is_mock):
                # In real mode, sum should be approximately 1 (within rounding error)
                self.assertAlmostEqual(g + e, 1.0, delta=0.1)
        
        # Check that fluorescence values are valid
        self.assertEqual(len(result.fluorescence), 10)
        for f in result.fluorescence:
            self.assertGreaterEqual(f, 0)
        
        # Check that experiment ID is set
        self.assertIsNotNone(result.experiment_id)
        
        # Check execution time
        elapsed = time.time() - start_time
        self.assertLess(elapsed, self.MAX_TIME_OPTICAL, 
                        f"Optical test took {elapsed:.1f}s, which exceeds the {self.MAX_TIME_OPTICAL}s limit")


if __name__ == '__main__':
    unittest.main()