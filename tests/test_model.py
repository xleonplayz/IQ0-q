"""
Tests for the NV center simulator.
"""

import unittest
import numpy as np
from src import PhysicalNVModel


class TestPhysicalNVModel(unittest.TestCase):
    """Tests for the PhysicalNVModel class."""

    def setUp(self):
        """Set up the test environment."""
        self.model = PhysicalNVModel()
    
    def test_initialization(self):
        """Test if the model initializes correctly."""
        self.assertIsInstance(self.model, PhysicalNVModel)
        
        # Check default configuration
        self.assertEqual(self.model.config["zero_field_splitting"], 2.87e9)
        self.assertEqual(self.model.config["gyromagnetic_ratio"], 28.0e9)
        
        # Check initial state
        populations = self.model.get_populations()
        self.assertAlmostEqual(populations["ms0"], 1.0)
        self.assertAlmostEqual(populations["ms+1"], 0.0)
        self.assertAlmostEqual(populations["ms-1"], 0.0)
    
    def test_magnetic_field(self):
        """Test setting and getting magnetic field."""
        # Test default field
        field = self.model.magnetic_field
        self.assertEqual(len(field), 3)
        self.assertEqual(field[0], 0.0)
        self.assertEqual(field[1], 0.0)
        self.assertEqual(field[2], 0.0)
        
        # Test setting field
        self.model.set_magnetic_field([0.1, 0.2, 0.3])
        field = self.model.magnetic_field
        self.assertEqual(field[0], 0.1)
        self.assertEqual(field[1], 0.2)
        self.assertEqual(field[2], 0.3)
    
    def test_microwave(self):
        """Test applying microwave."""
        # Default state
        self.assertFalse(self.model.mw_on)
        
        # Apply microwave
        self.model.apply_microwave(2.87e9, -10.0, True)
        self.assertTrue(self.model.mw_on)
        self.assertEqual(self.model.mw_frequency, 2.87e9)
        self.assertEqual(self.model.mw_power, -10.0)
        
        # Turn off microwave
        self.model.apply_microwave(2.87e9, -10.0, False)
        self.assertFalse(self.model.mw_on)
    
    def test_reset_state(self):
        """Test resetting the quantum state."""
        # Change state
        self.model.initialize_state(ms="-1")
        populations = self.model.get_populations()
        self.assertAlmostEqual(populations["ms-1"], 1.0)
        
        # Reset state
        self.model.reset_state()
        populations = self.model.get_populations()
        self.assertAlmostEqual(populations["ms0"], 1.0)
    
    def test_initialize_state(self):
        """Test initializing to specific states."""
        # Test ms=0
        self.model.initialize_state(ms="0")
        populations = self.model.get_populations()
        self.assertAlmostEqual(populations["ms0"], 1.0)
        
        # Test ms=+1
        self.model.initialize_state(ms="+1")
        populations = self.model.get_populations()
        self.assertAlmostEqual(populations["ms+1"], 1.0)
        
        # Test ms=-1
        self.model.initialize_state(ms="-1")
        populations = self.model.get_populations()
        self.assertAlmostEqual(populations["ms-1"], 1.0)
    
    def test_simulate(self):
        """Test simulating evolution."""
        # Initialize to ms=0
        self.model.reset_state()
        
        # Apply resonant microwave
        zfs = self.model.config["zero_field_splitting"]
        self.model.apply_microwave(zfs, 0.0, True)
        
        # Simulate evolution
        self.model.simulate(1e-6)  # 1 µs
        
        # Check populations changed
        populations = self.model.get_populations()
        self.assertLess(populations["ms0"], 0.99)
    
    def test_odmr(self):
        """Test ODMR simulation."""
        # Set magnetic field
        self.model.set_magnetic_field([0, 0, 0.001])  # 1 mT
        
        # Calculate expected resonance
        zfs = self.model.config["zero_field_splitting"]
        gamma = self.model.config["gyromagnetic_ratio"]
        b_z = 0.001  # 1 mT
        expected_resonance = zfs + gamma * b_z
        
        # Run ODMR around expected resonance
        result = self.model.simulate_odmr(
            expected_resonance - 50e6,
            expected_resonance + 50e6,
            11,
            -10.0
        )
        
        # Check result
        self.assertEqual(result.type, "ODMR")
        self.assertIsNotNone(result.center_frequency)
        
        # Resonance frequency should be close to expected
        self.assertLess(abs(result.center_frequency - expected_resonance), 10e6)
    
    def test_rabi(self):
        """Test Rabi oscillation simulation."""
        # Set resonant microwave
        zfs = self.model.config["zero_field_splitting"]
        
        # Run Rabi experiment
        result = self.model.simulate_rabi(5e-6, 21, zfs, 0.0)
        
        # Check result
        self.assertEqual(result.type, "Rabi")
        
        # Should show oscillation in ms=0 population
        ms0_pop = result.populations[:, 0]
        amplitude = np.max(ms0_pop) - np.min(ms0_pop)
        self.assertGreater(amplitude, 0.1)


if __name__ == "__main__":
    unittest.main()