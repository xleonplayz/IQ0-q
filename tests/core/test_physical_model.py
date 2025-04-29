"""
Unit tests for the PhysicalNVModel class.
"""

import unittest
import numpy as np
from unittest.mock import patch

from simos_nv_simulator.core.physical_model import PhysicalNVModel


class TestPhysicalNVModel(unittest.TestCase):
    """Test suite for the PhysicalNVModel class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.model = PhysicalNVModel()
    
    def test_initialization_with_default_config(self):
        """Test initialization with default configuration."""
        # Check that the model initializes with default config
        self.assertIsNotNone(self.model.config)
        self.assertEqual(self.model.config['zero_field_splitting'], 2.87e9)
        self.assertEqual(self.model.config['strain'], 5e6)
        self.assertEqual(self.model.mw_frequency, 2.87e9)  # Should default to ZFS
        self.assertEqual(self.model.mw_power, 0.0)
        self.assertFalse(self.model.mw_on)
        self.assertEqual(self.model.laser_power, 0.0)
        self.assertFalse(self.model.laser_on)
        np.testing.assert_array_equal(self.model.magnetic_field, np.array([0.0, 0.0, 0.0]))
    
    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = {
            'zero_field_splitting': 2.88e9,
            'strain': 6e6,
            'T1': 2e-3,
            'simulation_timestep': 2e-9
        }
        model = PhysicalNVModel(config=custom_config)
        
        # Check that custom values are set
        self.assertEqual(model.config['zero_field_splitting'], 2.88e9)
        self.assertEqual(model.config['strain'], 6e6)
        self.assertEqual(model.config['T1'], 2e-3)
        self.assertEqual(model.dt, 2e-9)
        
        # Check that other values are set to defaults
        self.assertEqual(model.config['gyromagnetic_ratio'], 2.8025e10)
    
    def test_set_magnetic_field(self):
        """Test setting the magnetic field."""
        test_field = [0.1, 0.2, 0.3]
        self.model.set_magnetic_field(test_field)
        np.testing.assert_array_equal(self.model.magnetic_field, np.array(test_field))
        
        # Test with numpy array
        test_field_np = np.array([0.4, 0.5, 0.6])
        self.model.set_magnetic_field(test_field_np)
        np.testing.assert_array_equal(self.model.magnetic_field, test_field_np)
    
    def test_get_magnetic_field(self):
        """Test getting the magnetic field."""
        test_field = [0.1, 0.2, 0.3]
        self.model.set_magnetic_field(test_field)
        
        # Get the field and verify it's a copy, not a reference
        field = self.model.get_magnetic_field()
        np.testing.assert_array_equal(field, np.array(test_field))
        
        # Modify the returned field and verify the original is unchanged
        field[0] = 99.9
        np.testing.assert_array_equal(self.model.magnetic_field, np.array(test_field))
    
    def test_apply_microwave(self):
        """Test applying microwave excitation."""
        frequency = 2.9e9
        power = -10.0
        
        # Test turning microwave on
        self.model.apply_microwave(frequency, power, True)
        self.assertEqual(self.model.mw_frequency, frequency)
        self.assertEqual(self.model.mw_power, power)
        self.assertTrue(self.model.mw_on)
        
        # Test turning microwave off
        self.model.apply_microwave(frequency, power, False)
        self.assertEqual(self.model.mw_frequency, frequency)
        self.assertEqual(self.model.mw_power, power)
        self.assertFalse(self.model.mw_on)
    
    def test_apply_laser(self):
        """Test applying laser excitation."""
        power = 5.0
        
        # Test turning laser on
        self.model.apply_laser(power, True)
        self.assertEqual(self.model.laser_power, power)
        self.assertTrue(self.model.laser_on)
        
        # Test turning laser off
        self.model.apply_laser(power, False)
        self.assertEqual(self.model.laser_power, power)
        self.assertFalse(self.model.laser_on)
    
    def test_reset_state(self):
        """Test resetting the quantum state."""
        # Set some non-default values
        self.model.apply_microwave(2.9e9, -10.0, True)
        self.model.apply_laser(5.0, True)
        
        # Reset state
        self.model.reset_state()
        
        # Verify control parameters are reset
        self.assertFalse(self.model.mw_on)
        self.assertFalse(self.model.laser_on)
    
    def test_get_config(self):
        """Test getting the configuration."""
        config = self.model.get_config()
        
        # Verify it's a copy, not a reference
        self.assertEqual(config, self.model.config)
        self.assertIsNot(config, self.model.config)
        
        # Modify the returned config and verify the original is unchanged
        original_zfs = self.model.config['zero_field_splitting']
        config['zero_field_splitting'] = 99e9
        self.assertEqual(self.model.config['zero_field_splitting'], original_zfs)
    
    def test_update_config(self):
        """Test updating the configuration."""
        updates = {
            'zero_field_splitting': 2.88e9,
            'T1': 2e-3,
            'new_param': 'test_value'
        }
        
        self.model.update_config(updates)
        
        # Verify updates are applied
        self.assertEqual(self.model.config['zero_field_splitting'], 2.88e9)
        self.assertEqual(self.model.config['T1'], 2e-3)
        self.assertEqual(self.model.config['new_param'], 'test_value')
        
        # Verify other values are unchanged
        self.assertEqual(self.model.config['strain'], 5e6)
    
    def test_get_state_info(self):
        """Test getting state information."""
        # Set some state
        self.model.set_magnetic_field([0.1, 0.2, 0.3])
        self.model.apply_microwave(2.9e9, -10.0, True)
        self.model.apply_laser(5.0, True)
        
        # Get state info
        info = self.model.get_state_info()
        
        # Verify state info
        self.assertEqual(info['magnetic_field'], [0.1, 0.2, 0.3])
        self.assertEqual(info['mw_frequency'], 2.9e9)
        self.assertEqual(info['mw_power'], -10.0)
        self.assertTrue(info['mw_on'])
        self.assertEqual(info['laser_power'], 5.0)
        self.assertTrue(info['laser_on'])
    
    @patch('threading.RLock')
    def test_thread_safety(self, mock_lock):
        """Test that methods use thread lock properly."""
        model = PhysicalNVModel()
        
        # Test various methods to ensure they acquire the lock
        model.set_magnetic_field([0.1, 0.2, 0.3])
        model.get_magnetic_field()
        model.apply_microwave(2.9e9, -10.0)
        model.apply_laser(5.0)
        model.reset_state()
        model.get_config()
        model.update_config({'test': 'value'})
        model.get_state_info()
        
        # Verify the lock's __enter__ and __exit__ were called for each method
        # 8 methods * 2 calls per context manager = 16 calls
        self.assertEqual(mock_lock.return_value.__enter__.call_count, 8)
        self.assertEqual(mock_lock.return_value.__exit__.call_count, 8)


if __name__ == '__main__':
    unittest.main()