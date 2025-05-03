"""
Unit tests for the PhysicalNVModel class.
"""

import unittest
import numpy as np
from unittest.mock import patch, MagicMock
import threading
import time

from simos_nv_simulator.core.physical_model import PhysicalNVModel, ODMRResult, RabiResult, T1Result


class TestPhysicalNVModel(unittest.TestCase):
    """Test suite for the PhysicalNVModel class."""
    
    # Maximum run times for time-sensitive tests
    MAX_TIME_ODMR = 60  # seconds
    MAX_TIME_RABI = 30  # seconds
    MAX_TIME_T1 = 5  # seconds

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
        model.get_fluorescence()
        
        # Also check the simulation control methods
        model.start_simulation_loop()
        model.stop_simulation_loop()
        
        # Verify the lock's __enter__ and __exit__ were called for each method
        # Each method should use a context manager that calls __enter__ and __exit__
        self.assertGreaterEqual(mock_lock.return_value.__enter__.call_count, 10) 
        self.assertGreaterEqual(mock_lock.return_value.__exit__.call_count, 10)


    def test_get_fluorescence(self):
        """Test getting fluorescence counts."""
        # Check that fluorescence returns a reasonable number
        fluor = self.model.get_fluorescence()
        self.assertGreater(fluor, 0.0)
        self.assertLess(fluor, 2e6)  # Should be less than max count rate
        
        # Test fluorescence contrast with different spin states
        self.model.reset_state()  # Initialize to ms=0
        ms0_fluor = self.model.get_fluorescence()
        
        # Apply microwave to change state to ms=-1/+1
        self.model.apply_microwave(self.model.config['zero_field_splitting'], 0.0, True)
        
        # Let the state evolve a bit
        for _ in range(100):
            if hasattr(self.model.nv_system, 'evolve'):
                self.model.nv_system.evolve(self.model.dt)
        
        # Measure fluorescence in the new state
        ms1_fluor = self.model.get_fluorescence()
        
        # Fluorescence in ms=0 should be higher than ms=±1 states
        self.assertGreater(ms0_fluor, ms1_fluor)
        
    def test_simulation_loop(self):
        """Test starting and stopping simulation loop."""
        # Check initial state
        self.assertFalse(self.model.is_simulating)
        
        # Start simulation
        self.model.start_simulation_loop()
        self.assertTrue(self.model.is_simulating)
        self.assertTrue(self.model.simulation_thread.is_alive())
        
        # Stop simulation
        self.model.stop_simulation_loop()
        self.assertFalse(self.model.is_simulating)
        
        # Wait a bit for thread to fully stop
        time.sleep(0.1)
        self.assertFalse(self.model.simulation_thread.is_alive())
        
    def test_simulate_odmr(self):
        """Test ODMR simulation."""
        import time
        start_time = time.time()
        # Run ODMR simulation around zero-field splitting
        zfs = self.model.config['zero_field_splitting']
        # Use small frequency range and very short averaging time for test
        result = self.model.simulate_odmr(zfs - 10e6, zfs + 10e6, 5, 0.001)  # Short averaging time
        
        # Check result type
        self.assertIsInstance(result, ODMRResult)
        
        # Check that frequencies are as expected
        self.assertEqual(len(result.frequencies), 5)
        self.assertEqual(result.frequencies[0], zfs - 10e6)
        self.assertEqual(result.frequencies[-1], zfs + 10e6)
        
        # Check that signal values are valid
        self.assertEqual(len(result.signal), 5)
        for s in result.signal:
            self.assertGreaterEqual(s, 0.5)  # Normalized, shouldn't go too low
            self.assertLessEqual(s, 1.5)  # Might be above 1 due to noise
            
        # Check that the result is cached
        self.assertTrue(any('odmr_' in k for k in self.model.cached_results.keys()))
        
        # Check that center frequency is in a reasonable range
        self.assertGreaterEqual(result.center_frequency, zfs - 50e6)
        self.assertLessEqual(result.center_frequency, zfs + 50e6)
        
        # Check that contrast and linewidth exist
        self.assertIsNotNone(result.contrast)
        self.assertIsNotNone(result.linewidth)
        
        # Check that experiment ID is set
        self.assertIsNotNone(result.experiment_id)
        
        # Check execution time
        elapsed = time.time() - start_time
        self.assertLess(elapsed, self.MAX_TIME_ODMR, f"ODMR test took {elapsed:.1f}s, which exceeds the {self.MAX_TIME_ODMR}s limit")
        
    def test_simulate_rabi(self):
        """Test Rabi oscillation simulation."""
        import time
        start_time = time.time()
        # Run Rabi simulation with small number of points for speed
        result = self.model.simulate_rabi(1e-7, 5)  # 100 ns, 5 points
        
        # Check result type
        self.assertIsInstance(result, RabiResult)
        
        # Check that times are as expected
        self.assertEqual(len(result.times), 5)
        self.assertEqual(result.times[0], 0.0)
        self.assertEqual(result.times[-1], 1e-7)
        
        # Check that population values are valid
        self.assertEqual(len(result.population), 5)
        for p in result.population:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
            
        # Check that the result is cached
        self.assertTrue(any('rabi_' in k for k in self.model.cached_results.keys()))
        
        # Check that experiment ID is set
        self.assertIsNotNone(result.experiment_id)
        
        # Check execution time
        elapsed = time.time() - start_time
        self.assertLess(elapsed, self.MAX_TIME_RABI, f"Rabi test took {elapsed:.1f}s, which exceeds the {self.MAX_TIME_RABI}s limit")
        
    def test_simulate_t1(self):
        """Test T1 relaxation simulation."""
        import time
        start_time = time.time()
        # Set a known T1 time
        self.model.update_config({'T1': 1e-4})  # 100 µs (shorter for test speed)
        
        # Run T1 simulation (very brief for test)
        result = self.model.simulate_t1(3e-4, 3)  # 300 µs, 3 points (short test)
        
        # Check result type
        self.assertIsInstance(result, T1Result)
        
        # Check that times are as expected
        self.assertEqual(len(result.times), 3)
        self.assertEqual(result.times[0], 0.0)
        self.assertEqual(result.times[-1], 3e-4)
        
        # Check that population values are valid
        self.assertEqual(len(result.population), 3)
        for p in result.population:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)
            
        # Check that the result is cached
        self.assertTrue(any('t1_' in k for k in self.model.cached_results.keys()))
        
        # Check that experiment ID is set
        self.assertIsNotNone(result.experiment_id)
        
        # Check execution time
        elapsed = time.time() - start_time
        self.assertLess(elapsed, self.MAX_TIME_T1, f"T1 test took {elapsed:.1f}s, which exceeds the {self.MAX_TIME_T1}s limit")
        
if __name__ == '__main__':
    unittest.main()