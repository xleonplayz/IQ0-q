# -*- coding: utf-8 -*-

"""
Test module for the Qudi hardware interface adapters for the NV center simulator.

Copyright (c) 2023
"""

import unittest
import numpy as np
import sys
import os
import time

# Add the src directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the interfaces
from src.qudi_interface import NVSimulatorDevice, NVSimulatorMicrowave, NVSimulatorScanner
from qudi.util.enums import SamplingOutputMode


class TestQudiInterface(unittest.TestCase):
    """
    Test class for the Qudi hardware interface adapters.
    """
    
    def setUp(self):
        """
        Set up the test environment.
        """
        # Create a simulator device instance
        self.device = NVSimulatorDevice()
        self.microwave = self.device.microwave
        self.scanner = self.device.scanner
        
    def tearDown(self):
        """
        Clean up after the tests.
        """
        # Turn off any active outputs
        if self.microwave.module_state() != 'idle':
            self.microwave.off()
        
        if self.scanner.module_state() != 'idle':
            self.scanner.stop_buffered_acquisition()
    
    def test_microwave_initialization(self):
        """
        Test the initialization of the microwave interface.
        """
        # Check constraints
        self.assertIsNotNone(self.microwave.constraints)
        self.assertEqual(self.microwave.constraints.min_frequency, 2.7e9)
        self.assertEqual(self.microwave.constraints.max_frequency, 3.1e9)
        
        # Check default values
        self.assertEqual(self.microwave.cw_frequency, 2.87e9)
        self.assertEqual(self.microwave.cw_power, 0.0)
        self.assertFalse(self.microwave.is_scanning)
        
    def test_scanner_initialization(self):
        """
        Test the initialization of the scanner interface.
        """
        # Check constraints
        self.assertIsNotNone(self.scanner.constraints)
        self.assertEqual(self.scanner.constraints.min_frame_size, 1)
        self.assertEqual(self.scanner.constraints.max_frame_size, 1000)
        
        # Check default values
        self.assertEqual(self.scanner.sample_rate, 100.0)
        self.assertEqual(self.scanner.frame_size, 100)
        self.assertEqual(self.scanner.samples_in_buffer, 0)
        
    def test_cw_mode(self):
        """
        Test the CW mode of the microwave interface.
        """
        # Set CW parameters
        test_freq = 2.85e9
        test_power = -10.0
        self.microwave.set_cw(test_freq, test_power)
        
        # Check if parameters were set correctly
        self.assertEqual(self.microwave.cw_frequency, test_freq)
        self.assertEqual(self.microwave.cw_power, test_power)
        
        # Turn on CW mode
        self.microwave.cw_on()
        self.assertEqual(self.microwave.module_state(), 'locked')
        self.assertFalse(self.microwave.is_scanning)
        
        # Turn off
        self.microwave.off()
        self.assertEqual(self.microwave.module_state(), 'idle')
        
    def test_scan_mode(self):
        """
        Test the scan mode of the microwave interface.
        """
        # Set scan parameters - jump list mode
        test_power = -10.0
        test_freqs = np.linspace(2.8e9, 2.9e9, 11)
        test_mode = SamplingOutputMode.JUMP_LIST
        test_rate = 50.0
        
        self.microwave.configure_scan(test_power, test_freqs, test_mode, test_rate)
        
        # Check if parameters were set correctly
        self.assertEqual(self.microwave.scan_power, test_power)
        self.assertEqual(self.microwave.scan_mode, test_mode)
        self.assertEqual(self.microwave.scan_sample_rate, test_rate)
        np.testing.assert_array_equal(self.microwave.scan_frequencies, test_freqs)
        
        # Start scan
        self.microwave.start_scan()
        self.assertEqual(self.microwave.module_state(), 'locked')
        self.assertTrue(self.microwave.is_scanning)
        
        # Reset scan
        self.microwave.reset_scan()
        self.assertTrue(self.microwave.is_scanning)
        
        # Turn off
        self.microwave.off()
        self.assertEqual(self.microwave.module_state(), 'idle')
        self.assertFalse(self.microwave.is_scanning)
        
        # Test equidistant sweep mode
        test_freqs = (2.8e9, 2.9e9, 11)  # Start, stop, points
        test_mode = SamplingOutputMode.EQUIDISTANT_SWEEP
        
        self.microwave.configure_scan(test_power, test_freqs, test_mode, test_rate)
        
        # Check if frequencies were converted correctly
        expected_freqs = np.linspace(2.8e9, 2.9e9, 11)
        np.testing.assert_array_almost_equal(self.microwave.scan_frequencies, expected_freqs)
        
    def test_scanner_acquisition(self):
        """
        Test the scanner acquisition functionality.
        """
        # Set parameters
        test_frame_size = 10
        test_rate = 100.0
        
        self.scanner.set_frame_size(test_frame_size)
        self.scanner.set_sample_rate(test_rate)
        
        # Start acquisition
        self.scanner.start_buffered_acquisition()
        self.assertEqual(self.scanner.module_state(), 'locked')
        
        # Wait for samples
        time.sleep(0.2)  # Should be enough for some samples at 100Hz
        
        # Get samples
        self.assertGreater(self.scanner.samples_in_buffer, 0)
        samples = self.scanner.get_buffered_samples(5)
        
        # Check sample format
        self.assertIn('default', samples)
        self.assertEqual(len(samples['default']), 5)
        
        # Stop acquisition
        self.scanner.stop_buffered_acquisition()
        self.assertEqual(self.scanner.module_state(), 'idle')
        
    def test_acquire_frame(self):
        """
        Test the acquire_frame method of the scanner.
        """
        # Set parameters
        test_frame_size = 5
        
        # Acquire frame
        frame = self.scanner.acquire_frame(test_frame_size)
        
        # Check frame format
        self.assertIn('default', frame)
        self.assertEqual(len(frame['default']), test_frame_size)
        
        # Check data values are reasonable
        self.assertTrue(np.all(frame['default'] >= 0))  # Fluorescence should be positive
        
    def test_odmr_scan(self):
        """
        Test a simple ODMR scan through the interfaces.
        """
        # Configure microwave for scan
        test_power = -10.0
        test_freqs = np.linspace(2.85e9, 2.89e9, 5)
        test_mode = SamplingOutputMode.JUMP_LIST
        test_rate = 10.0
        
        self.microwave.configure_scan(test_power, test_freqs, test_mode, test_rate)
        
        # Configure scanner
        self.scanner.set_frame_size(5)
        self.scanner.set_sample_rate(test_rate)
        
        # Start microwave scan
        self.microwave.start_scan()
        
        # Acquire frame
        frame = self.scanner.acquire_frame()
        
        # Check results
        self.assertEqual(len(frame['default']), 5)
        
        # Should have measurable difference in fluorescence at different frequencies
        # Since we're scanning around the NV resonance
        self.assertGreater(np.std(frame['default']), 0)
        
        # Turn off microwave
        self.microwave.off()
        
    def test_run_simulation(self):
        """
        Test the run_simulation method of the device.
        """
        # Run ODMR simulation
        result = self.device.run_simulation('odmr', f_min=2.8e9, f_max=2.9e9, n_points=11)
        
        # Check result format
        self.assertIn('frequencies', result)
        self.assertIn('signal', result)
        self.assertEqual(len(result['frequencies']), 11)
        self.assertEqual(len(result['signal']), 11)


if __name__ == '__main__':
    unittest.main()