"""
Tests for enhanced experimental interfaces in PhysicalNVModel.
"""

import sys
import os
import unittest
import numpy as np
from unittest.mock import patch
import tempfile

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import PhysicalNVModel, SimulationResult


class TestEnhancedExperiments(unittest.TestCase):
    """Test case for enhanced experimental interfaces."""

    def setUp(self):
        """Set up test case."""
        # Create model instance with quantum mechanical features
        self.model = PhysicalNVModel(
            zero_field_splitting=2.87e9,  # Hz
            gyromagnetic_ratio=28.0e9,    # Hz/T
            t1=5e-3,                      # s (5 ms)
            t2=500e-6,                    # s (500 µs)
            t2_star=1e-6,                 # s (1 µs)
            optics=True,                  # Include optical levels
            nitrogen=False,               # No nitrogen nuclear spin
            method="qutip"                # Use QuTiP backend
        )
        
        # Set a small magnetic field
        self.model.set_magnetic_field([0, 0, 2e-4])  # 0.2 mT along z

    def test_odmr_linewidth(self):
        """Test ODMR with linewidth analysis."""
        # Run a quick ODMR experiment
        result = self.model.simulate_odmr(
            f_min=2.86e9,
            f_max=2.88e9,
            n_points=11,  # Fewer points for speed
            mw_power=-10.0
        )
        
        # Verify type and basic properties
        self.assertEqual(result.type, "ODMR")
        self.assertEqual(len(result.frequencies), 11)
        self.assertEqual(len(result.signal), 11)
        
        # Check that center frequency is near ZFS range (SimOS vs config discrepancy)
        # SimOS uses 2.871 GHz, our config might have 2.87 GHz
        self.assertTrue(2.86e9 <= result.center_frequency <= 2.88e9)
        
        # Check that we have contrast
        self.assertGreater(result.contrast, 0.0)
        
        # Check if linewidth is present (may be None if no clear peak)
        if hasattr(result, "linewidth") and result.linewidth is not None:
            self.assertGreater(result.linewidth, 0.0)

    def test_t2_echo(self):
        """Test T2 echo experiment."""
        # Run a quick T2 echo experiment
        result = self.model.simulate_t2_echo(
            t_max=1e-3,
            n_points=5,  # Fewer points for speed
            mw_power=-5.0
        )
        
        # Verify type and basic properties
        self.assertEqual(result.type, "T2_Echo")
        self.assertEqual(len(result.times), 5)
        self.assertEqual(len(result.signal), 5)
        
        # Check if T2 is fitted
        if hasattr(result, "t2") and result.t2 is not None:
            # Should be close to the T2 value in config
            self.assertAlmostEqual(result.t2, self.model.config["t2"], delta=self.model.config["t2"]*0.5)

    def test_dynamical_decoupling(self):
        """Test dynamical decoupling sequences."""
        # Test various sequence types
        for seq_type in ["CPMG", "XY4", "XY8"]:
            # Run a quick DD experiment
            result = self.model.simulate_dynamical_decoupling(
                sequence_type=seq_type,
                t_max=5e-4,
                n_points=3,  # Minimum for test
                n_pulses=4,
                mw_power=-5.0
            )
            
            # Verify type and basic properties
            self.assertEqual(result.type, f"DynamicalDecoupling_{seq_type}")
            self.assertEqual(len(result.times), 3)
            self.assertEqual(len(result.signal), 3)
            self.assertEqual(result.n_pulses, 4)
            self.assertEqual(result.sequence_type, seq_type)

    def test_result_save_load(self):
        """Test saving and loading simulation results."""
        # Create a simple result
        result = SimulationResult(
            type="Test",
            data=np.array([1.0, 2.0, 3.0]),
            parameter=42
        )
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.npz') as tmp:
            result.save_data(tmp.name)
            
            # Load the result back
            loaded = SimulationResult.load_data(tmp.name)
            
            # Verify it loaded correctly
            self.assertEqual(loaded.type, "Test")
            np.testing.assert_array_equal(loaded.data, np.array([1.0, 2.0, 3.0]))
            self.assertEqual(loaded.parameter, 42)

    def test_result_plotting(self):
        """Test result plotting functionality."""
        # Create a simple ODMR result
        result = SimulationResult(
            type="ODMR",
            frequencies=np.linspace(2.86e9, 2.88e9, 11),
            signal=np.random.random(11),
            center_frequency=2.87e9,
            contrast=0.3,
            linewidth=5e6
        )
        
        # Try to plot (will fail if plot method has errors)
        try:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            result.plot(ax=ax)
            plt.close(fig)  # Clean up
        except ImportError:
            # Skip test if matplotlib not available
            pass


if __name__ == "__main__":
    unittest.main()