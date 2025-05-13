"""
Basic tests for SimOS integration in PhysicalNVModel.
"""

import sys
import os
import unittest
import numpy as np
from unittest.mock import patch

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import PhysicalNVModel, SimulationResult


class TestSimOSIntegration(unittest.TestCase):
    """Test case for SimOS integration into PhysicalNVModel."""

    def setUp(self):
        """Set up test case."""
        # Create model instance with simple configuration
        self.model = PhysicalNVModel(
            zero_field_splitting=2.87e9,
            gyromagnetic_ratio=28.0e9,
            t1=1e-3,
            t2=500e-6,
            optics=True,
            method="qutip"
        )

    def test_initialization(self):
        """Test proper initialization of quantum state."""
        # State should be initialized to ms=0
        populations = self.model.get_populations()
        
        # ms=0 population should be close to 1.0
        self.assertAlmostEqual(populations['ms0'], 1.0, delta=0.01)
        self.assertAlmostEqual(populations['ms+1'], 0.0, delta=0.01)
        self.assertAlmostEqual(populations['ms-1'], 0.0, delta=0.01)

    def test_hamiltonian_construction(self):
        """Test that Hamiltonian is properly constructed from SimOS."""
        # Initialize with non-zero magnetic field
        model = PhysicalNVModel(
            optics=False,  # Only spin system for simplicity
            method="qutip"
        )
        
        # Check that the Hamiltonian has been initialized
        self.assertTrue(hasattr(model, '_H_free'))
        self.assertIsNotNone(model._H_free)

    def test_state_manipulation(self):
        """Test state manipulation methods."""
        # Initialize in ms=+1 state
        self.model.initialize_state(ms="+1")
        populations = self.model.get_populations()
        self.assertAlmostEqual(populations['ms+1'], 1.0, delta=0.01)
        
        # Reset to ms=0
        self.model.reset_state()
        populations = self.model.get_populations()
        self.assertAlmostEqual(populations['ms0'], 1.0, delta=0.01)

    def test_fluorescence_calculation(self):
        """Test fluorescence calculation."""
        # ms=0 should have higher fluorescence
        self.model.initialize_state(ms="0")
        fluor_ms0 = self.model.get_fluorescence()
        
        # ms=±1 should have lower fluorescence
        self.model.initialize_state(ms="+1")
        fluor_ms1 = self.model.get_fluorescence()
        
        # Verify contrast
        self.assertGreater(fluor_ms0, fluor_ms1)
        contrast = (fluor_ms0 - fluor_ms1) / fluor_ms0
        self.assertAlmostEqual(contrast, 0.3, delta=0.05)  # Should be ~30%


if __name__ == "__main__":
    unittest.main()