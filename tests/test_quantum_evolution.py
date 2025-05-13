"""
Tests for quantum evolution in PhysicalNVModel.
"""

import sys
import os
import unittest
import numpy as np
from unittest.mock import patch

# Add parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import PhysicalNVModel, SimulationResult


class TestQuantumEvolution(unittest.TestCase):
    """Test case for quantum evolution methods."""

    def setUp(self):
        """Set up test case."""
        # Create model instance with quantum mechanical features
        self.model = PhysicalNVModel(
            zero_field_splitting=2.87e9,  # Hz
            gyromagnetic_ratio=28.0e9,    # Hz/T
            t1=5e-3,                      # s
            t2=500e-6,                    # s
            t2_star=1e-6,                 # s
            optics=True,                  # Include optical levels
            nitrogen=False,               # No nitrogen nuclear spin
            method="qutip"                # Use QuTiP backend
        )

    def test_pulse_sequence(self):
        """Test basic pulse sequence execution."""
        # Define a simple sequence
        sequence = [
            # Initial pi/2 pulse
            ("pi/2", None, {"power": -10.0}),
            
            # Wait
            ("wait", 1e-6, {}),
            
            # Final pi/2 pulse
            ("pi/2", None, {"power": -10.0, "measure": True})
        ]
        
        # Execute the sequence
        result = self.model.evolve_pulse_sequence(sequence)
        
        # Verify result contains expected keys
        self.assertIn("final_state", result)
        self.assertIn("final_population", result)
        self.assertIn("final_fluorescence", result)

    def test_ramsey_evolution(self):
        """Test Ramsey sequence."""
        # Execute a Ramsey sequence
        result = self.model.evolve_with_ramsey(
            free_evolution_time=1e-6,
            mw_power=-10.0
        )
        
        # Verify result contains expected keys
        self.assertIn("final_state", result)
        self.assertIn("final_population", result)
        self.assertIn("final_fluorescence", result)

    def test_spin_echo_evolution(self):
        """Test spin echo sequence."""
        # Execute a spin echo sequence
        result = self.model.evolve_with_spin_echo(
            free_evolution_time=1e-6,
            mw_power=-10.0
        )
        
        # Verify result contains expected keys
        self.assertIn("final_state", result)
        self.assertIn("final_population", result)
        self.assertIn("final_fluorescence", result)

    def test_t1_experiment(self):
        """Test T1 relaxation experiment."""
        # Set up and run T1 experiment
        result = self.model.simulate_t1(t_max=1e-3, n_points=5)
        
        # Verify result type
        self.assertEqual(result.type, "T1")
        
        # Verify time points
        self.assertEqual(len(result.times), 5)
        
        # Verify population shape
        self.assertEqual(result.populations.shape, (5, 3))

    def test_ramsey_experiment(self):
        """Test Ramsey (T2*) experiment."""
        # Set up and run Ramsey experiment
        result = self.model.simulate_ramsey(t_max=1e-6, n_points=5, detuning=1e6)
        
        # Verify result type
        self.assertEqual(result.type, "Ramsey")
        
        # Verify time points
        self.assertEqual(len(result.times), 5)
        
        # Verify fluorescence data
        self.assertEqual(len(result.fluorescence), 5)

    def test_spin_echo_experiment(self):
        """Test spin echo (T2) experiment."""
        # Set up and run spin echo experiment
        result = self.model.simulate_spin_echo(t_max=1e-3, n_points=5)
        
        # Verify result type
        self.assertEqual(result.type, "Spin Echo")
        
        # Verify time points
        self.assertEqual(len(result.times), 5)
        
        # Verify fluorescence data
        self.assertEqual(len(result.fluorescence), 5)


if __name__ == "__main__":
    unittest.main()