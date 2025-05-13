"""
Tests for the nuclear spin environment module.
"""

import unittest
import numpy as np
import sys
import os

# Add the path to the source directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, 'src')
sys.path.insert(0, parent_dir)

# Try to import the required modules
try:
    from src.nuclear_environment import (
        NuclearSpinBath, SpinConfig, HyperfineCalculator, 
        NuclearControl, SpinBathDecoherence
    )
    from src import PhysicalNVModel
    _IMPORTS_SUCCEEDED = True
except ImportError as e:
    print(f"Import error: {e}")
    _IMPORTS_SUCCEEDED = False


@unittest.skipIf(not _IMPORTS_SUCCEEDED, "Nuclear environment modules not available")
class TestNuclearSpinBath(unittest.TestCase):
    """Tests for the NuclearSpinBath class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.bath = NuclearSpinBath(
            concentration=0.01,  # 1% abundance
            bath_size=10,
            random_seed=42  # For reproducibility
        )
    
    def test_initialization(self):
        """Test if the bath initializes correctly."""
        self.assertIsInstance(self.bath, NuclearSpinBath)
        
        # Check if spins were generated
        self.assertEqual(len(self.bath), 10)
        
        # Check if SpinConfig objects are created
        spins = self.bath.get_spins()
        self.assertTrue(all(isinstance(s, SpinConfig) for s in spins))
    
    def test_custom_spin(self):
        """Test adding a custom spin."""
        bath = NuclearSpinBath(bath_size=0)  # Empty bath
        
        # Add a custom spin
        position = (1e-10, 2e-10, 3e-10)
        spin = bath.add_custom_spin(position, species='13C', index=0, name='test_spin')
        
        # Check if the spin was added
        self.assertEqual(len(bath), 1)
        
        # Check spin properties
        self.assertEqual(spin.position, position)
        self.assertEqual(spin.species, '13C')
        self.assertEqual(spin.name, 'test_spin')
        
        # Test gyromagnetic ratio accessor
        self.assertAlmostEqual(spin.gyromagnetic_ratio, 10.705e6, delta=1e3)
    
    def test_get_spins_by_species(self):
        """Test filtering spins by species."""
        # All spins should be 13C in our test setup
        c13_spins = self.bath.get_spins(species='13C')
        self.assertEqual(len(c13_spins), 10)
        
        # No 14N spins in default setup
        n14_spins = self.bath.get_spins(species='14N')
        self.assertEqual(len(n14_spins), 0)
        
        # Add a nitrogen spin
        bath = NuclearSpinBath(bath_size=0)  # Empty bath
        bath.add_custom_spin((0, 0, 1e-10), species='14N')
        
        # Check if it's returned correctly
        n14_spins = bath.get_spins(species='14N')
        self.assertEqual(len(n14_spins), 1)


@unittest.skipIf(not _IMPORTS_SUCCEEDED, "Nuclear environment modules not available")
class TestHyperfineCalculator(unittest.TestCase):
    """Tests for the HyperfineCalculator class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.calculator = HyperfineCalculator()
        
        # Create a simple bath with one spin
        self.bath = NuclearSpinBath(bath_size=0)
        self.bath.add_custom_spin((0, 0, 2e-10), species='13C')  # 2 Å along z-axis
    
    def test_initialization(self):
        """Test if the calculator initializes correctly."""
        self.assertIsInstance(self.calculator, HyperfineCalculator)
    
    def test_dipolar_tensor(self):
        """Test calculation of dipolar tensor."""
        # Get the spin
        spin = self.bath.get_spins()[0]
        
        # Calculate tensor
        tensor = self.calculator.calculate_dipolar_tensor(
            spin.position, spin.gyromagnetic_ratio
        )
        
        # Check tensor shape
        self.assertEqual(tensor.shape, (3, 3))
        
        # For a spin along z-axis, the tensor should be diagonal in this form
        # with Azz = 2*A and Axx = Ayy = -A
        # Check if Azz is approximately twice the magnitude of Axx
        self.assertAlmostEqual(tensor[2, 2] / tensor[0, 0], -2.0, delta=0.1)
        
        # Check if Axx == Ayy
        self.assertAlmostEqual(tensor[0, 0] / tensor[1, 1], 1.0, delta=0.01)
    
    def test_hyperfine_tensor(self):
        """Test calculation of full hyperfine tensor."""
        # Get the spin
        spin = self.bath.get_spins()[0]
        
        # Calculate tensor
        tensor = self.calculator.calculate_hyperfine_tensor(
            spin.position, spin.species
        )
        
        # Check tensor shape
        self.assertEqual(tensor.shape, (3, 3))
        
        # The tensor should follow the dipolar form for distant spins
        # with corrections from the contact term
        # We just check it's reasonable in magnitude
        # (typical values for a 13C at 2Å are 10-100 kHz)
        self.assertTrue(np.max(np.abs(tensor)) > 1e3)  # > 1 kHz
        self.assertTrue(np.max(np.abs(tensor)) < 1e6)  # < 1 MHz


@unittest.skipIf(not _IMPORTS_SUCCEEDED, "Nuclear environment modules not available")
class TestNuclearControl(unittest.TestCase):
    """Tests for the NuclearControl class."""
    
    def setUp(self):
        """Set up the test environment."""
        self.control = NuclearControl()
    
    def test_initialization(self):
        """Test if the control initializes correctly."""
        self.assertIsInstance(self.control, NuclearControl)
    
    def test_rf_hamiltonian(self):
        """Test calculation of RF Hamiltonian parameters."""
        # Calculate RF parameters
        params = self.control.calculate_rf_hamiltonian(
            frequency=500e3,    # 500 kHz
            power=0.1,          # 0.1 W
            phase=0.0,
            target_species='13C',
            polarization='x'
        )
        
        # Check if parameters are returned
        self.assertIn('frequency', params)
        self.assertIn('rabi_frequency', params)
        self.assertIn('direction', params)
        
        # Check values
        self.assertEqual(params['frequency'], 500e3)
        self.assertTrue(params['rabi_frequency'] > 0)
        self.assertEqual(len(params['direction']), 3)
        
        # For x-polarization, direction should be along x
        self.assertAlmostEqual(params['direction'][0], 1.0)
        self.assertAlmostEqual(params['direction'][1], 0.0)
        self.assertAlmostEqual(params['direction'][2], 0.0)


@unittest.skipIf(not _IMPORTS_SUCCEEDED, "Nuclear environment modules not available")
class TestSpinBathDecoherence(unittest.TestCase):
    """Tests for the SpinBathDecoherence class."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a bath with 5 spins
        self.bath = NuclearSpinBath(bath_size=5, random_seed=42)
        self.decoherence = SpinBathDecoherence(self.bath)
    
    def test_initialization(self):
        """Test if the decoherence model initializes correctly."""
        self.assertIsInstance(self.decoherence, SpinBathDecoherence)
        self.assertEqual(self.decoherence.spin_bath, self.bath)
    
    def test_t2_star(self):
        """Test calculation of T2*."""
        # Calculate T2*
        t2_star = self.decoherence.calculate_t2_star()
        
        # T2* should be positive and finite
        self.assertTrue(t2_star > 0)
        self.assertTrue(np.isfinite(t2_star))
        
        # T2* should be in a reasonable range for NV centers (ns to µs)
        self.assertTrue(t2_star > 1e-9)   # > 1 ns
        self.assertTrue(t2_star < 1e-5)   # < 10 µs
    
    def test_t2(self):
        """Test calculation of T2 from cluster expansion."""
        # Calculate T2
        t2 = self.decoherence.calculate_t2_from_cluster_expansion(max_order=2)
        
        # T2 should be positive and finite
        self.assertTrue(t2 > 0)
        self.assertTrue(np.isfinite(t2))
        
        # T2 should be in a reasonable range for NV centers (µs to ms)
        self.assertTrue(t2 > 1e-6)   # > 1 µs
        self.assertTrue(t2 < 1e-2)   # < 10 ms
        
        # T2 should be longer than T2*
        t2_star = self.decoherence.calculate_t2_star()
        self.assertTrue(t2 > t2_star)
    
    def test_sequence_decoherence(self):
        """Test sequence-specific decoherence model."""
        # Test times
        times = np.linspace(0, 100e-6, 101)  # 0 to 100 µs
        
        # Calculate decoherence for Hahn echo
        hahn_coherence = self.decoherence.apply_decoherence_to_sequence(
            times, sequence_type='hahn'
        )
        
        # Calculate decoherence for CPMG
        cpmg_coherence = self.decoherence.apply_decoherence_to_sequence(
            times, sequence_type='cpmg'
        )
        
        # Check shapes
        self.assertEqual(len(hahn_coherence), len(times))
        self.assertEqual(len(cpmg_coherence), len(times))
        
        # Coherence should start at 1 and decrease
        self.assertAlmostEqual(hahn_coherence[0], 1.0, delta=0.01)
        self.assertAlmostEqual(cpmg_coherence[0], 1.0, delta=0.01)
        
        # CPMG should have better coherence than Hahn echo
        # (at least at the middle of the time range)
        mid_idx = len(times) // 2
        self.assertTrue(cpmg_coherence[mid_idx] > hahn_coherence[mid_idx])


@unittest.skipIf(not _IMPORTS_SUCCEEDED, "Nuclear environment modules not available")
class TestModelIntegration(unittest.TestCase):
    """Tests for the integration with PhysicalNVModel."""
    
    def setUp(self):
        """Set up the test environment."""
        self.model = PhysicalNVModel(
            nuclear_spins=True,
            c13_concentration=0.01,
            bath_size=5,
            random_seed=42
        )
    
    def test_initialization(self):
        """Test if the model initializes with nuclear spins."""
        self.assertTrue(self.model._nuclear_enabled)
        self.assertIsNotNone(self.model.nuclear_bath)
        self.assertIsNotNone(self.model.hyperfine_calculator)
        self.assertIsNotNone(self.model.nuclear_control)
        self.assertIsNotNone(self.model.decoherence_model)
    
    def test_coherence_times(self):
        """Test calculation of coherence times."""
        coherence_times = self.model.calculate_coherence_times()
        
        self.assertIn('t1', coherence_times)
        self.assertIn('t2', coherence_times)
        self.assertIn('t2_star', coherence_times)
        
        # Check values are reasonable
        self.assertTrue(coherence_times['t1'] > 0)
        self.assertTrue(coherence_times['t2'] > 0)
        self.assertTrue(coherence_times['t2_star'] > 0)
        
        # T1 > T2 > T2*
        self.assertTrue(coherence_times['t1'] > coherence_times['t2'])
        self.assertTrue(coherence_times['t2'] > coherence_times['t2_star'])


if __name__ == '__main__':
    unittest.main()