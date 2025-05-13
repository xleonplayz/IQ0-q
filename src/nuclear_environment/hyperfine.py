"""
Hyperfine interaction calculator for NV centers.

This module provides utilities for calculating hyperfine interactions
between an NV center electron spin and surrounding nuclear spins in diamond.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import sys
import os

# Configure logging
logger = logging.getLogger(__name__)

# Physical constants
MU_0 = 4 * np.pi * 1e-7  # Vacuum permeability (H/m)
HBAR = 1.054571817e-34  # Reduced Planck constant (J·s)
MU_B = 9.274010e-24  # Bohr magneton (J/T)
MU_N = 5.050783746e-27  # Nuclear magneton (J/T)
ELECTRON_GYROMAG = 28.0e9  # Electron gyromagnetic ratio (Hz/T)

# Import gyromagnetic ratios from spin_bath for consistency
from .spin_bath import GYROMAGNETIC_RATIOS


class HyperfineCalculator:
    """
    Calculator for hyperfine interactions between NV electron and nuclear spins.
    
    This class provides methods for calculating hyperfine tensors and
    resulting Hamiltonians for nuclear spins in the vicinity of an NV center.
    """
    
    def __init__(self, simos_compatible: bool = True):
        """
        Initialize hyperfine calculator.
        
        Parameters
        ----------
        simos_compatible : bool
            Whether to generate SimOS-compatible operators
        """
        self._simos_compatible = simos_compatible
    
    def calculate_dipolar_tensor(self, position: Tuple[float, float, float], 
                                gyro_ratio_nuclear: float) -> np.ndarray:
        """
        Calculate dipolar hyperfine tensor for a nuclear spin.
        
        Parameters
        ----------
        position : tuple
            (x, y, z) position of nuclear spin in meters
        gyro_ratio_nuclear : float
            Gyromagnetic ratio of the nuclear spin in Hz/T
            
        Returns
        -------
        numpy.ndarray
            3x3 dipolar hyperfine tensor in Hz
        """
        # Extract position components
        x, y, z = position
        r = np.sqrt(x**2 + y**2 + z**2)
        
        if r < 1e-15:  # Avoid division by zero
            logger.warning("Nuclear spin too close to NV center, returning zero tensor")
            return np.zeros((3, 3))
        
        # Unit vector pointing from NV to nuclear spin
        n = np.array([x, y, z]) / r
        
        # Calculate prefactor (in Hz)
        # A = μ0 * γe * γn * ħ / (4π * r³)
        prefactor = (MU_0 / (4 * np.pi)) * ELECTRON_GYROMAG * gyro_ratio_nuclear * (HBAR / (2 * np.pi)) / (r**3)
        prefactor *= 1e-6  # Convert to MHz for numerical stability
        
        # Calculate 3x3 dipolar tensor: A [3(S·n)(I·n) - S·I]
        # This is represented as a 3x3 matrix: A [3n⊗n - I] where I is identity
        tensor = prefactor * (3 * np.outer(n, n) - np.eye(3))
        
        return tensor * 1e6  # Convert back to Hz
    
    def calculate_contact_term(self, position: Tuple[float, float, float], 
                              species: str) -> float:
        """
        Calculate Fermi contact hyperfine interaction.
        
        Parameters
        ----------
        position : tuple
            (x, y, z) position of nuclear spin in meters
        species : str
            Nuclear species ('13C', '14N', etc.)
            
        Returns
        -------
        float
            Contact term in Hz
        """
        # Extract position components
        x, y, z = position
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # Get nuclear gyromagnetic ratio
        gyro_ratio_nuclear = GYROMAGNETIC_RATIOS.get(species, 0.0)
        
        # Calculate electron wavefunction density at nuclear position
        # Simplified model using exponential decay from NV center
        # In reality, this would be calculated using DFT or other quantum chemistry methods
        # For demonstration, we use a simple Gaussian decay
        wavefunction_density = 0.0
        
        # Different parameters for different species
        if species == '13C':
            # Carbon atoms in the lattice
            # Only significant for very close carbon atoms
            decay_length = 1.5e-10  # 1.5 Å
            amplitude = 0.01
            wavefunction_density = amplitude * np.exp(-(r / decay_length)**2)
        elif species in ('14N', '15N'):
            # Host nitrogen has strong contact term
            # This is more accurate for the host nitrogen
            if r < 2e-10:  # Only for host nitrogen
                wavefunction_density = 0.1  # Arbitrary units for demonstration
        
        # Calculate contact term (proportional to wavefunction density)
        # A_contact = (8π/3) * μ0 * γe * γn * ħ * |ψ(0)|²
        contact_term = (8 * np.pi / 3) * (MU_0 / (4 * np.pi)) * ELECTRON_GYROMAG * gyro_ratio_nuclear
        contact_term *= (HBAR / (2 * np.pi)) * wavefunction_density
        
        return contact_term
    
    def calculate_hyperfine_tensor(self, position: Tuple[float, float, float], 
                                  species: str) -> np.ndarray:
        """
        Calculate full hyperfine tensor combining dipolar and contact terms.
        
        Parameters
        ----------
        position : tuple
            (x, y, z) position of nuclear spin in meters
        species : str
            Nuclear species ('13C', '14N', etc.)
            
        Returns
        -------
        numpy.ndarray
            3x3 hyperfine tensor in Hz
        """
        # Get nuclear gyromagnetic ratio
        gyro_ratio_nuclear = GYROMAGNETIC_RATIOS.get(species, 0.0)
        
        # Calculate dipolar tensor
        dipolar_tensor = self.calculate_dipolar_tensor(position, gyro_ratio_nuclear)
        
        # Calculate contact term
        contact_term = self.calculate_contact_term(position, species)
        
        # Combine: full tensor = dipolar + contact * identity
        hyperfine_tensor = dipolar_tensor + contact_term * np.eye(3)
        
        return hyperfine_tensor
    
    def calculate_hyperfine_hamiltonian(self, nuclear_bath, nv_system=None):
        """
        Calculate hyperfine Hamiltonian for a nuclear spin bath.
        
        Parameters
        ----------
        nuclear_bath : NuclearSpinBath
            Nuclear spin bath configuration
        nv_system : object, optional
            SimOS NV system object (if using SimOS backend)
            
        Returns
        -------
        object
            SimOS Hamiltonian object or numpy matrix
        """
        if self._simos_compatible and nv_system is not None:
            # Use SimOS to calculate hyperfine Hamiltonian
            try:
                from simos.systems.NV import auto_pairwise_coupling
                
                # Get hyperfine couplings using SimOS
                h_hyperfine = auto_pairwise_coupling(
                    nv_system,
                    approx=False,
                    only_to_NV=True  # Only include coupling to NV electron
                )
                
                logger.info("Hyperfine Hamiltonian calculated using SimOS")
                return h_hyperfine
                
            except ImportError as e:
                logger.warning(f"Failed to use SimOS for hyperfine calculation: {e}")
                logger.info("Falling back to manual calculation")
        
        # Manual calculation if SimOS not available or not requested
        # This is a simplified implementation for demonstration
        # A full implementation would construct the proper quantum operators
        
        # Calculate hyperfine tensors for all spins
        h_components = []
        for spin in nuclear_bath.get_spins():
            tensor = self.calculate_hyperfine_tensor(spin.position, spin.species)
            h_components.append((spin, tensor))
        
        logger.info(f"Calculated hyperfine tensors for {len(h_components)} nuclear spins")
        
        # Return tensors for now (a complete implementation would construct operators)
        return h_components
    
    def calculate_quadrupolar_interaction(self, position: Tuple[float, float, float], 
                                         species: str) -> np.ndarray:
        """
        Calculate quadrupolar interaction tensor for I > 1/2 nuclei.
        
        Parameters
        ----------
        position : tuple
            (x, y, z) position of nuclear spin in meters
        species : str
            Nuclear species ('14N', etc.)
            
        Returns
        -------
        numpy.ndarray
            3x3 quadrupolar tensor in Hz
        """
        # Only valid for I > 1/2 nuclei like 14N (I=1)
        from .spin_bath import SPIN_QUANTUM
        spin = SPIN_QUANTUM.get(species, 0)
        
        if spin <= 0.5:
            # No quadrupolar interaction for I=1/2 nuclei
            return np.zeros((3, 3))
        
        # Calculate electric field gradient at nuclear position
        # In reality, this would be calculated from electronic structure
        # For demonstration, we use a simple model for 14N
        
        if species == '14N' and np.linalg.norm(position) < 2e-10:
            # Host nitrogen has strong quadrupolar interaction along NV axis
            # Principal values of the EFG tensor
            p_values = np.array([-5.1e6, -5.1e6, 10.2e6])  # Hz
            
            # Principal axes (z is along NV axis)
            tensor = np.diag(p_values)
            
            return tensor
        
        # For other nuclei or positions, return zero
        return np.zeros((3, 3))