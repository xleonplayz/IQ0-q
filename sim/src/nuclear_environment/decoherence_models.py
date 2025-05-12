"""
Decoherence models for NV center systems with nuclear spin environments.

This module provides classes and functions for modeling decoherence effects
due to nuclear spin baths, including T2* dephasing, T2 relaxation, and
spectral diffusion processes.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import sys
import os

# Configure logging
logger = logging.getLogger(__name__)

# Import spin bath components for consistency
from .spin_bath import GYROMAGNETIC_RATIOS, SPIN_QUANTUM


class SpinBathDecoherence:
    """
    Class for modeling decoherence effects from nuclear spin environments.
    
    This class provides methods for calculating various decoherence processes
    affecting an NV center due to surrounding nuclear spins, including dephasing,
    relaxation, and spectral diffusion.
    """
    
    def __init__(self, spin_bath=None):
        """
        Initialize decoherence model.
        
        Parameters
        ----------
        spin_bath : NuclearSpinBath, optional
            Nuclear spin bath configuration
        """
        self.spin_bath = spin_bath
        
        # Default decoherence parameters
        self._default_params = {
            't1': 5e-3,         # s, intrinsic electron T1 (not from nuclear bath)
            't2_intrinsic': 1e-3,  # s, intrinsic electron T2 (not from nuclear bath)
            't2_star_intrinsic': 1e-6,  # s, intrinsic electron T2* (not from nuclear bath)
            'b_field': 0.05,    # T, magnetic field magnitude
            'b_field_dir': [0, 0, 1],  # Magnetic field direction (default along z/NV axis)
            'temperature': 300.0  # K, temperature for thermal fluctuations
        }
    
    def calculate_t2_star(self, magnetic_field: Optional[List[float]] = None):
        """
        Calculate T2* due to quasi-static nuclear fields.
        
        T2* represents the dephasing time due to inhomogeneous broadening
        from the quasi-static nuclear spin bath.
        
        Parameters
        ----------
        magnetic_field : list, optional
            [Bx, By, Bz] magnetic field in Tesla
            
        Returns
        -------
        float
            T2* time in seconds
        """
        if self.spin_bath is None:
            logger.warning("No spin bath defined, returning intrinsic T2*")
            return self._default_params['t2_star_intrinsic']
        
        # Set magnetic field
        if magnetic_field is None:
            b_mag = self._default_params['b_field']
            b_dir = np.array(self._default_params['b_field_dir'])
            b_dir = b_dir / np.linalg.norm(b_dir)
            magnetic_field = b_mag * b_dir
        
        # Get all nuclear spins
        nuclear_spins = self.spin_bath.get_spins()
        
        if not nuclear_spins:
            logger.warning("Empty spin bath, returning intrinsic T2*")
            return self._default_params['t2_star_intrinsic']
        
        # Calculate quasi-static field distribution from nuclear spins
        # This is a simplified model based on random orientations of nuclear spins
        
        # Sum squares of hyperfine couplings for variance calculation
        from .hyperfine import HyperfineCalculator
        calculator = HyperfineCalculator()
        
        hyperfine_variance = 0.0
        
        for spin in nuclear_spins:
            # Calculate full hyperfine tensor
            A_tensor = calculator.calculate_hyperfine_tensor(
                spin.position, spin.species
            )
            
            # For T2*, we care about the z-component (along NV axis)
            # assuming NV axis is along z
            A_z = A_tensor[2, 2]  # Hz
            
            # Each nuclear spin contributes A_z^2 * I(I+1)/3 to the field variance
            # where I is the nuclear spin quantum number
            I = spin.spin_quantum
            hyperfine_variance += (A_z**2) * I * (I + 1) / 3
        
        # Convert variance to T2* time
        # T2* = 1/(2π * σ), where σ is the standard deviation of the field
        field_std = np.sqrt(hyperfine_variance)  # Hz
        t2_star = 1 / (2 * np.pi * field_std)  # seconds
        
        # Combine with intrinsic T2*
        t2_star_intrinsic = self._default_params['t2_star_intrinsic']
        t2_star_combined = 1 / (1/t2_star + 1/t2_star_intrinsic)
        
        logger.info(f"Calculated T2* from nuclear bath: {t2_star:.2e} s")
        logger.info(f"Combined T2* with intrinsic T2*: {t2_star_combined:.2e} s")
        
        return t2_star_combined
    
    def calculate_t2_from_cluster_expansion(self, max_order: int = 2):
        """
        Calculate T2 using cluster correlation expansion (CCE) method.
        
        Parameters
        ----------
        max_order : int
            Maximum cluster size to consider
            
        Returns
        -------
        float
            T2 time in seconds
        """
        # This is a simplified implementation of CCE
        # A full implementation would perform proper cluster expansion
        # and quantum evolution of each cluster
        
        if self.spin_bath is None:
            logger.warning("No spin bath defined, returning intrinsic T2")
            return self._default_params['t2_intrinsic']
        
        # Get all nuclear spins
        nuclear_spins = self.spin_bath.get_spins()
        
        if not nuclear_spins:
            logger.warning("Empty spin bath, returning intrinsic T2")
            return self._default_params['t2_intrinsic']
        
        # For demonstration, we'll implement a simplified model
        # where T2 is estimated from pairwise flip-flop contributions
        
        # First, calculate dipolar coupling between all nuclear spin pairs
        from scipy.constants import mu_0, hbar
        
        # Simple function to calculate dipolar coupling strength between two spins
        def calculate_dipolar_coupling(pos1, pos2, gamma1, gamma2):
            # Vector between spins
            r_vec = np.array(pos1) - np.array(pos2)
            r = np.linalg.norm(r_vec)
            
            if r < 1e-15:  # Avoid division by zero
                return 0.0
            
            # Unit vector
            n = r_vec / r
            
            # Calculate dipolar coupling constant (in Hz)
            # D = (μ0/4π) * γ1 * γ2 * ħ / r³
            coupling = (mu_0 / (4 * np.pi)) * gamma1 * gamma2 * (hbar / (2 * np.pi)) / (r**3)
            
            return abs(coupling)  # Hz
        
        # Calculate pairwise couplings
        couplings = []
        for i, spin1 in enumerate(nuclear_spins):
            for j in range(i+1, len(nuclear_spins)):
                spin2 = nuclear_spins[j]
                
                # Skip if different species (approximate)
                if spin1.species != spin2.species:
                    continue
                
                coupling = calculate_dipolar_coupling(
                    spin1.position, spin2.position,
                    spin1.gyromagnetic_ratio, spin2.gyromagnetic_ratio
                )
                
                couplings.append(coupling)
        
        # Sort couplings from strongest to weakest
        couplings.sort(reverse=True)
        
        # Truncate to max_order strongest couplings
        if max_order < len(couplings):
            couplings = couplings[:max_order]
        
        # Estimate T2 from strongest couplings (simplified model)
        if couplings:
            # T2 ~ 1/D_rms where D_rms is root-mean-square coupling
            d_rms = np.sqrt(np.mean(np.array(couplings)**2))
            t2_bath = 1 / (2 * np.pi * d_rms)  # seconds
        else:
            t2_bath = float('inf')
        
        # Combine with intrinsic T2
        t2_intrinsic = self._default_params['t2_intrinsic']
        t2_combined = 1 / (1/t2_bath + 1/t2_intrinsic)
        
        logger.info(f"Calculated T2 from nuclear bath (CCE order {max_order}): {t2_bath:.2e} s")
        logger.info(f"Combined T2 with intrinsic T2: {t2_combined:.2e} s")
        
        return t2_combined
    
    def calculate_spectral_diffusion(self, tau_values: List[float]):
        """
        Calculate spectral diffusion contribution to decoherence.
        
        Parameters
        ----------
        tau_values : list
            List of time points (in seconds) at which to calculate coherence
            
        Returns
        -------
        numpy.ndarray
            Coherence values (0-1) for each tau value
        """
        if self.spin_bath is None:
            logger.warning("No spin bath defined, returning exponential decay with intrinsic T2")
            return np.exp(-(np.array(tau_values) / self._default_params['t2_intrinsic'])**2)
        
        # Get all nuclear spins
        nuclear_spins = self.spin_bath.get_spins()
        
        if not nuclear_spins:
            logger.warning("Empty spin bath, returning exponential decay with intrinsic T2")
            return np.exp(-(np.array(tau_values) / self._default_params['t2_intrinsic'])**2)
        
        # Calculate spectral diffusion decay
        # Simplified model: coherence ~ exp(-(t/T_SD)^n)
        # where n depends on the diffusion mechanism
        
        # Estimate spectral diffusion time constant from bath properties
        # For demonstration, we use a simple model
        t_sd = self.estimate_spectral_diffusion_time()
        
        # Spectral diffusion typically gives n between 2 and 3
        # n=2 for fluctuations with short correlation time
        # n>2 for fluctuations with long correlation time
        n = 2.5
        
        # Calculate coherence decay
        coherence = np.exp(-(np.array(tau_values) / t_sd)**n)
        
        # Combine with intrinsic T2 decay
        t2_intrinsic = self._default_params['t2_intrinsic']
        intrinsic_decay = np.exp(-(np.array(tau_values) / t2_intrinsic)**2)
        
        combined_coherence = coherence * intrinsic_decay
        
        return combined_coherence
    
    def estimate_spectral_diffusion_time(self):
        """
        Estimate spectral diffusion time constant.
        
        Returns
        -------
        float
            Spectral diffusion time constant in seconds
        """
        # Simplified model for spectral diffusion time constant
        # In reality, this depends on many factors including
        # bath concentration, temperature, and magnetic field
        
        # Get all nuclear spins
        nuclear_spins = self.spin_bath.get_spins()
        
        # Count 13C spins (main source of spectral diffusion)
        c13_count = len([s for s in nuclear_spins if s.species == '13C'])
        
        # No spins means no spectral diffusion
        if c13_count == 0:
            return float('inf')
        
        # Simple scaling with concentration
        # T_SD ~ 1/sqrt(concentration)
        base_time = 100e-6  # Base time at natural abundance (1.1%)
        concentration = c13_count / len(nuclear_spins) if nuclear_spins else 0.011
        
        # Scale with concentration relative to natural abundance
        t_sd = base_time * np.sqrt(0.011 / max(concentration, 1e-6))
        
        # Temperature dependence (simplified model)
        # Higher temperatures increase flip-flop rates
        temperature = self._default_params['temperature']
        temp_factor = np.sqrt(300.0 / temperature)  # Scale relative to room temp
        t_sd *= temp_factor
        
        # Magnetic field dependence (simplified model)
        # Higher fields reduce flip-flops due to energy mismatch
        b_field = self._default_params['b_field']
        field_factor = np.sqrt(b_field / 0.05)  # Scale relative to 500 gauss
        t_sd *= field_factor
        
        logger.info(f"Estimated spectral diffusion time: {t_sd:.2e} s")
        
        return t_sd
    
    def calculate_noise_spectrum(self, frequencies: List[float]):
        """
        Calculate noise spectrum from nuclear spin bath.
        
        Parameters
        ----------
        frequencies : list
            List of frequencies (Hz) at which to calculate noise
            
        Returns
        -------
        numpy.ndarray
            Noise power spectral density at each frequency (T²/Hz)
        """
        if self.spin_bath is None:
            logger.warning("No spin bath defined, returning flat noise spectrum")
            return np.ones_like(frequencies) * 1e-12
        
        # Get all nuclear spins
        nuclear_spins = self.spin_bath.get_spins()
        
        if not nuclear_spins:
            logger.warning("Empty spin bath, returning flat noise spectrum")
            return np.ones_like(frequencies) * 1e-12
        
        # Calculate noise spectrum
        # Simplified model based on assumed forms for different mechanisms
        
        # Convert frequencies to numpy array
        freqs = np.array(frequencies)
        
        # 1. Low-frequency noise (f^-α, with α typically 0.5-1.5)
        # This represents slow fluctuations like spectral diffusion
        alpha = 1.0
        low_freq_amplitude = 1e-10  # T²/Hz at 1 Hz
        low_freq_noise = low_freq_amplitude * (freqs / 1.0)**(-alpha)
        
        # 2. Lorentzian peaks from nuclear Larmor precession
        # Each nuclear species contributes a peak at its Larmor frequency
        b_field = self._default_params['b_field']
        lorentzian_noise = np.zeros_like(freqs)
        
        # Count spins by species
        species_count = {}
        for spin in nuclear_spins:
            species_count[spin.species] = species_count.get(spin.species, 0) + 1
        
        # Add Lorentzian peak for each species
        for species, count in species_count.items():
            gamma = GYROMAGNETIC_RATIOS.get(species, 0.0)
            if gamma == 0:
                continue
                
            # Larmor frequency
            larmor_freq = abs(gamma * b_field)
            
            # Peak width (depends on spin-spin interactions)
            # Typically kHz for 13C in diamond
            peak_width = 1e3  # Hz
            
            # Peak amplitude scales with number of spins
            peak_amplitude = 1e-12 * count  # T²/Hz
            
            # Lorentzian function
            lorentzian = peak_amplitude * (peak_width**2) / ((freqs - larmor_freq)**2 + peak_width**2)
            
            # Add peak to total noise
            lorentzian_noise += lorentzian
        
        # 3. Flat background (white noise from other sources)
        background = 1e-14  # T²/Hz
        
        # Combine all noise sources
        total_noise = low_freq_noise + lorentzian_noise + background
        
        return total_noise
    
    def apply_decoherence_to_sequence(self, sequence_times: List[float], 
                                    sequence_type: str = 'hahn'):
        """
        Apply decoherence to a pulse sequence.
        
        Parameters
        ----------
        sequence_times : list
            List of time points (in seconds) for the sequence
        sequence_type : str
            Type of sequence ('hahn', 'cpmg', 'xy4', etc.)
            
        Returns
        -------
        numpy.ndarray
            Coherence values (0-1) for each time point
        """
        # Different sequences have different filtering properties
        # For a more accurate model, we would calculate filter functions
        # and convolve with the noise spectrum
        
        # Simplified model: use different decay exponents for different sequences
        # based on empirical observations
        
        # Convert times to numpy array
        times = np.array(sequence_times)
        
        # Calculate T2 for reference
        t2 = self.calculate_t2_from_cluster_expansion()
        
        # Sequence-specific exponents and scaling factors
        sequence_params = {
            'fid': {'n': 2, 'scale': 1/self.calculate_t2_star()},
            'hahn': {'n': 3, 'scale': 1/t2},
            'cpmg': {'n': 4, 'scale': 0.7/t2},
            'xy4': {'n': 4, 'scale': 0.5/t2},
            'xy8': {'n': 4, 'scale': 0.4/t2},
            'xy16': {'n': 4, 'scale': 0.3/t2},
            'kdd': {'n': 4, 'scale': 0.25/t2}
        }
        
        # Get parameters for the requested sequence type
        if sequence_type not in sequence_params:
            logger.warning(f"Unknown sequence type: {sequence_type}, using Hahn echo model")
            sequence_type = 'hahn'
            
        params = sequence_params[sequence_type]
        
        # Calculate coherence decay
        coherence = np.exp(-(times * params['scale'])**params['n'])
        
        logger.info(f"Applied {sequence_type} decoherence model with T2 = {t2:.2e} s")
        
        return coherence