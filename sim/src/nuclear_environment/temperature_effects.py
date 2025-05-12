"""Temperature-dependent effects for NV center simulation.

This module implements physically accurate temperature dependence for
relaxation rates, decoherence processes, and other temperature-sensitive
parameters in NV center simulations.
"""

import numpy as np
import logging
from .thread_safety import thread_safe

# Configure module logger
logger = logging.getLogger(__name__)

class TemperatureEffects:
    """
    Manager for temperature-dependent effects in NV center simulations.
    
    This class calculates temperature-dependent parameters based on physical
    models, including T1 and T2 relaxation times, phonon-assisted processes,
    and ZPL/PSB ratios.
    """
    
    def __init__(self, physical_params=None):
        """
        Initialize the temperature effects manager.
        
        Parameters
        ----------
        physical_params : PhysicalParameters, optional
            Physical parameters manager instance
        """
        from .physical_parameters import PhysicalParameters
        
        self.params = physical_params if physical_params is not None else PhysicalParameters()
        
        # Physical constants
        self.k_B = 1.380649e-23  # Boltzmann constant (J/K)
        self.h = 6.62607015e-34  # Planck constant (J*s)
        
        # Diamond-specific parameters
        self.debye_temperature = 2230  # Debye temperature for diamond (K)
        self.zpl_energy = 1.945  # Zero-phonon line energy (eV)
        self.zpl_energy_J = self.zpl_energy * 1.602176634e-19  # Convert to J
        
        # Initialize calculated values
        self._update_temperature_dependent_params()
    
    @thread_safe
    def set_temperature(self, temperature_K):
        """
        Set temperature and update all temperature-dependent parameters.
        
        Parameters
        ----------
        temperature_K : float
            Temperature in Kelvin
        """
        # Update temperature in parameters
        self.params.set('temperature', temperature_K, 'K', 'Operating temperature')
        
        # Update all calculated parameters
        self._update_temperature_dependent_params()
        
        logger.info(f"Temperature set to {temperature_K} K, parameters updated")
    
    def _update_temperature_dependent_params(self):
        """Update all temperature-dependent parameters."""
        # Get current temperature
        temperature = self.params.get('temperature', 300.0)
        
        # Update various parameters
        self._update_relaxation_rates(temperature)
        self._update_optical_parameters(temperature)
        self._update_phonon_parameters(temperature)
    
    def _update_relaxation_rates(self, temperature):
        """
        Update T1 and T2 relaxation times based on temperature.
        
        Parameters
        ----------
        temperature : float
            Temperature in Kelvin
        """
        # Get base values at reference temperature (300K)
        base_t1 = self.params.get('t1_electron_300K', 5.0e-3)  # seconds
        base_t2 = self.params.get('t2_electron_300K', 500e-6)  # seconds
        
        # T1 temperature dependence:
        # Direct process dominates at low T: T1 ∝ coth(ħω/2kT)
        # Raman process dominates at higher T: T1 ∝ T^-5 for T > ΘD/10
        # Orbach process: T1 ∝ exp(Δ/kT) where Δ is activation energy
        
        # Simplified model based on experimental results
        # From: de Lange et al., Science 330, 60 (2010)
        if temperature < 100:
            # Low temperature regime: T1 nearly constant
            t1 = base_t1
        elif temperature < 200:
            # Transition regime (simple interpolation)
            t1 = base_t1 * (1 - (temperature - 100) / 100 * 0.5)
        else:
            # High temperature: T^-5 scaling from Raman process
            # Normalize to base_t1 at 300K
            t1 = base_t1 * (300 / temperature)**5
        
        # T2 temperature dependence:
        # T2 is limited by T1 at very low T: T2 <= 2*T1
        # At intermediate temperatures, T2 has weaker temperature dependence
        # In NV centers in high purity diamond, phonon-induced dephasing
        # typically leads to T2 ∝ T^-1 to T^-2 for T > 100K
        
        if temperature < 100:
            # Low temperature: T2 limited by T1
            t2 = min(base_t2, 2 * t1)
        else:
            # Moderate temperature: T^-1 scaling
            t2 = base_t2 * (300 / temperature)
        
        # T2* typically has stronger temperature dependence due to 
        # inhomogeneous broadening effects
        t2_star = self.params.get('t2_star_electron_300K', 1.0e-6)  # seconds
        t2_star_temp = t2_star * (300 / max(50, temperature))**2
        
        # Update parameters
        self.params.set('t1_electron', t1, 's', 'Temperature-adjusted T1')
        self.params.set('t2_electron', t2, 's', 'Temperature-adjusted T2')
        self.params.set('t2_star_electron', t2_star_temp, 's', 'Temperature-adjusted T2*')
        
        # Also store the rates (1/time)
        self.params.set('t1_rate', 1/t1 if t1 > 0 else 0, 'Hz', 'T1 relaxation rate')
        self.params.set('t2_rate', 1/t2 if t2 > 0 else 0, 'Hz', 'T2 dephasing rate')
        self.params.set('t2_star_rate', 1/t2_star_temp if t2_star_temp > 0 else 0, 'Hz', 'T2* dephasing rate')
        
        logger.debug(f"Updated relaxation times: T1={t1:.2e}s, T2={t2:.2e}s, T2*={t2_star_temp:.2e}s")
    
    def _update_optical_parameters(self, temperature):
        """
        Update optical parameters based on temperature.
        
        Parameters
        ----------
        temperature : float
            Temperature in Kelvin
        """
        # Base fluorescence parameters at 300K
        base_lifetime = self.params.get('fluorescence_lifetime_300K', 12e-9)  # seconds
        base_quantum_efficiency = self.params.get('quantum_efficiency_300K', 0.7)  # dimensionless
        
        # Temperature effects on excited state lifetime
        # Lifetime typically decreases with temperature due to increased 
        # non-radiative processes, but the effect is small for NV centers
        # We use a simple phenomenological model with weak temperature dependence
        lifetime_factor = 1.0 + 0.05 * np.tanh((temperature - 300) / 100)
        lifetime = base_lifetime / lifetime_factor
        
        # Quantum efficiency also decreases with temperature
        # This affects the overall fluorescence intensity
        qe_factor = 1.0 + 0.1 * np.tanh((temperature - 300) / 100)
        quantum_efficiency = base_quantum_efficiency / qe_factor
        
        # Zero-phonon line (ZPL) fraction decreases with temperature
        # as phonon sidebands (PSB) increase
        # Based on Debye-Waller factor: exp(-2W) where W is the Huang-Rhys factor
        # For NV centers, Huang-Rhys factor S ≈ 3.5 at low T
        base_zpl_fraction = self.params.get('zpl_fraction_0K', np.exp(-3.5))
        
        # Temperature dependent Huang-Rhys factor (increases with T)
        S_T = 3.5 * (1.0 + 0.0008 * temperature)
        zpl_fraction = np.exp(-S_T)
        
        # Update parameters
        self.params.set('fluorescence_lifetime', lifetime, 's', 'Temperature-adjusted fluorescence lifetime')
        self.params.set('quantum_efficiency', quantum_efficiency, 'dimensionless', 
                        'Temperature-adjusted quantum efficiency')
        self.params.set('zpl_fraction', zpl_fraction, 'dimensionless', 
                        'Zero-phonon line fraction at current temperature')
        
        logger.debug(f"Updated optical parameters: lifetime={lifetime:.2e}s, QE={quantum_efficiency:.2f}, ZPL={zpl_fraction:.3f}")
    
    def _update_phonon_parameters(self, temperature):
        """
        Update phonon-related parameters based on temperature.
        
        Parameters
        ----------
        temperature : float
            Temperature in Kelvin
        """
        # Calculate average phonon number for ZPL transition (Bose-Einstein distribution)
        zpl_freq = self.zpl_energy_J / self.h  # Frequency of ZPL transition
        n_phonon_zpl = 1.0 / (np.exp(self.zpl_energy_J / (self.k_B * temperature)) - 1.0)
        
        # Phonon-assisted transition rates scale with temperature
        # Upward transitions scale with n+1, downward with n
        phonon_factor_up = max(0.01, n_phonon_zpl + 1)
        phonon_factor_down = max(0.01, n_phonon_zpl)
        
        # Average phonon occupation for acoustic modes at 2.87 GHz (ZFS)
        zfs_energy_J = 2.87e9 * self.h
        n_phonon_zfs = 1.0 / (np.exp(zfs_energy_J / (self.k_B * temperature)) - 1.0)
        
        # Update parameters
        self.params.set('phonon_occupation_zpl', n_phonon_zpl, 'dimensionless', 
                        'Average phonon occupation for ZPL transition')
        self.params.set('phonon_factor_up', phonon_factor_up, 'dimensionless', 
                        'Phonon factor for upward transitions')
        self.params.set('phonon_factor_down', phonon_factor_down, 'dimensionless', 
                        'Phonon factor for downward transitions')
        self.params.set('phonon_occupation_zfs', n_phonon_zfs, 'dimensionless',
                        'Average phonon occupation for ZFS frequency')
        
        logger.debug(f"Updated phonon parameters: n_zpl={n_phonon_zpl:.2e}, n_zfs={n_phonon_zfs:.2f}")
    
    def get_collapse_operators(self, quantum_system=None):
        """
        Get temperature-dependent collapse operators for quantum evolution.
        
        Parameters
        ----------
        quantum_system : object, optional
            Quantum system object (for SimOS integration)
            
        Returns
        -------
        list
            List of collapse operators
        """
        # This implementation will depend on the specific quantum framework used
        # For now, we return a summary dictionary of rates
        
        rates = {
            't1_rate': self.params.get('t1_rate'),
            't2_rate': self.params.get('t2_rate'),
            'phonon_factor_up': self.params.get('phonon_factor_up'),
            'phonon_factor_down': self.params.get('phonon_factor_down')
        }
        
        # If a quantum system is provided, create proper operators
        if quantum_system is not None:
            try:
                # This is a placeholder for SimOS integration
                # A complete implementation would construct the proper operators
                # based on the quantum system's operators and the calculated rates
                logger.debug("Quantum system provided but operator construction not implemented")
                return rates
            except Exception as e:
                logger.warning(f"Failed to create operators: {e}")
        
        return rates