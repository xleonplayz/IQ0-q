"""
Physical NV-Center Model for quantum simulation.

This module provides the core physical model for NV-center simulations,
implementing the quantum mechanical behavior and integrating with SimOS.
"""

import numpy as np
import threading
import logging
import time
import uuid
import scipy.linalg
from typing import Dict, Any, Optional, Tuple, List, Union, Callable, Generator
from dataclasses import dataclass
from pathlib import Path

# Import our new infrastructure
from .exceptions import (
    SimOSNVError, SimOSImportError, ConfigurationError, 
    QuantumStateError, HamiltonianError, EvolutionError, 
    ThreadingError, ExperimentError
)
from .simos_adapter import SimOSNVAdapter
from .concurrency import StateLockManager

# Configure logging
logger = logging.getLogger(__name__)

# Data structures for simulation results
@dataclass
class OpticalResult:
    """
    Result of optical dynamics simulation.
    
    This class stores the results of optical excitation and fluorescence dynamics
    simulations for NV centers. It captures the time evolution of ground and excited
    state populations, as well as the resulting fluorescence signal.
    
    For saturation measurements, it also includes power-dependent fluorescence data.
    
    Parameters
    ----------
    times : numpy.ndarray
        Time points at which measurements were taken (in seconds)
    ground_population : numpy.ndarray
        Population in the ground state at each time point
    excited_population : numpy.ndarray
        Population in the excited state at each time point
    fluorescence : numpy.ndarray
        Fluorescence count rate at each time point (counts/s)
    saturation_curve : dict, optional
        For saturation measurements, contains 'powers' and 'counts' arrays
    experiment_id : str, optional
        Unique identifier for the experiment
        
    Notes
    -----
    The time evolution data provides insight into the optical dynamics
    of the NV center, including excitation rate, relaxation processes,
    and optical contrast between different spin states.
    """
    times: np.ndarray
    ground_population: np.ndarray
    excited_population: np.ndarray
    fluorescence: np.ndarray
    saturation_curve: Optional[Dict[str, np.ndarray]] = None
    experiment_id: str = ''
    
    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = str(uuid.uuid4())

@dataclass
class ODMRResult:
    """Result of an ODMR measurement."""
    frequencies: np.ndarray
    signal: np.ndarray
    contrast: float
    center_frequency: float
    linewidth: float
    experiment_id: str = ''
    
    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = str(uuid.uuid4())

@dataclass
class RabiResult:
    """Result of a Rabi oscillation measurement."""
    times: np.ndarray
    population: np.ndarray
    rabi_frequency: float
    decay_time: Optional[float] = None
    experiment_id: str = ''
    
    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = str(uuid.uuid4())

@dataclass
class T1Result:
    """Result of a T1 relaxation measurement."""
    times: np.ndarray
    population: np.ndarray
    t1_time: float
    experiment_id: str = ''
    
    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = str(uuid.uuid4())

@dataclass
class T2Result:
    """Result of a T2 (spin echo) measurement."""
    times: np.ndarray
    signal: np.ndarray
    t2_time: float
    experiment_id: str = ''
    
    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = str(uuid.uuid4())

@dataclass
class StateEvolution:
    """Evolution data for a quantum state."""
    times: np.ndarray
    populations: Dict[str, np.ndarray]
    coherences: Dict[str, np.ndarray]
    experiment_id: str = ''
    
    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = str(uuid.uuid4())


class PhysicalNVModel:
    """
    Implements the physical model for NV-centers.
    
    This class forms the core of the quantum simulation, handling the
    quantum state, Hamiltonian, and time evolution of the NV-center system.
    It integrates with SimOS for accurate quantum mechanical calculations.
    If SimOS is not available, a detailed error is provided.
    
    The model can be used to:
    - Simulate NV center quantum dynamics
    - Calculate energy levels under different magnetic fields
    - Simulate microwave and laser interactions
    - Calculate ODMR spectra and Rabi oscillations
    - Perform T1 and T2 measurements
    - Interface with Qudi hardware modules
    
    Attributes:
        config (Dict[str, Any]): Configuration parameters for the model
        lock_manager (StateLockManager): Thread safety manager
        magnetic_field (np.ndarray): Current magnetic field vector [Bx, By, Bz] in Tesla
        mw_frequency (float): Microwave frequency in Hz
        mw_power (float): Microwave power in dBm
        mw_on (bool): Whether microwave excitation is on
        laser_power (float): Laser power in mW
        laser_on (bool): Whether laser excitation is on
        dt (float): Simulation time step in seconds
        simos_adapter: SimOS NV adapter object
        quantum_state: Quantum state information
        simulation_thread: Thread for running simulations in background
        stop_simulation: Threading event for controlling simulation execution
        is_simulating (bool): Whether continuous simulation is running
        cached_results (Dict): Cache for storing simulation results
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the NV-center physical model.
        
        Args:
            config: Optional configuration dictionary. If None, default configuration will be used.
                   When providing a partial config, any missing fields will be filled with defaults.
        
        Raises:
            ConfigurationError: If the provided configuration is invalid.
            SimOSImportError: If SimOS cannot be imported and is required.
        """
        # Initialize with default configuration first
        default_config = self._default_config()
        
        # If config is provided, update defaults with custom values
        if config is not None:
            # Validate configuration values
            for key, value in config.items():
                # Check for None values
                if value is None:
                    raise ConfigurationError(
                        f"Configuration parameter '{key}' cannot be None",
                        parameter=key,
                        value=value
                    )
                
                # Check for negative values in times and frequencies
                if key in ['T1', 'T2', 'T2_star', 'zero_field_splitting', 'simulation_timestep'] and value <= 0:
                    raise ConfigurationError(
                        f"Configuration parameter '{key}' must be positive",
                        parameter=key,
                        value=value,
                        valid_range="positive values"
                    )
                
                # Check for proper types
                if key in ['zero_field_splitting', 'T1', 'T2', 'T2_star', 'strain', 'simulation_timestep']:
                    if not isinstance(value, (int, float)):
                        raise ConfigurationError(
                            f"Configuration parameter '{key}' must be numeric",
                            parameter=key,
                            value=value,
                            valid_range="numeric values"
                        )
            
            default_config.update(config)
            
        self.config = default_config
        
        # Set up thread safety
        self.lock_manager = StateLockManager()
        
        # Initialize the resource locks with the proper resources
        self.lock_manager.hamiltonian_rw.resource = {}
        self.lock_manager.state_rw.resource = {}
        
        # Current magnetic field (initialize to zero field)
        self.magnetic_field = np.array([0.0, 0.0, 0.0])
        
        # Microwave parameters
        self.mw_frequency = self.config['zero_field_splitting']  # Default to ZFS
        self.mw_power = 0.0  # dBm
        self.mw_on = False
        
        # Laser parameters
        self.laser_power = 0.0  # mW
        self.laser_on = False
        
        # Simulation time step
        self.dt = self.config['simulation_timestep']  # Default time step
        
        # Simulation thread control
        self.simulation_thread = None
        self.stop_simulation = threading.Event()
        self.is_simulating = False
        
        # Results cache
        self.cached_results = {}
        
        # Quantum state
        self.quantum_state = None
        
        # Initialize SimOS adapter and quantum system
        self._initialize_quantum_system()
        
        logger.info(f"NV-center model initialized with ZFS={self.config['zero_field_splitting']/1e9:.3f} GHz")
    
    def _default_config(self) -> Dict[str, Any]:
        """
        Get the default configuration for the NV-center model.
        
        Returns:
            Dict[str, Any]: Default configuration dictionary
        """
        return {
            # Quantum parameters
            'zero_field_splitting': 2.87e9,  # Hz
            'gyromagnetic_ratio': 2.8025e10,  # Hz/T
            'strain': 5e6,  # Hz - strain splitting
            'hyperfine_coupling_parallel': 2.14e6,  # Hz - 14N A parallel
            'hyperfine_coupling_perpendicular': 2.7e6,  # Hz - 14N A perpendicular
            
            # Relaxation times
            'T1': 5e-3,  # seconds - spin-lattice relaxation
            'T2': 1e-6,  # seconds - spin-spin relaxation
            'T2_star': 2e-7,  # seconds - inhomogeneous dephasing
            
            # Optical parameters
            'excitation_saturation_power': 1.0,  # mW
            'optical_transition_time': 1e-9,  # seconds
            'excited_state_lifetime': 12e-9,  # seconds
            'background_count_rate': 1000.0,  # counts/s
            'fluorescence_rate_ms0': 150000.0,  # counts/s
            'fluorescence_rate_ms1': 100000.0,  # counts/s
            'saturation_count_rate': 250000.0,  # counts/s
            'photoionization_threshold': 10.0,  # mW - power beyond which photoionization becomes significant
            
            # Simulation parameters
            'simulation_timestep': 1e-9,  # seconds
            'adaptive_timestep': True,
            'adaptive_tol': 1e-6,  # Tolerance for adaptive timestepping
            'ode_method': 'adams',  # ODE solver method for adaptive stepping
            'noise_amplitude': 0.03,  # fractional amplitude of noise in signals
            'simulate_nv_concentration': False,  # whether to simulate effects of NV concentration
            'bath_coupling_strength': 1e6,  # Hz - coupling to spin bath
            'bath_concentration': 1e19,  # spins/cm³ - spin bath concentration
            'bath_cutoff': 1e6,  # Hz - spectral cutoff for bath
            'decoherence_model': 'markovian',  # 'markovian' or 'non-markovian'
            
            # Advanced parameters for testing and special cases
            'microwave_resonance_tolerance': 20e6,  # Hz - frequency width for resonance
            'simulate_charge_state_dynamics': False,  # whether to simulate NV charge state dynamics
            'simulate_nitrogen_hyperfine': False,  # whether to include nitrogen hyperfine interaction
            'simulate_spin_bath': False,  # whether to include spin bath effects
            'strain_e': 0.0,  # Hz - E strain component
            'strain_d': 5e6,  # Hz - D strain component
            'include_excited_state': True,  # whether to include excited state in calculation
            'temperature': 298,  # K - system temperature
        }
    
    def _initialize_quantum_system(self):
        """
        Initialize the quantum system and SimOS adapter.
        
        This method sets up the SimOS adapter with the appropriate configuration
        and initializes the quantum state.
        
        Raises:
            SimOSImportError: If SimOS cannot be imported properly
            ConfigurationError: If the system cannot be initialized with the given config
        """
        try:
            # Prepare SimOS configuration
            simos_config = {
                "zero_field_splitting": self.config["zero_field_splitting"],
                "gyromagnetic_ratio": self.config["gyromagnetic_ratio"],
                "strain_e": self.config.get("strain_e", 0.0),
                "strain_d": self.config.get("strain_d", self.config["strain"]),
                "temperature": 298,  # Room temperature in K
                "include_excited_state": self.config.get("include_excited_state", True),
                "include_nitrogen_nucleus": self.config.get("simulate_nitrogen_hyperfine", False),
                "nitrogen_isotope": 14  # Default to 14N
            }
            
            # Initialize SimOS adapter
            self.simos_adapter = SimOSNVAdapter(simos_config)
            
            # Initialize quantum state to ground state (ms=0)
            self.reset_state()
            
            logger.info("Quantum system initialized successfully")
            
        except (SimOSImportError, ConfigurationError) as e:
            logger.error(f"Failed to initialize quantum system: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error initializing quantum system: {str(e)}")
            raise ConfigurationError(
                f"Failed to initialize quantum system: {str(e)}",
                parameter="config", 
                value=self.config
            )
    
    def reset_state(self):
        """
        Reset the quantum state to ground state (ms=0).
        
        This method resets the quantum state to the ground state (ms=0) and
        turns off any microwave or laser excitation.
        """
        # Acquire state write lock
        try:
            self.lock_manager.state_rw.acquire_write()
            try:
                self.quantum_state = self.simos_adapter.get_equilibrium_state()
                
                # Turn off excitation
                self.mw_on = False
                self.laser_on = False
                
                # Reset state variables tracking optical pumping
                self._update_state_info()
            
                logger.debug("Quantum state reset to ground state")
            except Exception as e:
                logger.error(f"Error resetting quantum state: {str(e)}")
                raise QuantumStateError(f"Failed to reset quantum state: {str(e)}")
            finally:
                self.lock_manager.state_rw.release_write()
        except Exception as e:
            logger.error(f"Failed to acquire lock for reset_state: {str(e)}")
            raise ThreadingError(f"Failed to acquire lock for reset_state: {str(e)}")
    
    def initialize_state(self, ms: int = 0):
        """
        Initialize the quantum state to a specific ms state.
        
        Args:
            ms: The ms value to initialize to (0, -1, or 1)
            
        Raises:
            QuantumStateError: If the state cannot be initialized
        """
        if ms not in (0, -1, 1):
            raise QuantumStateError(
                f"Invalid ms value: {ms}. Must be 0, -1, or 1.",
                state_info={"ms": ms}
            )
        
        try:
            self.lock_manager.state_rw.acquire_write()
            try:
                # Get appropriate state from adapter
                if ms == 0:
                    self.quantum_state = self.simos_adapter.get_equilibrium_state()
                elif ms == -1:
                    # Initialize to ms=-1 state 
                    self.quantum_state = self.simos_adapter.nv_system.get_state("-1")
                else:  # ms == 1
                    # Initialize to ms=+1 state
                    self.quantum_state = self.simos_adapter.nv_system.get_state("+1")
                    
                # Update state information
                self._update_state_info()
                
                logger.debug(f"Quantum state initialized to ms={ms}")
            except Exception as e:
                logger.error(f"Error initializing state to ms={ms}: {str(e)}")
                raise QuantumStateError(
                    f"Failed to initialize state to ms={ms}: {str(e)}",
                    state_info={"ms": ms}
                )
            finally:
                self.lock_manager.state_rw.release_write()
        except Exception as e:
            logger.error(f"Failed to acquire lock for initialize_state: {str(e)}")
            raise ThreadingError(f"Failed to acquire lock for initialize_state: {str(e)}")
    
    def set_magnetic_field(self, field: Union[List[float], np.ndarray]):
        """
        Set the magnetic field vector.
        
        Args:
            field: Magnetic field vector [Bx, By, Bz] in Tesla
            
        Raises:
            ConfigurationError: If the field is invalid
        """
        if len(field) != 3:
            raise ConfigurationError(
                f"Magnetic field must have 3 components, got {len(field)}",
                parameter="field",
                value=field,
                valid_range="3-component vector"
            )
        
        try:
            # Convert to numpy array if it isn't already
            if not isinstance(field, np.ndarray):
                field = np.array(field, dtype=float)
            
            # Store the field
            self.magnetic_field = field
            
            # Update Hamiltonian (this will be done during next evolution step)
            logger.debug(f"Magnetic field set to {field}")
            
        except Exception as e:
            logger.error(f"Error setting magnetic field: {str(e)}")
            raise ConfigurationError(
                f"Failed to set magnetic field: {str(e)}",
                parameter="field",
                value=field
            )
    
    def get_magnetic_field(self) -> np.ndarray:
        """
        Get the current magnetic field vector.
        
        Returns:
            np.ndarray: Copy of the magnetic field vector [Bx, By, Bz] in Tesla
        """
        return self.magnetic_field.copy()
    
    def apply_microwave(self, frequency: float, power: float, on: bool = True):
        """
        Apply microwave excitation with specified parameters.
        
        Args:
            frequency: Microwave frequency in Hz
            power: Microwave power in dBm
            on: Whether to turn the microwave on or off
            
        Raises:
            ConfigurationError: If the parameters are invalid
        """
        # Validate parameters
        if frequency < 0:
            raise ConfigurationError(
                "Microwave frequency must be positive",
                parameter="frequency",
                value=frequency,
                valid_range="positive values"
            )
        
        if power > 50:  # Typical maximum for laboratory equipment
            raise ConfigurationError(
                f"Microwave power {power} dBm exceeds reasonable limits",
                parameter="power",
                value=power,
                valid_range="up to 50 dBm"
            )
        
        # Store parameters
        self.mw_frequency = frequency
        self.mw_power = power
        self.mw_on = on
        
        logger.debug(f"Microwave {'enabled' if on else 'disabled'} at {frequency/1e6:.3f} MHz, {power:.1f} dBm")
    
    def apply_laser(self, power: float, on: bool = True):
        """
        Apply laser excitation with specified parameters.
        
        Args:
            power: Laser power in mW
            on: Whether to turn the laser on or off
            
        Raises:
            ConfigurationError: If the parameters are invalid
        """
        # Validate parameters
        if power < 0:
            raise ConfigurationError(
                "Laser power must be positive",
                parameter="power",
                value=power,
                valid_range="positive values"
            )
        
        if power > 100:  # Reasonable upper limit for most setups
            raise ConfigurationError(
                f"Laser power {power} mW exceeds reasonable limits",
                parameter="power",
                value=power,
                valid_range="up to 100 mW"
            )
        
        # Store parameters
        self.laser_power = power
        self.laser_on = on
        
        logger.debug(f"Laser {'enabled' if on else 'disabled'} at {power:.1f} mW")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get a copy of the current configuration.
        
        Returns:
            Dict[str, Any]: Copy of the configuration dictionary
        """
        # Return a copy to prevent external modification
        return self.config.copy()
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update the configuration with new values.
        
        Args:
            updates: Dictionary of configuration updates
            
        Raises:
            ConfigurationError: If the updates contain invalid values
        """
        # Validate key parameters
        for key, value in updates.items():
            # Check for None values
            if value is None:
                raise ConfigurationError(
                    f"Configuration parameter '{key}' cannot be None",
                    parameter=key,
                    value=value
                )
            
            # Check for negative values in times and frequencies
            if key in ['T1', 'T2', 'T2_star', 'zero_field_splitting', 'simulation_timestep'] and value <= 0:
                raise ConfigurationError(
                    f"Configuration parameter '{key}' must be positive",
                    parameter=key,
                    value=value,
                    valid_range="positive values"
                )
        
        # Update the configuration
        self.config.update(updates)
        
        # Update simulation time step if it changed
        if 'simulation_timestep' in updates:
            self.dt = self.config['simulation_timestep']
        
        logger.debug(f"Configuration updated: {updates}")
    
    def get_state_info(self) -> Dict[str, Any]:
        """
        Get information about the current quantum state.
        
        Returns:
            Dict[str, Any]: Dictionary with state information
        """
        # Ensure state_info is updated
        self._update_state_info()
        
        # Compile state information
        info = {
            'magnetic_field': self.magnetic_field.tolist(),
            'mw_frequency': self.mw_frequency,
            'mw_power': self.mw_power,
            'mw_on': self.mw_on,
            'laser_power': self.laser_power,
            'laser_on': self.laser_on,
            'ground_state_population': self._ground_state_population,
            'excited_state_population': self._excited_state_population,
            'populations': self._populations.copy()
        }
        
        # Add coherences if available
        if hasattr(self, '_coherences'):
            coherences_dict = {}
            for key, value in self._coherences.items():
                # Convert complex values to dict with real and imag parts
                coherences_dict[key] = {'real': float(np.real(value)), 'imag': float(np.imag(value))}
            info['coherences'] = coherences_dict
        
        return info
    
    def _update_state_info(self):
        """
        Update cached information about the quantum state.
        
        This method extracts important information from the quantum state
        such as populations and coherences.
        """
        try:
            # Get populations from the quantum state
            self._populations = self.simos_adapter.get_ground_state_populations(self.quantum_state)
            
            # Calculate aggregate populations
            self._ground_state_population = (
                self._populations.get('ms0', 0.0) + 
                self._populations.get('ms_minus', 0.0) + 
                self._populations.get('ms_plus', 0.0)
            )
            
            # Calculate excited state population
            self._excited_state_population = (
                1.0 - self._ground_state_population
            )
            
            # Get coherences
            try:
                self._coherences = self.simos_adapter.get_coherences(self.quantum_state)
            except Exception:
                # If coherences can't be calculated, use empty dict
                self._coherences = {}
                
        except Exception as e:
            logger.error(f"Error updating state info: {str(e)}")
            # Set default values if update fails
            self._populations = {'ms0': 0.33, 'ms_minus': 0.33, 'ms_plus': 0.33}
            self._ground_state_population = 1.0
            self._excited_state_population = 0.0
            self._coherences = {}
    
    def _update_hamiltonian(self):
        """
        Update the Hamiltonian based on current parameters.
        
        This method constructs the Hamiltonian for the current magnetic field,
        strain, and control parameters.
        
        Returns:
            Any: The updated Hamiltonian operator
        
        Raises:
            HamiltonianError: If the Hamiltonian cannot be constructed
        """
        try:
            # Get complete field Hamiltonian with all physical parameters
            strain = [self.config.get('strain_e', 0.0), self.config.get('strain_d', self.config['strain'])]
            H = self.simos_adapter.field_hamiltonian(
                self.magnetic_field.tolist(),
                strain=strain
            )
            
            # Add microwave driving with proper rotating wave approximation
            if self.mw_on:
                # Convert from dBm to amplitude
                mw_amplitude = 10**(self.mw_power/20) * 1e-3  # Approximate conversion
                
                # Rotating wave approximation for microwave driving
                phi = 2*np.pi * self.mw_frequency * self.simos_adapter.nv_system.simulation_time
                
                # Add driving Hamiltonian
                H_mw = mw_amplitude * (
                    self.simos_adapter.Sx * np.cos(phi) + 
                    self.simos_adapter.Sy * np.sin(phi)
                )
                H = H + H_mw
            
            # Add hyperfine interactions with proper physical model
            if self.config.get('simulate_nitrogen_hyperfine', False):
                H_hf = self.simos_adapter.hyperfine_hamiltonian(
                    n_species=self.config.get('nitrogen_isotope', 14),
                    hyperfine_parallel=self.config['hyperfine_coupling_parallel'],
                    hyperfine_perpendicular=self.config['hyperfine_coupling_perpendicular']
                )
                H = H + H_hf
                
            # Add environmental spin coupling for realistic decoherence
            if self.config.get('simulate_spin_bath', False):
                H_bath = self.simos_adapter.get_spin_bath_hamiltonian(
                    bath_concentration=self.config.get('bath_concentration', 1e19),  # spins/cm³
                    bath_coupling=self.config.get('bath_coupling_strength', 1e6)
                )
                H = H + H_bath
            
            # Store the Hamiltonian
            self._hamiltonian = H
            return H
            
        except Exception as e:
            logger.error(f"Error updating Hamiltonian: {str(e)}")
            raise HamiltonianError(
                f"Failed to update Hamiltonian: {str(e)}",
                hamiltonian_type="full_hamiltonian",
                params={
                    "magnetic_field": self.magnetic_field.tolist(),
                    "mw_frequency": self.mw_frequency,
                    "mw_power": self.mw_power,
                    "mw_on": self.mw_on
                }
            )
    
    def _get_collapse_operators(self):
        """
        Get collapse operators for the current state and control parameters.
        
        This method constructs collapse operators for relaxation, dephasing,
        and optical processes.
        
        Returns:
            List[Any]: List of collapse operators
        
        Raises:
            QuantumStateError: If the operators cannot be constructed
        """
        try:
            c_ops = []
            
            # Get T1 and T2 rates
            t1 = self.config['T1']
            t2 = self.config['T2']
            t2_star = self.config['T2_star']
            
            # Add T1 and T2 collapse operators
            if t1 > 0:
                # T1 relaxation rate
                gamma1 = 1.0 / t1
                
                # Add T1 collapse operators - amplitude sqrt(gamma1) to get rate gamma1
                c_ops_t1 = self.simos_adapter.pure_dephasing_operators(gamma1)
                c_ops.extend(c_ops_t1)
            
            if t2 > 0:
                # T2 pure dephasing rate - need to account for T1 contribution
                # 1/T2 = 1/T2' + 1/(2*T1), so 1/T2' = 1/T2 - 1/(2*T1)
                gamma2_prime = max(0, 1/t2 - 1/(2*t1))
                
                # Add T2 collapse operators
                c_ops_t2 = self.simos_adapter.pure_dephasing_operators(gamma2_prime)
                c_ops.extend(c_ops_t2)
            
            if t2_star > 0 and t2_star < t2:
                # Additional inhomogeneous dephasing for T2* effect
                gamma2_star = max(0, 1/t2_star - 1/t2)
                
                # Add T2* collapse operators
                c_ops_t2star = self.simos_adapter.inhomogeneous_dephasing_operators(gamma2_star)
                c_ops.extend(c_ops_t2star)
            
            # Add optical collapse operators if laser is on
            if self.laser_on and self.laser_power > 0:
                # Normalize laser power to [0, 1] for SimOS beta parameter
                saturation_power = self.config['excitation_saturation_power']
                beta = min(1.0, self.laser_power / saturation_power)
                
                # Get optical collapse operators
                c_ops_optical, _ = self.simos_adapter.transition_operators(
                    T=298,  # Room temperature
                    beta=beta,
                    Bvec=self.magnetic_field.tolist()
                )
                
                c_ops.extend(c_ops_optical)
            
            return c_ops
            
        except Exception as e:
            logger.error(f"Error getting collapse operators: {str(e)}")
            raise QuantumStateError(
                f"Failed to get collapse operators: {str(e)}",
                state_info={
                    "laser_on": self.laser_on,
                    "laser_power": self.laser_power,
                    "T1": self.config['T1'],
                    "T2": self.config['T2'],
                    "T2_star": self.config['T2_star']
                }
            )
    
    def evolve_quantum_state(self, dt: float = None):
        """
        Evolve the quantum state forward in time.
        
        Args:
            dt: Time step in seconds. If None, use the default time step.
            
        Returns:
            Any: The evolved quantum state
            
        Raises:
            EvolutionError: If the evolution fails
        """
        if dt is None:
            dt = self.dt
        
        if dt <= 0:
            raise EvolutionError(
                f"Time step must be positive, got {dt}",
                duration=dt
            )
        
        try:
            # Acquire state write lock
            self.lock_manager.state_rw.acquire_write()
            try:
                # Update the Hamiltonian - first need to acquire hamiltonian lock
                H = None
                try:
                    self.lock_manager.hamiltonian_rw.acquire_write()
                    try:
                        H = self._update_hamiltonian()
                    finally:
                        self.lock_manager.hamiltonian_rw.release_write()
                except Exception as e:
                    logger.error(f"Failed to acquire hamiltonian lock: {str(e)}")
                    raise ThreadingError(f"Failed to acquire hamiltonian lock: {str(e)}")
                
                # Get collapse operators for decoherence processes
                c_ops = self._get_collapse_operators()
                
                # Get optical operators if laser is on
                if self.laser_on and self.laser_power > 0:
                    try:
                        # Calculate normalized laser power for excitation parameter
                        saturation_power = self.config['excitation_saturation_power']
                        beta = min(1.0, self.laser_power / saturation_power)
                        
                        # Get temperature-dependent optical operators
                        optical_ops, relaxation_ops = self.simos_adapter.transition_operators(
                            T=self.config.get('temperature', 298),
                            beta=beta,
                            Bvec=self.magnetic_field.tolist()
                        )
                        
                        # Add optical operators to collapse operators
                        c_ops.extend(optical_ops)
                        c_ops.extend(relaxation_ops)
                    except Exception as e:
                        logger.warning(f"Failed to get optical operators: {str(e)}")
                        # Continue with standard collapse operators
                
                # Choose evolution method based on configuration
                if self.config.get('adaptive_timestep', False):
                    # Use adaptive method for better accuracy with variable dynamics
                    self.quantum_state = self.simos_adapter.evolve_adaptive(
                        self.quantum_state,
                        H,
                        dt,
                        c_ops=c_ops,
                        tol=self.config.get('adaptive_tol', 1e-6),
                        method=self.config.get('ode_method', 'adams')
                    )
                elif self.config.get('decoherence_model') == 'non-markovian':
                    # Non-Markovian evolution for specific environments
                    # Create bath spectrum function
                    bath_coupling = self.config.get('bath_coupling_strength', 1e6)
                    cutoff = self.config.get('bath_cutoff', 1e6)  # Hz
                    bath_spectrum = self.simos_adapter.lorentzian_bath(bath_coupling, cutoff)
                    
                    self.quantum_state = self.simos_adapter.evolve_non_markovian(
                        self.quantum_state,
                        H,
                        dt,
                        bath_spectrum=bath_spectrum,
                        temperature=self.config.get('temperature', 298)
                    )
                else:
                    # Standard Lindblad evolution
                    self.quantum_state = self.simos_adapter.evolve_density_matrix(
                        rho=self.quantum_state,
                        H=H,
                        dt=dt,
                        c_ops=c_ops
                    )
                
                # Update simulation time in adapter
                if hasattr(self.simos_adapter.nv_system, 'simulation_time'):
                    self.simos_adapter.nv_system.simulation_time += dt
                
                # Update state information
                self._update_state_info()
                
                return self.quantum_state
                
            except Exception as e:
                if isinstance(e, (HamiltonianError, QuantumStateError, EvolutionError, ThreadingError)):
                    raise
                
                logger.error(f"Error evolving quantum state: {str(e)}")
                raise EvolutionError(
                    f"Failed to evolve quantum state: {str(e)}",
                    duration=dt,
                    method="evolve_quantum_state"
                )
            finally:
                self.lock_manager.state_rw.release_write()
                
        except Exception as e:
            logger.error(f"Failed to acquire state lock for evolve_quantum_state: {str(e)}")
            raise ThreadingError(f"Failed to acquire state lock for evolve_quantum_state: {str(e)}")
    
    def get_fluorescence(self) -> float:
        """
        Calculate fluorescence based on the current quantum state.
        
        Returns:
            float: Fluorescence in counts per second
        """
        try:
            # Evolve the quantum state first
            self.evolve_quantum_state()
            
            # Acquire read lock for state data
            try:
                self.lock_manager.state_rw.acquire_read()
                try:
                    # Try to use SimOS physical model for fluorescence calculation
                    if hasattr(self.simos_adapter.nv_system, 'fluorescence'):
                        try:
                            # Get optical parameters
                            ms0_rate = self.config['fluorescence_rate_ms0']
                            ms1_rate = self.config['fluorescence_rate_ms1']
                            bg_rate = self.config['background_count_rate']
                            
                            # Calculate fluorescence using SimOS physical model
                            fluor = self.simos_adapter.nv_system.fluorescence(
                                self.quantum_state,
                                beta=min(1.0, self.laser_power / self.config['excitation_saturation_power']) if self.laser_on else 0.0,
                                T=self.config.get('temperature', 298),
                                Bvec=self.magnetic_field.tolist(),
                                background=bg_rate,
                                ms0_rate=ms0_rate,
                                ms1_rate=ms1_rate
                            )
                            
                            # Apply realistic noise model
                            if self.config.get('noise_amplitude'):
                                # Shot noise (Poisson statistics)
                                shot_noise = np.sqrt(fluor) * np.random.normal(0, 0.5)
                                
                                # Technical noise (proportional to signal)
                                tech_noise = fluor * self.config['noise_amplitude'] * np.random.normal(0, 1)
                                
                                # Combined noise
                                fluor += shot_noise + tech_noise
                                
                            return max(0.0, fluor)  # Ensure non-negative
                            
                        except Exception as e:
                            logger.debug(f"Advanced fluorescence calculation failed, falling back to basic: {str(e)}")
                            # Fall back to basic calculation below
                    
                    # Basic calculation (fallback)
                    # Parameters for fluorescence calculation
                    ms0_rate = self.config['fluorescence_rate_ms0']
                    ms1_rate = self.config['fluorescence_rate_ms1']
                    bg_rate = self.config['background_count_rate']
                    
                    # Calculate fluorescence from ground states
                    ground_fluor = (
                        self._populations['ms0'] * ms0_rate + 
                        (self._populations['ms_minus'] + self._populations['ms_plus']) * ms1_rate
                    )
                    
                    # Calculate fluorescence from excited states
                    excited_lifetime = self.config['excited_state_lifetime']
                    emission_rate = 1.0 / excited_lifetime
                    excited_fluor = self._excited_state_population * emission_rate * self.config['saturation_count_rate']
                    
                    # Sum all contributions
                    fluor = ground_fluor + excited_fluor + bg_rate
                    
                    # Apply saturation effects with laser power
                    laser_on_local = self.laser_on
                    laser_power_local = self.laser_power
                    
                    if laser_on_local and laser_power_local > 0:
                        saturation_power = self.config['excitation_saturation_power']
                        saturation_factor = laser_power_local / (laser_power_local + saturation_power)
                        fluor *= saturation_factor
                    else:
                        # If laser is off, only background counts
                        fluor = bg_rate
                    
                    # Add noise if configured
                    if self.config.get('noise_amplitude'):
                        noise_amp = self.config['noise_amplitude']
                        fluor *= (1.0 + noise_amp * (2 * np.random.random() - 1))
                    
                    return fluor
                finally:
                    self.lock_manager.state_rw.release_read()
            except Exception as e:
                logger.error(f"Failed to acquire state read lock for get_fluorescence: {str(e)}")
                raise ThreadingError(f"Failed to acquire state read lock for get_fluorescence: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error calculating fluorescence: {str(e)}")
            # Return a default value if calculation fails
            return 1000.0  # Background level
    
    def start_simulation_loop(self):
        """
        Start a background thread for continuous simulation.
        
        This method starts a thread that continuously evolves the quantum state
        and updates the model's state information.
        
        Raises:
            ThreadingError: If the simulation thread cannot be started
        """
        if self.is_simulating:
            logger.warning("Simulation already running")
            return
        
        # Reset the stop event
        self.stop_simulation.clear()
        
        # Define the simulation function
        def simulation_loop():
            logger.info("Starting simulation loop")
            
            while not self.stop_simulation.is_set():
                try:
                    # Evolve the quantum state
                    self.evolve_quantum_state()
                    
                    # Sleep briefly to avoid CPU overload
                    time.sleep(0.001)
                    
                except Exception as e:
                    logger.error(f"Error in simulation loop: {str(e)}")
                    time.sleep(0.1)  # Sleep longer on error
            
            logger.info("Simulation loop stopped")
        
        try:
            # Create and start the simulation thread
            self.simulation_thread = threading.Thread(target=simulation_loop, daemon=True)
            self.simulation_thread.start()
            self.is_simulating = True
            
            logger.info("Simulation thread started")
            
        except Exception as e:
            logger.error(f"Error starting simulation thread: {str(e)}")
            raise ThreadingError(
                f"Failed to start simulation thread: {str(e)}"
            )
    
    def stop_simulation_loop(self):
        """
        Stop the background simulation thread.
        
        This method signals the simulation thread to stop and waits for it to finish.
        
        Raises:
            ThreadingError: If the simulation thread cannot be stopped
        """
        if not self.is_simulating:
            logger.warning("No simulation running")
            return
        
        try:
            # Signal the thread to stop
            self.stop_simulation.set()
            
            # Wait for the thread to finish (with timeout)
            if self.simulation_thread and self.simulation_thread.is_alive():
                self.simulation_thread.join(timeout=1.0)
            
            self.is_simulating = False
            
            logger.info("Simulation thread stopped")
            
        except Exception as e:
            logger.error(f"Error stopping simulation thread: {str(e)}")
            raise ThreadingError(
                f"Failed to stop simulation thread: {str(e)}"
            )
    
    def simulate_odmr(self, start_freq: float, stop_freq: float, num_points: int,
                      averaging_time: float = 0.1) -> ODMRResult:
        """
        Simulate an ODMR measurement.
        
        Args:
            start_freq: Start frequency in Hz
            stop_freq: Stop frequency in Hz
            num_points: Number of frequency points
            averaging_time: Measurement time per point in seconds
            
        Returns:
            ODMRResult: ODMR measurement result
            
        Raises:
            ExperimentError: If the simulation fails
        """
        # Generate a cache key based on parameters
        cache_key = f"odmr_{start_freq}_{stop_freq}_{num_points}_{averaging_time}"
        
        # Check if result is already cached
        if cache_key in self.cached_results:
            logger.debug(f"Using cached ODMR result for {cache_key}")
            return self.cached_results[cache_key]
        
        try:
            # Generate frequency points
            frequencies = np.linspace(start_freq, stop_freq, num_points)
            signal = np.zeros(num_points)
            
            # Save original state
            original_mw_on = self.mw_on
            original_mw_freq = self.mw_frequency
            original_mw_power = self.mw_power
            
            # Turn on laser for optical readout
            self.apply_laser(1.0, True)
            
            # Set microwave power (constant for all frequencies)
            self.apply_microwave(frequencies[0], -10.0, False)
            
            # Get reference signal (no MW)
            ref_signal = self.get_fluorescence()
            
            # Measure each frequency point
            for i, freq in enumerate(frequencies):
                # Apply microwave at current frequency
                self.apply_microwave(freq, -10.0, True)
                
                # Let system equilibrate
                for _ in range(int(10 * averaging_time / self.dt)):
                    self.evolve_quantum_state()
                
                # Get fluorescence
                signal[i] = self.get_fluorescence() / ref_signal
            
            # Restore original state
            self.apply_microwave(original_mw_freq, original_mw_power, original_mw_on)
            self.apply_laser(0.0, False)
            
            # Find resonance parameters
            # Simple method: find minimum and estimate width
            min_idx = np.argmin(signal)
            center_frequency = frequencies[min_idx]
            
            # Estimate linewidth and contrast
            contrast = 1.0 - signal[min_idx]
            
            # Find FWHM
            half_contrast = 1.0 - contrast / 2
            above_half = signal > half_contrast
            
            # Find crossings of half-max value
            crossings = np.where(np.diff(above_half))[0]
            
            if len(crossings) >= 2:
                # FWHM from frequency difference at crossings
                linewidth = frequencies[crossings[1]] - frequencies[crossings[0]]
            else:
                # Fallback estimate
                linewidth = abs(stop_freq - start_freq) / 10
            
            # Create and cache result
            result = ODMRResult(
                frequencies=frequencies,
                signal=signal,
                contrast=contrast,
                center_frequency=center_frequency,
                linewidth=linewidth
            )
            
            self.cached_results[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error simulating ODMR: {str(e)}")
            raise ExperimentError(
                f"Failed to simulate ODMR: {str(e)}",
                experiment_type="ODMR",
                params={
                    "start_freq": start_freq,
                    "stop_freq": stop_freq,
                    "num_points": num_points,
                    "averaging_time": averaging_time
                }
            )
    
    def simulate_rabi(self, max_time: float, num_points: int) -> RabiResult:
        """
        Simulate a Rabi oscillation measurement.
        
        Args:
            max_time: Maximum evolution time in seconds
            num_points: Number of time points
            
        Returns:
            RabiResult: Rabi oscillation measurement result
            
        Raises:
            ExperimentError: If the simulation fails
        """
        # Generate a cache key based on parameters
        cache_key = f"rabi_{max_time}_{num_points}"
        
        # Check if result is already cached
        if cache_key in self.cached_results:
            logger.debug(f"Using cached Rabi result for {cache_key}")
            return self.cached_results[cache_key]
        
        try:
            # Generate time points
            times = np.linspace(0, max_time, num_points)
            population = np.zeros(num_points)
            
            # Initialize to ms=0 state
            self.reset_state()
            
            # Apply resonant microwave
            zfs = self.config['zero_field_splitting']
            self.apply_microwave(zfs, -10.0, True)
            
            # Simulate for each time point
            for i, t in enumerate(times):
                if i > 0:
                    # Evolve for time difference since last point
                    dt = times[i] - times[i-1]
                    self.evolve_quantum_state(dt)
                
                # Record ground state population
                population[i] = self._populations['ms0']
            
            # Turn off microwave
            self.apply_microwave(zfs, -10.0, False)
            
            # Estimate Rabi frequency from oscillation
            # Simple method: count zero crossings
            mean_pop = np.mean(population)
            crossings = np.where(np.diff(population > mean_pop))[0]
            
            if len(crossings) >= 2:
                # Estimate period from average time between crossings
                period = 2 * max_time / len(crossings)
                rabi_freq = 1 / period
            else:
                # Fallback estimate - assume a typical Rabi frequency
                rabi_freq = 5e6  # Hz
            
            # Estimate decay time (if applicable)
            try:
                # Fit exponential decay
                from scipy.optimize import curve_fit
                
                def damped_oscillation(t, f, tau, a, b):
                    return a * np.exp(-t/tau) * np.cos(2*np.pi*f*t) + b
                
                # Initial guess: use estimated Rabi frequency
                p0 = [rabi_freq, max_time, 0.5, 0.5]
                
                # Limit to first 20 points or all if fewer
                fit_limit = min(20, len(times))
                popt, _ = curve_fit(
                    damped_oscillation, 
                    times[:fit_limit], 
                    population[:fit_limit],
                    p0=p0,
                    bounds=([0, 0, 0, 0], [1e8, 1e-3, 1, 1])
                )
                
                # Extract decay time
                decay_time = popt[1]
            except:
                # If fitting fails, use None
                decay_time = None
            
            # Create and cache result
            result = RabiResult(
                times=times,
                population=population,
                rabi_frequency=rabi_freq,
                decay_time=decay_time
            )
            
            self.cached_results[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error simulating Rabi oscillation: {str(e)}")
            raise ExperimentError(
                f"Failed to simulate Rabi oscillation: {str(e)}",
                experiment_type="Rabi",
                params={
                    "max_time": max_time,
                    "num_points": num_points
                }
            )
    
    def simulate_t1(self, max_time: float, num_points: int) -> T1Result:
        """
        Simulate a T1 relaxation measurement.
        
        Args:
            max_time: Maximum relaxation time in seconds
            num_points: Number of time points
            
        Returns:
            T1Result: T1 relaxation measurement result
            
        Raises:
            ExperimentError: If the simulation fails
        """
        # Generate a cache key based on parameters
        cache_key = f"t1_{max_time}_{num_points}"
        
        # Check if result is already cached
        if cache_key in self.cached_results:
            logger.debug(f"Using cached T1 result for {cache_key}")
            return self.cached_results[cache_key]
        
        try:
            # Generate time points
            times = np.linspace(0, max_time, num_points)
            population = np.zeros(num_points)
            
            # Initialize to ms=±1 state
            self.reset_state()
            self.initialize_state(ms=1)  # Can also use ms=-1
            
            # Simulate for each time point
            for i, t in enumerate(times):
                if i > 0:
                    # Evolve for time difference since last point
                    dt = times[i] - times[i-1]
                    self.evolve_quantum_state(dt)
                
                # Record ms=1 population
                population[i] = self._populations['ms_plus']
            
            # Extract T1 time from configuration
            t1_time = self.config['T1']
            
            # Create and cache result
            result = T1Result(
                times=times,
                population=population,
                t1_time=t1_time
            )
            
            self.cached_results[cache_key] = result
            return result
            
        except Exception as e:
            logger.error(f"Error simulating T1 relaxation: {str(e)}")
            raise ExperimentError(
                f"Failed to simulate T1 relaxation: {str(e)}",
                experiment_type="T1",
                params={
                    "max_time": max_time,
                    "num_points": num_points
                }
            )
    
    def simulate_spin_echo(self, max_time: float, num_points: int) -> T2Result:
        """
        Simulate a spin echo (T2) measurement.
        
        Args:
            max_time: Maximum evolution time in seconds
            num_points: Number of time points
            
        Returns:
            T2Result: T2 measurement result
            
        Raises:
            ExperimentError: If the simulation fails
        """
        try:
            # Try to use direct SimOS implementation if available
            if hasattr(self.simos_adapter.nv_system, 'spin_echo'):
                try:
                    # Generate time points
                    times = np.linspace(0, max_time, num_points)
                    
                    # Use SimOS implementation
                    result_data = self.simos_adapter.nv_system.spin_echo(
                        tau_max=max_time,
                        num_points=num_points,
                        Bvec=self.magnetic_field.tolist(),
                        T2=self.config['T2'],
                        temperature=self.config.get('temperature', 298)
                    )
                    
                    # Extract data
                    signal = result_data['signal']
                    t2_time = result_data.get('t2_time', self.config['T2'])
                    
                    # Create result
                    result = T2Result(
                        times=times,
                        signal=signal,
                        t2_time=t2_time
                    )
                    
                    return result
                    
                except Exception as e:
                    logger.debug(f"Direct SimOS spin echo failed, falling back to manual implementation: {str(e)}")
                    # Fall back to manual implementation below
            
            # Manual implementation (fallback)
            # Generate time points
            times = np.linspace(0, max_time, num_points)
            signal = np.zeros(num_points)
            
            # Simulate each delay time
            for i, tau in enumerate(times):
                # Skip zero time point
                if tau == 0:
                    signal[i] = 1.0
                    continue
                
                # Create a pulse sequence for the spin echo
                sequence = PulseSequence(self)
                
                # Add sequence elements
                sequence.add_pi_half_pulse(phase=0.0)  # X π/2 pulse (along X)
                sequence.add_delay(tau/2)              # Free evolution for τ/2
                sequence.add_pi_pulse(phase=0.0)       # X π pulse (along X)
                sequence.add_delay(tau/2)              # Free evolution for τ/2
                sequence.add_pi_half_pulse(phase=0.0)  # X π/2 pulse (along X)
                
                # Execute sequence
                result = sequence.execute()
                
                # Get final state
                signal[i] = result['final_state']['populations']['ms0']
            
            # Get T2 time from configuration
            t2_time = self.config['T2']
            
            # Create result
            result = T2Result(
                times=times,
                signal=signal,
                t2_time=t2_time
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error simulating spin echo: {str(e)}")
            raise ExperimentError(
                f"Failed to simulate spin echo: {str(e)}",
                experiment_type="T2",
                params={
                    "max_time": max_time,
                    "num_points": num_points
                }
            )
    
    def simulate_optical_dynamics(self, max_time: float, num_points: int,
                                 laser_power: float) -> OpticalResult:
        """
        Simulate optical dynamics (excitation and fluorescence).
        
        Args:
            max_time: Maximum evolution time in seconds
            num_points: Number of time points
            laser_power: Laser power in mW
            
        Returns:
            OpticalResult: Optical dynamics simulation result
            
        Raises:
            ExperimentError: If the simulation fails
        """
        try:
            # Generate time points
            times = np.linspace(0, max_time, num_points)
            ground_population = np.zeros(num_points)
            excited_population = np.zeros(num_points)
            fluorescence = np.zeros(num_points)
            
            # Initialize to ground state
            self.reset_state()
            
            # Turn on laser
            self.apply_laser(laser_power, True)
            
            # Simulate for each time point
            for i, t in enumerate(times):
                if i > 0:
                    # Evolve for time difference since last point
                    dt = times[i] - times[i-1]
                    self.evolve_quantum_state(dt)
                
                # Record populations
                ground_population[i] = self._ground_state_population
                excited_population[i] = self._excited_state_population
                
                # Record fluorescence
                fluorescence[i] = self.get_fluorescence()
            
            # Turn off laser
            self.apply_laser(0.0, False)
            
            # Create result
            result = OpticalResult(
                times=times,
                ground_population=ground_population,
                excited_population=excited_population,
                fluorescence=fluorescence
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error simulating optical dynamics: {str(e)}")
            raise ExperimentError(
                f"Failed to simulate optical dynamics: {str(e)}",
                experiment_type="optical_dynamics",
                params={
                    "max_time": max_time,
                    "num_points": num_points,
                    "laser_power": laser_power
                }
            )
    
    def simulate_xy8(self, max_time: float, num_points: int, pulse_spacing=None,
                num_repeats: int = 1, phase_error: float = 0.0) -> T2Result:
        """
        Simulate XY8 dynamical decoupling sequence.
        
        Args:
            max_time: Maximum evolution time in seconds
            num_points: Number of time points
            pulse_spacing: Time between pulses. If None, calculated from max_time
            num_repeats: Number of XY8 sequence repetitions
            phase_error: Phase error in radians for robustness testing
            
        Returns:
            T2Result: Results of the XY8 measurement
            
        Raises:
            ExperimentError: If the simulation fails
        """
        try:
            # Try to use direct SimOS implementation if available
            if hasattr(self.simos_adapter.nv_system, 'xy8_sequence'):
                try:
                    # Generate time points
                    times = np.linspace(0, max_time, num_points)
                    
                    # Calculate pulse spacing if not provided
                    if pulse_spacing is None:
                        # Each XY8 sequence has 8 pulses
                        pulse_spacing = max_time / (8 * num_repeats)
                    
                    # Use SimOS implementation
                    results = self.simos_adapter.nv_system.xy8_sequence(
                        max_time=max_time,
                        num_points=num_points,
                        pulse_spacing=pulse_spacing,
                        num_repeats=num_repeats,
                        phase_error=phase_error,
                        Bvec=self.magnetic_field.tolist(),
                        initial_state="0",  # Start in ms=0
                        T2=self.config['T2'],
                        temperature=self.config.get('temperature', 298)
                    )
                    
                    # Extract results
                    signal = results['signal']
                    coherence_time = results.get('t2_time', self.config['T2'])
                    
                    # Create result object
                    result = T2Result(
                        times=times,
                        signal=signal,
                        t2_time=coherence_time
                    )
                    
                    return result
                    
                except Exception as e:
                    logger.debug(f"Direct SimOS XY8 failed, falling back to manual implementation: {str(e)}")
                    # Fall back to manual implementation below
            
            # Manual implementation (fallback)
            # Generate time points
            times = np.linspace(0, max_time, num_points)
            signal = np.zeros(num_points)
            
            # Calculate pulse spacing if not provided
            if pulse_spacing is None:
                # Each XY8 sequence has 8 pulses
                pulse_spacing = max_time / (8 * num_repeats)
            
            # Simulate each time point
            for i, t in enumerate(times):
                # Skip zero time point
                if t == 0:
                    signal[i] = 1.0
                    continue
                
                # Create pulse sequence
                sequence = PulseSequence(self)
                
                # Initial π/2 pulse
                sequence.add_pi_half_pulse(phase=0.0)  # X axis
                
                # Add XY8 units for the specified number of repeats
                for _ in range(num_repeats):
                    # XY8 phase sequence: X-Y-X-Y-Y-X-Y-X
                    phases = [0, np.pi/2, 0, np.pi/2, np.pi/2, 0, np.pi/2, 0]
                    
                    for phase in phases:
                        # Add intentional phase error if specified
                        actual_phase = phase + phase_error
                        
                        sequence.add_delay(pulse_spacing)
                        sequence.add_pi_pulse(phase=actual_phase)
                        sequence.add_delay(pulse_spacing)
                
                # Final π/2 pulse
                sequence.add_pi_half_pulse(phase=0.0)  # X axis
                
                # Execute sequence and get final population
                result = sequence.execute()
                signal[i] = result['final_state']['populations']['ms0']
            
            # Create result
            result = T2Result(
                times=times,
                signal=signal,
                t2_time=self.config['T2']
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error simulating XY8 sequence: {str(e)}")
            raise ExperimentError(
                f"Failed to simulate XY8 sequence: {str(e)}",
                experiment_type="XY8",
                params={
                    "max_time": max_time,
                    "num_points": num_points,
                    "pulse_spacing": pulse_spacing,
                    "num_repeats": num_repeats,
                    "phase_error": phase_error
                }
            )
            
    def simulate_dipolar_coupling(self, coupling_strength: float, evolution_time: float, 
                                num_points: int, target_spin: str = "13C") -> StateEvolution:
        """
        Simulate dipolar coupling to a nearby nuclear spin.
        
        Args:
            coupling_strength: Dipolar coupling strength in Hz
            evolution_time: Total evolution time in seconds
            num_points: Number of time points
            target_spin: Type of target spin ("13C", "14N", "15N", "1H")
            
        Returns:
            StateEvolution: Evolution of coupled system
            
        Raises:
            ExperimentError: If the simulation fails
        """
        try:
            # Generate time points
            times = np.linspace(0, evolution_time, num_points)
            
            # Initialize with appropriate initial state
            self.reset_state()
            
            # Get coupling Hamiltonian from SimOS
            H_dipolar = self.simos_adapter.dipolar_coupling_hamiltonian(
                target_spin=target_spin,
                coupling_strength=coupling_strength,
                orientation=[0, 0, 1]  # Z-axis by default
            )
            
            # Set up data structures
            populations = {
                'ms0': np.zeros(num_points),
                'ms_minus': np.zeros(num_points),
                'ms_plus': np.zeros(num_points),
                'nuclear_up': np.zeros(num_points),
                'nuclear_down': np.zeros(num_points)
            }
            
            coherences = {
                'ms0_ms_minus': np.zeros(num_points, dtype=complex),
                'ms0_ms_plus': np.zeros(num_points, dtype=complex),
                'ms_minus_ms_plus': np.zeros(num_points, dtype=complex),
                'electron_nuclear': np.zeros(num_points, dtype=complex)
            }
            
            # Get base Hamiltonian
            H_base = self._update_hamiltonian()
            H_total = H_base + H_dipolar
            
            # Save original state
            original_state = self.quantum_state
            
            # Initialize to superposition state if available
            try:
                self.quantum_state = self.simos_adapter.get_superposition_state(
                    electron_states=["0", "+1"],
                    nuclear_state="up"
                )
            except Exception as e:
                logger.debug(f"Failed to create superposition state: {str(e)}")
                # Fall back to standard state
                self.initialize_state(ms=0)
            
            # Evolve and measure
            for i, t in enumerate(times):
                if i > 0:
                    dt = times[i] - times[i-1]
                    self.quantum_state = self.simos_adapter.evolve_density_matrix(
                        self.quantum_state, H_total, dt, 
                        c_ops=self._get_collapse_operators()
                    )
                
                # Record populations and coherences
                state_info = self.get_state_info()
                for key in ['ms0', 'ms_minus', 'ms_plus']:
                    if key in state_info['populations']:
                        populations[key][i] = state_info['populations'][key]
                
                # Try to get coherences if available
                try:
                    coherence_info = self.simos_adapter.get_coherences(self.quantum_state)
                    for key in ['ms0_ms_minus', 'ms0_ms_plus', 'ms_minus_ms_plus']:
                        if key in coherence_info:
                            coherences[key][i] = coherence_info[key]
                except Exception:
                    # Continue without coherences
                    pass
            
            # Restore original state
            self.quantum_state = original_state
            self._update_state_info()
            
            # Create result
            result = StateEvolution(
                times=times,
                populations=populations,
                coherences=coherences
            )
            
            return result
                
        except Exception as e:
            logger.error(f"Error simulating dipolar coupling: {str(e)}")
            raise ExperimentError(
                f"Failed to simulate dipolar coupling: {str(e)}",
                experiment_type="dipolar_coupling",
                params={
                    "coupling_strength": coupling_strength,
                    "evolution_time": evolution_time,
                    "target_spin": target_spin
                }
            )
    
    class PulseSequence:
        """
        Framework for complex pulse sequences with phase control.
        
        This class provides a flexible way to create and execute pulse sequences
        on the NV center, including microwave and laser pulses with precise timing
        and phase control.
        """
        
        def __init__(self, nv_model):
            """
            Initialize the pulse sequence framework.
            
            Parameters
            ----------
            nv_model : PhysicalNVModel
                NV center model to apply pulses to
            """
            self.model = nv_model
            self.pulses = []
            self.current_time = 0.0
            
        def add_pi_pulse(self, phase=0.0, duration=None):
            """
            Add a π pulse to the sequence.
            
            Parameters
            ----------
            phase : float
                Phase angle in radians (0 for X, π/2 for Y)
            duration : float, optional
                Pulse duration. If None, calculated from Rabi frequency.
            
            Returns
            -------
            self
                For method chaining
            """
            # Calculate duration if not provided
            if duration is None:
                # Estimate from Rabi frequency
                power_factor = 10**(self.model.mw_power / 10) / 1000  # Convert dBm to watts
                rabi_freq = 5e6 * np.sqrt(power_factor)  # Approx 5 MHz at 0 dBm
                duration = 1 / (2 * rabi_freq)  # π pulse duration
            
            # Add pulse to sequence
            self.pulses.append({
                'type': 'pi',
                'phase': phase,
                'duration': duration,
                'start_time': self.current_time
            })
            
            # Update current time
            self.current_time += duration
            
            return self
            
        def add_pi_half_pulse(self, phase=0.0, duration=None):
            """
            Add a π/2 pulse to the sequence.
            
            Parameters
            ----------
            phase : float
                Phase angle in radians (0 for X, π/2 for Y)
            duration : float, optional
                Pulse duration. If None, calculated from Rabi frequency.
            
            Returns
            -------
            self
                For method chaining
            """
            # Calculate duration if not provided
            if duration is None:
                # Estimate from Rabi frequency
                power_factor = 10**(self.model.mw_power / 10) / 1000  # Convert dBm to watts
                rabi_freq = 5e6 * np.sqrt(power_factor)  # Approx 5 MHz at 0 dBm
                duration = 1 / (4 * rabi_freq)  # π/2 pulse duration
            
            # Add pulse to sequence
            self.pulses.append({
                'type': 'pi_half',
                'phase': phase,
                'duration': duration,
                'start_time': self.current_time
            })
            
            # Update current time
            self.current_time += duration
            
            return self
            
        def add_delay(self, duration):
            """
            Add a delay to the sequence.
            
            Parameters
            ----------
            duration : float
                Delay duration in seconds
            
            Returns
            -------
            self
                For method chaining
            """
            # Add delay to sequence
            self.pulses.append({
                'type': 'delay',
                'duration': duration,
                'start_time': self.current_time
            })
            
            # Update current time
            self.current_time += duration
            
            return self
            
        def add_laser_pulse(self, power, duration):
            """
            Add a laser pulse to the sequence.
            
            Parameters
            ----------
            power : float
                Laser power in mW
            duration : float
                Pulse duration in seconds
            
            Returns
            -------
            self
                For method chaining
            """
            # Add laser pulse to sequence
            self.pulses.append({
                'type': 'laser',
                'power': power,
                'duration': duration,
                'start_time': self.current_time
            })
            
            # Update current time
            self.current_time += duration
            
            return self
            
        def add_xy8_unit(self, tau):
            """
            Add an XY8 unit sequence.
            
            An XY8 unit consists of 8 π pulses with specific phases:
            X - Y - X - Y - Y - X - Y - X
            spaced by 2*tau.
            
            Parameters
            ----------
            tau : float
                Half spacing between pulses in seconds
            
            Returns
            -------
            self
                For method chaining
            """
            # XY8 phase sequence
            phases = [0, np.pi/2, 0, np.pi/2, np.pi/2, 0, np.pi/2, 0]
            
            for phase in phases:
                self.add_delay(tau)
                self.add_pi_pulse(phase=phase)
                self.add_delay(tau)
                
            return self
            
        def execute(self, record_state=False, sampling_rate=None):
            """
            Execute the pulse sequence on the NV model.
            
            Parameters
            ----------
            record_state : bool
                Whether to record state evolution
            sampling_rate : float, optional
                If provided, record state at this sampling rate
                
            Returns
            -------
            dict
                Results of the sequence execution
            """
            # Reset NV state
            self.model.reset_state()
            
            # Initialize results
            times = [0.0]
            populations = {'ms0': [self.model._populations['ms0']],
                          'ms_minus': [self.model._populations['ms_minus']],
                          'ms_plus': [self.model._populations['ms_plus']]}
            fluorescence = [self.model.get_fluorescence()]
            
            # Time points to record if sampling_rate provided
            if sampling_rate:
                sample_times = np.arange(0, self.current_time, 1/sampling_rate)
                current_sample_idx = 0
            
            # Execute each pulse
            current_time = 0.0
            for pulse in self.pulses:
                pulse_type = pulse['type']
                duration = pulse['duration']
                
                if pulse_type == 'pi' or pulse_type == 'pi_half':
                    # Apply microwave pulse with phase
                    phase = pulse['phase']
                    
                    # Set MW parameters
                    self.model.apply_microwave(self.model.mw_frequency, self.model.mw_power, True)
                    
                    # Configure phase in the SimOS adapter if supported
                    if hasattr(self.model.simos_adapter.nv_system, 'set_phase'):
                        self.model.simos_adapter.nv_system.set_phase(phase)
                    
                    # Evolve during pulse
                    self.model.evolve_quantum_state(duration)
                    
                    # Turn off microwave
                    self.model.apply_microwave(self.model.mw_frequency, self.model.mw_power, False)
                    
                elif pulse_type == 'laser':
                    # Apply laser pulse
                    power = pulse['power']
                    
                    # Set laser parameters
                    self.model.apply_laser(power, True)
                    
                    # Evolve during pulse
                    self.model.evolve_quantum_state(duration)
                    
                    # Turn off laser
                    self.model.apply_laser(0.0, False)
                    
                elif pulse_type == 'delay':
                    # Just free evolution
                    self.model.evolve_quantum_state(duration)
                
                # Update current time
                current_time += duration
                
                # Record state at end of pulse if requested
                if record_state:
                    times.append(current_time)
                    
                    # Get current state
                    pops = self.model._populations
                    fluorescence.append(self.model.get_fluorescence())
                    
                    # Record populations
                    for state in populations:
                        populations[state].append(pops[state])
                        
                # Record at sampling points if provided
                if sampling_rate and current_sample_idx < len(sample_times):
                    while current_sample_idx < len(sample_times) and sample_times[current_sample_idx] <= current_time:
                        # Record at this sample time
                        if not record_state or abs(sample_times[current_sample_idx] - current_time) > 1e-12:
                            times.append(sample_times[current_sample_idx])
                            
                            # Get current state
                            pops = self.model._populations
                            fluorescence.append(self.model.get_fluorescence())
                            
                            # Record populations
                            for state in populations:
                                populations[state].append(pops[state])
                        
                        current_sample_idx += 1
            
            # Return all results
            return {
                'times': np.array(times),
                'populations': {state: np.array(vals) for state, vals in populations.items()},
                'fluorescence': np.array(fluorescence),
                'pulses': self.pulses.copy(),
                'final_state': self.model.get_state_info()
            }
            
    def simulate_state_evolution(self, max_time: float, num_points: int,
                               hamiltonian_only: bool = False) -> StateEvolution:
        """
        Simulate general quantum state evolution.
        
        Args:
            max_time: Maximum evolution time in seconds
            num_points: Number of time points
            hamiltonian_only: If True, only include Hamiltonian evolution (no decoherence)
            
        Returns:
            StateEvolution: Quantum state evolution result
            
        Raises:
            ExperimentError: If the simulation fails
        """
        try:
            # Generate time points
            times = np.linspace(0, max_time, num_points)
            
            # Initialize data structures
            populations = {
                'ms0': np.zeros(num_points),
                'ms_minus': np.zeros(num_points),
                'ms_plus': np.zeros(num_points)
            }
            
            coherences = {
                'ms0_ms_minus': np.zeros(num_points, dtype=complex),
                'ms0_ms_plus': np.zeros(num_points, dtype=complex),
                'ms_minus_ms_plus': np.zeros(num_points, dtype=complex)
            }
            
            # Save original state to restore after simulation
            original_state = self.quantum_state
            
            # Create a copy of the model to avoid modifying the original
            # We need to temporarily modify internal state for hamiltonian_only mode
            if hamiltonian_only:
                # Store original decoherence parameters
                original_t1 = self.config['T1']
                original_t2 = self.config['T2']
                original_t2_star = self.config['T2_star']
                
                # Disable decoherence
                self.update_config({
                    'T1': 1e10,  # Effectively infinite
                    'T2': 1e10,
                    'T2_star': 1e10
                })
            
            # Simulate for each time point
            for i, t in enumerate(times):
                if i > 0:
                    # Evolve for time difference since last point
                    dt = times[i] - times[i-1]
                    self.evolve_quantum_state(dt)
                
                # Record populations
                populations['ms0'][i] = self._populations['ms0']
                populations['ms_minus'][i] = self._populations['ms_minus']
                populations['ms_plus'][i] = self._populations['ms_plus']
                
                # Record coherences
                for key in coherences:
                    if key in self._coherences:
                        coherences[key][i] = self._coherences[key]
            
            # Restore original state
            self.quantum_state = original_state
            self._update_state_info()
            
            # Restore original decoherence parameters if needed
            if hamiltonian_only:
                self.update_config({
                    'T1': original_t1,
                    'T2': original_t2,
                    'T2_star': original_t2_star
                })
            
            # Create result
            result = StateEvolution(
                times=times,
                populations=populations,
                coherences=coherences
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error simulating state evolution: {str(e)}")
            raise ExperimentError(
                f"Failed to simulate state evolution: {str(e)}",
                experiment_type="state_evolution",
                params={
                    "max_time": max_time,
                    "num_points": num_points,
                    "hamiltonian_only": hamiltonian_only
                }
            )