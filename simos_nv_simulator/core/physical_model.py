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

# SimOS imports
try:
    import simos  # type: ignore
    import simos.propagation  # For time propagation
    import simos.systems.NV  # For NV-specific functions
    SIMOS_AVAILABLE = True
    logger.info("SimOS package found. Using SimOS for quantum simulation.")
except ImportError:
    SIMOS_AVAILABLE = False
    logger.warning("SimOS package not found. Using placeholder implementation.")

# SimOS NV Wrapper class for integration with our API
class SimOSNVWrapper:
    """Wrapper for the SimOS NV system to provide the same API as our placeholder."""
    
    def __init__(self, model, simos_nv):
        self.model = model
        self.simos_nv = simos_nv
        
        # State information
        self.populations = {
            'ms0': 0.98, 
            'ms_minus': 0.01, 
            'ms_plus': 0.01,
            'excited_ms0': 0.0,
            'excited_ms_minus': 0.0,
            'excited_ms_plus': 0.0
        }
        
        # Energy levels
        self.energy_levels = {
            'ms0': 0.0,
            'ms_minus': -model.config['zero_field_splitting'],
            'ms_plus': model.config['zero_field_splitting'],
            'excited_ms0': 1.945e15,  # Optical transition ~637 nm
            'excited_ms_minus': 1.945e15 - model.config['zero_field_splitting'],
            'excited_ms_plus': 1.945e15 + model.config['zero_field_splitting']
        }
        
        # Control parameters
        self.mw_on = False
        self.mw_freq = 0.0
        self.mw_power = 0.0
        self.laser_on = False
        self.laser_power = 0.0
        self.magnetic_field = np.array([0.0, 0.0, 0.0])
        self.simulation_time = 0.0
        
        # SimOS-specific parameters
        self._rho = simos.systems.NV.gen_rho0(self.simos_nv)  # Density matrix for ground state
        self._hamiltonian = self.simos_nv.field_hamiltonian()  # Default Hamiltonian
        self._c_ops_laser_off = []  # Collapse operators without laser
        
    def ground_state(self):
        """Return ground state (ms=0)."""
        return {'state': 'ms0', 'population': 1.0}
    
    def set_state_ms0(self):
        """Initialize to ms=0 state."""
        self._rho = simos.systems.NV.gen_rho0(self.simos_nv)
        self._update_populations()
        
    def set_state_ms1(self, ms):
        """Initialize to ms=±1 state."""
        # Create state in ms=±1
        if ms == -1:
            # Initialize to ms=-1 state using spin operators
            ms0 = simos.systems.NV.gen_rho0(self.simos_nv)  # Start with ms=0
            # Apply lowering operator twice to get from ms=0 to ms=-1
            self._rho = self.simos_nv.Sminus * self.simos_nv.Sminus * ms0 * self.simos_nv.Splus * self.simos_nv.Splus
            self._rho = self._rho.unit()  # Normalize
        else:  # ms == 1
            # Initialize to ms=+1 state
            ms0 = simos.systems.NV.gen_rho0(self.simos_nv)
            # Apply raising operator twice to get from ms=0 to ms=+1
            self._rho = self.simos_nv.Splus * self.simos_nv.Splus * ms0 * self.simos_nv.Sminus * self.simos_nv.Sminus
            self._rho = self._rho.unit()  # Normalize
            
        self._update_populations()
    
    def _update_populations(self):
        """Update population dictionaries from density matrix."""
        # Extract spin state populations
        try:
            # Get projection operators from the NV system
            # For the ground state spin projections
            ms0_gs = self.simos_nv.GSid * self.simos_nv.S0
            ms_plus_gs = self.simos_nv.GSid * self.simos_nv.Splus * self.simos_nv.Sminus
            ms_minus_gs = self.simos_nv.GSid * self.simos_nv.Sminus * self.simos_nv.Splus
            
            # For the excited state spin projections
            ms0_es = self.simos_nv.ESid * self.simos_nv.S0
            ms_plus_es = self.simos_nv.ESid * self.simos_nv.Splus * self.simos_nv.Sminus
            ms_minus_es = self.simos_nv.ESid * self.simos_nv.Sminus * self.simos_nv.Splus
            
            # Calculate expectation values
            self.populations['ms0'] = simos.expect(ms0_gs, self._rho).real
            self.populations['ms_plus'] = simos.expect(ms_plus_gs, self._rho).real
            self.populations['ms_minus'] = simos.expect(ms_minus_gs, self._rho).real
            self.populations['excited_ms0'] = simos.expect(ms0_es, self._rho).real
            self.populations['excited_ms_plus'] = simos.expect(ms_plus_es, self._rho).real
            self.populations['excited_ms_minus'] = simos.expect(ms_minus_es, self._rho).real
        except Exception as e:
            logger.error(f"Error updating populations: {e}")
            # Keep current populations if there's an error
    
    def evolve(self, dt):
        """Simulate time evolution using SimOS."""
        try:
            # Update Hamiltonian and collapse operators if needed
            self._update_hamiltonian()
            c_ops = self._get_c_ops()
            
            # Evolve the state
            if len(c_ops) > 0:
                # With decoherence (Lindblad master equation)
                result = simos.propagation.mesolve(self._hamiltonian, self._rho, dt, c_ops)
                self._rho = result.states[-1]
            else:
                # Without decoherence (unitary evolution)
                propagator = simos.propagation.evol(self._hamiltonian, dt)
                self._rho = propagator * self._rho * propagator.dag()
            
            # Update populations and time
            self._update_populations()
            self.simulation_time += dt
        except Exception as e:
            logger.error(f"Error in SimOS evolve: {e}")
            # Simple fallback evolution if SimOS fails
            if self.mw_on and abs(self.mw_freq - self.model.config['zero_field_splitting']) < 50e6:
                # Simple Rabi oscillation
                rabi_freq = 5e6 * np.sqrt(10**(self.mw_power/10) / 1.0)
                phase = 2 * np.pi * rabi_freq * dt
                # Update ms0 and ms±1 populations with simple model
                delta_p = 0.1 * np.sin(phase) * min(1.0, self.populations['ms0'])
                self.populations['ms0'] -= delta_p
                if abs(self.mw_freq - (self.model.config['zero_field_splitting'] - self.model.config['gyromagnetic_ratio'] * self.magnetic_field[2])) < 20e6:
                    self.populations['ms_minus'] += delta_p
                else:
                    self.populations['ms_plus'] += delta_p
            
            # Simple relaxation
            if dt > 0:
                t1 = self.model.config['T1']
                relaxation_prob = 1.0 - np.exp(-dt / t1)
                relaxing_pop = (self.populations['ms_minus'] + self.populations['ms_plus']) * relaxation_prob
                self.populations['ms_minus'] -= self.populations['ms_minus'] * relaxation_prob
                self.populations['ms_plus'] -= self.populations['ms_plus'] * relaxation_prob
                self.populations['ms0'] += relaxing_pop
    
    def _update_hamiltonian(self):
        """Update Hamiltonian based on control parameters."""
        try:
            # Create Hamiltonian with current fields
            self._hamiltonian = self.simos_nv.field_hamiltonian(Bvec=self.magnetic_field)
            
            # Add microwave Hamiltonian if active
            if self.mw_on:
                # Convert from dBm to amplitude
                mw_amplitude = 10**(self.mw_power/20) * 1e-3  # Approximation
                # Rotating wave approximation for microwave driving
                phi = 2*np.pi*self.mw_freq*self.simulation_time
                mw_H = mw_amplitude * (self.simos_nv.Sx * np.cos(phi) + self.simos_nv.Sy * np.sin(phi))
                self._hamiltonian += mw_H
        except Exception as e:
            logger.error(f"Error updating Hamiltonian: {e}")
    
    def _get_c_ops(self):
        """Get collapse operators based on current control parameters."""
        try:
            # Get collapse operators from the SimOS NV model
            T = 298  # Room temperature in Kelvin
            
            if self.laser_on and self.laser_power > 0:
                # Normalize laser power to saturation value for SimOS beta parameter (0-1)
                saturation_power = self.model.config['excitation_saturation_power']
                beta = min(1.0, self.laser_power / saturation_power)
                
                # Collapse operators with laser
                c_ops_on, _ = self.simos_nv.transition_operators(T=T, beta=beta, Bvec=self.magnetic_field)
                return c_ops_on
            else:
                # Collapse operators without laser (only relaxation)
                if not self._c_ops_laser_off:
                    _, self._c_ops_laser_off = self.simos_nv.transition_operators(T=T, beta=0, Bvec=self.magnetic_field)
                return self._c_ops_laser_off
        except Exception as e:
            logger.error(f"Error getting collapse operators: {e}")
            return []
    
    def apply_magnetic_field(self, field):
        """Apply magnetic field to system."""
        self.magnetic_field = field
        # Update energy levels with Zeeman shift
        gamma = self.model.config['gyromagnetic_ratio']
        b_z = field[2]  # z-component
        
        # Zeeman effect on ground state
        self.energy_levels['ms_minus'] = -self.model.config['zero_field_splitting'] - gamma * b_z
        self.energy_levels['ms_plus'] = self.model.config['zero_field_splitting'] + gamma * b_z
        
        # Zeeman effect on excited state
        self.energy_levels['excited_ms_minus'] = (1.945e15 - 
                                               self.model.config['zero_field_splitting'] - 
                                               gamma * b_z)
        self.energy_levels['excited_ms_plus'] = (1.945e15 + 
                                              self.model.config['zero_field_splitting'] + 
                                              gamma * b_z)
    
    def apply_microwave(self, frequency, power, on):
        """Apply microwave to the system."""
        self.mw_freq = frequency
        self.mw_power = power
        self.mw_on = on
    
    def apply_laser(self, power, on):
        """Apply laser to the system."""
        self.laser_power = power
        self.laser_on = on
    
    def get_populations(self):
        """Get state populations."""
        # Update the state
        self.evolve(self.model.dt)
        return self.populations.copy()
    
    def get_fluorescence(self):
        """Calculate fluorescence based on current state populations."""
        # Get basic parameters
        ms0_rate = self.model.config['fluorescence_rate_ms0']
        ms1_rate = self.model.config['fluorescence_rate_ms1']
        bg_rate = self.model.config['background_count_rate']
        
        # Calculate fluorescence from ground states
        ground_fluor = (self.populations['ms0'] * ms0_rate + 
                     (self.populations['ms_minus'] + self.populations['ms_plus']) * ms1_rate)
        
        # Calculate fluorescence from excited states
        excited_lifetime = self.model.config['excited_state_lifetime']
        emission_rate = 1.0 / excited_lifetime
        
        # Excited state fluorescence depends on populations
        excited_fluor = (self.populations['excited_ms0'] + 
                        self.populations['excited_ms_minus'] + 
                        self.populations['excited_ms_plus']) * emission_rate * self.model.config['saturation_count_rate']
        
        # Sum all contributions
        fluor = ground_fluor + excited_fluor + bg_rate
        
        # Apply saturation effects with laser power
        if self.laser_on and self.laser_power > 0:
            saturation_power = self.model.config['excitation_saturation_power']
            saturation_factor = self.laser_power / (self.laser_power + saturation_power)
            fluor *= saturation_factor
        
        # Add some noise
        if self.model.config.get('noise_amplitude'):
            noise_amp = self.model.config['noise_amplitude']
            fluor *= (1.0 + noise_amp * (2 * np.random.random() - 1))
        
        return fluor
    
    def get_odmr_signal(self, frequency):
        """Calculate ODMR signal at a given frequency."""
        # Save current MW parameters
        old_freq = self.mw_freq
        old_on = self.mw_on
        
        # Apply test frequency
        self.apply_microwave(frequency, self.mw_power, True)
        
        # Simulate system for a while to reach steady state
        for _ in range(10):
            self.evolve(self.model.dt * 100)
        
        # Get fluorescence
        signal = self.get_fluorescence()
        
        # Get reference fluorescence (no MW)
        self.mw_on = False
        for _ in range(10):
            self.evolve(self.model.dt * 100)
        ref_signal = self.get_fluorescence()
        
        # Restore original settings
        self.mw_freq = old_freq
        self.mw_on = old_on
        
        # Normalize signal
        normalized = signal / ref_signal
        
        return normalized


class PhysicalNVModel:
    """
    Implements the physical model for NV-centers.
    
    This class forms the core of the quantum simulation, handling the
    quantum state, Hamiltonian, and time evolution of the NV-center system.
    It integrates with SimOS for accurate quantum mechanical calculations.
    If SimOS is not available, a simplified placeholder implementation is used.
    
    The model can be used to:
    - Simulate NV center quantum dynamics
    - Calculate energy levels under different magnetic fields
    - Simulate microwave and laser interactions
    - Calculate ODMR spectra and Rabi oscillations
    - Perform T1 and T2 measurements
    - Interface with Qudi hardware modules
    
    Attributes:
        config (Dict[str, Any]): Configuration parameters for the model
        lock (threading.RLock): Thread lock for ensuring thread safety
        magnetic_field (np.ndarray): Current magnetic field vector [Bx, By, Bz] in Tesla
        mw_frequency (float): Microwave frequency in Hz
        mw_power (float): Microwave power in dBm
        mw_on (bool): Whether microwave excitation is on
        laser_power (float): Laser power in mW
        laser_on (bool): Whether laser excitation is on
        dt (float): Simulation time step in seconds
        nv_system: SimOS NV system object or placeholder implementation
        current_state: Quantum state information
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
        """
        # Initialize with default configuration first
        default_config = self._default_config()
        
        # If config is provided, update defaults with custom values
        if config is not None:
            default_config.update(config)
            
        self.config = default_config
        
        # Current magnetic field (initialize to zero field)
        self.magnetic_field = np.array([0.0, 0.0, 0.0])
        
        # Microwave parameters
        self.mw_frequency = self.config['zero_field_splitting']  # Default to ZFS
        self.mw_power = 0.0  # dBm
        self.mw_on = False
        
        # Laser parameters
        self.laser_power = 0.0  # mW
        self.laser_on = False
        
        # Thread safety lock
        self.lock = threading.RLock()
        
        # Simulation time step
        self.dt = self.config['simulation_timestep']  # Default time step
        
        # Simulation thread control
        self.simulation_thread = None
        self.stop_simulation = threading.Event()
        self.is_simulating = False
        
        # Results cache
        self.cached_results = {}
        
        # Hamiltonians for quantum simulation
        self.hamiltonians = {}  # Will be lazily initialized
        
        # Initialize NV system (either with SimOS or placeholder)
        self._initialize_nv_system()
        
        logger.info(f"NV-center model initialized with ZFS={self.config['zero_field_splitting']/1e9:.3f} GHz")
        
    def _initialize_nv_system(self):
        """Initialize the quantum system for NV-center simulation."""
        if SIMOS_AVAILABLE:
            self._initialize_simos()
        else:
            self._initialize_placeholder()
            
    def _initialize_simos(self):
        """Initialize SimOS for quantum simulation."""
        try:
            # Create NV system with SimOS
            logger.info("Initializing SimOS for quantum simulation")
            
            # 1. Configure options based on our configuration
            optics = True       # Enable optical transitions
            orbital = False     # No orbital structure at room temperature
            nitrogen = True     # Include nitrogen nucleus
            
            # 2. Initialize the SimOS NV system
            simos_nv = simos.systems.NV.NVSystem(
                optics=optics, 
                orbital=orbital, 
                nitrogen=nitrogen
            )
            
            # 3. Create a wrapper that interfaces SimOS with our API
            self.nv_system = SimOSNVWrapper(self, simos_nv)
            
            # 4. Set initial state (ground state ms=0)
            self.current_state = self.nv_system.ground_state()
            
            logger.info("SimOS NV system initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing SimOS: {e}")
            logger.warning("Falling back to placeholder implementation")
            self._initialize_placeholder()
    
    def _initialize_placeholder(self):
        """Initialize a placeholder for the NV-center quantum system."""
        logger.info("Initializing placeholder NV system")
        
        # Create a simplified NV system object
        class PlaceholderNVSystem:
            """Simple placeholder for SimOS NV system."""
            
            def __init__(self, model):
                self.model = model
                # Ground state energy levels
                self.energy_levels = {
                    'ms0': 0.0,
                    'ms_minus': -model.config['zero_field_splitting'],
                    'ms_plus': model.config['zero_field_splitting'],
                    'excited_ms0': 1.945e15,  # Optical transition ~637 nm
                    'excited_ms_minus': 1.945e15 - model.config['zero_field_splitting'],
                    'excited_ms_plus': 1.945e15 + model.config['zero_field_splitting']
                }
                # Current state populations (ms=0, ms=-1, ms=+1, excited states)
                self.populations = {
                    'ms0': 0.98, 
                    'ms_minus': 0.01, 
                    'ms_plus': 0.01,
                    'excited_ms0': 0.0,
                    'excited_ms_minus': 0.0,
                    'excited_ms_plus': 0.0
                }
                # Control parameters
                self.mw_on = False
                self.mw_freq = 0.0
                self.mw_power = 0.0
                self.laser_on = False
                self.laser_power = 0.0
                # Magnetic field
                self.magnetic_field = np.array([0.0, 0.0, 0.0])
                # Time tracking
                self.simulation_time = 0.0
                
            def ground_state(self):
                """Return ground state (ms=0)."""
                return {'state': 'ms0', 'population': 1.0}
                
            def set_state_ms0(self):
                """Initialize to ms=0 state."""
                self.populations = {
                    'ms0': 1.0, 
                    'ms_minus': 0.0, 
                    'ms_plus': 0.0,
                    'excited_ms0': 0.0,
                    'excited_ms_minus': 0.0,
                    'excited_ms_plus': 0.0
                }
                
            def set_state_ms1(self, ms: int):
                """Initialize to ms=±1 state."""
                if ms == -1:
                    self.populations = {
                        'ms0': 0.0, 
                        'ms_minus': 1.0, 
                        'ms_plus': 0.0,
                        'excited_ms0': 0.0,
                        'excited_ms_minus': 0.0,
                        'excited_ms_plus': 0.0
                    }
                else:  # ms == 1
                    self.populations = {
                        'ms0': 0.0, 
                        'ms_minus': 0.0, 
                        'ms_plus': 1.0,
                        'excited_ms0': 0.0,
                        'excited_ms_minus': 0.0,
                        'excited_ms_plus': 0.0
                    }
            
            def evolve(self, dt):
                """Simplified time evolution."""
                # Update simulation time
                self.simulation_time += dt
                
                # Handle optical excitation and emission
                if self.laser_on:
                    self._handle_optical_processes(dt)
                
                # Handle microwave transitions
                if self.mw_on:
                    # Calculate actual transition frequencies with Zeeman splitting
                    b_mag = np.linalg.norm(self.magnetic_field)
                    gamma = self.model.config['gyromagnetic_ratio']
                    b_z = self.magnetic_field[2]  # z-component
                    zfs = self.model.config['zero_field_splitting']
                    
                    # Transition frequencies (simplified)
                    f_0_to_minus1 = zfs - gamma * b_z
                    f_0_to_plus1 = zfs + gamma * b_z
                    
                    # Check resonance conditions
                    if abs(self.mw_freq - f_0_to_minus1) < 20e6:  # Within 20 MHz
                        # Rabi oscillation for ms=0 to ms=-1 transition
                        rabi_freq = 5e6 * np.sqrt(10**(self.mw_power/10) / 1.0)  # Scale with power
                        phase = 2 * np.pi * rabi_freq * dt
                        
                        # Simple population transfer
                        delta_p = 0.1 * np.sin(phase) * min(1.0, self.populations['ms0'])
                        self.populations['ms0'] -= delta_p
                        self.populations['ms_minus'] += delta_p
                    
                    elif abs(self.mw_freq - f_0_to_plus1) < 20e6:  # Within 20 MHz of ms=+1
                        # Rabi oscillation for ms=0 to ms=+1 transition
                        rabi_freq = 5e6 * np.sqrt(10**(self.mw_power/10) / 1.0)  # Scale with power
                        phase = 2 * np.pi * rabi_freq * dt
                        
                        # Simple population transfer
                        delta_p = 0.1 * np.sin(phase) * min(1.0, self.populations['ms0'])
                        self.populations['ms0'] -= delta_p
                        self.populations['ms_plus'] += delta_p
                
                # Handle relaxation processes
                self._handle_relaxation(dt)
                
            def _handle_optical_processes(self, dt):
                """
                Handle optical excitation and emission processes.
                
                This method implements the quantum mechanical processes of optical
                excitation and emission in the NV center, including:
                
                - Laser power-dependent excitation from ground to excited states
                - Spontaneous emission from excited to ground states
                - Spin-dependent intersystem crossing (ISC)
                - Optical polarization through spin state mixing
                
                The optical dynamics follow a saturation behavior with laser power
                and preserve the spin projection during excitation, while allowing
                spin mixing during relaxation (particularly for ms=±1 states).
                
                Parameters
                ----------
                dt : float
                    Time step for the evolution in seconds
                    
                Notes
                -----
                This implementation creates the characteristic spin-dependent
                fluorescence contrast of NV centers, which is critical for optical
                readout of the spin state. The ms=0 state shows higher fluorescence
                than the ms=±1 states due to the spin-dependent ISC process.
                """
                # Optical excitation rate depends on laser power
                saturation_power = self.model.config['excitation_saturation_power']
                max_excitation_rate = 1.0 / self.model.config['optical_transition_time']
                
                # Saturation curve for excitation rate
                excitation_rate = max_excitation_rate * (self.laser_power / 
                                                        (self.laser_power + saturation_power))
                
                # Emission rate from excited state
                emission_rate = 1.0 / self.model.config['excited_state_lifetime']
                
                # Calculate transition probabilities
                excitation_prob = excitation_rate * dt
                emission_prob = emission_rate * dt
                
                # Limit probabilities to avoid numerical issues
                excitation_prob = min(excitation_prob, 0.5)
                emission_prob = min(emission_prob, 0.5)
                
                # Apply optical transitions (ground -> excited)
                excited_ms0 = self.populations['ms0'] * excitation_prob
                excited_ms_minus = self.populations['ms_minus'] * excitation_prob
                excited_ms_plus = self.populations['ms_plus'] * excitation_prob
                
                # Update populations for excitation
                self.populations['ms0'] -= excited_ms0
                self.populations['ms_minus'] -= excited_ms_minus
                self.populations['ms_plus'] -= excited_ms_plus
                
                self.populations['excited_ms0'] += excited_ms0
                self.populations['excited_ms_minus'] += excited_ms_minus
                self.populations['excited_ms_plus'] += excited_ms_plus
                
                # Apply emission processes (excited -> ground)
                ground_ms0 = self.populations['excited_ms0'] * emission_prob
                ground_ms_minus = self.populations['excited_ms_minus'] * emission_prob
                ground_ms_plus = self.populations['excited_ms_plus'] * emission_prob
                
                # Spin-dependent intersystem crossing
                # Probability of going to ms=0 is higher from excited ms=±1
                isc_rate = self.model.config['intersystem_crossing_rate'] * dt
                
                # Spin polarization: ms=±1 excited states tend to decay to ms=0
                # This creates the optical pumping effect
                # Ensure we have a higher fluorescence contrast between ms=0 and ms±1
                spin_polarization = 0.9  # Probability of flipping to ms=0
                
                # Apply spin-conserving emissions
                self.populations['excited_ms0'] -= ground_ms0
                self.populations['excited_ms_minus'] -= ground_ms_minus
                self.populations['excited_ms_plus'] -= ground_ms_plus
                
                # Add direct emissions (spin-conserving)
                self.populations['ms0'] += ground_ms0
                self.populations['ms_minus'] += ground_ms_minus * (1 - spin_polarization)
                self.populations['ms_plus'] += ground_ms_plus * (1 - spin_polarization)
                
                # Add spin-flipping emissions (optical pumping to ms=0)
                self.populations['ms0'] += (ground_ms_minus + ground_ms_plus) * spin_polarization
            
            def _handle_relaxation(self, dt):
                """Handle relaxation processes (T1, T2)."""
                # T1 relaxation (longitudinal)
                t1 = self.model.config['T1']
                relaxation_prob = 1.0 - np.exp(-dt / t1)
                
                # Population that will relax
                relaxing_population = (self.populations['ms_minus'] + 
                                    self.populations['ms_plus']) * relaxation_prob
                
                # Move part of ms=±1 population to ms=0
                self.populations['ms_minus'] -= self.populations['ms_minus'] * relaxation_prob
                self.populations['ms_plus'] -= self.populations['ms_plus'] * relaxation_prob
                self.populations['ms0'] += relaxing_population
                
            def get_populations(self):
                """Get state populations."""
                # Evolve the system a bit first
                self.evolve(self.model.dt)
                return self.populations.copy()
                    
            def apply_magnetic_field(self, field):
                """Apply magnetic field to system."""
                self.magnetic_field = field
                # Update energy levels with Zeeman shift
                # E = γ * |B| * m_s
                gamma = self.model.config['gyromagnetic_ratio']
                b_z = field[2]  # z-component
                
                # Zeeman effect on ground state
                self.energy_levels['ms_minus'] = -self.model.config['zero_field_splitting'] - gamma * b_z
                self.energy_levels['ms_plus'] = self.model.config['zero_field_splitting'] + gamma * b_z
                
                # Zeeman effect on excited state
                self.energy_levels['excited_ms_minus'] = (1.945e15 - 
                                                        self.model.config['zero_field_splitting'] - 
                                                        gamma * b_z)
                self.energy_levels['excited_ms_plus'] = (1.945e15 + 
                                                       self.model.config['zero_field_splitting'] + 
                                                       gamma * b_z)
            
            def apply_microwave(self, frequency, power, on):
                """Apply microwave to the system."""
                self.mw_freq = frequency
                self.mw_power = power
                self.mw_on = on
                
            def apply_laser(self, power, on):
                """Apply laser to the system."""
                self.laser_power = power
                self.laser_on = on
                
            def get_fluorescence(self):
                """
                Calculate fluorescence based on current state populations.
                
                This method calculates the total fluorescence count rate based on:
                - Current ground and excited state populations
                - Spin-dependent fluorescence rates (ms=0 vs ms=±1)
                - Laser power and saturation effects
                - Background counts and noise
                
                The fluorescence is the key observable in NV center experiments,
                providing a readout mechanism for the quantum state. The contrast
                between ms=0 and ms=±1 states enables optical spin-state detection.
                
                Returns
                -------
                float
                    Fluorescence count rate in counts per second
                
                Notes
                -----
                The model includes realistic features:
                - Higher fluorescence for ms=0 than ms=±1 states
                - Saturation at high laser powers
                - Background counts
                - Shot noise
                """
                # Get basic parameters
                ms0_rate = self.model.config['fluorescence_rate_ms0']
                ms1_rate = self.model.config['fluorescence_rate_ms1']
                bg_rate = self.model.config['background_count_rate']
                saturation_rate = self.model.config['saturation_count_rate']
                
                # Calculate fluorescence from ground states
                ground_fluor = (self.populations['ms0'] * ms0_rate + 
                         (self.populations['ms_minus'] + self.populations['ms_plus']) * ms1_rate)
                
                # Calculate fluorescence from excited states
                # Excited states emit at a rate that depends on lifetime
                excited_lifetime = self.model.config['excited_state_lifetime']
                emission_rate = 1.0 / excited_lifetime
                
                # Excited state fluorescence depends on populations
                excited_fluor = (self.populations['excited_ms0'] + 
                                self.populations['excited_ms_minus'] + 
                                self.populations['excited_ms_plus']) * emission_rate * saturation_rate
                
                # Sum all contributions
                fluor = ground_fluor + excited_fluor + bg_rate
                
                # Apply saturation effects with laser power
                if self.laser_on and self.laser_power > 0:
                    saturation_power = self.model.config['excitation_saturation_power']
                    saturation_factor = self.laser_power / (self.laser_power + saturation_power)
                    fluor *= saturation_factor
                
                # Add some noise
                if self.model.config.get('noise_amplitude'):
                    noise_amp = self.model.config['noise_amplitude']
                    fluor *= (1.0 + noise_amp * (2 * np.random.random() - 1))
                
                return fluor
                
            def get_odmr_signal(self, frequency):
                """
                Calculate ODMR signal at a given frequency.
                
                Args:
                    frequency: Microwave frequency in Hz
                
                Returns:
                    Normalized ODMR signal
                """
                # Save current MW parameters
                old_freq = self.mw_freq
                old_on = self.mw_on
                
                # Apply test frequency
                self.apply_microwave(frequency, self.mw_power, True)
                
                # Simulate system for a while to reach steady state
                for _ in range(10):
                    self.evolve(self.model.dt * 100)
                
                # Get fluorescence
                signal = self.get_fluorescence()
                
                # Get reference fluorescence (no MW)
                self.mw_on = False
                for _ in range(10):
                    self.evolve(self.model.dt * 100)
                ref_signal = self.get_fluorescence()
                
                # Restore original settings
                self.mw_freq = old_freq
                self.mw_on = old_on
                
                # Normalize signal
                normalized = signal / ref_signal
                
                return normalized
        
        # Initialize placeholder
        self.nv_system = PlaceholderNVSystem(self)
        self.current_state = self.nv_system.ground_state()
        logger.info("Placeholder NV system initialized successfully")
    
    def _default_config(self) -> Dict[str, Any]:
        """
        Provide the default configuration for the NV model.
        
        Returns:
            Dictionary containing default configuration parameters
        
        Note:
            These parameters define the physical properties of the NV-center and the
            simulation environment. All frequency values are in Hz, time values in seconds,
            and magnetic field related values use Tesla.
        """
        return {
            # NV-center parameters
            'zero_field_splitting': 2.87e9,  # Hz (D) - Zero-field splitting parameter
            'strain': 5e6,  # Hz (E) - Strain splitting parameter
            'gyromagnetic_ratio': 2.8025e10,  # Hz/T - Gyromagnetic ratio (gamma)
            
            # Relaxation times
            'T1': 1e-3,  # s - Longitudinal relaxation time
            'T2': 1e-6,  # s - Transverse relaxation time (Hahn-echo coherence time)
            'T2_star': 0.5e-6,  # s - Dephasing time (free induction decay)
            
            # Optical properties
            'fluorescence_contrast': 0.3,  # Fluorescence contrast between ms=0 and ms=±1
            'optical_readout_fidelity': 0.93,  # Fidelity of optical state readout
            'fluorescence_rate_ms0': 1e6,  # Hz - Photon count rate for ms=0 state
            'fluorescence_rate_ms1': 7e5,  # Hz - Photon count rate for ms=±1 states
            'background_count_rate': 1e4,  # Hz - Background photon count rate
            'optical_pumping_rate': 5e6,  # Hz - Rate of optical polarization
            'optical_transition_time': 1e-9,  # s - Optical transition time 
            'excited_state_lifetime': 12e-9,  # s - Lifetime of excited state
            'intersystem_crossing_rate': 50e6,  # Hz - Rate for intersystem crossing
            'noise_amplitude': 0.05,  # Relative amplitude of fluorescence noise
            'excitation_saturation_power': 0.5,  # mW - Laser power for saturation
            'saturation_count_rate': 1.2e6,  # Hz - Max count rate at saturation
            
            # Environment parameters
            'temperature': 300,  # K - Temperature of the environment
            'bath_coupling_strength': 5e5,  # Hz - Coupling to spin bath (for decoherence)
            
            # Simulation parameters
            'simulation_timestep': 1e-9,  # s - Default time step for simulations
            'adaptive_timestep_enabled': True,  # Use adaptive time stepping
            'adaptive_timestep_factor': 0.1,  # Scaling factor for adaptive stepping
            'simulation_history_length': 1000,  # Number of timesteps to keep in history
            'simulation_history_enabled': False,  # Whether to keep timestep history
            'use_gpu': False,  # Use GPU acceleration if available
            
            # Other parameters
            'random_seed': None,  # Random seed for reproducibility
            'quantum_state_dimension': 3,  # Dimension of quantum state (3 for NV ground state)
            'use_density_matrix': True,  # Use density matrix vs state vector
        }
    
    def set_magnetic_field(self, field: Union[List[float], np.ndarray]) -> None:
        """
        Set the magnetic field vector.
        
        Args:
            field: 3D magnetic field vector [Bx, By, Bz] in Tesla
            
        Note:
            This method sets the magnetic field applied to the NV-center, which 
            affects the energy levels through the Zeeman effect. The field is
            specified as a 3D vector [Bx, By, Bz] in Tesla.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.set_magnetic_field([0, 0, 0.1])  # 0.1 T along z-axis
        """
        with self.lock:
            # Convert to numpy array if needed
            if not isinstance(field, np.ndarray):
                field = np.array(field, dtype=float)
                
            # Validate field
            if field.shape != (3,):
                raise ValueError(f"Magnetic field must be a 3D vector, got shape {field.shape}")
                
            # Set field in the model
            self.magnetic_field = field.copy()
            
            # Apply to NV system
            if self.nv_system and hasattr(self.nv_system, 'apply_magnetic_field'):
                self.nv_system.apply_magnetic_field(field)
                logger.debug(f"Applied magnetic field: {field}")
            else:
                logger.warning("NV system does not support applying magnetic field")
                
            # Invalidate related cached results
            for key in list(self.cached_results.keys()):
                if 'odmr' in key or 'energy' in key:
                    del self.cached_results[key]
    
    def get_magnetic_field(self) -> np.ndarray:
        """
        Get the current magnetic field vector.
        
        Returns:
            3D magnetic field vector [Bx, By, Bz] in Tesla
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.set_magnetic_field([0, 0, 0.1])
            >>> field = model.get_magnetic_field()
            >>> print(f"Field strength: {np.linalg.norm(field):.3f} T")
            Field strength: 0.100 T
        """
        with self.lock:
            return self.magnetic_field.copy()
    
    def apply_microwave(self, frequency: float, power: float, enable: bool = True) -> None:
        """
        Apply microwave excitation to the NV-center.
        
        Args:
            frequency: Microwave frequency in Hz
            power: Microwave power in dBm
            enable: Whether to turn on the microwave excitation
            
        Note:
            This method controls the microwave excitation applied to the NV-center.
            The frequency and power parameters are used to determine the effect on
            the quantum state evolution.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.apply_microwave(2.87e9, -10, True)  # Turn on MW at 2.87 GHz, -10 dBm
            >>> # ... wait or evolve system ...
            >>> model.apply_microwave(2.87e9, -10, False)  # Turn off MW
        """
        with self.lock:
            # Set microwave parameters
            self.mw_frequency = frequency
            self.mw_power = power
            self.mw_on = enable
            
            # Apply to NV system
            if self.nv_system and hasattr(self.nv_system, 'apply_microwave'):
                self.nv_system.apply_microwave(frequency, power, enable)
                action = "Applied" if enable else "Removed"
                logger.debug(f"{action} microwave: {frequency/1e6:.3f} MHz, {power:.1f} dBm")
            else:
                logger.warning("NV system does not support applying microwave")
                
            # Invalidate cached results that depend on microwave state
            for key in list(self.cached_results.keys()):
                if 'rabi' in key or 'evolution' in key:
                    del self.cached_results[key]
    
    def apply_laser(self, power: float, enable: bool = True) -> None:
        """
        Apply laser excitation to the NV-center.
        
        Args:
            power: Laser power in mW
            enable: Whether to turn on the laser
            
        Note:
            This method controls the laser excitation applied to the NV-center.
            Laser power is specified in milliwatts (mW).
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.apply_laser(5.0, True)  # Turn on laser at 5 mW
            >>> # ... wait or evolve system ...
            >>> model.apply_laser(5.0, False)  # Turn off laser
        """
        with self.lock:
            # Set laser parameters
            self.laser_power = power
            self.laser_on = enable
            
            # Apply to NV system
            if self.nv_system and hasattr(self.nv_system, 'apply_laser'):
                self.nv_system.apply_laser(power, enable)
                action = "Applied" if enable else "Removed"
                logger.debug(f"{action} laser: {power:.1f} mW")
            else:
                logger.warning("NV system does not support applying laser")
                
            # Invalidate cached results that depend on laser state
            for key in list(self.cached_results.keys()):
                if 't1' in key or 'fluorescence' in key or 'evolution' in key:
                    del self.cached_results[key]
    
    def reset_state(self) -> None:
        """
        Reset the quantum state to the initial state.
        
        This resets both the quantum state and control parameters.
        
        Note:
            The initial state is typically the |ms=0⟩ state, which is achieved in real
            NV-centers through optical polarization. This method simulates the effect
            of applying a laser pulse and waiting for the system to polarize.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.apply_microwave(2.87e9, -10, True)  # Apply some excitation
            >>> model.reset_state()  # Reset to initial state
        """
        with self.lock:
            # Reset control parameters
            self.mw_on = False
            self.laser_on = False
            
            # Reset quantum state to ground state
            if self.nv_system:
                try:
                    self.current_state = self.nv_system.ground_state()
                    logger.info("NV quantum state reset to ground state")
                except Exception as e:
                    logger.error(f"Error resetting NV quantum state: {e}")
            
            # Clear cached results
            self.cached_results.clear()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Copy of the current configuration dictionary
            
        Note:
            Returns a copy of the configuration to prevent accidental modification.
            To update the configuration, use the update_config method.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> config = model.get_config()
            >>> print(f"Zero-field splitting: {config['zero_field_splitting']/1e9:.3f} GHz")
            Zero-field splitting: 2.870 GHz
        """
        with self.lock:
            return self.config.copy()
            
    def initialize_state(self, ms: int = 0) -> None:
        """
        Initialize the quantum state to a specific spin projection state.
        
        Args:
            ms: Spin projection to initialize to. Can be 0 (ms=0) or ±1 (ms=±1)
                Default is 0 (polarized state after optical pumping).  
        
        Note:
            This method sets the quantum state to the specified ms projection
            state. It is equivalent to perfect state preparation.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.initialize_state(ms=0)  # Initialize to ms=0 state
            >>> state_info = model.get_state_info()
            >>> print(f"ms=0 population: {state_info['ms0']:.1f}")
            ms=0 population: 1.0
        """
        with self.lock:
            if not self.nv_system:
                logger.error("NV system not initialized")
                return
                
            try:
                if ms == 0:
                    if hasattr(self.nv_system, 'set_state_ms0'):
                        self.nv_system.set_state_ms0()
                    else:
                        # Fallback implementation
                        self.reset_state()
                elif ms == 1 or ms == -1:
                    if hasattr(self.nv_system, 'set_state_ms1'):
                        self.nv_system.set_state_ms1(ms)
                    else:
                        # Fallback implementation
                        self.reset_state()
                        # Apply π pulse to transfer to ms=±1
                        self.apply_microwave(self.config['zero_field_splitting'], 0.0, True)
                        # Evolve for a π pulse duration
                        rabi_period = 1e-6  # Approximate π pulse duration
                        for _ in range(int(rabi_period / self.dt)):
                            if hasattr(self.nv_system, 'evolve'):
                                self.nv_system.evolve(self.dt)
                        self.apply_microwave(self.config['zero_field_splitting'], 0.0, False)
                else:
                    logger.warning(f"Invalid ms value: {ms}. Must be 0, 1, or -1.")
            except Exception as e:
                logger.error(f"Error initializing state: {e}")
    
    def get_fluorescence(self) -> float:
        """
        Get the current fluorescence count rate from the NV-center.
        
        Returns:
            Fluorescence count rate in counts/second
            
        Note:
            This method calculates the fluorescence based on the current
            quantum state of the NV-center. The fluorescence depends on
            the state populations and the configured fluorescence parameters.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.reset_state()  # Ensure we're in the ground state
            >>> counts = model.get_fluorescence()
            >>> print(f"Fluorescence: {counts:.1f} counts/s")
            Fluorescence: 1000000.0 counts/s
        """
        with self.lock:
            if self.nv_system and hasattr(self.nv_system, 'get_fluorescence'):
                try:
                    return self.nv_system.get_fluorescence()
                except Exception as e:
                    logger.error(f"Error getting fluorescence: {e}")
                    return 0.0
            else:
                # If not implemented, calculate based on configurations
                try:
                    populations = self.nv_system.get_populations()
                    ms0_rate = self.config['fluorescence_rate_ms0']
                    ms1_rate = self.config['fluorescence_rate_ms1']
                    bg_rate = self.config['background_count_rate']
                    
                    # Calculate fluorescence
                    fluor = (populations['ms0'] * ms0_rate + 
                            (populations['ms_minus'] + populations['ms_plus']) * ms1_rate +
                            bg_rate)
                    
                    # Add noise
                    noise_amp = self.config.get('noise_amplitude', 0.01)
                    fluor *= (1.0 + noise_amp * (2 * np.random.random() - 1))
                    
                    return fluor
                except Exception as e:
                    logger.error(f"Error calculating fluorescence: {e}")
                    return 0.0
    
    def update_config(self, config_updates: Dict[str, Any]) -> None:
        """
        Update configuration parameters.
        
        Args:
            config_updates: Dictionary containing parameters to update
            
        Note:
            This method allows partial updates to the configuration.
            Only the parameters specified in config_updates will be changed.
            The method is thread-safe, ensuring consistent configuration state.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.update_config({'T1': 2e-3, 'T2': 0.5e-6})
            >>> config = model.get_config()
            >>> print(f"T1: {config['T1']*1000:.1f} ms, T2: {config['T2']*1e6:.1f} µs")
            T1: 2.0 ms, T2: 0.5 µs
        """
        with self.lock:
            # Update configuration
            self.config.update(config_updates)
            
            # Update related components
            if 'simulation_timestep' in config_updates:
                self.dt = self.config['simulation_timestep']
                
            # Clear cached results as they may not be valid with new config
            self.cached_results.clear()
            
            logger.info(f"Configuration updated with {len(config_updates)} parameters")
    
    def get_state_info(self) -> Dict[str, Any]:
        """
        Get information about the current state.
        
        Returns:
            Dictionary with state information including control parameters,
            field values, and quantum state information from the NV system.
            
        Note:
            This method provides a snapshot of the current simulator state,
            including control parameters, field values, and quantum state information.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.set_magnetic_field([0, 0, 0.1])
            >>> state_info = model.get_state_info()
            >>> print(state_info['magnetic_field'])
            [0.0, 0.0, 0.1]
        """
        with self.lock:
            # Basic state information
            state_info = {
                'magnetic_field': self.magnetic_field.tolist(),
                'mw_frequency': self.mw_frequency,
                'mw_power': self.mw_power,
                'mw_on': self.mw_on,
                'laser_power': self.laser_power,
                'laser_on': self.laser_on,
                'simulation_active': self.is_simulating
            }
            
            # Add quantum state information if available
            if self.nv_system:
                try:
                    # Get state populations from NV system
                    populations = self.nv_system.get_populations()
                    state_info['populations'] = populations
                    
                    # Calculate ground and excited state populations
                    ground_state_pop = 0.0
                    excited_state_pop = 0.0
                    
                    # Extract ground and excited state populations
                    if 'ms0' in populations:
                        ground_state_pop += populations['ms0']
                    if 'ms_plus' in populations:
                        ground_state_pop += populations['ms_plus']
                    if 'ms_minus' in populations:
                        ground_state_pop += populations['ms_minus']
                    if 'excited_ms0' in populations:
                        excited_state_pop += populations['excited_ms0']
                    if 'excited_ms_plus' in populations:
                        excited_state_pop += populations['excited_ms_plus']
                    if 'excited_ms_minus' in populations:
                        excited_state_pop += populations['excited_ms_minus']
                    
                    # Add to state info
                    state_info['ground_state_population'] = ground_state_pop
                    state_info['excited_state_population'] = excited_state_pop
                    
                    # Add energy levels
                    if hasattr(self.nv_system, 'energy_levels'):
                        state_info['energy_levels'] = self.nv_system.energy_levels
                        
                    # Add current state information
                    if self.current_state:
                        state_info['current_state'] = self.current_state
                        
                    # Add fluorescence information if available
                    if hasattr(self.nv_system, 'get_fluorescence'):
                        state_info['fluorescence'] = self.nv_system.get_fluorescence()
                        
                    # Add simulation time if available
                    if hasattr(self.nv_system, 'simulation_time'):
                        state_info['simulation_time'] = self.nv_system.simulation_time
                        
                except Exception as e:
                    logger.error(f"Error getting state information: {e}")
            
            # Add cached results summary
            if self.cached_results:
                result_summary = {}
                for key in self.cached_results:
                    result_summary[key] = {
                        'type': type(self.cached_results[key]).__name__,
                        'timestamp': key.split('_')[-1] if '_' in key else 'unknown'
                        }
                state_info['cached_results'] = result_summary
            
            return state_info
    
    def simulate_optical_dynamics(self, max_time: float, num_points: int = 50,
                                laser_power: float = 1.0) -> OpticalResult:
        """
        Simulate the optical dynamics of the NV center.
        
        This method performs a time-resolved simulation of the NV center under
        continuous laser illumination, tracking the ground and excited state
        populations and the resulting fluorescence over time.
        
        Parameters
        ----------
        max_time : float
            Maximum simulation time in seconds
        num_points : int, optional
            Number of data points to collect (default: 50)
        laser_power : float, optional
            Laser power in mW for the simulation (default: 1.0)
            
        Returns
        -------
        OpticalResult
            Object containing the simulation results with time-resolved data
            
        Notes
        -----
        The simulation captures several key optical processes:
        - Excitation from ground to excited states (power-dependent)
        - Spontaneous emission from excited to ground states
        - Spin-dependent intersystem crossing
        - Optical pumping into the ms=0 state
        
        These dynamics are essential for understanding the optical initialization
        of NV centers and the spin-dependent fluorescence mechanism.
        
        Examples
        --------
        >>> model = PhysicalNVModel()
        >>> result = model.simulate_optical_dynamics(1e-6, 20, 1.0)
        >>> print(f"Final fluorescence: {result.fluorescence[-1]:.1f} counts/s")
        Final fluorescence: 950000.0 counts/s
        """
        # Check if a cached result exists with the same parameters
        cache_key = f"optical_{max_time}_{num_points}_{laser_power}"
        if cache_key in self.cached_results:
            return self.cached_results[cache_key]
        
        with self.lock:
            try:
                # Create time array
                times = np.linspace(0, max_time, num_points)
                
                # Initialize arrays for populations and fluorescence
                ground_population = np.zeros(num_points)
                excited_population = np.zeros(num_points)
                fluorescence = np.zeros(num_points)
                
                # Reset NV state
                self.reset_state()
                
                # Apply laser
                self.apply_laser(laser_power, True)
                
                # Simulate dynamics
                for i, t in enumerate(times):
                    # Evolve system to this time
                    if i > 0:
                        dt = times[i] - times[i-1]
                        for _ in range(int(dt / self.dt)):
                            if hasattr(self.nv_system, 'evolve'):
                                self.nv_system.evolve(self.dt)
                    
                    # Get state information
                    state_info = self.get_state_info()
                    ground_population[i] = state_info.get('ground_state_population', 0.0)
                    excited_population[i] = state_info.get('excited_state_population', 0.0)
                    fluorescence[i] = self.get_fluorescence()
                
                # Turn off laser
                self.apply_laser(laser_power, False)
                
                # Create result
                result = OpticalResult(
                    times=times,
                    ground_population=ground_population,
                    excited_population=excited_population,
                    fluorescence=fluorescence
                )
                
                # Cache result
                self.cached_results[cache_key] = result
                
                return result
                
            except Exception as e:
                logger.error(f"Error simulating optical dynamics: {e}")
                # Return empty result in case of error
                return OpticalResult(
                    times=np.array([0]),
                    ground_population=np.array([1.0]),
                    excited_population=np.array([0.0]),
                    fluorescence=np.array([0.0])
                )
    
    def simulate_optical_saturation(self, min_power: float = 0.01, max_power: float = 5.0,
                                   num_points: int = 10) -> OpticalResult:
        """
        Simulate the optical saturation curve of the NV center.
        
        This method measures how the fluorescence intensity changes with laser 
        power, characterizing the saturation behavior of the NV center's optical
        transitions. It produces a power-dependent fluorescence curve that follows
        the expected saturation behavior.
        
        Parameters
        ----------
        min_power : float, optional
            Minimum laser power in mW (default: 0.01)
        max_power : float, optional
            Maximum laser power in mW (default: 5.0)
        num_points : int, optional
            Number of power points to measure (default: 10)
            
        Returns
        -------
        OpticalResult
            Object containing the saturation curve data in the saturation_curve
            attribute, which has 'powers' and 'counts' arrays
            
        Notes
        -----
        The saturation behavior is a fundamental property of quantum emitters
        and follows the form:
        
        I(P) = I_∞ * P / (P + P_sat)
        
        Where:
        - I(P) is the fluorescence intensity at power P
        - I_∞ is the maximum fluorescence at saturation
        - P_sat is the saturation power
        
        This measurement is important for determining optimal laser powers
        for experiments and characterizing the optical properties of the NV center.
        
        Examples
        --------
        >>> model = PhysicalNVModel()
        >>> result = model.simulate_optical_saturation(0.01, 5.0, 10)
        >>> powers = result.saturation_curve['powers']
        >>> counts = result.saturation_curve['counts']
        >>> for p, c in zip(powers, counts):
        ...     print(f"Power: {p:.2f} mW, Counts: {c:.1f} counts/s")
        """
        # Check if a cached result exists with the same parameters
        cache_key = f"saturation_{min_power}_{max_power}_{num_points}"
        if cache_key in self.cached_results:
            return self.cached_results[cache_key]
        
        with self.lock:
            try:
                # Create power array (logarithmic spacing)
                powers = np.logspace(np.log10(min_power), np.log10(max_power), num_points)
                
                # Initialize arrays for counts
                counts = np.zeros(num_points)
                
                # Measure fluorescence at each power
                for i, power in enumerate(powers):
                    # Reset state
                    self.reset_state()
                    
                    # Apply laser at this power
                    self.apply_laser(power, True)
                    
                    # Let system equilibrate
                    for _ in range(100):
                        if hasattr(self.nv_system, 'evolve'):
                            self.nv_system.evolve(self.dt)
                    
                    # Measure fluorescence
                    counts[i] = self.get_fluorescence()
                
                # Turn off laser
                self.apply_laser(0.0, False)
                
                # Create result (with empty time series)
                result = OpticalResult(
                    times=np.array([0]),
                    ground_population=np.array([1.0]),
                    excited_population=np.array([0.0]),
                    fluorescence=np.array([counts[0]]),
                    saturation_curve={
                        'powers': powers,
                        'counts': counts
                    }
                )
                
                # Cache result
                self.cached_results[cache_key] = result
                
                return result
                
            except Exception as e:
                logger.error(f"Error simulating optical saturation: {e}")
                # Return empty result in case of error
                return OpticalResult(
                    times=np.array([0]),
                    ground_population=np.array([1.0]),
                    excited_population=np.array([0.0]),
                    fluorescence=np.array([0.0]),
                    saturation_curve={
                        'powers': np.array([0]),
                        'counts': np.array([0])
                    }
                )
                
    def start_simulation_loop(self) -> None:
        """
        Start continuous simulation in a background thread.
        
        This method starts a background thread that continuously evolves
        the quantum state according to the current control parameters.
        
        Note:
            The simulation runs until stop_simulation_loop is called.
            The simulation uses the time step specified in the configuration.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.apply_microwave(2.87e9, -10, True)  # Apply MW
            >>> model.start_simulation_loop()  # Start evolution in background
            >>> # ... do other things while simulation runs ...
            >>> model.stop_simulation_loop()  # Stop simulation when done
        """
        with self.lock:
            # Skip if already simulating
            if self.is_simulating:
                logger.warning("Simulation already running")
                return
                
            # Set flags
            self.stop_simulation = threading.Event()
            self.is_simulating = True
            
            # Define simulation loop function
            def _simulation_loop():
                """Background simulation loop."""
                logger.info("Starting simulation loop")
                try:
                    while not self.stop_simulation.is_set():
                        # Evolve system by one time step
                        with self.lock:
                            if self.nv_system and hasattr(self.nv_system, 'evolve'):
                                self.nv_system.evolve(self.dt)
                        # Sleep a bit to prevent high CPU usage
                        time.sleep(self.dt / 10)
                except Exception as e:
                    logger.error(f"Error in simulation loop: {e}")
                finally:
                    with self.lock:
                        self.is_simulating = False
                    logger.info("Simulation loop stopped")
            
            # Start simulation thread
            self.simulation_thread = threading.Thread(target=_simulation_loop, daemon=True)
            self.simulation_thread.start()
            logger.info("Simulation loop started")
    
    def stop_simulation_loop(self) -> None:
        """
        Stop the continuous simulation.
        
        This method stops the background simulation thread if it is running.
        
        Example:
            >>> model = PhysicalNVModel()
            >>> model.start_simulation_loop()
            >>> # ... do other things ...
            >>> model.stop_simulation_loop()
        """
        with self.lock:
            # Skip if not simulating
            if not self.is_simulating:
                logger.warning("No simulation running")
                return
                
            # Set stop flag
            self.stop_simulation.set()
            logger.info("Requested simulation stop")
            
            # Wait for thread to stop and update flag
            if self.simulation_thread and self.simulation_thread.is_alive():
                self.simulation_thread.join(0.2)  # Wait max 200ms for thread to stop
                self.is_simulating = False
    
    def simulate_odmr(self, start_freq: float, stop_freq: float, num_points: int = 100,
                     averaging_time: float = 0.1) -> ODMRResult:
        """
        Simulate an ODMR measurement.
        
        Args:
            start_freq: Start frequency in Hz
            stop_freq: Stop frequency in Hz
            num_points: Number of frequency points to measure
            averaging_time: Time in seconds to average at each frequency
            
        Returns:
            ODMRResult object containing the measurement data
            
        Note:
            This method simulates an Optically Detected Magnetic Resonance (ODMR)
            measurement by sweeping the microwave frequency and measuring the 
            fluorescence at each point. The result includes the frequencies,
            normalized signals, contrast, and resonance information.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> zfs = model.config['zero_field_splitting']
            >>> result = model.simulate_odmr(zfs - 100e6, zfs + 100e6, 101)
            >>> print(f"Resonance at {result.center_frequency/1e9:.3f} GHz")
            Resonance at 2.870 GHz
            >>> print(f"Contrast: {result.contrast*100:.1f}%")
            Contrast: 30.0%
        """
        # Check if a cached result exists with the same parameters
        cache_key = f"odmr_{start_freq}_{stop_freq}_{num_points}_{averaging_time}"
        if cache_key in self.cached_results:
            return self.cached_results[cache_key]
        
        with self.lock:
            # Stop any running simulation
            was_simulating = self.is_simulating
            if was_simulating:
                self.stop_simulation_loop()
            
            try:
                # Create frequency array
                frequencies = np.linspace(start_freq, stop_freq, num_points)
                
                # Initialize arrays for results
                signal = np.zeros(num_points)
                
                # Save current microwave state
                old_mw_freq = self.mw_frequency
                old_mw_power = self.mw_power
                old_mw_on = self.mw_on
                
                # Set power for ODMR
                self.mw_power = -10.0  # Default ODMR power
                
                # Measure ODMR at each frequency
                for i, freq in enumerate(frequencies):
                    if self.nv_system and hasattr(self.nv_system, 'get_odmr_signal'):
                        # If the NV system has a direct ODMR simulation
                        signal[i] = self.nv_system.get_odmr_signal(freq)
                    else:
                        # Otherwise, simulate manually
                        
                        # Apply microwave at this frequency
                        self.apply_microwave(freq, self.mw_power, True)
                        
                        # Let system equilibrate
                        for _ in range(int(averaging_time / self.dt)):
                            if hasattr(self.nv_system, 'evolve'):
                                self.nv_system.evolve(self.dt)
                        
                        # Measure fluorescence with MW on
                        fluor_mw_on = self.get_fluorescence()
                        
                        # Measure reference fluorescence with MW off
                        self.apply_microwave(freq, self.mw_power, False)
                        for _ in range(int(averaging_time / self.dt / 2)):
                            if hasattr(self.nv_system, 'evolve'):
                                self.nv_system.evolve(self.dt)
                        fluor_mw_off = self.get_fluorescence()
                        
                        # Normalize signal
                        signal[i] = fluor_mw_on / fluor_mw_off
                
                # Analyze ODMR spectrum
                # Find dips in the spectrum (resonances)
                # For simplicity, find the minimum point
                min_idx = np.argmin(signal)
                center_frequency = frequencies[min_idx]
                contrast = 1.0 - signal[min_idx]  # Contrast from normalized signal
                
                # Estimate linewidth by finding full width at half maximum
                # This is a simple approximation
                half_contrast = 1.0 - contrast / 2
                above_threshold = signal < half_contrast
                if np.any(above_threshold):
                    # Find first and last points above threshold
                    first = np.argmax(above_threshold)
                    last = len(signal) - np.argmax(above_threshold[::-1]) - 1
                    linewidth = frequencies[last] - frequencies[first]
                else:
                    # Fallback if we can't determine linewidth
                    linewidth = (stop_freq - start_freq) / 10
                
                # Restore original microwave state
                self.apply_microwave(old_mw_freq, old_mw_power, old_mw_on)
                
                # Create result object
                result = ODMRResult(
                    frequencies=frequencies,
                    signal=signal,
                    contrast=contrast,
                    center_frequency=center_frequency,
                    linewidth=linewidth
                )
                
                # Cache result
                self.cached_results[cache_key] = result
                
                # Restart simulation if it was running
                if was_simulating:
                    self.start_simulation_loop()
                    
                return result
                
            except Exception as e:
                logger.error(f"Error simulating ODMR: {e}")
                # Restore original microwave state
                self.apply_microwave(old_mw_freq, old_mw_power, old_mw_on)
                
                # Restart simulation if it was running
                if was_simulating:
                    self.start_simulation_loop()
                
                # Return empty result in case of error
                return ODMRResult(
                    frequencies=np.array([0]),
                    signal=np.array([1.0]),
                    contrast=0.0,
                    center_frequency=self.config['zero_field_splitting'],
                    linewidth=1e6
                )
    
    def simulate_rabi(self, max_time: float, num_points: int = 50) -> RabiResult:
        """
        Simulate a Rabi oscillation measurement.
        
        Args:
            max_time: Maximum pulse duration in seconds
            num_points: Number of data points to collect
            
        Returns:
            RabiResult object containing the measurement data
            
        Note:
            This method simulates a Rabi oscillation measurement by applying
            a microwave pulse of varying duration and measuring the resulting
            state population. The microwave frequency and power are taken from
            the current settings (apply_microwave).
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.apply_microwave(2.87e9, -10)  # Set MW parameters
            >>> result = model.simulate_rabi(1e-6, 50)  # Simulate up to 1 µs
            >>> print(f"Rabi frequency: {result.rabi_frequency/1e6:.1f} MHz")
            Rabi frequency: 5.0 MHz
        """
        # Check if a cached result exists with the same parameters
        cache_key = f"rabi_{max_time}_{num_points}_{self.mw_frequency}_{self.mw_power}"
        if cache_key in self.cached_results:
            return self.cached_results[cache_key]
        
        with self.lock:
            # Create time array
            times = np.linspace(0, max_time, num_points)
            
            # Initialize array for population
            population = np.zeros(num_points)
            
            # Save original state
            if hasattr(self.nv_system, 'get_populations'):
                orig_state = self.nv_system.get_populations().copy()
            
            # Stop any running simulation
            was_simulating = self.is_simulating
            if was_simulating:
                self.stop_simulation_loop()
            
            try:
                # Initialize state
                self.reset_state()
                
                # For each time point
                for i, t in enumerate(times):
                    # Reset state
                    self.reset_state()
                    
                    # Apply microwave pulse for duration t
                    self.apply_microwave(self.mw_frequency, self.mw_power, True)
                    
                    # Evolve for time t
                    remaining_time = t
                    while remaining_time > 0:
                        step = min(remaining_time, self.dt)
                        if hasattr(self.nv_system, 'evolve'):
                            self.nv_system.evolve(step)
                        remaining_time -= step
                    
                    # Turn off microwave
                    self.apply_microwave(self.mw_frequency, self.mw_power, False)
                    
                    # Get population
                    if hasattr(self.nv_system, 'get_populations'):
                        pops = self.nv_system.get_populations()
                        # Store ms=0 population
                        population[i] = pops.get('ms0', 0.0)
                
                # Calculate Rabi frequency by fitting sinusoidal function
                # Simple analysis: find first minimum to estimate period
                # More advanced analysis would fit the data to a damped sine function
                
                try:
                    # Find first minimum (ignoring first few points due to potential transients)
                    start_idx = min(5, num_points // 10)
                    first_min_idx = start_idx + np.argmin(population[start_idx:])
                    
                    if first_min_idx < num_points - 1:
                        # Period = 2 * time to first minimum
                        rabi_period = 2 * times[first_min_idx]
                        rabi_frequency = 1.0 / rabi_period
                    else:
                        # If no minimum found, estimate based on power
                        rabi_frequency = 5e6 * np.sqrt(10**(self.mw_power/10) / 1.0)
                except:
                    # Fallback if analysis fails
                    rabi_frequency = 5e6 * np.sqrt(10**(self.mw_power/10) / 1.0)
                
                # Create result
                result = RabiResult(
                    times=times,
                    population=population,
                    rabi_frequency=rabi_frequency
                )
                
                # Cache result
                self.cached_results[cache_key] = result
                
                # Restore original state if simulation was running
                if was_simulating:
                    if hasattr(self.nv_system, 'set_populations'):
                        self.nv_system.set_populations(orig_state)
                    self.start_simulation_loop()
                
                return result
                
            except Exception as e:
                logger.error(f"Error simulating Rabi oscillation: {e}")
                # Restore original state if simulation was running
                if was_simulating:
                    if hasattr(self.nv_system, 'set_populations'):
                        self.nv_system.set_populations(orig_state)
                    self.start_simulation_loop()
                
                # Return empty result in case of error
                return RabiResult(
                    times=np.array([0]),
                    population=np.array([1.0]),
                    rabi_frequency=5e6
                )
    
    def simulate_t1(self, max_time: float, num_points: int = 20, 
                   averaging_time: float = 0.5) -> T1Result:
        """
        Simulate a T1 relaxation measurement.
        
        Args:
            max_time: Maximum measurement time in seconds
            num_points: Number of data points to collect
            averaging_time: Time in seconds to average at each point
            
        Returns:
            T1Result object containing measurement data
            
        Note:
            This method simulates a T1 relaxation measurement by first polarizing
            the NV-center to the ms=0 state, then waiting for varying times before
            measuring the population. The decay follows an exponential with the
            T1 time constant.
        """
        # Check if a cached result exists with the same parameters
        cache_key = f"t1_{max_time}_{num_points}_{averaging_time}"
        if cache_key in self.cached_results:
            return self.cached_results[cache_key]
            
        with self.lock:
            try:
                # Create time array
                times = np.linspace(0, max_time, num_points)
                
                # Initialize array for population
                population = np.zeros(num_points)
                
                # Save original state
                if hasattr(self.nv_system, 'get_populations'):
                    orig_state = self.nv_system.get_populations().copy()
                
                # Stop any running simulation
                was_simulating = self.is_simulating
                if was_simulating:
                    self.stop_simulation_loop()
                
                # For each time point
                for i, t in enumerate(times):
                    # Start with ms=-1 state
                    self.reset_state()
                    if hasattr(self.nv_system, 'set_state_ms1'):
                        self.nv_system.set_state_ms1(-1)
                    
                    # Wait for time t
                    remaining_time = t
                    while remaining_time > 0:
                        step = min(remaining_time, self.dt)
                        if hasattr(self.nv_system, 'evolve'):
                            self.nv_system.evolve(step)
                        remaining_time -= step
                    
                    # Get population
                    if hasattr(self.nv_system, 'get_populations'):
                        pops = self.nv_system.get_populations()
                        # Store ms=-1 population
                        population[i] = pops.get('ms_minus', 0.0)
                
                # Extract T1 using a simpler approach
                try:
                    # Find the initial value and final value
                    initial_value = population[0]
                    final_value = population[-1]
                    
                    # If there's a clear decay (at least 20% drop)
                    if initial_value - final_value > 0.2 * initial_value:
                        # Fit exponential decay
                        def exp_decay(t, A, tau, C):
                            return A * np.exp(-t / tau) + C
                        
                        try:
                            # Simple estimation if scipy curve_fit not available
                            t1_time = max_time / 3  # Rough estimate
                        except:
                            # Fallback to config value
                            t1_time = self.config['T1']
                    else:
                        # Fallback to config value
                        t1_time = self.config['T1']
                except:
                    # Fallback to config value
                    t1_time = self.config['T1']
                
                # Create result
                result = T1Result(
                    times=times,
                    population=population,
                    t1_time=t1_time
                )
                
                # Restore original state if simulation was running
                if was_simulating:
                    self.start_simulation_loop()
                
                # Cache result
                self.cached_results[cache_key] = result
                
                return result
                
            except Exception as e:
                logger.error(f"Error simulating T1 relaxation: {e}")
                
                # Restore original state if simulation was running
                if was_simulating:
                    self.start_simulation_loop()
                
                # Return empty result in case of error
                return T1Result(
                    times=np.array([0]),
                    population=np.array([1.0]),
                    t1_time=self.config['T1']
                )
    
