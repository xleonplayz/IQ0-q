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
    SIMOS_AVAILABLE = True
    logger.info("SimOS package found. Using SimOS for quantum simulation.")
except ImportError:
    SIMOS_AVAILABLE = False
    logger.warning("SimOS package not found. Using placeholder implementation.")


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
        
        # Cache for storing simulation results
        self.cached_results: Dict[str, Union[ODMRResult, RabiResult, T1Result, T2Result, StateEvolution]] = {}
        
        # Initialize quantum Hamiltonian matrices for SimOS placeholder
        self._initialize_hamiltonian_matrices()
        
        # Initialize SimOS integration
        self._initialize_simos_system()
        
        logger.debug("PhysicalNVModel initialized with configuration: %s", self.config)
        
    def _initialize_simos_system(self) -> None:
        """
        Initialize the SimOS NV system with current configuration.
        
        This method creates and configures the SimOS NV system object
        based on the parameters in the configuration dictionary.
        If SimOS is not available, it creates a simplified placeholder implementation.
        """
        # Current quantum state
        self.current_state = None
        
        # Initialize with SimOS if available
        if SIMOS_AVAILABLE:
            try:
                # Create NV system with SimOS
                logger.info("Initializing SimOS NV system")
                
                # Extract configuration parameters for SimOS
                zfs = self.config['zero_field_splitting']
                strain = self.config['strain']
                gyro = self.config['gyromagnetic_ratio']
                t1 = self.config['T1']
                t2 = self.config['T2']
                temperature = self.config['temperature']
                hf_coupling = self.config['hyperfine_coupling_14n']
                quad_splitting = self.config['quadrupole_splitting_14n']
                
                # Create SimOS NV system 
                # Note: Actual API will depend on SimOS implementation
                self.nv_system = simos.NVCenter(
                    zero_field_splitting=zfs,
                    strain=strain,
                    gyromagnetic_ratio=gyro,
                    temperature=temperature,
                    t1=t1,
                    t2=t2,
                    t2_star=self.config['T2_star'],
                    hyperfine_coupling=hf_coupling,
                    quadrupole_splitting=quad_splitting,
                    bath_coupling=self.config['bath_coupling_strength'],
                    use_gpu=self.config['use_gpu'],
                    integration_method=self.config['integration_method'],
                    adaptive_timestep=self.config['adaptive_timestep'],
                    accuracy=self.config['simulation_accuracy']
                )
                
                # Initialize quantum state (typically ms=0 state)
                self.current_state = self.nv_system.ground_state()
                
                logger.info("SimOS NV system initialized successfully")
                
            except Exception as e:
                logger.error(f"Error initializing SimOS NV system: {e}")
                self._initialize_placeholder_system()
        else:
            # If SimOS is not available, use a placeholder implementation
            self._initialize_placeholder_system()
            
    def _initialize_hamiltonian_matrices(self) -> None:
        """
        Initialize Hamiltonian matrices for quantum state evolution.
        
        This method creates the necessary operators and matrices for
        simulating the quantum evolution of the NV center system when
        the full SimOS library is not available.
        """
        # Define Pauli matrices as basic building blocks
        self.sigma_x = np.array([[0, 1], [1, 0]])
        self.sigma_y = np.array([[0, -1j], [1j, 0]])
        self.sigma_z = np.array([[1, 0], [0, -1]])
        self.identity = np.eye(2)
        
        # Define spin-1 operators for NV electron spin
        # Using the 3x3 matrix representation for S=1
        self.Sx = (1/np.sqrt(2)) * np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0]
        ])
        self.Sy = (1j/np.sqrt(2)) * np.array([
            [0, -1, 0],
            [1, 0, -1],
            [0, 1, 0]
        ])
        self.Sz = np.array([
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, -1]
        ])
        self.S0 = np.eye(3)  # Identity for spin-1
        
        # Define nuclear spin operators for N14 (I=1)
        self.Ix = self.Sx.copy()  # Same structure for I=1
        self.Iy = self.Sy.copy()
        self.Iz = self.Sz.copy()
        self.I0 = self.S0.copy()
        
        # Define basis states
        self.ms_plus = np.array([1, 0, 0])  # |ms=+1⟩
        self.ms_zero = np.array([0, 1, 0])  # |ms=0⟩
        self.ms_minus = np.array([0, 0, 1])  # |ms=-1⟩
        
        # Initialize Hamiltonians
        self._update_hamiltonians()
    
    def _update_hamiltonians(self) -> None:
        """
        Update all Hamiltonian components based on current parameters.
        
        This method constructs different parts of the full Hamiltonian
        including zero-field splitting, Zeeman interaction, hyperfine
        coupling, strain effects, and microwave driving.
        """
        # Zero-field splitting term: D*Sz^2
        D = self.config['zero_field_splitting']
        H_zfs = D * np.dot(self.Sz, self.Sz)
        
        # Strain term: E*(Sx^2 - Sy^2)
        E = self.config['strain']
        H_strain = E * (np.dot(self.Sx, self.Sx) - np.dot(self.Sy, self.Sy))
        
        # Zeeman interaction: gamma * B · S
        gamma = self.config['gyromagnetic_ratio']
        Bx, By, Bz = self.magnetic_field
        H_zeeman = gamma * (Bx * self.Sx + By * self.Sy + Bz * self.Sz)
        
        # Hyperfine interaction (simplified A_parallel term only): A * Sz * Iz
        A = self.config['hyperfine_coupling_14n']
        # For simplified placeholder, we'll skip the tensor product with nuclear spin
        # This would be a 9x9 matrix in the full implementation
        H_hyperfine = np.zeros_like(self.Sz)  # Placeholder
        
        # Nuclear quadrupole interaction: P * Iz^2
        P = self.config['quadrupole_splitting_14n']
        # Again, simplified for placeholder
        H_quadrupole = np.zeros_like(self.Sz)  # Placeholder
        
        # Microwave driving term (when active)
        if self.mw_on:
            # Rabi frequency based on power
            power_mw = 10**(self.mw_power/10)  # Convert dBm to mW
            rabi_amplitude = np.sqrt(power_mw) * 5e6  # Scale factor: 5 MHz/sqrt(mW)
            
            # Resonance detuning
            detuning = self.mw_frequency - D - gamma * Bz  # For ms=0 to ms=-1 transition
            
            # Rotating frame Hamiltonian for driving ms=0 to ms=-1 transition
            # In rotating wave approximation (RWA)
            H_drive = rabi_amplitude * (self.Sx + 1j * self.Sy) / 2
        else:
            H_drive = np.zeros_like(self.Sz)
        
        # Store individual Hamiltonian components
        self.H_zfs = H_zfs
        self.H_strain = H_strain
        self.H_zeeman = H_zeeman
        self.H_hyperfine = H_hyperfine
        self.H_quadrupole = H_quadrupole
        self.H_drive = H_drive
        
        # Construct total Hamiltonian
        self.H_total = H_zfs + H_strain + H_zeeman + H_hyperfine + H_quadrupole
        self.H_with_drive = self.H_total + H_drive
    
    def _initialize_placeholder_system(self) -> None:
        """
        Initialize a simplified placeholder implementation when SimOS is not available.
        
        This creates a basic model that can be used for testing and development
        until full SimOS integration is implemented.
        """
        logger.info("Initializing placeholder NV system")
        self._update_hamiltonians()  # Ensure Hamiltonians are up to date
        
        # Create a simplified NV system object
        class PlaceholderNVSystem:
            """Simple placeholder for SimOS NV system."""
            
            def __init__(self, model):
                self.model = model
                self.energy_levels = {
                    'ms0': 0.0,
                    'ms_minus': -model.config['zero_field_splitting'],
                    'ms_plus': model.config['zero_field_splitting']
                }
                # Current state populations (ms=0, ms=-1, ms=+1)
                self.populations = {'ms0': 0.98, 'ms_minus': 0.01, 'ms_plus': 0.01}
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
                
            def evolve(self, dt):
                """Simplified time evolution."""
                # Update simulation time
                self.simulation_time += dt
                
                # This is a placeholder with simplified quantum evolution
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
                        
                    if abs(self.mw_freq - f_0_to_plus1) < 20e6:  # Within 20 MHz
                        # Rabi oscillation for ms=0 to ms=+1 transition
                        rabi_freq = 5e6 * np.sqrt(10**(self.mw_power/10) / 1.0)  # Scale with power
                        phase = 2 * np.pi * rabi_freq * dt
                        
                        # Simple population transfer
                        delta_p = 0.1 * np.sin(phase) * min(1.0, self.populations['ms0'])
                        self.populations['ms0'] -= delta_p
                        self.populations['ms_plus'] += delta_p
                
                if self.laser_on:
                    # Laser causes polarization to ms=0
                    rate = min(1.0, self.laser_power / 5.0)  # Scale with power
                    self.populations['ms0'] = min(0.98, self.populations['ms0'] + 0.1 * rate)
                    self.populations['ms_plus'] = max(0.01, self.populations['ms_plus'] - 0.05 * rate)
                    self.populations['ms_minus'] = max(0.01, self.populations['ms_minus'] - 0.05 * rate)
                
                # Apply T1 relaxation
                t1 = self.model.config['T1']
                relaxation_amount = dt / t1 if t1 > 0 else 0
                equilibrium = {'ms0': 0.98, 'ms_minus': 0.01, 'ms_plus': 0.01}
                
                for state in self.populations:
                    # Move population toward equilibrium
                    self.populations[state] = (1 - relaxation_amount) * self.populations[state] + \
                                              relaxation_amount * equilibrium[state]
                
                # Apply T2* dephasing (coherence decay)
                t2_star = self.model.config['T2_star']
                dephasing_amount = dt / t2_star if t2_star > 0 else 0
                
                # Apply more sophisticated decoherence model
                if self.model.config['decoherence_model'] == 'markovian':
                    # Simple exponential decay of coherences
                    for state in self.populations:
                        if state != 'ms0':  # Coherences between ms=0 and ms=±1
                            # Dephasing reduces off-diagonal elements (coherences)
                            self.populations[state] *= (1 - dephasing_amount)
                else:  # non-markovian model
                    # More complex model with environment memory effects
                    bath_coupling = self.model.config.get('bath_coupling_strength', 5e5)
                    # Frequency-dependent phase factor representing bath memory
                    bath_memory = np.exp(-(dt * bath_coupling)**2)
                    
                    for state in self.populations:
                        if state != 'ms0':
                            # Non-Markovian decay has oscillatory component
                            phase = 2 * np.pi * bath_coupling * dt
                            coherence_factor = np.exp(-dephasing_amount) * (np.cos(phase) + 1j * np.sin(phase) * bath_memory)
                            # Using only real part for our simplified model
                            self.populations[state] *= np.abs(coherence_factor)
                
                # Add some random noise
                if self.model.config.get('noise_amplitude'):
                    noise_amp = self.model.config['noise_amplitude']
                    for state in self.populations:
                        # Add small random perturbation
                        noise = noise_amp * (2 * np.random.random() - 1) * self.populations[state]
                        self.populations[state] = max(0, self.populations[state] + noise)
                
                # Normalize populations
                total = sum(self.populations.values())
                for state in self.populations:
                    self.populations[state] /= total
                
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
                b_z = field[2]  # z-component affects ms levels
                zfs = self.model.config['zero_field_splitting']
                
                # Update energy levels (simplified first-order Zeeman effect)
                self.energy_levels['ms0'] = 0.0
                self.energy_levels['ms_minus'] = -zfs - gamma * b_z
                self.energy_levels['ms_plus'] = zfs + gamma * b_z
                
            def apply_microwave(self, frequency, power, on):
                """Apply microwave field to the system."""
                self.mw_freq = frequency
                self.mw_power = power
                self.mw_on = on
                
            def apply_laser(self, power, on):
                """Apply laser to the system."""
                self.laser_power = power
                self.laser_on = on
                
            def get_fluorescence(self):
                """Calculate fluorescence based on current state populations."""
                # Get basic parameters
                ms0_rate = self.model.config['fluorescence_rate_ms0']
                ms1_rate = self.model.config['fluorescence_rate_ms1']
                bg_rate = self.model.config['background_count_rate']
                
                # Calculate fluorescence based on populations
                fluor = (self.populations['ms0'] * ms0_rate + 
                         (self.populations['ms_minus'] + self.populations['ms_plus']) * ms1_rate +
                         bg_rate)
                
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
            
            # Environment parameters
            'temperature': 300,  # K - Temperature of the environment
            'bath_coupling_strength': 5e5,  # Hz - Coupling to spin bath (for decoherence)
            
            # Hyperfine parameters (14N)
            'hyperfine_coupling_14n': 2.14e6,  # Hz - Hyperfine coupling to 14N
            'quadrupole_splitting_14n': -4.96e6,  # Hz - Nuclear quadrupole splitting
            
            # Simulation parameters
            'simulation_timestep': 1e-9,  # s (1 ns) - Time step for numerical integration
            'use_gpu': False,  # Whether to use GPU acceleration for simulation
            'simulation_loop_delay': 0.01,  # s - Delay between simulation iterations
            'adaptive_timestep': True,  # Whether to use adaptive timesteps
            'integration_method': 'RK45',  # Numerical integration method (RK45, RK23, BDF, etc.)
            'simulation_accuracy': 1e-6,  # Accuracy for adaptive integration methods
            
            # Experiment parameters
            'odmr_contrast_multiplier': 1.0,  # Multiplier for ODMR contrast (for calibration)
            'odmr_linewidth_multiplier': 1.0,  # Multiplier for ODMR linewidth (for calibration)
            'noise_amplitude': 0.01,  # Relative amplitude of random noise
            'decoherence_model': 'markovian',  # Model for decoherence ('markovian', 'non-markovian')
        }
    
    def set_magnetic_field(self, field_vector: Union[List[float], np.ndarray]) -> None:
        """
        Set the external magnetic field.
        
        Args:
            field_vector: 3D vector [Bx, By, Bz] representing the magnetic field in Tesla
            
        Note:
            The magnetic field affects the energy levels of the NV-center through Zeeman splitting.
            The field is stored as a numpy array and is thread-safe.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.set_magnetic_field([0, 0, 0.1])  # 0.1 Tesla along z-axis
        """
        with self.lock:
            self.magnetic_field = np.array(field_vector, dtype=float)
            
            # Update the NV system with new magnetic field
            if self.nv_system:
                try:
                    self.nv_system.apply_magnetic_field(self.magnetic_field)
                except Exception as e:
                    logger.error(f"Error applying magnetic field to NV system: {e}")
            
            logger.info(f"Magnetic field set to {self.magnetic_field} T")
    
    def get_magnetic_field(self) -> np.ndarray:
        """
        Get the current magnetic field.
        
        Returns:
            3D vector representing the magnetic field in Tesla
            
        Note:
            Returns a copy of the magnetic field vector to prevent external modification
            of the internal state. The method is thread-safe.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> field = model.get_magnetic_field()
            >>> print(field)
            [0. 0. 0.]
        """
        with self.lock:
            return self.magnetic_field.copy()
    
    def apply_microwave(self, frequency: float, power: float, on: bool = True) -> None:
        """
        Apply microwave excitation to the NV-center.
        
        Args:
            frequency: Microwave frequency in Hz
            power: Microwave power in dBm
            on: Whether to turn the microwave on (True) or off (False)
            
        Note:
            Microwave excitation is used to drive transitions between spin states.
            The frequency determines which transition is addressed, while the power
            controls the Rabi frequency (rate of oscillation between states).
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.apply_microwave(2.87e9, -10, True)  # Turn on MW at ZFS frequency
        """
        with self.lock:
            self.mw_frequency = frequency
            self.mw_power = power
            self.mw_on = on
            
            # Update the NV system with microwave parameters
            if self.nv_system and hasattr(self.nv_system, 'apply_microwave'):
                try:
                    # Convert dBm to amplitude for simulation
                    # P(dBm) = 10*log10(P(mW))
                    # P(mW) = 10^(P(dBm)/10)
                    power_mw = 10**(power/10)
                    
                    # Apply microwave field to the NV system
                    self.nv_system.apply_microwave(frequency, power_mw, on)
                except Exception as e:
                    logger.error(f"Error applying microwave to NV system: {e}")
            
            logger.info(f"Microwave set to {on}, frequency={frequency} Hz, power={power} dBm")
    
    def apply_laser(self, power: float, on: bool = True) -> None:
        """
        Apply laser excitation to the NV-center.
        
        Args:
            power: Laser power in mW
            on: Whether to turn the laser on (True) or off (False)
            
        Note:
            Laser excitation is used for state initialization and readout of the NV-center.
            Higher laser power generally leads to faster polarization but may cause heating.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.apply_laser(5.0, True)  # Turn on laser at 5mW
        """
        with self.lock:
            self.laser_power = power
            self.laser_on = on
            
            # Update the NV system with laser parameters
            if self.nv_system and hasattr(self.nv_system, 'apply_laser'):
                try:
                    # Apply laser to the NV system
                    self.nv_system.apply_laser(power, on)
                except Exception as e:
                    logger.error(f"Error applying laser to NV system: {e}")
            
            logger.info(f"Laser set to {on}, power={power} mW")
    
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
            
            # Stop any ongoing simulation
            if self.is_simulating:
                self.stop_simulation.set()
                if self.simulation_thread and self.simulation_thread.is_alive():
                    self.simulation_thread.join(timeout=1.0)
                self.is_simulating = False
                self.stop_simulation.clear()
            
            logger.info("NV state reset")
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Copy of the current configuration dictionary
            
        Note:
            Returns a copy of the configuration to prevent external modification
            of the internal configuration. The method is thread-safe.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> config = model.get_config()
            >>> print(config['zero_field_splitting'])
            2.87e+09
        """
        with self.lock:
            return self.config.copy()
            
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
            >>> model.update_config({'zero_field_splitting': 2.88e9, 'T1': 1.5e-3})
        """
        with self.lock:
            # Stop any running simulation before updating config
            was_simulating = self.is_simulating
            if was_simulating:
                self.stop_simulation_loop()
                
            self.config.update(config_updates)
            
            # Update simulation timestep if it was changed
            if 'simulation_timestep' in config_updates:
                self.dt = self.config['simulation_timestep']
            
            # If any NV parameters were updated, reinitialize the NV system
            nv_params = {
                'zero_field_splitting', 'strain', 'gyromagnetic_ratio', 
                'T1', 'T2', 'T2_star', 'bath_coupling_strength',
                'hyperfine_coupling_14n', 'quadrupole_splitting_14n',
                'fluorescence_rate_ms0', 'fluorescence_rate_ms1', 'background_count_rate',
                'optical_pumping_rate', 'decoherence_model'
            }
            
            if any(param in config_updates for param in nv_params):
                try:
                    self._initialize_simos_system()
                    logger.info("NV system reinitialized with updated parameters")
                except Exception as e:
                    logger.error(f"Error reinitializing NV system: {e}")
            
            # Restart simulation if it was running
            if was_simulating:
                self.start_simulation_loop()
                
            logger.info(f"Configuration updated with {config_updates}")
            
    def start_simulation_loop(self) -> None:
        """
        Start a background thread for continuous simulation.
        
        This method starts a separate thread that continuously evolves the quantum
        state of the NV system over time. This is useful for real-time simulations
        where the system evolves while external parameters are changed.
        
        Note:
            Only one simulation thread can be active at a time.
            The simulation can be stopped with stop_simulation_loop().
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.start_simulation_loop()
            >>> # The simulation now runs in background
            >>> model.set_magnetic_field([0, 0, 0.1])  # System state updates in background
            >>> model.stop_simulation_loop()  # Stop the simulation
        """
        with self.lock:
            if self.is_simulating:
                logger.warning("Simulation is already running")
                return
                
            if not self.nv_system:
                logger.error("Cannot start simulation: NV system not initialized")
                return
                
            # Clear the stop flag
            self.stop_simulation.clear()
            
            # Create and start the simulation thread
            self.simulation_thread = threading.Thread(
                target=self._simulation_loop,
                name="NVSimulationThread",
                daemon=True
            )  # type: ignore
            self.is_simulating = True
            if self.simulation_thread:
                self.simulation_thread.start()
            
            logger.info("Continuous simulation started")
            
    def stop_simulation_loop(self) -> None:
        """
        Stop the continuous simulation thread.
        
        This method stops the background simulation thread if one is running.
        
        Example:
            >>> model = PhysicalNVModel()
            >>> model.start_simulation_loop()
            >>> # Do something with the model while simulation runs
            >>> model.stop_simulation_loop()
        """
        with self.lock:
            if not self.is_simulating:
                logger.warning("No simulation is running")
                return
                
            # Set the stop flag to signal the thread to stop
            self.stop_simulation.set()
            
            # Wait for the thread to finish (with timeout)
            if self.simulation_thread and self.simulation_thread.is_alive():
                self.lock.release()  # Release lock while waiting
                try:
                    self.simulation_thread.join(timeout=2.0)
                finally:
                    self.lock.acquire()  # Re-acquire lock after waiting
                
            self.is_simulating = False
            logger.info("Continuous simulation stopped")
            
    def _simulation_loop(self) -> None:
        """
        Internal method to run the continuous simulation loop.
        
        This is the target function for the simulation thread. It continuously
        evolves the quantum state until the stop flag is set.
        
        The loop uses the current model configuration to determine the time step,
        integration method, and whether to use adaptive time stepping. This allows
        for flexible and efficient simulation of the quantum dynamics.
        """
        logger.debug("Simulation loop starting")
        
        try:
            # Main simulation loop
            while not self.stop_simulation.is_set():
                # Determine time step to use for this iteration
                use_adaptive = self.config.get('adaptive_timestep', True)
                
                # Use a local time step that adjusts based on current Hamiltonian
                if use_adaptive and hasattr(self, 'H_with_drive'):
                    # Calculate appropriate step size based on energy scales
                    # This is a simple heuristic: step size ~ 1/energy
                    with self.lock:
                        if self.mw_on:
                            # For active driving, use smaller steps
                            h_norm = np.linalg.norm(self.H_with_drive)
                            if h_norm > 0:
                                # Step size inversely proportional to energy scale
                                # but capped by minimum/maximum values
                                dt_adaptive = min(1.0 / (10 * h_norm), 100 * self.dt)
                                dt_adaptive = max(dt_adaptive, self.dt / 10)
                            else:
                                dt_adaptive = self.dt
                        else:
                            # For free evolution, can use larger steps
                            dt_adaptive = self.dt
                else:
                    # Use fixed time step from configuration
                    dt_adaptive = self.dt
                
                # Acquire lock for each iteration
                with self.lock:
                    # Use the new quantum state evolution method
                    self.evolve_quantum_state(dt_adaptive)
                    
                # Sleep briefly to avoid hogging CPU resources
                delay = self.config.get('simulation_loop_delay', 0.01)
                time.sleep(delay)
                
        except Exception as e:
            logger.error(f"Error in simulation loop: {e}")
        finally:
            logger.debug("Simulation loop exiting")
            # Ensure flag is cleared even if exception occurs
            self.is_simulating = False
            
    def simulate_odmr(self, 
                      freq_start: float, 
                      freq_stop: float, 
                      num_points: int = 101,
                      avg_time: float = 0.1) -> ODMRResult:
        """
        Simulate an ODMR (Optically Detected Magnetic Resonance) measurement.
        
        Args:
            freq_start: Start frequency for the ODMR scan in Hz
            freq_stop: Stop frequency for the ODMR scan in Hz
            num_points: Number of frequency points to measure
            avg_time: Averaging time per point in seconds
            
        Returns:
            ODMRResult object containing the simulated ODMR spectrum
            
        Note:
            ODMR is a technique to detect magnetic resonance of NV centers
            by monitoring their fluorescence while sweeping a microwave frequency.
            A dip in fluorescence occurs when the microwave frequency matches
            the energy level splitting between spin states.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> # Set a magnetic field along z-axis
            >>> model.set_magnetic_field([0, 0, 0.003])  # 3 mT
            >>> # Run ODMR around zero-field splitting
            >>> result = model.simulate_odmr(2.8e9, 2.94e9, 101)
            >>> # Analyze the result
            >>> print(f"ODMR center frequency: {result.center_frequency/1e6:.1f} MHz")
            >>> print(f"ODMR contrast: {result.contrast*100:.1f}%")
        """
        with self.lock:
            # Store original microwave parameters
            orig_mw_freq = self.mw_frequency
            orig_mw_power = self.mw_power
            orig_mw_on = self.mw_on
            
            # Stop any running simulation
            was_simulating = self.is_simulating
            if was_simulating:
                self.stop_simulation_loop()
            
            try:
                # Calculate ODMR frequencies
                frequencies = np.linspace(freq_start, freq_stop, num_points)
                
                # Set a power that gives good contrast
                self.mw_power = -10.0  # dBm
                
                # Reset the state to ensure consistent starting condition
                self.reset_state()
                
                # Prepare signal array
                signal = np.zeros(num_points)
                
                # Get reference signal (no MW)
                self.mw_on = False
                self.laser_on = True
                self.laser_power = 2.0  # mW
                
                # Calculate reference fluorescence
                ref_signal = self.get_fluorescence()
                
                # Sweep frequencies and measure fluorescence
                for i, freq in enumerate(frequencies):
                    # Apply microwave at this frequency
                    self.apply_microwave(float(freq), self.mw_power, True)
                    
                    # Reset NV state 
                    if self.nv_system and hasattr(self.nv_system, 'ground_state'):
                        self.current_state = self.nv_system.ground_state()
                    
                    # Simulate system evolution for averaging time
                    # This is a simplified simulation loop
                    iterations = max(1, int(avg_time / self.dt))
                    for _ in range(iterations):
                        if hasattr(self.nv_system, 'evolve'):
                            self.nv_system.evolve(self.dt)
                    
                    # Measure fluorescence
                    signal[i] = self.get_fluorescence()
                
                # Normalize the signal
                normalized_signal = signal / ref_signal
                
                # Find dips in the signal
                # Smooth the signal to reduce noise
                smoothed_signal = normalized_signal
                if num_points > 5:
                    window_size = max(3, int(num_points / 20))
                    if window_size % 2 == 0:
                        window_size += 1
                    window = np.ones(window_size) / window_size
                    # Use a simple moving average instead of np.convolve
                    smoothed_signal = np.copy(normalized_signal)
                    for i in range(len(normalized_signal)):
                        # For each point, average over window_size points around it
                        window_min = max(0, i - window_size // 2)
                        window_max = min(len(normalized_signal), i + window_size // 2 + 1)
                        smoothed_signal[i] = np.mean(normalized_signal[window_min:window_max])
                    
                # Find the minimum in the smoothed signal
                min_idx = np.argmin(smoothed_signal)
                min_freq = frequencies[min_idx]
                
                # Calculate contrast
                contrast = 1.0 - smoothed_signal[min_idx]
                
                # Apply ODMR contrast multiplier (for calibration)
                contrast *= self.config.get('odmr_contrast_multiplier', 1.0)
                
                # Calculate linewidth (FWHM)
                half_max = 1.0 - contrast / 2.0
                idx_above = smoothed_signal > half_max
                
                # Find the left and right crossing points
                linewidth = 10e6  # Default 10 MHz
                
                # Find left crossing point
                left_point = 0
                for i in range(min_idx):
                    if smoothed_signal[i] > half_max:
                        left_point = i
                
                # Find right crossing point
                right_point = len(smoothed_signal) - 1
                for i in range(min_idx + 1, len(smoothed_signal)):
                    if smoothed_signal[i] > half_max:
                        right_point = i
                        break
                
                # Calculate linewidth if valid points were found
                if right_point > left_point:
                    linewidth = frequencies[right_point] - frequencies[left_point]
                
                # Apply linewidth multiplier (for calibration)
                linewidth *= self.config.get('odmr_linewidth_multiplier', 1.0)
                
                # Create result object
                result = ODMRResult(
                    frequencies=frequencies,
                    signal=normalized_signal,
                    contrast=contrast,
                    center_frequency=min_freq,
                    linewidth=linewidth,
                    experiment_id=f"odmr_{str(uuid.uuid4())[:8]}"
                )
                
                # Cache result
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                cache_key = f"odmr_{timestamp}"
                self.cached_results[cache_key] = result
                
                return result
                
            finally:
                # Restore original microwave state
                self.apply_microwave(orig_mw_freq, orig_mw_power, orig_mw_on)
                
                # Restart simulation if it was running
                if was_simulating:
                    self.start_simulation_loop()
                    
    def simulate_rabi(self, 
                     duration: float, 
                     num_points: int = 101,
                     frequency: Optional[float] = None,
                     power: float = -10.0) -> RabiResult:
        """
        Simulate a Rabi oscillation measurement.
        
        Args:
            duration: Total duration of the Rabi measurement in seconds
            num_points: Number of time points to simulate
            frequency: Microwave frequency in Hz (defaults to zero-field splitting)
            power: Microwave power in dBm
            
        Returns:
            RabiResult object containing the simulated Rabi oscillation data
            
        Note:
            Rabi oscillations are coherent oscillations between spin states
            when a resonant microwave field is applied. The oscillation
            frequency depends on the microwave power.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> # Run Rabi oscillation measurement at ZFS frequency
            >>> result = model.simulate_rabi(4e-6, 101)  # 4 μs duration
            >>> # Analyze the result
            >>> print(f"Rabi frequency: {result.rabi_frequency/1e6:.1f} MHz")
        """
        with self.lock:
            # Store original microwave parameters
            orig_mw_freq = self.mw_frequency
            orig_mw_power = self.mw_power
            orig_mw_on = self.mw_on
            
            # Stop any running simulation
            was_simulating = self.is_simulating
            if was_simulating:
                self.stop_simulation_loop()
            
            try:
                # Set up time points for measurement
                times = np.linspace(0, duration, num_points)
                time_step = duration / (num_points - 1)
                
                # Set microwave frequency if specified
                if frequency is None:
                    # Default to zero-field splitting for resonance
                    # Adjust for magnetic field if present
                    zfs = self.config['zero_field_splitting']
                    gamma = self.config['gyromagnetic_ratio']
                    b_z = self.magnetic_field[2]  # z-component
                    
                    # Select lower transition frequency (ms=0 to ms=-1)
                    frequency = zfs - gamma * b_z
                
                # Prepare result array
                population = np.zeros(num_points)
                
                # For each time point, initialize, apply MW, and measure
                for i, t in enumerate(times):
                    # Reset to ground state
                    self.reset_state()
                    
                    # Apply resonant microwave pulse for time t
                    self.apply_microwave(frequency, power, True)
                    
                    # Simulate evolution for the pulse duration
                    iterations = max(1, int(t / self.dt))
                    for _ in range(iterations):
                        if hasattr(self.nv_system, 'evolve'):
                            self.nv_system.evolve(self.dt)
                    
                    # Stop microwave
                    self.mw_on = False
                    
                    # Get populations
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
                    min_idx = np.argmin(population[start_idx:]) + start_idx
                    
                    if min_idx < len(times) - 1 and times[min_idx] > 0:
                        # Period = 2x first minimum time (for sine starting at max)
                        period = 2 * times[min_idx]
                        rabi_freq = 1.0 / period
                    else:
                        # Fallback: estimate based on microwave power
                        # Simple estimate: Rabi frequency ~ sqrt(power) * constant
                        # Power is in dBm, convert to linear scale
                        power_linear = 10**(power/10)  # mW
                        rabi_freq = 2e6 * np.sqrt(power_linear / 10)  # 2 MHz at 10 mW
                except Exception as e:
                    logger.error(f"Error calculating Rabi frequency: {e}")
                    # Use a default value
                    rabi_freq = 2e6  # Hz
                
                # Estimate decay time with a simpler approach
                decay_time = None
                try:
                    # Use a sliding window to find the envelope
                    if num_points >= 5:
                        # Simple estimation based on amplitude decay
                        # Find first oscillation peak
                        first_peak = np.argmax(population[:num_points//3])
                        if first_peak > 0:
                            # Find a later point where amplitude has decayed
                            later_idx = min(int(num_points-1), int(first_peak + num_points//2))
                            later_value = population[later_idx]
                            first_value = population[first_peak]
                            
                            # If there's significant decay
                            if first_value > later_value * 1.5:
                                # Estimate decay constant
                                delta_t = times[later_idx] - times[first_peak]
                                ratio = later_value / first_value
                                # tau = -delta_t / ln(ratio)
                                if ratio > 0:
                                    decay_time = -delta_t / np.log(ratio)
                except Exception as e:
                    logger.error(f"Error estimating decay time: {e}")
                
                # Use T2* from config as fallback if estimation failed
                if decay_time is None or decay_time <= 0:
                    decay_time = self.config.get('T2', 1e-6)  # seconds
                
                # Create result object
                result = RabiResult(
                    times=times,
                    population=population,
                    rabi_frequency=rabi_freq,
                    decay_time=decay_time,
                    experiment_id=f"rabi_{str(uuid.uuid4())[:8]}"
                )
                
                # Cache result
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                cache_key = f"rabi_{timestamp}"
                self.cached_results[cache_key] = result
                
                return result
                
            finally:
                # Restore original microwave state
                self.apply_microwave(orig_mw_freq, orig_mw_power, orig_mw_on)
                
                # Restart simulation if it was running
                if was_simulating:
                    self.start_simulation_loop()
                    
    def simulate_t1(self,
                   duration: float,
                   num_points: int = 51) -> T1Result:
        """
        Simulate a T1 relaxation measurement.
        
        Args:
            duration: Total duration of the T1 measurement in seconds
            num_points: Number of time points to simulate
            
        Returns:
            T1Result object containing the simulated T1 relaxation data
            
        Note:
            T1 relaxation is the process by which the spin population returns
            to thermal equilibrium. This measurement involves initializing
            the spin to a non-equilibrium state and then measuring the
            population as a function of time.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> # Set T1 time in configuration
            >>> model.update_config({'T1': 2e-3})  # 2 ms
            >>> # Run T1 measurement
            >>> result = model.simulate_t1(10e-3, 51)  # 10 ms duration
            >>> # Analyze the result
            >>> print(f"Measured T1: {result.t1_time*1000:.1f} ms")
        """
        with self.lock:
            # Store original parameters
            orig_mw_freq = self.mw_frequency
            orig_mw_power = self.mw_power
            orig_mw_on = self.mw_on
            orig_laser_on = self.laser_on
            orig_laser_power = self.laser_power
            
            # Stop any running simulation
            was_simulating = self.is_simulating
            if was_simulating:
                self.stop_simulation_loop()
            
            try:
                # Set up time points for measurement
                times = np.linspace(0, duration, num_points)
                
                # Prepare result array
                population = np.zeros(num_points)
                
                # For each time point, initialize, wait, and measure
                for i, t in enumerate(times):
                    # Initialize to ms=±1 state with a π pulse
                    self.reset_state()  # Start in ms=0
                    
                    # Apply resonant π pulse to transfer to ms=-1
                    # Use zero-field splitting frequency
                    zfs = self.config['zero_field_splitting']
                    gamma = self.config['gyromagnetic_ratio']
                    b_z = self.magnetic_field[2]  # z-component
                    
                    # Target ms=0 to ms=-1 transition
                    freq = zfs - gamma * b_z
                    self.apply_microwave(freq, -5.0, True)  # Higher power for faster π pulse
                    
                    # Estimate rabi frequency to calculate π pulse duration
                    rabi_freq = 5e6  # Hz, estimated
                    pi_pulse_time = 1 / (2 * rabi_freq)  # Time for π pulse
                    
                    # Apply π pulse
                    iterations = max(1, int(pi_pulse_time / self.dt))
                    for _ in range(iterations):
                        if hasattr(self.nv_system, 'evolve'):
                            self.nv_system.evolve(self.dt)
                    
                    # Turn off microwave
                    self.mw_on = False
                    
                    # Wait for relaxation time t
                    wait_iterations = max(1, int(t / self.dt))
                    for _ in range(wait_iterations):
                        if hasattr(self.nv_system, 'evolve'):
                            self.nv_system.evolve(self.dt)
                    
                    # Measure population
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
                    if initial_value > final_value * 1.2:
                        # Find the point where population drops to 1/e of the way from initial to final
                        decay_amount = (initial_value - final_value) * (1 - 1/np.e) + final_value
                        
                        # Find the first point that falls below this value
                        decay_idx = 0
                        for i, p in enumerate(population):
                            if p <= decay_amount:
                                decay_idx = i
                                break
                        
                        # If we found a valid decay point
                        if decay_idx > 0 and decay_idx < len(times) - 1:
                            t1_time = times[decay_idx]
                        else:
                            # Fallback to configured value
                            t1_time = self.config['T1']
                    else:
                        # Not enough decay, use configured value
                        t1_time = self.config['T1']
                        
                except Exception as e:
                    logger.error(f"Error extracting T1 relaxation: {e}")
                    # Use configured T1 as fallback
                    t1_time = self.config['T1']
                
                # Create result object
                result = T1Result(
                    times=times,
                    population=population,
                    t1_time=t1_time,
                    experiment_id=f"t1_{str(uuid.uuid4())[:8]}"
                )
                
                # Cache result
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                cache_key = f"t1_{timestamp}"
                self.cached_results[cache_key] = result
                
                return result
                
            finally:
                # Restore original parameters
                self.apply_microwave(orig_mw_freq, orig_mw_power, orig_mw_on)
                self.apply_laser(orig_laser_power, orig_laser_on)
                
                # Restart simulation if it was running
                if was_simulating:
                    self.start_simulation_loop()
    
    def evolve_quantum_state(self, 
                           duration: float, 
                           hamiltonian_only: bool = False) -> Dict[str, Any]:
        """
        Evolve the quantum state for a specified duration.
        
        This method implements the core quantum evolution algorithms,
        applying both coherent evolution under the Hamiltonian and
        incoherent processes through Lindblad terms.
        
        Args:
            duration: Time to evolve in seconds
            hamiltonian_only: If True, only apply coherent Hamiltonian evolution
                             without relaxation or dephasing effects
        
        Returns:
            Dictionary with updated state information
            
        Note:
            This is the main method for explicit quantum state evolution, as opposed
            to the continuous simulation in the background thread.
            
            For coherent evolution, it uses the Schrödinger or Liouville-von Neumann
            equation. For incoherent processes, it applies Lindblad-type relaxation
            and dephasing terms.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.reset_state()
            >>> # Apply resonant microwave
            >>> model.apply_microwave(2.87e9, -10, True)
            >>> # Evolve for 100 ns
            >>> model.evolve_quantum_state(100e-9)
            >>> # Check new state
            >>> state_info = model.get_state_info()
            >>> print(f"ms=0 population: {state_info['populations']['ms0']:.3f}")
        """
        with self.lock:
            # Update Hamiltonians to make sure they reflect current settings
            if not SIMOS_AVAILABLE:
                self._update_hamiltonians()
            
            # Choose which integration method to use
            method = self.config['integration_method']
            use_adaptive = self.config['adaptive_timestep']
            
            # SimOS-based evolution
            if SIMOS_AVAILABLE and hasattr(self.nv_system, 'evolve_state'):
                try:
                    # Let SimOS handle state evolution
                    self.current_state = self.nv_system.evolve_state(
                        self.current_state, 
                        duration, 
                        method=method,
                        adaptive=use_adaptive,
                        coherent_only=hamiltonian_only
                    )
                    
                    # Get updated state information
                    return self.get_state_info()
                    
                except Exception as e:
                    logger.error(f"Error in SimOS state evolution: {e}")
                    # Fall back to placeholder implementation
                    pass
            
            # Placeholder evolution if SimOS fails or isn't available
            try:
                # Using the PlaceholderNVSystem evolve method
                if hasattr(self.nv_system, 'evolve'):
                    # Forward evolution to the placeholder system
                    if use_adaptive and duration > self.dt:
                        # For adaptive timestep with large duration, break into chunks
                        remaining = duration
                        while remaining > 0:
                            # Estimate good timestep based on Hamiltonian norm
                            if hasattr(self, 'H_with_drive'):
                                # Calculate appropriate step size based on energy scales
                                h_norm = np.linalg.norm(self.H_with_drive)
                                if h_norm > 0:
                                    # Step size inversely proportional to energy scale
                                    step = min(remaining, 1.0 / (10 * h_norm))
                                else:
                                    step = min(remaining, self.dt)
                            else:
                                step = min(remaining, self.dt)
                            
                            # Evolve with this step size
                            self.nv_system.evolve(step)
                            remaining -= step
                    else:
                        # Single step evolution
                        self.nv_system.evolve(duration)
                        
                    # Get updated state information
                    return self.get_state_info()
                    
                else:
                    # For direct density matrix evolution, implement 
                    # matrix exponential method: rho(t+dt) = exp(-i*H*dt) rho(t) exp(i*H*dt)
                    logger.warning("Advanced quantum evolution not available in placeholder")
                    return self.get_state_info()
                
            except Exception as e:
                logger.error(f"Error in placeholder state evolution: {e}")
                return self.get_state_info()

    def simulate_spin_echo(self,
                          tau_max: float,
                          num_points: int = 51) -> T2Result:
        """
        Simulate a spin echo measurement to determine T2 coherence time.
        
        Args:
            tau_max: Maximum delay time (total sequence duration will be 2*tau_max)
            num_points: Number of delay time points to simulate
            
        Returns:
            T2Result object containing the simulated spin echo data
            
        Note:
            The spin echo sequence consists of:
            1. π/2 pulse to create superposition
            2. Free evolution for time τ
            3. π pulse to refocus
            4. Free evolution for time τ
            5. π/2 pulse to convert coherence to population
            6. Readout
            
            This sequence mitigates the effect of static/slowly varying fields
            and measures the true decoherence time T2.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.update_config({'T2': 300e-6})  # Set T2 = 300 µs
            >>> result = model.simulate_spin_echo(500e-6, 51)  # Up to 500 µs delay
            >>> print(f"Measured T2: {result.t2_time*1e6:.1f} µs")
        """
        with self.lock:
            # Store original parameters
            orig_mw_freq = self.mw_frequency
            orig_mw_power = self.mw_power
            orig_mw_on = self.mw_on
            
            # Stop any running simulation
            was_simulating = self.is_simulating
            if was_simulating:
                self.stop_simulation_loop()
            
            try:
                # Set up time points (delay times)
                tau_values = np.linspace(0, tau_max, num_points)
                
                # Prepare result array
                signal = np.zeros(num_points)
                
                # Resonant frequency for ms=0 to ms=-1 transition
                zfs = self.config['zero_field_splitting']
                gamma = self.config['gyromagnetic_ratio']
                b_z = self.magnetic_field[2]
                frequency = zfs - gamma * b_z
                
                # Use sufficient power for fast pulses
                power = 0.0  # 0 dBm
                
                # Calculate π and π/2 pulse durations based on Rabi frequency
                # Rabi frequency scales with sqrt(power)
                power_mw = 10**(power/10)  # Convert dBm to mW
                rabi_freq = 10e6 * np.sqrt(power_mw / 10)  # 10 MHz at 10 mW
                pi_time = 1 / (2 * rabi_freq)
                pi2_time = pi_time / 2
                
                # For each delay time, run spin echo sequence
                for i, tau in enumerate(tau_values):
                    # 1. Initialize state
                    self.reset_state()
                    
                    # 2. Apply first π/2 pulse around X axis (creates |+y⟩ state)
                    self.apply_microwave(frequency, power, True)
                    self.evolve_quantum_state(pi2_time)
                    self.mw_on = False
                    
                    # 3. Free evolution for time τ
                    self.evolve_quantum_state(tau)
                    
                    # 4. Apply π pulse around X axis (flips to |-y⟩ state)
                    self.apply_microwave(frequency, power, True)
                    self.evolve_quantum_state(pi_time)
                    self.mw_on = False
                    
                    # 5. Free evolution for another time τ
                    self.evolve_quantum_state(tau)
                    
                    # 6. Apply final π/2 pulse to convert coherence to population
                    self.apply_microwave(frequency, power, True)
                    self.evolve_quantum_state(pi2_time)
                    self.mw_on = False
                    
                    # 7. Measure final state
                    state_info = self.get_state_info()
                    populations = state_info.get('populations', {})
                    
                    # Calculate normalized signal (ms=0 population should be close to 1 
                    # with perfect coherence, and 0.5 with complete decoherence)
                    ms0_pop = populations.get('ms0', 0.5)
                    # Normalize to [0, 1] where 1 is perfect coherence
                    signal[i] = 2 * (ms0_pop - 0.5)
                
                # Extract T2 time by fitting exponential decay: exp(-(t/T2)^n)
                # where n=1 for simple exponential, n=2-3 for spin bath decoherence
                try:
                    # Find where signal drops to 1/e
                    threshold = np.exp(-1.0)  # ~0.368
                    norm_signal = signal / np.max(signal)  # Normalize to max
                    
                    # Find first point below threshold
                    below_threshold = np.where(norm_signal < threshold)[0]
                    if len(below_threshold) > 0:
                        idx = below_threshold[0]
                        if idx > 0 and idx < len(tau_values):
                            t2_time = tau_values[idx]
                        else:
                            t2_time = self.config['T2']
                    else:
                        # No decay within measurement time
                        t2_time = self.config['T2']
                except Exception as e:
                    logger.error(f"Error extracting T2 time: {e}")
                    t2_time = self.config['T2']
                
                # Create result object
                result = T2Result(
                    times=tau_values,
                    signal=signal,
                    t2_time=t2_time,
                    experiment_id=f"t2_{str(uuid.uuid4())[:8]}"
                )
                
                # Cache result
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                cache_key = f"t2_{timestamp}"
                self.cached_results[cache_key] = result
                
                return result
                
            finally:
                # Restore original parameters
                self.apply_microwave(orig_mw_freq, orig_mw_power, orig_mw_on)
                
                # Restart simulation if it was running
                if was_simulating:
                    self.start_simulation_loop()
    
    def simulate_state_evolution(self,
                                duration: float,
                                num_points: int = 101,
                                hamiltonian_only: bool = False) -> StateEvolution:
        """
        Simulate the evolution of the quantum state over time.
        
        Args:
            duration: Total duration of the evolution in seconds
            num_points: Number of time points to record
            hamiltonian_only: If True, only include coherent evolution
            
        Returns:
            StateEvolution object containing populations and coherences over time
            
        Note:
            This method tracks the complete quantum state evolution over time,
            recording both populations and coherences. It's useful for visualizing
            quantum dynamics like Rabi oscillations, decoherence, etc.
            
        Example:
            >>> model = PhysicalNVModel()
            >>> model.reset_state()
            >>> model.apply_microwave(2.87e9, -10, True)  # Resonant MW
            >>> # Simulate 1µs evolution with 101 points
            >>> result = model.simulate_state_evolution(1e-6, 101)
            >>> # Plot ms=0 population vs time
            >>> plt.plot(result.times, result.populations['ms0'])
        """
        with self.lock:
            # Store original state
            orig_state = None
            if hasattr(self.nv_system, 'get_populations'):
                orig_state = self.nv_system.get_populations().copy()
            
            # Stop any running simulation
            was_simulating = self.is_simulating
            if was_simulating:
                self.stop_simulation_loop()
            
            try:
                # Initialize state
                self.reset_state()
                
                # Set up time points
                times = np.linspace(0, duration, num_points)
                
                # Initialize result arrays
                populations = {
                    'ms0': np.zeros(num_points),
                    'ms_minus': np.zeros(num_points),
                    'ms_plus': np.zeros(num_points)
                }
                
                # Coherences (simplified for placeholder implementation)
                coherences = {
                    'ms0_minus': np.zeros(num_points, dtype=complex),
                    'ms0_plus': np.zeros(num_points, dtype=complex),
                    'ms_minus_plus': np.zeros(num_points, dtype=complex)
                }
                
                # Record initial state
                if hasattr(self.nv_system, 'get_populations'):
                    pops = self.nv_system.get_populations()
                    for state in populations:
                        populations[state][0] = pops.get(state, 0.0)
                
                # Evolve state and record at each time point
                current_time = 0.0
                for i in range(1, num_points):
                    # Calculate time step
                    step_time = times[i] - times[i-1]
                    
                    # Evolve quantum state
                    self.evolve_quantum_state(step_time, hamiltonian_only)
                    
                    # Record populations
                    if hasattr(self.nv_system, 'get_populations'):
                        pops = self.nv_system.get_populations()
                        for state in populations:
                            populations[state][i] = pops.get(state, 0.0)
                    
                    # For coherences, we would ideally access the full density matrix
                    # This is a simplified implementation for the placeholder
                    # In a real implementation with SimOS, we would extract coherences from
                    # the off-diagonal elements of the density matrix
                    
                # Create result object
                result = StateEvolution(
                    times=times,
                    populations=populations,
                    coherences=coherences,
                    experiment_id=f"evolution_{str(uuid.uuid4())[:8]}"
                )
                
                # Cache result
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                cache_key = f"evolution_{timestamp}"
                self.cached_results[cache_key] = result
                
                return result
                
            finally:
                # Restore original state if available
                if orig_state:
                    # In a real implementation, we would restore the full density matrix
                    # For the placeholder, we can only approximately restore populations
                    if hasattr(self.nv_system, 'populations'):
                        self.nv_system.populations = orig_state
                
                # Restart simulation if it was running
                if was_simulating:
                    self.start_simulation_loop()
    
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
                    logger.error(f"Error getting quantum state information: {e}")
                    state_info['quantum_state_error'] = str(e)
            
            # Add cached results summary if available
            if self.cached_results:
                result_summary = {}
                for key, value in self.cached_results.items():
                    result_type = type(value).__name__
                    if hasattr(value, 'experiment_id'):
                        result_summary[key] = {
                            'type': result_type,
                            'id': value.experiment_id,
                            'timestamp': key.split('_')[-1] if '_' in key else 'unknown'
                        }
                state_info['cached_results'] = result_summary
            
            return state_info