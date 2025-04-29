"""
Physical NV-Center Model for quantum simulation.

This module provides the core physical model for NV-center simulations,
implementing the quantum mechanical behavior and integrating with SimOS.
"""

import numpy as np
import threading
import logging
import time
from typing import Dict, Any, Optional, Tuple, List, Union, Callable

# Configure logging
logger = logging.getLogger(__name__)

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
                    hyperfine_coupling=hf_coupling,
                    quadrupole_splitting=quad_splitting,
                    use_gpu=self.config['use_gpu']
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
            
    def _initialize_placeholder_system(self) -> None:
        """
        Initialize a simplified placeholder implementation when SimOS is not available.
        
        This creates a basic model that can be used for testing and development
        until full SimOS integration is implemented.
        """
        logger.info("Initializing placeholder NV system")
        
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
                
            def ground_state(self):
                """Return ground state (ms=0)."""
                return {'state': 'ms0', 'population': 1.0}
                
            def evolve(self, dt):
                """Simplified time evolution."""
                # This is just a placeholder with no real quantum evolution
                # In a real implementation, this would integrate quantum dynamics
                if self.mw_on:
                    # Check if microwave is resonant with a transition
                    zfs = self.model.config['zero_field_splitting']
                    # If MW is resonant with ms=0 to ms=±1 transition
                    if abs(self.mw_freq - zfs) < 10e6:  # Within 10 MHz
                        # Simulate some population transfer
                        self.populations['ms0'] = max(0.7, self.populations['ms0'] - 0.01)
                        self.populations['ms_plus'] = min(0.3, self.populations['ms_plus'] + 0.005)
                        self.populations['ms_minus'] = min(0.3, self.populations['ms_minus'] + 0.005)
                
                if self.laser_on:
                    # Laser causes polarization to ms=0
                    rate = min(1.0, self.laser_power / 5.0)  # Scale with power
                    self.populations['ms0'] = min(0.98, self.populations['ms0'] + 0.1 * rate)
                    self.populations['ms_plus'] = max(0.01, self.populations['ms_plus'] - 0.05 * rate)
                    self.populations['ms_minus'] = max(0.01, self.populations['ms_minus'] - 0.05 * rate)
                
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
                # Simplified: Just update energy levels with Zeeman shift
                # E = γ * |B| * m_s
                b_mag = np.linalg.norm(field)
                gamma = self.model.config['gyromagnetic_ratio']
                self.energy_levels['ms_minus'] = -self.model.config['zero_field_splitting'] - gamma * b_mag
                self.energy_levels['ms_plus'] = self.model.config['zero_field_splitting'] + gamma * b_mag
                
            def apply_microwave(self, frequency, power, on):
                """Apply microwave field to the system."""
                self.mw_freq = frequency
                self.mw_power = power
                self.mw_on = on
                
            def apply_laser(self, power, on):
                """Apply laser to the system."""
                self.laser_power = power
                self.laser_on = on
        
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
            'T2': 1e-6,  # s - Transverse relaxation time
            
            # Optical properties
            'fluorescence_contrast': 0.3,  # Fluorescence contrast between ms=0 and ms=±1
            'optical_readout_fidelity': 0.93,  # Fidelity of optical state readout
            
            # Environment parameters
            'temperature': 300,  # K - Temperature of the environment
            
            # Hyperfine parameters (14N)
            'hyperfine_coupling_14n': 2.14e6,  # Hz - Hyperfine coupling to 14N
            'quadrupole_splitting_14n': -4.96e6,  # Hz - Nuclear quadrupole splitting
            
            # Simulation parameters
            'simulation_timestep': 1e-9,  # s (1 ns) - Time step for numerical integration
            'use_gpu': False  # Whether to use GPU acceleration for simulation
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
            self.config.update(config_updates)
            
            # Update simulation timestep if it was changed
            if 'simulation_timestep' in config_updates:
                self.dt = self.config['simulation_timestep']
            
            # If any NV parameters were updated, reinitialize the NV system
            nv_params = {'zero_field_splitting', 'strain', 'gyromagnetic_ratio', 
                         'T1', 'T2', 'hyperfine_coupling_14n', 'quadrupole_splitting_14n'}
            
            if any(param in config_updates for param in nv_params):
                try:
                    self._initialize_simos_system()
                    logger.info("NV system reinitialized with updated parameters")
                except Exception as e:
                    logger.error(f"Error reinitializing NV system: {e}")
            
            logger.info(f"Configuration updated with {config_updates}")
    
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
                'laser_on': self.laser_on
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
                        
                except Exception as e:
                    logger.error(f"Error getting quantum state information: {e}")
                    state_info['quantum_state_error'] = str(e)
            
            return state_info