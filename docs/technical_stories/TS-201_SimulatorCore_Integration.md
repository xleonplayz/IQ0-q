# Technical Story: TS-201 SimulatorCore Integration

## Objective
Create a central simulator core manager that integrates the NV center simulator with Qudi's dummy hardware modules. This component will serve as a continuous background service accessible to all dummy modules, providing physically accurate simulations for NV center experiments.

## Requirements
- Implement a singleton pattern for consistent access across all modules
- Provide comprehensive access to all simulator features
- Maintain backward compatibility with existing interfaces
- Handle thread safety and concurrent access
- Support configurable parameters via Qudi's configuration system
- Ensure proper error handling and recovery mechanisms

## Technical Specifications

### Package Structure

The simulator will be integrated directly into the Qudi package structure to ensure reliable imports and proper operation:

```
/src/qudi/hardware/nv_simulator/
  __init__.py                   # Package initialization
  simulator_manager.py          # Central manager class
  model.py                      # NV center physics model
  confocal/                     # Confocal scanning components
    __init__.py
    confocal_simulator.py
    diamond_lattice.py
    focused_laser.py
  nuclear/                      # Nuclear environment simulation
    __init__.py
    spin_bath.py
    hyperfine.py
  simos/                        # Minimal version of quantum simulation engine
    __init__.py
    core.py
    states.py
  interfaces/                   # Direct implementations of Qudi interfaces
    __init__.py
    microwave.py
    fast_counter.py
    scanning_probe.py
```

This structure ensures all simulator components are properly packaged within Qudi's namespace, avoiding import problems.

### Class Structure

```python
class SimulatorManager:
    """
    Central manager for NV center simulator integration with Qudi dummy modules.
    
    This class implements a singleton pattern to ensure a single simulator instance
    is shared across all dummy hardware modules. It provides a simplified access layer
    to the underlying NV simulator functionality with robust error handling.
    """
    
    _instance = None
    _DEFAULT_CONFIG = {
        'simulator': {
            'zero_field_splitting': 2.87e9,
            'gyromagnetic_ratio': 28.025e9,
            't1': 5.0e-3,
            't2': 1.0e-5,
            'optics': True,
            'nitrogen': False,
            'method': 'qutip'
        },
        'confocal': {
            'lattice': {
                'nv_density': 0.5,
                'size': [50e-6, 50e-6, 50e-6]
            },
            'laser': {
                'wavelength': 532e-9,
                'numerical_aperture': 0.8,
                'power': 1.0
            }
        },
        'nuclear_environment': {
            'enabled': False,
            'c13_concentration': 0.011,
            'bath_size': 50
        }
    }
    
    def __new__(cls, config=None):
        """Ensure singleton implementation."""
        if cls._instance is None:
            cls._instance = super(SimulatorManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config=None):
        """Initialize the simulator manager with configuration."""
        # Skip re-initialization if already done
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        # Mark as not fully initialized yet to prevent recursion issues
        self._initialized = False
        
        # Set up thread lock early for thread-safe initialization
        self._thread_lock = self._create_thread_lock()
        
        # Configure logging
        self.log = self._setup_logging()
        self.log.info("Initializing NV Simulator Manager")
        
        with self._thread_lock:
            try:
                # Merge configurations with precedence: provided config > defaults
                self.config = self._merge_configs(self._DEFAULT_CONFIG, config or {})
                
                # Initialize the NV model and components
                self._initialize_simulator()
                
                # Module registration tracking with watchdog
                self.active_modules = set()
                self._module_watchdogs = {}
                
                # Health monitoring
                self._last_health_check = time.time()
                self._is_running = True
                self._init_health_monitoring()
                
                # Mark as successfully initialized
                self._initialized = True
                self.log.info("NV Simulator Manager initialized successfully")
                
            except Exception as e:
                self.log.error(f"Failed to initialize simulator manager: {str(e)}")
                # Partial initialization for graceful fallback
                self._initialized = False
                self._is_running = False
                # Re-raise for proper handling by caller
                raise
    
    def _create_thread_lock(self):
        """Create the appropriate thread lock based on environment."""
        try:
            # Use Qudi's mutex if available
            from qudi.util.mutex import RecursiveMutex
            return RecursiveMutex()
        except ImportError:
            # Fall back to standard threading library
            return threading.RLock()
    
    def _setup_logging(self):
        """Set up proper logging for the simulator."""
        try:
            # Try to use Qudi's logging
            from qudi.core.logger import get_logger
            return get_logger(__name__)
        except ImportError:
            # Fall back to standard logging
            logger = logging.getLogger(__name__)
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)
            return logger
    
    def _merge_configs(self, base, override):
        """
        Merge configurations with clear priority rules.
        
        @param dict base: Base configuration
        @param dict override: Configuration to override base values
        @return dict: Merged configuration
        """
        result = copy.deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def _initialize_simulator(self):
        """Initialize the NV simulator with proper error handling."""
        try:
            # Import model from local package structure
            from qudi.hardware.nv_simulator.model import PhysicalNVModel
            
            # Create simulator instance with parameters
            simulator_params = self.config.get('simulator', {})
            self.nv_model = PhysicalNVModel(**simulator_params)
            self.log.info("NV center simulator model initialized")
            
            # Initialize additional components
            self._init_confocal_simulator()
            self._init_nuclear_environment()
            
        except Exception as e:
            self.log.error(f"Failed to initialize simulator components: {str(e)}")
            # Create a fallback minimal simulator or raise exception
            self._setup_fallback_simulator()
    
    def _init_confocal_simulator(self):
        """Initialize confocal simulator with error handling."""
        try:
            # Use local package imports
            from qudi.hardware.nv_simulator.confocal import ConfocalSimulator, DiamondLattice, FocusedLaserBeam
            
            # Get confocal configuration
            confocal_config = self.config.get('confocal', {})
            
            # Create diamond lattice with NV centers
            lattice_config = confocal_config.get('lattice', {})
            self.diamond_lattice = DiamondLattice(
                nv_density=lattice_config.get('nv_density', 1.0),
                size=lattice_config.get('size', (50e-6, 50e-6, 50e-6)),
                seed=lattice_config.get('seed', None)
            )
            
            # Configure laser beam
            laser_config = confocal_config.get('laser', {})
            self.laser_beam = FocusedLaserBeam(
                wavelength=laser_config.get('wavelength', 532e-9),
                numerical_aperture=laser_config.get('numerical_aperture', 0.8),
                power=laser_config.get('power', 1.0)
            )
            
            # Create confocal simulator
            self.confocal_simulator = ConfocalSimulator(
                diamond_lattice=self.diamond_lattice,
                laser_beam=self.laser_beam,
                nv_model=self.nv_model
            )
            self.log.info("Confocal simulator component initialized")
            
        except Exception as e:
            self.log.warning(f"Could not initialize confocal simulator: {str(e)}")
            self.confocal_simulator = None
            self.diamond_lattice = None
            self.laser_beam = None
    
    def _init_nuclear_environment(self):
        """Initialize nuclear environment simulator with error handling."""
        try:
            # Use local package imports
            from qudi.hardware.nv_simulator.nuclear import SpinBath, HyperfineInteraction
            
            # Initialize nuclear spin environment if configured
            nuclear_config = self.config.get('nuclear_environment', {})
            if nuclear_config.get('enabled', False):
                self.spin_bath = SpinBath(
                    c13_concentration=nuclear_config.get('c13_concentration', 0.011),
                    bath_size=nuclear_config.get('bath_size', 50)
                )
                self.hyperfine = HyperfineInteraction(
                    nv_model=self.nv_model,
                    spin_bath=self.spin_bath
                )
                self.log.info("Nuclear spin environment initialized")
            else:
                self.spin_bath = None
                self.hyperfine = None
                
        except Exception as e:
            self.log.warning(f"Could not initialize nuclear environment: {str(e)}")
            self.spin_bath = None
            self.hyperfine = None
    
    def _init_health_monitoring(self):
        """Initialize health monitoring according to environment."""
        try:
            # Try to use Qt timer if in Qt environment
            from qtpy import QtCore
            
            self._health_timer = QtCore.QTimer()
            self._health_timer.timeout.connect(self._health_check)
            self._health_timer.start(5000)  # Check every 5 seconds
            self.log.debug("Health monitoring initialized with Qt timer")
            
        except ImportError:
            # Fall back to thread-based monitoring
            self._health_thread = threading.Thread(
                target=self._health_monitor_loop,
                daemon=True
            )
            self._health_thread.start()
            self.log.debug("Health monitoring initialized with background thread")
    
    def _health_monitor_loop(self):
        """Background thread for health monitoring."""
        while self._is_running:
            try:
                self._health_check()
            except Exception as e:
                self.log.error(f"Error in health monitor: {str(e)}")
                
            # Sleep between checks
            time.sleep(5)
    
    def _health_check(self):
        """Check simulator health and attempt recovery if needed."""
        with self._thread_lock:
            try:
                # Simple test operation to verify simulator is responsive
                if hasattr(self, 'nv_model') and self.nv_model is not None:
                    test = self.nv_model.get_fluorescence() is not None
                    
                    # If test fails, attempt recovery
                    if not test:
                        self.log.warning("Simulator health check failed, attempting recovery")
                        self._reinitialize_simulator()
                        
                # Check for stale module registrations
                self._check_module_watchdogs()
                
                # Update last health check timestamp
                self._last_health_check = time.time()
                
            except Exception as e:
                self.log.error(f"Health check failed: {str(e)}")
                self._reinitialize_simulator()
    
    def _check_module_watchdogs(self):
        """Check for stale module registrations."""
        current_time = time.time()
        stale_modules = []
        
        # Identify modules that haven't been active for 30 seconds
        for module, last_seen in self._module_watchdogs.items():
            if current_time - last_seen > 30:
                stale_modules.append(module)
        
        # Remove stale modules
        for module in stale_modules:
            if module in self.active_modules:
                self.active_modules.remove(module)
            self._module_watchdogs.pop(module, None)
            self.log.warning(f"Module {module} appears to be stale, unregistered")
    
    def _reinitialize_simulator(self):
        """Attempt to recover a damaged simulator instance."""
        try:
            # Save important settings
            saved_config = copy.deepcopy(self.config)
            saved_modules = copy.copy(self.active_modules)
            
            # Reinitialize
            self._initialized = False
            self._initialize_simulator()
            
            # Restore module registrations
            self.active_modules = saved_modules
            
            self.log.info("Successfully reinitialized simulator")
            return True
        except Exception as e:
            self.log.error(f"Failed to reinitialize simulator: {str(e)}")
            return False
    
    def _setup_fallback_simulator(self):
        """Set up a minimal fallback simulator for graceful degradation."""
        # Create a minimal stand-in with the same API but simplified behavior
        class MinimalNVModel:
            def get_fluorescence(self):
                return 1e5
                
            def apply_microwave(self, *args, **kwargs):
                pass
                
            def apply_laser(self, *args, **kwargs):
                pass
                
            def set_magnetic_field(self, *args, **kwargs):
                pass
                
            def reset_state(self):
                pass
                
            def simulate_odmr(self, f_min, f_max, n_points, mw_power):
                # Return a simple ODMR spectrum with two dips
                frequencies = np.linspace(f_min, f_max, n_points)
                signal = np.ones(n_points)
                
                # Create dips at typical NV frequencies
                center = 2.87e9
                width = 5e6
                depth = 0.3
                signal -= depth * width**2 / ((frequencies - center - 5e6)**2 + width**2)
                signal -= depth * width**2 / ((frequencies - center + 5e6)**2 + width**2)
                
                # Add noise
                signal += np.random.normal(0, 0.01, n_points)
                
                class Result:
                    pass
                result = Result()
                result.frequencies = frequencies
                result.signal = signal
                return result
                
            def simulate_rabi(self, t_max, n_points, mw_power, mw_frequency):
                # Return a simple damped oscillation
                times = np.linspace(0, t_max, n_points)
                signal = 1 - 0.3 * np.sin(2 * np.pi * 5e6 * times)**2 * np.exp(-times/5e-6)
                
                class Result:
                    pass
                result = Result()
                result.times = times
                result.signal = signal * 1e5
                return result
        
        self.nv_model = MinimalNVModel()
        self.log.warning("Using minimal fallback simulator")
    
    def register_module(self, module_name):
        """Register a module as active with the simulator."""
        with self._thread_lock:
            # Ensure initialization
            if not hasattr(self, '_initialized') or not self._initialized:
                self._init_on_demand()
                
            self.active_modules.add(module_name)
            self._module_watchdogs[module_name] = time.time()
            self.log.debug(f"Module {module_name} registered with simulator")
    
    def unregister_module(self, module_name):
        """Unregister a module from the simulator."""
        with self._thread_lock:
            if module_name in self.active_modules:
                self.active_modules.remove(module_name)
                self._module_watchdogs.pop(module_name, None)
                self.log.debug(f"Module {module_name} unregistered from simulator")
            
            # If no more active modules, consider shutting down
            if len(self.active_modules) == 0:
                self._consider_shutdown()
    
    def _init_on_demand(self):
        """Handle lazy initialization if needed."""
        if not hasattr(self, '_initialized') or not self._initialized:
            self.log.info("On-demand initialization of simulator")
            try:
                self.__init__()
            except Exception as e:
                self.log.error(f"On-demand initialization failed: {str(e)}")
                # Set up minimal functionality
                self._setup_fallback_simulator()
                self._initialized = True
    
    def _consider_shutdown(self):
        """Consider shutting down simulator resources when not in use."""
        if len(self.active_modules) == 0:
            self.log.info("No active modules, simulator resources can be released")
            # For now just log, in the future could release more resources
    
    def shutdown(self):
        """Explicitly shut down the simulator manager."""
        with self._thread_lock:
            self.log.info("Shutting down simulator manager")
            
            # Stop health monitoring
            if hasattr(self, '_health_timer') and self._health_timer is not None:
                try:
                    self._health_timer.stop()
                except:
                    pass
                
            # Set state
            self._is_running = False
            
            # Mark for garbage collection
            self._initialized = False
            
            # Clear registrations
            self.active_modules.clear()
            
            # Class-level cleanup
            SimulatorManager._instance = None
    
    def is_active(self):
        """Check if the simulator is currently active and healthy."""
        if not hasattr(self, '_initialized'):
            return False
            
        return self._initialized and self._is_running and len(self.active_modules) > 0
    
    def ping(self, module_name=None):
        """
        Update the watchdog timer for a module, or check simulator health.
        
        @param str module_name: Optional name of module to update watchdog
        @return bool: True if simulator is healthy
        """
        with self._thread_lock:
            # Update module watchdog if provided
            if module_name is not None and module_name in self.active_modules:
                self._module_watchdogs[module_name] = time.time()
            
            # Return health status
            if hasattr(self, 'nv_model') and self.nv_model is not None:
                return True
            return False
    
    # ===== Safe method wrapper =====
    
    def _safe_call(self, method_name, *args, **kwargs):
        """
        Safely call a method with standardized error handling.
        
        @param str method_name: Name of method to call
        @param *args, **kwargs: Arguments to pass to method
        @return: Result of method call or None if failed
        """
        with self._thread_lock:
            # Ensure we're initialized
            self._init_on_demand()
            
            try:
                # Get the method and call it
                if not hasattr(self.nv_model, method_name):
                    self.log.error(f"Method {method_name} not found in simulator")
                    return None
                    
                method = getattr(self.nv_model, method_name)
                return method(*args, **kwargs)
            except Exception as e:
                self.log.error(f"Error calling {method_name}: {str(e)}")
                return None
    
    # ===== Core simulator access methods =====
    
    def reset_state(self):
        """Reset the NV center state."""
        with self._thread_lock:
            try:
                if hasattr(self, 'nv_model') and self.nv_model is not None:
                    self.nv_model.reset_state()
                    self.log.debug("NV state reset")
                    return True
            except Exception as e:
                self.log.error(f"Failed to reset NV state: {str(e)}")
            return False
    
    def apply_magnetic_field(self, field_vector):
        """
        Set the magnetic field vector.
        
        @param field_vector: [Bx, By, Bz] in Gauss
        @return bool: Success or failure
        """
        return self._safe_call('set_magnetic_field', field_vector)
    
    def apply_laser(self, power, on=True):
        """
        Control the laser for optical excitation.
        
        @param power: Laser power in normalized units (0.0-1.0)
        @param on: Bool whether laser is on/off
        @return bool: Success or failure
        """
        with self._thread_lock:
            try:
                self.nv_model.apply_laser(power, on)
                self.log.debug(f"Laser {'on' if on else 'off'} with power {power}")
                return True
            except Exception as e:
                self.log.error(f"Failed to apply laser: {str(e)}")
                return False
    
    def apply_microwave(self, frequency, power_dbm, on=True):
        """
        Control the microwave excitation.
        
        @param frequency: Microwave frequency in Hz
        @param power_dbm: Microwave power in dBm
        @param on: Bool whether microwave is on/off
        @return bool: Success or failure
        """
        with self._thread_lock:
            try:
                self.nv_model.apply_microwave(frequency, power_dbm, on)
                self.log.debug(f"Microwave {'on' if on else 'off'} at {frequency/1e6:.3f} MHz, {power_dbm} dBm")
                return True
            except Exception as e:
                self.log.error(f"Failed to apply microwave: {str(e)}")
                return False
    
    def get_fluorescence(self):
        """
        Get the current fluorescence signal.
        
        @return float: Fluorescence count rate in counts/s
        """
        with self._thread_lock:
            try:
                value = self.nv_model.get_fluorescence()
                return value
            except Exception as e:
                self.log.error(f"Failed to get fluorescence: {str(e)}")
                return 1e5  # Default fluorescence level
    
    def evolve(self, duration):
        """
        Evolve the quantum state for specified duration.
        
        @param duration: Time to evolve in seconds
        @return bool: Success or failure
        """
        return self._safe_call('evolve', duration)
    
    # ===== Fast Counter Interface Methods =====
    
    def generate_time_trace(self, bin_width_s, record_length_s, number_of_gates=0):
        """
        Generate a time-resolved fluorescence trace.
        
        @param bin_width_s: Bin width in seconds
        @param record_length_s: Total record length in seconds
        @param number_of_gates: Number of gates (0 for ungated mode)
        
        @return: Simulated time trace data
        """
        with self._thread_lock:
            try:
                # Calculate number of bins
                num_bins = int(record_length_s / bin_width_s)
                
                # Check if we're in gated mode
                if number_of_gates > 0:
                    # Gated mode (2D array)
                    trace = np.zeros((number_of_gates, num_bins))
                    
                    # For each gate, generate a fluorescence trace
                    for gate in range(number_of_gates):
                        # Get the current NV state fluorescence
                        fluorescence_level = self.get_fluorescence()
                        
                        # Generate decay pattern
                        time_bins = np.arange(num_bins) * bin_width_s
                        decay_trace = fluorescence_level * np.exp(-time_bins / 12e-9)  # 12 ns decay time
                        
                        # Add Poisson noise
                        background = 0.001 * np.ones(num_bins)
                        mean_counts = (decay_trace + background) * bin_width_s * 0.1  # 10% detection efficiency
                        noisy_trace = np.random.poisson(mean_counts)
                        
                        # Store in the trace array
                        trace[gate, :] = noisy_trace
                else:
                    # Ungated mode (1D array)
                    # Get fluorescence level
                    fluorescence_level = self.get_fluorescence()
                    
                    # Generate decay pattern
                    time_bins = np.arange(num_bins) * bin_width_s
                    decay_trace = fluorescence_level * np.exp(-time_bins / 12e-9)
                    
                    # Add Poisson noise
                    background = 0.001 * np.ones(num_bins)
                    mean_counts = (decay_trace + background) * bin_width_s * 0.1
                    trace = np.random.poisson(mean_counts)
                    
                return trace
                
            except Exception as e:
                self.log.error(f"Failed to generate time trace: {str(e)}")
                # Return zeros as fallback
                if number_of_gates > 0:
                    return np.zeros((number_of_gates, num_bins))
                else:
                    return np.zeros(num_bins)
    
    # ===== Microwave Interface Methods =====
    
    def simulate_odmr(self, f_min, f_max, n_points, mw_power=-10.0):
        """
        Simulate an ODMR experiment.
        
        @param f_min: Start frequency in Hz
        @param f_max: End frequency in Hz
        @param n_points: Number of frequency points
        @param mw_power: Microwave power in dBm
        
        @return: Dictionary with frequencies and signal
        """
        with self._thread_lock:
            try:
                # Use the simulator's ODMR simulation
                result = self.nv_model.simulate_odmr(f_min, f_max, n_points, mw_power)
                return {
                    'frequencies': result.frequencies,
                    'signal': result.signal
                }
            except Exception as e:
                self.log.error(f"Failed to simulate ODMR: {str(e)}")
                
                # Fallback to synthetic data
                frequencies = np.linspace(f_min, f_max, n_points)
                center = 2.87e9  # Zero-field splitting
                width = 5e6  # 5 MHz linewidth
                depth = 0.3  # 30% contrast
                signal = np.ones(n_points)
                signal -= depth * width**2 / ((frequencies - center)**2 + width**2)
                signal *= 1e5  # Scale to counts/s
                signal += np.random.normal(0, 0.01 * 1e5, n_points)  # Add noise
                
                return {
                    'frequencies': frequencies,
                    'signal': signal
                }
    
    def simulate_rabi(self, t_max, n_points, mw_power=0.0, mw_frequency=None):
        """
        Simulate a Rabi oscillation experiment.
        
        @param t_max: Maximum time in seconds
        @param n_points: Number of time points
        @param mw_power: Microwave power in dBm
        @param mw_frequency: Microwave frequency in Hz (or None for resonance)
        
        @return: Dictionary with times and signal
        """
        with self._thread_lock:
            try:
                # Use the simulator's Rabi simulation
                result = self.nv_model.simulate_rabi(t_max, n_points, mw_power, mw_frequency)
                return {
                    'times': result.times,
                    'signal': result.signal
                }
            except Exception as e:
                self.log.error(f"Failed to simulate Rabi: {str(e)}")
                
                # Fallback to synthetic data
                times = np.linspace(0, t_max, n_points)
                # Approx 5 MHz Rabi frequency
                signal = 1 - 0.3 * np.sin(2 * np.pi * 5e6 * times)**2 * np.exp(-times/5e-6)
                signal *= 1e5  # Scale to counts/s
                signal += np.random.normal(0, 0.01 * 1e5, n_points)  # Add noise
                
                return {
                    'times': times,
                    'signal': signal
                }
    
    # ===== Scanning Probe Interface Methods =====
    
    def set_position(self, x, y, z):
        """
        Set the position for scanning probe.
        
        @param x: X position in meters
        @param y: Y position in meters
        @param z: Z position in meters
        @return bool: Success or failure
        """
        with self._thread_lock:
            try:
                if self.confocal_simulator is not None:
                    self.confocal_simulator.set_position(x, y, z)
                    return True
                return False
            except Exception as e:
                self.log.error(f"Failed to set position: {str(e)}")
                return False
    
    def get_confocal_image(self, x_range, y_range, z_position, resolution):
        """
        Generate a confocal image based on current settings.
        
        @param x_range: (min, max) for x axis in meters
        @param y_range: (min, max) for y axis in meters
        @param z_position: Z position in meters
        @param resolution: Number of pixels per dimension
        
        @return: 2D array of confocal image data
        """
        with self._thread_lock:
            try:
                if self.confocal_simulator is not None:
                    # Use the actual confocal simulator
                    x_vals = np.linspace(x_range[0], x_range[1], resolution)
                    y_vals = np.linspace(y_range[0], y_range[1], resolution)
                    
                    # Create a grid and scan it
                    image = np.zeros((resolution, resolution))
                    
                    for i, y in enumerate(y_vals):
                        for j, x in enumerate(x_vals):
                            self.confocal_simulator.set_position(x, y, z_position)
                            image[i, j] = self.confocal_simulator.get_intensity()
                    
                    return image
                else:
                    # Generate synthetic data without confocal simulator
                    return self._generate_synthetic_confocal_image(x_range, y_range, resolution)
            except Exception as e:
                self.log.error(f"Failed to generate confocal image: {str(e)}")
                # Return empty image as fallback
                return self._generate_synthetic_confocal_image(x_range, y_range, resolution)
    
    def _generate_synthetic_confocal_image(self, x_range, y_range, resolution):
        """Generate a synthetic confocal image with random spots as fallback."""
        x_vals = np.linspace(x_range[0], x_range[1], resolution)
        y_vals = np.linspace(y_range[0], y_range[1], resolution)
        
        # Create a grid
        x_grid, y_grid = np.meshgrid(x_vals, y_vals)
        
        # Generate some random Gaussian spots
        num_spots = 10
        image = np.zeros((resolution, resolution))
        
        for _ in range(num_spots):
            # Random position
            x_pos = np.random.uniform(x_range[0], x_range[1])
            y_pos = np.random.uniform(y_range[0], y_range[1])
            
            # Random intensity and size
            intensity = np.random.uniform(0.5, 1.0)
            sigma = np.random.uniform(0.5e-6, 2e-6)
            
            # Add Gaussian spot
            image += intensity * np.exp(-((x_grid - x_pos)**2 + (y_grid - y_pos)**2) / (2 * sigma**2))
        
        # Add noise
        image += np.random.normal(0, 0.05, (resolution, resolution))
        image = np.clip(image, 0, None)
        
        return image
```

### Configuration Integration

The simulator manager is integrated into Qudi's configuration system with a comprehensive, hierarchical parameter structure:

```python
# In default.cfg
nv_simulator:
    module.Class: 'hardware.nv_simulator.simulator_manager.SimulatorManager'
    options:
        simulator:
            # Core NV parameters
            zero_field_splitting: 2.87e9  # Hz
            gyromagnetic_ratio: 28.025e9  # Hz/T
            t1: 5.0e-3  # T1 relaxation time (s)
            t2: 1.0e-5  # T2 dephasing time (s)
            optics: True
            nitrogen: False
            method: "qutip"
        
        # Environmental parameters
        magnetic_field: [0, 0, 500]  # Gauss
        temperature: 300.0  # Kelvin
        
        # Confocal parameters
        confocal:
            lattice:
                nv_density: 0.5  # NVs per cubic micron
                size: [50e-6, 50e-6, 50e-6]  # Sample size in meters
            laser:
                wavelength: 532e-9  # m
                numerical_aperture: 0.8
                power: 1.0  # mW
        
        # Nuclear environment parameters
        nuclear_environment:
            enabled: False
            c13_concentration: 0.011  # Natural abundance
            bath_size: 50  # Number of nuclear spins
        
        # Performance parameters
        performance:
            use_analytical_models: True  # Use analytical approximations for speed
            cache_results: True  # Cache resource-intensive calculations
            max_calculation_time: 1.0  # Maximum time (s) for heavy calculations
```

### Thread Safety and Error Handling

Thread safety is implemented with a comprehensive approach:

1. **Recursive Mutex**: Using Qudi's `RecursiveMutex` for thread safety
2. **Lock Hierarchy**: Consistent locking pattern to prevent deadlocks
3. **Safe Method Wrappers**: All simulator calls wrapped in try-except blocks
4. **Standardized Error Handling**: Consistent error reporting and fallbacks
5. **Method Timeouts**: Protection against long-running quantum simulations

Error handling mechanisms include:

1. **Robust Initialization**: Graceful handling of import and initialization errors
2. **Fallback Behaviors**: Simplified synthetic data when advanced simulation fails
3. **Health Monitoring**: Periodic health checks with automatic recovery
4. **Structured Logging**: Detailed error logging for diagnostics
5. **Safe Call Wrapper**: Standardized error handling for all simulator methods

### Module Registration and Health Monitoring

The simulator implements a robust registry system for tracking active modules:

```python
# In dummy modules
def on_activate(self):
    """Initialization performed during activation of the module."""
    # Get reference to simulator manager
    try:
        self._simulator = SimulatorManager()
        self._simulator.register_module(self.__class__.__name__)
        self.log.info("Connected to NV Simulator")
    except Exception as e:
        self._simulator = None
        self.log.warning(f"Could not connect to NV Simulator: {str(e)}. Using fallback behavior.")
    
def on_deactivate(self):
    """Cleanup performed during deactivation of the module."""
    # Unregister from simulator
    if hasattr(self, '_simulator') and self._simulator is not None:
        try:
            self._simulator.unregister_module(self.__class__.__name__)
            
            # Check if we're the last module
            if not self._simulator.active_modules:
                # Signal for resource cleanup
                self._simulator.shutdown()
        except Exception as e:
            self.log.error(f"Error during simulator unregistration: {str(e)}")
```

The simulator includes health monitoring to detect and recover from problems:

1. **Watchdog Timers**: Tracking module activity and detecting stale registrations
2. **Periodic Health Checks**: Regular verification of simulator functionality
3. **Self-Healing**: Automatic reinitialization of damaged simulator components
4. **Resource Management**: Proper cleanup when simulator is no longer needed
5. **Qt-Compatible Design**: Works correctly in Qudi's Qt-based environment

## Integration with Dummy Modules

### Example: Fast Counter Interface

```python
# Inside fast_counter_dummy.py
def get_data_trace(self):
    """Get the current timetrace of the fast counter."""
    # Include an artificial waiting time to simulate hardware
    time.sleep(0.1)
    
    # Generate trace using simulator if available
    if self._simulator is not None:
        try:
            # Update watchdog to show we're active
            self._simulator.ping(self.__class__.__name__)
            
            # Use simulator to generate time trace
            trace = self._simulator.generate_time_trace(
                bin_width_s=self._binwidth, 
                record_length_s=self._record_length_s,
                number_of_gates=self._number_of_gates
            )
            return trace, {'elapsed_sweeps': None, 'elapsed_time': None}
        except Exception as e:
            self.log.warning(f"Simulator trace generation failed: {str(e)}. Using fallback.")
    
    # Fallback to original behavior if simulator unavailable
    if self._count_data is None:
        try:
            self._count_data = np.loadtxt(self.trace_path, dtype='int64')
            if self._gated:
                self._count_data = self._count_data.transpose()
        except Exception as e:
            self.log.error(f"Failed to load trace file: {str(e)}")
            # Create empty trace
            num_bins = int(self._record_length_s / self._binwidth)
            if self._gated:
                self._count_data = np.zeros((self._number_of_gates, num_bins), dtype='int64')
            else:
                self._count_data = np.zeros(num_bins, dtype='int64')
    
    info_dict = {'elapsed_sweeps': None, 'elapsed_time': None}
    return self._count_data, info_dict
```

### Example: Microwave Interface

```python
# Inside microwave_dummy.py
def cw_on(self):
    """Switches on cw microwave output."""
    with self._thread_lock:
        if self.module_state() == 'idle':
            self.log.debug(f'Starting CW microwave output with {self._cw_frequency:.6e} Hz '
                          f'and {self._cw_power:.6f} dBm')
            
            # Apply microwave in simulator if available
            if self._simulator is not None:
                try:
                    self._simulator.ping(self.__class__.__name__)
                    success = self._simulator.apply_microwave(
                        frequency=self._cw_frequency,
                        power_dbm=self._cw_power,
                        on=True
                    )
                    if not success:
                        self.log.warning("Simulator reported failure in applying microwave")
                except Exception as e:
                    self.log.warning(f"Failed to apply microwave in simulator: {str(e)}")
            
            time.sleep(0.1)  # Reduced delay for responsiveness
            self._is_scanning = False
            self.module_state.lock()
        elif self._is_scanning:
            raise RuntimeError(
                'Unable to start microwave CW output. Frequency scanning in progress.'
            )
        else:
            self.log.debug('CW microwave output already running')
```

### Example: Scanning Probe Interface

```python
# Inside scanning_probe_dummy.py
def set_position(self, x=None, y=None, z=None):
    """Move the scanner to a specific position in absolute coordinates."""
    with self._thread_lock:
        # Update target position with provided values
        if x is not None:
            self._target_position['x'] = x
        if y is not None:
            self._target_position['y'] = y
        if z is not None:
            self._target_position['z'] = z
        
        # Update current position (immediate move for simulator)
        self._position = self._target_position.copy()
        
        # Apply to simulator if available
        if self._simulator is not None:
            try:
                # Update watchdog
                self._simulator.ping(self.__class__.__name__)
                
                # Update position in simulator
                success = self._simulator.set_position(
                    x=self._position['x'],
                    y=self._position['y'],
                    z=self._position['z']
                )
                if not success:
                    self.log.debug("Simulator position update failed, using local position")
            except Exception as e:
                self.log.warning(f"Failed to update position in simulator: {str(e)}")
        
        self.log.debug(f"Scanner position set to x={self._position['x']}, y={self._position['y']}, z={self._position['z']}")
        
        # Return the current position
        return self._position.copy()
```

## Implementation Timeline

1. **Package Structure Setup** (Week 1)
   - Create `nv_simulator` directory structure in Qudi
   - Set up basic imports and minimal functionality

2. **Core Simulator Integration** (Week 1-2)
   - Implement `simulator_manager.py` with singleton pattern
   - Set up thread safety and proper error handling
   - Create minimal integration tests

3. **Module Interfaces** (Week 2-3)
   - Implement Fast Counter interface methods
   - Implement Microwave interface methods
   - Implement Scanning Probe interface methods

4. **Dummy Module Adaptation** (Week 3)
   - Modify `fast_counter_dummy.py` to use simulator
   - Modify `microwave_dummy.py` to use simulator
   - Modify `scanning_probe_dummy.py` to use simulator

5. **Robustness and Health Monitoring** (Week 4)
   - Add health checks and watchdog timers
   - Implement fallback behaviors
   - Create comprehensive tests

6. **Documentation and Finalization** (Week 4)
   - Update documentation and comments
   - Optimize performance
   - Review and finalize

## Risks and Mitigations

1. **Risk**: Import path problems due to package structure
   **Mitigation**: Use Qudi-native package structure and imports

2. **Risk**: Package installation issues with external simulator
   **Mitigation**: Integrate simulator directly into Qudi's package hierarchy

3. **Risk**: Initialization order problems in singleton pattern
   **Mitigation**: Protect initialization with locks and clearly separate phases

4. **Risk**: Thread safety issues with shared simulator instance
   **Mitigation**: Use Qudi's `RecursiveMutex` consistently with proper lock hierarchy

5. **Risk**: Inconsistent error handling across modules
   **Mitigation**: Implement standardized `_safe_call` wrapper with consistent logging

6. **Risk**: Configuration conflicts between modules
   **Mitigation**: Implement clear configuration hierarchy with defined precedence

7. **Risk**: Missing shutdown mechanism causing resource leaks
   **Mitigation**: Add explicit `shutdown()` method and resource tracking

8. **Risk**: Stale module registrations causing memory leaks
   **Mitigation**: Implement watchdog timers to detect and clean up stale registrations

9. **Risk**: Damaged simulator state affecting all modules
   **Mitigation**: Add health monitoring with automatic recovery mechanisms

10. **Risk**: Qt event loop conflicts in threaded operations
    **Mitigation**: Use Qt-compatible timers and thread-safe signal patterns

11. **Risk**: Time-intensive quantum simulations blocking UI
    **Mitigation**: Implement timeouts for long operations and analytical fallbacks

12. **Risk**: Module-specific configuration conflicts
    **Mitigation**: Use module-specific configuration sections with clear namespace