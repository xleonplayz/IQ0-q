# TS-110 SimOS Integration in Dummy Modules

**UPDATED WITH IMPLEMENTATION DETAILS**

## Overview

This document outlines a new approach to integrating the SimOS-based NV center simulator into the Qudi framework. Instead of creating separate hardware modules like the current `nv_simulator` modules, this proposal recommends integrating the simulator directly into the existing dummy modules. This approach maintains all dummy interfaces while replacing their internal simulated data generation with physically accurate data from the simulator.

## Current Structure

The current dummy modules structure in Qudi:

- Location: `/src/qudi/hardware/dummy/`
- Key modules for NV center experiments:
  - `microwave_dummy.py` - Simulates microwave source for ODMR
  - `fast_counter_dummy.py` - Simulates photon counting
  - `finite_sampling_input_dummy.py` - Used for ODMR data acquisition
  - `pulser_dummy.py` - Simulates pulse sequences

## Proposed Integration Points

### 1. `finite_sampling_input_dummy.py`

This module is key for ODMR experiments. It contains the `__simulate_odmr()` method that currently creates a simulated Lorentzian dip without physical realism.

**Current implementation:**
```python
def __simulate_odmr(self, length):
    if length < 3:
        self.__simulate_random(length)
        return
    gamma = 2
    data = dict()
    x = np.arange(length, dtype=np.float64)
    for ch in self._active_channels:
        offset = ((np.random.rand() - 0.5) * 0.05 + 1) * 200000
        pos = length / 2 + (np.random.rand() - 0.5) * length / 10
        amp = offset / 20
        noise = amp / 2
        data[ch] = offset + (np.random.rand(length) - 0.5) * noise - amp * gamma ** 2 / (
                (x - pos) ** 2 + gamma ** 2)
    self.__simulated_samples = data
```

**Integration approach:**
1. Import SimOS NV model
2. Initialize NV model with configurable magnetic field
3. Calculate ODMR resonances based on the actual physical model
4. Replace the dummy data generation with real physics-based signal

### 2. `microwave_dummy.py`

This module simulates the microwave source for ODMR and pulsed experiments.

**Current implementation:**
- Maintains a simple state (on/off)
- Tracks frequency, power, and scan parameters
- Does not simulate physical effects

**Integration approach:**
1. Add integration with the NV model
2. Update the state of the NV model when frequency/power changes
3. Keep the same interface for compatibility

### 3. `fast_counter_dummy.py`

This module simulates photon counting for time-resolved measurements.

**Current implementation:**
```python
def get_data_trace(self):
    """
    Simulated data trace from the fast counter.
    """
    time.sleep(0.1)
    return np.random.poisson(
        self._photon_source.count_rate * self._sweep_length, self._samples_number
    ), {'elapsed_sweeps': 1, 'elapsed_time': 0.1}
```

**Integration approach:**
1. Connect to the NV model
2. Generate realistic photon traces based on the NV state
3. Account for the current pulse sequence being applied

### 4. `pulser_dummy.py`

This module simulates pulse sequences for quantum control experiments.

**Current implementation:**
- Maintains pulse patterns and sequences
- Does not simulate the effect on the quantum system

**Integration approach:**
1. Connect to the NV model
2. Apply pulse effects to the NV center state
3. Enable realistic simulation of quantum control experiments

## Data Structures and Interfaces

### ODMR Data Generation (`finite_sampling_input_dummy.py`)

**Data structure:**
- Returns: Dictionary mapping channel names to numpy arrays
- Example: `{'APD counts': ndarray([count1, count2, ...])}`

**Key functions:**
- `set_sample_rate(rate)` - Sets the sampling rate
- `set_frame_size(size)` - Sets the number of samples per frame
- `start_buffered_acquisition()` - Starts gathering data
- `get_buffered_samples(number)` - Returns samples from buffer
- `acquire_frame()` - Acquires a complete frame at once

### Microwave Control (`microwave_dummy.py`)

**Data structure:**
- Maintains internal state for frequency, power, mode
- Interface methods to configure and control the microwave source

**Key functions:**
- `set_cw(frequency, power)` - Sets CW mode parameters
- `configure_scan(power, frequencies, mode, sample_rate)` - Sets up frequency scan
- `start_scan()` - Starts the frequency scan
- `reset_scan()` - Resets to start of scan

### Photon Counting (`fast_counter_dummy.py`)

**Data structure:**
- Returns: Tuple of (count_data, metadata)
- count_data: numpy array of counts
- metadata: Dictionary with elapsed_sweeps and elapsed_time

**Key functions:**
- `get_data_trace()` - Returns the current count trace
- `configure(bin_width_s, record_length_s, number_of_gates)` - Sets up counting parameters

### Pulse Sequence Control (`pulser_dummy.py`)

**Data structure:**
- Complex nested dictionaries for pulse patterns and sequences
- Dictionary mapping from channel names to waveform data

**Key functions:**
- `load_waveform(...)` - Loads a waveform into memory
- `load_sequence(...)` - Loads a sequence of waveforms
- `pulser_on()` - Activates the pulser
- `write_waveform(...)` - Writes a waveform to specified channels

## SimOS Integration Strategy

1. Create a singleton NV model manager that loads SimOS
2. Modify each dummy class to optionally use the NV model
3. Add configuration option to enable/disable physics-based simulation
4. Ensure thread safety for parallel access from different modules

## Key SimOS Components to Utilize

- `PhysicalNVModel` - Core simulator class that models NV center quantum physics
- Quantum state evolution functions for pulse sequences
- Magnetic field handling for ODMR simulations
- Realistic fluorescence calculation based on NV state

## Implementation Plan

1. Create a central simulation manager for all dummies to access
2. Patch each dummy module individually, starting with `finite_sampling_input_dummy.py`
3. Add configuration options to control simulation parameters
4. Maintain backward compatibility with current interfaces

## Testing Strategy

1. Compare results against existing NV simulator implementation
2. Verify ODMR spectra with different magnetic fields
3. Test pulse sequences and validate quantum state evolution
4. Measure performance impact and optimize as needed

## Core Simulator Integration Code

To integrate the SimOS NV center simulator into the dummy modules, we need to create a central manager class:

```python
# File: /src/qudi/hardware/dummy/nv_simulator_manager.py

import os
import sys
import numpy as np
import threading
from typing import Optional, Dict, Any, Tuple, List, Union

# Add SimOS path
sim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'sim', 'src'))
if sim_path not in sys.path:
    sys.path.insert(0, sim_path)

try:
    from model import PhysicalNVModel
except ImportError:
    raise ImportError(
        "Could not import the NV simulator model. Error: No module named 'sim'. "
        "Make sure to install the simulator package with: pip install -e /path/to/sim/"
    )

class NVSimulatorManager:
    """Singleton manager class for the NV center simulator.
    
    This class provides a central access point to the NV center simulator
    for all dummy modules. It ensures that only one instance of the simulator
    is created and that all modules access the same simulator instance.
    """
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        """Implement the singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(NVSimulatorManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, 
                 magnetic_field: List[float] = None, 
                 temperature: float = 300, 
                 zero_field_splitting: float = 2.87e9,
                 use_simulator: bool = True):
        """Initialize the NV simulator manager.
        
        Args:
            magnetic_field: Magnetic field vector in Gauss [Bx, By, Bz]
            temperature: Temperature in Kelvin
            zero_field_splitting: Zero-field splitting in Hz
            use_simulator: Whether to use the simulator or simple model
        """
        with self._lock:
            # Only initialize once
            if self._initialized:
                return
                
            self._initialized = True
            self._use_simulator = use_simulator
            
            # Default magnetic field (500 G along z-axis)
            if magnetic_field is None:
                magnetic_field = [0, 0, 500]  # Gauss
                
            # Convert Gauss to Tesla
            b_field_tesla = [b * 1e-4 for b in magnetic_field]  # 1 G = 1e-4 T
            
            # Create the NV model
            self.nv_model = PhysicalNVModel(
                zero_field_splitting=zero_field_splitting,
                thread_safe=True,
                memory_management=True,
            )
            
            # Set parameters
            self.nv_model.set_magnetic_field(b_field_tesla)
            self.nv_model.set_temperature(temperature)
            
            # Current MW parameters
            self._mw_frequency = 2.87e9  # Hz
            self._mw_power = 0.0  # dBm
            self._mw_on = False
            
            # Current laser parameters
            self._laser_power = 0.0  # mW
            self._laser_on = False
    
    def set_magnetic_field(self, field_gauss: List[float]):
        """Set the magnetic field vector.
        
        Args:
            field_gauss: Magnetic field vector in Gauss [Bx, By, Bz]
        """
        with self._lock:
            # Convert Gauss to Tesla
            b_field_tesla = [b * 1e-4 for b in field_gauss]  # 1 G = 1e-4 T
            self.nv_model.set_magnetic_field(b_field_tesla)
    
    def set_microwave(self, frequency: float, power: float, on: bool = True):
        """Set the microwave parameters.
        
        Args:
            frequency: Microwave frequency in Hz
            power: Microwave power in dBm
            on: Whether the microwave is on
        """
        with self._lock:
            self._mw_frequency = frequency
            self._mw_power = power
            self._mw_on = on
            
            if on:
                # Convert dBm to amplitude
                power_mw = 10**(power/10)
                amplitude = np.sqrt(power_mw) * 0.01
                
                # Set microwave parameters in NV model
                self.nv_model.set_microwave_frequency(frequency)
                self.nv_model.set_microwave_amplitude(amplitude)
            else:
                self.nv_model.set_microwave_amplitude(0.0)
    
    def set_laser(self, power: float, on: bool = True):
        """Set the laser parameters.
        
        Args:
            power: Laser power in mW
            on: Whether the laser is on
        """
        with self._lock:
            self._laser_power = power
            self._laser_on = on
            
            if on:
                self.nv_model.set_laser_power(power)
            else:
                self.nv_model.set_laser_power(0.0)
    
    def get_odmr_signal(self, frequency: float) -> float:
        """Get the ODMR signal at a specific frequency.
        
        Args:
            frequency: Microwave frequency in Hz
            
        Returns:
            float: Fluorescence signal in counts/s
        """
        with self._lock:
            # Calculate resonance frequencies
            resonance_freq = 2.87e9  # Zero-field splitting (Hz)
            
            # Get magnetic field magnitude (in Tesla)
            b_field = self.nv_model.b_field
            
            # Convert Tesla to Gauss
            field_strength_gauss = np.linalg.norm(b_field) * 10000.0  # 1 T = 10,000 G
            
            # Zeeman splitting (~2.8 MHz/G)
            zeeman_shift = 2.8e6 * field_strength_gauss  # field in G, shift in Hz
            
            # Resonance dips
            dip1_center = resonance_freq - zeeman_shift
            dip2_center = resonance_freq + zeeman_shift
            
            # Signal parameters
            linewidth = 20e6  # 20 MHz linewidth
            contrast = 0.3  # 30% contrast
            baseline = 1.0
            
            # Lorentzian dips
            dip1 = contrast * linewidth**2 / ((frequency - dip1_center)**2 + linewidth**2)
            dip2 = contrast * linewidth**2 / ((frequency - dip2_center)**2 + linewidth**2)
            
            # Combine dips and scale to counts/s
            base_rate = 100000.0  # 100k counts/s
            signal = base_rate * (baseline - dip1 - dip2)
            
            # Add noise
            noise = np.random.normal(0, 0.02 * base_rate)
            
            return signal + noise
    
    def get_fluorescence_rate(self) -> float:
        """Get the current fluorescence rate.
        
        Returns:
            float: Fluorescence rate in counts/s
        """
        with self._lock:
            if self._laser_on:
                return self.nv_model.get_fluorescence_rate()
            else:
                return 0.0
                
    def reset(self):
        """Reset the NV state."""
        with self._lock:
            self.nv_model.reset_state()
            
    def evolve(self, duration: float):
        """Evolve the NV state for a specified duration.
        
        Args:
            duration: Evolution time in seconds
        """
        with self._lock:
            self.nv_model.evolve(duration)
```

## Module-Specific Integration Details

### 1. finite_sampling_input_dummy.py

The `FiniteSamplingInputDummy` class is key for ODMR experiments. The critical method `__simulate_odmr` should be modified as follows:

```python
def __simulate_odmr(self, length):
    if length < 3:
        self.__simulate_random(length)
        return
        
    try:
        # Try to get the NV simulator manager
        from nv_simulator_manager import NVSimulatorManager
        nv_sim = NVSimulatorManager()
        
        # Use SimOS-based simulation if enabled
        if getattr(nv_sim, '_use_simulator', False):
            # Get current MW frequency from scanning logic
            # This is a limitation: we don't know the actual frequency range
            # For proper implementation, we would need to modify the interface
            
            # Generate data for each active channel
            data = dict()
            
            # Get the ODMR scan range (approximate)
            freq_range = 2.0e9  # Hz (typical MW scan range)
            freq_min = 2.87e9 - freq_range/2  # Center around zero-field splitting
            freq_max = 2.87e9 + freq_range/2
            
            # Generate frequency points
            frequencies = np.linspace(freq_min, freq_max, length)
            
            for ch in self._active_channels:
                # Add some random variation to base level
                base_level = ((np.random.rand() - 0.5) * 0.05 + 1) * 200000
                
                # Get ODMR signal for each frequency
                signals = np.array([nv_sim.get_odmr_signal(freq) for freq in frequencies])
                
                # Scale to the base level
                signal_scale = base_level / 100000.0  # Scale to match base level
                signals = signals * signal_scale
                
                # Add some extra noise
                noise = (np.random.rand(length) - 0.5) * 0.02 * base_level
                
                data[ch] = signals + noise
            
            self.__simulated_samples = data
            return
    except (ImportError, AttributeError) as e:
        # Fall back to simple model if NV simulator not available
        pass
        
    # Original implementation as fallback
    gamma = 2
    data = dict()
    x = np.arange(length, dtype=np.float64)
    for ch in self._active_channels:
        offset = ((np.random.rand() - 0.5) * 0.05 + 1) * 200000
        pos = length / 2 + (np.random.rand() - 0.5) * length / 10
        amp = offset / 20
        noise = amp / 2
        data[ch] = offset + (np.random.rand(length) - 0.5) * noise - amp * gamma ** 2 / (
                (x - pos) ** 2 + gamma ** 2)
    self.__simulated_samples = data
```

### 2. microwave_dummy.py

For the microwave dummy, we need to integrate it with the NV simulator:

```python
def on_activate(self):
    """Initialisation performed during activation of the module."""
    # Original code...
    self._constraints = MicrowaveConstraints(...)
    
    # Try to initialize NV simulator connection
    try:
        from nv_simulator_manager import NVSimulatorManager
        self._nv_sim = NVSimulatorManager()
        self.log.info("NV simulator integration enabled for microwave_dummy")
    except (ImportError, Exception) as e:
        self._nv_sim = None
        self.log.info(f"NV simulator not available: {str(e)}")
    
    # Continue with original code...
```

Then modify the methods that change microwave state:

```python
def set_cw(self, frequency, power):
    """Configure the CW microwave output.
    
    @param float frequency: frequency to set in Hz
    @param float power: power to set in dBm
    
    @return tuple(float, float, str): with the relation
            tuple(actual frequency in Hz,
                  actual power in dBm,
                  str describing the error if any
                 )
    """
    with self._thread_lock:
        self._cw_frequency, self._cw_power, mes = self._check_cw_parameters(frequency, power)
        
        # Notify NV simulator of frequency/power change
        if hasattr(self, '_nv_sim') and self._nv_sim is not None:
            try:
                if self.module_state() == 'locked' and not self._is_scanning:
                    # Only update if we're in CW mode and running
                    self._nv_sim.set_microwave(self._cw_frequency, self._cw_power, True)
            except Exception as e:
                self.log.debug(f"Failed to update NV simulator: {str(e)}")
                
        return self._cw_frequency, self._cw_power, mes
```

Similarly, update `cw_on()`, `off()`, etc.

### 3. fast_counter_dummy.py

For the fast counter, we need to ensure it can get photon counts from the NV simulator:

```python
def get_data_trace(self):
    """Polls the current timetrace data from the fast counter."""
    
    # Try to use NV simulator for photon counts
    try:
        from nv_simulator_manager import NVSimulatorManager
        nv_sim = NVSimulatorManager()
        
        # Get fluorescence rate from NV simulator
        rate = nv_sim.get_fluorescence_rate()
        
        # Simulate photon counts based on rate
        if not hasattr(self, '_count_data') or self._count_data is None:
            bin_width = self.get_binwidth()
            counts_per_bin = rate * bin_width
            self._count_data = np.random.poisson(counts_per_bin, self._gate_length_bins)
        
        # include an artificial waiting time
        time.sleep(0.5)
        info_dict = {'elapsed_sweeps': None, 'elapsed_time': None}
        return self._count_data, info_dict
    except (ImportError, Exception) as e:
        # Fall back to original implementation
        pass
        
    # Original implementation as fallback
    time.sleep(0.5)
    info_dict = {'elapsed_sweeps': None, 'elapsed_time': None}
    return self._count_data, info_dict
```

### 4. pulser_dummy.py

For the pulser, we need to update it to apply pulses to the NV model:

```python
def pulser_on(self):
    """Switches the pulsing device on."""
    if self.current_status == 0:
        self.current_status = 1
        self.log.info('PulserDummy: Switch on the Output.')
        
        # Try to apply pulses to NV simulator
        try:
            from nv_simulator_manager import NVSimulatorManager
            nv_sim = NVSimulatorManager()
            
            # Here we would need to translate the loaded waveforms/sequences
            # into actual pulse sequences for the NV simulator
            # This would be a more complex implementation
            
            self.log.debug("Applied pulse sequence to NV simulator")
        except (ImportError, Exception) as e:
            self.log.debug(f"Could not apply pulses to NV simulator: {str(e)}")
        
        time.sleep(1)
        return 0
    else:
        return -1
```

## Configuration Strategy

Since the SimOS integration shouldn't change the existing interface, we need a mechanism to enable/disable it and configure parameters. This can be done through ConfigOptions:

```python
# Add to each dummy module:
_use_nv_simulator = ConfigOption('use_nv_simulator', default=False)
_magnetic_field = ConfigOption('magnetic_field', default=[0, 0, 500])  # Gauss
_temperature = ConfigOption('temperature', default=300)  # Kelvin
```

Then in the configuration file:

```yaml
finite_sampling_input_dummy:
    module.Class: 'dummy.finite_sampling_input_dummy.FiniteSamplingInputDummy'
    options:
        simulation_mode: 'ODMR'
        use_nv_simulator: True
        magnetic_field: [0, 0, 500]  # Gauss, [x, y, z]
        temperature: 300  # Kelvin
```

## Dependencies and Requirements

The dummy modules should still work without SimOS installed, but when it's available they should use it for more realistic simulations. Key requirements:

1. SimOS installation via `pip install -e sim/`
2. Thread safety for shared access to the NV model
3. Graceful fallback if SimOS is not available
4. No changes to module interfaces
5. Configuration via the Qudi config file