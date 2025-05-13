# TS-111: Hardware Fallback Mechanisms

## Summary
This technical story documents the fallback mechanisms implemented in the NV simulator hardware modules, ensuring robustness when primary components are unavailable or fail.

## Implementation Details

### Model.py Fallbacks
The core `PhysicalNVModel` class in `sim/src/model.py` implements multiple fallback mechanisms to ensure simulation can continue even without SimOS:

1. **State Reset Fallback** (Lines 182-190)
   ```python
   # Fallback to simplified representation
   self.state = np.zeros(3)
   self.state[0] = 1.0  # |ms=0⟩ state
   ```
   When SimOS quantum state initialization fails, this fallback ensures the system still starts in the ground state using a simplified representation.

2. **Quantum Evolution Fallback** (Lines 431-489)
   ```python
   # Fallback to simplified model
   self._evolve_fallback(duration)
   ```
   The `_evolve_fallback` method provides a complete alternative implementation for time evolution when SimOS is unavailable, including:
   - Rabi oscillations with microwave drive
   - T1 relaxation effects toward thermal equilibrium
   - State normalization

3. **Fluorescence Calculation Fallback** (Lines 530-549)
   ```python
   # Fallback to simplified model
   ms0_pop = self.state[0]
   # ms=0 has higher fluorescence than ms=±1
   base_fluorescence = 1e5  # counts/s
   contrast = 0.3  # 30% contrast
   # Scale by collection efficiency
   return base_fluorescence * self._collection_efficiency * (1.0 - contrast * (1.0 - ms0_pop))
   ```
   Provides a simplified fluorescence calculation based on ms=0 population when SimOS is unavailable.

4. **Population Calculation Fallback** (Lines 576-583)
   ```python
   # Basic implementation for 3-level system
   return {
       'ms0': self.state[0],
       'ms+1': self.state[1],
       'ms-1': self.state[2]
   }
   ```
   Returns a basic population distribution using the simplified state representation.

5. **Analytical Experiment Models**
   The following analytical fallbacks implement physically accurate models when quantum simulation isn't possible:
   
   - **ODMR Simulation** (Lines 691-742): `_simulate_odmr_analytical`
     Implements Lorentzian dips at expected resonance frequencies based on magnetic field and zero-field splitting.

   - **Rabi Oscillation Simulation** (Lines 847-900): `_simulate_rabi_analytical`
     Models Rabi oscillations with detuning effects and T2 damping.

   - **T1 Relaxation Simulation** (Lines 962-987): `_simulate_t1_analytical`
     Models exponential relaxation from ms=±1 to thermal equilibrium.

   - **Ramsey Simulation** (Lines 1001-1002): `_simulate_ramsey_analytical`
     Performs analytical calculation of Ramsey free induction decay.

   - **Hahn Echo Simulation** (Lines 1016-1017): `_simulate_echo_analytical`
     Implements analytical model for spin echo measurements.

### Hardware Module Fallbacks

1. **QudiFacade Import Fallbacks** (Lines 51-106 in `qudi_facade.py`)
   ```python
   # Fallback to direct import checks
   print(f"Wrapper module not found at {wrapper_path}, falling back to direct imports")
   
   # 1. First check if it's in the dummy/sim/src directory
   dummy_sim_path = os.path.join(current_dir, '..', 'dummy', 'sim', 'src')
   # [additional path setting logic...]
   ```
   Implements a multi-stage fallback system for finding and importing the required model classes:
   - First checks for wrapper module
   - Tries dummy/sim/src directory
   - Checks base directory path
   - Adds sim directory for module imports
   - Finally checks local directory

2. **Finite Sampler Connection Fallback** (Lines 103-116 in `finite_sampler.py`)
   ```python
   # Try direct import as fallback
   try:
       from qudi_facade import QudiFacade
       self._qudi_facade = QudiFacade()
       self.log.info('Using direct QudiFacade instantiation as fallback')
       
       # Initialize data buffer
       self._data_buffer = np.zeros((len(self._active_channels), self._current_frame_size))
       
       self.log.info('NV Simulator Finite Sampler initialized with fallback')
   except Exception as e2:
       self.log.error(f"Fallback also failed: {str(e2)}")
       raise
   ```
   When connector-based initialization fails, directly instantiates QudiFacade for sampling operations.

3. **Fast Counter Connection Fallback** (Lines 87-93 in `fast_counter.py`)
   ```python
   except Exception as e:
       self.log.error(f"Failed to get QudiFacade from connector: {str(e)}")
       # Fallback to direct instantiation
       from qudi_facade import QudiFacade
       self._qudi_facade = QudiFacade()
       self.log.info("Using directly instantiated QudiFacade as fallback")
   ```
   Ensures the fast counter can operate by directly instantiating QudiFacade when connector fails.

4. **Pulser Connection Fallback** (Lines 132-136 in `pulser.py`)
   ```python
   except Exception as e:
       self.log.error(f"Failed to get QudiFacade from connector: {str(e)}")
       # Fallback to direct instantiation
       from qudi_facade import QudiFacade
       self._qudi_facade = QudiFacade()
       self.log.info("Using directly instantiated QudiFacade as fallback")
   ```
   Enables the pulser module to operate independently of the connector system when needed.

## Benefits and Impact

1. **Robustness**: The simulator continues to function even when components are missing or fail.

2. **Graceful Degradation**: Rather than failing completely, the system falls back to simplified but physically accurate models.

3. **Modularity**: Each component can operate independently when needed, improving isolation and testability.

4. **Physics Accuracy**: Even the fallback mechanisms implement physically accurate models based on NV center physics, maintaining scientific validity.

5. **Error Handling**: Comprehensive error handling with detailed logging helps diagnose issues while maintaining operation.

## Usage Examples

When SimOS is unavailable but NV simulation is needed:

```python
from model import PhysicalNVModel

# This will automatically use fallback mechanisms if SimOS is not available
model = PhysicalNVModel()

# Run an ODMR experiment - will use analytical model if quantum simulation fails
result = model.simulate_odmr(2.8e9, 2.9e9, 100)

# Examine results
print(f"Resonance frequencies: {result.resonances}")
```

## Testing Considerations

To validate fallback mechanisms:

1. Test with SimOS unavailable to verify analytical models produce physically accurate results
2. Validate consistency between quantum and analytical results where both are available
3. Ensure error messages are clear and help diagnose the root cause
4. Test connector failures to verify direct instantiation works correctly
5. Verify that all experiments (ODMR, Rabi, T1, etc.) function in fallback mode

## Limitations

1. Simplified models lack quantum coherence effects beyond basic T1/T2 relaxation
2. No nuclear spin interactions in fallback models
3. Some complex pulse sequences may not be fully supported in fallback mode
4. Direct instantiation bypasses the connector system, potentially missing configuration options

## Conclusion

The implemented fallback mechanisms significantly improve the robustness of the NV simulator, ensuring it can operate under various failure conditions while maintaining physical accuracy. This approach balances the desire for high-fidelity quantum simulation with practical robustness requirements.