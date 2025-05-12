# TS-104: Qudi Experiment Mode Integration

## Summary
Implement support for standard Qudi experiment modes in the NV center simulator, including ODMR, Rabi, Ramsey, Hahn Echo, and T1 measurements, with a consistent interface that can be used by Qudi's experimental logic modules.

## Motivation
To fully replace a hardware setup for testing and development, the simulator must support all common experimental modes used in Qudi. Currently, there's a mismatch between the simulator's experiment API and Qudi's expectations. This technical story bridges that gap by implementing standardized experiment modes that match Qudi's typical hardware interfaces.

## Description
This technical story enhances the simulator with standardized experiment modes that align with Qudi's experimental workflows. Each mode will be implemented using the SimOS quantum backend but exposed through Qudi-compatible interfaces to enable seamless integration with Qudi's logic modules.

### Requirements
1. Implement comprehensive experiment mode support:
   - ODMR (continuous wave and pulsed)
   - Rabi oscillations
   - Ramsey interferometry
   - Spin Echo (Hahn echo)
   - T1 relaxation
   - Custom pulse sequences

2. Create a consistent API that:
   - Matches Qudi's expected hardware interface signatures
   - Returns data in Qudi-compatible formats
   - Handles experimental parameters appropriately

3. Ensure accurate simulation results that:
   - Match expected theoretical behavior
   - Include realistic noise and imperfections
   - Scale properly with experimental parameters

4. Provide configuration options for each experiment mode

### Implementation Details
1. Create a dedicated module for experiment modes:
   ```
   src/
     qudi_interface/
       __init__.py
       experiments/
         __init__.py
         odmr.py
         rabi.py
         ramsey.py
         spin_echo.py
         t1.py
         custom_sequence.py
         utils.py
   ```

2. Create an abstract base class for experiment modes:
   ```python
   class ExperimentMode:
       def __init__(self, simulator):
           self._simulator = simulator
           self._params = {}
           
       def configure(self, **params):
           """Configure the experiment with parameters"""
           self._params.update(params)
           
       def run(self):
           """Run the experiment and return results"""
           raise NotImplementedError
           
       def get_default_parameters(self):
           """Return default parameters for this experiment"""
           raise NotImplementedError
   ```

3. Implement the ODMR experiment mode:
   ```python
   class ODMRMode(ExperimentMode):
       def __init__(self, simulator):
           super().__init__(simulator)
           self._default_params = {
               'freq_start': 2.82e9,  # Hz
               'freq_stop': 2.92e9,   # Hz
               'num_points': 101,     # Number of frequency points
               'power': -10.0,        # dBm
               'avg_count': 3,        # Number of averages
               'pulsed': False        # CW or pulsed ODMR
           }
           self._params = self._default_params.copy()
           
       def get_default_parameters(self):
           return self._default_params.copy()
           
       def run(self):
           """Run the ODMR experiment"""
           # Extract parameters
           f_min = self._params['freq_start']
           f_max = self._params['freq_stop']
           num_points = self._params['num_points']
           power = self._params['power']
           avg_count = self._params['avg_count']
           pulsed = self._params['pulsed']
           
           # Run appropriate simulation based on mode
           if not pulsed:
               # CW ODMR - use simulator's ODMR function
               result = self._simulator.simulate_odmr(f_min, f_max, num_points, power)
               
               # Average multiple runs if requested
               if avg_count > 1:
                   signal = result.signal
                   for _ in range(avg_count - 1):
                       result_new = self._simulator.simulate_odmr(f_min, f_max, num_points, power)
                       signal += result_new.signal
                   result.signal = signal / avg_count
           else:
               # Pulsed ODMR - use pulse sequence
               # ... implementation for pulsed ODMR ...
           
           # Format result for Qudi compatibility
           qudi_result = {
               'frequencies': result.frequencies,
               'odmr_signal': result.signal,
               'parameters': self._params.copy()
           }
           
           return qudi_result
   ```

4. Implement the Rabi experiment mode:
   ```python
   class RabiMode(ExperimentMode):
       def __init__(self, simulator):
           super().__init__(simulator)
           self._default_params = {
               'rabi_times': np.linspace(0, 500e-9, 51),  # s
               'mw_frequency': 2.87e9,                    # Hz
               'mw_power': -10.0,                         # dBm
               'avg_count': 3                             # Number of averages
           }
           self._params = self._default_params.copy()
           
       def get_default_parameters(self):
           return self._default_params.copy()
           
       def run(self):
           """Run the Rabi experiment"""
           # Extract parameters
           rabi_times = self._params['rabi_times']
           mw_freq = self._params['mw_frequency']
           mw_power = self._params['mw_power']
           avg_count = self._params['avg_count']
           
           # Create and run pulse sequence for Rabi
           signal = np.zeros(len(rabi_times))
           
           for i in range(avg_count):
               result = self._simulator.simulate_rabi(
                   t_max=max(rabi_times),
                   n_points=len(rabi_times),
                   mw_frequency=mw_freq,
                   mw_power=mw_power
               )
               signal += result.populations[:, 0]  # ms=0 population
           
           signal /= avg_count
           
           # Format result for Qudi compatibility
           qudi_result = {
               'times': rabi_times,
               'signal': signal,
               'parameters': self._params.copy()
           }
           
           return qudi_result
   ```

5. Implement Ramsey, Spin Echo, and T1 modes following similar patterns

6. Create utility functions for Qudi format conversion:
   ```python
   def convert_to_qudi_format(simulator_result, experiment_type):
       """Convert simulator results to Qudi-compatible format"""
       if experiment_type == 'odmr':
           # ... convert ODMR results ...
       elif experiment_type == 'rabi':
           # ... convert Rabi results ...
       # ... and so on for other experiments ...
   ```

7. Integrate experiment modes with the Qudi hardware interface:
   ```python
   class NVSimulatorDevice:
       def __init__(self):
           self._simulator = PhysicalNVModel()
           
           # Create experiment mode instances
           self._experiment_modes = {
               'odmr': ODMRMode(self._simulator),
               'rabi': RabiMode(self._simulator),
               'ramsey': RamseyMode(self._simulator),
               'spin_echo': SpinEchoMode(self._simulator),
               't1': T1Mode(self._simulator),
               'custom': CustomSequenceMode(self._simulator)
           }
           
           # Create Qudi interface adapters
           self.microwave = NVSimulatorMicrowave(self._simulator)
           self.scanner = NVSimulatorScanner(self._simulator)
           self.pulser = NVSimulatorPulser(self._simulator)
           
       def get_experiment_mode(self, mode_name):
           """Get an experiment mode by name"""
           return self._experiment_modes.get(mode_name)
           
       def run_experiment(self, mode_name, **params):
           """Run an experiment with the given parameters"""
           mode = self.get_experiment_mode(mode_name)
           if mode is None:
               raise ValueError(f"Unknown experiment mode: {mode_name}")
               
           mode.configure(**params)
           return mode.run()
   ```

### API Integration
The experiment modes will be accessible through the Qudi interface adapter, which will internally use these standardized experiment modes:

```python
# Example of using experiment modes through the NVSimulatorDevice
simulator_device = NVSimulatorDevice()

# Configure and run ODMR
odmr_params = {
    'freq_start': 2.83e9,
    'freq_stop': 2.91e9,
    'num_points': 201,
    'power': -5.0,
    'avg_count': 5
}
odmr_result = simulator_device.run_experiment('odmr', **odmr_params)

# Access through Qudi hardware interface
microwave = simulator_device.microwave
scanner = simulator_device.scanner

# Qudi ODMR logic would interact with these interfaces
microwave.configure_scan(
    odmr_params['power'],
    (odmr_params['freq_start'], odmr_params['freq_stop'], odmr_params['num_points']),
    SamplingOutputMode.EQUIDISTANT_SWEEP,
    100.0
)
microwave.start_scan()
data = scanner.acquire_frame()
```

### Testing Strategy
1. Create test cases for each experiment mode with known parameters
2. Compare simulation results with theoretical predictions
3. Validate Qudi interface compatibility
4. Test averaging and noise behavior
5. Compare results with experimental data when available

## Technical Risks
1. Performance impact of abstraction layers
2. Accuracy of simulated experimental results compared to real experiments
3. Interface compatibility issues with Qudi modules
4. Parameter mapping complexity between Qudi and simulator

## Effort Estimation
- Experiment mode base framework: 1 day
- ODMR mode implementation: 1 day
- Rabi mode implementation: 0.5 day
- Ramsey mode implementation: 0.5 day
- Spin Echo mode implementation: 0.5 day
- T1 mode implementation: 0.5 day
- Custom sequence mode: 1 day
- Integration and testing: 2 days
- Total: 7 days