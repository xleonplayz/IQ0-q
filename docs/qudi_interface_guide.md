# Qudi Hardware Interface for NV Center Simulator

This documentation describes how to use the Qudi hardware interface adapters for the NV center quantum simulator. These interfaces allow the NV simulator to be used as a hardware backend for Qudi experiments, particularly ODMR measurements.

## Overview

The NV center simulator provides an accurate quantum mechanical model of an NV center in diamond, including:
- Realistic ODMR spectra with proper resonance profiles
- Rabi oscillations with accurate frequency dependence
- T1 relaxation behavior with proper three-level dynamics
- T2 echo and dynamical decoupling sequences with realistic decoherence

The Qudi interface layer now allows this simulator to be used directly within the Qudi experimental framework, functioning as a drop-in replacement for actual hardware.

## Installation

1. Ensure the simulator code is properly installed including all dependencies
2. Add the `sim/src` directory to your Python path
3. Copy the `qudi_nv_simulator.cfg` configuration file to your Qudi config directory
4. Modify paths in the configuration as needed for your setup

## Available Interfaces

The implementation provides the following Qudi hardware interfaces:

### 1. MicrowaveInterface

The `NVSimulatorMicrowave` class implements Qudi's `MicrowaveInterface`, providing:
- CW microwave control with frequency and power settings
- Frequency scanning with both jump-list and equidistant sweep modes
- Proper state management and thread safety

### 2. FiniteSamplingInputInterface

The `NVSimulatorScanner` class implements Qudi's `FiniteSamplingInputInterface`, providing:
- Fluorescence detection synchronized with microwave control
- Configurable sampling rate and frame size
- Buffered acquisition with proper threading

### 3. Main Device Class

The `NVSimulatorDevice` is the central class that creates and manages the interfaces:
- Initializes the core simulator with configurable parameters
- Instantiates and connects all required interfaces
- Provides utilities for direct simulator control and configuration

## Configuration Options

The simulator can be configured with the following parameters:

```yaml
simulator_params:
    t1: 5.0e-3            # T1 relaxation time in seconds
    t2: 1.0e-5            # T2 dephasing time in seconds
    laser_power: 1.0      # laser power in mW
    zero_field_splitting: 2.87e9  # in Hz
    magnetic_field: [0, 0, 0.5e-3]  # B-field vector in Tesla
```

## Example Usage in Qudi

### ODMR Experiment

With the interfaces properly configured, you can run ODMR experiments in Qudi using the simulator:

1. Start Qudi with the NV simulator configuration
2. Start the ODMR logic and GUI modules
3. Configure the scan parameters in the ODMR GUI
4. Run the scan to observe the simulated ODMR spectrum

The simulator will accurately model the quantum response of an NV center to the microwave sweep, including:
- Proper resonance profiles at the expected transition frequencies
- Power-dependent broadening of resonance lines
- Realistic fluorescence contrast based on simulation parameters

### Direct Simulator Access

For more advanced control, you can also access the simulator directly:

```python
# Get the simulator device instance from Qudi
simulator_device = qudi_main.objectmanager.get('nv_simulator')

# Get the underlying simulator model
simulator = simulator_device.get_simulator_instance()

# Configure the simulator
simulator_device.configure_simulator(
    magnetic_field=[0, 0, 1e-3],  # 1 mT along z-axis
    laser_power=2.0               # 2 mW laser power
)

# Run a specific simulation
result = simulator_device.run_simulation('odmr', 
                                        f_min=2.8e9, 
                                        f_max=2.9e9, 
                                        n_points=101)
```

## Integration with Real Hardware

The simulator interfaces can also be used alongside real hardware components:

1. Use the simulator's microwave interface with a real detector
2. Use a real microwave source with the simulator's detector
3. Use the simulator for development and testing before deploying to real hardware

## Performance and Threading Considerations

- The simulator uses threading for non-blocking acquisitions
- All interfaces implement proper thread safety mechanisms
- Long simulations with complex quantum dynamics may impact performance

## Limitations

- The simulator does not model all hardware imperfections of real systems
- Timing synchronization is approximate and may not match hardware timing precision
- Some advanced features like arbitrary waveform generation are not fully supported

## Troubleshooting

Common issues:

1. **Module import errors** - Ensure the simulator code is in your Python path
2. **Configuration errors** - Check that all required parameters are set correctly
3. **Performance issues** - Reduce the quantum system complexity for faster simulations
4. **Unexpected resonances** - Verify the magnetic field and other physical parameters