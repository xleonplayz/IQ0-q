# NV Center Simulator - Qudi Interface

This package provides a complete integration between the NV center simulator and Qudi.
It implements various hardware interfaces that allow Qudi to use the NV simulator
as a complete virtual replacement for real NV center hardware.

## Features

- **Complete Hardware Integration**: Implements all major Qudi hardware interfaces for NV center experiments
- **Physically Accurate Simulations**: Uses the NV center quantum model to produce realistic experimental data
- **Plug-and-Play Configuration**: Simple configuration file to set up a complete virtual system
- **Experiment Mode Support**: Pre-built experiment modes for common measurements (ODMR, Rabi, etc.)
- **Confocal Scanning**: Simulated 3D scanning with realistic point spread function

## Available Hardware Interfaces

| Interface | Implementation | Description |
|-----------|---------------|-------------|
| Microwave | `NVSimMicrowaveDevice` | Controls microwave parameters for ODMR and other experiments |
| Fast Counter | `NVSimFastCounter` | Simulates photon counting with realistic statistics |
| Pulser | `NVSimPulser` | Pulse sequence generation for quantum control experiments |
| Scanning Probe | `NVSimScanningProbe` | Confocal scanning for spatial mapping |
| Laser | `NVSimLaser` | Laser control for optical excitation |

## Getting Started

1. Ensure you have both the NV simulator and Qudi installed
2. Copy `nv_simulator_qudi.cfg` to your Qudi configuration directory
3. Start Qudi using this configuration:
   ```
   qudi -c nv_simulator_qudi.cfg
   ```

## Core Components

### QudiFacade

The `QudiFacade` class serves as the central manager for all simulator resources. It initializes and maintains:

- The physical NV model with quantum mechanical properties
- The confocal simulator for spatial scanning
- The laser controller for optical excitation
- Environment parameters like magnetic field and temperature

All the hardware interfaces access the simulator through this facade to maintain a consistent state.

### Experiment Modes

The simulator includes pre-defined experiment modes for common measurements:

- ODMR for electron spin resonance
- Rabi for coherent control
- Ramsey for dephasing measurements
- Spin Echo for decoherence measurements
- T1 for relaxation measurements
- Custom sequences for advanced protocols

## Configuration Options

The simulator behavior can be customized through configuration parameters:

```yaml
nv_microwave:
    module.Class: 'qudi_interface.hardware.NVSimMicrowaveDevice'
    magnetic_field: [0, 0, 100]  # Magnetic field in Gauss [Bx, By, Bz]
    field_inhomogeneity: 0.01  # Relative field inhomogeneity
```

See the `nv_simulator_qudi.cfg` file for a complete list of configuration options.

## Examples

The `examples` directory contains example scripts that demonstrate how to:

- Run ODMR experiments with simulated NV centers
- Perform coherent control experiments (Rabi, Ramsey, etc.)
- Use confocal scanning to image simulated NV centers
- Create custom pulse sequences

## Development and Extension

The modular design allows easy extension with new features:

- Add new experiment modes by extending the `ExperimentMode` base class
- Enhance the physical model with additional quantum effects
- Implement additional Qudi interfaces as needed

## Troubleshooting

If you encounter issues with the simulation:

1. Check that your configuration paths are correct
2. Ensure the NV simulator core is properly initialized
3. Verify that appropriate parameters are set for the physical model
4. Check Qudi logs for any error messages

For more help, please submit an issue in the project repository.