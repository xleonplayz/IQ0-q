# NV Center Simulator for Qudi

This module provides a comprehensive NV center simulator that integrates with the Qudi framework. It allows for realistic quantum simulations of NV centers in diamond, including ODMR experiments, confocal microscopy, and pulsed experiments.

## Features

- **Realistic NV Center Physics**: Simulates NV center quantum dynamics, including spin states, Zeeman interactions, and hyperfine coupling
- **Optical Properties**: Models the optical cycle of NV centers, including fluorescence emission
- **Confocal Microscopy**: Simulates 3D confocal scanning with position-dependent fluorescence
- **Pulse Sequences**: Supports complex pulse sequences for quantum control
- **Qudi Integration**: Seamlessly replaces dummy modules in Qudi with physically accurate simulations

## Modules

- **QudiFacade** (`qudi_facade.py`): Central manager for all simulator resources
- **NVSimMicrowaveDevice** (`microwave_device.py`): Microwave source for ODMR experiments
- **NVSimFastCounter** (`fast_counter.py`): Photon counter for time-resolved measurements
- **NVSimPulser** (`pulser.py`): Pulse generator for quantum control experiments
- **NVSimScanningProbe** (`scanning_probe.py`): Scanner for confocal microscopy
- **NVSimLaser** (`laser.py`): Laser controller for optical excitation

## Installation

1. Ensure the simulator core (`/sim/` directory) is properly installed
2. Copy the `nv_simulator` directory to your Qudi installation under `hardware/`
3. Use the provided `nv_simulator.cfg` as a base for your configuration

## Configuration

The NV simulator can be configured through the Qudi config file. Parameters include:

- Magnetic field strength and direction
- Sample properties (NV density, etc.)
- Optical parameters (laser power, wavelength)
- Scanning parameters (position ranges, resolution)

See the `nv_simulator.cfg` file for a complete example configuration.

## Usage

To use the NV simulator:

1. Load Qudi with the provided configuration file: `python -m qudi -c /path/to/nv_simulator.cfg`
2. The simulator modules will be used instead of the corresponding hardware modules
3. Run experiments as you would with real hardware

## Examples

### ODMR Experiment

The NV simulator accurately reproduces ODMR spectra with the expected features:

- Zero-field splitting at 2.87 GHz
- Zeeman splitting dependent on magnetic field
- Hyperfine interaction features
- Realistic noise and signal levels

### Confocal Imaging

The simulator creates realistic confocal images:

- Position-dependent fluorescence based on NV center distribution
- 3D scanning capability
- Point spread function effects

### Pulsed Experiments

Supports various pulse sequences:

- Rabi oscillations
- Ramsey interferometry
- Spin echo
- Dynamical decoupling sequences

## Architecture

The simulator is built on the PhysicalNVModel class that provides the core quantum simulation. This model is then wrapped by various adapter classes that implement the Qudi hardware interfaces.

## License

This software is released under the GNU Lesser General Public License v3.0.

## Contact

For questions and support, please contact the development team or open an issue on the project repository.