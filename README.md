# qudi-iqo-modules with NV Center Simulator
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

---
A collection of qudi measurement modules originally developed for experiments on color centers in 
semiconductor materials, now enhanced with an integrated NV center simulator.

## NV Center Simulator Features

The NV center simulator provides realistic quantum simulations of NV centers in diamond:

- Physically accurate simulation of quantum dynamics
- ODMR, Rabi, T1, T2 experiments
- Confocal microscopy simulation
- Pulse sequence support for quantum control
- Direct replacement for hardware modules

## Using the Simulator

### Standalone Mode

Run simulations without the Qudi framework:

```
python -m src.qudi.hardware.nv_simulator
```

Optional arguments:
- `--odmr`: Run only the ODMR simulation
- `--rabi`: Run only the Rabi oscillation simulation
- `--confocal`: Run only the confocal scan simulation
- `--all`: Run all simulations (default)

### With Qudi Framework

1. Use the provided configuration file:
   ```
   cp src/qudi/hardware/nv_simulator/nv_simulator.cfg /path/to/your/qudi/config/
   ```

2. Launch Qudi with this configuration:
   ```
   python -m qudi -c /path/to/your/qudi/config/nv_simulator.cfg
   ```

For more details, see [NV Simulator Documentation](/src/qudi/hardware/nv_simulator/README.md).


## Installation
For installation instructions please refer to our
[iqo-modules installation guide](https://github.com/Ulm-IQO/qudi-iqo-modules/blob/main/docs/installation_guide.md).


## More information
The best starting point for further researching the qudi documentation is the [readme file](https://github.com/Ulm-IQO/qudi-core) of the qudi-core repo.

## Forum
For questions concerning qudi or the iqo-modules, there is a [forum](https://github.com/Ulm-IQO/qudi-core/discussions) to discuss with the qudi community. Feel free to ask!
If you found a bug and located it already, please note GitHub's [issue tracking](https://github.com/Ulm-IQO/qudi-iqo-modules/issues) feature.

## Copyright
Check [AUTHORS.md](AUTHORS.md) for a list of authors and the git history for their individual
contributions.
