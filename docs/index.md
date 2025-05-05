# SimOS NV Simulator Documentation

Welcome to the SimOS NV Simulator documentation. This project provides a comprehensive quantum simulator for Nitrogen-Vacancy (NV) centers in diamond.

*Developed by Leon Kaiser*

```{toctree}
:maxdepth: 2
:caption: Contents:

physical_model
api_reference
```

## Overview

The NV center simulator implements a full quantum mechanical model of NV centers, including:

- Ground and excited state dynamics
- Magnetic field interactions (Zeeman effect)
- Strain effects
- Hyperfine coupling to nitrogen nucleus
- Optical cycling and polarization
- Microwave control of spin states
- Quantum decoherence processes (T1, T2, T2*)

## Basic Usage

```python
from simos_nv_simulator.core.physical_model import PhysicalNVModel

# Create NV model with default parameters
nv = PhysicalNVModel()

# Set magnetic field (in Tesla)
nv.set_magnetic_field([0, 0, 0.005])

# Apply microwave drive
nv.apply_microwave(frequency=2.87e9, power=-20.0, on=True)

# Run simulation for 1 microsecond
nv.simulate(dt=1e-9, steps=1000)

# Get fluorescence count rate
counts = nv.get_fluorescence()
```

## Installation

```bash
# Install from source
git clone https://github.com/xleonplayz/IQ0-q.git
cd IQ0-q
pip install -e .
```

## Configuration

The simulator can be configured with various parameters:

```python
model = PhysicalNVModel({
    'zero_field_splitting': 2.87e9,  # Hz
    'gyromagnetic_ratio': 28.0e9,    # Hz/T
    'T1': 1e-3,                      # seconds
    'T2': 1e-6,                      # seconds
    'strain_e': 0.0,                 # Strain parameter E
    'strain_d': 0.0,                 # Strain parameter D
    'temperature': 298,              # Kelvin
})
```

## Development

See the [GitHub repository](https://github.com/xleonplayz/IQ0-q) for more information on contributing to this project.