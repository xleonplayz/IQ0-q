# SIMOS NV Simulator

A Python simulator for Nitrogen-Vacancy (NV) centers in diamond, integrating with SimOS.

## Overview

This package provides a high-performance simulator for NV center quantum dynamics, focusing on:

1. Full quantum mechanical evolution using SimOS for accurate simulations
2. High-level API for common quantum sensing and quantum information protocols
3. Thread-safe operation for integration with control hardware

## Installation

For development:

```bash
# Clone the repository
git clone https://github.com/your-org/simos-nv-simulator.git
cd simos-nv-simulator

# Install in development mode
pip install -e .
```

## Dependencies

- Python 3.9+
- NumPy
- SciPy
- SimOS (Simulation of Optically-addressable Spins)

The simulator uses a local version of SimOS from the `simos_repo` directory instead of requiring the SimOS package to be installed. This approach allows for easier development and modification of both the simulator and SimOS code.

### Testing Without SimOS

For basic testing without a fully functional SimOS installation, the test suite includes mock implementations that allow the basic functionality tests to pass without requiring the actual quantum simulation capabilities.

## Usage

Basic usage example:

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

## Testing

Run the tests with:

```bash
pytest
```

Or to run a specific test file:

```bash
pytest tests/core/test_physical_model.py
```

## License

[MIT License](LICENSE)