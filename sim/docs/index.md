# NV Center Simulator

A simple Python simulator for NV centers in diamond.

## Overview

This package provides tools to simulate the quantum behavior of NV centers in diamond, including:

- Quantum state evolution
- Interaction with magnetic fields
- Microwave and optical control
- Common experiments (ODMR, Rabi oscillations)

## Installation

```bash
pip install -e .
```

## Basic Usage

```python
from src import PhysicalNVModel

# Create a model
model = PhysicalNVModel()

# Set magnetic field
model.set_magnetic_field([0, 0, 0.05])  # 50 mT

# Run ODMR experiment
result = model.simulate_odmr(2.7e9, 2.9e9, 101, -10.0)
print(result)
```

## Available Experiments

- **ODMR**: Optically Detected Magnetic Resonance for determining energy level splitting
- **Rabi**: Rabi oscillations between energy levels

## Testing

Run the tests with:

```bash
python -m unittest discover
```