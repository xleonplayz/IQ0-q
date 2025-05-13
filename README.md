# NV Center Quantum Simulator

A quantum-mechanical simulator for NV centers in diamond, built on SimOS.

## Structure

```
./
├── docs/          # Documentation
│   └── technical_stories/  # Implementation plans
├── src/           # Source code
│   ├── __init__.py
│   ├── model.py   # Core simulation model
│   └── sim/       # SimOS integration
└── tests/         # Test suite
    ├── __init__.py
    ├── test_model.py
    └── test_simos_integration.py
```

## Installation

For normal use:
```bash
pip install -e .
```

For Qudi integration, install using the full path:
```bash
# Linux/macOS
pip install -e /full/path/to/IQO-q/sim

# Windows
pip install -e C:\Path\to\IQO-q\sim
```

## Troubleshooting

If you see errors like:
```
ImportError: Could not import the NV simulator model. Error: No module named 'sim'
```

This means the simulator package is not in the Python path. Make sure you've installed it with the commands above.

## Basic Usage

```python
from src import PhysicalNVModel

# Create a model with quantum mechanical features
model = PhysicalNVModel(
    optics=True,           # Include optical levels
    nitrogen=False,        # No nitrogen nuclear spin
    method="qutip"         # Use QuTiP backend
)

# Set magnetic field (50 mT along z-axis)
model.set_magnetic_field([0, 0, 0.05])

# Run ODMR experiment
result = model.simulate_odmr(2.7e9, 2.9e9, 101)
print(result)

# Run Rabi oscillation experiment
result = model.simulate_rabi(t_max=1e-6, n_points=50, mw_power=-10.0)
print(result)
```

## Features

- Full quantum mechanical simulation using density matrices
- Proper Hamiltonian construction with SimOS quantum physics
- Accurate modeling of decoherence using collapse operators
- Magnetic field interaction with proper Zeeman effect
- Microwave and laser control 
- ODMR and Rabi simulations
- Thread-safe implementation

## Quantum Simulation Details

The simulator uses SimOS to implement a full quantum mechanical model:

- Density matrix formalism for mixed quantum states
- Lindblad master equation for open quantum system dynamics
- Includes optical levels (ground, excited, and shelving states)
- Models T1 relaxation and T2 dephasing processes
- Configurable microwave driving and laser excitation

## Testing

```bash
python -m unittest discover
```