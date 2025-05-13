"""
Test visualization for Rabi oscillation experiment.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import PhysicalNVModel

# Create a model with quantum mechanical features
model = PhysicalNVModel(
    zero_field_splitting=2.87e9,  # Hz
    gyromagnetic_ratio=28.0e9,    # Hz/T
    t1=5e-3,                      # s (5 ms)
    t2=500e-6,                    # s (500 µs)
    t2_star=1e-6,                 # s (1 µs)
    optics=True,                  # Include optical levels
    nitrogen=False,               # No nitrogen nuclear spin
    method="qutip"                # Use QuTiP backend
)

# Set a small magnetic field
model.set_magnetic_field([0, 0, 2e-4])  # 0.2 mT along z

# Run Rabi experiment
print("\nRunning Rabi oscillation experiment...")
rabi_result = model.simulate_rabi(
    t_max=0.5e-6,
    n_points=51,
    mw_power=0.0  # Higher power for faster oscillations
)
print(rabi_result)

# Plot the result
fig, ax = plt.subplots(figsize=(10, 6))
rabi_result.plot(ax=ax)
plt.tight_layout()
plt.savefig("rabi_test.png", dpi=150)
print("\nPlot saved as rabi_test.png")