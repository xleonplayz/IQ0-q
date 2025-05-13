"""
Test visualization for T1 relaxation experiment.
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

# Run T1 experiment
print("\nRunning T1 relaxation experiment...")
t1_result = model.simulate_t1(
    t_max=15e-3,  # 15 ms, about 3 times T1
    n_points=41
)
print(t1_result)

# Plot the result
fig, ax = plt.subplots(figsize=(10, 6))
t1_result.plot(ax=ax)
plt.tight_layout()
plt.savefig("t1_test.png", dpi=150)
print("\nPlot saved as t1_test.png")