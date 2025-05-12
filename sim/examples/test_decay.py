"""
Test script for T2 echo and dynamical decoupling measurements.
This script tests only the decoherence simulations that were having flatline issues.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import PhysicalNVModel

# Create a model with quantum mechanical features and longer coherence times for testing
model = PhysicalNVModel(
    zero_field_splitting=2.87e9,  # Hz
    gyromagnetic_ratio=28.0e9,    # Hz/T
    t1=5e-3,                      # s (5 ms)
    t2=1e-3,                      # s (1 ms) - longer T2 for testing
    t2_star=10e-6,                # s (10 µs)
    optics=False,                 # Skip optical levels for faster simulation
    nitrogen=False,               # No nitrogen nuclear spin
    method="qutip"                # Use QuTiP backend
)

# Set a small magnetic field
model.set_magnetic_field([0, 0, 1e-4])  # 0.1 mT along z

# Create a figure for plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 1. T2 measurement with Hahn echo - use shorter time to avoid integration issues
print("Running T2 echo experiment...")
t2_result = model.simulate_t2_echo(
    t_max=0.5e-3,  # 0.5 ms (half the T2 value)
    n_points=11,    # Fewer points for faster simulation
    mw_power=-5.0
)
print(t2_result)

# Plot the result
t2_result.plot(ax=ax1)

# 2. Dynamical decoupling with XY8 sequence - also use shorter time
print("\nRunning XY8 dynamical decoupling experiment...")
dd_result = model.simulate_dynamical_decoupling(
    sequence_type="XY8",
    t_max=0.8e-3,  # 0.8 ms
    n_points=11,   # Fewer points
    n_pulses=8,    # 8 π pulses
    mw_power=-5.0
)
print(dd_result)

# Plot the result
dd_result.plot(ax=ax2)

# Adjust layout and save
plt.tight_layout()
plt.savefig("test_decay.png", dpi=150)
print("\nPlot saved as test_decay.png")