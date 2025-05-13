"""
Example demonstrating enhanced experimental interfaces for NV centers.

This script showcases all the new experimental interfaces added in TS-003:
- ODMR with linewidth analysis
- Rabi oscillations with damped sine fitting
- T1 relaxation measurements
- T2 measurements with Hahn echo
- Dynamical decoupling sequences (CPMG, XY8)
- Result plotting and analysis
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add the parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import PhysicalNVModel
from src.model import SimulationResult

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

# Create a figure for plotting
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(3, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, :])

# 1. Enhanced ODMR with linewidth analysis
print("Running ODMR experiment...")
odmr_result = model.simulate_odmr(
    f_min=2.86e9,
    f_max=2.88e9,
    n_points=21,
    mw_power=-10.0
)
print(odmr_result)

# Plot the result
odmr_result.plot(ax=ax1)

# 2. Enhanced Rabi oscillations with damped sine fitting
print("\nRunning Rabi oscillation experiment...")
rabi_result = model.simulate_rabi(
    t_max=1e-6,
    n_points=31,
    mw_power=-5.0  # Higher power for faster oscillations
)
print(rabi_result)

# Plot the result
rabi_result.plot(ax=ax2)

# 3. T1 relaxation measurement
print("\nRunning T1 relaxation experiment...")
t1_result = model.simulate_t1(
    t_max=15e-3,  # 15 ms, about 3 times T1
    n_points=21
)
print(t1_result)

# Plot the result
t1_result.plot(ax=ax3)

# 4. T2 measurement with Hahn echo
print("\nRunning T2 echo experiment...")
t2_result = model.simulate_t2_echo(
    t_max=1.5e-3,  # 1.5 ms, about 3 times T2
    n_points=21,
    mw_power=-5.0
)
print(t2_result)

# Plot the result
t2_result.plot(ax=ax4)

# 5. Dynamical decoupling with XY8 sequence
print("\nRunning XY8 dynamical decoupling experiment...")
dd_result = model.simulate_dynamical_decoupling(
    sequence_type="XY8",
    t_max=2e-3,    # 2 ms
    n_points=16,
    n_pulses=8,    # 8 π pulses
    mw_power=-5.0
)
print(dd_result)

# Plot the result
dd_result.plot(ax=ax5)

# Adjust layout and save
plt.tight_layout()
plt.savefig("enhanced_experiments.png", dpi=150)
print("\nPlot saved as enhanced_experiments.png")

# Print a summary of all results
print("\nSummary of experiment results:")
print("1.", odmr_result)
print("2.", rabi_result)
print("3.", t1_result)
print("4.", t2_result)
print("5.", dd_result)
print("\nAll experiments complete.")