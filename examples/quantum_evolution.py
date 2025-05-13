"""
Example demonstrating quantum evolution methods for NV centers.

This script shows how to use the new quantum evolution engine to simulate
common NV center experiments like Ramsey, spin echo, and T1 relaxation.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Add the parent directory to path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import PhysicalNVModel

# Create a model with quantum mechanical features
model = PhysicalNVModel(
    zero_field_splitting=2.87e9,  # Hz
    gyromagnetic_ratio=28.0e9,    # Hz/T
    t1=5e-3,                      # s
    t2=500e-6,                    # s
    t2_star=1e-6,                 # s
    optics=True,                  # Include optical levels
    nitrogen=False,               # No nitrogen nuclear spin
    method="qutip"                # Use QuTiP backend
)

# Set a small magnetic field
model.set_magnetic_field([0, 0, 2e-4])  # 0.2 mT along z

# Create a figure with 3 subplots
fig = plt.figure(figsize=(15, 10))
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, :])

# 1. T1 Relaxation Experiment
print("Simulating T1 relaxation...")
t1_result = model.simulate_t1(t_max=10e-3, n_points=21)
print(t1_result)

# Plot T1 results
times_ms = t1_result.times * 1000  # Convert to ms
ax1.plot(times_ms, t1_result.populations[:, 0], 'bo-', label='ms=0')
ax1.plot(times_ms, t1_result.populations[:, 2], 'ro-', label='ms=-1')

# Fit curve if available
if hasattr(t1_result, 't1') and t1_result.t1 is not None:
    t1_fit = t1_result.t1
    fit_x = np.linspace(0, t1_result.times[-1], 100)
    fit_y = 1.0 - np.exp(-fit_x / t1_fit)
    ax1.plot(fit_x * 1000, fit_y, 'k--', label=f'Fit: T1 = {t1_fit*1e6:.1f} µs')

ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Population')
ax1.set_title('T1 Relaxation')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Ramsey Experiment (T2*)
print("\nSimulating Ramsey interference...")
ramsey_result = model.simulate_ramsey(t_max=5e-6, n_points=21, detuning=2e6)
print(ramsey_result)

# Plot Ramsey results
times_us = ramsey_result.times * 1e6  # Convert to µs
ax2.plot(times_us, ramsey_result.fluorescence, 'go-', label='Fluorescence')

# Fit curve if available
if hasattr(ramsey_result, 't2_star') and ramsey_result.t2_star is not None:
    t2_star = ramsey_result.t2_star
    freq = ramsey_result.frequency
    fit_x = np.linspace(0, ramsey_result.times[-1], 100)
    fit_y = np.mean(ramsey_result.fluorescence) + 0.5 * np.exp(-fit_x / t2_star) * np.cos(2 * np.pi * freq * fit_x)
    ax2.plot(fit_x * 1e6, fit_y, 'k--', 
             label=f'Fit: T2* = {t2_star*1e6:.1f} µs, f = {freq/1e6:.2f} MHz')

ax2.set_xlabel('Time (µs)')
ax2.set_ylabel('Fluorescence (a.u.)')
ax2.set_title('Ramsey Interference (T2*)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Spin Echo Experiment (T2)
print("\nSimulating Spin Echo...")
echo_result = model.simulate_spin_echo(t_max=1e-3, n_points=21)
print(echo_result)

# Plot Spin Echo results
times_us = echo_result.times * 1e6  # Convert to µs
ax3.plot(times_us, echo_result.fluorescence, 'mo-', label='Fluorescence')

# Fit curve if available
if hasattr(echo_result, 't2') and echo_result.t2 is not None:
    t2 = echo_result.t2
    fit_x = np.linspace(0, echo_result.times[-1], 100)
    
    # Get min/max values for fit
    min_val = np.min(echo_result.fluorescence)
    max_val = np.max(echo_result.fluorescence)
    amplitude = max_val - min_val
    
    fit_y = min_val + amplitude * np.exp(-fit_x / t2)
    ax3.plot(fit_x * 1e6, fit_y, 'k--', label=f'Fit: T2 = {t2*1e6:.1f} µs')

ax3.set_xlabel('Time (µs)')
ax3.set_ylabel('Fluorescence (a.u.)')
ax3.set_title('Spin Echo (T2)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Adjust layout and save
plt.tight_layout()
plt.savefig("quantum_evolution.png", dpi=150)
print("\nPlot saved as quantum_evolution.png")

# Example of a custom pulse sequence
print("\nRunning a custom pulse sequence (XY8 dynamical decoupling)...")

# Define an XY8 sequence (simplified)
xy8_sequence = [
    # Initialize with laser
    ("laser", 1e-6, {"power": 2.0}),
    
    # Initial pi/2
    ("pi/2", None, {"power": -10.0}),
    
    # XY8 sequence (4 X-Y pairs)
    ("wait", 1e-6, {}),  # tau
    ("pi", None, {"power": -10.0, "phase": 0}),  # X
    ("wait", 2e-6, {}),  # 2*tau
    ("pi", None, {"power": -10.0, "phase": 90}),  # Y
    ("wait", 2e-6, {}),  # 2*tau
    ("pi", None, {"power": -10.0, "phase": 0}),  # X
    ("wait", 2e-6, {}),  # 2*tau
    ("pi", None, {"power": -10.0, "phase": 90}),  # Y
    ("wait", 2e-6, {}),  # 2*tau
    ("pi", None, {"power": -10.0, "phase": 90}),  # Y
    ("wait", 2e-6, {}),  # 2*tau
    ("pi", None, {"power": -10.0, "phase": 0}),  # X
    ("wait", 2e-6, {}),  # 2*tau
    ("pi", None, {"power": -10.0, "phase": 90}),  # Y
    ("wait", 2e-6, {}),  # 2*tau
    ("pi", None, {"power": -10.0, "phase": 0}),  # X
    ("wait", 1e-6, {}),  # tau
    
    # Final pi/2
    ("pi/2", None, {"power": -10.0, "phase": 0, "measure": True, "fluorescence": True})
]

# Execute the sequence
result = model.evolve_pulse_sequence(xy8_sequence)
print("\nFinal fluorescence after XY8 sequence:", result["final_fluorescence"])
print("Final state populations:", result["final_population"])