"""
Example demonstrating basic ODMR simulation with the quantum NV simulator.
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
    t1=5e-3,                      # s
    t2=500e-6,                    # s
    optics=True,                  # Include optical levels
    nitrogen=False,               # No nitrogen nuclear spin
    method="qutip"                # Use QuTiP backend
)

# Experiment parameters
magnetic_field_range = np.linspace(0, 5e-3, 6)  # 0 to 5 mT
f_center = 2.87e9  # Hz (ZFS)
f_span = 100e6     # Hz (±50 MHz around ZFS)
n_points = 101

# Prepare plot
plt.figure(figsize=(10, 6))
plt.title("ODMR Spectra at Different Magnetic Fields")
plt.xlabel("Frequency (GHz)")
plt.ylabel("Fluorescence (normalized)")

# Run ODMR for different magnetic fields
for B in magnetic_field_range:
    # Set magnetic field along z-axis
    model.set_magnetic_field([0, 0, B])
    
    # Run ODMR experiment
    f_min = f_center - f_span/2
    f_max = f_center + f_span/2
    result = model.simulate_odmr(f_min, f_max, n_points, mw_power=-5.0)
    
    # Plot normalized signal
    freqs_GHz = result.frequencies / 1e9
    signal_norm = result.signal / np.max(result.signal)
    plt.plot(freqs_GHz, signal_norm, label=f"B = {B*1000:.1f} mT")

# Add legend and grid
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
plt.savefig("odmr_quantum_simulation.png", dpi=150)
print(f"Plot saved as odmr_quantum_simulation.png")

# Additional information
print("\nResonance frequencies:")
for B in magnetic_field_range:
    # Calculate expected resonance
    gamma = model.config["gyromagnetic_ratio"]
    zfs = model.config["zero_field_splitting"]
    res_freq = zfs + gamma * B
    print(f"B = {B*1000:.1f} mT: {res_freq/1e9:.6f} GHz")