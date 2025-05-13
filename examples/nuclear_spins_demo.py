#!/usr/bin/env python3
"""
Nuclear Spin Environment Demonstration

This script demonstrates the use of the nuclear spin environment module
for simulating NV centers in diamond with surrounding nuclear spins.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import the NV center model
from src import PhysicalNVModel

def main():
    print("Nuclear Spin Environment Demonstration")
    print("======================================")
    
    # Create a model with nuclear spin environment
    print("\nInitializing NV center with nuclear spin environment...")
    
    model = PhysicalNVModel(
        # Basic NV configuration
        zero_field_splitting=2.87e9,  # Hz
        gyromagnetic_ratio=28.0e9,    # Hz/T
        temperature=298.0,            # K
        t1=5e-3,                      # s
        t2=1e-3,                      # s
        t2_star=2e-6,                 # s
        optics=True,
        
        # Nuclear spin configuration
        nuclear_spins=True,
        c13_concentration=0.011,      # Natural abundance (1.1%)
        bath_size=15,                 # 15 spins in the environment
        include_nitrogen_nuclear=True,
        nitrogen_species="14N"
    )
    
    # Apply a magnetic field
    model.set_magnetic_field([0, 0, 0.05])  # 500 G along z-axis
    print(f"Applied magnetic field: {model.magnetic_field} T")

    # Part 1: Calculate coherence times with nuclear spins
    print("\n1. Coherence Times with Nuclear Spins")
    print("-----------------------------------")
    
    coherence_times = model.calculate_coherence_times()
    print(f"T1:   {coherence_times['t1']*1e6:.2f} μs")
    print(f"T2:   {coherence_times['t2']*1e6:.2f} μs")
    print(f"T2*:  {coherence_times['t2_star']*1e6:.2f} μs")
    print(f"Note: {coherence_times['note']}")
    
    # Part 2: Simulate DEER experiment
    print("\n2. DEER Experiment Simulation")
    print("----------------------------")
    
    # Generate tau values for DEER experiment
    tau_values = np.linspace(0, 50e-6, 51)  # 0 to 50 μs
    print(f"Simulating DEER sequence with {len(tau_values)} time points...")
    
    # Run DEER simulation
    deer_result = model.simulate_deer(tau_values, target_nuclear='13C')
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(tau_values * 1e6, deer_result.signal)
    plt.xlabel('Tau (μs)')
    plt.ylabel('DEER Signal')
    plt.title('DEER Signal vs. Evolution Time')
    plt.grid(True)
    plt.savefig('deer_experiment.png')
    print(f"DEER experiment results saved to deer_experiment.png")
    
    # Part 3: RF Control
    print("\n3. RF Pulse Control")
    print("------------------")
    
    # Calculate Larmor frequency for 13C at 0.05 T
    b0 = 0.05  # T
    gamma_c13 = 10.705e6  # Hz/T
    larmor_c13 = gamma_c13 * b0
    print(f"13C Larmor frequency at {b0*1e4:.0f} G: {larmor_c13/1e3:.2f} kHz")
    
    # Apply RF pulse to manipulate 13C
    rf_power = 0.1  # W
    rf_duration = 1e-5  # 10 μs
    
    print(f"Applying RF pulse at {larmor_c13/1e3:.2f} kHz with {rf_power} W power for {rf_duration*1e6:.1f} μs...")
    rf_result = model.apply_rf_pulse(
        frequency=larmor_c13,
        power=rf_power,
        duration=rf_duration,
        target_nuclear='13C'
    )
    
    if rf_result['success']:
        print(f"RF pulse applied successfully")
        print(f"Rotation angle: {rf_result['rotation_angle']/np.pi:.2f}π radians")
    else:
        print(f"RF pulse failed: {rf_result.get('error', 'unknown error')}")
    
    # Part 4: Dynamical Decoupling with Nuclear Spins
    print("\n4. Dynamical Decoupling with Nuclear Spin Environment")
    print("--------------------------------------------------")
    
    # Generate time points for dynamical decoupling
    t_max = 100e-6  # 100 μs
    n_points = 101
    
    # Compare different sequences
    sequences = ['hahn', 'cpmg', 'xy4', 'xy8']
    
    plt.figure(figsize=(12, 8))
    
    for seq in sequences:
        print(f"Simulating {seq.upper()} sequence...")
        result = model.simulate_dynamical_decoupling(
            sequence_type=seq,
            t_max=t_max,
            n_points=n_points,
            n_pulses=4,  # Use 4 pulses for all sequences
            mw_frequency=model.get_resonance_frequency(),
            mw_power=20.0  # 20 dBm power
        )
        
        # Plot normalized signal
        signal = result.signal
        signal_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
        plt.plot(result.times * 1e6, signal_norm, label=f"{seq.upper()}")
    
    plt.xlabel('Free Evolution Time (μs)')
    plt.ylabel('Normalized Signal')
    plt.title('Dynamical Decoupling with Nuclear Spin Environment')
    plt.legend()
    plt.grid(True)
    plt.savefig('dd_sequences.png')
    print(f"Dynamical decoupling results saved to dd_sequences.png")
    
    print("\nDemonstration complete!")

if __name__ == "__main__":
    main()