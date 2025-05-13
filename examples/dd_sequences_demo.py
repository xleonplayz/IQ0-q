#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demonstration of dynamical decoupling sequences using the quantum-accurate framework.

This script compares different dynamical decoupling sequences and their effectiveness in
preserving quantum coherence.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Add parent directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

# Import from sequences module
from src.sequences import (
    DynamicalDecouplingSequence, PulseParameters, PulseError,
    create_hahn_echo, create_cpmg, create_xy4, create_xy8, create_xy16, create_kdd,
    plot_filter_function, compare_sequences, plot_sequence_comparison
)

# Import the NV model
from src.model import PhysicalNVModel

# Create an NV center model
def create_nv_model():
    model = PhysicalNVModel(
        t1=5e-3,      # T1 relaxation time (5 ms)
        t2=1e-6,      # T2 dephasing time (1 µs)
        d_gs=2.87e9,  # Zero-field splitting (2.87 GHz)
        gyro_e=28e9   # Gyromagnetic ratio (28 GHz/T)
    )
    # Apply a small magnetic field
    model.set_magnetic_field([0, 0, 5e-4])  # 0.5 mT along z
    return model


def plot_dd_sequences():
    """Plot all dynamical decoupling sequences for visualization."""
    # Create model
    model = create_nv_model()
    
    # Create figure
    fig, axs = plt.subplots(3, 2, figsize=(15, 12))
    axs = axs.flatten()
    
    # Create and plot sequences
    tau = 500e-9  # 500 ns delay time
    
    # Hahn Echo
    seq_hahn = create_hahn_echo(model, tau)
    seq_hahn.plot_sequence(ax=axs[0])
    axs[0].set_title("Hahn Echo")
    
    # CPMG-4
    seq_cpmg = create_cpmg(model, tau, 4)
    seq_cpmg.plot_sequence(ax=axs[1])
    axs[1].set_title("CPMG-4")
    
    # XY4
    seq_xy4 = create_xy4(model, tau)
    seq_xy4.plot_sequence(ax=axs[2])
    axs[2].set_title("XY4")
    
    # XY8
    seq_xy8 = create_xy8(model, tau)
    seq_xy8.plot_sequence(ax=axs[3])
    axs[3].set_title("XY8")
    
    # XY16
    seq_xy16 = create_xy16(model, tau)
    seq_xy16.plot_sequence(ax=axs[4])
    axs[4].set_title("XY16")
    
    # KDD
    seq_kdd = create_kdd(model, tau)
    seq_kdd.plot_sequence(ax=axs[5])
    axs[5].set_title("KDD")
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "dd_sequences.png"), dpi=200)
    plt.close()
    
    print(f"Sequence diagrams saved to dd_sequences.png")


def compare_filter_functions():
    """Compare filter functions of different DD sequences."""
    # Create model
    model = create_nv_model()
    
    # Create sequences with the same total sequence time
    total_time = 10e-6  # 10 µs total evolution time
    
    # Hahn Echo (1 π pulse)
    seq_hahn = create_hahn_echo(model, total_time/2)
    
    # CPMG-4 (4 π pulses)
    seq_cpmg4 = create_cpmg(model, total_time/(2*4), 4)
    
    # XY4 (4 π pulses in 1 repetition)
    seq_xy4 = create_xy4(model, total_time/(2*4))
    
    # XY8 (8 π pulses in 1 repetition)
    seq_xy8 = create_xy8(model, total_time/(2*8))
    
    # XY16 (16 π pulses in 1 repetition)
    seq_xy16 = create_xy16(model, total_time/(2*16))
    
    # KDD (5 composite π pulses in 1 repetition)
    seq_kdd = create_kdd(model, total_time/(2*5))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot filter functions
    sequences = [seq_hahn, seq_cpmg4, seq_xy4, seq_xy8, seq_xy16, seq_kdd]
    ax = plot_filter_function(sequences, omega_min=1e3, omega_max=1e9, ax=ax)
    
    # Set up logarithmic scale
    ax.set_xscale('log')
    ax.set_yscale('log')
    
    # Add title and labels
    ax.set_title('Filter Functions of Dynamical Decoupling Sequences')
    ax.set_xlabel('Angular Frequency ω (rad/s)')
    ax.set_ylabel('Filter Function F(ω)')
    
    # Add a legend
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "dd_filter_functions.png"), dpi=200)
    plt.close()
    
    print(f"Filter function comparison saved to dd_filter_functions.png")


def simulate_decoherence_times():
    """Simulate decoherence for different sequences and pulse numbers."""
    # Create model
    model = create_nv_model()
    
    # Set up t_max based on the base T2
    t_max = model.config["t2"] * 5  # 5x the base T2 time
    n_points = 51
    
    # Define pulse numbers to test
    pulse_numbers = [1, 4, 8, 16, 32, 64]
    
    # Define sequences to test
    sequence_types = ["hahn", "cpmg", "xy4", "xy8", "xy16", "kdd"]
    
    # Plot colors
    colors = {
        "hahn": "k", 
        "cpmg": "r", 
        "xy4": "g", 
        "xy8": "b", 
        "xy16": "m", 
        "kdd": "c"
    }
    
    # Create figure for decoherence curves
    fig_curves, ax_curves = plt.subplots(figsize=(10, 6))
    
    # Create figure for T2 vs n_pulses
    fig_t2, ax_t2 = plt.subplots(figsize=(10, 6))
    
    # Fixed number of pulses for decoherence curves
    fixed_pulses = 16
    
    # Store T2 times
    t2_data = {seq_type: [] for seq_type in sequence_types}
    
    # For each sequence type
    for seq_type in sequence_types:
        # For decoherence curves with fixed pulse number
        result = model.simulate_dynamical_decoupling(
            seq_type, t_max, n_points, fixed_pulses
        )
        
        # Normalize the signal
        norm_signal = (result.signal - min(result.signal)) / (max(result.signal) - min(result.signal))
        
        # Plot the decoherence curve
        ax_curves.plot(
            result.times * 1e6,  # Convert to µs
            norm_signal,
            label=f"{seq_type.upper()} (n={fixed_pulses})",
            color=colors[seq_type],
            marker='o',
            markersize=4,
            alpha=0.7
        )
        
        # For T2 vs pulse number
        for n_pulses in pulse_numbers:
            # Skip KDD for very high pulse numbers (computational complexity)
            if seq_type == "kdd" and n_pulses > 32:
                t2_data[seq_type].append(None)
                continue
                
            # Simulate with appropriate t_max (scale based on expected T2 enhancement)
            # Use the empirical scaling law: T2(n) = T2 * n^p where p is typically 2/3
            expected_t2 = model.config["t2"] * (n_pulses ** 0.67)
            sim_t_max = min(expected_t2 * 3, 100e-6)  # Cap at 100 µs for computational efficiency
            
            # Simulate
            result = model.simulate_dynamical_decoupling(
                seq_type, sim_t_max, 31, n_pulses
            )
            
            # Store T2 time
            t2_data[seq_type].append(result.t2)
    
    # Finalize decoherence curves plot
    ax_curves.set_xlabel('Time (µs)')
    ax_curves.set_ylabel('Normalized Signal')
    ax_curves.set_title(f'Decoherence Curves for Different DD Sequences (n={fixed_pulses})')
    ax_curves.grid(True, alpha=0.3)
    ax_curves.legend()
    
    # Save decoherence curves
    fig_curves.tight_layout()
    fig_curves.savefig(os.path.join(script_dir, "dd_decoherence_curves.png"), dpi=200)
    plt.close(fig_curves)
    
    # Plot T2 vs pulse number
    for seq_type in sequence_types:
        # Filter out None values
        valid_indices = [i for i, t2 in enumerate(t2_data[seq_type]) if t2 is not None]
        valid_pulses = [pulse_numbers[i] for i in valid_indices]
        valid_t2 = [t2_data[seq_type][i] for i in valid_indices]
        
        if valid_t2:
            # Plot T2 vs pulse number
            ax_t2.loglog(
                valid_pulses, 
                [t2 * 1e6 for t2 in valid_t2],  # Convert to µs
                label=seq_type.upper(),
                color=colors[seq_type],
                marker='o',
                markersize=6,
                linestyle='-'
            )
    
    # Add reference scaling lines
    max_pulses = max(pulse_numbers)
    base_t2 = model.config["t2"] * 1e6  # Base T2 in µs
    
    # n^(2/3) scaling
    ref_pulses = np.array(pulse_numbers)
    ref_t2_1 = base_t2 * (ref_pulses ** (2/3))
    ax_t2.loglog(ref_pulses, ref_t2_1, 'k--', alpha=0.5, label='n^(2/3) scaling')
    
    # n^(1) scaling (linear)
    ref_t2_2 = base_t2 * ref_pulses
    ax_t2.loglog(ref_pulses, ref_t2_2, 'k:', alpha=0.5, label='n^1 scaling')
    
    # Finalize T2 vs pulse number plot
    ax_t2.set_xlabel('Number of π Pulses (n)')
    ax_t2.set_ylabel('T2 Time (µs)')
    ax_t2.set_title('T2 Enhancement with Different DD Sequences')
    ax_t2.grid(True, which='both', alpha=0.3)
    ax_t2.legend()
    
    # Save T2 vs pulse number plot
    fig_t2.tight_layout()
    fig_t2.savefig(os.path.join(script_dir, "dd_t2_enhancement.png"), dpi=200)
    plt.close(fig_t2)
    
    print(f"Decoherence curves saved to dd_decoherence_curves.png")
    print(f"T2 enhancement plot saved to dd_t2_enhancement.png")


def test_pulse_errors():
    """Test the effect of pulse errors on different DD sequences."""
    # Create model
    model = create_nv_model()
    
    # Create sequences with the same total evolution time
    tau = 500e-9  # 500 ns delay time
    
    # Error magnitudes to test (as fraction)
    error_magnitudes = [0.0, 0.02, 0.05, 0.1, 0.2]
    
    # Create figure
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))
    
    # Test amplitude errors
    for seq_type, color, marker in [
        ("cpmg", "r", "o"), 
        ("xy4", "g", "s"), 
        ("xy8", "b", "^"), 
        ("xy16", "m", "d"),
        ("kdd", "c", "x")
    ]:
        # Store fidelities
        fidelities = []
        
        for error in error_magnitudes:
            # Create pulse parameters with amplitude error
            pulse_params = PulseParameters(
                error_type=PulseError.AMPLITUDE,
                error_amplitude=error
            )
            
            # Create sequence based on type
            if seq_type == "cpmg":
                seq = create_cpmg(model, tau, 4, pulse_params)
            elif seq_type == "xy4":
                seq = create_xy4(model, tau, 1, pulse_params)
            elif seq_type == "xy8":
                seq = create_xy8(model, tau, 1, pulse_params)
            elif seq_type == "xy16":
                seq = create_xy16(model, tau, 1, pulse_params)
            elif seq_type == "kdd":
                seq = create_kdd(model, tau, 1, pulse_params)
            
            # Prepare initial state
            model.reset_state()
            initial_state = model.state.copy()
            
            # Simulate with error
            result = seq.simulate(
                initial_state=initial_state,
                include_decoherence=False  # Isolate pulse error effects
            )
            
            # Calculate fidelity
            if 'coherence_values' in result and result['coherence_values']:
                # Get the last coherence value
                fidelity = abs(result['coherence_values'][-1]['value']) ** 2
            else:
                fidelity = 0.0
                
            fidelities.append(fidelity)
        
        # Plot fidelity vs error magnitude
        axs[0].plot(
            error_magnitudes,
            fidelities,
            label=seq_type.upper(),
            color=color,
            marker=marker,
            markersize=8,
            linestyle='-',
            linewidth=2
        )
    
    # Set up amplitude error plot
    axs[0].set_xlabel('Amplitude Error (fraction)')
    axs[0].set_ylabel('Fidelity')
    axs[0].set_title('Effect of Amplitude Errors on DD Sequences')
    axs[0].grid(True, alpha=0.3)
    axs[0].set_ylim(0, 1.05)
    axs[0].legend()
    
    # Test phase errors
    for seq_type, color, marker in [
        ("cpmg", "r", "o"), 
        ("xy4", "g", "s"), 
        ("xy8", "b", "^"), 
        ("xy16", "m", "d"),
        ("kdd", "c", "x")
    ]:
        # Store fidelities
        fidelities = []
        
        for error in error_magnitudes:
            # Create pulse parameters with phase error
            pulse_params = PulseParameters(
                error_type=PulseError.PHASE,
                error_phase=error * np.pi/2  # Convert to radians (fraction of π/2)
            )
            
            # Create sequence based on type
            if seq_type == "cpmg":
                seq = create_cpmg(model, tau, 4, pulse_params)
            elif seq_type == "xy4":
                seq = create_xy4(model, tau, 1, pulse_params)
            elif seq_type == "xy8":
                seq = create_xy8(model, tau, 1, pulse_params)
            elif seq_type == "xy16":
                seq = create_xy16(model, tau, 1, pulse_params)
            elif seq_type == "kdd":
                seq = create_kdd(model, tau, 1, pulse_params)
            
            # Prepare initial state
            model.reset_state()
            initial_state = model.state.copy()
            
            # Simulate with error
            result = seq.simulate(
                initial_state=initial_state,
                include_decoherence=False  # Isolate pulse error effects
            )
            
            # Calculate fidelity
            if 'coherence_values' in result and result['coherence_values']:
                # Get the last coherence value
                fidelity = abs(result['coherence_values'][-1]['value']) ** 2
            else:
                fidelity = 0.0
                
            fidelities.append(fidelity)
        
        # Plot fidelity vs error magnitude
        axs[1].plot(
            error_magnitudes,
            fidelities,
            label=seq_type.upper(),
            color=color,
            marker=marker,
            markersize=8,
            linestyle='-',
            linewidth=2
        )
    
    # Set up phase error plot
    axs[1].set_xlabel('Phase Error (fraction of π/2)')
    axs[1].set_ylabel('Fidelity')
    axs[1].set_title('Effect of Phase Errors on DD Sequences')
    axs[1].grid(True, alpha=0.3)
    axs[1].set_ylim(0, 1.05)
    axs[1].legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(script_dir, "dd_error_robustness.png"), dpi=200)
    plt.close()
    
    print(f"Pulse error robustness plot saved to dd_error_robustness.png")


if __name__ == "__main__":
    # Plot sequence diagrams
    plot_dd_sequences()
    
    # Compare filter functions
    compare_filter_functions()
    
    # Simulate decoherence times
    simulate_decoherence_times()
    
    # Test pulse errors
    test_pulse_errors()
    
    print("All demonstrations completed.")