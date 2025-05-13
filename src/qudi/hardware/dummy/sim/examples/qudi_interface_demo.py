#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demonstration script for the Qudi hardware interface adapters for the NV center simulator.
This script shows how to use the interfaces outside of the Qudi framework for testing.

Copyright (c) 2023
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from qudi.util.enums import SamplingOutputMode

# Add the src directory to the path
sim_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, sim_dir)

# Import the interfaces
from src.qudi_interface import NVSimulatorDevice

def main():
    """
    Main demonstration function.
    """
    print("NV Simulator Qudi Interface Demo")
    print("--------------------------------")
    
    # Create simulator device
    print("Initializing NV simulator device...")
    device = NVSimulatorDevice(config={
        'simulator_params': {
            't1': 5.0e-3,  # T1 relaxation time in seconds
            't2': 1.0e-5,  # T2 dephasing time in seconds
            'laser_power': 1.0,  # laser power in mW
        }
    })
    
    # Access the interfaces
    microwave = device.microwave
    scanner = device.scanner
    
    # Configure the simulator with a magnetic field
    print("Configuring simulator with magnetic field...")
    device.configure_simulator(
        magnetic_field=[0, 0, 0.5e-3]  # 0.5 mT along z-axis
    )
    
    # Print some device information
    print("\nMicrowave interface constraints:")
    print(f"  Frequency range: {microwave.constraints.frequency_limits[0]/1e9:.3f} - {microwave.constraints.frequency_limits[1]/1e9:.3f} GHz")
    print(f"  Power range: {microwave.constraints.power_limits[0]} - {microwave.constraints.power_limits[1]} dBm")
    
    print("\nScanner interface constraints:")
    print(f"  Sample rate range: {scanner.constraints.sample_rate_limits[0]} - {scanner.constraints.sample_rate_limits[1]} Hz")
    print(f"  Frame size range: {scanner.constraints.frame_size_limits[0]} - {scanner.constraints.frame_size_limits[1]} samples")
    
    # Run a simple ODMR scan demo using the interfaces
    print("\nRunning ODMR scan using the interfaces...")
    run_odmr_scan(microwave, scanner)
    
    # Run simulations directly using the device
    print("\nRunning simulations using the device...")
    run_simulations(device)
    
    print("\nDemo completed.")

def run_odmr_scan(microwave, scanner):
    """
    Run a simple ODMR scan using the microwave and scanner interfaces.
    
    @param microwave: The microwave interface
    @param scanner: The scanner interface
    """
    try:
        # Configure scan parameters
        freq_min = 2.85e9  # Hz
        freq_max = 2.89e9  # Hz
        n_points = 101
        power = -10.0  # dBm
        
        # Create frequency list
        frequencies = np.linspace(freq_min, freq_max, n_points)
        
        # Configure microwave for scan
        microwave.configure_scan(
            power=power,
            frequencies=frequencies,
            mode=SamplingOutputMode.JUMP_LIST,
            sample_rate=10.0  # Hz
        )
        
        # Configure scanner
        scanner.set_frame_size(n_points)
        scanner.set_sample_rate(10.0)  # Hz
        
        # Start the scan
        print("Starting microwave scan...")
        microwave.start_scan()
        
        # Acquire a frame
        print("Acquiring fluorescence data...")
        frame = scanner.acquire_frame()
        
        # Stop the scan
        microwave.off()
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies / 1e9, frame['default'], 'b.-')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Fluorescence (counts/s)')
        plt.title('ODMR Scan Using Qudi Interfaces')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(sim_dir, 'qudi_interface_odmr.png'))
        print(f"Plot saved to qudi_interface_odmr.png")
        
    except Exception as e:
        print(f"Error in ODMR scan: {str(e)}")
        if microwave.module_state() != 'idle':
            microwave.off()
        if scanner.module_state() != 'idle':
            scanner.stop_buffered_acquisition()

def run_simulations(device):
    """
    Run simulations directly using the device's built-in methods.
    
    @param device: The NV simulator device
    """
    # Run an ODMR simulation
    print("Running ODMR simulation...")
    odmr_result = device.run_simulation('odmr',
                                        f_min=2.8e9,
                                        f_max=2.9e9,
                                        n_points=101,
                                        mw_power=-10.0)
    
    # Run a Rabi simulation
    print("Running Rabi oscillation simulation...")
    rabi_result = device.run_simulation('rabi',
                                       t_max=1e-6,
                                       n_points=101,
                                       mw_power=-10.0)
    
    # Plot the results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot ODMR
    ax1.plot(odmr_result['frequencies'] / 1e9, odmr_result['signal'], 'b.-')
    ax1.set_xlabel('Frequency (GHz)')
    ax1.set_ylabel('Fluorescence (norm.)')
    ax1.set_title('ODMR Simulation')
    ax1.grid(True)
    
    # Plot Rabi
    ax2.plot(rabi_result['times'] * 1e6, rabi_result['signal'], 'r.-')
    ax2.set_xlabel('Time (µs)')
    ax2.set_ylabel('Fluorescence (norm.)')
    ax2.set_title('Rabi Oscillation Simulation')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(sim_dir, 'qudi_interface_simulations.png'))
    print(f"Plot saved to qudi_interface_simulations.png")

if __name__ == '__main__':
    main()