# -*- coding: utf-8 -*-

"""
This file allows running the NV simulator as a standalone module for testing.

Copyright (c) 2023, IQO

This file is part of qudi.

Qudi is free software: you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

Qudi is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with qudi.
If not, see <https://www.gnu.org/licenses/>.
"""

import os
import sys
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add the parent directory to the Python path
parent_dir = str(Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Add the simulator directory to the Python path
sim_dir = str(Path(os.path.dirname(os.path.abspath(__file__))).parent.parent.parent.parent / 'sim')
if sim_dir not in sys.path:
    sys.path.append(sim_dir)

from qudi.hardware.nv_simulator.qudi_facade import QudiFacade


def setup_simulator():
    """Set up the simulator and return the facade."""
    facade = QudiFacade({
        'magnetic_field': [0, 0, 300],  # Gauss
        'temperature': 300  # Kelvin
    })
    return facade


def run_odmr_simulation(facade, freq_range=(2.7e9, 3.1e9), num_points=401):
    """Run a basic ODMR simulation."""
    print("Running ODMR simulation...")
    
    # Create frequency points
    frequencies = np.linspace(freq_range[0], freq_range[1], num_points)
    
    # Turn on laser
    facade.laser_controller.on()
    
    # Get baseline fluorescence
    baseline = facade.nv_model.get_fluorescence_rate()
    
    # Array to store results
    odmr_signal = np.zeros(num_points)
    
    # Run ODMR
    for i, freq in enumerate(frequencies):
        # Set microwave frequency
        facade.microwave_controller.set_frequency(freq)
        
        # Turn on microwave
        facade.microwave_controller.set_power(10.0)  # 10 dBm
        facade.microwave_controller.on()
        
        # Let the system evolve
        facade.nv_model.evolve(1e-6)  # 1 µs
        
        # Measure fluorescence
        fluorescence = facade.nv_model.get_fluorescence_rate()
        odmr_signal[i] = fluorescence
        
        # Turn off microwave
        facade.microwave_controller.off()
        
        # Reset the system state
        facade.nv_model.reset_state()
        facade.laser_controller.on()
        
        # Progress indication
        if i % 40 == 0:
            print(f"Progress: {i}/{num_points}")
    
    # Turn off laser
    facade.laser_controller.off()
    
    # Normalize signal
    normalized_signal = odmr_signal / baseline
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot((frequencies - 2.87e9) / 1e6, normalized_signal)
    plt.xlabel('Frequency - 2.87 GHz (MHz)')
    plt.ylabel('Normalized Fluorescence')
    plt.title('ODMR Spectrum')
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('odmr_simulation.png')
    print("ODMR simulation completed. Results saved to 'odmr_simulation.png'")
    
    return frequencies, normalized_signal


def run_rabi_simulation(facade, duration_range=(0, 1e-6), num_points=101):
    """Run a Rabi oscillation simulation."""
    print("Running Rabi oscillation simulation...")
    
    # Create time points
    durations = np.linspace(duration_range[0], duration_range[1], num_points)
    
    # Array to store results
    fluorescence = np.zeros(num_points)
    
    # Set microwave frequency to resonance
    facade.microwave_controller.set_frequency(2.87e9)
    facade.microwave_controller.set_power(20.0)  # 20 dBm
    
    for i, duration in enumerate(durations):
        # Reset the system state
        facade.nv_model.reset_state()
        
        # Apply microwave pulse
        facade.microwave_controller.on()
        facade.nv_model.evolve(duration)
        facade.microwave_controller.off()
        
        # Apply laser readout
        facade.laser_controller.on()
        facade.nv_model.evolve(1e-6)  # 1 µs readout
        
        # Measure fluorescence
        fluorescence[i] = facade.nv_model.get_fluorescence_rate()
        
        # Turn off laser
        facade.laser_controller.off()
        
        # Progress indication
        if i % 10 == 0:
            print(f"Progress: {i}/{num_points}")
    
    # Normalize fluorescence
    max_fluorescence = np.max(fluorescence)
    normalized_fluorescence = fluorescence / max_fluorescence
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(durations * 1e9, normalized_fluorescence)
    plt.xlabel('Pulse Duration (ns)')
    plt.ylabel('Normalized Fluorescence')
    plt.title('Rabi Oscillation')
    plt.grid(True)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('rabi_simulation.png')
    print("Rabi simulation completed. Results saved to 'rabi_simulation.png'")
    
    return durations, normalized_fluorescence


def run_confocal_simulation(facade, scan_range=(-5, 5), resolution=101):
    """Run a basic confocal scan simulation."""
    print("Running confocal scan simulation...")
    
    # Create scan points
    x_points = np.linspace(scan_range[0], scan_range[1], resolution) * 1e-6  # Convert to meters
    y_points = np.linspace(scan_range[0], scan_range[1], resolution) * 1e-6  # Convert to meters
    
    # Array to store results
    image = np.zeros((resolution, resolution))
    
    # Turn on laser
    facade.laser_controller.on()
    
    # Run scan
    for i, x in enumerate(x_points):
        for j, y in enumerate(y_points):
            # Set position
            facade.confocal_simulator.set_position(x, y, 0)
            
            # Measure fluorescence
            image[i, j] = facade.nv_model.get_fluorescence_rate()
            
        # Progress indication
        if i % 10 == 0:
            print(f"Scan progress: {i}/{resolution}")
    
    # Turn off laser
    facade.laser_controller.off()
    
    # Plot results
    plt.figure(figsize=(10, 8))
    plt.imshow(image, extent=[scan_range[0], scan_range[1], scan_range[0], scan_range[1]], 
               origin='lower', cmap='viridis')
    plt.colorbar(label='Fluorescence (counts/s)')
    plt.xlabel('X Position (µm)')
    plt.ylabel('Y Position (µm)')
    plt.title('Confocal Scan')
    plt.grid(False)
    plt.tight_layout()
    
    # Save plot
    plt.savefig('confocal_simulation.png')
    print("Confocal scan completed. Results saved to 'confocal_simulation.png'")
    
    return image


def main():
    """Main function to run simulations."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run NV simulator demonstrations')
    parser.add_argument('--odmr', action='store_true', help='Run ODMR simulation')
    parser.add_argument('--rabi', action='store_true', help='Run Rabi oscillation simulation')
    parser.add_argument('--confocal', action='store_true', help='Run confocal scan simulation')
    parser.add_argument('--all', action='store_true', help='Run all simulations')
    
    args = parser.parse_args()
    
    # If no arguments provided, run all simulations
    if not (args.odmr or args.rabi or args.confocal or args.all):
        args.all = True
    
    # Set up simulator
    facade = setup_simulator()
    
    # Run requested simulations
    if args.odmr or args.all:
        run_odmr_simulation(facade)
        
    if args.rabi or args.all:
        run_rabi_simulation(facade)
        
    if args.confocal or args.all:
        run_confocal_simulation(facade)
    
    print("All simulations completed!")


if __name__ == "__main__":
    main()