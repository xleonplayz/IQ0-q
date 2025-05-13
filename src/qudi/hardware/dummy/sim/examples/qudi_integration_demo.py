#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script demonstrating the use of NV simulator with Qudi hardware interfaces.
This script shows how to use the simulator interfaces directly without running the full Qudi application.

Copyright (c) 2023
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

# Add the simulator to the python path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

# Import the simulator interfaces
from src.qudi_interface.hardware import QudiFacade
from src.qudi_interface.hardware import NVSimMicrowaveDevice
from src.qudi_interface.hardware import NVSimFastCounter
from src.qudi_interface.hardware import NVSimPulser
from src.qudi_interface.hardware import NVSimScanningProbe
from src.qudi_interface.hardware import NVSimLaser


def setup_simulator():
    """
    Create and configure the simulator components.
    
    @return tuple: (facade, microwave, fast_counter, pulser, scanning_probe, laser)
    """
    print("Setting up NV simulator components...")
    
    # Create the facade with initial configuration
    simulator_config = {
        'simulator': {
            'magnetic_field': [0, 0, 100]  # 100 Gauss along z
        },
        'confocal': {
            'lattice': {
                'nv_density': 1.0,  # NV centers per cubic micrometer
                'size': [50e-6, 50e-6, 50e-6]  # 50 μm cube sample
            },
            'laser': {
                'wavelength': 532e-9,  # 532 nm
                'numerical_aperture': 0.8
            }
        }
    }
    
    facade = QudiFacade(simulator_config)
    
    # Create hardware interfaces
    microwave = NVSimMicrowaveDevice(config={'magnetic_field': [0, 0, 100]})
    fast_counter = NVSimFastCounter(config={'bin_width_s': 1e-9, 'record_length_s': 1e-6})
    pulser = NVSimPulser()
    scanning_probe = NVSimScanningProbe()
    laser = NVSimLaser(config={'power_range': [0.0, 100.0], 'initial_power': 30.0})
    
    # Activate each module (this would be done by Qudi normally)
    microwave.on_activate()
    fast_counter.on_activate()
    pulser.on_activate()
    scanning_probe.on_activate()
    laser.on_activate()
    
    return facade, microwave, fast_counter, pulser, scanning_probe, laser


def run_odmr_example(microwave, fast_counter, laser):
    """
    Run a simple ODMR measurement using the simulator interfaces.
    
    @param microwave: The microwave device interface
    @param fast_counter: The fast counter interface
    @param laser: The laser interface
    
    @return tuple: (frequencies, counts) - the ODMR data
    """
    print("\nRunning ODMR example...")
    
    # Define ODMR parameters
    freq_start = 2.8e9  # 2.8 GHz
    freq_stop = 2.95e9  # 2.95 GHz
    num_points = 101
    power = -10.0  # dBm
    
    # Turn on the laser
    laser.on()
    print("Laser turned on")
    
    # Configure the microwave scan
    freq_range = (freq_start, freq_stop, num_points)
    microwave.configure_scan(power, freq_range, 0, 100.0)  # mode 0 = JUMP_LIST
    print(f"Microwave configured for scan from {freq_start/1e9:.4f} GHz to {freq_stop/1e9:.4f} GHz")
    
    # Configure fast counter
    fast_counter.configure(1e-6, 1e-3)  # 1 μs bins, 1 ms record length
    
    # Create arrays to store results
    frequencies = np.linspace(freq_start, freq_stop, num_points)
    counts = np.zeros(num_points)
    
    # Start the microwave scan
    microwave.start_scan()
    print("Starting scan...")
    
    # Collect data for each frequency point
    for i in range(num_points):
        # Set the microwave frequency
        current_freq = frequencies[i]
        
        # If using the real microwave scan we'd wait for a trigger here
        # For this example, we manually step to the next frequency
        if i > 0:  # Skip first frequency since that's set by start_scan
            microwave.scan_next()
        
        # Wait for equilibration
        sleep(0.01)
        
        # Get count data from the fast counter
        fast_counter.start_measure()
        sleep(0.05)  # Wait for acquisition
        data = fast_counter.get_data_trace()
        fast_counter.stop_measure()
        
        # Store the result (average count rate)
        counts[i] = np.mean(data)
        
        # Progress update
        if i % 10 == 0:
            print(f"Progress: {i}/{num_points} frequencies")
    
    # Stop the microwave
    microwave.off()
    
    # Turn off the laser
    laser.off()
    
    print("ODMR scan completed")
    return frequencies, counts


def run_confocal_example(scanning_probe, laser):
    """
    Run a simple confocal scan using the simulator interfaces.
    
    @param scanning_probe: The scanning probe interface
    @param laser: The laser interface
    
    @return np.ndarray: The 2D scan image
    """
    print("\nRunning confocal scanning example...")
    
    # Define scan parameters
    x_range = (-10e-6, 10e-6)  # 20 μm x 20 μm scan area
    y_range = (-10e-6, 10e-6)
    resolution = 20  # 20x20 pixels
    
    # Turn on the laser
    laser.on()
    print("Laser turned on")
    
    # Configure the scanner
    from qudi.interface.scanning_probe_interface import ScannerSettings
    settings = ScannerSettings(
        resolution=resolution,
        forward_range={'x': x_range, 'y': y_range, 'z': (0, 0)},
        forward_axes=frozenset({'x', 'y'}),
        static_axes=frozenset({'z'})
    )
    scanning_probe.configure_scan(settings)
    print(f"Scanner configured for {resolution}x{resolution} pixel scan")
    
    # Create array to store results
    image = np.zeros((resolution, resolution))
    
    # Manual scan (in a real setup, this would be handled by Qudi's ScanningProbeLogic)
    print("Starting scan...")
    for y_idx in range(resolution):
        # Calculate y position
        y_pos = y_range[0] + (y_range[1] - y_range[0]) * y_idx / (resolution - 1)
        
        for x_idx in range(resolution):
            # Calculate x position
            x_pos = x_range[0] + (x_range[1] - x_range[0]) * x_idx / (resolution - 1)
            
            # Move the scanner
            scanning_probe.set_position(x=x_pos, y=y_pos)
            
            # Wait for move and equilibration
            sleep(0.02)
            
            # Get photon counts from the current position
            # In a real setup, this would come from a counter, but here we cheat and
            # ask the confocal simulator directly through the scanning probe
            if hasattr(scanning_probe, '_confocal_simulator') and scanning_probe._confocal_simulator is not None:
                counts = scanning_probe._confocal_simulator.get_counts()
                image[y_idx, x_idx] = counts
            
        # Progress update
        print(f"Progress: {y_idx+1}/{resolution} lines")
    
    # Turn off the laser
    laser.off()
    
    print("Confocal scan completed")
    return image


def run_pulse_example(pulser, fast_counter, laser):
    """
    Run a simple pulse sequence example using the simulator interfaces.
    
    @param pulser: The pulser interface
    @param fast_counter: The fast counter interface
    @param laser: The laser interface
    
    @return np.ndarray: The measurement result
    """
    print("\nRunning pulse sequence example...")
    
    # Define a simple pi pulse sequence (for Rabi oscillation)
    # Create a waveform with:
    # 1. Initial laser pulse for polarization
    # 2. Microwave pulse with variable duration
    # 3. Final laser pulse for readout
    
    # Turn on the laser initially (to ensure it's configured)
    laser.on()
    sleep(0.1)
    laser.off()
    
    # Setup active channels for the pulser
    pulser.set_active_channels({
        'a_ch1': True,  # MW amplitude
        'd_ch1': True,  # MW switch
        'd_ch2': True   # Laser
    })
    print("Pulser channels configured")
    
    # Create a simple Rabi sequence with different microwave pulse durations
    pulse_durations = np.linspace(0, 1000e-9, 21)  # 0 to 1000 ns
    results = np.zeros(len(pulse_durations))
    
    for i, duration in enumerate(pulse_durations):
        # Create analog samples
        # - First channel is MW amplitude (constant during pulse)
        num_samples = 1000  # Total samples in sequence
        
        # Fixed timing parameters (in samples)
        init_laser_samples = 100  # Initial laser pulse
        mw_start_sample = 200  # Start of MW pulse
        mw_duration_samples = int(duration * 1e9)  # Convert ns to samples
        readout_start_sample = mw_start_sample + mw_duration_samples + 100
        readout_duration_samples = 100
        
        # Create the sample arrays
        analog_samples = np.zeros((num_samples, 1))  # One analog channel
        digital_samples = np.zeros((num_samples, 2))  # Two digital channels
        
        # Set the MW amplitude (a_ch1)
        analog_samples[mw_start_sample:mw_start_sample+mw_duration_samples, 0] = 1.0
        
        # Set the MW switch (d_ch1)
        digital_samples[mw_start_sample:mw_start_sample+mw_duration_samples, 0] = 1
        
        # Set the laser pulses (d_ch2)
        digital_samples[:init_laser_samples, 1] = 1  # Initial laser pulse
        digital_samples[readout_start_sample:readout_start_sample+readout_duration_samples, 1] = 1  # Readout laser pulse
        
        # Create a waveform name
        wfm_name = f"rabi_pulse_{i}"
        
        # Write the waveform to the pulser
        pulser.write_waveform(wfm_name, analog_samples, digital_samples, True, True)
        
        # Load the waveform
        pulser.load_waveform(wfm_name)
        
        # Run the waveform
        pulser.pulser_on()
        
        # Wait for the sequence to complete
        sleep(0.1)
        
        # Get the measurement result (in a real setup this would come from a counter during the readout)
        fast_counter.start_measure()
        sleep(0.05)
        data = fast_counter.get_data_trace()
        fast_counter.stop_measure()
        
        # Store the result (integrated readout window counts)
        readout_start_bin = readout_start_sample
        readout_end_bin = readout_start_sample + readout_duration_samples
        results[i] = np.sum(data[readout_start_bin:readout_end_bin])
        
        # Turn off pulser
        pulser.pulser_off()
        
        # Progress update
        if i % 5 == 0:
            print(f"Progress: {i+1}/{len(pulse_durations)} pulse durations")
    
    print("Pulse sequence example completed")
    return pulse_durations, results


def plot_results(odmr_data, confocal_data, pulse_data):
    """
    Plot the results from the different examples.
    
    @param odmr_data: Tuple of (frequencies, counts) from ODMR example
    @param confocal_data: 2D array from confocal example
    @param pulse_data: Tuple of (pulse_durations, results) from pulse example
    """
    plt.figure(figsize=(15, 5))
    
    # Plot ODMR data
    plt.subplot(1, 3, 1)
    frequencies, counts = odmr_data
    plt.plot(frequencies / 1e9, counts, 'b.-')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Counts')
    plt.title('ODMR Spectrum')
    plt.grid(True)
    
    # Plot confocal data
    plt.subplot(1, 3, 2)
    plt.imshow(confocal_data, cmap='viridis', origin='lower',
              extent=[-10, 10, -10, 10])  # ±10 μm
    plt.colorbar(label='Counts')
    plt.xlabel('X Position (μm)')
    plt.ylabel('Y Position (μm)')
    plt.title('Confocal Scan')
    
    # Plot pulse data
    plt.subplot(1, 3, 3)
    durations, results = pulse_data
    plt.plot(durations * 1e9, results, 'r.-')
    plt.xlabel('Pulse Duration (ns)')
    plt.ylabel('Readout Counts')
    plt.title('Rabi Oscillation')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('simulator_examples.png')
    plt.show()


def main():
    """Main function to run all examples"""
    # Setup the simulator
    facade, microwave, fast_counter, pulser, scanning_probe, laser = setup_simulator()
    
    try:
        # Run examples
        odmr_data = run_odmr_example(microwave, fast_counter, laser)
        confocal_data = run_confocal_example(scanning_probe, laser)
        pulse_data = run_pulse_example(pulser, fast_counter, laser)
        
        # Plot the results
        plot_results(odmr_data, confocal_data, pulse_data)
        
    finally:
        # Clean up
        microwave.on_deactivate()
        fast_counter.on_deactivate()
        pulser.on_deactivate()
        scanning_probe.on_deactivate()
        laser.on_deactivate()
        print("\nAll modules deactivated")


if __name__ == '__main__':
    main()