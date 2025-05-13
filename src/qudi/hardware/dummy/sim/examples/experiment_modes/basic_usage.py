#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example demonstrating the use of the experiment modes in the NV center simulator.

This script shows how to use the experiment modes framework to run various
quantum experiments with the NV center simulator.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add the root directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
sys.path.insert(0, root_dir)

# Import the simulator device
from sim.src.qudi_interface.simulator_device import NVSimulatorDevice

# Create the simulator device
simulator = NVSimulatorDevice()

# Function to plot experiment results
def plot_experiment_result(result, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    
    if 'odmr_signal' in result:
        # ODMR experiment
        x_data = result['frequencies'] / 1e9  # Convert to GHz
        y_data = result['odmr_signal']
        xlabel = 'Frequency (GHz)'
    elif 'times' in result and 'signal' in result:
        # Rabi or T1 experiment
        x_data = result['times'] * 1e6  # Convert to 탎
        y_data = result['signal']
        xlabel = 'Time (탎)'
    
    plt.plot(x_data, y_data, 'o-')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    return plt

# Configure the simulator
simulator.configure_simulator(
    magnetic_field=[0, 0, 0.05],  # 500 G along z-axis
    laser_power=1.0
)

print("Running ODMR experiment...")
# Run ODMR experiment
odmr_result = simulator.run_experiment('odmr', 
                                      freq_start=2.82e9, 
                                      freq_stop=2.92e9, 
                                      num_points=101,
                                      power=-10.0, 
                                      avg_count=3,
                                      noise_level=0.01)

# Plot ODMR result
odmr_plot = plot_experiment_result(odmr_result, 
                                  'ODMR Spectrum', 
                                  'Frequency (GHz)', 
                                  'Fluorescence (a.u.)')
odmr_plot.savefig(os.path.join(script_dir, 'odmr_result.png'))

print("Running Rabi experiment...")
# Run Rabi experiment
rabi_times = np.linspace(0, 500e-9, 51)  # 0 to 500 ns
rabi_result = simulator.run_experiment('rabi',
                                      rabi_times=rabi_times,
                                      mw_frequency=2.87e9,
                                      mw_power=-10.0,
                                      avg_count=3,
                                      noise_level=0.01)

# Plot Rabi result
rabi_plot = plot_experiment_result(rabi_result, 
                                  'Rabi Oscillations', 
                                  'Time (탎)', 
                                  'Population')
rabi_plot.savefig(os.path.join(script_dir, 'rabi_result.png'))

print("Running T1 relaxation experiment...")
# Run T1 experiment with logarithmic time spacing
t1_times = np.logspace(-7, -3, 51)  # 100 ns to 1 ms
t1_result = simulator.run_experiment('t1',
                                    tau_times=t1_times,
                                    avg_count=3,
                                    noise_level=0.01)

# Plot T1 result
plt.figure(figsize=(10, 6))
plt.semilogx(t1_result['times'] * 1e6, t1_result['signal'], 'o-')
plt.title('T1 Relaxation')
plt.xlabel('Time (탎)')
plt.ylabel('Population')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 't1_result.png'))

print("Running Ramsey experiment...")
# Run Ramsey experiment
ramsey_times = np.linspace(0, 5e-6, 51)  # 0 to 5 탎
ramsey_result = simulator.run_experiment('ramsey',
                                        tau_times=ramsey_times,
                                        mw_frequency=2.87e9,
                                        detuning=5e6,  # 5 MHz detuning
                                        mw_power=-10.0,
                                        avg_count=3,
                                        noise_level=0.01)

# Plot Ramsey result
ramsey_plot = plot_experiment_result(ramsey_result, 
                                    'Ramsey Interference', 
                                    'Time (탎)', 
                                    'Signal')
ramsey_plot.savefig(os.path.join(script_dir, 'ramsey_result.png'))

print("Running Spin Echo experiment...")
# Run Spin Echo experiment
echo_times = np.linspace(50e-9, 10e-6, 51)  # 50 ns to 10 탎
echo_result = simulator.run_experiment('spin_echo',
                                      tau_times=echo_times,
                                      mw_frequency=2.87e9,
                                      mw_power=-10.0,
                                      avg_count=3,
                                      noise_level=0.01)

# Plot Spin Echo result
echo_plot = plot_experiment_result(echo_result, 
                                  'Spin Echo', 
                                  'Time (탎)', 
                                  'Signal')
echo_plot.savefig(os.path.join(script_dir, 'spin_echo_result.png'))

print("Running CPMG experiment...")
# Run CPMG experiment using the custom sequence mode
cpmg_times = np.linspace(100e-9, 20e-6, 51)  # 100 ns to 20 탎
cpmg_result = simulator.run_experiment('custom',
                                      sequence_name='cpmg',
                                      tau_times=cpmg_times,
                                      mw_frequency=2.87e9,
                                      mw_power=-10.0,
                                      avg_count=3,
                                      noise_level=0.01,
                                      sequence_params={'n_pulses': 8})

# Plot CPMG result
cpmg_plot = plot_experiment_result(cpmg_result, 
                                  'CPMG Sequence (8 pulses)', 
                                  'Time (탎)', 
                                  'Signal')
cpmg_plot.savefig(os.path.join(script_dir, 'cpmg_result.png'))

print("All experiments completed. Results saved as PNG files.")