#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Runner script for ODMR flow test with result visualization.

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
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Import environment setup to ensure consistent test environment
from env_setup import env_info

# Import the test module
from test_odmr_flow import run_odmr_flow_test, logger


def analyze_log_file(log_file_path):
    """Analyze the log file for key information about the ODMR test."""
    try:
        if not os.path.exists(log_file_path):
            print(f"Log file not found: {log_file_path}")
            return None
            
        # Parse log file for data
        scan_frequencies = []
        mw_frequency_updates = []
        scan_indices = []
        detected_resonances = []
        
        with open(log_file_path, 'r') as f:
            for line in f:
                try:
                    # Extract scan frequency information
                    if "[SCAN DEBUG] Setting frequency to" in line:
                        parts = line.split("Setting frequency to ")
                        if len(parts) > 1:
                            freq_str = parts[1].split(" GHz")[0]
                            freq = float(freq_str) * 1e9  # Convert GHz to Hz
                            mw_frequency_updates.append(freq)
                    
                    # Extract scan index information
                    if "[SCAN DEBUG] Moving to frequency index" in line:
                        parts = line.split("Moving to frequency index ")
                        if len(parts) > 1:
                            index = int(parts[1].strip())
                            scan_indices.append(index)
                    
                    # Extract resonance information
                    if "Resonance at " in line and "Value:" in line:
                        parts = line.split("Resonance at ")
                        if len(parts) > 1:
                            freq_str = parts[1].split(" GHz")[0]
                            value_str = parts[1].split("Value: ")[1].strip()
                            resonance = (float(freq_str) * 1e9, float(value_str))
                            detected_resonances.append(resonance)
                    
                    # Extract scan frequencies configuration
                    if "Scan frequencies: " in line and "[(1." in line:
                        parts = line.split("Scan frequencies: ")[1]
                        # Parse the frequency ranges, e.g., [(1.4e+09, 4.4e+09, 301)]
                        # This is a bit hacky but works for this specific format
                        for part in parts.strip("[]").split("), ("):
                            freq_parts = part.strip("()").split(", ")
                            if len(freq_parts) == 3:
                                try:
                                    start = float(freq_parts[0])
                                    stop = float(freq_parts[1])
                                    num = int(float(freq_parts[2]))
                                    scan_frequencies.append((start, stop, num))
                                except ValueError:
                                    continue
                            
                    # Extract ODMR signal information at specific frequencies
                    # This is more complex and might need custom parsing depending on log format
                
                except Exception as e:
                    print(f"Error parsing line: {line.strip()}")
                    print(f"Error: {e}")
                    continue
                
        return {
            'scan_frequencies': scan_frequencies,
            'mw_frequency_updates': mw_frequency_updates,
            'scan_indices': scan_indices,
            'detected_resonances': detected_resonances
        }
    
    except Exception as e:
        print(f"Error analyzing log file: {e}")
        return None


def plot_results(analysis_data, output_path=None):
    """Plot the ODMR test results for visualization."""
    if not analysis_data:
        print("No analysis data to plot")
        return
        
    try:
        # Create figure
        fig = plt.figure(figsize=(12, 10))
        gs = GridSpec(3, 2, figure=fig)
        
        # Plot scan frequency ranges
        ax1 = fig.add_subplot(gs[0, 0])
        if analysis_data['scan_frequencies']:
            for i, (start, stop, num) in enumerate(analysis_data['scan_frequencies']):
                ax1.plot([i, i], [start/1e9, stop/1e9], 'b-', linewidth=2)
                ax1.text(i, stop/1e9, f"{num} pts", ha='center', va='bottom')
            ax1.set_title("Configured Frequency Ranges")
            ax1.set_xlabel("Range Index")
            ax1.set_ylabel("Frequency (GHz)")
            ax1.grid(True)
        else:
            ax1.text(0.5, 0.5, "No scan frequency data", ha='center', va='center', transform=ax1.transAxes)
        
        # Plot microwave frequency updates
        ax2 = fig.add_subplot(gs[0, 1])
        if analysis_data['mw_frequency_updates']:
            freqs = analysis_data['mw_frequency_updates']
            ax2.plot(freqs, 'r-')
            ax2.set_title(f"Microwave Frequency Updates ({len(freqs)} points)")
            ax2.set_xlabel("Update Index")
            ax2.set_ylabel("Frequency (Hz)")
            ax2.ticklabel_format(axis='y', style='sci', scilimits=(9, 9))
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, "No frequency update data", ha='center', va='center', transform=ax2.transAxes)
        
        # Plot scan indices over time
        ax3 = fig.add_subplot(gs[1, 0])
        if analysis_data['scan_indices']:
            indices = analysis_data['scan_indices']
            ax3.plot(indices, 'g-')
            ax3.set_title(f"Scan Indices ({len(indices)} updates)")
            ax3.set_xlabel("Update Count")
            ax3.set_ylabel("Scan Index")
            ax3.grid(True)
        else:
            ax3.text(0.5, 0.5, "No scan index data", ha='center', va='center', transform=ax3.transAxes)
        
        # Plot detected resonances
        ax4 = fig.add_subplot(gs[1, 1])
        if analysis_data['detected_resonances']:
            resonances = analysis_data['detected_resonances']
            freqs = [r[0]/1e9 for r in resonances]
            values = [r[1] for r in resonances]
            
            # Sort by frequency for better visualization
            sorted_data = sorted(zip(freqs, values))
            freqs = [x[0] for x in sorted_data]
            values = [x[1] for x in sorted_data]
            
            ax4.plot(freqs, values, 'bD-')
            for i, (f, v) in enumerate(zip(freqs, values)):
                ax4.text(f, v, f"{f:.3f}", ha='center', va='bottom')
            ax4.set_title(f"Detected Resonances ({len(resonances)} found)")
            ax4.set_xlabel("Frequency (GHz)")
            ax4.set_ylabel("Signal Value")
            ax4.grid(True)
        else:
            ax4.text(0.5, 0.5, "No resonances detected", ha='center', va='center', transform=ax4.transAxes)
        
        # Calculate expected ODMR signal based on 500G magnetic field
        ax5 = fig.add_subplot(gs[2, :])
        try:
            # Constants for NV center
            zfs = 2.87e9  # Zero-field splitting (Hz)
            gyro = 2.8e6  # Gyromagnetic ratio (Hz/G)
            field = 500   # Magnetic field (G)
            
            # Calculate expected resonances
            zeeman_shift = gyro * field
            dip1_center = zfs - zeeman_shift
            dip2_center = zfs + zeeman_shift
            
            # Generate model ODMR spectrum
            freq_range = np.linspace(1.4e9, 4.4e9, 1000)
            linewidth = 20e6  # 20 MHz linewidth
            contrast = 0.3    # 30% contrast
            baseline = 1.0
            
            # Lorentzian dips
            dip1 = contrast * linewidth**2 / ((freq_range - dip1_center)**2 + linewidth**2)
            dip2 = contrast * linewidth**2 / ((freq_range - dip2_center)**2 + linewidth**2)
            
            # Combined ODMR signal
            odmr_signal = baseline - dip1 - dip2
            
            # Plot the model spectrum
            ax5.plot(freq_range/1e9, odmr_signal)
            ax5.axvline(x=dip1_center/1e9, color='r', linestyle='--', label=f'Expected: {dip1_center/1e9:.3f} GHz')
            ax5.axvline(x=dip2_center/1e9, color='r', linestyle='--', label=f'Expected: {dip2_center/1e9:.3f} GHz')
            
            # Plot detected resonances if available
            if analysis_data['detected_resonances']:
                for freq, value in analysis_data['detected_resonances']:
                    ax5.axvline(x=freq/1e9, color='g', linestyle='-', alpha=0.5)
                    
            ax5.set_title("Expected ODMR Signal for 500 Gauss Field")
            ax5.set_xlabel("Frequency (GHz)")
            ax5.set_ylabel("Normalized Signal")
            ax5.grid(True)
            ax5.legend()
            
        except Exception as e:
            print(f"Error generating ODMR model: {e}")
            ax5.text(0.5, 0.5, "Error generating ODMR model", ha='center', va='center', transform=ax5.transAxes)
        
        fig.tight_layout()
        
        # Save the figure if output path provided
        if output_path:
            fig.savefig(output_path)
            print(f"Results saved to {output_path}")
        
        # Show the figure
        plt.show()
        
    except Exception as e:
        print(f"Error plotting results: {e}")


def run_test_and_visualize():
    """Run the ODMR flow test and visualize the results."""
    try:
        # Run the test
        print("=== Running ODMR Flow Test ===")
        # Import the original function to patch it before running
        import types
        import test_odmr_flow
        
        # Patch the QudiFacade creation in run_odmr_flow_test
        original_run_test = test_odmr_flow.run_odmr_flow_test
        def patched_run_test():
            """Patched version that ensures proper QudiFacade parameters"""
            try:
                # Before running the test, reset the QudiFacade singleton
                from qudi.hardware.nv_simulator.qudi_facade import QudiFacade
                QudiFacade.reset_instance()
                print("Reset QudiFacade singleton for clean test")
                
                # Now run the original test
                original_run_test()
            except Exception as e:
                print(f"Error running test: {e}")
        
        # Replace the original function with our patched version
        test_odmr_flow.run_odmr_flow_test = patched_run_test
        
        # Now run the patched function
        run_odmr_flow_test()
        
        # Wait a moment for files to be flushed
        time.sleep(1)
        
        # Analyze the log file
        log_file_path = os.path.join(current_dir, 'odmr_flow_test.log')
        print(f"\n=== Analyzing log file: {log_file_path} ===")
        analysis_data = analyze_log_file(log_file_path)
        
        if analysis_data:
            # Generate output path for plots
            output_dir = os.path.join(current_dir, 'results')
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f'odmr_test_results_{time.strftime("%Y%m%d_%H%M%S")}.png')
            
            # Plot and save results
            print("\n=== Plotting Results ===")
            plot_results(analysis_data, output_path)
        else:
            print("No analysis data available, cannot plot results")
            
    except Exception as e:
        print(f"Error running test and visualizing: {e}")


if __name__ == "__main__":
    run_test_and_visualize()