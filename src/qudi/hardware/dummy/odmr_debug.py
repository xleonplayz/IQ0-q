#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ODMR Debugging Utility for Qudi NV Simulator

This module provides diagnostic tools to help debug ODMR measurements
when using the NV simulator.

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
import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s'))
logger.addHandler(handler)

def test_odmr_simulation():
    """Test the NV simulator ODMR response independently."""
    logger.info("Testing ODMR simulation...")
    
    try:
        # Try to import the NV simulator manager
        from nv_simulator_manager import NVSimulatorManager
        
        # Create the simulator with 500G field
        sim = NVSimulatorManager(magnetic_field=[0, 0, 500], temperature=300, use_simulator=True)
        logger.info("Successfully initialized NV simulator manager")
        
        # Calculate ODMR resonance frequencies
        zero_field = 2.87e9  # Hz
        b_field_gauss = 500  # Gauss
        zeeman_shift = 2.8e6 * b_field_gauss  # Hz (2.8 MHz/G)
        
        # Expected resonances
        res1 = zero_field - zeeman_shift
        res2 = zero_field + zeeman_shift
        
        logger.info(f"Expected resonances at: {res1/1e9:.4f} GHz and {res2/1e9:.4f} GHz")
        
        # Scan a wide frequency range
        freq_min = 1.0e9   # 1.0 GHz
        freq_max = 5.0e9   # 5.0 GHz
        n_points = 1000    # 1000 points
        
        # Simulate ODMR
        logger.info(f"Simulating ODMR from {freq_min/1e9:.2f} GHz to {freq_max/1e9:.2f} GHz with {n_points} points")
        result = sim.simulate_odmr(freq_min, freq_max, n_points)
        
        # Check if result exists
        if result is None or 'signal' not in result or 'frequencies' not in result:
            logger.error("Simulation returned no valid data!")
            return
            
        # Check signal properties
        frequencies = result['frequencies']
        signal = result['signal']
        
        logger.info(f"Frequency range: {frequencies[0]/1e9:.4f} GHz to {frequencies[-1]/1e9:.4f} GHz")
        logger.info(f"Signal range: {np.min(signal):.1f} to {np.max(signal):.1f}")
        logger.info(f"Signal shape: {signal.shape}")
        
        # Determine if there are dips in the signal
        baseline = np.percentile(signal, 95)  # Top 5% as baseline
        min_val = np.min(signal)
        contrast = (baseline - min_val) / baseline
        
        logger.info(f"Signal baseline: {baseline:.1f}")
        logger.info(f"Signal minimum: {min_val:.1f}")
        logger.info(f"Contrast: {contrast:.1%}")
        
        if contrast < 0.05:
            logger.warning("Signal contrast is very low (< 5%). ODMR dips may not be visible.")
        
        # Find and report dips
        from scipy import signal as scipysignal
        
        # Smooth the signal to avoid minor fluctuations
        smoothed = scipysignal.savgol_filter(signal, 51, 3)
        
        # Find peaks (which are dips in our case as ODMR shows fluorescence dips)
        peaks, properties = scipysignal.find_peaks(-smoothed, prominence=baseline*0.05)
        
        if len(peaks) == 0:
            logger.warning("No ODMR dips detected in the simulated signal!")
        else:
            logger.info(f"Detected {len(peaks)} ODMR dips at:")
            for peak in peaks:
                freq = frequencies[peak]
                logger.info(f"  {freq/1e9:.4f} GHz (signal: {signal[peak]:.1f})")
        
        # Plot the results
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies/1e9, signal, 'b-', label='ODMR Signal')
        
        # Mark the expected resonances
        plt.axvline(x=res1/1e9, color='r', linestyle='--', label=f'Expected Res 1: {res1/1e9:.4f} GHz')
        plt.axvline(x=res2/1e9, color='g', linestyle='--', label=f'Expected Res 2: {res2/1e9:.4f} GHz')
        
        # Mark the detected dips
        for peak in peaks:
            plt.plot(frequencies[peak]/1e9, signal[peak], 'ro')
        
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Fluorescence (counts/s)')
        plt.title('NV Simulator ODMR Response Test')
        plt.legend()
        plt.grid(True)
        
        # Save plot to a file
        plot_file = os.path.join(os.getcwd(), 'odmr_test_plot.png')
        plt.savefig(plot_file)
        logger.info(f"Saved plot to: {plot_file}")
        
        # Show the plot if possible (may fail in some environments)
        try:
            plt.show()
        except:
            pass
        
        return {
            'frequencies': frequencies,
            'signal': signal,
            'expected_resonances': [res1, res2],
            'detected_peaks': [frequencies[p] for p in peaks],
            'contrast': contrast
        }
        
    except Exception as e:
        logger.error(f"Error in test_odmr_simulation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def test_odmr_frequency_range():
    """Test if the OdmrLogic is scanning the correct frequency range."""
    logger.info("Checking ODMR frequency range from OdmrLogic...")
    
    try:
        # Try to import OdmrLogic
        from qudi.logic.odmr_logic import OdmrLogic
        
        # Import StatusVar to get default values
        from qudi.core.statusvariable import StatusVar
        
        # Get the default scan frequency ranges from the OdmrLogic class
        for attr_name in dir(OdmrLogic):
            if attr_name == '_scan_frequency_ranges':
                attr = getattr(OdmrLogic, attr_name)
                if isinstance(attr, StatusVar):
                    ranges = attr.default
                    logger.info(f"Default ODMR scan ranges: {ranges}")
                    
                    # Analyze the ranges
                    for i, r in enumerate(ranges):
                        start, stop, points = r
                        logger.info(f"Range {i+1}: {start/1e9:.4f} GHz to {stop/1e9:.4f} GHz with {points} points")
                        
                        # Check if range covers expected resonances
                        zero_field = 2.87e9  # Hz
                        b_field_gauss = 500  # Gauss
                        zeeman_shift = 2.8e6 * b_field_gauss  # Hz (2.8 MHz/G)
                        
                        # Expected resonances
                        res1 = zero_field - zeeman_shift
                        res2 = zero_field + zeeman_shift
                        
                        if start <= res1 <= stop and start <= res2 <= stop:
                            logger.info(f"Range covers both expected resonances: {res1/1e9:.4f} GHz and {res2/1e9:.4f} GHz")
                        elif start <= res1 <= stop:
                            logger.warning(f"Range only covers lower resonance: {res1/1e9:.4f} GHz")
                        elif start <= res2 <= stop:
                            logger.warning(f"Range only covers upper resonance: {res2/1e9:.4f} GHz")
                        else:
                            logger.error(f"Range does not cover any expected resonances!")
                            logger.error(f"Resonances should be at {res1/1e9:.4f} GHz and {res2/1e9:.4f} GHz")
        
        return True
    except Exception as e:
        logger.error(f"Error in test_odmr_frequency_range: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("Running ODMR diagnostics...")
    
    # Test frequency range
    test_odmr_frequency_range()
    
    # Test ODMR simulation
    result = test_odmr_simulation()
    
    if result:
        logger.info("ODMR tests completed. If you still don't see ODMR signal:")
        logger.info("1. Check that your ODMR frequency range includes the resonances")
        logger.info("2. Ensure the microwave power is high enough (try -10 dBm)")
        logger.info("3. Verify that the simulator is properly initialized")
        logger.info("4. Check the log files for errors during ODMR scanning")
    else:
        logger.error("ODMR tests failed. See errors above.")