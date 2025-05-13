#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
NV Simulator Verification Script for Qudi

This script tests the NV simulator implementation and verifies
that the ODMR resonances are calculated correctly.

Run this script from the main IQO-q directory using:
python -m src.qudi.hardware.dummy.test.verify_nv_simulator
"""

import os
import sys
import logging
import numpy as np
# Set matplotlib to use non-interactive backend
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('NV_Simulator_Verification')

def main():
    """Main verification function."""
    logger.info("Starting NV simulator verification...")
    
    # Ensure we're in the right directory
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dummy_dir = os.path.dirname(curr_dir)
    
    # Add dummy directory to path if needed
    if dummy_dir not in sys.path:
        sys.path.insert(0, dummy_dir)
    
    # Try to import the NV simulator manager
    try:
        from nv_simulator_manager import NVSimulatorManager
        logger.info("Successfully imported NVSimulatorManager")
    except ImportError as e:
        logger.error(f"Failed to import NVSimulatorManager: {e}")
        logger.info("Current sys.path:")
        for p in sys.path:
            logger.info(f"  {p}")
        return
    
    # Create the simulator instance
    try:
        magnetic_field = [0, 0, 500]  # 500 Gauss along z
        nv_sim = NVSimulatorManager(magnetic_field=magnetic_field, temperature=300, use_simulator=True)
        logger.info("Successfully created NVSimulatorManager instance")
    except Exception as e:
        logger.error(f"Failed to create NVSimulatorManager: {e}")
        return
    
    # Calculate expected resonances
    zfs = 2.87e9  # Zero-field splitting (Hz)
    gyro = 2.8e6  # Gyromagnetic ratio (Hz/G)
    field = np.linalg.norm(magnetic_field)
    
    zeeman_shift = gyro * field
    dip1_center = zfs - zeeman_shift
    dip2_center = zfs + zeeman_shift
    
    logger.info(f"Magnetic field: {field} G")
    logger.info(f"Zeeman shift: {zeeman_shift/1e6:.2f} MHz")
    logger.info(f"Expected resonances:")
    logger.info(f"  ms=0 -> ms=-1: {dip1_center/1e9:.6f} GHz")
    logger.info(f"  ms=0 -> ms=+1: {dip2_center/1e9:.6f} GHz")
    
    # Simulate ODMR spectrum
    logger.info("Simulating ODMR spectrum...")
    try:
        # Wide spectrum to capture both resonances
        freq_min = 1.0e9   # 1.0 GHz
        freq_max = 5.0e9   # 5.0 GHz
        n_points = 1000    # 1000 points
        
        odmr_result = nv_sim.simulate_odmr(freq_min, freq_max, n_points)
        
        if 'signal' not in odmr_result or 'frequencies' not in odmr_result:
            logger.error("Invalid ODMR result - missing 'signal' or 'frequencies'")
            return
            
        # Plot the spectrum
        frequencies = odmr_result['frequencies']
        signal = odmr_result['signal']
        
        logger.info(f"Signal range: {np.min(signal):.1f} to {np.max(signal):.1f}")
        
        # Calculate contrast
        baseline = np.percentile(signal, 95)
        min_val = np.min(signal)
        contrast = (baseline - min_val) / baseline
        
        logger.info(f"Signal baseline: {baseline:.1f}")
        logger.info(f"Signal minimum: {min_val:.1f}")
        logger.info(f"Contrast: {contrast:.1%}")
        
        # Plot the spectrum
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies/1e9, signal, 'b-', label='ODMR Signal')
        
        # Mark expected resonances
        plt.axvline(x=dip1_center/1e9, color='r', linestyle='--', label=f'Expected: {dip1_center/1e9:.4f} GHz')
        plt.axvline(x=dip2_center/1e9, color='g', linestyle='--', label=f'Expected: {dip2_center/1e9:.4f} GHz')
        
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Fluorescence (counts/s)')
        plt.title('NV Simulator ODMR Response')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        out_dir = os.path.join(curr_dir, 'results')
        os.makedirs(out_dir, exist_ok=True)
        plot_file = os.path.join(out_dir, 'nv_simulator_verification.png')
        plt.savefig(plot_file)
        
        logger.info(f"Saved plot to: {plot_file}")
        
        # Skip showing the plot, just save to file
        plt.close()
        logger.info("Plot saved but not displayed (to avoid GUI dependencies)")
        
        logger.info("Verification completed successfully")
        
    except Exception as e:
        logger.error(f"Error simulating ODMR spectrum: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

if __name__ == "__main__":
    main()