#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Start script for running Qudi with the NV simulator configuration.

This script provides a convenient way to launch Qudi with the NV simulator
configuration for ODMR measurements. It verifies that the simulator is
working properly before starting Qudi.

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
import logging
import subprocess
import importlib.util
import time
import argparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('NV_Simulator_Launcher')

def verify_nv_simulator():
    """Verify that the NV simulator integration is working."""
    logger.info("Verifying NV simulator integration...")
    
    # Check if nv_simulator_manager.py exists
    nv_sim_path = os.path.join('src', 'qudi', 'hardware', 'dummy', 'nv_simulator_manager.py')
    if not os.path.exists(nv_sim_path):
        logger.error(f"NV simulator manager not found at {nv_sim_path}")
        return False
    
    # Attempt to import the NV simulator manager
    try:
        sys.path.insert(0, os.path.join('src', 'qudi', 'hardware', 'dummy'))
        from nv_simulator_manager import NVSimulatorManager
        logger.info("Successfully imported NVSimulatorManager")
        
        # Create the simulator instance
        nv_sim = NVSimulatorManager(magnetic_field=[0, 0, 500], temperature=300, use_simulator=True)
        logger.info("Successfully created NVSimulatorManager instance")
        
        # Calculate expected resonances
        zfs = 2.87e9  # Zero-field splitting (Hz)
        gyro = 2.8e6  # Gyromagnetic ratio (Hz/G)
        field = 500.0  # Field in Gauss
        
        zeeman_shift = gyro * field
        dip1_center = zfs - zeeman_shift
        dip2_center = zfs + zeeman_shift
        
        logger.info(f"Magnetic field: {field} G")
        logger.info(f"Expected resonances at:")
        logger.info(f"  ms=0 -> ms=-1: {dip1_center/1e9:.6f} GHz")
        logger.info(f"  ms=0 -> ms=+1: {dip2_center/1e9:.6f} GHz")
        
        return True
    except Exception as e:
        logger.error(f"Error verifying NV simulator: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def check_qudi_available():
    """Check if Qudi is available on the system."""
    try:
        # Try to find the qudi package
        spec = importlib.util.find_spec("qudi")
        if spec is None:
            logger.warning("Qudi package not found in Python path")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error checking for Qudi: {e}")
        return False

def run_qudi_with_config(config_path):
    """Run Qudi with the specified configuration file."""
    logger.info(f"Starting Qudi with configuration: {config_path}")
    
    try:
        # Check if the config file exists
        if not os.path.exists(config_path):
            logger.error(f"Configuration file not found: {config_path}")
            return False
        
        # Run Qudi with the specified config
        if check_qudi_available():
            logger.info("Running Qudi as a Python module...")
            cmd = [sys.executable, "-m", "qudi", "-c", config_path]
        else:
            logger.info("Running Qudi using the qudi command...")
            cmd = ["qudi", "-c", config_path]
            
        logger.info(f"Command: {' '.join(cmd)}")
        
        # Run Qudi
        return subprocess.call(cmd)
    except Exception as e:
        logger.error(f"Error running Qudi: {e}")
        return False

def main():
    """Main function to start Qudi with the NV simulator."""
    parser = argparse.ArgumentParser(description='Start Qudi with the NV simulator configuration')
    parser.add_argument('-c', '--config', type=str, 
                        default='src/qudi/hardware/dummy/nv_simulator_config.cfg',
                        help='Path to the configuration file')
    parser.add_argument('--skip-verify', action='store_true',
                        help='Skip NV simulator verification')
    args = parser.parse_args()
    
    logger.info("Starting NV simulator launcher...")
    
    # Verify NV simulator integration
    if not args.skip_verify:
        if not verify_nv_simulator():
            logger.error("NV simulator verification failed")
            logger.info("Try running the script with --skip-verify to bypass the check")
            return 1
    else:
        logger.info("Skipping NV simulator verification")
    
    # Run Qudi with the NV simulator configuration
    return run_qudi_with_config(args.config)

if __name__ == "__main__":
    sys.exit(main())