#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains a self-checking adapter to ensure proper communication between
the microwave device and the ODMR module for SimOS integration.

This is for debugging purposes only.

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
import logging

# Initialize logger
logger = logging.getLogger(__name__)

class SimDebugger:
    """Helper class to debug Simulator issues.
    
    This class provides methods to monitor activity and help track down issues
    with the SimOS integration in Qudi.
    """
    
    @staticmethod
    def test_scan_sequence(mw_device, frequencies, steps=10, power=-20.0):
        """Test a scanning sequence with the provided microwave device.
        
        Args:
            mw_device: The microwave device to test
            frequencies: Either a list of frequencies or a tuple (start, stop, num_points)
            steps: Number of steps to scan if using JUMP_LIST
            power: Power level in dBm
        """
        from qudi.util.enums import SamplingOutputMode
        
        # Default to JUMP_LIST mode
        mode = SamplingOutputMode.JUMP_LIST
        
        # If frequencies is a tuple like (start, stop, num), use EQUIDISTANT_SWEEP
        if isinstance(frequencies, tuple) and len(frequencies) == 3:
            mode = SamplingOutputMode.EQUIDISTANT_SWEEP
            logger.info(f"Testing sweep mode from {frequencies[0]/1e9:.6f} to {frequencies[1]/1e9:.6f} GHz")
        else:
            # Create a list of frequencies if not provided
            if frequencies is None:
                center = 2.87e9  # Zero-field splitting
                span = 1.0e9  # 1 GHz span
                frequencies = np.linspace(center - span/2, center + span/2, steps)
            logger.info(f"Testing jump list mode with {len(frequencies)} frequencies")
        
        # Configure the scan
        mw_device.configure_scan(power, frequencies, mode, 10.0)
        
        # Start the scan
        mw_device.start_scan()
        
        # For JUMP_LIST mode, manually step through frequencies
        if mode == SamplingOutputMode.JUMP_LIST:
            for i in range(len(frequencies)):
                freq = mw_device.get_current_frequency()
                logger.info(f"Current frequency: {freq/1e9:.6f} GHz")
                if callable(getattr(mw_device, 'scan_next', None)):
                    mw_device.scan_next()
                time.sleep(0.2)
        # For EQUIDISTANT_SWEEP, wait for the expected duration
        else:
            start, stop, num = frequencies
            total_time = num / 10.0  # sample_rate is 10.0 Hz
            logger.info(f"Waiting for sweep to complete (expected duration: {total_time:.1f}s)")
            time.sleep(total_time + 1.0)  # Add 1 second margin
        
        # Stop the scan
        mw_device.off()
        logger.info("Scan test completed")
        
    @staticmethod
    def test_odmr_data_acquisition(mw_device, fs_device, frequencies=None, steps=50):
        """Test ODMR data acquisition with the microwave device and finite sampler.
        
        Args:
            mw_device: The microwave device to use
            fs_device: The finite sampling device to use
            frequencies: Either a list of frequencies or a tuple (start, stop, num_points)
            steps: Number of steps to scan
        """
        from qudi.util.enums import SamplingOutputMode
        
        # Configure the finite sampler
        fs_device.set_active_channels(['APD counts'])
        fs_device.set_sample_rate(10.0)  # 10 Hz sampling rate
        fs_device.set_frame_size(steps)
        
        # Default frequencies around NV center zero-field splitting
        if frequencies is None:
            center = 2.87e9  # Zero-field splitting
            span = 1.0e9  # 1 GHz span
            frequencies = (center - span/2, center + span/2, steps)
        
        # Configure the microwave scan
        mode = SamplingOutputMode.EQUIDISTANT_SWEEP
        if not isinstance(frequencies, tuple):
            mode = SamplingOutputMode.JUMP_LIST
        
        mw_device.configure_scan(-20.0, frequencies, mode, 10.0)
        
        # Start data acquisition and MW scan
        fs_device.start_buffered_acquisition()
        mw_device.start_scan()
        
        # Let it run for the expected duration
        if mode == SamplingOutputMode.EQUIDISTANT_SWEEP:
            total_time = steps / 10.0  # sample_rate is 10.0 Hz
        else:
            total_time = len(frequencies) / 10.0
            
        logger.info(f"Waiting for ODMR scan to complete (expected duration: {total_time:.1f}s)")
        time.sleep(total_time / 2)  # Wait for half the expected time
        
        # Check if data is being acquired
        samples = fs_device.samples_in_buffer()
        logger.info(f"Samples in buffer after {total_time/2:.1f}s: {samples}")
        
        # Wait for the rest of the time
        time.sleep(total_time / 2 + 1.0)  # Add 1 second margin
        
        # Get the acquired data
        data = fs_device.get_buffered_samples()
        fs_device.stop_buffered_acquisition()
        mw_device.off()
        
        # Log the data
        for ch, samples in data.items():
            logger.info(f"Channel {ch}: min={np.min(samples):.1f}, max={np.max(samples):.1f}, len={len(samples)}")
        
        logger.info("ODMR test completed")
        return data