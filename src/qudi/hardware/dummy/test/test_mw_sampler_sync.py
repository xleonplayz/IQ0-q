#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Direct test of microwave to sampler synchronization.

This test checks the direct communication between the microwave device and the finite sampler
through the shared state mechanism, bypassing the ODMR logic.

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
import logging
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import qudi modules
current_dir = os.path.dirname(os.path.abspath(__file__))
qudi_dir = os.path.abspath(os.path.join(current_dir, '../../../..'))
if qudi_dir not in sys.path:
    sys.path.insert(0, qudi_dir)

from qudi.hardware.nv_simulator.qudi_facade import QudiFacade
from qudi.hardware.nv_simulator.microwave_device import NVSimMicrowaveDevice
from qudi.hardware.nv_simulator.finite_sampler import NVSimFiniteSampler

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(current_dir, 'mw_sampler_sync_test.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('MW_Sampler_Sync_Test')


class DummyConnector:
    """Dummy connector class for providing module instances."""
    def __init__(self, module_instance):
        self._module = module_instance
    def __call__(self):
        return self._module


def test_direct_frequency_setting():
    """Test direct frequency setting and synchronization."""
    
    logger.info("=== Starting MW-Sampler Sync Test ===")
    
    # Initialize modules
    logger.info("Initializing modules...")
    # Remove test_mode parameter to avoid Qt property error
    qudi_facade = QudiFacade(name="QudiFacade_Test")
    qudi_facade.on_activate()
    
    # Set magnetic field to 500 Gauss along z-axis
    b_field_tesla = [0, 0, 500e-4]  # 500 G = 0.05 T
    qudi_facade.nv_model.set_magnetic_field(b_field_tesla)
    logger.info(f"Set magnetic field to {b_field_tesla} Tesla")
    
    # Initialize microwave device
    microwave_device = NVSimMicrowaveDevice()
    microwave_device.simulator = DummyConnector(qudi_facade)
    microwave_device.on_activate()
    
    # Initialize finite sampler
    finite_sampler = NVSimFiniteSampler()
    finite_sampler.simulator = DummyConnector(qudi_facade)
    finite_sampler.on_activate()
    
    try:
        # Configure finite sampler
        sample_rate = 1000  # Hz
        frame_size = 1000   # Samples
        finite_sampler.configure(['APD counts'], sample_rate, frame_size)
        
        # Calculate expected resonance frequencies
        zfs = 2.87e9  # Zero-field splitting (Hz)
        gyro = 2.8e6  # Gyromagnetic ratio (Hz/G)
        field = 500   # Magnetic field (G)
        
        zeeman_shift = gyro * field
        dip1_center = zfs - zeeman_shift
        dip2_center = zfs + zeeman_shift
        
        logger.info(f"Expected resonance frequencies: {dip1_center/1e9:.6f} GHz and {dip2_center/1e9:.6f} GHz")
        
        # Test frequency sweep from 1.4 GHz to 4.4 GHz
        frequencies = np.linspace(1.4e9, 4.4e9, 31)
        signals = []
        
        # Turn on microwave
        microwave_device.set_cw(frequencies[0], -20)
        microwave_device.cw_on()
        
        # Start sampling
        finite_sampler.start_buffered_acquisition()
        
        # Sweep through frequencies and get data
        for freq in frequencies:
            # Set frequency
            microwave_device.set_cw(freq, -20)
            
            # Wait a bit for internal state to update
            time.sleep(0.05)
            
            # Get current shared state
            current_freq = qudi_facade._shared_state['current_mw_frequency']
            current_mw_on = qudi_facade._shared_state['current_mw_on']
            
            logger.info(f"Setting frequency: {freq/1e9:.6f} GHz")
            logger.info(f"Shared state frequency: {current_freq/1e9:.6f} GHz, MW on: {current_mw_on}")
            
            # Verify update was applied
            if abs(current_freq - freq) > 1:
                logger.error(f"Frequency mismatch! Set: {freq/1e9:.6f} GHz, Got: {current_freq/1e9:.6f} GHz")
            
            # Get sample data
            try:
                sample_data = finite_sampler.get_buffered_data()
                avg_signal = np.mean(sample_data['APD counts'])
                signals.append(avg_signal)
                logger.info(f"Frequency: {freq/1e9:.6f} GHz, Signal: {avg_signal:.2f}")
            except Exception as e:
                logger.error(f"Error getting sample data: {e}")
                signals.append(np.nan)
        
        # Stop sampling
        finite_sampler.stop_buffered_acquisition()
        
        # Turn off microwave
        microwave_device.off()
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(frequencies/1e9, signals, 'b-o')
        plt.axvline(x=dip1_center/1e9, color='r', linestyle='--', label=f'Expected: {dip1_center/1e9:.3f} GHz')
        plt.axvline(x=dip2_center/1e9, color='r', linestyle='--', label=f'Expected: {dip2_center/1e9:.3f} GHz')
        plt.title('Frequency Sweep Test')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Signal (counts)')
        plt.grid(True)
        plt.legend()
        
        # Save the figure
        output_dir = os.path.join(current_dir, 'results')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'mw_sampler_sync_test_{time.strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(output_path)
        logger.info(f"Results saved to {output_path}")
        
        plt.show()
        
    except Exception as e:
        logger.exception(f"Error during test: {e}")
    finally:
        # Cleanup
        try:
            # Deactivate modules
            microwave_device.on_deactivate()
            finite_sampler.on_deactivate()
            qudi_facade.on_deactivate()
        except:
            logger.exception("Error deactivating modules")
            
        logger.info("=== MW-Sampler Sync Test Completed ===")


def test_scan_mode_synchronization():
    """Test scan mode frequency synchronization between the microwave device and finite sampler."""
    
    logger.info("=== Starting Scan Mode Synchronization Test ===")
    
    # Initialize modules
    logger.info("Initializing modules...")
    # Remove test_mode parameter to avoid Qt property error
    qudi_facade = QudiFacade(name="QudiFacade_Test")
    qudi_facade.on_activate()
    
    # Set magnetic field to 500 Gauss along z-axis
    b_field_tesla = [0, 0, 500e-4]  # 500 G = 0.05 T
    qudi_facade.nv_model.set_magnetic_field(b_field_tesla)
    logger.info(f"Set magnetic field to {b_field_tesla} Tesla")
    
    # Initialize microwave device
    microwave_device = NVSimMicrowaveDevice()
    microwave_device.simulator = DummyConnector(qudi_facade)
    microwave_device.on_activate()
    
    # Initialize finite sampler
    finite_sampler = NVSimFiniteSampler()
    finite_sampler.simulator = DummyConnector(qudi_facade)
    finite_sampler.on_activate()
    
    try:
        # Configure finite sampler
        sample_rate = 1000  # Hz
        frame_size = 1000   # Samples
        finite_sampler.configure(['APD counts'], sample_rate, frame_size)
        
        # Calculate expected resonance frequencies
        zfs = 2.87e9  # Zero-field splitting (Hz)
        gyro = 2.8e6  # Gyromagnetic ratio (Hz/G)
        field = 500   # Magnetic field (G)
        
        zeeman_shift = gyro * field
        dip1_center = zfs - zeeman_shift
        dip2_center = zfs + zeeman_shift
        
        logger.info(f"Expected resonance frequencies: {dip1_center/1e9:.6f} GHz and {dip2_center/1e9:.6f} GHz")
        
        # Configure scan range (from 1.4 GHz to 4.4 GHz, 31 points)
        frequencies = np.linspace(1.4e9, 4.4e9, 31)
        
        # Configure microwave scan
        from qudi.util.enums import SamplingOutputMode
        microwave_device.configure_scan(-20, frequencies, SamplingOutputMode.JUMP_LIST, sample_rate)
        
        # Start sampling
        finite_sampler.start_buffered_acquisition()
        
        # Start scan
        microwave_device.start_scan()
        
        # Collect data points
        signals = []
        actual_freqs = []
        
        max_points = len(frequencies)
        for i in range(max_points):
            # Get current scan state from shared state
            scan_active = qudi_facade._shared_state['scanning_active']
            scan_index = qudi_facade._shared_state['current_scan_index']
            current_freq = qudi_facade._shared_state['current_mw_frequency']
            
            logger.info(f"Scan point {i}, Active: {scan_active}, Index: {scan_index}, Frequency: {current_freq/1e9:.6f} GHz")
            
            # Get sample data
            try:
                sample_data = finite_sampler.get_buffered_data()
                avg_signal = np.mean(sample_data['APD counts'])
                signals.append(avg_signal)
                actual_freqs.append(current_freq)
                logger.info(f"Signal: {avg_signal:.2f}")
            except Exception as e:
                logger.error(f"Error getting sample data: {e}")
                signals.append(np.nan)
                actual_freqs.append(np.nan)
            
            # Move to next frequency
            if not microwave_device.scan_next():
                logger.info(f"Scan completed after {i+1} points")
                break
                
            # Wait a bit for internal state to update
            time.sleep(0.05)
        
        # Stop sampling
        finite_sampler.stop_buffered_acquisition()
        
        # Turn off microwave
        microwave_device.off()
        
        # Plot results
        plt.figure(figsize=(10, 6))
        plt.plot(np.array(actual_freqs)/1e9, signals, 'b-o')
        plt.axvline(x=dip1_center/1e9, color='r', linestyle='--', label=f'Expected: {dip1_center/1e9:.3f} GHz')
        plt.axvline(x=dip2_center/1e9, color='r', linestyle='--', label=f'Expected: {dip2_center/1e9:.3f} GHz')
        plt.title('Scan Mode Test')
        plt.xlabel('Frequency (GHz)')
        plt.ylabel('Signal (counts)')
        plt.grid(True)
        plt.legend()
        
        # Save the figure
        output_dir = os.path.join(current_dir, 'results')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'scan_mode_test_{time.strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(output_path)
        logger.info(f"Results saved to {output_path}")
        
        plt.show()
        
    except Exception as e:
        logger.exception(f"Error during test: {e}")
    finally:
        # Cleanup
        try:
            # Deactivate modules
            microwave_device.on_deactivate()
            finite_sampler.on_deactivate()
            qudi_facade.on_deactivate()
        except:
            logger.exception("Error deactivating modules")
            
        logger.info("=== Scan Mode Synchronization Test Completed ===")


if __name__ == "__main__":
    # Run both tests
    test_direct_frequency_setting()
    test_scan_mode_synchronization()