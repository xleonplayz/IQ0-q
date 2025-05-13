#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test for the shared_state mechanism in QudiFacade.
Focuses specifically on testing the frequency update communication.

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

# Import environment setup to ensure consistent test environment
from env_setup import env_info

from qudi.hardware.nv_simulator.qudi_facade import QudiFacade
from qudi.hardware.nv_simulator.microwave_device import NVSimMicrowaveDevice
from qudi.hardware.nv_simulator.finite_sampler import NVSimFiniteSampler

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'shared_state_test.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('Shared_State_Test')


class DummyConnector:
    """Dummy connector class for providing module instances."""
    def __init__(self, module_instance):
        self._module = module_instance
    def __call__(self):
        return self._module


def test_direct_shared_state_update():
    """Test direct update and read from shared state."""
    
    logger.info("=== Starting Direct Shared State Test ===")
    
    # Initialize QudiFacade
    logger.info("Initializing QudiFacade...")
    qudi_facade = QudiFacade(name="SharedStateTest")
    qudi_facade.on_activate()
    
    # Test 1: Direct shared state manipulation
    logger.info("\nTest 1: Direct shared state manipulation")
    
    # Set values directly in shared state
    test_frequency = 3.142e9  # ~3.14 GHz
    test_power = -10.0
    
    # Check current values first
    initial_freq = qudi_facade._shared_state['current_mw_frequency']
    initial_power = qudi_facade._shared_state['current_mw_power']
    logger.info(f"Initial shared state values: frequency={initial_freq/1e9:.6f} GHz, power={initial_power} dBm")
    
    # Modify directly
    qudi_facade._shared_state['current_mw_frequency'] = test_frequency
    qudi_facade._shared_state['current_mw_power'] = test_power
    logger.info(f"Set shared state directly: frequency={test_frequency/1e9:.6f} GHz, power={test_power} dBm")
    
    # Read back values
    current_freq = qudi_facade.get_current_frequency()
    current_power = qudi_facade.get_current_power()
    logger.info(f"Read through accessor methods: frequency={current_freq/1e9:.6f} GHz, power={current_power} dBm")
    
    # Verify values match
    if abs(current_freq - test_frequency) < 1:
        logger.info("✅ Frequency read back correctly")
    else:
        logger.error(f"❌ Frequency mismatch: set {test_frequency/1e9:.6f} GHz, got {current_freq/1e9:.6f} GHz")
    
    if abs(current_power - test_power) < 0.1:
        logger.info("✅ Power read back correctly")
    else:
        logger.error(f"❌ Power mismatch: set {test_power} dBm, got {current_power} dBm")


def test_microwave_controller_update():
    """Test update of shared state through microwave controller."""
    
    logger.info("\n=== Starting Microwave Controller Update Test ===")
    
    # Initialize QudiFacade
    logger.info("Initializing QudiFacade...")
    qudi_facade = QudiFacade(name="SharedStateTest")
    qudi_facade.on_activate()
    
    # Test 2: Update via microwave controller
    logger.info("\nTest 2: Shared state update via microwave controller")
    
    # Test values
    test_frequency = 2.718e9  # ~2.72 GHz
    test_power = -15.0
    
    # Check initial values
    initial_freq = qudi_facade.get_current_frequency()
    initial_power = qudi_facade.get_current_power()
    initial_mw_on = qudi_facade.is_microwave_on()
    logger.info(f"Initial values via accessors: frequency={initial_freq/1e9:.6f} GHz, power={initial_power} dBm, MW on={initial_mw_on}")
    
    # Update via controller
    logger.info(f"Setting microwave controller: frequency={test_frequency/1e9:.6f} GHz, power={test_power} dBm")
    qudi_facade.microwave_controller.set_frequency(test_frequency)
    qudi_facade.microwave_controller.set_power(test_power)
    
    # Read back via shared state directly
    direct_freq = qudi_facade._shared_state['current_mw_frequency']
    direct_power = qudi_facade._shared_state['current_mw_power']
    logger.info(f"Directly from shared state: frequency={direct_freq/1e9:.6f} GHz, power={direct_power} dBm")
    
    # Read back via accessors
    access_freq = qudi_facade.get_current_frequency()
    access_power = qudi_facade.get_current_power()
    logger.info(f"Via accessor methods: frequency={access_freq/1e9:.6f} GHz, power={access_power} dBm")
    
    # Verify values match
    if abs(direct_freq - test_frequency) < 1:
        logger.info("✅ Frequency updated correctly in shared state")
    else:
        logger.error(f"❌ Frequency mismatch in shared state: set {test_frequency/1e9:.6f} GHz, got {direct_freq/1e9:.6f} GHz")
    
    if abs(access_freq - test_frequency) < 1:
        logger.info("✅ Frequency read correctly via accessor")
    else:
        logger.error(f"❌ Frequency mismatch via accessor: set {test_frequency/1e9:.6f} GHz, got {access_freq/1e9:.6f} GHz")
    
    # Turn on the microwave
    logger.info("Turning microwave ON")
    qudi_facade.microwave_controller.on()
    
    # Check if state was updated
    mw_on_state = qudi_facade.is_microwave_on()
    direct_mw_on = qudi_facade._shared_state['current_mw_on']
    
    if mw_on_state and direct_mw_on:
        logger.info("✅ Microwave ON state correctly updated and accessible")
    else:
        logger.error(f"❌ Microwave ON state mismatch: accessor={mw_on_state}, shared_state={direct_mw_on}")
    
    # Turn off microwave
    logger.info("Turning microwave OFF")
    qudi_facade.microwave_controller.off()
    
    # Verify off state
    mw_off_state = qudi_facade.is_microwave_on()
    direct_mw_off = qudi_facade._shared_state['current_mw_on']
    
    if not mw_off_state and not direct_mw_off:
        logger.info("✅ Microwave OFF state correctly updated and accessible")
    else:
        logger.error(f"❌ Microwave OFF state mismatch: accessor={mw_off_state}, shared_state={direct_mw_off}")


def test_scan_frequency_update():
    """Test frequency update during scanning process."""
    
    logger.info("\n=== Starting Scan Frequency Update Test ===")
    
    # Initialize QudiFacade
    logger.info("Initializing QudiFacade...")
    qudi_facade = QudiFacade(name="SharedStateTest")
    qudi_facade.on_activate()
    
    # Initialize microwave device
    logger.info("Initializing Microwave Device...")
    microwave_device = NVSimMicrowaveDevice(name="MW_Test")
    microwave_device.simulator = DummyConnector(qudi_facade)
    microwave_device.on_activate()
    
    # Force microwave controllers to be in a known state
    logger.info("Ensuring clean initial state...")
    microwave_device.off()
    
    # Use force_unlock_state if it exists (for compatibility with older versions)
    if hasattr(microwave_device, 'force_unlock_state'):
        microwave_device.force_unlock_state()
        
    qudi_facade._shared_state['current_mw_on'] = False
    qudi_facade._shared_state['scanning_active'] = False
    logger.info(f"Module state before scan setup: {microwave_device.module_state()}")
    logger.info(f"Microwave on: {qudi_facade.is_microwave_on()}, scanning active: {qudi_facade.is_scanning()}")
    
    # Test 3: Frequency update during scan
    logger.info("\nTest 3: Frequency update during scanning")
    
    # Check initial state
    initial_scan_active = qudi_facade.is_scanning()
    initial_scan_index = qudi_facade.get_current_scan_index()
    logger.info(f"Initial scan state: active={initial_scan_active}, index={initial_scan_index}")
    
    # Configure scan with 5 frequencies
    scan_frequencies = np.linspace(2.8e9, 3.0e9, 5)
    scan_power = -20.0
    
    from qudi.util.enums import SamplingOutputMode
    logger.info(f"Configuring scan with frequencies: {scan_frequencies/1e9} GHz")
    try:
        # Use force=True for test reliability
        microwave_device.configure_scan(scan_power, scan_frequencies, SamplingOutputMode.JUMP_LIST, 10, force=True)
    except RuntimeError as e:
        # Special error handling for this test
        if "Unable to configure scan. Microwave output is active" in str(e):
            logger.error("Caught module state error, applying emergency fix...")
            # Force unlock and try again with force=True
            if hasattr(microwave_device, 'force_unlock_state'):
                microwave_device.force_unlock_state()
                logger.info(f"Module state after force_unlock: {microwave_device.module_state()}")
                microwave_device.configure_scan(scan_power, scan_frequencies, SamplingOutputMode.JUMP_LIST, 10, force=True)
            else:
                # Raise original error if we can't force unlock
                raise e
        else:
            raise e
    
    # Start scan
    logger.info("Starting scan")
    microwave_device.start_scan()
    
    # Verify scan started
    scan_active = qudi_facade.is_scanning()
    scan_index = qudi_facade.get_current_scan_index()
    current_freq = qudi_facade.get_current_frequency()
    
    logger.info(f"After start_scan: active={scan_active}, index={scan_index}, frequency={current_freq/1e9:.6f} GHz")
    
    # First frequency should match first frequency in list
    if abs(current_freq - scan_frequencies[0]) < 1:
        logger.info("✅ Initial scan frequency set correctly")
    else:
        logger.error(f"❌ Initial scan frequency mismatch: expected {scan_frequencies[0]/1e9:.6f} GHz, got {current_freq/1e9:.6f} GHz")
    
    # Move through scan frequencies
    freq_log = []
    index_log = []
    
    for i in range(len(scan_frequencies)):
        # Log current state before scan_next
        current_freq = qudi_facade.get_current_frequency()
        current_index = qudi_facade.get_current_scan_index()
        freq_log.append(current_freq)
        index_log.append(current_index)
        
        logger.info(f"Before scan_next ({i}): index={current_index}, frequency={current_freq/1e9:.6f} GHz")
        
        # Call scan_next
        has_more = microwave_device.scan_next()
        logger.info(f"scan_next() returned: {has_more}")
        
        # Get updated state
        updated_freq = qudi_facade.get_current_frequency()
        updated_index = qudi_facade.get_current_scan_index()
        
        logger.info(f"After scan_next ({i}): index={updated_index}, frequency={updated_freq/1e9:.6f} GHz")
        
        # For all but the last iteration, the frequency should change
        if i < len(scan_frequencies) - 1:
            if abs(updated_freq - current_freq) > 1:
                logger.info(f"✅ Frequency updated after scan_next ({i})")
            else:
                logger.error(f"❌ Frequency did not change after scan_next ({i}): {current_freq/1e9:.6f} GHz -> {updated_freq/1e9:.6f} GHz")
        
        # Small pause to ensure updates propagate
        time.sleep(0.1)
    
    # Plot the frequency changes
    plt.figure(figsize=(10, 6))
    plt.plot(index_log, np.array(freq_log)/1e9, 'b-o', label="Observed Frequencies")
    plt.plot(range(len(scan_frequencies)), scan_frequencies/1e9, 'r--x', label="Expected Frequencies")
    plt.xlabel("Scan Index")
    plt.ylabel("Frequency (GHz)")
    plt.title("Scan Frequency Progression")
    plt.grid(True)
    plt.legend()
    
    # Save the figure
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'scan_frequency_test_{time.strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(output_path)
    logger.info(f"Frequency progression plot saved to {output_path}")
    
    # Stop scan
    microwave_device.off()
    

def test_microwave_sampler_coordination():
    """Test coordination between microwave device and finite sampler."""
    
    logger.info("\n=== Starting Microwave-Sampler Coordination Test ===")
    
    # Initialize QudiFacade
    logger.info("Initializing QudiFacade...")
    qudi_facade = QudiFacade(name="SharedStateTest")
    qudi_facade.on_activate()
    
    # Set magnetic field to 500 Gauss along z-axis
    b_field_tesla = [0, 0, 500e-4]  # 500 G = 0.05 T
    qudi_facade.nv_model.set_magnetic_field(b_field_tesla)
    logger.info(f"Set magnetic field to {b_field_tesla} Tesla")
    
    # Initialize microwave device
    logger.info("Initializing Microwave Device...")
    microwave_device = NVSimMicrowaveDevice(name="MW_Test")
    microwave_device.simulator = DummyConnector(qudi_facade)
    microwave_device.on_activate()
    
    # Initialize finite sampler
    logger.info("Initializing Finite Sampler...")
    finite_sampler = NVSimFiniteSampler(name="Sampler_Test")
    finite_sampler.simulator = DummyConnector(qudi_facade)
    finite_sampler.on_activate()
    
    # Test 4: Coordination between microwave device and sampler
    logger.info("\nTest 4: Coordination between microwave and sampler")
    
    # Configure finite sampler
    sample_rate = 1000  # Hz
    frame_size = 1000   # Samples
    logger.info(f"Configuring sampler: rate={sample_rate} Hz, frame_size={frame_size}")
    finite_sampler.configure(['APD counts'], sample_rate, frame_size)
    
    # Configure scan with resonance frequencies for 500G field
    # For 500G, expect resonances at ~1.47 GHz and ~4.27 GHz
    scan_frequencies = np.linspace(1.4e9, 4.4e9, 11)  # 11 points from 1.4 to 4.4 GHz
    scan_power = -20.0
    
    from qudi.util.enums import SamplingOutputMode
    logger.info(f"Configuring scan with {len(scan_frequencies)} frequencies: {scan_frequencies[0]/1e9:.2f}-{scan_frequencies[-1]/1e9:.2f} GHz")
    try:
        # Use force=True for test reliability
        microwave_device.configure_scan(scan_power, scan_frequencies, SamplingOutputMode.JUMP_LIST, 10, force=True)
    except RuntimeError as e:
        # Special error handling for this test
        if "Unable to configure scan. Microwave output is active" in str(e):
            logger.error("Caught module state error, applying emergency fix...")
            # Force unlock and try again with force=True
            if hasattr(microwave_device, 'force_unlock_state'):
                microwave_device.force_unlock_state()
                logger.info(f"Module state after force_unlock: {microwave_device.module_state()}")
                microwave_device.configure_scan(scan_power, scan_frequencies, SamplingOutputMode.JUMP_LIST, 10, force=True)
            else:
                # Raise original error if we can't force unlock
                raise e
        else:
            raise e
    
    # Start scan
    logger.info("Starting scan")
    microwave_device.start_scan()
    
    # Start sampler
    logger.info("Starting sampler")
    finite_sampler.start_buffered_acquisition()
    
    # Collect signals at each frequency
    freq_log = []
    signal_log = []
    
    for i in range(len(scan_frequencies)):
        # Get current frequency
        current_freq = qudi_facade.get_current_frequency()
        freq_log.append(current_freq)
        logger.info(f"Frequency {i}: {current_freq/1e9:.6f} GHz")
        
        # Get sample data
        try:
            sample_data = finite_sampler.get_buffered_data()
            avg_signal = np.mean(sample_data['APD counts'])
            signal_log.append(avg_signal)
            logger.info(f"  Signal: {avg_signal:.2f}")
        except Exception as e:
            logger.error(f"Error getting sample data: {e}")
            signal_log.append(np.nan)
        
        # Move to next frequency
        if i < len(scan_frequencies) - 1:
            microwave_device.scan_next()
            time.sleep(0.1)  # Small delay to ensure update propagates
    
    # Stop sampler and scan
    finite_sampler.stop_buffered_acquisition()
    microwave_device.off()
    
    # Plot the signal vs frequency
    plt.figure(figsize=(12, 6))
    
    # Plot measured frequencies and signals
    plt.subplot(1, 2, 1)
    plt.plot(np.array(freq_log)/1e9, signal_log, 'b-o')
    
    # Expected resonances
    zfs = 2.87e9  # Zero-field splitting (Hz)
    gyro = 2.8e6  # Gyromagnetic ratio (Hz/G)
    field = 500   # Magnetic field (G)
    
    zeeman_shift = gyro * field
    dip1_center = zfs - zeeman_shift
    dip2_center = zfs + zeeman_shift
    
    plt.axvline(x=dip1_center/1e9, color='r', linestyle='--', label=f'Expected: {dip1_center/1e9:.3f} GHz')
    plt.axvline(x=dip2_center/1e9, color='r', linestyle='--', label=f'Expected: {dip2_center/1e9:.3f} GHz')
    
    plt.title('Measured Signal vs Frequency')
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Signal (counts)')
    plt.grid(True)
    plt.legend()
    
    # Plot requested vs actual frequencies
    plt.subplot(1, 2, 2)
    plt.plot(scan_frequencies/1e9, np.array(freq_log)/1e9, 'g-o')
    plt.plot([1.4, 4.4], [1.4, 4.4], 'k--', label='Expected 1:1')
    plt.title('Requested vs. Actual Frequency')
    plt.xlabel('Requested Frequency (GHz)')
    plt.ylabel('Actual Frequency (GHz)')
    plt.grid(True)
    plt.legend()
    
    # Save the figure
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'mw_sampler_coordination_{time.strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(output_path)
    logger.info(f"Coordination test results saved to {output_path}")


if __name__ == "__main__":
    # Run all tests
    try:
        test_direct_shared_state_update()
        QudiFacade.force_reset_shared_state()
        QudiFacade.reset_instance()  # Reset singleton between tests
    except Exception as e:
        logger.exception(f"Error in direct shared state test: {e}")
    
    try:
        test_microwave_controller_update()
        QudiFacade.force_reset_shared_state()
        QudiFacade.reset_instance()  # Reset singleton between tests
    except Exception as e:
        logger.exception(f"Error in microwave controller test: {e}")
    
    try:
        test_scan_frequency_update()
        QudiFacade.force_reset_shared_state()
        QudiFacade.reset_instance()  # Reset singleton between tests
    except Exception as e:
        logger.exception(f"Error in scan frequency test: {e}")
    
    try:
        test_microwave_sampler_coordination()
    except Exception as e:
        logger.exception(f"Error in microwave-sampler coordination test: {e}")