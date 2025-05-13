#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test focusing specifically on the frequency chain from ODMR logic to microwave to sampler.
This test isolates and verifies each step in the chain to identify where frequency updates might fail.

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
from threading import Thread
from PySide2 import QtCore

# Import environment setup to ensure consistent test environment
from env_setup import env_info

from qudi.hardware.nv_simulator.qudi_facade import QudiFacade
from qudi.hardware.nv_simulator.microwave_device import NVSimMicrowaveDevice
from qudi.hardware.nv_simulator.finite_sampler import NVSimFiniteSampler
from qudi.util.mutex import Mutex
from qudi.util.enums import SamplingOutputMode
from qudi.logic.odmr_logic import OdmrLogic

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'frequency_chain_test.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('Frequency_Chain_Test')


class DummyConnector:
    """Dummy connector class for providing module instances."""
    def __init__(self, module_instance):
        self._module = module_instance
    def __call__(self):
        return self._module


class SignalCollector(QtCore.QObject):
    """Collects signals from OdmrLogic for testing."""
    
    def __init__(self):
        super().__init__()
        self.scan_data_updated_count = 0
        self.scan_state_updated_values = []
        self.elapsed_updated_values = []
        self.last_data = None
        self.signals_lock = Mutex()
        
    @QtCore.Slot()
    def on_scan_data_updated(self):
        with self.signals_lock:
            self.scan_data_updated_count += 1
            logger.debug(f"Signal received: scan_data_updated (count: {self.scan_data_updated_count})")
    
    @QtCore.Slot(bool)
    def on_scan_state_updated(self, state):
        with self.signals_lock:
            self.scan_state_updated_values.append(state)
            logger.debug(f"Signal received: scan_state_updated = {state}")
    
    @QtCore.Slot(float, int)
    def on_elapsed_updated(self, time_elapsed, sweeps):
        with self.signals_lock:
            self.elapsed_updated_values.append((time_elapsed, sweeps))
            logger.debug(f"Signal received: elapsed_updated = {time_elapsed:.2f}s, {sweeps} sweeps")
    
    def store_data(self, odmr_logic):
        """Store the current ODMR data from logic for analysis."""
        with self.signals_lock:
            self.last_data = {
                'frequency_data': odmr_logic.frequency_data,
                'signal_data': odmr_logic.signal_data,
                'raw_data': odmr_logic.raw_data
            }
            
            # Log first 5 frequencies and corresponding signal values
            for ch, signals in odmr_logic.signal_data.items():
                for range_idx, signal in enumerate(signals):
                    freqs = odmr_logic.frequency_data[range_idx]
                    sample_indices = np.linspace(0, len(freqs)-1, min(5, len(freqs))).astype(int)
                    
                    logger.info(f"Channel: {ch}, Range: {range_idx}")
                    for i in sample_indices:
                        logger.info(f"  Freq: {freqs[i]/1e9:.6f} GHz, Signal: {signal[i]:.2f}")


def test_direct_scan_next():
    """Test direct frequency updates with scan_next."""
    
    logger.info("=== Starting Direct Scan Next Test ===")
    
    # Initialize QudiFacade
    logger.info("Initializing QudiFacade...")
    qudi_facade = QudiFacade(name="FrequencyChainTest")
    qudi_facade.on_activate()
    
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
    
    # Test direct scan_next functionality
    logger.info("\nTest 1: Direct scan_next test")
    
    # Configure scan with 5 frequencies
    scan_frequencies = np.linspace(2.8e9, 3.0e9, 5)
    scan_power = -20.0
    
    logger.info(f"Configuring scan with frequencies: {scan_frequencies/1e9} GHz")
    microwave_device.configure_scan(scan_power, scan_frequencies, SamplingOutputMode.JUMP_LIST, 10)
    
    # Start scan
    logger.info("Starting scan")
    microwave_device.start_scan()
    
    # Configure sampler
    sample_rate = 1000
    frame_size = 1000
    finite_sampler.configure(['APD counts'], sample_rate, frame_size)
    finite_sampler.start_buffered_acquisition()
    
    # Track all frequency states
    freq_log = []
    freq_shared_log = []
    sampler_freq_log = []
    signal_log = []
    
    for i in range(len(scan_frequencies) * 2):  # Run through twice
        # Directly read frequency from shared state
        current_freq_shared = qudi_facade._shared_state['current_mw_frequency']
        freq_shared_log.append(current_freq_shared)
        
        # Get frequency using accessor method
        current_freq = qudi_facade.get_current_frequency()
        freq_log.append(current_freq)
        
        # Get sample data and record frequency seen by sampler
        sample_data = finite_sampler.get_buffered_data()
        # The frequency used by the sampler is logged in the sampler's debug output
        # We need to capture this from qudi_facade directly as sampler can't report it
        sampler_freq = qudi_facade._shared_state['current_mw_frequency']
        sampler_freq_log.append(sampler_freq)
        
        # Get average signal
        avg_signal = np.mean(sample_data['APD counts'])
        signal_log.append(avg_signal)
        
        logger.info(f"Step {i}:")
        logger.info(f"  Shared state frequency: {current_freq_shared/1e9:.6f} GHz")
        logger.info(f"  get_current_frequency: {current_freq/1e9:.6f} GHz")
        logger.info(f"  Sampler seeing frequency: {sampler_freq/1e9:.6f} GHz")
        logger.info(f"  Signal: {avg_signal:.2f}")
        
        # Call scan_next to move to next frequency
        if i < len(scan_frequencies) * 2 - 1:  # Don't advance on last iteration
            has_more = microwave_device.scan_next()
            logger.info(f"  scan_next returned: {has_more}")
            
            # Wait a bit for state to propagate
            time.sleep(0.1)
    
    # Stop acquisition
    finite_sampler.stop_buffered_acquisition()
    microwave_device.off()
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Frequency over steps
    plt.subplot(2, 1, 1)
    plt.plot(np.array(freq_shared_log)/1e9, 'b-o', label="Shared State")
    plt.plot(np.array(freq_log)/1e9, 'r-x', label="get_current_frequency")
    plt.plot(np.array(sampler_freq_log)/1e9, 'g-^', label="Sampler View")
    plt.xlabel("Step")
    plt.ylabel("Frequency (GHz)")
    plt.title("Frequency in Different Parts of the Chain")
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Signal vs Frequency
    plt.subplot(2, 1, 2)
    plt.plot(np.array(freq_log)/1e9, signal_log, 'b-o')
    
    # Expected resonances (if field is set)
    try:
        magnetic_field = qudi_facade.nv_model.b_field
        field_strength_gauss = np.linalg.norm(magnetic_field) * 10000.0  # Convert Tesla to Gauss
        
        zfs = 2.87e9  # Zero-field splitting (Hz)
        gyro = 2.8e6  # Gyromagnetic ratio (Hz/G)
        
        zeeman_shift = gyro * field_strength_gauss
        dip1_center = zfs - zeeman_shift
        dip2_center = zfs + zeeman_shift
        
        plt.axvline(x=dip1_center/1e9, color='r', linestyle='--', label=f'Expected Dip: {dip1_center/1e9:.3f} GHz')
        plt.axvline(x=dip2_center/1e9, color='r', linestyle='--', label=f'Expected Dip: {dip2_center/1e9:.3f} GHz')
        plt.legend()
    except:
        pass
    
    plt.xlabel("Frequency (GHz)")
    plt.ylabel("Signal")
    plt.title("Signal vs Frequency")
    plt.grid(True)
    
    # Save plot
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'direct_scan_next_test_{time.strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(output_path)
    logger.info(f"Direct scan_next test plot saved to {output_path}")


def test_odmr_logic_scan():
    """Test ODMR logic's scanning behavior."""
    
    logger.info("\n=== Starting ODMR Logic Scan Test ===")
    
    # Initialize QudiFacade
    logger.info("Initializing QudiFacade...")
    qudi_facade = QudiFacade(name="FrequencyChainTest")
    qudi_facade.on_activate()
    
    # Set magnetic field
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
    
    # Initialize ODMR logic
    logger.info("Initializing ODMR Logic...")
    odmr_logic = OdmrLogic()
    odmr_logic._microwave = DummyConnector(microwave_device)
    odmr_logic._data_scanner = DummyConnector(finite_sampler)
    
    # Create signal collector
    signal_collector = SignalCollector()
    
    # Connect signals
    odmr_logic.sigScanDataUpdated.connect(signal_collector.on_scan_data_updated)
    odmr_logic.sigScanStateUpdated.connect(signal_collector.on_scan_state_updated)
    odmr_logic.sigElapsedUpdated.connect(signal_collector.on_elapsed_updated)
    
    # Activate ODMR logic
    odmr_logic.on_activate()
    
    # Set up ODMR scan parameters for NV center resonances
    logger.info("Setting up ODMR scan parameters...")
    
    # Configure scan to cover both resonances (around 1.47 GHz and 4.27 GHz)
    odmr_logic.set_frequency_range(1.4e9, 4.4e9, 21, 0)  # 21 points from 1.4 to 4.4 GHz
    
    # Set scan power
    odmr_logic.set_scan_power(-20)
    
    # Set short runtime for test
    odmr_logic.set_runtime(5)
    
    # Log current configuration
    logger.info(f"Scan frequencies: {odmr_logic.frequency_ranges}")
    logger.info(f"Scan power: {odmr_logic.scan_power} dBm")
    logger.info(f"Scan runtime: {odmr_logic.runtime} s")
    
    # Frequency tracking data
    freq_history = []
    scan_index_history = []
    time_points = []
    start_time = time.time()
    
    # Start frequency tracking thread
    stop_tracking = False
    
    def track_frequencies():
        while not stop_tracking:
            freq = qudi_facade.get_current_frequency()
            scan_idx = qudi_facade.get_current_scan_index()
            elapsed = time.time() - start_time
            
            freq_history.append(freq)
            scan_index_history.append(scan_idx)
            time_points.append(elapsed)
            
            # Log every few samples
            if len(freq_history) % 5 == 0:
                logger.debug(f"Frequency tracking - Time: {elapsed:.2f}s, Index: {scan_idx}, Freq: {freq/1e9:.6f} GHz")
                
            time.sleep(0.1)  # 100ms sampling
    
    # Start tracking thread
    tracking_thread = Thread(target=track_frequencies)
    tracking_thread.daemon = True
    tracking_thread.start()
    
    try:
        # Start ODMR scan
        logger.info("Starting ODMR scan...")
        odmr_logic.start_odmr_scan()
        
        # Wait for scan to complete
        timeout = 20  # seconds
        start_time = time.time()
        logger.info(f"Waiting up to {timeout}s for scan to complete...")
        
        while odmr_logic.module_state() == 'locked' and (time.time() - start_time) < timeout:
            # Wait a bit
            time.sleep(0.2)
            
            # Periodically check status and store data
            if int((time.time() - start_time) * 5) % 5 == 0:  # Check every ~1s
                signal_collector.store_data(odmr_logic)
                
                # Log status
                logger.info("ODMR scan status:")
                logger.info(f"  Elapsed time: {time.time() - start_time:.2f}s")
                logger.info(f"  Scan state: {odmr_logic.module_state()}")
                logger.info(f"  Data updates: {signal_collector.scan_data_updated_count}")
                logger.info(f"  Current MW frequency: {qudi_facade.get_current_frequency()/1e9:.6f} GHz")
                logger.info(f"  Current scan index: {qudi_facade.get_current_scan_index()}")
                
        # Check if scan completed or timed out
        if odmr_logic.module_state() == 'locked':
            logger.warning(f"ODMR scan did not complete within {timeout}s, stopping manually")
            odmr_logic.stop_odmr_scan()
        else:
            logger.info("ODMR scan completed successfully")
        
        # Stop frequency tracking
        stop_tracking = True
        tracking_thread.join(1.0)
        
        # Get final data
        signal_collector.store_data(odmr_logic)
        
        # Plot results
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Frequency tracking over time
        plt.subplot(3, 1, 1)
        plt.plot(time_points, np.array(freq_history)/1e9, 'b-')
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (GHz)")
        plt.title("Frequency vs Time during ODMR Scan")
        plt.grid(True)
        
        # Plot 2: Scan index over time
        plt.subplot(3, 1, 2)
        plt.plot(time_points, scan_index_history, 'g-')
        plt.xlabel("Time (s)")
        plt.ylabel("Scan Index")
        plt.title("Scan Index vs Time")
        plt.grid(True)
        
        # Plot 3: ODMR spectrum
        plt.subplot(3, 1, 3)
        
        if signal_collector.last_data:
            for ch, signals in signal_collector.last_data['signal_data'].items():
                for range_idx, signal in enumerate(signals):
                    freqs = signal_collector.last_data['frequency_data'][range_idx]
                    plt.plot(freqs/1e9, signal, label=f"{ch} - Range {range_idx}")
            
            # Add expected resonances
            zfs = 2.87e9  # Zero-field splitting (Hz)
            gyro = 2.8e6  # Gyromagnetic ratio (Hz/G)
            field = 500   # Magnetic field (G)
            
            zeeman_shift = gyro * field
            dip1_center = zfs - zeeman_shift
            dip2_center = zfs + zeeman_shift
            
            plt.axvline(x=dip1_center/1e9, color='r', linestyle='--', label=f'Expected: {dip1_center/1e9:.3f} GHz')
            plt.axvline(x=dip2_center/1e9, color='r', linestyle='--', label=f'Expected: {dip2_center/1e9:.3f} GHz')
            
        plt.xlabel("Frequency (GHz)")
        plt.ylabel("Signal")
        plt.title("ODMR Spectrum")
        plt.grid(True)
        plt.legend()
        
        # Save plot
        output_dir = os.path.join(os.path.dirname(__file__), 'results')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'odmr_logic_scan_test_{time.strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(output_path)
        logger.info(f"ODMR logic scan test plot saved to {output_path}")
        
    except Exception as e:
        logger.exception(f"Error during ODMR logic scan test: {e}")
    finally:
        # Clean up
        stop_tracking = True
        if tracking_thread.is_alive():
            tracking_thread.join(1.0)
            
        try:
            # Disconnect signals
            odmr_logic.sigScanDataUpdated.disconnect(signal_collector.on_scan_data_updated)
            odmr_logic.sigScanStateUpdated.disconnect(signal_collector.on_scan_state_updated)
            odmr_logic.sigElapsedUpdated.disconnect(signal_collector.on_elapsed_updated)
        except:
            logger.exception("Error disconnecting signals")
        
        try:
            # Deactivate modules
            odmr_logic.on_deactivate()
            microwave_device.on_deactivate()
            finite_sampler.on_deactivate()
            qudi_facade.on_deactivate()
        except:
            logger.exception("Error deactivating modules")


def test_scan_next_implementation():
    """Test the scan_next implementation in the microwave device."""
    
    logger.info("\n=== Starting Scan Next Implementation Test ===")
    
    # Initialize QudiFacade
    logger.info("Initializing QudiFacade...")
    qudi_facade = QudiFacade(name="FrequencyChainTest")
    qudi_facade.on_activate()
    
    # Initialize microwave device
    logger.info("Initializing Microwave Device...")
    microwave_device = NVSimMicrowaveDevice(name="MW_Test")
    microwave_device.simulator = DummyConnector(qudi_facade)
    microwave_device.on_activate()
    
    # Test scan_next implementation details
    logger.info("\nTest 3: Scan Next Implementation Details")
    
    # Configure scan with 5 frequencies
    scan_frequencies = np.linspace(2.8e9, 3.0e9, 5)
    scan_power = -20.0
    
    logger.info(f"Configuring scan with frequencies: {scan_frequencies/1e9} GHz")
    microwave_device.configure_scan(scan_power, scan_frequencies, SamplingOutputMode.JUMP_LIST, 10)
    
    # Start scan
    logger.info("Starting scan")
    microwave_device.start_scan()
    
    # Check implementation of scan_next
    logger.info("Testing scan_next implementation:")
    
    # Check how scan_next updates the QudiFacade shared state
    for i in range(len(scan_frequencies) + 1):  # +1 to include end of scan case
        # Log current state
        scan_index = microwave_device._current_scan_index
        shared_index = qudi_facade._shared_state['current_scan_index']
        current_freq = qudi_facade.get_current_frequency()
        direct_freq = qudi_facade._shared_state['current_mw_frequency']
        scanning_active = qudi_facade._shared_state['scanning_active']
        
        logger.info(f"Step {i}:")
        logger.info(f"  Microwave scan index: {scan_index}")
        logger.info(f"  Shared state scan index: {shared_index}")
        logger.info(f"  get_current_frequency: {current_freq/1e9:.6f} GHz")
        logger.info(f"  Shared state frequency: {direct_freq/1e9:.6f} GHz")
        logger.info(f"  Scanning active: {scanning_active}")
        
        # Step through and verify scan_next behavior
        if i < len(scan_frequencies):
            expected_freq = scan_frequencies[i if i == 0 else min(i, len(scan_frequencies) - 1)]
            logger.info(f"  Expected frequency: {expected_freq/1e9:.6f} GHz")
            
            # Check if current frequency matches expected
            if abs(current_freq - expected_freq) < 1:
                logger.info("  ✅ Current frequency matches expected")
            else:
                logger.error(f"  ❌ Frequency mismatch: expected {expected_freq/1e9:.6f}, got {current_freq/1e9:.6f}")
            
            # Check if shared state is consistent
            if abs(direct_freq - current_freq) < 1:
                logger.info("  ✅ Shared state consistent with accessor")
            else:
                logger.error(f"  ❌ Shared state inconsistent: accessor {current_freq/1e9:.6f}, direct {direct_freq/1e9:.6f}")
            
            # Call scan_next
            if i < len(scan_frequencies) - 1:
                logger.info("  Calling scan_next()...")
                has_more = microwave_device.scan_next()
                logger.info(f"  scan_next returned: {has_more}")
                time.sleep(0.1)  # Wait for state propagation
            else:
                # Last step, expect no more
                logger.info("  Calling scan_next() at end of sequence...")
                has_more = microwave_device.scan_next()
                logger.info(f"  scan_next returned: {has_more} (should be False at end of sequence)")
                time.sleep(0.1)
    
    # Reset scan and try again - this is to verify reset behavior
    logger.info("\nTesting reset_scan implementation:")
    microwave_device.reset_scan()
    
    # Get reset state
    reset_index = microwave_device._current_scan_index
    reset_freq = qudi_facade.get_current_frequency()
    expected_first_freq = scan_frequencies[0]
    
    logger.info(f"After reset_scan:")
    logger.info(f"  Scan index: {reset_index}")
    logger.info(f"  Frequency: {reset_freq/1e9:.6f} GHz")
    logger.info(f"  Expected first frequency: {expected_first_freq/1e9:.6f} GHz")
    
    if reset_index == 0:
        logger.info("  ✅ Scan index correctly reset to 0")
    else:
        logger.error(f"  ❌ Scan index not reset: {reset_index}")
        
    if abs(reset_freq - expected_first_freq) < 1:
        logger.info("  ✅ Frequency correctly reset to first frequency")
    else:
        logger.error(f"  ❌ Frequency not reset properly: got {reset_freq/1e9:.6f}, expected {expected_first_freq/1e9:.6f}")
    
    # Stop scan
    microwave_device.off()
    

if __name__ == "__main__":
    # Run all tests
    try:
        test_direct_scan_next()
        QudiFacade.reset_instance()  # Reset singleton between tests
    except Exception as e:
        logger.exception(f"Error in direct scan next test: {e}")
        
    try:
        test_scan_next_implementation()
        QudiFacade.reset_instance()  # Reset singleton between tests
    except Exception as e:
        logger.exception(f"Error in scan next implementation test: {e}")
    
    try:
        test_odmr_logic_scan()
    except Exception as e:
        logger.exception(f"Error in ODMR logic scan test: {e}")