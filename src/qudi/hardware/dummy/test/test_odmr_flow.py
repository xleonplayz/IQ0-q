# -*- coding: utf-8 -*-

"""
Test module for checking the information flow between ODMR GUI and hardware components.

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
from PySide2 import QtCore

# Add parent directory to path to import qudi modules
current_dir = os.path.dirname(os.path.abspath(__file__))
qudi_dir = os.path.abspath(os.path.join(current_dir, '../../../..'))
if qudi_dir not in sys.path:
    sys.path.insert(0, qudi_dir)

from qudi.util.mutex import Mutex
from qudi.util.enums import SamplingOutputMode
from qudi.hardware.nv_simulator.qudi_facade import QudiFacade
from qudi.hardware.nv_simulator.microwave_device import NVSimMicrowaveDevice
from qudi.hardware.nv_simulator.finite_sampler import NVSimFiniteSampler
from qudi.logic.odmr_logic import OdmrLogic

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(current_dir, 'odmr_flow_test.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('ODMR_Flow_Test')


class DummyConnector:
    """Dummy connector class for providing module instances to logic."""
    
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
            logger.debug(f"Signal: scan_data_updated (count: {self.scan_data_updated_count})")
    
    @QtCore.Slot(bool)
    def on_scan_state_updated(self, state):
        with self.signals_lock:
            self.scan_state_updated_values.append(state)
            logger.debug(f"Signal: scan_state_updated = {state}")
    
    @QtCore.Slot(float, int)
    def on_elapsed_updated(self, time_elapsed, sweeps):
        with self.signals_lock:
            self.elapsed_updated_values.append((time_elapsed, sweeps))
            logger.debug(f"Signal: elapsed_updated = {time_elapsed:.2f}s, {sweeps} sweeps")
    
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


def run_odmr_flow_test():
    """Run the ODMR information flow test."""
    
    logger.info("=== Starting ODMR Flow Test ===")
    
    # Initialize QudiFacade
    logger.info("Initializing QudiFacade...")
    qudi_facade = QudiFacade(name="QudiFacade_Test")
    qudi_facade.on_activate()
    
    # Set magnetic field to 500 Gauss along z-axis
    b_field_tesla = [0, 0, 500e-4]  # 500 G = 0.05 T
    qudi_facade.nv_model.set_magnetic_field(b_field_tesla)
    logger.info(f"Set magnetic field to {b_field_tesla} Tesla")
    
    # Initialize microwave device
    logger.info("Initializing NVSimMicrowaveDevice...")
    microwave_device = NVSimMicrowaveDevice(name="MW_Test")
    microwave_device.simulator = DummyConnector(qudi_facade)
    microwave_device.on_activate()
    
    # Initialize finite sampler
    logger.info("Initializing NVSimFiniteSampler...")
    finite_sampler = NVSimFiniteSampler(name="FiniteSampler_Test")
    finite_sampler.simulator = DummyConnector(qudi_facade)
    finite_sampler.on_activate()
    
    # Initialize ODMR logic
    logger.info("Initializing OdmrLogic...")
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
    # At 500 Gauss, resonances should be around 1.47 GHz and 4.27 GHz
    logger.info("Setting up ODMR scan parameters...")
    
    # Configure scan to cover both resonances with 301 points
    odmr_logic.set_frequency_range(1.4e9, 4.4e9, 301, 0)
    
    # Set scan power
    odmr_logic.set_scan_power(-20)
    
    # Set short runtime for test
    odmr_logic.set_runtime(5)
    
    # Log current configuration
    logger.info(f"Scan frequencies: {odmr_logic.frequency_ranges}")
    logger.info(f"Scan power: {odmr_logic.scan_power} dBm")
    logger.info(f"Scan runtime: {odmr_logic.runtime} s")
    
    try:
        # First test - get current state
        logger.info("\n=== Test 1: Check current state ===")
        
        # Verify initial state
        logger.info(f"MW device scanning: {microwave_device.is_scanning}")
        logger.info(f"Finite sampler running: {finite_sampler._is_running}")
        logger.info(f"ODMR logic state: {odmr_logic.module_state()}")
        
        # Test direct frequency setting
        logger.info("\n=== Test 2: Direct frequency control ===")
        
        # Set CW frequency and power
        microwave_device.set_cw(2.87e9, -20)
        microwave_device.cw_on()
        
        # Check shared state
        logger.info(f"Shared state MW frequency: {qudi_facade._shared_state['current_mw_frequency']/1e9:.6f} GHz")
        logger.info(f"Shared state MW on: {qudi_facade._shared_state['current_mw_on']}")
        
        # Get sample data directly from finite sampler
        logger.info("Getting sample data directly from finite sampler...")
        finite_sampler.start_buffered_acquisition()
        sample_data = finite_sampler.get_buffered_data()
        finite_sampler.stop_buffered_acquisition()
        
        # Log sample data summary for each channel
        for ch, data in sample_data.items():
            logger.info(f"Channel {ch}: mean={np.mean(data):.2f}, std={np.std(data):.2f}, min={np.min(data):.2f}, max={np.max(data):.2f}")
        
        # Turn off microwave
        microwave_device.off()
        
        # Test ODMR scan
        logger.info("\n=== Test 3: ODMR Scan ===")
        
        # Start ODMR scan
        logger.info("Starting ODMR scan...")
        odmr_logic.start_odmr_scan()
        
        # Wait for scan to complete
        timeout = 30  # seconds
        start_time = time.time()
        logger.info(f"Waiting up to {timeout}s for scan to complete...")
        
        while odmr_logic.module_state() == 'locked' and (time.time() - start_time) < timeout:
            # Wait a bit
            time.sleep(0.1)
            
            # Periodically check microwave state and shared state
            if int((time.time() - start_time) * 10) % 10 == 0:  # Check every ~1s
                logger.info("Periodic scan status check:")
                logger.info(f"  ODMR state: {odmr_logic.module_state()}")
                logger.info(f"  MW scanning: {microwave_device.is_scanning}")
                logger.info(f"  Shared state scanning: {qudi_facade._shared_state['scanning_active']}")
                logger.info(f"  Shared state scan index: {qudi_facade._shared_state['current_scan_index']}")
                if qudi_facade._shared_state['scan_frequencies'] is not None:
                    scan_freqs = qudi_facade._shared_state['scan_frequencies']
                    logger.info(f"  Scan frequencies: {len(scan_freqs)} points from {scan_freqs[0]/1e9:.6f} to {scan_freqs[-1]/1e9:.6f} GHz")
                
                # Store current data for analysis
                signal_collector.store_data(odmr_logic)
        
        # Check if scan completed or timed out
        if odmr_logic.module_state() == 'locked':
            logger.warning(f"ODMR scan did not complete within {timeout}s, stopping manually")
            odmr_logic.stop_odmr_scan()
        else:
            logger.info("ODMR scan completed successfully")
        
        # Analyze final data
        logger.info("\n=== Test 4: Analyzing Results ===")
        
        # Store final data
        signal_collector.store_data(odmr_logic)
        
        # Log signal stats
        logger.info(f"Scan data updated signal count: {signal_collector.scan_data_updated_count}")
        logger.info(f"Scan state updates: {signal_collector.scan_state_updated_values}")
        logger.info(f"Elapsed time updates: {len(signal_collector.elapsed_updated_values)}")
        
        # Verify data contains ODMR resonances
        if signal_collector.last_data:
            signal_data = signal_collector.last_data['signal_data']
            freq_data = signal_collector.last_data['frequency_data']
            
            for ch, signals in signal_data.items():
                for range_idx, signal in enumerate(signals):
                    freqs = freq_data[range_idx]
                    
                    # Calculate min/max/mean
                    min_val = np.min(signal)
                    max_val = np.max(signal)
                    mean_val = np.mean(signal)
                    std_val = np.std(signal)
                    
                    logger.info(f"Channel: {ch}, Range: {range_idx}")
                    logger.info(f"  Min: {min_val:.2f}, Max: {max_val:.2f}, Mean: {mean_val:.2f}, Std: {std_val:.2f}")
                    
                    # Check for significant contrast (>5% difference between min and max)
                    if max_val > 0 and (max_val - min_val) / max_val > 0.05:
                        logger.info("  Detected significant contrast in the data")
                        
                        # Find minima in signal (resonances)
                        # Use a simple approach: find local minima below the average value
                        minima_indices = []
                        for i in range(1, len(signal) - 1):
                            if signal[i] < signal[i-1] and signal[i] < signal[i+1] and signal[i] < mean_val:
                                minima_indices.append(i)
                        
                        logger.info(f"  Found {len(minima_indices)} potential resonances:")
                        for idx in minima_indices:
                            logger.info(f"    Resonance at {freqs[idx]/1e9:.6f} GHz, Value: {signal[idx]:.2f}")
                    else:
                        logger.warning("  No significant contrast detected in the data")
        else:
            logger.warning("No data was collected for analysis")
        
    except Exception as e:
        logger.exception(f"Error during ODMR flow test: {e}")
    finally:
        # Cleanup
        logger.info("\n=== Cleaning up ===")
        
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
        
        logger.info("ODMR flow test completed")


if __name__ == "__main__":
    run_odmr_flow_test()