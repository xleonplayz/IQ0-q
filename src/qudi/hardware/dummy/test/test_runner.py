#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified test runner script for NV simulator tests.

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
import argparse
import importlib
import traceback

# Configure basic logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'test_runner.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('NV_Sim_Test_Runner')

# Add parent directories to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
qudi_dir = os.path.abspath(os.path.join(current_dir, '../../../..'))

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
if qudi_dir not in sys.path:
    sys.path.insert(0, qudi_dir)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Ensure fixed_modules can be imported
fixed_modules_dir = os.path.join(current_dir, 'fixed_modules')
if fixed_modules_dir not in sys.path:
    sys.path.insert(0, fixed_modules_dir)

# Add path for simulator import
simulator_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'nv_simulator'))
if simulator_dir not in sys.path:
    sys.path.insert(0, simulator_dir)

def prepare_environment():
    """Prepare the environment for testing."""
    logger.info("Preparing test environment...")
    
    # Create results directory
    results_dir = os.path.join(current_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # First check if we can import core modules directly
    core_modules_available = True
    try:
        from qudi.core.module import Base
        logger.info("Real Qudi core modules are available")
    except ImportError:
        core_modules_available = False
        logger.warning("Real Qudi core modules not available, will use mock implementation")
        
        # Add our fixed modules to the path for tests to use
        if fixed_modules_dir not in sys.path:
            sys.path.insert(0, fixed_modules_dir)
    
    return {
        'core_modules_available': core_modules_available,
        'results_dir': results_dir
    }

def patch_imports():
    """Patch imports to use fixed modules if needed."""
    try:
        # Try importing from real Qudi
        from qudi.core.module import Base
        from qudi.interface.microwave_interface import MicrowaveInterface
        from qudi.interface.finite_sampling_input_interface import FiniteSamplingInputInterface
        logger.info("Using real Qudi imports")
        
        # Patch the QudiFacade to support test_mode even when real Qudi is available
        try:
            from qudi.hardware.nv_simulator.qudi_facade import QudiFacade
            original_new = QudiFacade.__new__
            
            # If __new__ is not already patched
            if not hasattr(QudiFacade, '_original_new'):
                # Store the original for reference
                QudiFacade._original_new = original_new
                
                # Define the patched method
                def patched_new(cls, *args, **kwargs):
                    # Handle test_mode parameter
                    test_mode = kwargs.pop('test_mode', False)
                    
                    if test_mode:
                        # In test mode, create a new instance
                        instance = object.__new__(cls)
                        instance._initialized = False
                        return instance
                    
                    # Otherwise use the original implementation
                    return original_new(cls, *args, **kwargs)
                
                # Apply the patch
                QudiFacade.__new__ = patched_new
                logger.info("Patched QudiFacade.__new__ to support test_mode")
        except ImportError:
            logger.warning("Could not patch QudiFacade, it might not be imported yet")
    except ImportError:
        logger.info("Patching Qudi imports with fixed modules")
        
        # Create a sys.modules entry for qudi.core.module
        import fixed_modules.qudi_core as qudi_core
        sys.modules['qudi.core.module'] = qudi_core
        
        # Create a sys.modules entry for qudi.interface.microwave_interface
        import fixed_modules.microwave_interface as microwave_interface
        sys.modules['qudi.interface.microwave_interface'] = microwave_interface
        
        # Create a sys.modules entry for qudi.interface.finite_sampling_input_interface
        import fixed_modules.finite_sampling_interface as finite_sampling_interface
        sys.modules['qudi.interface.finite_sampling_input_interface'] = finite_sampling_interface
        
        # Create a minimal enum module if needed
        if 'qudi.util.enums' not in sys.modules:
            from fixed_modules.microwave_interface import SamplingOutputMode
            enums_module = type('EnumsModule', (), {'SamplingOutputMode': SamplingOutputMode})
            sys.modules['qudi.util.enums'] = enums_module

def print_separator():
    """Print a separator line."""
    print("\n" + "="*80 + "\n")

def run_test(test_name):
    """Run a specific test."""
    logger.info(f"Running test: {test_name}")
    print_separator()
    print(f"RUNNING TEST: {test_name}")
    print_separator()
    
    try:
        if test_name == 'mw_sampler_sync':
            import test_mw_sampler_sync
            test_mw_sampler_sync.test_direct_frequency_setting()
            test_mw_sampler_sync.test_scan_mode_synchronization()
            logger.info("MW-Sampler sync test completed successfully")
            return True
        elif test_name == 'odmr_flow':
            import test_odmr_flow
            test_odmr_flow.run_odmr_flow_test()
            logger.info("ODMR flow test completed successfully")
            return True
        elif test_name == 'run_odmr_test':
            import run_odmr_test
            run_odmr_test.run_test_and_visualize()
            logger.info("ODMR test and visualization completed successfully")
            return True
        else:
            logger.error(f"Unknown test: {test_name}")
            return False
    except Exception as e:
        logger.exception(f"Error running test {test_name}: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='NV Simulator Test Runner')
    parser.add_argument('tests', nargs='*', help='Tests to run (mw_sampler_sync, odmr_flow, run_odmr_test, or all)')
    args = parser.parse_args()
    
    # Default to all tests if none specified
    if not args.tests:
        args.tests = ['all']
    
    # Prepare environment
    env = prepare_environment()
    
    # Patch imports if needed
    patch_imports()
    
    # Run specified tests
    if 'all' in args.tests:
        tests = ['mw_sampler_sync', 'odmr_flow', 'run_odmr_test']
    else:
        tests = args.tests
    
    results = {}
    for test in tests:
        start_time = time.time()
        success = run_test(test)
        end_time = time.time()
        results[test] = {
            'success': success,
            'duration': end_time - start_time
        }
    
    # Print summary
    print_separator()
    print("TEST RESULTS SUMMARY")
    print_separator()
    
    for test, result in results.items():
        status = "PASSED" if result['success'] else "FAILED"
        print(f"{test}: {status} (took {result['duration']:.2f}s)")
    
    # Check if all tests passed
    all_passed = all(result['success'] for result in results.values())
    
    print_separator()
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print_separator()
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())