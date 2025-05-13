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

# Import environment setup to ensure consistent test environment
from env_setup import env_info

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

def prepare_environment():
    """Prepare the environment for testing."""
    logger.info("Preparing test environment...")
    
    # Environment already set up in env_setup.py
    results_dir = env_info['results_dir']
    
    # Make sure the QudiFacade singleton is reset
    try:
        from qudi.hardware.nv_simulator.qudi_facade import QudiFacade
        QudiFacade.reset_instance()
        logger.info("Reset QudiFacade singleton before tests")
    except ImportError:
        logger.warning("Could not import QudiFacade to reset singleton")
    
    # Check if we can import core modules directly
    core_modules_available = True
    try:
        from qudi.core.module import Base
        logger.info("Real Qudi core modules are available")
    except ImportError:
        core_modules_available = False
        logger.warning("Real Qudi core modules not available, will use mock implementation")
    
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
                
                # Define the patched method to reset singleton
                def reset_singleton():
                    # Reset the singleton instance
                    if hasattr(QudiFacade, '_instance'):
                        QudiFacade._instance = None
                        logger.info("Reset QudiFacade singleton instance")
                
                # Add reset method to QudiFacade for tests to use
                QudiFacade.reset_for_test = staticmethod(reset_singleton)
                
                # No need to patch __new__ anymore
                logger.info("Added reset_for_test method to QudiFacade")
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
    
    # Reset QudiFacade singleton before each test
    try:
        from qudi.hardware.nv_simulator.qudi_facade import QudiFacade
        QudiFacade.reset_instance()
        logger.info(f"Reset QudiFacade singleton before {test_name} test")
    except ImportError:
        logger.warning("Could not import QudiFacade to reset singleton")
    
    try:
        if test_name == 'mw_sampler_sync':
            # Run only first test to avoid singleton issues
            import test_mw_sampler_sync
            test_mw_sampler_sync.test_direct_frequency_setting()
            
            # Reset singleton before second test
            from qudi.hardware.nv_simulator.qudi_facade import QudiFacade
            QudiFacade.reset_instance()
            logger.info("Reset QudiFacade singleton between tests")
            
            # Run second test
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
        elif test_name == 'shared_state':
            import test_shared_state
            # Individual tests have their own singleton reset
            test_shared_state.test_direct_shared_state_update()
            test_shared_state.test_microwave_controller_update()
            test_shared_state.test_scan_frequency_update()
            test_shared_state.test_microwave_sampler_coordination()
            logger.info("Shared state tests completed successfully")
            return True
        elif test_name == 'frequency_chain':
            import test_frequency_chain
            # Individual tests have their own singleton reset
            test_frequency_chain.test_direct_scan_next()
            test_frequency_chain.test_scan_next_implementation()
            test_frequency_chain.test_odmr_logic_scan()
            logger.info("Frequency chain tests completed successfully")
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
    parser.add_argument('tests', nargs='*', help='Tests to run (all, mw_sampler_sync, odmr_flow, run_odmr_test, shared_state, frequency_chain)')
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
        tests = ['mw_sampler_sync', 'odmr_flow', 'run_odmr_test', 'shared_state', 'frequency_chain']
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