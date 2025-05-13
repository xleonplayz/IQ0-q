#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Environment setup helper for NV simulator tests.

Copyright (c) 2023, IQO

This file is part of qudi.

Qudi is free software: you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

Qudi is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with qudi.
If not, see <https://www.gnu.org/licenses/>.
"""

import os
import sys
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('NV_Sim_Test_Environment')

def setup_environment():
    """Setup environment for testing."""
    # Set testing environment variable
    os.environ['QUDI_NV_TEST_MODE'] = '1'
    logger.info("Set QUDI_NV_TEST_MODE=1 for testing")
    
    # Add parent directories to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    qudi_dir = os.path.abspath(os.path.join(current_dir, '../../../..'))
    
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        logger.debug(f"Added {parent_dir} to sys.path")
        
    if qudi_dir not in sys.path:
        sys.path.insert(0, qudi_dir)
        logger.debug(f"Added {qudi_dir} to sys.path")
    
    # Add fixed modules directory to path
    fixed_modules_dir = os.path.join(current_dir, 'fixed_modules')
    if fixed_modules_dir not in sys.path:
        sys.path.insert(0, fixed_modules_dir)
        logger.debug(f"Added {fixed_modules_dir} to sys.path")
    
    # Reset QudiFacade singleton before each test
    try:
        from qudi.hardware.nv_simulator.qudi_facade import QudiFacade
        QudiFacade.reset_instance()
        logger.info("Reset QudiFacade singleton")
    except ImportError:
        logger.warning("Could not import QudiFacade to reset singleton")
    
    # Create results directory
    results_dir = os.path.join(current_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    logger.debug(f"Created results directory: {results_dir}")
    
    return {
        'test_mode': True,
        'results_dir': results_dir
    }

# Auto-execute when imported
env_info = setup_environment()