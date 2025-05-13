"""
Wrapper module to import and use the PhysicalNVModel from the sim package.

This wrapper helps avoid import path issues when working with the simulator.
"""

import os
import sys
import importlib.util
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Try local import from src/
src_dir = os.path.join(current_dir, 'src')
if os.path.exists(src_dir):
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    logger.info(f"Added {src_dir} to Python path")

# 2. Make sure sim is in path for sim.simos imports
sim_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..', '..', 'sim'))
if os.path.exists(sim_dir):
    if sim_dir not in sys.path:
        sys.path.insert(0, sim_dir)
    logger.info(f"Added {sim_dir} to Python path")

# 3. For imports within model.py that use sim.simos...
dummy_sim_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
if dummy_sim_dir not in sys.path:
    sys.path.insert(0, dummy_sim_dir)
    logger.info(f"Added {dummy_sim_dir} to Python path")

# Import the model 
try:
    from model import PhysicalNVModel
    logger.info("Successfully imported PhysicalNVModel from model.py")
except ImportError as e:
    logger.error(f"Could not import PhysicalNVModel: {e}")
    logger.error(f"Python path: {sys.path}")
    raise

# Export the PhysicalNVModel class
__all__ = ['PhysicalNVModel']