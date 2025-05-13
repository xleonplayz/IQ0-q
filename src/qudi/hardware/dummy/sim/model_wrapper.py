"""
Wrapper module to import and use the PhysicalNVModel from the sim package.

This wrapper helps avoid import path issues when working with the simulator.
"""

import os
import sys
import importlib.util
import logging
import traceback

# Configure logging
logger = logging.getLogger(__name__)

def import_model():
    """Import and return the PhysicalNVModel class.
    
    Returns:
        class: The PhysicalNVModel class
    """
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    logger.debug(f"model_wrapper.py located at: {current_dir}")
    
    # Add all possible paths to sys.path to ensure we can find the model
    paths_to_check = [
        # 1. Try local import from src/
        os.path.join(current_dir, 'src'),
        
        # 2. Make sure sim is in path for sim.simos imports
        os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..', '..', 'sim')),
        
        # 3. For imports within model.py that use sim.simos...
        os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..')),
        
        # 4. For Windows path style
        os.path.abspath(os.path.join(current_dir, 'src')),
        
        # 5. Absolute path to sim/src directory
        os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..', '..', 'sim', 'src')),
    ]
    
    # Add all valid paths to sys.path
    for path in paths_to_check:
        if os.path.exists(path) and path not in sys.path:
            sys.path.insert(0, path)
            logger.debug(f"Added {path} to Python path")
    
    logger.debug(f"Python path now contains: {sys.path}")
    
    # Try to import the model in various ways
    try:
        # Try direct import first
        try:
            from model import PhysicalNVModel
            logger.info("Successfully imported PhysicalNVModel directly")
            return PhysicalNVModel
        except ImportError as e1:
            logger.debug(f"Direct import failed: {e1}")
            
            # Try importing from src
            try:
                from src.model import PhysicalNVModel
                logger.info("Successfully imported PhysicalNVModel from src")
                return PhysicalNVModel
            except ImportError as e2:
                logger.debug(f"src.model import failed: {e2}")
                
                # Try using importlib
                try:
                    spec = None
                    for path in paths_to_check:
                        model_path = os.path.join(path, 'model.py')
                        if os.path.exists(model_path):
                            logger.debug(f"Found model.py at {model_path}")
                            spec = importlib.util.spec_from_file_location("model", model_path)
                            break
                    
                    if spec is not None:
                        model_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(model_module)
                        PhysicalNVModel = getattr(model_module, 'PhysicalNVModel')
                        logger.info("Successfully imported PhysicalNVModel using importlib")
                        return PhysicalNVModel
                    else:
                        raise ImportError("Could not find model.py in any of the search paths")
                except Exception as e3:
                    logger.debug(f"importlib import failed: {e3}")
                    # Final fallback
                    try:
                        # Try importing from sim.src
                        from sim.src.model import PhysicalNVModel
                        logger.info("Successfully imported PhysicalNVModel from sim.src")
                        return PhysicalNVModel
                    except ImportError as e4:
                        logger.debug(f"sim.src.model import failed: {e4}")
                        raise ImportError("All import methods failed")
    except Exception as e:
        logger.error(f"Failed to import PhysicalNVModel: {e}")
        logger.error(f"Python path: {sys.path}")
        logger.error(traceback.format_exc())
        raise

# Try the import
try:
    PhysicalNVModel = import_model()
    # Export the PhysicalNVModel class
    __all__ = ['PhysicalNVModel', 'import_model']
except Exception:
    logger.warning("Could not import PhysicalNVModel in model_wrapper.py initialization")
    # Still export the import_model function
    __all__ = ['import_model']