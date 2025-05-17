#!/usr/bin/env python
"""
Start the NV Simulator API server.

This script starts the FastAPI server that provides the NV simulator API,
which can be used by Qudi-IQO modules to interact with a realistic NV center simulation.
"""

import argparse
import logging
import os
import sys
import time
import numpy as np
from pathlib import Path

# Add package to Python path
script_dir = Path(__file__).resolve().parent
package_dir = Path(script_dir).parent.parent
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))

from nv.api.server import run_server
from nv.config_loader import load_config

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Start the NV Simulator API server")
    parser.add_argument("-H", "--host", type=str, default="0.0.0.0",
                      help="Host address to bind to (default: 0.0.0.0)")
    parser.add_argument("-p", "--port", type=int, default=5000,
                      help="Port to bind to (default: 5000)")
    parser.add_argument("-k", "--api-key", type=str,
                      help="API key for authentication (default: from environment or 'dev-key')")
    parser.add_argument("-v", "--verbose", action="store_true",
                      help="Enable verbose logging")
    parser.add_argument("-m", "--magnetic-field", type=float, default=0.001,
                      help="Initial magnetic field magnitude in Tesla (default: 0.001)")
    parser.add_argument("-d", "--magnetic-field-dir", type=str, default="0,0,1",
                      help="Initial magnetic field direction (comma-separated, default: 0,0,1)")
    return parser.parse_args()

def main():
    """Main entry point."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Set API key if provided
    if args.api_key:
        os.environ["NV_API_KEY"] = args.api_key
    
    # Parse magnetic field direction
    try:
        mag_field_dir = [float(x) for x in args.magnetic_field_dir.split(',')]
        if len(mag_field_dir) != 3:
            raise ValueError("Magnetic field direction must have 3 components")
    except Exception as e:
        logger.error(f"Invalid magnetic field direction: {e}")
        sys.exit(1)
    
    # Normalize magnetic field direction
    mag_field_dir = np.array(mag_field_dir)
    norm = np.linalg.norm(mag_field_dir)
    if norm > 0:
        mag_field_dir = mag_field_dir / norm
    else:
        mag_field_dir = np.array([0, 0, 1])
    
    # Try to load and update configuration
    try:
        config_loader = load_config()
        config = config_loader.get_nv_system_config()
        
        # Update magnetic field in config
        config["magnetic_field"] = [
            args.magnetic_field * mag_field_dir[0],
            args.magnetic_field * mag_field_dir[1],
            args.magnetic_field * mag_field_dir[2]
        ]
        
        logger.info(f"Updated magnetic field in configuration: {config['magnetic_field']}")
    except Exception as e:
        logger.warning(f"Failed to update configuration: {e}")
    
    # Print information
    print("=" * 80)
    print(f"NV Simulator API Server")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"API Key: {'default' if not args.api_key else 'custom'}")
    print(f"Magnetic Field: {args.magnetic_field} T along [{mag_field_dir[0]:.2f}, {mag_field_dir[1]:.2f}, {mag_field_dir[2]:.2f}]")
    print(f"Starting time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print("\nPress Ctrl+C to stop the server\n")
    
    # Run server
    run_server(host=args.host, port=args.port)

if __name__ == "__main__":
    main()