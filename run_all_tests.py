#!/usr/bin/env python3
"""
Test runner for the simos_nv_simulator package.

This script runs all tests for the package, including:
- Standard unit tests
- Performance tests
- Integration tests

Example usage:
    python run_all_tests.py              # Run all tests
    python run_all_tests.py -v           # Run all tests with verbose output
    python run_all_tests.py -k quantum   # Run only tests with 'quantum' in the name
    python run_all_tests.py -m slow      # Run only tests marked as 'slow'
"""

import argparse
import sys
import pytest


def main():
    """Run all tests for the simos_nv_simulator package."""
    parser = argparse.ArgumentParser(description='Run tests for the simos_nv_simulator package')
    parser.add_argument('-v', '--verbose', action='store_true', 
                        help='Verbose output')
    parser.add_argument('-k', '--keyword', type=str, default=None,
                        help='Only run tests that match the given keyword expression')
    parser.add_argument('-m', '--marker', type=str, default=None,
                        help='Only run tests with the given marker')
    parser.add_argument('--skip-slow', action='store_true',
                        help='Skip slow tests')
    
    args = parser.parse_args()
    
    # Build pytest arguments
    pytest_args = []
    
    # Add verbosity
    if args.verbose:
        pytest_args.append('-v')
    
    # Add keyword filter
    if args.keyword:
        pytest_args.extend(['-k', args.keyword])
    
    # Add marker filter
    if args.marker:
        pytest_args.extend(['-m', args.marker])
    elif args.skip_slow:
        pytest_args.extend(['-m', 'not slow'])
    
    # Run tests and return exit code
    return pytest.main(pytest_args)


if __name__ == '__main__':
    sys.exit(main())