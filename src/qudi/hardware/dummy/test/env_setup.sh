#!/bin/bash
# Shell script to set environment variables and run tests

# Set test mode environment variable
export QUDI_NV_TEST_MODE=1
echo "Set QUDI_NV_TEST_MODE=1 for testing"

# Run the test runner with all tests
python test_runner.py all