@echo off
REM Batch file to set environment variables and run tests

REM Set test mode environment variable
SET QUDI_NV_TEST_MODE=1
echo Set QUDI_NV_TEST_MODE=1 for testing

REM Run the test runner with all tests
python test_runner.py all

REM Pause to see results
pause