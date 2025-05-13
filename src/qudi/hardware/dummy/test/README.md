# NV Simulator ODMR Tests

This directory contains test scripts to diagnose communication issues between the microwave device, finite sampler, ODMR logic, and GUI.

## Key Files

- `test_mw_sampler_sync.py`: Direct test of communication between microwave device and finite sampler
- `test_odmr_flow.py`: Test of the full ODMR information flow through the logic layer
- `run_odmr_test.py`: Runner script that executes tests and visualizes results
- `test_runner.py`: **NEW** Unified test runner that works with or without Qudi installed

## Common Issues Fixed

1. **Module Import Error**: Fixed missing `qudi_main_weakref` and `name` parameters in all modules
2. **Base Class Availability**: Added mock implementations of Qudi core classes in `fixed_modules/`
3. **Missing Logging**: Ensured logging is properly configured for all test modules
4. **Singleton Initialization Error**: Added environment variable control and singleton reset method
5. **Missing scan_next Method**: Added scan_next() method to microwave_dummy for compatibility
6. **Multiple Test Issues**: Modified test runner to reset singleton between tests
7. **Environment Setup**: Added centralized environment setup script for consistency

## Running the Tests

### Recommended Method

Use the provided scripts which set the necessary environment variables:

Windows:
```
start_tests.bat
```

Linux/Mac:
```
./env_setup.sh
```

### Alternative Method (Works on all platforms)

Use the unified test runner:

```bash
# Set environment variable first (Windows)
SET QUDI_NV_TEST_MODE=1

# Set environment variable first (Linux/Mac)
export QUDI_NV_TEST_MODE=1

# Then run tests
python test_runner.py all

# Or specific test(s)
python test_runner.py mw_sampler_sync odmr_flow
```

### Alternative Methods

Individual test scripts can also be run directly:

```bash
# Basic frequency sweep test
python test_mw_sampler_sync.py

# Complete ODMR flow test
python test_odmr_flow.py

# Visualization and analysis
python run_odmr_test.py
```

## Expected Results

With a magnetic field of 500 Gauss along the z-axis, you should observe resonance dips at:
- 1.47 GHz (2.87 GHz - 1.4 GHz)
- 4.27 GHz (2.87 GHz + 1.4 GHz)

## Troubleshooting

If you encounter errors:

1. Use the `test_runner.py` script first, which handles most import issues automatically
2. Check logs in the `.log` files for detailed error information
3. Verify that `fixed_modules/` is in your Python path when running without Qudi
4. Make sure the NV model can be imported properly
5. If you encounter QtProperty errors with test_mode, it's been removed as it caused incompatibility with real Qudi

## ODMR GUI Fix

If your ODMR GUI is showing a flat line instead of resonance dips, these issues have been fixed:

1. **Main Issue**: The ODMR logic was calling `reset_scan()` after each data point, which resets to the first frequency. It now calls `scan_next()` when available to properly step through the frequency list.

2. **Compatibility Fix**: Added `scan_next()` method to the dummy microwave device for compatibility with the NV simulator.

3. **Singleton Management**: Added proper singleton reset capability to avoid initialization conflicts.

4. **Test Environment**: Added environment variable control to allow tests to create new instances.

The most important change is in `odmr_logic.py`, which now has this improved code:

```python
# Instead of just calling reset_scan()
if hasattr(self._microwave(), 'scan_next'):
    self._microwave().scan_next()  # Move to next frequency
else:
    self._microwave().reset_scan()  # Fall back for backward compatibility
```

## Mock Modules (fixed_modules/)

The `fixed_modules/` directory contains mock implementations of Qudi core classes needed for testing:

- `qudi_core.py`: Base, ModuleState, ConfigOption, Connector classes
- `microwave_interface.py`: MicrowaveInterface and constraints
- `finite_sampling_interface.py`: FiniteSamplingInputInterface and constraints

These are automatically used if the real Qudi modules are not available.

## Results Directory

Test results and plots are saved in a `results/` subdirectory, created automatically by the tests.