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
4. **Singleton Initialization Error**: Added `test_mode=True` parameter to allow multiple QudiFacade instances 
5. **API Compatibility**: Made modules work with both older and newer Qudi versions

## Running the Tests

### Recommended Method (Works with or without Qudi installed)

Use the unified test runner:

```bash
# Run all tests
python test_runner.py all

# Run specific test(s)
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

## Mock Modules (fixed_modules/)

The `fixed_modules/` directory contains mock implementations of Qudi core classes needed for testing:

- `qudi_core.py`: Base, ModuleState, ConfigOption, Connector classes
- `microwave_interface.py`: MicrowaveInterface and constraints
- `finite_sampling_interface.py`: FiniteSamplingInputInterface and constraints

These are automatically used if the real Qudi modules are not available.

## Results Directory

Test results and plots are saved in a `results/` subdirectory, created automatically by the tests.