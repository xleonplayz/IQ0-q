# NV Simulator ODMR Tests

This directory contains test scripts to diagnose communication issues between the microwave device, finite sampler, ODMR logic, and GUI.

## Key Files

- `test_mw_sampler_sync.py`: Direct test of communication between microwave device and finite sampler
- `test_odmr_flow.py`: Test of the full ODMR information flow through the logic layer
- `run_odmr_test.py`: Runner script that executes tests and visualizes results

## Common Issues Fixed

1. **Module Import Error**: Updated `QudiFacade` class to accept required Qudi parameters (`qudi_main_weakref` and `name`).
2. **Base Class Availability**: Added fallback mechanisms for importing the `Base` class when running in a non-Qudi environment.
3. **Missing Logging**: Ensured logging is properly configured for all test modules.

## Running the Tests

For basic frequency sweep test:
```
python test_mw_sampler_sync.py
```

For complete ODMR flow test:
```
python test_odmr_flow.py
```

For visualization and analysis:
```
python run_odmr_test.py
```

## Expected Results

With a magnetic field of 500 Gauss along the z-axis, you should observe resonance dips at:
- 1.47 GHz (2.87 GHz - 1.4 GHz)
- 4.27 GHz (2.87 GHz + 1.4 GHz)

## Troubleshooting

If you encounter additional issues:

1. Check for any import errors in the console output
2. Look for debugging information in the log files:
   - `mw_sampler_sync_test.log` (for test_mw_sampler_sync.py)
   - `odmr_flow_test.log` (for test_odmr_flow.py)
3. Examine the plots in the `results` directory
4. Check the shared state in QudiFacade to see if frequency updates are being properly tracked

## Results Directory

Test results and plots are saved in a `results` subdirectory, including:
- Frequency sweep plots
- ODMR signal plots
- Resonance analysis