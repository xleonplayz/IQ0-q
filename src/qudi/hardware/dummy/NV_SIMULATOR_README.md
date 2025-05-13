# NV Center Simulator for Qudi

This directory contains the NV center simulator integration for Qudi, which allows simulating ODMR (Optically Detected Magnetic Resonance) experiments with NV centers in diamond.

## Overview

The simulator provides realistic ODMR signals with resonance dips at frequencies determined by the zero-field splitting and the applied magnetic field. It can be used to test Qudi's ODMR functionality without actual hardware.

### Key Components

- **nv_simulator_manager.py**: Main manager class that maintains the NV center state and handles communication between modules
- **nv_simple_model.py**: Simplified NV model for when the full simulator is not available
- **microwave_dummy.py**: Microwave device that interfaces with the NV simulator
- **finite_sampling_input_dummy.py**: Data acquisition module that generates ODMR signals
- **nv_simulator_config.cfg**: Configuration file for running Qudi with the NV simulator

## Running ODMR with the NV Simulator

You can use the `start_nv_simulator.py` script in the main project directory to launch Qudi with the NV simulator:

```bash
python start_nv_simulator.py
```

Or use Qudi directly with the simulator configuration:

```bash
qudi -c src/qudi/hardware/dummy/nv_simulator_config.cfg
```

### Expected ODMR Resonances

With the default magnetic field of 500 Gauss along the z-axis:

- First resonance: ~1.47 GHz (ms=0 → ms=-1)
- Second resonance: ~4.27 GHz (ms=0 → ms=+1)

## Troubleshooting

If you encounter issues with the simulator:

1. **Verify NV simulator initialization**:
   ```bash
   python -m src.qudi.hardware.dummy.test.verify_nv_simulator
   ```
   This will check that the simulator is working and plot the expected ODMR spectrum.

2. **Check ODMR frequency range**:
   Make sure your ODMR scan covers the expected resonance frequencies. For 500 Gauss, scan from 1.4 GHz to 4.4 GHz.

3. **Check logs**:
   Examine the Qudi logs for errors related to the NV simulator.

4. **Run diagnostic test**:
   ```bash
   python -m src.qudi.hardware.dummy.odmr_debug
   ```
   This will run various diagnostics to help identify issues.

## Configuration Options

The NV simulator can be configured with these options:

```python
microwave_dummy:
    module.Class: 'dummy.microwave_dummy.MicrowaveDummy'
    options:
        use_nv_simulator: True  # Enable NV simulator
        magnetic_field: [0, 0, 500]  # 500 Gauss along z-axis
        temperature: 300  # Room temperature

finite_sampling_input_dummy:
    module.Class: 'dummy.finite_sampling_input_dummy.FiniteSamplingInputDummy'
    options:
        simulation_mode: "ODMR"
        use_nv_simulator: True  # Enable NV simulator 
        magnetic_field: [0, 0, 500]  # 500 Gauss along z-axis
        temperature: 300  # Room temperature
```

## Advanced Usage

### Changing Magnetic Field

You can modify the magnetic field to see how it affects the ODMR spectrum. The resonance frequencies are determined by:

```
resonance1 = zero_field_splitting - gyromagnetic_ratio * magnetic_field
resonance2 = zero_field_splitting + gyromagnetic_ratio * magnetic_field
```

Where:
- Zero-field splitting = 2.87 GHz
- Gyromagnetic ratio = 2.8 MHz/Gauss

## Testing

The NV simulator includes several test scripts:

- `verify_nv_simulator.py`: Verify the simulator is working and plot the ODMR spectrum
- `test_frequency_chain.py`: Test the frequency scanning chain
- `test_mw_sampler_sync.py`: Test synchronization between microwave and sampler
- `odmr_debug.py`: Run diagnostics for ODMR setup

## Notes

- The simulator requires Python 3.7 or later
- For best results, run with microwave power around -20 dBm
- The simulated contrast is around 5%, which is realistic for NV centers