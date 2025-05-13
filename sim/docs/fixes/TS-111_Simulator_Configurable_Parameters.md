# TS-111: Simulator Configurable Parameters

## Summary
This document provides a comprehensive list of all configurable parameters in the NV simulator that can be manipulated by users to adjust the physical model, hardware behavior, and experimental settings.

## Physical Model Parameters

### Core NV Center Physics
| Parameter | Default Value | Units | Location | Configuration Method |
|-----------|---------------|-------|----------|---------------------|
| magnetic_field | [0, 0, 0] | Gauss (vector) | qudi_facade.py | Config option or `set_magnetic_field(field)` |
| zero_field_splitting | 2.87e9 | Hz | model.py, qudi_facade.py | Config option or initialization parameter |
| gyromagnetic_ratio | 2.8025e10 | Hz/T | model.py, qudi_facade.py | Config option or initialization parameter |
| strain | 0.0 | Hz | model.py | Initialization parameter |
| temperature | 300 | Kelvin | qudi_facade.py | Config option or `set_temperature(temp)` |

### Relaxation and Coherence Times
| Parameter | Default Value | Units | Location | Configuration Method |
|-----------|---------------|-------|----------|---------------------|
| t1 | 5.0e-3 | seconds | model.py, qudi_facade.py | Config option or initialization parameter |
| t2 | 1.0e-5 | seconds | model.py, qudi_facade.py | Config option or initialization parameter |

### Optical Properties
| Parameter | Default Value | Units | Location | Configuration Method |
|-----------|---------------|-------|----------|---------------------|
| collection_efficiency | 1.0 | relative (0-1) | model.py | `set_collection_efficiency(efficiency)` |
| laser_power | 0.0 | mW | model.py | `set_laser_power(power)` or `apply_laser(power, on=True)` |

### Nuclear Spin Environment
| Parameter | Default Value | Units | Location | Configuration Method |
|-----------|---------------|-------|----------|---------------------|
| c13_concentration | 0.011 | relative | model.py | Initialization parameter |
| nitrogen | False | boolean | model.py | Initialization parameter |

### Simulation Method
| Parameter | Default Value | Units | Location | Configuration Method |
|-----------|---------------|-------|----------|---------------------|
| optics | True | boolean | model.py | Initialization parameter |
| method | "qutip" | string | model.py | Initialization parameter |
| thread_safe | True | boolean | model.py | Initialization parameter |

## Hardware Module Parameters

### Laser Module
| Parameter | Default Value | Units | Location | Configuration Method |
|-----------|---------------|-------|----------|---------------------|
| wavelength | 532 | nm | nv_simulator.cfg | Config option |
| max_power | 100.0 | mW | nv_simulator.cfg | Config option |
| power_noise | 0.01 | relative | nv_simulator.cfg | Config option |

### Microwave Module
| Parameter | Default Value | Units | Location | Configuration Method |
|-----------|---------------|-------|----------|---------------------|
| microwave_frequency | 2.87e9 | Hz | model.py | `set_microwave_frequency(freq)` or `apply_microwave(freq, power_dbm, on=True)` |
| microwave_amplitude | 0.0 | relative | model.py | `set_microwave_amplitude(amp)` |
| fixed_startup_time | 0.2 | seconds | nv_simulator.cfg | Config option |

### Fast Counter
| Parameter | Default Value | Units | Location | Configuration Method |
|-----------|---------------|-------|----------|---------------------|
| photon_rate | 100000 | counts/s | nv_simulator.cfg | Config option |
| noise_factor | 0.1 | relative | nv_simulator.cfg | Config option |
| dark_counts | 200 | counts/s | nv_simulator.cfg | Config option |
| time_jitter | 0.5e-9 | seconds | nv_simulator.cfg | Config option |

### Scanning Probe
| Parameter | Default Value | Units | Location | Configuration Method |
|-----------|---------------|-------|----------|---------------------|
| position_ranges | x: [0, 100e-6],<br> y: [0, 100e-6],<br> z: [-50e-6, 50e-6] | meters | nv_simulator.cfg | Config option |
| frequency_ranges | x: [1, 1000],<br> y: [1, 1000],<br> z: [1, 500] | Hz | nv_simulator.cfg | Config option |
| resolution_ranges | x: [1, 1000],<br> y: [1, 1000],<br> z: [2, 500] | pixels | nv_simulator.cfg | Config option |
| position_accuracy | x: 10e-9,<br> y: 10e-9,<br> z: 50e-9 | meters | nv_simulator.cfg | Config option |
| nv_density | 1e15 | 1/m³ | nv_simulator.cfg | Config option |

## Configuration Examples

### Example 1: Setting up a custom NV center with specific parameters

```python
# Import the model
from model import PhysicalNVModel

# Create a model with custom physics parameters
model = PhysicalNVModel(
    zero_field_splitting=2.87e9,  # D value in Hz
    gyromagnetic_ratio=28.025e9,  # γe in Hz/T
    t1=4.5e-3,                    # T1 time in seconds
    t2=2.0e-5,                    # T2 time in seconds
    strain=10e6,                  # Strain in Hz  
    temperature=293,              # Room temperature in K
    c13_concentration=0.01,       # 1% 13C
    optics=True,                  # Include optical levels
    nitrogen=True,                # Include nitrogen nuclear spin
    method="qutip"                # Use QuTiP solver
)

# Set magnetic field (in Tesla)
model.set_magnetic_field([0, 0, 0.05])  # 500 Gauss along z-axis

# Set collection efficiency
model.set_collection_efficiency(0.8)  # 80% collection efficiency
```

### Example 2: Configuration File Section for NV Simulator

```
# NV Simulator Configuration
nv_simulator:
    module.Class: 'nv_simulator.qudi_facade.QudiFacade'
    options:
        magnetic_field: [0, 0, 100]        # 100 Gauss along z-axis
        temperature: 293                   # Room temperature in K
        zero_field_splitting: 2.87e9       # D value in Hz
        gyromagnetic_ratio: 2.8025e10      # γe in Hz/T
        t1: 5.0e-3                         # T1 time in s
        t2: 1.0e-5                         # T2 time in s
        thread_safe: True                  # Use thread locking

# Fast Counter Configuration
nv_sim_fastcounter:
    module.Class: 'nv_simulator.fast_counter.NVSimFastCounter'
    options:
        gated: True                        # Enable gated counting
        photon_rate: 120000                # Photon count rate in c/s
        noise_factor: 0.05                 # 5% noise level
        dark_counts: 100                   # Lower dark counts
        time_jitter: 0.2e-9                # Better timing resolution
        t1: 5.5e-6                         # T1 for this module
        t2: 2.0e-6                         # T2 for this module
    connect:
        simulator: nv_simulator
```

## Usage Examples for Physical Simulations

### ODMR Experiment
```python
# Run ODMR with specific parameters
odmr_result = model.simulate_odmr(
    f_min=2.8e9,        # Start frequency in Hz
    f_max=2.9e9,        # End frequency in Hz
    n_points=101,       # Number of frequency points
    mw_power=-10.0      # Microwave power in dBm
)

# Access ODMR results
frequencies = odmr_result.frequencies
signal = odmr_result.signal
resonances = odmr_result.resonances
zeeman_shift = odmr_result.zeeman_shift
```

### Rabi Oscillation Experiment
```python
# Run Rabi oscillation experiment
rabi_result = model.simulate_rabi(
    t_max=1e-6,         # Maximum Rabi time in seconds
    n_points=101,       # Number of time points
    mw_power=0.0,       # Microwave power in dBm
    mw_frequency=2.87e9 # Microwave frequency in Hz
)

# Access Rabi results
times = rabi_result.times
signal = rabi_result.signal
rabi_frequency = rabi_result.rabi_frequency
```

### T1 Relaxation Measurement
```python
# Run T1 relaxation experiment
t1_result = model.simulate_t1(
    t_max=20e-3,        # Maximum delay time in seconds
    n_points=101        # Number of time points
)

# Access T1 results
times = t1_result.times
signal = t1_result.signal
t1_value = t1_result.t1
```

## Best Practices

1. **Realistic Parameters**: Use physically realistic parameters for NV centers in diamond
   - Zero-field splitting: 2.87 GHz
   - Gyromagnetic ratio: 28.025 GHz/T
   - T1: 1-6 ms (depending on diamond quality)
   - T2: 1-300 μs (depending on 13C concentration)

2. **Strain Consideration**: For realistic simulations, include some strain (1-10 MHz)

3. **Temperature Effects**: Room temperature (293-300K) is typical for most NV experiments

4. **Magnetic Field**: Set field in Gauss units when configuring
   - Common range: 0-1000 Gauss
   - ODMR splitting: 5.6 MHz/Gauss

5. **Collection Efficiency**: Realistic values are 0.1-0.3 for standard confocal setups

6. **Configuration Changes**: Use API calls to change parameters dynamically during experiments