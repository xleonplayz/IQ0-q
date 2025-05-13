# Mock Qudi Interfaces

This directory contains mock implementations of Qudi interfaces for standalone testing without requiring a full Qudi installation. This is particularly useful for:

1. Testing the NV simulator interfaces without Qudi
2. Developing and debugging the simulator code independently
3. Running examples and demonstrations without dependencies

## Usage

To use the mock interfaces instead of real Qudi interfaces, modify your import statements as follows:

### Instead of importing from Qudi:

```python
from qudi.interface.microwave_interface import MicrowaveInterface, MicrowaveConstraints
from qudi.interface.fast_counter_interface import FastCounterInterface
from qudi.core.module import Base
```

### Import from the mock module:

```python
from sim.src.qudi_interface.mock import Base, MicrowaveInterface, MicrowaveConstraints, FastCounterInterface
```

## Available Mock Interfaces

The mock module provides simplified versions of these Qudi components:

- **Core Components**:
  - `Base` - Base class for all Qudi modules
  - `ModuleState` - Module state handling

- **Interfaces**:
  - `MicrowaveInterface` - Microwave source control
  - `FastCounterInterface` - Photon counting and time-resolved measurements 
  - `PulserInterface` - Pulse sequence generation and control
  - `ScanningProbeInterface` - Scanning probe and confocal microscopy
  - `SimpleLaserInterface` - Laser control and monitoring

- **Supporting Classes**:
  - `MicrowaveConstraints` - Constraints for microwave devices
  - `PulserConstraints` - Constraints for pulse generators
  - `ScanConstraints` - Constraints for scanning devices
  - `ScannerAxis` - Axis definition for scanners
  - `ScannerChannel` - Channel definition for scanners
  - `ScannerSettings` - Settings for scanner operation

- **Enums**:
  - `ControlMode` - Laser control modes
  - `ShutterState` - Laser shutter states 
  - `LaserState` - Laser operational states
  - `BackScanCapability` - Scanner back scan capabilities
  - `SamplingOutputMode` - Microwave scan modes

## Limitations

The mock interfaces provide only the minimum functionality needed for testing:

- Error handling is simplified
- Some complex validations are omitted
- GUI components are not included
- Only the interfaces used by the NV simulator are implemented

## Example

```python
# Example using mock interfaces for testing

from sim.src.qudi_interface.mock import Base, SimpleLaserInterface, LaserState
from sim.src.qudi_interface.hardware import NVSimLaser

# Create and test the laser interface
laser = NVSimLaser()
laser.on_activate()

# Control the laser
laser.on()
print(f"Laser state: {laser.get_laser_state() == LaserState.ON}")
print(f"Laser power: {laser.get_power()} mW")

# Clean up
laser.off()
laser.on_deactivate()
```