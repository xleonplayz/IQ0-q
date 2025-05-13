# Fixed Modules for Testing

This directory contains mock implementations of Qudi core modules needed for testing 
the NV simulator without a full Qudi installation.

## Contents

- `qudi_core.py`: Base class implementations (Base, ModuleState, ConfigOption, Connector)
- `microwave_interface.py`: MicrowaveInterface and MicrowaveConstraints
- `finite_sampling_interface.py`: FiniteSamplingInputInterface and FiniteSamplingInputConstraints

## Usage

These modules should automatically be used if the real Qudi modules are not available.
The test scripts have been modified to handle the import of these modules as fallbacks.

## Mock Implementation

These are simplified versions of the real Qudi modules, containing only the minimum
functionality needed for testing. They implement the same interfaces but with simplified
logic and without dependencies on the full Qudi framework.