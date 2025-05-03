# SimOS NV Simulator

[![CI](https://github.com/xleonplayz/IQ0-q/actions/workflows/ci.yml/badge.svg)](https://github.com/xleonplayz/IQ0-q/actions/workflows/ci.yml)

A realistic NV-center simulator with SimOS integration for the Qudi framework.

## Overview

The SimOS NV Simulator provides a comprehensive simulation of NV-centers in diamond, 
integrated with the Qudi framework for quantum experiments. It offers:

- Realistic quantum physics simulation using the SimOS library
- Implementation of all relevant Qudi hardware interfaces
- Thread-safe design for concurrent access from multiple modules
- Configurable simulation parameters for various experimental conditions
- Web-based configuration and monitoring interface

## Features

- Accurate simulation of NV-center quantum dynamics
- Support for ODMR, Rabi oscillations, and other standard quantum experiments
- Realistic simulation of environmental effects and noise
- Thread-safe access from multiple Qudi hardware interfaces
- Customizable parameters for different experimental scenarios

## Installation

### Requirements

- Python 3.8 or higher
- NumPy

### Installation from source

```bash
git clone https://github.com/xleonplayz/IQ0-q.git
cd IQ0-q
pip install -e .
```

For development, install the extra development dependencies:

```bash
pip install -e ".[dev]"
```

## Usage

The simulator provides implementations of Qudi hardware interfaces that can be configured
in your Qudi configuration file:

```python
# Qudi configuration example
hardware:
    microwave:
        module.Class: 'simos_nv_simulator.qudi_interfaces.microwave.MicrowaveSimulator'
        connect:
            port: 'localhost:5555'
            
    fast_counter:
        module.Class: 'simos_nv_simulator.qudi_interfaces.fast_counter.FastCounterSimulator'
        connect:
            port: 'localhost:5555'
```

## Development

### Running Tests

To run tests:

```bash
pytest
```

To run with coverage:

```bash
pytest --cov=simos_nv_simulator
```

### Code Formatting

This project uses Black for code formatting:

```bash
black simos_nv_simulator tests
```

## License

MIT

## Acknowledgements

This project builds upon the SimOS library for quantum simulation of optically
addressable spins.