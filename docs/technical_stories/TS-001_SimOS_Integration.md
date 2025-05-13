# Technical Story TS-001: SimOS Integration

## Overview

Integrate the SimOS quantum physics library into our NV simulator to replace the current simplified implementation with a fully quantum-mechanical model.

## Description

The current `PhysicalNVModel` class in `src/model.py` implements a simplified model of NV center physics using basic population vectors and simplified evolution equations. We need to replace this with SimOS's scientifically accurate quantum mechanical model to enable proper simulation of complex quantum phenomena.

## Goals

- Replace the simplified population-based model with SimOS's full density matrix approach
- Implement proper Hamiltonian construction using SimOS's NVSystem class
- Enable accurate simulation of quantum dynamics with decoherence effects
- Maintain compatibility with existing interface methods and experimental simulations

## Implementation Plan

### 1. Setup Imports and Directory Structure

```python
import numpy as np
import logging
import threading
from typing import List, Dict, Any, Optional, Tuple, Union
import sys
import os

# Add SimOS to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'sim', 'simos'))

# Import SimOS components
import simos
from simos.systems.NV import NVSystem, decay_rates
from simos.core import globalclock
from simos.states import thermal_dm
from simos.coherent import auto_zeeman_interaction
from simos.propagation import evol

# Configure logging
logger = logging.getLogger(__name__)
```

### 2. Modify the PhysicalNVModel Class Constructor

Update the constructor to initialize a SimOS NVSystem instance and set up proper quantum state representation:

```python
def __init__(self, **config):
    """
    Initialize the NV center physical model.
    
    Parameters
    ----------
    **config
        Configuration parameters for the NV system
    """
    # Lock for thread safety
    self.lock = threading.RLock()
    
    with self.lock:
        # Default configuration
        self.config = {
            "zero_field_splitting": 2.87e9,  # Hz
            "gyromagnetic_ratio": 28.0e9,    # Hz/T
            "temperature": 298.0,            # K
            "strain": 0.0,                   # Hz
            "t1": 5e-3,                      # s
            "t2": 500e-6,                    # s
            "t2_star": 1e-6,                 # s
            "nitrogen": False,               # Include nitrogen nuclear spin
            "optics": True,                  # Include optical levels
            "orbital": False,                # Include orbital structure (low temp only)
            "method": "qutip"                # Numerical backend (qutip, numpy, sparse)
        }
        
        # Update with provided configuration
        self.config.update(config)
        
        # Initialize state variables
        self._magnetic_field = [0.0, 0.0, 0.0]  # T
        self.mw_frequency = 2.87e9  # Hz
        self.mw_power = 0.0         # dBm
        self.mw_on = False          # Microwave on/off
        self.laser_power = 0.0      # mW
        self.laser_on = False       # Laser on/off
        
        # Create NV System with SimOS
        self._nv_system = NVSystem(
            optics=self.config["optics"],
            orbital=self.config["orbital"],
            nitrogen=self.config["nitrogen"],
            method=self.config["method"]
        )
        
        # Initialize quantum state
        self._initialize_state()
        
        # Initialize Hamiltonian
        self._update_hamiltonian()
        
        # Initialize collapse operators for laser and relaxation
        self._update_collapse_operators()
```

### 3. Implement State Initialization

Replace the simplified state representation with SimOS's density matrix approach:

```python
def _initialize_state(self):
    """Initialize the quantum state to ground state (ms=0)."""
    # Use proper density matrix representation from SimOS
    if self.config["optics"]:
        # Initialize in ground state with ms=0
        self._state = thermal_dm(self._nv_system, 
                                 [("GS", 1.0), ("ES", 0.0), ("SS", 0.0), ("S", [1.0, 0.0, 0.0])], 
                                 method=self.config["method"])
    else:
        # Initialize in ms=0 state
        self._state = thermal_dm(self._nv_system, 
                                 [("S", [1.0, 0.0, 0.0])], 
                                 method=self.config["method"])
```

### 4. Implement Hamiltonian Construction

Create a method to update the Hamiltonian based on current field and MW conditions:

```python
def _update_hamiltonian(self):
    """Update the system Hamiltonian based on current fields."""
    # Convert magnetic field to proper units
    B_vec = np.array(self._magnetic_field)
    
    # Get Hamiltonian from NVSystem
    if self.config["optics"]:
        # For system with optical levels
        H_gs, H_es = self._nv_system.field_hamiltonian(B_vec)
        self._H_free = H_gs + H_es
    else:
        # For electronic spin only
        self._H_free = self._nv_system.field_hamiltonian(B_vec)
    
    # Add microwave driving term if enabled
    if self.mw_on and self.mw_power > 0:
        # Calculate Rabi frequency based on power
        rabi_freq = 10e6 * 10**((self.mw_power + 20) / 20)  # Hz
        
        # Convert to angular frequency
        omega_R = 2 * np.pi * rabi_freq
        
        # Create microwave driving Hamiltonian (rotating wave approximation)
        if self.config["optics"]:
            # Apply only to ground state
            mw_drive = 0.5 * omega_R * (
                self._nv_system.Sx * self._nv_system.GSid * np.cos(2*np.pi*self.mw_frequency*globalclock.time) +
                self._nv_system.Sy * self._nv_system.GSid * np.sin(2*np.pi*self.mw_frequency*globalclock.time)
            )
        else:
            # Apply to spin directly
            mw_drive = 0.5 * omega_R * (
                self._nv_system.Sx * np.cos(2*np.pi*self.mw_frequency*globalclock.time) +
                self._nv_system.Sy * np.sin(2*np.pi*self.mw_frequency*globalclock.time)
            )
        
        # Add driving to Hamiltonian
        self._H = self._H_free + mw_drive
    else:
        # No driving
        self._H = self._H_free
```

### 5. Add Collapse Operators for Incoherent Processes

SimOS provides accurate models for relaxation and optical processes:

```python
def _update_collapse_operators(self):
    """Update the system collapse operators for optical and relaxation processes."""
    if self.config["optics"]:
        # Get optical collapse operators with/without laser
        beta = 0.0 if not self.laser_on else self.laser_power / 5.0  # Normalize to saturation at ~5mW
        c_ops_with_laser, c_ops_without_laser = self._nv_system.transition_operators(
            T=self.config["temperature"],
            beta=beta,
            Bvec=np.array(self._magnetic_field)
        )
        
        # Use appropriate collapse operators based on laser state
        if self.laser_on:
            self._c_ops = c_ops_with_laser
        else:
            self._c_ops = c_ops_without_laser
    else:
        # Only spin relaxation for spin-only model
        self._c_ops = []
        
        # Add T1 relaxation (not fully accurate but simplified)
        t1 = self.config["t1"]
        if t1 > 0:
            gamma1 = 1.0 / t1
            # ms=±1 to ms=0 relaxation
            self._c_ops.append(np.sqrt(gamma1) * self._nv_system.Splus)
            self._c_ops.append(np.sqrt(gamma1) * self._nv_system.Sminus)
        
        # Add T2 pure dephasing (not fully accurate but simplified)
        t2 = self.config["t2"]
        t2_star = self.config["t2_star"]
        if t2 > 0 and t1 > 0:
            # Calculate pure dephasing rate (1/T2' = 1/T2 - 1/2T1)
            gamma_phi = 1.0/t2 - 1.0/(2*t1)
            if gamma_phi > 0:
                self._c_ops.append(np.sqrt(gamma_phi) * self._nv_system.Sz)
```

## Technical Risks

1. **Performance Impact**: The full quantum mechanical simulation is significantly more computationally intensive than the simplified model.

2. **Dependency Management**: Integrating SimOS introduces a complex dependency that needs to be properly managed.

3. **Backend Compatibility**: SimOS supports multiple backends (qutip, numpy, sparse) which might require conditional imports.

4. **API Compatibility**: Ensuring the new implementation maintains compatibility with existing API consumers.

## Testing Strategy

1. **Unit Tests**:
   - Test Hamiltonian generation with different magnetic field values
   - Test initialization to specific states
   - Test evolution with different parameters

2. **Validation Tests**:
   - Compare ODMR results against theoretical predictions
   - Validate Rabi oscillation frequency vs. power relationship
   - Check that T1 and T2 relaxation match expected values

3. **Integration Tests**:
   - Verify all existing experiments (ODMR, Rabi) produce expected results
   - Test compatibility with existing experiment scripts

## Acceptance Criteria

1. Unit tests pass with at least 90% coverage
2. ODMR and Rabi experiments produce results within 5% of theoretical values
3. Performance impact is acceptable (simulation time not more than 5x slower)
4. All existing API functionality maintained
5. Documentation updated to reflect the new implementation

## Estimation

- Core SimOS integration: 2 days
- Refactoring ODMR and Rabi methods: 1 day
- Unit and validation tests: 1 day
- Documentation: 0.5 day
- Total: 4.5 days of effort