# Technical Story TS-002: Quantum Evolution Implementation

## Overview

Implement an accurate quantum evolution engine using SimOS to replace the simplified population-based model, allowing for proper simulation of quantum coherence, decoherence, and quantum state evolution.

## Description

The current implementation uses a simplified population-based model for time evolution that cannot accurately represent quantum coherences. This story focuses on implementing a proper quantum mechanical evolution system based on SimOS, which will enable accurate simulation of quantum superposition states, decoherence processes, and more complex quantum experiments.

## Goals

- Replace the simplified evolution model with SimOS's full quantum mechanical evolution
- Implement both unitary evolution for closed systems and master equation solvers for open systems
- Support decoherence processes with configurable T1, T2, and T2* parameters
- Ensure proper handling of time-dependent Hamiltonians for microwave driving

## Implementation Plan

### 1. Implement State Evolution Method

Replace the current `_evolve` method with proper quantum evolution based on SimOS:

```python
def simulate(self, dt, steps=1):
    """
    Simulate the quantum evolution for a given time.
    
    Parameters
    ----------
    dt : float
        Time step in seconds
    steps : int, optional
        Number of time steps to simulate
    """
    with self.lock:
        total_time = dt * steps
        
        # If no collapse operators, use unitary evolution
        if not self._c_ops:
            # For unitary evolution, use efficient propagator
            propagator = evol(self._H, total_time)
            self._state = propagator * self._state * propagator.dag()
        else:
            # For open quantum systems with collapse operators, 
            # use master equation evolution (requires Qutip backend)
            if self.config["method"] != "qutip":
                logger.warning("Switching to qutip backend for master equation solver")
                
            # Use mesolve from qutip
            import qutip as qt
            
            # Define time points
            tlist = np.linspace(0, total_time, steps + 1)
            
            # Solve master equation
            result = qt.mesolve(self._H, self._state, tlist, self._c_ops, [])
            
            # Update state to final result
            self._state = result.states[-1]
        
        # Update global time
        globalclock.inc(total_time)
```

### 2. Implement Specialized Evolution for Different Scenarios

Extend the basic evolution with specialized methods for different types of experiments:

```python
def evolve_with_rabi(self, dt, mw_frequency, mw_power):
    """
    Evolve the system with a microwave drive at specified frequency and power.
    
    Parameters
    ----------
    dt : float
        Time step in seconds
    mw_frequency : float
        Microwave frequency in Hz
    mw_power : float
        Microwave power in dBm
    """
    with self.lock:
        # Save current microwave state
        original_mw = (self.mw_frequency, self.mw_power, self.mw_on)
        
        try:
            # Apply microwave drive
            self.apply_microwave(mw_frequency, mw_power, True)
            
            # Evolve
            self.simulate(dt)
            
        finally:
            # Restore original state
            self.apply_microwave(original_mw[0], original_mw[1], original_mw[2])
```

```python
def evolve_pulse_sequence(self, sequence, dt=1e-9):
    """
    Evolve the system through a pulse sequence.
    
    Parameters
    ----------
    sequence : list
        List of (pulse_type, duration, parameters) tuples
    dt : float, optional
        Time step in seconds for continuous evolution
    
    Returns
    -------
    dict
        Results of the pulse sequence
    """
    with self.lock:
        # Save original state
        original_mw = (self.mw_frequency, self.mw_power, self.mw_on)
        original_laser = (self.laser_power, self.laser_on)
        
        try:
            # Process sequence
            results = {}
            
            for i, (pulse_type, duration, params) in enumerate(sequence):
                if pulse_type == "wait":
                    # Free evolution
                    self.apply_microwave(0.0, 0.0, False)
                    self.apply_laser(0.0, False)
                    self.simulate(duration)
                    
                elif pulse_type == "mw":
                    # Microwave pulse
                    frequency = params.get("frequency", self.mw_frequency)
                    power = params.get("power", 0.0)
                    self.apply_microwave(frequency, power, True)
                    self.simulate(duration)
                    
                elif pulse_type == "laser":
                    # Laser initialization
                    power = params.get("power", 1.0)
                    self.apply_laser(power, True)
                    self.simulate(duration)
                    
                # Store state after each pulse if requested
                if params.get("store_state", False):
                    results[f"state_after_pulse_{i}"] = self._state.copy()
                    
                # Measure after each pulse if requested
                if params.get("measure", False):
                    results[f"population_after_pulse_{i}"] = self.get_populations()
                    if params.get("fluorescence", False):
                        results[f"fluorescence_after_pulse_{i}"] = self.get_fluorescence()
            
            return results
            
        finally:
            # Restore original state
            self.apply_microwave(original_mw[0], original_mw[1], original_mw[2])
            self.apply_laser(original_laser[0], original_laser[1])
```

### 3. Implement Advanced Evolution Methods for Specific NV Experiments

Add specialized methods for common NV experiments:

```python
def evolve_with_ramsey(self, free_evolution_time, pi_half_pulse_duration, 
                       mw_frequency=None, mw_power=0.0):
    """
    Perform a Ramsey experiment.
    
    Parameters
    ----------
    free_evolution_time : float
        Time between pi/2 pulses in seconds
    pi_half_pulse_duration : float
        Duration of pi/2 pulses in seconds
    mw_frequency : float, optional
        Microwave frequency in Hz. If None, use resonance frequency.
    mw_power : float, optional
        Microwave power in dBm
    
    Returns
    -------
    dict
        State after the sequence
    """
    with self.lock:
        # If mw_frequency is not provided, use resonance frequency
        if mw_frequency is None:
            zfs = self.config["zero_field_splitting"]
            gamma = self.config["gyromagnetic_ratio"]
            b_z = self._magnetic_field[2]
            mw_frequency = zfs + gamma * b_z
        
        # Define the pulse sequence
        sequence = [
            # Initial pi/2 pulse
            ("mw", pi_half_pulse_duration, {"frequency": mw_frequency, "power": mw_power}),
            
            # Free evolution
            ("wait", free_evolution_time, {}),
            
            # Final pi/2 pulse
            ("mw", pi_half_pulse_duration, {"frequency": mw_frequency, "power": mw_power, "measure": True, "fluorescence": True})
        ]
        
        # Execute sequence
        return self.evolve_pulse_sequence(sequence)
```

```python
def evolve_with_spin_echo(self, free_evolution_time, pi_pulse_duration, 
                         pi_half_pulse_duration, mw_frequency=None, mw_power=0.0):
    """
    Perform a Hahn echo experiment.
    
    Parameters
    ----------
    free_evolution_time : float
        Total free evolution time in seconds (half before, half after pi pulse)
    pi_pulse_duration : float
        Duration of pi pulse in seconds
    pi_half_pulse_duration : float
        Duration of pi/2 pulses in seconds
    mw_frequency : float, optional
        Microwave frequency in Hz. If None, use resonance frequency.
    mw_power : float, optional
        Microwave power in dBm
    
    Returns
    -------
    dict
        State after the sequence
    """
    with self.lock:
        # If mw_frequency is not provided, use resonance frequency
        if mw_frequency is None:
            zfs = self.config["zero_field_splitting"]
            gamma = self.config["gyromagnetic_ratio"]
            b_z = self._magnetic_field[2]
            mw_frequency = zfs + gamma * b_z
        
        # Define the pulse sequence
        sequence = [
            # Initial pi/2 pulse
            ("mw", pi_half_pulse_duration, {"frequency": mw_frequency, "power": mw_power}),
            
            # First free evolution
            ("wait", free_evolution_time/2, {}),
            
            # Pi pulse
            ("mw", pi_pulse_duration, {"frequency": mw_frequency, "power": mw_power}),
            
            # Second free evolution
            ("wait", free_evolution_time/2, {}),
            
            # Final pi/2 pulse
            ("mw", pi_half_pulse_duration, {"frequency": mw_frequency, "power": mw_power, "measure": True, "fluorescence": True})
        ]
        
        # Execute sequence
        return self.evolve_pulse_sequence(sequence)
```

### 4. Adapt Existing State Population and Measurement Methods

Update methods for state population and fluorescence to use SimOS's expectation value calculations:

```python
def get_populations(self):
    """
    Get the populations of different spin states.
    
    Returns
    -------
    dict
        Dictionary with keys 'ms0', 'ms_plus', 'ms_minus' and their probabilities
    """
    with self.lock:
        from simos.core import expect
        
        # Get population of ms=0 state
        p0 = expect(self._nv_system.Sp[0], self._state)
        
        # Get population of ms=+1 state
        pp = expect(self._nv_system.Sp[1], self._state)
        
        # Get population of ms=-1 state
        pm = expect(self._nv_system.Sp[-1], self._state)
        
        return {
            'ms0': float(np.real(p0)),
            'ms+1': float(np.real(pp)),
            'ms-1': float(np.real(pm))
        }
```

```python
def get_fluorescence(self):
    """
    Get the fluorescence signal for the current state.
    
    Returns
    -------
    float
        Fluorescence signal in counts per second
    """
    with self.lock:
        if self.config["optics"]:
            # Get population in different electronic states
            from simos.core import expect
            
            # Ground state population
            p_gs = expect(self._nv_system.GSid, self._state)
            
            # Excited state population
            p_es = expect(self._nv_system.ESid, self._state)
            
            # Shelving state population
            p_ss = expect(self._nv_system.SSid, self._state)
            
            # ms=0 population in ground state
            p0_gs = expect(self._nv_system.GSid * self._nv_system.Sp[0], self._state)
            
            # Fluorescence model
            # - Ground state emits no photons
            # - Excited state emits photons at rate dependent on laser power and state
            # - Shelving state emits no photons
            
            # Get optical rates
            rates = decay_rates(self.config["temperature"])
            emission_rate = rates["optical_emission"]
            
            # Simplified fluorescence model
            base_rate = 1e5  # photons/s at saturation
            contrast = 0.3   # fluorescence contrast between ms=0 and ms=±1
            
            # Calculate fluorescence
            if p_es > 0:
                # If in excited state, emit with state-dependent rate
                p0_es = expect(self._nv_system.ESid * self._nv_system.Sp[0], self._state) / p_es
                return base_rate * (1.0 - contrast * (1.0 - float(np.real(p0_es))))
            else:
                # If not in excited state, use ground state population
                p0_gs = p0_gs / max(p_gs, 1e-10)  # Avoid division by zero
                return base_rate * (1.0 - contrast * (1.0 - float(np.real(p0_gs))))
        else:
            # Simplified model for spin-only system
            ms0_pop = self.get_populations()['ms0']
            base_fluorescence = 1e5  # counts/s
            contrast = 0.3  # 30% contrast
            
            return base_fluorescence * (1.0 - contrast * (1.0 - ms0_pop))
```

## Technical Risks

1. **Performance Degradation**: Quantum evolution with density matrices is much more computationally intensive than the simplified model.

2. **Numerical Stability**: Time-dependent Hamiltonians and complex decoherence processes can lead to numerical instabilities in long simulations.

3. **Memory Usage**: Full density matrix representation requires significantly more memory than population vectors.

4. **Backend Dependencies**: Master equation solvers may require specific numerical backends with additional dependencies.

## Testing Strategy

1. **Unit Tests**:
   - Test evolution with different magnetic field configurations
   - Test evolution with microwave driving at different detunings
   - Test evolution with optical processes

2. **Validation Tests**:
   - Test coherent Rabi oscillations against analytical solutions
   - Test T1 relaxation dynamics
   - Test T2 decoherence in spin echo experiments
   - Test optical pumping dynamics

3. **Performance Tests**:
   - Measure simulation performance for different system configurations
   - Test memory usage for complex simulations

## Acceptance Criteria

1. Rabi oscillations match theoretical frequency within 1% error
2. Relaxation times (T1, T2, T2*) match configured values within 5% error
3. Ramsey and spin echo experiments demonstrate correct decoherence behavior
4. Optical processes correctly demonstrate spin-dependent fluorescence
5. Performance degradation is documented and minimized where possible

## Estimation

- Core evolution engine: 2 days
- Pulse sequence implementation: 1 day
- Specialized experiments (Ramsey, spin echo): 1 day
- Testing and validation: 1.5 days
- Documentation: 0.5 day
- Total: 6 days of effort