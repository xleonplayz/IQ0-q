# Technical Story TS-003: Experimental Interface Implementation

## Overview

Implement high-level experimental interfaces for the SimOS-enhanced NV center simulator to facilitate common quantum sensing and quantum information experiments on NV centers.

## Description

The core simulator with SimOS integration provides the foundation for quantum-accurate simulations, but we need to expose user-friendly experimental interfaces that match common laboratory protocols. This story focuses on implementing experiment-specific methods for ODMR, Rabi oscillations, T1/T2 measurements, and other common NV center experiments, while maintaining backward compatibility with the existing API.

## Goals

- Maintain backward compatibility with existing experiment methods
- Implement high-level interfaces for common NV center experiments
- Provide comprehensive experiment result analysis and processing
- Enable more complex experiments like dynamical decoupling sequences

## Implementation Plan

### 1. ODMR Experiment Implementation

Keep the `simulate_odmr` method signature but update the implementation to use SimOS:

```python
def simulate_odmr(self, f_min, f_max, n_points, mw_power=-10.0):
    """
    Run an ODMR (Optically Detected Magnetic Resonance) experiment.
    
    Parameters
    ----------
    f_min : float
        Start frequency in Hz
    f_max : float
        End frequency in Hz
    n_points : int
        Number of frequency points
    mw_power : float, optional
        Microwave power in dBm
        
    Returns
    -------
    SimulationResult
        Object containing frequencies, signal, and analysis
    """
    with self.lock:
        # Generate frequency points
        frequencies = np.linspace(f_min, f_max, n_points)
        signal = np.zeros(n_points)
        
        # Save original state
        original_state = self._state.copy()
        original_mw = (self.mw_frequency, self.mw_power, self.mw_on)
        original_laser = (self.laser_power, self.laser_on)
        
        try:
            # Scan frequencies
            for i, f in enumerate(frequencies):
                # Initialize (optical pumping to ms=0)
                self.reset_state()
                
                # Apply microwave and evolve
                self.apply_microwave(f, mw_power, True)
                self.simulate(1e-6)  # 1 µs
                
                # Read out (fluorescence measurement)
                self.apply_microwave(f, 0.0, False)
                self.apply_laser(1.0, True)
                signal[i] = self.get_fluorescence()
            
            # Find resonance (minimum in fluorescence)
            min_idx = np.argmin(signal)
            center_frequency = frequencies[min_idx]
            
            # Calculate contrast
            max_signal = np.max(signal)
            min_signal = np.min(signal)
            contrast = (max_signal - min_signal) / max_signal if max_signal > 0 else 0
            
            # Calculate linewidth (FWHM)
            half_contrast = (max_signal + min_signal) / 2
            above_half = signal > half_contrast
            rising_edges = np.where(np.diff(above_half.astype(int)) == 1)[0]
            falling_edges = np.where(np.diff(above_half.astype(int)) == -1)[0]
            
            if len(rising_edges) > 0 and len(falling_edges) > 0:
                # Find closest pair of rising and falling edges around resonance
                linewidth = frequencies[falling_edges[0]] - frequencies[rising_edges[0]]
            else:
                linewidth = None
            
            # Return results
            return SimulationResult(
                type="ODMR",
                frequencies=frequencies,
                signal=signal,
                center_frequency=center_frequency,
                contrast=contrast,
                linewidth=linewidth
            )
        
        finally:
            # Restore original state
            self._state = original_state
            self.apply_microwave(original_mw[0], original_mw[1], original_mw[2])
            self.apply_laser(original_laser[0], original_laser[1])
```

### 2. Rabi Oscillation Experiment

Update the Rabi oscillation implementation to use SimOS:

```python
def simulate_rabi(self, t_max, n_points, mw_frequency=None, mw_power=-10.0):
    """
    Run a Rabi oscillation experiment.
    
    Parameters
    ----------
    t_max : float
        Maximum time in seconds
    n_points : int
        Number of time points
    mw_frequency : float, optional
        Microwave frequency in Hz. If None, use resonance frequency.
    mw_power : float, optional
        Microwave power in dBm
        
    Returns
    -------
    SimulationResult
        Object containing times, populations, and oscillation parameters
    """
    with self.lock:
        # If mw_frequency is not provided, use resonance frequency
        if mw_frequency is None:
            zfs = self.config["zero_field_splitting"]
            gamma = self.config["gyromagnetic_ratio"]
            b_z = self._magnetic_field[2]
            mw_frequency = zfs + gamma * b_z
        
        # Generate time points
        times = np.linspace(0, t_max, n_points)
        populations = np.zeros((n_points, 3))  # ms0, ms+1, ms-1
        
        # Save original state
        original_state = self._state.copy()
        original_mw = (self.mw_frequency, self.mw_power, self.mw_on)
        original_laser = (self.laser_power, self.laser_on)
        
        try:
            # For each time point
            for i, t in enumerate(times):
                # Initialize
                self.reset_state()
                
                # Apply microwave and evolve
                self.evolve_with_rabi(t, mw_frequency, mw_power)
                
                # Get populations
                pops = self.get_populations()
                populations[i, 0] = pops["ms0"]
                populations[i, 1] = pops["ms+1"]
                populations[i, 2] = pops["ms-1"]
            
            # Analyze results: fit damped sine wave
            ms0_pop = populations[:, 0]
            
            # Use FFT to estimate Rabi frequency
            if len(ms0_pop) > 4:
                # Remove DC
                ms0_centered = ms0_pop - np.mean(ms0_pop)
                
                # Compute FFT
                fft = np.abs(np.fft.rfft(ms0_centered))
                freqs = np.fft.rfftfreq(len(ms0_centered), d=t_max/(n_points-1))
                
                # Find peak
                peak_idx = np.argmax(fft[1:]) + 1  # Skip DC
                rabi_frequency = freqs[peak_idx]
                
                # Fit damped sine (more accurate)
                try:
                    from scipy.optimize import curve_fit
                    
                    def damped_sine(t, A, f, phi, tau, C):
                        return A * np.exp(-t/tau) * np.cos(2*np.pi*f*t + phi) + C
                    
                    # Initial guess
                    p0 = [
                        (np.max(ms0_pop) - np.min(ms0_pop))/2,  # Amplitude
                        rabi_frequency,                          # Frequency
                        0,                                       # Phase
                        t_max*2,                                 # Decay time
                        np.mean(ms0_pop)                         # Offset
                    ]
                    
                    # Fit
                    popt, _ = curve_fit(damped_sine, times, ms0_pop, p0=p0, bounds=(
                        [0, 0, -np.pi, 0, 0],
                        [1, np.inf, np.pi, np.inf, 1]
                    ))
                    
                    rabi_frequency = popt[1]  # Use fitted frequency
                    rabi_decay_time = popt[3]  # T2* from Rabi decay
                else:
                    rabi_decay_time = None
            else:
                rabi_frequency = None
                rabi_decay_time = None
            
            # Calculate amplitude
            amplitude = np.max(ms0_pop) - np.min(ms0_pop)
            
            # Return results
            return SimulationResult(
                type="Rabi",
                times=times,
                populations=populations,
                rabi_frequency=rabi_frequency,
                rabi_decay_time=rabi_decay_time,
                amplitude=amplitude
            )
        
        finally:
            # Restore original state
            self._state = original_state
            self.apply_microwave(original_mw[0], original_mw[1], original_mw[2])
            self.apply_laser(original_laser[0], original_laser[1])
```

### 3. Implement T1 Relaxation Measurement

Add new method for T1 measurement:

```python
def simulate_t1(self, t_max, n_points):
    """
    Run a T1 relaxation measurement.
    
    Parameters
    ----------
    t_max : float
        Maximum time in seconds
    n_points : int
        Number of time points
        
    Returns
    -------
    SimulationResult
        Object containing times, populations, and relaxation parameters
    """
    with self.lock:
        # Generate time points
        times = np.linspace(0, t_max, n_points)
        populations = np.zeros((n_points, 3))  # ms0, ms+1, ms-1
        
        # Save original state
        original_state = self._state.copy()
        original_mw = (self.mw_frequency, self.mw_power, self.mw_on)
        original_laser = (self.laser_power, self.laser_on)
        
        try:
            # For each time point
            for i, t in enumerate(times):
                # Initialize to ms=-1 state
                self.reset_state()
                self.initialize_state(ms="-1")
                
                # Wait for time t (free evolution)
                self.apply_microwave(0.0, 0.0, False)
                self.apply_laser(0.0, False)
                self.simulate(t)
                
                # Get populations
                pops = self.get_populations()
                populations[i, 0] = pops["ms0"]
                populations[i, 1] = pops["ms+1"]
                populations[i, 2] = pops["ms-1"]
            
            # Fit exponential decay
            ms0_pop = populations[:, 0]
            
            try:
                from scipy.optimize import curve_fit
                
                def exponential(t, A, tau, C):
                    return A * (1 - np.exp(-t/tau)) + C
                
                # Initial guess
                p0 = [
                    np.max(ms0_pop) - np.min(ms0_pop),  # Amplitude
                    self.config["t1"],                  # Time constant (using config value as guess)
                    np.min(ms0_pop)                     # Offset
                ]
                
                # Fit
                popt, _ = curve_fit(exponential, times, ms0_pop, p0=p0)
                
                t1_measured = popt[1]  # Fitted T1
                
            except:
                t1_measured = None
                
            # Return results
            return SimulationResult(
                type="T1",
                times=times,
                populations=populations,
                t1=t1_measured
            )
        
        finally:
            # Restore original state
            self._state = original_state
            self.apply_microwave(original_mw[0], original_mw[1], original_mw[2])
            self.apply_laser(original_laser[0], original_laser[1])
```

### 4. Implement T2 Measurement (Hahn Echo)

Add method for T2 measurement with Hahn echo:

```python
def simulate_t2_echo(self, t_max, n_points, mw_frequency=None, mw_power=0.0):
    """
    Run a T2 measurement using Hahn echo.
    
    Parameters
    ----------
    t_max : float
        Maximum free evolution time in seconds
    n_points : int
        Number of time points
    mw_frequency : float, optional
        Microwave frequency in Hz. If None, use resonance frequency.
    mw_power : float, optional
        Microwave power in dBm
        
    Returns
    -------
    SimulationResult
        Object containing times, signals, and coherence parameters
    """
    with self.lock:
        # If mw_frequency is not provided, use resonance frequency
        if mw_frequency is None:
            zfs = self.config["zero_field_splitting"]
            gamma = self.config["gyromagnetic_ratio"]
            b_z = self._magnetic_field[2]
            mw_frequency = zfs + gamma * b_z
        
        # Calculate pulse durations from power
        rabi_freq = 10e6 * 10**((mw_power + 20) / 20)  # Hz
        pi_pulse_duration = 1.0 / (2.0 * rabi_freq)    # seconds
        pi_half_pulse_duration = pi_pulse_duration / 2.0
        
        # Generate time points (free evolution time)
        times = np.linspace(0, t_max, n_points)
        signal = np.zeros(n_points)
        
        # Save original state
        original_state = self._state.copy()
        
        try:
            # For each time point
            for i, tau in enumerate(times):
                # Reset for each measurement
                self.reset_state()
                
                # Execute Hahn echo sequence and measure
                result = self.evolve_with_spin_echo(
                    free_evolution_time=tau,
                    pi_pulse_duration=pi_pulse_duration,
                    pi_half_pulse_duration=pi_half_pulse_duration,
                    mw_frequency=mw_frequency,
                    mw_power=mw_power
                )
                
                # Get fluorescence measurement after final pulse
                if "fluorescence_after_pulse_4" in result:
                    signal[i] = result["fluorescence_after_pulse_4"]
                else:
                    # Measure fluorescence if not already done
                    self.apply_laser(1.0, True)
                    signal[i] = self.get_fluorescence()
            
            # Fit decay curve
            try:
                from scipy.optimize import curve_fit
                
                def exponential_decay(t, A, tau, C):
                    return A * np.exp(-((t/tau)**n)) + C
                
                # Normalize signal
                sig_norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal))
                
                # Try different decay models
                results = []
                for n in [1, 2, 3]:  # Try different exponents
                    try:
                        # Initial guess
                        p0 = [
                            1.0,                    # Amplitude
                            self.config["t2"],      # Time constant (using config value as guess)
                            0.0                     # Offset
                        ]
                        
                        # Fit with fixed exponent
                        popt, pcov = curve_fit(lambda t, A, tau, C: exponential_decay(t, A, tau, C), 
                                              times, sig_norm, p0=p0)
                        
                        # Calculate fit error
                        fit = exponential_decay(times, *popt)
                        mse = np.mean((sig_norm - fit)**2)
                        
                        results.append((n, popt[1], mse))
                    except:
                        continue
                
                # Choose best fit based on MSE
                if results:
                    best_fit = min(results, key=lambda x: x[2])
                    decay_exponent = best_fit[0]
                    t2_measured = best_fit[1]
                else:
                    decay_exponent = None
                    t2_measured = None
                    
            except:
                decay_exponent = None
                t2_measured = None
                
            # Return results
            return SimulationResult(
                type="T2_Echo",
                times=times,
                signal=signal,
                t2=t2_measured,
                decay_exponent=decay_exponent
            )
        
        finally:
            # Restore original state
            self._state = original_state
```

### 5. Implement Advanced Dynamical Decoupling Method

Add method for advanced dynamical decoupling sequences like XY8:

```python
def simulate_dynamical_decoupling(self, sequence_type, t_max, n_points, n_pulses, 
                                mw_frequency=None, mw_power=0.0):
    """
    Run a dynamical decoupling sequence.
    
    Parameters
    ----------
    sequence_type : str
        Type of sequence, one of: "CPMG", "XY4", "XY8", "XY16"
    t_max : float
        Maximum free evolution time in seconds
    n_points : int
        Number of time points
    n_pulses : int
        Number of π pulses in the sequence
    mw_frequency : float, optional
        Microwave frequency in Hz. If None, use resonance frequency.
    mw_power : float, optional
        Microwave power in dBm
        
    Returns
    -------
    SimulationResult
        Object containing times, signals, and coherence parameters
    """
    with self.lock:
        # If mw_frequency is not provided, use resonance frequency
        if mw_frequency is None:
            zfs = self.config["zero_field_splitting"]
            gamma = self.config["gyromagnetic_ratio"]
            b_z = self._magnetic_field[2]
            mw_frequency = zfs + gamma * b_z
        
        # Calculate pulse durations from power
        rabi_freq = 10e6 * 10**((mw_power + 20) / 20)  # Hz
        pi_pulse_duration = 1.0 / (2.0 * rabi_freq)    # seconds
        pi_half_pulse_duration = pi_pulse_duration / 2.0
        
        # Generate time points (total sequence time)
        times = np.linspace(0, t_max, n_points)
        signal = np.zeros(n_points)
        
        # Save original state
        original_state = self._state.copy()
        
        try:
            # For each time point
            for i, tau in enumerate(times):
                # Reset for each measurement
                self.reset_state()
                
                # Calculate time between pulses
                tau_pulse = tau / n_pulses
                
                # Build the specific sequence
                sequence = []
                
                # Initial π/2 pulse
                sequence.append(("mw", pi_half_pulse_duration, 
                              {"frequency": mw_frequency, "power": mw_power, "phase": 0}))
                
                # Add dynamical decoupling pulses
                if sequence_type == "CPMG":
                    # CPMG: all pulses along Y
                    for j in range(n_pulses):
                        sequence.append(("wait", tau_pulse/2, {}))
                        sequence.append(("mw", pi_pulse_duration, 
                                      {"frequency": mw_frequency, "power": mw_power, "phase": 90}))
                        sequence.append(("wait", tau_pulse/2, {}))
                        
                elif sequence_type == "XY4":
                    # XY4: X-Y-X-Y
                    phases = [0, 90, 0, 90] * (n_pulses // 4)
                    for j in range(min(n_pulses, len(phases))):
                        sequence.append(("wait", tau_pulse/2, {}))
                        sequence.append(("mw", pi_pulse_duration, 
                                      {"frequency": mw_frequency, "power": mw_power, "phase": phases[j]}))
                        sequence.append(("wait", tau_pulse/2, {}))
                        
                elif sequence_type == "XY8":
                    # XY8: X-Y-X-Y-Y-X-Y-X
                    phases = [0, 90, 0, 90, 90, 0, 90, 0] * (n_pulses // 8)
                    for j in range(min(n_pulses, len(phases))):
                        sequence.append(("wait", tau_pulse/2, {}))
                        sequence.append(("mw", pi_pulse_duration, 
                                      {"frequency": mw_frequency, "power": mw_power, "phase": phases[j]}))
                        sequence.append(("wait", tau_pulse/2, {}))
                        
                elif sequence_type == "XY16":
                    # XY16: X-Y-X-Y-Y-X-Y-X-X-Y-X-Y-Y-X-Y-X
                    phases = [0, 90, 0, 90, 90, 0, 90, 0, 0, 90, 0, 90, 90, 0, 90, 0] * (n_pulses // 16)
                    for j in range(min(n_pulses, len(phases))):
                        sequence.append(("wait", tau_pulse/2, {}))
                        sequence.append(("mw", pi_pulse_duration, 
                                      {"frequency": mw_frequency, "power": mw_power, "phase": phases[j]}))
                        sequence.append(("wait", tau_pulse/2, {}))
                
                # Final π/2 pulse
                sequence.append(("mw", pi_half_pulse_duration, 
                              {"frequency": mw_frequency, "power": mw_power, "phase": 0}))
                
                # Measure
                sequence.append(("laser", 1e-6, {"power": 1.0, "measure": True, "fluorescence": True}))
                
                # Execute sequence
                result = self.evolve_pulse_sequence(sequence)
                
                # Get fluorescence
                for key in result:
                    if "fluorescence" in key:
                        signal[i] = result[key]
                        break
            
            # Fit decay curve - similar to T2 analysis
            # (Code omitted for brevity, similar to T2 echo analysis)
                
            # Return results
            return SimulationResult(
                type=f"DynamicalDecoupling_{sequence_type}",
                times=times,
                signal=signal,
                n_pulses=n_pulses,
                sequence_type=sequence_type,
                # Add fitted parameters here
            )
        
        finally:
            # Restore original state
            self._state = original_state
```

### 6. Enhance SimulationResult Class

Update the `SimulationResult` class to provide richer analysis capabilities:

```python
class SimulationResult:
    """Container for simulation results with analysis capabilities."""
    
    def __init__(self, type, **kwargs):
        """
        Initialize simulation result.
        
        Parameters
        ----------
        type : str
            Type of simulation (e.g., "ODMR", "Rabi", "T1", "T2")
        **kwargs
            Additional data and analysis results
        """
        self.type = type
        
        # Store all provided data
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __str__(self):
        """String representation of the result."""
        if self.type == "ODMR":
            return f"ODMR Result: Center frequency = {self.center_frequency/1e6:.2f} MHz, Contrast = {self.contrast:.2%}"
        elif self.type == "Rabi":
            if hasattr(self, "rabi_frequency") and self.rabi_frequency is not None:
                return f"Rabi Result: Frequency = {self.rabi_frequency/1e6:.2f} MHz, Amplitude = {self.amplitude:.2f}"
            else:
                return f"Rabi Result: Amplitude = {self.amplitude:.2f} (no frequency detected)"
        elif self.type == "T1":
            if hasattr(self, "t1") and self.t1 is not None:
                return f"T1 Result: Relaxation time = {self.t1*1e3:.2f} ms"
            else:
                return "T1 Result: Could not determine relaxation time"
        elif self.type == "T2_Echo":
            if hasattr(self, "t2") and self.t2 is not None:
                return f"T2 Echo Result: Coherence time = {self.t2*1e6:.2f} µs"
            else:
                return "T2 Echo Result: Could not determine coherence time"
        elif "DynamicalDecoupling" in self.type:
            seq_type = self.type.split("_")[1]
            if hasattr(self, "t2") and self.t2 is not None:
                return f"{seq_type} Result: Coherence time = {self.t2*1e6:.2f} µs with {self.n_pulses} pulses"
            else:
                return f"{seq_type} Result: Could not determine coherence time"
        else:
            return f"{self.type} Result"
    
    def plot(self, ax=None, **kwargs):
        """
        Plot the simulation results.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes to plot on. If None, create a new figure.
        **kwargs
            Additional parameters for plotting.
            
        Returns
        -------
        matplotlib.axes.Axes
            The axes containing the plot.
        """
        import matplotlib.pyplot as plt
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        
        if self.type == "ODMR":
            # Plot ODMR spectrum
            ax.plot((self.frequencies - self.center_frequency) / 1e6, self.signal)
            ax.set_xlabel("Frequency Detuning (MHz)")
            ax.set_ylabel("Fluorescence (counts/s)")
            ax.set_title("ODMR Spectrum")
            
            # Add vertical line at resonance
            ax.axvline(0, color='r', linestyle='--')
            
            # Add text with results
            text = f"Center: {self.center_frequency/1e6:.3f} MHz\nContrast: {self.contrast:.1%}"
            if hasattr(self, "linewidth") and self.linewidth is not None:
                text += f"\nLinewidth: {self.linewidth/1e6:.2f} MHz"
            ax.text(0.05, 0.95, text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
            
        elif self.type == "Rabi":
            # Plot Rabi oscillations
            ax.plot(self.times * 1e6, self.populations[:, 0])
            ax.set_xlabel("Time (µs)")
            ax.set_ylabel("ms=0 Population")
            ax.set_title("Rabi Oscillations")
            
            # Add fitted curve if available
            if hasattr(self, "rabi_frequency") and self.rabi_frequency is not None:
                from scipy.optimize import curve_fit
                
                def damped_sine(t, A, f, phi, tau, C):
                    return A * np.exp(-t/tau) * np.cos(2*np.pi*f*t + phi) + C
                
                t_fit = np.linspace(0, self.times[-1], 1000)
                if hasattr(self, "rabi_decay_time") and self.rabi_decay_time is not None:
                    # Use fitted parameters
                    A = self.amplitude / 2
                    f = self.rabi_frequency
                    tau = self.rabi_decay_time
                    C = 0.5
                    
                    y_fit = damped_sine(t_fit, A, f, 0, tau, C)
                    ax.plot(t_fit * 1e6, y_fit, 'r--')
                    
                    # Add text with results
                    text = f"Frequency: {f/1e6:.2f} MHz\nT2*: {tau*1e6:.2f} µs"
                    ax.text(0.05, 0.95, text, transform=ax.transAxes, 
                           verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
                    
        elif self.type == "T1":
            # Plot T1 relaxation
            ax.plot(self.times * 1e3, self.populations[:, 0])
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("ms=0 Population")
            ax.set_title("T1 Relaxation")
            
            # Add fitted curve if available
            if hasattr(self, "t1") and self.t1 is not None:
                def exponential(t, A, tau, C):
                    return A * (1 - np.exp(-t/tau)) + C
                
                t_fit = np.linspace(0, self.times[-1], 1000)
                A = self.populations[:, 0].max() - self.populations[:, 0].min()
                C = self.populations[:, 0].min()
                
                y_fit = exponential(t_fit, A, self.t1, C)
                ax.plot(t_fit * 1e3, y_fit, 'r--')
                
                # Add text with results
                text = f"T1: {self.t1*1e3:.2f} ms"
                ax.text(0.05, 0.95, text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
                
        elif self.type == "T2_Echo" or "DynamicalDecoupling" in self.type:
            # Plot T2 decay
            ax.plot(self.times * 1e6, self.signal)
            ax.set_xlabel("Time (µs)")
            ax.set_ylabel("Fluorescence (counts/s)")
            
            if "DynamicalDecoupling" in self.type:
                seq_type = self.type.split("_")[1]
                ax.set_title(f"{seq_type} Dynamical Decoupling with {self.n_pulses} pulses")
            else:
                ax.set_title("Hahn Echo T2 Measurement")
            
            # Add fitted curve if available
            if hasattr(self, "t2") and self.t2 is not None:
                def exponential_decay(t, A, tau, n, C):
                    return A * np.exp(-((t/tau)**n)) + C
                
                t_fit = np.linspace(0, self.times[-1], 1000)
                
                # Normalize for plotting
                signal_norm = (self.signal - self.signal.min()) / (self.signal.max() - self.signal.min())
                A = 1.0
                C = 0.0
                n = self.decay_exponent if hasattr(self, "decay_exponent") else 2.0
                
                y_fit = exponential_decay(t_fit, A, self.t2, n, C)
                
                # Scale back to original range for plotting
                y_fit = y_fit * (self.signal.max() - self.signal.min()) + self.signal.min()
                ax.plot(t_fit * 1e6, y_fit, 'r--')
                
                # Add text with results
                text = f"T2: {self.t2*1e6:.2f} µs\nExponent: {n:.1f}"
                ax.text(0.05, 0.95, text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
                
        return ax
```

## Technical Risks

1. **Backward Compatibility**: Ensuring the new implementation maintains compatibility with existing code that uses the simulator.

2. **Numerical Robustness**: Complex pulse sequences and fitting routines may be numerically unstable in some edge cases.

3. **Performance Impact**: Advanced experiments with many pulses or long coherence times may be computationally expensive.

4. **Dependency Management**: The enhanced features may introduce additional dependencies (e.g., SciPy for curve fitting).

## Testing Strategy

1. **Functional Tests**:
   - Test each experiment type with standard parameters
   - Verify result structure and analysis
   - Test plotting capabilities

2. **Compatibility Tests**:
   - Test with existing code that uses the simulator
   - Verify that results are compatible with the previous implementation

3. **Robustness Tests**:
   - Test with extreme parameter values
   - Test with different numerical backends
   - Test with and without various options enabled

4. **Validation Tests**:
   - Compare experiment results against analytical predictions
   - Verify that simulated T1/T2 matches configured values

## Acceptance Criteria

1. All experimental interfaces maintain backward compatibility with previous API
2. ODMR experiments correctly identify resonance frequencies
3. Rabi oscillation frequency scales properly with microwave power
4. T1 and T2 measurements match configured coherence times within 5% error
5. Dynamical decoupling sequences show expected improvement in coherence times
6. All experiments produce valid results across a reasonable parameter range
7. Result analysis and plotting capabilities work as expected

## Estimation

- ODMR and Rabi implementation: 1 day
- T1 and T2 implementation: 1 day
- Dynamical decoupling implementation: 1 day
- Enhanced result analysis and plotting: 1 day
- Testing and validation: 1 day
- Documentation: 0.5 day
- Total: 5.5 days of effort