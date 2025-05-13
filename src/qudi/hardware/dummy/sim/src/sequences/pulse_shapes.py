# -*- coding: utf-8 -*-

"""
Pulse shape utilities for dynamical decoupling sequences.

This module provides functions for generating and manipulating pulse shapes
for quantum control of NV centers.
"""

import numpy as np
from enum import Enum, auto
from typing import Callable, List, Dict, Tuple, Any, Union, Optional
import matplotlib.pyplot as plt

class PulseShape(Enum):
    """Types of pulse shapes for DD sequences."""
    SQUARE = auto()     # Square/rectangular pulse
    GAUSSIAN = auto()   # Gaussian pulse
    SINC = auto()       # Sinc pulse
    HERMITE = auto()    # Hermite pulse
    DRAG = auto()       # Derivative Removal by Adiabatic Gate (DRAG) pulse
    OPTIMAL = auto()    # Numerically optimized pulse

def create_pulse_shape(shape_type: Union[str, PulseShape], 
                      time_resolution: int = 100, 
                      parameters: Optional[Dict] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a pulse shape with normalized amplitude.
    
    Parameters:
    -----------
    shape_type : str or PulseShape
        Type of pulse shape
    time_resolution : int
        Number of time points for the pulse
    parameters : dict, optional
        Additional parameters for specific pulse shapes
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Time points and corresponding amplitude values
    """
    # Convert string to enum if needed
    if isinstance(shape_type, str):
        shape_type = shape_type.upper()
        try:
            shape_type = PulseShape[shape_type]
        except KeyError:
            raise ValueError(f"Unknown pulse shape: {shape_type}")
    
    # Initialize parameters
    params = parameters or {}
    t = np.linspace(0, 1, time_resolution)
    
    # Generate pulse shape
    if shape_type == PulseShape.SQUARE:
        # Simple square pulse
        amplitude = np.ones_like(t)
        
    elif shape_type == PulseShape.GAUSSIAN:
        # Gaussian pulse
        sigma = params.get('sigma', 0.25)
        t_c = params.get('center', 0.5)
        amplitude = np.exp(-0.5 * ((t - t_c) / sigma)**2)
        
    elif shape_type == PulseShape.SINC:
        # Sinc pulse
        width = params.get('width', 5.0)
        t_c = params.get('center', 0.5)
        t_scaled = width * (t - t_c)
        amplitude = np.ones_like(t)
        nonzero = np.abs(t_scaled) > 1e-10
        amplitude[nonzero] = np.sin(np.pi * t_scaled[nonzero]) / (np.pi * t_scaled[nonzero])
        
    elif shape_type == PulseShape.HERMITE:
        # Hermite pulse (used for improved spectral properties)
        sigma = params.get('sigma', 0.25)
        t_c = params.get('center', 0.5)
        order = params.get('order', 1)
        
        # Compute Hermite polynomial of order n
        def hermite(x, n):
            if n == 0:
                return np.ones_like(x)
            elif n == 1:
                return 2 * x
            else:
                return 2 * x * hermite(x, n-1) - 2 * (n-1) * hermite(x, n-2)
        
        # Normalized time
        x = (t - t_c) / sigma
        
        # Hermite-Gaussian pulse
        amplitude = hermite(x, order) * np.exp(-0.5 * x**2)
        
        # Normalize
        amplitude = amplitude / np.max(np.abs(amplitude))
        
    elif shape_type == PulseShape.DRAG:
        # DRAG pulse: combination of Gaussian and its derivative
        sigma = params.get('sigma', 0.25)
        t_c = params.get('center', 0.5)
        beta = params.get('beta', 0.5)  # DRAG coefficient
        
        # Gaussian component
        x = (t - t_c) / sigma
        gaussian = np.exp(-0.5 * x**2)
        
        # Derivative component
        derivative = -x * gaussian
        
        # DRAG pulse (complex amplitude)
        amplitude = gaussian + 1j * beta * derivative
        
        # Take real part for now (in actual implementation, we'd use complex control)
        amplitude = np.real(amplitude)
        
        # Normalize
        amplitude = amplitude / np.max(np.abs(amplitude))
        
    elif shape_type == PulseShape.OPTIMAL:
        # Import optimal control package if available
        try:
            import scipy.optimize as opt
        except ImportError:
            logger.warning("SciPy not available. Using windowed pulse instead of optimal control.")
            amplitude = np.ones_like(t)
            window = np.blackman(len(t))
            amplitude = amplitude * window / np.max(window)
            return t, amplitude
        
        # Get optimal control parameters
        target_rotation = params.get('target_rotation', np.pi)  # Default π pulse
        phase = params.get('phase', 0.0)
        max_iterations = params.get('max_iterations', 100)
        fidelity_target = params.get('fidelity_target', 0.99)
        
        # Define cost function for optimization
        def pulse_cost_function(amplitudes):
            # Reshape amplitudes into pulse shape
            pulse_shape = np.reshape(amplitudes, (len(t),))
            
            # Calculate evolution under this pulse
            # Simplified model of quantum evolution
            dt = t[1] - t[0]
            phase_accumulation = np.cumsum(pulse_shape) * dt
            
            # Target is typically a specific rotation angle
            target = target_rotation
            final_rotation = phase_accumulation[-1]
            
            # Primary objective: achieve target rotation
            rotation_error = (final_rotation - target)**2
            
            # Secondary objectives: smoothness and power constraints
            smoothness = np.sum(np.diff(pulse_shape)**2)
            power = np.sum(pulse_shape**2)
            
            # Total cost with weights
            cost = rotation_error + 0.01 * smoothness + 0.001 * power
            return cost
        
        # Initial guess: Gaussian pulse
        sigma = params.get('sigma', 0.25)
        t_c = params.get('center', 0.5)
        initial_guess = np.exp(-0.5 * ((t - t_c) / sigma)**2)
        
        # Run optimization
        result = opt.minimize(
            pulse_cost_function,
            initial_guess,
            method='L-BFGS-B',
            bounds=[(0, 1) for _ in range(len(t))],
            options={'maxiter': max_iterations}
        )
        
        # Extract optimized pulse shape
        amplitude = result.x
        
        # Check if optimization was successful
        if result.success:
            logger.info(f"Optimal control pulse generation successful. Final cost: {result.fun:.6f}")
        else:
            logger.warning(f"Optimal control optimization did not converge: {result.message}. Using result anyway.")
        
        # Add phase modulation if requested
        if phase != 0.0:
            # For a real pulse implementation, we would use complex control
            # Here we approximate by modulating the amplitude
            t_norm = (t - t[0]) / (t[-1] - t[0])
            phase_mod = np.cos(phase * t_norm * 2 * np.pi)
            amplitude = amplitude * phase_mod
        
        # Normalize to peak amplitude of 1
        amplitude = amplitude / np.max(np.abs(amplitude))
        
    else:
        raise ValueError(f"Unsupported pulse shape: {shape_type}")
    
    # Return time and amplitude arrays
    return t, amplitude

def create_composite_pulse(pulse_sequence: List[Dict], 
                          time_resolution: int = 1000) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Create a composite pulse sequence with multiple axes.
    
    Parameters:
    -----------
    pulse_sequence : List[Dict]
        List of pulse dictionaries with axis, angle, duration, shape
    time_resolution : int
        Number of time points for the entire sequence
        
    Returns:
    --------
    Tuple[np.ndarray, Dict[str, np.ndarray]]
        Time points and dict of amplitudes for each control axis
    """
    # Calculate total duration
    total_duration = sum(pulse.get('duration', 0) for pulse in pulse_sequence)
    
    # Create time array
    time_points = np.linspace(0, total_duration, time_resolution)
    
    # Initialize amplitudes for each axis
    control_axes = {'x': np.zeros_like(time_points), 
                   'y': np.zeros_like(time_points), 
                   'z': np.zeros_like(time_points)}
    
    # Current time
    current_time = 0.0
    
    # Generate pulse amplitudes
    for pulse in pulse_sequence:
        # Skip if not a pulse
        if pulse.get('type') != 'pulse':
            current_time += pulse.get('duration', 0)
            continue
            
        # Extract pulse parameters
        axis = pulse['axis']
        if axis.startswith('-'):
            sign = -1.0
            axis = axis[1:]  # Remove negative sign
        else:
            sign = 1.0
            
        angle = pulse['angle']
        duration = pulse['duration']
        shape_type = pulse.get('shape', 'square')
        
        # Get normalized pulse shape
        _, shape_amplitude = create_pulse_shape(shape_type, time_resolution=100)
        
        # Find indices in time array that correspond to this pulse
        pulse_start = current_time
        pulse_end = current_time + duration
        
        pulse_indices = np.where((time_points >= pulse_start) & (time_points <= pulse_end))[0]
        
        if len(pulse_indices) > 0:
            # Calculate relative time within pulse for each index
            relative_times = (time_points[pulse_indices] - pulse_start) / duration
            
            # Interpolate shape amplitudes to match indices
            amplitudes = np.interp(relative_times, np.linspace(0, 1, len(shape_amplitude)), shape_amplitude)
            
            # Scale by angle/duration to get Rabi frequency in rad/s
            rabi_freq = angle / duration
            
            # Apply to correct axis
            if axis in control_axes:
                control_axes[axis][pulse_indices] += sign * rabi_freq * amplitudes
        
        # Update current time
        current_time += duration
    
    return time_points, control_axes

def calculate_spectrum(time_points: np.ndarray, 
                      amplitude: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the frequency spectrum of a pulse.
    
    Parameters:
    -----------
    time_points : np.ndarray
        Time points
    amplitude : np.ndarray
        Amplitude values
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Frequencies and power spectrum
    """
    # Time step
    dt = time_points[1] - time_points[0]
    
    # Calculate FFT
    fft = np.fft.fft(amplitude)
    freqs = np.fft.fftfreq(len(amplitude), dt)
    
    # Calculate power spectrum
    power = np.abs(fft)**2
    
    # Sort by frequency
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    power = power[idx]
    
    return freqs, power

def plot_pulse_shape(time_points: np.ndarray, 
                    amplitude: Union[np.ndarray, Dict[str, np.ndarray]], 
                    include_spectrum: bool = True, 
                    ax = None) -> Any:
    """
    Plot a pulse shape and optionally its spectrum.
    
    Parameters:
    -----------
    time_points : np.ndarray
        Time points
    amplitude : np.ndarray or Dict[str, np.ndarray]
        Amplitude values for each axis
    include_spectrum : bool
        Whether to include frequency spectrum
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns:
    --------
    matplotlib.axes.Axes or Tuple[matplotlib.axes.Axes, matplotlib.axes.Axes]
        The axes used for plotting
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib is required for plotting")
        
    # Create figure if needed
    if ax is None:
        if include_spectrum:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            ax = ax1
        else:
            fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot pulse shape
    if isinstance(amplitude, dict):
        # Multi-axis pulse
        for axis, values in amplitude.items():
            if np.any(values != 0):  # Only plot non-zero amplitudes
                color = {'x': 'r', 'y': 'g', 'z': 'b'}.get(axis, 'k')
                ax.plot(time_points, values, color=color, label=f"{axis}-axis")
    else:
        # Single pulse
        ax.plot(time_points, amplitude, 'b-')
    
    # Set up axis labels
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title('Pulse Shape')
    
    if isinstance(amplitude, dict):
        ax.legend()
    
    # Plot spectrum if requested
    if include_spectrum:
        if isinstance(amplitude, dict):
            # Use first non-zero axis for spectrum
            for axis, values in amplitude.items():
                if np.any(values != 0):
                    plot_amplitude = values
                    break
            else:
                plot_amplitude = np.zeros_like(time_points)
        else:
            plot_amplitude = amplitude
            
        # Calculate spectrum
        freqs, power = calculate_spectrum(time_points, plot_amplitude)
        
        # Only plot positive frequencies
        positive_idx = freqs >= 0
        freqs = freqs[positive_idx]
        power = power[positive_idx]
        
        # Normalize power
        power = power / np.max(power)
        
        # Plot spectrum
        ax2.semilogy(freqs, power, 'r-')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Power (normalized)')
        ax2.set_title('Frequency Spectrum')
        ax2.grid(True)
        
        return ax, ax2
    
    return ax