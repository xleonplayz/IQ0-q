# -*- coding: utf-8 -*-

"""
Sequence analyzer for dynamical decoupling sequences.

This module provides functions for analyzing and visualizing the performance
of dynamical decoupling sequences for NV center quantum control.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import matplotlib.pyplot as plt

from .base_sequence import DynamicalDecouplingSequence, PulseParameters, PulseError
from .standard_sequences import calculate_sequence_filter_function


def analyze_sequence_fidelity(sequence: DynamicalDecouplingSequence,
                             noise_spectral_density: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                             noise_parameters: Optional[Dict] = None,
                             omega_min: float = 1e2,
                             omega_max: float = 1e8,
                             n_points: int = 1000) -> Dict[str, Any]:
    """
    Analyze the fidelity of a dynamical decoupling sequence under different noise conditions.
    
    Parameters:
    -----------
    sequence : DynamicalDecouplingSequence
        The dynamical decoupling sequence to analyze
    noise_spectral_density : Callable, optional
        Function that takes frequency array and returns noise spectral density
    noise_parameters : Dict, optional
        Parameters for the noise model
    omega_min : float
        Minimum angular frequency for analysis
    omega_max : float
        Maximum angular frequency for analysis
    n_points : int
        Number of frequency points
        
    Returns:
    --------
    Dict[str, Any]
        Analysis results including coherence decay, filter functions, etc.
    """
    # Create frequency array
    omega = np.logspace(np.log10(omega_min), np.log10(omega_max), n_points)
    
    # Calculate filter function
    filter_function = calculate_sequence_filter_function(sequence, omega)
    
    # Default noise model if none provided
    if noise_spectral_density is None:
        if noise_parameters is None:
            noise_parameters = {
                'type': '1/f',
                'amplitude': 1.0,
                'cutoff': 1e3
            }
            
        # Define noise spectral density function
        def default_noise_spectral_density(omega):
            noise_type = noise_parameters.get('type', '1/f')
            amplitude = noise_parameters.get('amplitude', 1.0)
            cutoff = noise_parameters.get('cutoff', 1e3)
            
            if noise_type == '1/f':
                # 1/f noise (pink noise)
                S = amplitude / np.maximum(omega, 1e-10)
            elif noise_type == 'ohmic':
                # Ohmic noise (linear in frequency)
                S = amplitude * omega * np.exp(-omega / cutoff)
            elif noise_type == 'lorentzian':
                # Lorentzian noise (describes fluctuator)
                gamma = noise_parameters.get('gamma', 1e5)
                S = amplitude * gamma / (gamma**2 + omega**2)
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")
                
            return S
        
        noise_spectral_density = default_noise_spectral_density
    
    # Calculate noise spectral density
    S_omega = noise_spectral_density(omega)
    
    # Calculate decay integral
    integrand = filter_function * S_omega
    
    # Trapezoidal integration in log space
    log_omega = np.log(omega)
    d_log_omega = np.diff(log_omega)
    integrand_avg = 0.5 * (integrand[:-1] + integrand[1:])
    decay_exponent = np.sum(integrand_avg * d_log_omega * omega[:-1])
    
    # Calculate coherence time
    T2_sequence = 1.0 / decay_exponent if decay_exponent > 0 else float('inf')
    
    # Calculate coherence vs time
    times = np.linspace(0, sequence.total_duration * 5, 100)
    coherence = np.exp(-times / T2_sequence) if decay_exponent > 0 else np.ones_like(times)
    
    # Return results
    return {
        'sequence': sequence,
        'filter_function': {
            'omega': omega,
            'F': filter_function
        },
        'noise_spectrum': {
            'omega': omega,
            'S': S_omega
        },
        'coherence': {
            'times': times,
            'values': coherence,
            'T2': T2_sequence
        },
        'decay_exponent': decay_exponent
    }

def calculate_filter_function(sequence: DynamicalDecouplingSequence,
                            omega_min: float = 1e2,
                            omega_max: float = 1e8,
                            n_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate the filter function for a dynamical decoupling sequence.
    
    Parameters:
    -----------
    sequence : DynamicalDecouplingSequence
        The dynamical decoupling sequence
    omega_min : float
        Minimum angular frequency
    omega_max : float
        Maximum angular frequency
    n_points : int
        Number of frequency points
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Frequencies and filter function values
    """
    # Create frequency array
    omega = np.logspace(np.log10(omega_min), np.log10(omega_max), n_points)
    
    # Calculate filter function
    F = calculate_sequence_filter_function(sequence, omega)
    
    return omega, F

def plot_filter_function(sequences: Union[DynamicalDecouplingSequence, List[DynamicalDecouplingSequence]],
                         omega_min: float = 1e2,
                         omega_max: float = 1e8,
                         n_points: int = 1000,
                         ax = None) -> Any:
    """
    Plot the filter function for one or more dynamical decoupling sequences.
    
    Parameters:
    -----------
    sequences : DynamicalDecouplingSequence or List[DynamicalDecouplingSequence]
        Sequence(s) to analyze
    omega_min : float
        Minimum angular frequency
    omega_max : float
        Maximum angular frequency
    n_points : int
        Number of frequency points
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns:
    --------
    matplotlib.axes.Axes
        The axes used for plotting
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Convert to list if single sequence
    if not isinstance(sequences, list):
        sequences = [sequences]
    
    # Create frequency array
    omega = np.logspace(np.log10(omega_min), np.log10(omega_max), n_points)
    
    # Plot filter function for each sequence
    for sequence in sequences:
        # Calculate filter function
        F = calculate_sequence_filter_function(sequence, omega)
        
        # Plot
        ax.loglog(omega, F, label=sequence.name)
    
    # Add x-axis reference points
    tau_min = min(1e-9, min(sequence.total_duration for sequence in sequences) / 100)
    reference_times = [1e-9, 1e-6, 1e-3, 1]  # ns, µs, ms, s
    for t in reference_times:
        if 1/t >= omega_min and 1/t <= omega_max:
            ax.axvline(1/t, color='gray', linestyle='--', alpha=0.3)
            ax.text(1/t, ax.get_ylim()[0] * 2, f"{t*1e9:.0f} ns" if t < 1e-6 else
                                              f"{t*1e6:.0f} µs" if t < 1e-3 else
                                              f"{t*1e3:.0f} ms" if t < 1 else
                                              f"{t:.0f} s",
                   rotation=90, va='bottom', ha='right', color='gray')
    
    # Set up axis labels
    ax.set_xlabel('Angular Frequency ω (rad/s)')
    ax.set_ylabel('Filter Function F(ω)')
    ax.set_title('DD Sequence Filter Functions')
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    if len(sequences) > 1:
        ax.legend()
    
    return ax

def error_susceptibility(sequence: DynamicalDecouplingSequence,
                        error_types: List[PulseError],
                        error_magnitudes: List[float],
                        num_simulations: int = 10) -> Dict[str, Any]:
    """
    Analyze the susceptibility of a sequence to different types of pulse errors.
    
    Parameters:
    -----------
    sequence : DynamicalDecouplingSequence
        The dynamical decoupling sequence
    error_types : List[PulseError]
        List of error types to analyze
    error_magnitudes : List[float]
        List of error magnitudes
    num_simulations : int
        Number of simulations for each configuration
        
    Returns:
    --------
    Dict[str, Any]
        Error analysis results
    """
    # Initialize results
    results = {
        'error_types': error_types,
        'error_magnitudes': error_magnitudes,
        'fidelities': {}
    }
    
    # Reference simulation (no errors)
    original_params = sequence.pulse_parameters
    sequence._pulse_params = PulseParameters()  # Reset pulse parameters
    
    # Run reference simulation
    ref_result = sequence.simulate(include_decoherence=False)
    ref_state = ref_result['final_state']
    
    # For each error type
    for error_type in error_types:
        # Initialize fidelity array
        fidelities = np.zeros((len(error_magnitudes), num_simulations))
        
        # For each error magnitude
        for i, magnitude in enumerate(error_magnitudes):
            # For multiple simulations (to average out stochastic effects)
            for j in range(num_simulations):
                # Create pulse parameters with error
                if error_type == PulseError.AMPLITUDE:
                    # Amplitude error
                    params = PulseParameters(
                        error_type=error_type,
                        error_amplitude=magnitude
                    )
                elif error_type == PulseError.PHASE:
                    # Phase error (random for each simulation)
                    phase_error = magnitude * np.pi * (2 * np.random.random() - 1)
                    params = PulseParameters(
                        error_type=error_type,
                        error_phase=phase_error
                    )
                elif error_type == PulseError.DETUNING:
                    # Detuning error
                    params = PulseParameters(
                        error_type=error_type,
                        error_detuning=magnitude
                    )
                else:
                    # Combined error
                    amp_error = magnitude
                    phase_error = magnitude * np.pi * 0.1
                    params = PulseParameters(
                        error_type=error_type,
                        error_amplitude=amp_error,
                        error_phase=phase_error
                    )
                
                # Update sequence parameters
                sequence._pulse_params = params
                
                # Run simulation
                sim_result = sequence.simulate(include_decoherence=False)
                
                # Calculate fidelity
                fidelity = sequence._nv_system.state_overlap(ref_state, sim_result['final_state'])
                fidelities[i, j] = abs(fidelity) ** 2
        
        # Store results
        results['fidelities'][error_type.name] = {
            'values': fidelities,
            'mean': np.mean(fidelities, axis=1),
            'std': np.std(fidelities, axis=1)
        }
    
    # Restore original parameters
    sequence._pulse_params = original_params
    
    return results

def plot_error_susceptibility(results: Dict[str, Any],
                             ax = None) -> Any:
    """
    Plot the error susceptibility analysis results.
    
    Parameters:
    -----------
    results : Dict[str, Any]
        Error analysis results from error_susceptibility
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns:
    --------
    matplotlib.axes.Axes
        The axes used for plotting
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data
    error_magnitudes = results['error_magnitudes']
    
    # Colors for different error types
    colors = {
        'AMPLITUDE': 'r',
        'PHASE': 'g',
        'DETUNING': 'b',
        'COMBINED': 'm'
    }
    
    # Plot results for each error type
    for error_type, data in results['fidelities'].items():
        mean = data['mean']
        std = data['std']
        
        # Plot with error bars
        ax.errorbar(
            error_magnitudes,
            mean,
            yerr=std,
            fmt='o-',
            label=error_type,
            color=colors.get(error_type, 'k')
        )
    
    # Set up axis labels
    ax.set_xlabel('Error Magnitude')
    ax.set_ylabel('Fidelity')
    ax.set_title('Sequence Error Susceptibility')
    ax.grid(True)
    ax.legend()
    
    # Set y-axis limits
    ax.set_ylim(0, 1.05)
    
    return ax

def compare_sequences(sequences: List[DynamicalDecouplingSequence],
                     noise_model: Optional[Callable] = None,
                     pulse_error: Optional[PulseError] = None,
                     error_magnitude: float = 0.05) -> Dict[str, Any]:
    """
    Compare multiple dynamical decoupling sequences under the same conditions.
    
    Parameters:
    -----------
    sequences : List[DynamicalDecouplingSequence]
        List of sequences to compare
    noise_model : Callable, optional
        Noise spectral density function
    pulse_error : PulseError, optional
        Type of pulse error to simulate
    error_magnitude : float
        Magnitude of pulse error
        
    Returns:
    --------
    Dict[str, Any]
        Comparison results
    """
    # Initialize results
    results = {
        'sequences': [seq.name for seq in sequences],
        'filter_functions': {},
        'T2_times': {},
        'fidelities': {}
    }
    
    # Calculate filter functions
    omega = np.logspace(2, 8, 1000)
    for sequence in sequences:
        # Calculate filter function
        filter_function = calculate_sequence_filter_function(sequence, omega)
        results['filter_functions'][sequence.name] = filter_function
    
    # Analyze coherence times if noise model provided
    if noise_model is not None:
        for sequence in sequences:
            # Analyze sequence
            analysis = analyze_sequence_fidelity(sequence, noise_model)
            
            # Store coherence time
            results['T2_times'][sequence.name] = analysis['coherence']['T2']
    
    # Analyze pulse error susceptibility if error type provided
    if pulse_error is not None:
        for sequence in sequences:
            # Original parameters
            original_params = sequence.pulse_parameters
            
            # Create parameters with error
            if pulse_error == PulseError.AMPLITUDE:
                params = PulseParameters(
                    error_type=pulse_error,
                    error_amplitude=error_magnitude
                )
            elif pulse_error == PulseError.PHASE:
                params = PulseParameters(
                    error_type=pulse_error,
                    error_phase=error_magnitude * np.pi
                )
            elif pulse_error == PulseError.DETUNING:
                params = PulseParameters(
                    error_type=pulse_error,
                    error_detuning=error_magnitude
                )
            else:
                params = PulseParameters(
                    error_type=pulse_error,
                    error_amplitude=error_magnitude,
                    error_phase=error_magnitude * np.pi * 0.1
                )
            
            # Update sequence parameters
            sequence._pulse_params = params
            
            # Run simulation with error
            error_result = sequence.simulate(include_decoherence=False)
            
            # Run ideal simulation
            sequence._pulse_params = PulseParameters()
            ideal_result = sequence.simulate(include_decoherence=False)
            
            # Calculate fidelity
            fidelity = sequence._nv_system.state_overlap(
                ideal_result['final_state'],
                error_result['final_state']
            )
            results['fidelities'][sequence.name] = abs(fidelity) ** 2
            
            # Restore original parameters
            sequence._pulse_params = original_params
    
    return results

def plot_sequence_comparison(comparison_results: Dict[str, Any],
                           plot_type: str = 'filter',
                           ax = None) -> Any:
    """
    Plot the results of a sequence comparison.
    
    Parameters:
    -----------
    comparison_results : Dict[str, Any]
        Results from compare_sequences
    plot_type : str
        Type of plot ('filter', 'T2', 'fidelity')
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
        
    Returns:
    --------
    matplotlib.axes.Axes
        The axes used for plotting
    """
    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get sequence names
    sequences = comparison_results['sequences']
    
    if plot_type == 'filter':
        # Plot filter functions
        omega = np.logspace(2, 8, 1000)
        for sequence_name in sequences:
            filter_function = comparison_results['filter_functions'][sequence_name]
            ax.loglog(omega, filter_function, label=sequence_name)
            
        # Set up axis labels
        ax.set_xlabel('Angular Frequency ω (rad/s)')
        ax.set_ylabel('Filter Function F(ω)')
        ax.set_title('Filter Function Comparison')
        ax.grid(True, which='both', linestyle='--', alpha=0.5)
        ax.legend()
        
    elif plot_type == 'T2':
        # Plot T2 times
        if 'T2_times' not in comparison_results:
            raise ValueError("T2 times not available. Run comparison with noise_model.")
            
        # Get T2 times
        T2_times = [comparison_results['T2_times'][name] for name in sequences]
        
        # Create bar chart
        ax.bar(range(len(sequences)), T2_times)
        ax.set_xticks(range(len(sequences)))
        ax.set_xticklabels(sequences, rotation=45)
        ax.set_ylabel('T2 Time (s)')
        ax.set_title('Coherence Time Comparison')
        
    elif plot_type == 'fidelity':
        # Plot fidelities
        if 'fidelities' not in comparison_results:
            raise ValueError("Fidelity data not available. Run comparison with pulse_error.")
            
        # Get fidelities
        fidelities = [comparison_results['fidelities'][name] for name in sequences]
        
        # Create bar chart
        ax.bar(range(len(sequences)), fidelities)
        ax.set_xticks(range(len(sequences)))
        ax.set_xticklabels(sequences, rotation=45)
        ax.set_ylabel('Fidelity')
        ax.set_ylim(0, 1.05)
        ax.set_title('Pulse Error Robustness Comparison')
        
    else:
        raise ValueError(f"Unknown plot type: {plot_type}")
    
    return ax