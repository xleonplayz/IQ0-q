# -*- coding: utf-8 -*-

"""
Base class for dynamical decoupling sequences.

This module defines the framework for creating and simulating 
quantum-accurate dynamical decoupling pulse sequences.
"""

import numpy as np
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple, Union, Any, Callable
import time
import sys
import os

# Add the parent directory to the path to import SimOS
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import SimOS
try:
    from sim.src.sim.simos import core, propagation, states
except ImportError:
    # Try alternative import path
    from simos import core, propagation, states

class PulseError(Enum):
    """Types of pulse errors that can be simulated."""
    NONE = auto()
    AMPLITUDE = auto()  # Amplitude error (over/under rotation)
    PHASE = auto()      # Phase error
    DETUNING = auto()   # Frequency detuning error
    COMBINED = auto()   # Combination of multiple errors
    
class PulseParameters:
    """
    Parameters for dynamical decoupling pulses.
    
    This class holds the configuration for pulse shapes, errors, and durations.
    """
    
    def __init__(self, 
                 pi_pulse_duration: float = 50e-9,
                 pi_half_pulse_duration: float = 25e-9,
                 pulse_shape: str = 'square',
                 error_type: PulseError = PulseError.NONE,
                 error_amplitude: float = 0.0,
                 error_phase: float = 0.0,
                 error_detuning: float = 0.0,
                 time_resolution: int = 10):
        """
        Initialize pulse parameters.
        
        Parameters:
        -----------
        pi_pulse_duration : float
            Duration of π pulses in seconds
        pi_half_pulse_duration : float
            Duration of π/2 pulses in seconds
        pulse_shape : str
            Shape of the pulses ('square', 'gaussian', 'sinc', 'optimal')
        error_type : PulseError
            Type of pulse error to simulate
        error_amplitude : float
            Amplitude error (fractional, e.g., 0.05 = 5% error)
        error_phase : float
            Phase error in radians
        error_detuning : float
            Frequency detuning error in Hz
        time_resolution : int
            Number of time steps for simulating each pulse
        """
        self.pi_pulse_duration = pi_pulse_duration
        self.pi_half_pulse_duration = pi_half_pulse_duration
        self.pulse_shape = pulse_shape
        self.error_type = error_type
        self.error_amplitude = error_amplitude
        self.error_phase = error_phase
        self.error_detuning = error_detuning
        self.time_resolution = time_resolution
        
    def get_pulse_duration(self, angle: float) -> float:
        """
        Get the appropriate pulse duration for a given rotation angle.
        
        Parameters:
        -----------
        angle : float
            Rotation angle in radians
            
        Returns:
        --------
        float
            Pulse duration in seconds
        """
        if abs(abs(angle) - np.pi) < 1e-6:
            return self.pi_pulse_duration
        elif abs(abs(angle) - np.pi/2) < 1e-6:
            return self.pi_half_pulse_duration
        else:
            # Scale based on pi pulse duration
            return self.pi_pulse_duration * abs(angle) / np.pi
    
    def apply_errors(self, axis: str, angle: float) -> Tuple[str, float]:
        """
        Apply configured errors to a pulse.
        
        Parameters:
        -----------
        axis : str
            Pulse axis ('x', 'y', 'z')
        angle : float
            Rotation angle in radians
            
        Returns:
        --------
        Tuple[str, float]
            Modified (axis, angle) with errors applied
        """
        # No error
        if self.error_type == PulseError.NONE:
            return axis, angle
            
        # Amplitude error (over/under rotation)
        if self.error_type in (PulseError.AMPLITUDE, PulseError.COMBINED):
            angle *= (1.0 + self.error_amplitude)
            
        # Phase error
        if self.error_type in (PulseError.PHASE, PulseError.COMBINED):
            if axis == 'x':
                # Rotate in x-y plane by error_phase
                if abs(self.error_phase) > 1e-6:
                    if abs(self.error_phase - np.pi/2) < 1e-6:
                        axis = 'y'
                    elif abs(self.error_phase - np.pi) < 1e-6:
                        axis = '-x'
                    elif abs(self.error_phase - 3*np.pi/2) < 1e-6:
                        axis = '-y'
                    else:
                        # For other phases, we'll need more complex handling
                        # This is a simplified version - real implementation would
                        # need to handle arbitrary rotation axes
                        axis = f"x(φ={self.error_phase:.4f})"
            elif axis == 'y':
                # Similar logic for y-axis pulses
                if abs(self.error_phase) > 1e-6:
                    if abs(self.error_phase - np.pi/2) < 1e-6:
                        axis = '-x'
                    elif abs(self.error_phase - np.pi) < 1e-6:
                        axis = '-y'
                    elif abs(self.error_phase - 3*np.pi/2) < 1e-6:
                        axis = 'x'
                    else:
                        axis = f"y(φ={self.error_phase:.4f})"
                        
        return axis, angle


class DynamicalDecouplingSequence:
    """
    Dynamical decoupling sequence for NV center spin manipulation.
    
    This class provides the framework for creating and simulating
    quantum-accurate dynamical decoupling pulse sequences.
    """
    
    def __init__(self, 
                 nv_system: Any, 
                 pulse_params: Optional[PulseParameters] = None):
        """
        Initialize a dynamical decoupling sequence.
        
        Parameters:
        -----------
        nv_system : Any
            NV system object (compatible with SimOS)
        pulse_params : PulseParameters, optional
            Pulse parameters configuration
        """
        self._nv_system = nv_system
        self._pulse_params = pulse_params or PulseParameters()
        self._sequence = []
        self._total_sequence_time = 0.0
        self._name = "Generic DD Sequence"
        
    @property
    def name(self) -> str:
        """Get the sequence name."""
        return self._name
    
    @name.setter
    def name(self, value: str) -> None:
        """Set the sequence name."""
        self._name = value
        
    @property
    def sequence(self) -> List[Dict[str, Any]]:
        """Get the sequence elements."""
        return self._sequence.copy()
    
    @property
    def total_duration(self) -> float:
        """Get the total sequence duration in seconds."""
        return self._total_sequence_time
    
    @property
    def pulse_count(self) -> int:
        """Get the number of pulses in the sequence."""
        return sum(1 for e in self._sequence if e['type'] == 'pulse')
    
    @property
    def pi_pulse_count(self) -> int:
        """Get the number of π pulses in the sequence."""
        return sum(1 for e in self._sequence 
                  if e['type'] == 'pulse' and abs(abs(e['angle']) - np.pi) < 1e-6)
    
    @property
    def pulse_parameters(self) -> PulseParameters:
        """Get the pulse parameters."""
        return self._pulse_params
    
    def clear(self) -> None:
        """Clear the sequence."""
        self._sequence = []
        self._total_sequence_time = 0.0
    
    def add_pulse(self, 
                  axis: str, 
                  angle: float, 
                  duration: Optional[float] = None, 
                  shape: Optional[str] = None) -> None:
        """
        Add a pulse to the sequence.
        
        Parameters:
        -----------
        axis : str
            Rotation axis ('x', 'y', 'z')
        angle : float
            Rotation angle in radians
        duration : float, optional
            Pulse duration in seconds. If None, uses duration from pulse_params.
        shape : str, optional
            Pulse shape. If None, uses shape from pulse_params.
        """
        # Apply configured errors to the pulse
        axis, angle = self._pulse_params.apply_errors(axis, angle)
        
        # Determine pulse duration
        if duration is None:
            duration = self._pulse_params.get_pulse_duration(angle)
            
        # Determine pulse shape
        if shape is None:
            shape = self._pulse_params.pulse_shape
            
        # Add pulse to sequence
        self._sequence.append({
            'type': 'pulse',
            'axis': axis,
            'angle': angle,
            'duration': duration,
            'shape': shape
        })
        
        # Update total sequence time
        self._total_sequence_time += duration
        
    def add_delay(self, duration: float) -> None:
        """
        Add a delay (free evolution) to the sequence.
        
        Parameters:
        -----------
        duration : float
            Delay duration in seconds
        """
        # Add delay to sequence
        self._sequence.append({
            'type': 'delay',
            'duration': duration
        })
        
        # Update total sequence time
        self._total_sequence_time += duration
        
    def add_measurement(self, 
                        axis: Optional[str] = None,
                        duration: float = 0.0) -> None:
        """
        Add a measurement step to the sequence.
        
        Parameters:
        -----------
        axis : str, optional
            Measurement axis. If None, uses the default measurement axis.
        duration : float
            Measurement duration in seconds
        """
        # Add measurement to sequence
        self._sequence.append({
            'type': 'measurement',
            'axis': axis,
            'duration': duration
        })
        
        # Update total sequence time
        if duration > 0:
            self._total_sequence_time += duration
            
    def _create_pulse_hamiltonian(self, 
                                 element: Dict[str, Any], 
                                 time_fraction: float) -> Any:
        """
        Create the Hamiltonian for a pulse element at a specific time within the pulse.
        
        Parameters:
        -----------
        element : Dict[str, Any]
            Pulse element from sequence
        time_fraction : float
            Fraction of the pulse duration (0.0 to 1.0)
            
        Returns:
        --------
        Any
            Hamiltonian for the pulse (compatible with SimOS)
        """
        # Extract pulse parameters
        axis = element['axis']
        angle = element['angle']
        duration = element['duration']
        shape = element['shape']
        
        # Calculate pulse amplitude based on shape
        amplitude = self._get_pulse_amplitude(shape, time_fraction)
        
        # Scale angle by amplitude and convert to angular frequency
        angular_frequency = angle * amplitude / duration
        
        # Create rotation Hamiltonian based on axis
        if axis == 'x':
            return angular_frequency * self._nv_system.sx
        elif axis == 'y':
            return angular_frequency * self._nv_system.sy
        elif axis == 'z':
            return angular_frequency * self._nv_system.sz
        elif axis == '-x':
            return -angular_frequency * self._nv_system.sx
        elif axis == '-y':
            return -angular_frequency * self._nv_system.sy
        elif axis == '-z':
            return -angular_frequency * self._nv_system.sz
        else:
            raise ValueError(f"Unsupported pulse axis: {axis}")
            
    def _get_pulse_amplitude(self, shape: str, time_fraction: float) -> float:
        """
        Get the amplitude of a pulse at a specific time fraction.
        
        Parameters:
        -----------
        shape : str
            Pulse shape ('square', 'gaussian', 'sinc', 'optimal')
        time_fraction : float
            Fraction of the pulse duration (0.0 to 1.0)
            
        Returns:
        --------
        float
            Pulse amplitude (0.0 to 1.0)
        """
        if shape == 'square':
            return 1.0
        elif shape == 'gaussian':
            # Gaussian pulse with σ = 0.25
            sigma = 0.25
            t = time_fraction - 0.5  # Center at 0.5
            return np.exp(-0.5 * (t / sigma)**2)
        elif shape == 'sinc':
            # Sinc pulse
            t = time_fraction - 0.5  # Center at 0.5
            if abs(t) < 1e-10:
                return 1.0
            else:
                x = 5.0 * t  # Scale for appropriate width
                return np.sin(np.pi * x) / (np.pi * x)
        elif shape == 'optimal':
            # Placeholder for an optimal control pulse shape
            # In practice, this would be more sophisticated
            return 1.0
        else:
            raise ValueError(f"Unsupported pulse shape: {shape}")
            
    def simulate(self, 
                initial_state: Any = None, 
                magnetic_field: Optional[List[float]] = None,
                include_decoherence: bool = True,
                decoherence_model: str = 'simple',
                time_steps: Optional[int] = None) -> Dict[str, Any]:
        """
        Simulate the sequence using SimOS.
        
        Parameters:
        -----------
        initial_state : Any, optional
            Initial quantum state. If None, uses ground state.
        magnetic_field : List[float], optional
            Magnetic field vector [Bx, By, Bz] in Tesla.
        include_decoherence : bool
            Whether to include decoherence effects
        decoherence_model : str
            Decoherence model to use ('simple', 'bath', 'full')
        time_steps : int, optional
            Number of time steps for simulation. If None, auto-determined.
            
        Returns:
        --------
        Dict[str, Any]
            Simulation results including final state, coherence information
        """
        # Set up initial state
        if initial_state is None:
            initial_state = self._nv_system.ground_state()
        
        # Set up magnetic field
        if magnetic_field is not None:
            self._nv_system.set_magnetic_field(magnetic_field)
            
        # Prepare for simulation
        current_state = initial_state.copy()
        
        # Get system Hamiltonian (without control fields)
        H0 = self._nv_system.hamiltonian
        
        # Set up collapse operators for decoherence
        collapse_ops = None
        if include_decoherence:
            if decoherence_model == 'simple':
                # Simple T1/T2 decoherence
                collapse_ops = self._nv_system.get_decoherence_operators()
            elif decoherence_model == 'bath':
                # Nuclear spin bath decoherence
                collapse_ops = self._nv_system.get_bath_decoherence_operators()
            elif decoherence_model == 'full':
                # Full microscopic model
                collapse_ops = self._nv_system.get_full_decoherence_operators()
                
        # Determine timestep if not specified
        if time_steps is None:
            # Reasonable default: at least 10 steps per pulse or delay
            min_duration = min(e['duration'] for e in self._sequence 
                              if e.get('duration', 0) > 0)
            time_steps = max(1000, int(self._total_sequence_time / min_duration * 10))
        
        # Store measurement results
        measurements = []
        coherence_values = []
        
        # Time points for tracking
        time_points = np.linspace(0, self._total_sequence_time, time_steps+1)
        current_time = 0.0
        
        # Apply each element in the sequence
        sequence_idx = 0
        while sequence_idx < len(self._sequence):
            element = self._sequence[sequence_idx]
            
            # Determine next time steps within this element
            element_duration = element.get('duration', 0.0)
            element_end_time = current_time + element_duration
            
            # Find time points within this element
            element_time_indices = np.where(
                (time_points > current_time) & 
                (time_points <= element_end_time)
            )[0]
            
            if element['type'] == 'delay':
                # Free evolution under the system Hamiltonian
                if len(element_time_indices) > 0:
                    for i in element_time_indices:
                        step_duration = time_points[i] - time_points[i-1]
                        
                        # Evolve state
                        if collapse_ops:
                            # Use master equation for decoherence
                            current_state = propagation.mesolve(
                                H0, current_state, step_duration, collapse_ops
                            )
                        else:
                            # Pure unitary evolution
                            propagator = propagation.evol(H0, step_duration)
                            current_state = propagator * current_state
                else:
                    # Single step evolution for short elements
                    if collapse_ops:
                        current_state = propagation.mesolve(
                            H0, current_state, element_duration, collapse_ops
                        )
                    else:
                        propagator = propagation.evol(H0, element_duration)
                        current_state = propagator * current_state
                
            elif element['type'] == 'pulse':
                # Apply pulse with appropriate shape
                pulse_timesteps = max(10, len(element_time_indices))
                
                if len(element_time_indices) > 0:
                    for i in element_time_indices:
                        step_duration = time_points[i] - time_points[i-1]
                        time_fraction = (time_points[i] - current_time) / element_duration
                        
                        # Create pulse Hamiltonian for this time step
                        H_pulse = self._create_pulse_hamiltonian(element, time_fraction)
                        H_total = H0 + H_pulse
                        
                        # Evolve state
                        if collapse_ops:
                            current_state = propagation.mesolve(
                                H_total, current_state, step_duration, collapse_ops
                            )
                        else:
                            propagator = propagation.evol(H_total, step_duration)
                            current_state = propagator * current_state
                else:
                    # Discretize the pulse manually for short elements
                    dt = element_duration / pulse_timesteps
                    for j in range(pulse_timesteps):
                        time_fraction = (j + 0.5) / pulse_timesteps
                        H_pulse = self._create_pulse_hamiltonian(element, time_fraction)
                        H_total = H0 + H_pulse
                        
                        if collapse_ops:
                            current_state = propagation.mesolve(
                                H_total, current_state, dt, collapse_ops
                            )
                        else:
                            propagator = propagation.evol(H_total, dt)
                            current_state = propagator * current_state
            
            elif element['type'] == 'measurement':
                # Perform measurement
                measurement_axis = element['axis']
                if measurement_axis == 'x':
                    operator = self._nv_system.sx
                elif measurement_axis == 'y':
                    operator = self._nv_system.sy
                elif measurement_axis == 'z':
                    operator = self._nv_system.sz
                else:
                    # Default to population measurement
                    operator = self._nv_system.projection_0
                    
                # Calculate expectation value
                expectation = self._nv_system.expectation_value(
                    current_state, operator
                )
                
                # Store measurement
                measurements.append({
                    'time': current_time,
                    'value': expectation
                })
                
                # Calculate coherence if needed
                if initial_state is not None:
                    coherence = self._nv_system.state_overlap(
                        initial_state, current_state
                    )
                    coherence_values.append({
                        'time': current_time,
                        'value': coherence
                    })
            
            # Update current time
            current_time = element_end_time
            sequence_idx += 1
        
        # Final measurement if none was performed
        if not measurements:
            # Default to population measurement
            expectation = self._nv_system.expectation_value(
                current_state, self._nv_system.projection_0
            )
            
            # Store measurement
            measurements.append({
                'time': current_time,
                'value': expectation
            })
            
            # Calculate coherence
            if initial_state is not None:
                coherence = self._nv_system.state_overlap(
                    initial_state, current_state
                )
                coherence_values.append({
                    'time': current_time,
                    'value': coherence
                })
                
        # Return results
        return {
            'sequence_name': self._name,
            'sequence': self._sequence,
            'duration': self._total_sequence_time,
            'final_state': current_state,
            'measurements': measurements,
            'coherence_values': coherence_values
        }
                
    def plot_sequence(self, 
                     include_amplitude: bool = True, 
                     ax = None, 
                     **kwargs) -> Any:
        """
        Plot the pulse sequence diagram.
        
        Parameters:
        -----------
        include_amplitude : bool
            Whether to include pulse amplitude profiles
        ax : matplotlib.axes.Axes, optional
            Axes to plot on
        **kwargs : dict
            Additional keyword arguments for plotting
            
        Returns:
        --------
        matplotlib.axes.Axes
            The axes used for plotting
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
        except ImportError:
            raise ImportError("Matplotlib is required for plotting")
            
        # Create axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 4))
            
        # Set up colors for different axes
        colors = {
            'x': 'r',
            '-x': 'darkred',
            'y': 'g',
            '-y': 'darkgreen',
            'z': 'b',
            '-z': 'darkblue'
        }
        
        # Set up y-positions for different channels
        channels = {
            'x': 0,
            '-x': 0,
            'y': 1,
            '-y': 1,
            'z': 2,
            '-z': 2
        }
        
        # Plot sequence elements
        current_time = 0.0
        
        for element in self._sequence:
            if element['type'] == 'pulse':
                # Extract pulse parameters
                axis = element['axis']
                angle = element['angle']
                duration = element['duration']
                shape = element['shape']
                
                # Determine color and channel
                color = colors.get(axis, 'gray')
                channel = channels.get(axis, 3)
                
                # Plot pulse
                if include_amplitude and shape != 'square':
                    # Plot pulse shape with amplitude profile
                    t = np.linspace(0, duration, 100)
                    amp = np.array([self._get_pulse_amplitude(shape, i/100) for i in range(100)])
                    amp = amp * 0.4  # Scale for visibility
                    
                    # Plot pulse envelope
                    ax.fill_between(
                        current_time + t,
                        channel - amp,
                        channel + amp,
                        color=color,
                        alpha=0.4
                    )
                    
                    # Plot pulse edges
                    ax.plot([current_time, current_time + duration],
                          [channel, channel], 'k-', lw=1)
                else:
                    # Plot square pulse
                    rect = Rectangle(
                        (current_time, channel - 0.4),
                        duration,
                        0.8,
                        facecolor=color,
                        alpha=0.6,
                        edgecolor='k'
                    )
                    ax.add_patch(rect)
                
                # Label pulse angle
                if abs(abs(angle) - np.pi) < 1e-6:
                    label = 'π'
                elif abs(abs(angle) - np.pi/2) < 1e-6:
                    label = 'π/2'
                else:
                    label = f"{angle/np.pi:.2f}π"
                    
                # Add text label
                ax.text(
                    current_time + duration/2,
                    channel,
                    label,
                    ha='center',
                    va='center',
                    color='white',
                    fontweight='bold',
                    bbox=dict(facecolor='black', alpha=0.7, pad=1)
                )
                
            elif element['type'] == 'delay':
                # Plot delay
                duration = element['duration']
                ax.plot(
                    [current_time, current_time + duration],
                    [4, 4],  # Plot delays at the top
                    'k-',
                    lw=2
                )
                
                # Label delay
                ax.text(
                    current_time + duration/2,
                    4.2,
                    f"τ = {duration*1e9:.1f} ns",
                    ha='center',
                    va='bottom',
                    fontsize=8
                )
                
            # Update current time
            current_time += element.get('duration', 0.0)
            
        # Set up axis labels and limits
        ax.set_yticks([0, 1, 2, 4])
        ax.set_yticklabels(['X', 'Y', 'Z', 'Delay'])
        ax.set_xlabel('Time (s)')
        ax.set_xlim(0, self._total_sequence_time * 1.05)
        ax.set_ylim(-1, 5)
        ax.set_title(f"{self._name} Sequence")
        
        return ax
    
    def __str__(self) -> str:
        """String representation of the sequence."""
        pulse_count = sum(1 for e in self._sequence if e['type'] == 'pulse')
        delay_count = sum(1 for e in self._sequence if e['type'] == 'delay')
        
        return (f"{self._name}: {len(self._sequence)} elements "
                f"({pulse_count} pulses, {delay_count} delays), "
                f"total duration: {self._total_sequence_time*1e9:.1f} ns")