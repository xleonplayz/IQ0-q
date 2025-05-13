"""Adaptive time stepping for efficient quantum evolution.

This module provides classes and functions for adaptive time stepping in
quantum evolution simulations, allowing efficient handling of different
time scales in the simulation.
"""

import numpy as np
import logging
from .thread_safety import thread_safe

# Configure module logger
logger = logging.getLogger(__name__)

class AdaptiveEvolution:
    """
    Adaptive time stepping for quantum evolution.
    
    This class provides methods for evolving quantum systems with adaptive
    time steps, optimizing simulation performance while maintaining accuracy.
    """
    
    def __init__(self, simulator, min_step=None, max_step=None, tolerance=1e-6):
        """
        Initialize the adaptive evolution handler.
        
        Parameters
        ----------
        simulator : object
            Simulator object that provides _take_step method
        min_step : float, optional
            Minimum step size in seconds
        max_step : float, optional
            Maximum step size in seconds
        tolerance : float, optional
            Error tolerance for adaptive stepping
        """
        self.simulator = simulator
        self.min_step = min_step
        self.max_step = max_step
        self.tolerance = tolerance
        
        # Statistics for performance monitoring
        self.stats = {
            'total_steps': 0,
            'accepted_steps': 0,
            'rejected_steps': 0,
            'total_time': 0.0,
        }
    
    @thread_safe
    def evolve(self, state, hamiltonian, time_s, c_ops=None):
        """
        Evolve the quantum state with adaptive time stepping.
        
        Parameters
        ----------
        state : object
            Initial quantum state
        hamiltonian : object
            System Hamiltonian
        time_s : float
            Total evolution time in seconds
        c_ops : list, optional
            List of collapse operators for open system evolution
            
        Returns
        -------
        object
            Evolved quantum state
        dict
            Statistics about the evolution
        """
        # Reset statistics
        self.stats = {
            'total_steps': 0,
            'accepted_steps': 0,
            'rejected_steps': 0,
            'total_time': 0.0,
        }
        
        if time_s <= 0:
            return state, self.stats
            
        # Default step size limits
        if self.max_step is None:
            self.max_step = time_s / 10
        if self.min_step is None:
            self.min_step = time_s / 1000
        
        # Current step size - start with max_step
        step_size = self.max_step
        
        # Integrate with adaptive step size
        t = 0
        current_state = state
        
        while t < time_s:
            # Determine step size (don't exceed remaining time)
            step = min(step_size, time_s - t)
            
            # Track statistics
            self.stats['total_steps'] += 1
            
            # Take a trial step
            state_trial = self.simulator._take_step(current_state, hamiltonian, step, c_ops)
            
            # Take two half steps
            half_step = step / 2
            state_half = self.simulator._take_step(current_state, hamiltonian, half_step, c_ops)
            state_half_half = self.simulator._take_step(state_half, hamiltonian, half_step, c_ops)
            
            # Estimate error
            error = self._estimate_error(state_trial, state_half_half)
            
            # Accept or reject step
            if error < self.tolerance or step <= self.min_step:
                # Accept the step
                current_state = state_half_half  # Use the more accurate half-step result
                t += step
                self.stats['accepted_steps'] += 1
                self.stats['total_time'] += step
                
                # Adjust step size based on error for next step
                if error > 0:
                    # Safety factor of 0.9, exponent 0.2 for smooth size control
                    new_step = 0.9 * step * (self.tolerance / error)**0.2
                    step_size = min(self.max_step, max(self.min_step, new_step))
            else:
                # Reject the step and reduce step size
                self.stats['rejected_steps'] += 1
                step_size = max(self.min_step, step * 0.5)
                
            # Log progress periodically
            if self.stats['total_steps'] % 100 == 0:
                logger.debug(
                    f"Adaptive evolution: {t/time_s:.1%} complete, "
                    f"step size: {step_size:.2e}s, "
                    f"error: {error:.2e}"
                )
        
        # Log final statistics
        accept_ratio = self.stats['accepted_steps'] / max(1, self.stats['total_steps'])
        logger.info(
            f"Adaptive evolution completed: {self.stats['total_steps']} steps, "
            f"{accept_ratio:.1%} acceptance rate"
        )
        
        return current_state, self.stats
    
    def _estimate_error(self, state1, state2):
        """
        Estimate error between two states.
        
        Parameters
        ----------
        state1 : object
            First state
        state2 : object
            Second state
            
        Returns
        -------
        float
            Error estimate
        """
        # Use appropriate error metric based on state type
        # This implementation assumes SimOS/QuTiP-like states
        try:
            # Try to use trace distance which is usually available
            from simos.qmatrixmethods import tracedist
            return tracedist(state1, state2)
        except ImportError:
            # Fallback to simple difference norm
            try:
                diff = state1 - state2
                # For matrices, use Frobenius norm
                if hasattr(diff, 'norm'):
                    return diff.norm() / max(1e-15, state1.norm())
                # For arrays, use L2 norm
                return np.linalg.norm(diff) / max(1e-15, np.linalg.norm(state1))
            except Exception as e:
                logger.warning(f"Error estimation failed: {e}. Using default value.")
                return 0.1  # Default conservative error estimate


class RichardsonExtrapolation:
    """
    Richardson extrapolation for improving numerical accuracy.
    
    This class implements Richardson extrapolation to improve the accuracy
    of numerical quantum evolution by combining results from different step sizes.
    """
    
    def __init__(self, simulator, base_step=1e-9, order=4):
        """
        Initialize the Richardson extrapolation handler.
        
        Parameters
        ----------
        simulator : object
            Simulator object that provides _take_step method
        base_step : float, optional
            Base step size in seconds
        order : int, optional
            Extrapolation order (number of step sizes to use)
        """
        self.simulator = simulator
        self.base_step = base_step
        self.order = order
    
    @thread_safe
    def evolve(self, state, hamiltonian, time_s, c_ops=None):
        """
        Evolve the quantum state with Richardson extrapolation.
        
        Parameters
        ----------
        state : object
            Initial quantum state
        hamiltonian : object
            System Hamiltonian
        time_s : float
            Total evolution time in seconds
        c_ops : list, optional
            List of collapse operators for open system evolution
            
        Returns
        -------
        object
            Evolved quantum state
        """
        if time_s <= 0:
            return state
        
        # Create step sizes: h, h/2, h/4, ...
        step_sizes = [self.base_step / (2**i) for i in range(self.order)]
        
        # For each step size, evolve the system
        results = []
        for step_size in step_sizes:
            # Calculate number of steps
            n_steps = max(1, int(np.ceil(time_s / step_size)))
            actual_step = time_s / n_steps
            
            # Evolve using fixed step size
            current = state
            for _ in range(n_steps):
                current = self.simulator._take_step(current, hamiltonian, actual_step, c_ops)
            
            results.append(current)
        
        # Apply Richardson extrapolation
        final_state = self._richardson_extrapolate(results)
        
        return final_state
    
    def _richardson_extrapolate(self, results):
        """
        Apply Richardson extrapolation to results from different step sizes.
        
        Parameters
        ----------
        results : list
            List of states from different step sizes, ordered from largest to smallest step
            
        Returns
        -------
        object
            Extrapolated state
        """
        try:
            # This is a simplified implementation assuming state objects support
            # addition and scalar multiplication
            
            # Richardson extrapolation coefficients
            # For first-order methods with steps h, h/2, h/4, ...
            # the coefficients are 2^n / (2^n - 1) for the finer result
            # and -1 / (2^n - 1) for the coarser result
            extrapolated = results[-1]  # Start with finest resolution
            
            for i in range(self.order - 1):
                # Calculate Richardson coefficients for ith extrapolation
                j = self.order - i - 2  # Index into results list
                factor = 2**(i + 1)
                coeff1 = factor / (factor - 1)
                coeff2 = -1 / (factor - 1)
                
                # Combine results with appropriate coefficients
                extrapolated = coeff1 * extrapolated + coeff2 * results[j]
            
            return extrapolated
            
        except Exception as e:
            logger.warning(f"Richardson extrapolation failed: {e}. Using finest resolution result.")
            return results[-1]  # Return result from smallest step size
