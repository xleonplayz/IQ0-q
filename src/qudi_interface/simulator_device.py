# -*- coding: utf-8 -*-

"""
Qudi hardware interface adapter for the NV center simulator.
Main device class that integrates and manages all interfaces.

Copyright (c) 2023
"""

import os
import sys
import importlib.util
import logging
from typing import Optional, Dict, Any, Union

# Import the simulator
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'model.py')
spec = importlib.util.spec_from_file_location("model", model_path)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)
PhysicalNVModel = model.PhysicalNVModel

# Import our interface adapters
from .microwave_adapter import NVSimulatorMicrowave
from .scanner_adapter import NVSimulatorScanner

# Import experiment modes
from .experiments import (
    ODMRMode,
    RabiMode,
    RamseyMode,
    SpinEchoMode,
    T1Mode,
    CustomSequenceMode
)


class NVSimulatorDevice:
    """
    Main class that integrates the NV simulator with Qudi's hardware interfaces.
    This class creates and manages the interface adapters required for experiments.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, name: str = 'nvsim'):
        """
        Initialize the NV simulator device with all required interfaces.
        
        @param config: Optional configuration dictionary
        @param name: Base name for this device
        """
        self.log = logging.getLogger(name)
        self.log.info("Initializing NV Simulator Device")
        
        # Parse config if provided
        self._config = config or {}
        
        # Initialize the core simulator
        simulator_params = self._config.get('simulator_params', {})
        self._simulator = PhysicalNVModel(**simulator_params)
        
        # Initialize interfaces
        self._initialize_interfaces()
        
        # Initialize experiment modes
        self._initialize_experiment_modes()
        
        self.log.info("NV Simulator Device initialization complete")
        
    def _initialize_interfaces(self):
        """
        Initialize all the required interface adapters.
        """
        # Create microwave interface
        self.microwave = NVSimulatorMicrowave(
            nv_simulator=self._simulator, 
            name=self._config.get('microwave_name', 'nvmw')
        )
        
        # Create scanner interface
        self.scanner = NVSimulatorScanner(
            nv_simulator=self._simulator,
            microwave_adapter=self.microwave,
            name=self._config.get('scanner_name', 'nvscan')
        )
        
        # Create confocal scanner interface if confocal module is available
        try:
            from ..confocal import ConfocalSimulatorScanner
            self.confocal = ConfocalSimulatorScanner(
                name=self._config.get('confocal_name', 'nvconfocal')
            )
        except ImportError:
            self.log.info("Confocal scanner module not available")
            self.confocal = None
        
        # Additional interfaces can be added here as needed
    
    def _initialize_experiment_modes(self):
        """
        Initialize the experiment mode instances.
        """
        # Create experiment mode instances
        self._experiment_modes = {
            'odmr': ODMRMode(self._simulator),
            'rabi': RabiMode(self._simulator),
            'ramsey': RamseyMode(self._simulator),
            'spin_echo': SpinEchoMode(self._simulator),
            't1': T1Mode(self._simulator),
            'custom': CustomSequenceMode(self._simulator)
        }
        
        # Add confocal mode if confocal module is available
        try:
            from ..confocal import ConfocalSimulator
            if hasattr(self, 'confocal') and self.confocal is not None:
                class ConfocalMode:
                    def __init__(self, confocal_scanner):
                        self._confocal = confocal_scanner
                        self._params = {}
                        
                    def configure(self, **params):
                        self._params.update(params)
                        return self
                        
                    def run(self):
                        # Extract parameters with defaults
                        center = self._params.get('center', (10e-6, 10e-6, 0))
                        size = self._params.get('size', (10e-6, 10e-6))
                        resolution = self._params.get('resolution', (100, 100))
                        integration_time = self._params.get('integration_time', 0.01)
                        
                        # Run the confocal scan
                        image = self._confocal._confocal_simulator.scan_plane(
                            center, size, resolution, integration_time)
                        
                        # Return results
                        return {
                            'image': image,
                            'center': center,
                            'size': size,
                            'resolution': resolution,
                            'parameters': self._params.copy()
                        }
                
                self._experiment_modes['confocal'] = ConfocalMode(self.confocal)
        except ImportError:
            pass  # Confocal mode not available
        
    def get_simulator_instance(self):
        """
        Access to the underlying simulator instance for advanced usage.
        
        @return: The PhysicalNVModel instance
        """
        return self._simulator
    
    def configure_simulator(self, **kwargs):
        """
        Configure simulator parameters.
        
        @param kwargs: Parameters to configure
        """
        # Common configurations
        if 'magnetic_field' in kwargs:
            self._simulator.set_magnetic_field(kwargs['magnetic_field'])
            
        if 'laser_power' in kwargs:
            power = kwargs['laser_power']
            on = kwargs.get('laser_on', True)
            self._simulator.apply_laser(power, on)
            
        # Handle additional parameter updates
        for key, value in kwargs.items():
            if hasattr(self._simulator, key):
                setattr(self._simulator, key, value)
                
        self.log.info(f"Simulator configured with parameters: {kwargs}")
    
    def get_experiment_mode(self, mode_name: str):
        """
        Get an experiment mode by name.
        
        @param mode_name: Name of the experiment mode to retrieve
        @return: The experiment mode instance or None if not found
        """
        return self._experiment_modes.get(mode_name)
    
    def run_experiment(self, mode_name: str, **params) -> Dict[str, Any]:
        """
        Run an experiment with the given parameters using the specified mode.
        
        @param mode_name: Name of the experiment mode to use
        @param params: Experiment parameters to configure
        
        @return: Experiment results in Qudi-compatible format
        """
        self.log.info(f"Running experiment '{mode_name}' with parameters: {params}")
        
        # Get the experiment mode
        mode = self.get_experiment_mode(mode_name)
        if mode is None:
            error_msg = f"Unknown experiment mode: {mode_name}"
            self.log.error(error_msg)
            raise ValueError(error_msg)
            
        # Configure the experiment with parameters
        mode.configure(**params)
        
        # Run the experiment
        try:
            result = mode.run()
            self.log.info(f"Experiment '{mode_name}' completed successfully")
            return result
        except Exception as e:
            self.log.error(f"Error running experiment '{mode_name}': {str(e)}")
            raise
    
    def run_simulation(self, experiment_type: str, **params) -> Any:
        """
        Run a predefined experiment simulation with proper error handling.
        This is maintained for backward compatibility with older code.
        
        @param experiment_type: Type of experiment to run
        @param params: Experiment parameters
        
        @return: Simulation results
        """
        self.log.info(f"Running {experiment_type} simulation with parameters: {params}")
        
        # Map old experiment types to new experiment modes when possible
        if experiment_type == 'odmr':
            f_min = params.get('f_min', 2.7e9)
            f_max = params.get('f_max', 3.0e9)
            n_points = params.get('n_points', 101)
            mw_power = params.get('mw_power', -10.0)
            
            # Use new experiment mode system with proper error handling
            try:
                odmr_params = {
                    'freq_start': f_min,
                    'freq_stop': f_max,
                    'num_points': n_points,
                    'power': mw_power
                }
                result = self.run_experiment('odmr', **odmr_params)
                self.log.info("ODMR experiment completed successfully using experiment mode")
                return result
            except ValueError as e:
                # Handle parameter validation errors
                self.log.error(f"Parameter error in ODMR experiment: {str(e)}")
                self.log.info("Falling back to direct simulator call")
                result = self._simulator.simulate_odmr(f_min, f_max, n_points, mw_power)
            except NotImplementedError as e:
                # Handle missing implementation
                self.log.warning(f"Experiment mode not fully implemented: {str(e)}")
                self.log.info("Falling back to direct simulator call")
                result = self._simulator.simulate_odmr(f_min, f_max, n_points, mw_power)
            except Exception as e:
                # Log unexpected errors with full traceback
                self.log.error(f"Unexpected error in ODMR experiment: {str(e)}", exc_info=True)
                self.log.info("Falling back to direct simulator call")
                result = self._simulator.simulate_odmr(f_min, f_max, n_points, mw_power)
            
        elif experiment_type == 'rabi':
            t_max = params.get('t_max', 1e-6)
            n_points = params.get('n_points', 101)
            mw_frequency = params.get('mw_frequency', None)
            mw_power = params.get('mw_power', -10.0)
            
            # Try using new experiment mode system with proper error handling
            try:
                import numpy as np
                rabi_params = {
                    'rabi_times': np.linspace(0, t_max, n_points),
                    'mw_frequency': mw_frequency if mw_frequency is not None else 2.87e9,
                    'mw_power': mw_power
                }
                result = self.run_experiment('rabi', **rabi_params)
                self.log.info("Rabi experiment completed successfully using experiment mode")
                return result
            except ValueError as e:
                # Handle parameter validation errors
                self.log.error(f"Parameter error in Rabi experiment: {str(e)}")
                self.log.info("Falling back to direct simulator call")
                result = self._simulator.simulate_rabi(t_max, n_points, mw_frequency, mw_power)
            except NotImplementedError as e:
                # Handle missing implementation
                self.log.warning(f"Experiment mode not fully implemented: {str(e)}")
                self.log.info("Falling back to direct simulator call")
                result = self._simulator.simulate_rabi(t_max, n_points, mw_frequency, mw_power)
            except Exception as e:
                # Log unexpected errors with full traceback
                self.log.error(f"Unexpected error in Rabi experiment: {str(e)}", exc_info=True)
                self.log.info("Falling back to direct simulator call")
                result = self._simulator.simulate_rabi(t_max, n_points, mw_frequency, mw_power)
            
        elif experiment_type == 't1':
            t_max = params.get('t_max', 1e-3)
            n_points = params.get('n_points', 101)
            
            # Try using new experiment mode system with proper error handling
            try:
                import numpy as np
                t1_params = {
                    'tau_times': np.linspace(0, t_max, n_points)
                }
                result = self.run_experiment('t1', **t1_params)
                self.log.info("T1 experiment completed successfully using experiment mode")
                return result
            except ValueError as e:
                # Handle parameter validation errors
                self.log.error(f"Parameter error in T1 experiment: {str(e)}")
                self.log.info("Falling back to direct simulator call")
                result = self._simulator.simulate_t1(t_max, n_points)
            except NotImplementedError as e:
                # Handle missing implementation
                self.log.warning(f"Experiment mode not fully implemented: {str(e)}")
                self.log.info("Falling back to direct simulator call")
                result = self._simulator.simulate_t1(t_max, n_points)
            except Exception as e:
                # Log unexpected errors with full traceback
                self.log.error(f"Unexpected error in T1 experiment: {str(e)}", exc_info=True)
                self.log.info("Falling back to direct simulator call")
                result = self._simulator.simulate_t1(t_max, n_points)
            
        elif experiment_type == 't2_echo':
            t_max = params.get('t_max', 1e-5)
            n_points = params.get('n_points', 101)
            mw_frequency = params.get('mw_frequency', None)
            mw_power = params.get('mw_power', 0.0)
            
            # Try using new experiment mode system with proper error handling
            try:
                import numpy as np
                echo_params = {
                    'tau_times': np.linspace(0, t_max/2, n_points),  # t_max is total sequence time
                    'mw_frequency': mw_frequency if mw_frequency is not None else 2.87e9,
                    'mw_power': mw_power
                }
                result = self.run_experiment('spin_echo', **echo_params)
                self.log.info("Spin echo experiment completed successfully using experiment mode")
                return result
            except ValueError as e:
                # Handle parameter validation errors
                self.log.error(f"Parameter error in spin echo experiment: {str(e)}")
                self.log.info("Falling back to direct simulator call")
                result = self._simulator.simulate_t2_echo(t_max, n_points, mw_frequency, mw_power)
            except NotImplementedError as e:
                # Handle missing implementation
                self.log.warning(f"Experiment mode not fully implemented: {str(e)}")
                self.log.info("Falling back to direct simulator call")
                result = self._simulator.simulate_t2_echo(t_max, n_points, mw_frequency, mw_power)
            except Exception as e:
                # Log unexpected errors with full traceback
                self.log.error(f"Unexpected error in spin echo experiment: {str(e)}", exc_info=True)
                self.log.info("Falling back to direct simulator call")
                result = self._simulator.simulate_t2_echo(t_max, n_points, mw_frequency, mw_power)
            
        elif experiment_type == 'dynamical_decoupling':
            sequence_type = params.get('sequence_type', 'cpmg')
            t_max = params.get('t_max', 1e-5)
            n_points = params.get('n_points', 101)
            n_pulses = params.get('n_pulses', 4)
            mw_frequency = params.get('mw_frequency', None)
            mw_power = params.get('mw_power', 0.0)
            
            # Try using new experiment mode system with proper error handling
            try:
                import numpy as np
                dd_params = {
                    'sequence_name': sequence_type,
                    'tau_times': np.linspace(0, t_max, n_points),
                    'mw_frequency': mw_frequency if mw_frequency is not None else 2.87e9,
                    'mw_power': mw_power,
                    'sequence_params': {
                        'n_pulses': n_pulses
                    }
                }
                result = self.run_experiment('custom', **dd_params)
                self.log.info(f"Dynamical decoupling ({sequence_type}) experiment completed successfully using experiment mode")
                return result
            except ValueError as e:
                # Handle parameter validation errors
                self.log.error(f"Parameter error in dynamical decoupling experiment: {str(e)}")
                self.log.info("Falling back to direct simulator call")
                result = self._simulator.simulate_dynamical_decoupling(
                    sequence_type, t_max, n_points, n_pulses, mw_frequency, mw_power
                )
            except NotImplementedError as e:
                # Handle missing implementation
                self.log.warning(f"Experiment mode not fully implemented: {str(e)}")
                self.log.info("Falling back to direct simulator call")
                result = self._simulator.simulate_dynamical_decoupling(
                    sequence_type, t_max, n_points, n_pulses, mw_frequency, mw_power
                )
            except Exception as e:
                # Log unexpected errors with full traceback
                self.log.error(f"Unexpected error in dynamical decoupling experiment: {str(e)}", exc_info=True)
                self.log.info("Falling back to direct simulator call")
                result = self._simulator.simulate_dynamical_decoupling(
                    sequence_type, t_max, n_points, n_pulses, mw_frequency, mw_power
                )
            
        else:
            error_msg = f"Unknown experiment type: {experiment_type}"
            self.log.error(error_msg)
            raise ValueError(error_msg)
            
        self.log.info(f"Simulation completed successfully using direct simulator call")
        return result