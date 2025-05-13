# -*- coding: utf-8 -*-

"""
Base class for experiment modes in the NV center simulator.

Copyright (c) 2023
"""

from typing import Dict, Any
from abc import ABC, abstractmethod


class ExperimentMode(ABC):
    """
    Abstract base class for experiment modes in the NV center simulator.
    Each experiment mode encapsulates the parameters and execution logic
    for a particular type of quantum experiment (ODMR, Rabi, etc.).
    """
    
    def __init__(self, simulator):
        """
        Initialize the experiment mode with a simulator instance.
        
        @param simulator: PhysicalNVModel instance to run the experiment on
        """
        self._simulator = simulator
        self._params = {}
        self._default_params = {}
    
    def configure(self, **params):
        """
        Configure the experiment with parameters.
        
        @param params: Parameter key-value pairs to configure the experiment
        """
        self._params.update(params)
        return self
    
    @abstractmethod
    def run(self):
        """
        Run the experiment and return results.
        This method must be implemented by each specific experiment mode.
        
        @return: Experiment results in a Qudi-compatible format
        """
        raise NotImplementedError
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Return default parameters for this experiment.
        
        @return: Dictionary of default parameters
        """
        return self._default_params.copy()
    
    def get_current_parameters(self) -> Dict[str, Any]:
        """
        Return the current parameters for this experiment.
        
        @return: Dictionary of current parameters
        """
        return self._params.copy()
    
    def reset_parameters(self):
        """
        Reset parameters to default values.
        """
        self._params = self.get_default_parameters()
        return self