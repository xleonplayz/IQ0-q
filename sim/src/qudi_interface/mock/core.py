# -*- coding: utf-8 -*-
"""
Mock implementation of Qudi core classes for standalone testing.

Copyright (c) 2023
"""

import logging
import threading


class Base:
    """
    Base class for all Qudi modules. This is a simplified version
    for testing without Qudi installation.
    """
    
    def __init__(self, config=None, name=None, **kwargs):
        """
        Mock initialization for Base module.
        
        @param config: Module configuration as dict
        @param name: Unique name for this module instance
        """
        self._module_state = ModuleState()
        self.module_state = self._module_state
        self.config = config or {}
        self._name = name or self.__class__.__name__
        
        # Set up logging
        self.log = logging.getLogger(self._name)
        if not self.log.handlers:
            # Add console handler if none exists
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.log.addHandler(handler)
            self.log.setLevel(logging.INFO)
    
    def on_activate(self):
        """
        Method called when module is activated.
        Should be overridden by derived classes.
        """
        pass
    
    def on_deactivate(self):
        """
        Method called when module is deactivated.
        Should be overridden by derived classes.
        """
        pass


class Module(Base):
    """
    Mock implementation of Qudi module class.
    Just a placeholder for compatibility.
    """
    pass


class ModuleState:
    """
    Mock implementation of Qudi module state.
    Provides basic state machine functionality for testing.
    """
    
    def __init__(self):
        """Initialize with idle state and create lock"""
        self._state = 'idle'
        self._lock = threading.RLock()
    
    def lock(self):
        """Lock the module state to 'running'"""
        if self._state == 'idle':
            self._state = 'running'
    
    def unlock(self):
        """Unlock the module state to 'idle'"""
        if self._state == 'running':
            self._state = 'idle'
    
    def __call__(self):
        """Get the current state when called"""
        return self._state
    
    def lock_access(self):
        """Return the thread lock for exclusive access"""
        return self._lock