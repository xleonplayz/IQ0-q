#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mock Qudi core modules for standalone testing.

Copyright (c) 2023, IQO

This file is part of qudi.

Qudi is free software: you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

Qudi is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with qudi.
If not, see <https://www.gnu.org/licenses/>.
"""

import logging
import threading


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


class Base:
    """
    Base class for all Qudi modules. This is a simplified version
    for testing without Qudi installation.
    """
    
    def __init__(self, qudi_main_weakref=None, name=None, **kwargs):
        """
        Mock initialization for Base module.
        
        @param qudi_main_weakref: Weakref to Qudi main object
        @param name: Unique name for this module instance
        """
        self._module_state = ModuleState()
        self.module_state = self._module_state
        self.config = kwargs.get('config', {})
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


class ConfigOption:
    """
    Mock implementation of Qudi ConfigOption class.
    """
    
    def __init__(self, name, default=None, missing='error'):
        """
        Initialize a ConfigOption.
        
        @param name: str, option name in config
        @param default: default value if option is missing
        @param missing: behavior if option is missing ('error', 'warn', 'info', 'nothing')
        """
        self.name = name
        self.default = default
        self.missing = missing
        
    def __get__(self, instance, owner=None):
        """Descriptor get implementation."""
        if instance is None:
            return self
            
        if hasattr(instance, 'config') and self.name in instance.config:
            return instance.config[self.name]
            
        if self.missing == 'error':
            raise ValueError(f"Required config option '{self.name}' missing")
        elif self.missing == 'warn':
            instance.log.warning(f"Config option '{self.name}' missing, using default: {self.default}")
        elif self.missing == 'info':
            instance.log.info(f"Config option '{self.name}' missing, using default: {self.default}")
            
        return self.default


class Connector:
    """
    Mock implementation of Qudi Connector class.
    """
    
    def __init__(self, interface, name=None, optional=False):
        """
        Initialize a Connector.
        
        @param interface: str, interface name to connect to
        @param name: str, optional connector name
        @param optional: bool, whether connection is optional
        """
        self.interface = interface
        self.name = name or interface
        self.optional = optional
        self._module = None
        
    def __get__(self, instance, owner=None):
        """Descriptor get implementation."""
        if instance is None:
            return self
            
        # Create a callable that returns the connected module
        def _connector_method():
            if self._module is None and not self.optional:
                raise RuntimeError(f"Connector '{self.name}' not connected")
            return self._module
            
        return _connector_method
        
    def __set__(self, instance, value):
        """Descriptor set implementation."""
        self._module = value