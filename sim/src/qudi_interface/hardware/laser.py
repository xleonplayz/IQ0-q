# -*- coding: utf-8 -*-

"""
Qudi hardware interface adapter for NV simulator laser control.
This module implements the SimpleLaserInterface for the NV center simulator,
enabling laser control for optical excitation.

Copyright (c) 2023
"""

import time
from typing import List, Tuple, Dict, Any, Optional

from qudi.interface.simple_laser_interface import SimpleLaserInterface
from qudi.interface.simple_laser_interface import ControlMode, ShutterState, LaserState

from .qudi_facade import QudiFacade


class NVSimLaser(SimpleLaserInterface):
    """
    Hardware adapter that implements the SimpleLaserInterface for the NV center simulator.
    This interface enables laser control for optical excitation.
    """

    def __init__(self, config=None, **kwargs):
        """
        Initialize the laser adapter for the NV simulator.
        
        @param config: Configuration dictionary
        @param **kwargs: Additional keyword arguments for the base class
        """
        # Initialize the module base class
        super().__init__(config=config, **kwargs)
        
        # Get the Qudi facade instance
        self._qudi_facade = QudiFacade(config)
        self._simulator = self._qudi_facade.get_nv_model()
        self._laser_controller = self._qudi_facade.get_laser_controller()
        
        # Parse configuration
        self._config = self.config
        
        # Default laser settings
        self._laser_power = 0.0  # Power in mW
        self._laser_power_setpoint = 0.0  # Setpoint in mW
        self._power_range = (0.0, 100.0)  # Power range in mW
        
        # Update from config if provided
        if 'power_range' in self._config:
            self._power_range = self._config['power_range']
            
        if 'initial_power' in self._config:
            self._laser_power = self._config['initial_power']
            self._laser_power_setpoint = self._laser_power
        
        # State variables
        self._laser_state = LaserState.OFF
        self._control_mode = ControlMode.POWER  # Only power control is supported
        self._shutter_state = ShutterState.CLOSED  # Start with closed shutter
        
        # Thread lock for thread safety
        self._thread_lock = self.module_state.lock_access()
        
        self.log.info("NV Simulator laser initialized")

    def on_activate(self):
        """
        Called when module is activated
        """
        self.log.info('NV Simulator laser activated')
        
        # Initialize laser controller
        if self._laser_controller is not None:
            # Set initial power
            power_norm = self._laser_power / self._power_range[1]  # Normalize power to 0-1 range
            self._laser_controller.set_power(power_norm)
            
            # Ensure laser is off
            self._laser_controller.off()
            self._shutter_state = ShutterState.CLOSED
            self._laser_state = LaserState.OFF
        else:
            self.log.warning("Laser controller not available from QudiFacade")

    def on_deactivate(self):
        """
        Called when module is deactivated
        """
        # Turn off laser
        self.off()
        self.log.info('NV Simulator laser deactivated')

    def get_power_range(self):
        """ 
        Return laser power range
        
        @return float[2]: power range (min, max) in mW
        """
        return self._power_range

    def get_power(self):
        """ 
        Return actual laser power
        
        @return float: Laser power in mW
        """
        # For the simulator, we just return the set power if the laser is on
        if self._laser_state == LaserState.ON:
            return self._laser_power
        else:
            return 0.0

    def set_power(self, power):
        """ 
        Set power setpoint.
        
        @param float power: power to set in mW
        """
        with self._thread_lock:
            # Ensure power is within range
            if power < self._power_range[0]:
                power = self._power_range[0]
            elif power > self._power_range[1]:
                power = self._power_range[1]
                
            # Store the power setpoint
            self._laser_power_setpoint = power
            
            # If laser is on, apply the new power
            if self._laser_state == LaserState.ON:
                self._laser_power = self._laser_power_setpoint
                
                # Apply to simulator if laser controller is available
                if self._laser_controller is not None:
                    # Normalize power to 0-1 range
                    power_norm = self._laser_power / self._power_range[1]
                    self._laser_controller.set_power(power_norm)
                    
            self.log.debug(f"Laser power set to {power} mW")

    def get_power_setpoint(self):
        """ 
        Return laser power setpoint.
        
        @return float: power setpoint in mW
        """
        return self._laser_power_setpoint

    def get_current_unit(self):
        """ 
        Return the units of current for this laser.
        
        @return str: unit
        """
        return "A"  # Ampere

    def get_current_range(self):
        """ 
        Return laser current range.
        
        @return float[2]: current range (min, max) in appropriate current units
        """
        # Current control is not implemented for the simulator
        # We return a dummy range
        return (0.0, 1.0)

    def get_current(self):
        """ 
        Return actual laser current.
        
        @return float: Laser current in appropriate current units
        """
        # Simulate a direct relationship between power and current
        # For the simulator, we don't have actual current control
        if self._laser_state == LaserState.ON:
            # Calculate from power: 1A at max power
            return self._laser_power / self._power_range[1]
        else:
            return 0.0

    def get_current_setpoint(self):
        """ 
        Return laser current setpoint.
        
        @return float: Laser current setpoint in appropriate current units
        """
        # Calculate from power setpoint: 1A at max power
        return self._laser_power_setpoint / self._power_range[1]

    def set_current(self, current):
        """ 
        Set current setpoint.
        
        @param float current: current setpoint
        """
        # Convert current to power and set power
        power = current * self._power_range[1]
        self.set_power(power)

    def get_shutter_state(self):
        """ 
        Return shutter state.
        
        @return ShutterState: shutter state enum
        """
        return self._shutter_state

    def set_shutter_state(self, state):
        """ 
        Set shutter state.
        
        @param ShutterState state: desired state
        @return ShutterState: actual state
        """
        with self._thread_lock:
            if state == ShutterState.OPEN:
                # Open the shutter (turn on laser)
                if self._laser_controller is not None:
                    power_norm = self._laser_power / self._power_range[1]
                    self._laser_controller.set_power(power_norm)
                    self._laser_controller.on()
                
                self._shutter_state = ShutterState.OPEN
                self._laser_state = LaserState.ON
                self._laser_power = self._laser_power_setpoint
                self.log.debug("Laser shutter opened")
                
            elif state == ShutterState.CLOSED:
                # Close the shutter (turn off laser)
                if self._laser_controller is not None:
                    self._laser_controller.off()
                
                self._shutter_state = ShutterState.CLOSED
                self._laser_state = LaserState.OFF
                self.log.debug("Laser shutter closed")
                
            return self._shutter_state

    def get_laser_state(self):
        """ 
        Return laser state.
        
        @return LaserState: laser state enum
        """
        return self._laser_state

    def set_laser_state(self, state):
        """ 
        Set laser state.
        
        @param LaserState state: desired state
        @return LaserState: actual state
        """
        with self._thread_lock:
            if state == LaserState.ON:
                # Turn on the laser
                if self._laser_controller is not None:
                    power_norm = self._laser_power_setpoint / self._power_range[1]
                    self._laser_controller.set_power(power_norm)
                    self._laser_controller.on()
                
                self._laser_state = LaserState.ON
                self._shutter_state = ShutterState.OPEN
                self._laser_power = self._laser_power_setpoint
                self.log.debug(f"Laser turned ON with power {self._laser_power} mW")
                
            elif state == LaserState.OFF:
                # Turn off the laser
                if self._laser_controller is not None:
                    self._laser_controller.off()
                
                self._laser_state = LaserState.OFF
                self._shutter_state = ShutterState.CLOSED
                self.log.debug("Laser turned OFF")
                
            # LOCKED state is not supported in the simulator
            
            return self._laser_state

    def get_control_mode(self):
        """ 
        Return the currently active control mode of the laser
        
        @return ControlMode: control mode enum
        """
        return self._control_mode

    def set_control_mode(self, mode):
        """ 
        Change the control mode of the laser
        
        @param ControlMode mode: desired control mode enum
        @return ControlMode: actual control mode enum
        """
        # Only power control is supported in the simulator
        if mode == ControlMode.POWER:
            self._control_mode = mode
        else:
            self.log.warning(f"Control mode {mode} not supported, staying in POWER mode")
            
        return self._control_mode

    def get_temperatures(self):
        """ 
        Get all available temperatures from the laser.
        
        @return dict: dict of temperature name and value pairs
        """
        # Simulate a temperature 
        return {'base': 25.0, 'diode': 28.5}
    
    def get_temperature_setpoints(self):
        """ 
        Get all temperature setpoints from the laser.
        
        @return dict: dict of temperature name and setpoint pairs
        """
        # Simulated temperature setpoints
        return {'base': 25.0, 'diode': 28.0}
    
    def set_temperatures(self, temps):
        """ 
        Set laser temperatures.
        
        @param dict temps: dict of temperature name and setpoint pairs
        @return dict: dict of temperature name and actual setpoint pairs
        """
        # Temperatures are not settable in this simulator
        self.log.warning("Temperature control not implemented in simulator")
        return self.get_temperature_setpoints()

    def get_laser_extra_info(self):
        """ 
        Get extra information from laser.
        
        @return dict: dict with keys 'extra_info' and 'wavelength'
        """
        # Return some simulated extra info and wavelength
        return {
            'extra_info': 'NV Simulator Laser',
            'wavelength': 532.0  # in nm
        }

    def on(self):
        """ 
        Turn on laser.
        
        @return LaserState: actual laser state
        """
        return self.set_laser_state(LaserState.ON)

    def off(self):
        """ 
        Turn off laser.
        
        @return LaserState: actual laser state
        """
        return self.set_laser_state(LaserState.OFF)