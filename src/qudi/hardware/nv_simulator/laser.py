# -*- coding: utf-8 -*-

"""
This file contains the Qudi hardware module for the NV simulator laser.

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

import time
import os
import sys
from qudi.interface.simple_laser_interface import SimpleLaserInterface
from qudi.interface.simple_laser_interface import LaserState, ShutterState, ControlMode
from qudi.core.configoption import ConfigOption
from qudi.core.connector import Connector
from qudi.util.mutex import Mutex

# Import QudiFacade directly from current directory to avoid circular imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from qudi_facade import QudiFacade


class NVSimLaser(SimpleLaserInterface):
    """Laser control implementation for the NV simulator.
    
    This module provides laser control capabilities for the NV center simulator.
    
    Example config for copy-paste:

    nv_sim_laser:
        module.Class: 'nv_simulator.laser.NVSimLaser'
        options:
            wavelength: 532  # Laser wavelength in nm
            max_power: 100.0  # Maximum laser power in mW
            power_noise: 0.01  # Relative power noise
    """

    # Config options
    _wavelength = ConfigOption('wavelength', default=532, missing='warn')  # in nm
    _max_power = ConfigOption('max_power', default=100.0, missing='warn')  # in mW
    _power_noise = ConfigOption('power_noise', default=0.01, missing='warn')  # relative

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._thread_lock = Mutex()
        self._laser_state = LaserState.OFF
        self._shutter_state = ShutterState.CLOSED
        self._control_mode = ControlMode.POWER
        self._current_setpoint = 0  # in %
        self._power_setpoint = 0  # in mW
        self._temperature = 25.0  # in °C

    def on_activate(self):
        """Activate module."""
        # Get QudiFacade from connector
        self._qudi_facade = self.simulator()
        
        self.log.info('NV Simulator Laser initialized')

    def on_deactivate(self):
        """Deactivate module."""
        if self._laser_state == LaserState.ON:
            self.off()

    def get_current_power(self):
        """Get the actual laser output power.

        @return float: Actual laser power in mW
        """
        with self._thread_lock:
            if self._laser_state == LaserState.OFF or self._shutter_state == ShutterState.CLOSED:
                return 0.0
                
            # In POWER mode, return the setpoint with some noise
            if self._control_mode == ControlMode.POWER:
                power = self._power_setpoint
            # In CURRENT mode, calculate power from current (linear approximation)
            else:
                power = self._current_setpoint * self._max_power / 100.0
                
            # Add some noise to the reading
            power_with_noise = power * (1.0 + self._power_noise * (2.0 * (0.5 - time.time() % 1.0)))
            return power_with_noise

    def get_current_unit(self):
        """Get the current laser power unit.

        @return str: Unit of laser power
        """
        return 'mW'

    def get_laser_state(self):
        """Get the laser state.

        @return LaserState: Current laser state
        """
        return self._laser_state

    def get_shutter_state(self):
        """Get the laser shutter state.

        @return ShutterState: Current laser shutter state
        """
        return self._shutter_state

    def get_temperatures(self):
        """Get all available temperatures.

        @return dict: Dict of temperature names and values
        """
        return {'laser': self._temperature}

    def get_temperature_setpoints(self):
        """Get all available temperature setpoints.

        @return dict: Dict of temperature name and setpoint value
        """
        return {'laser': self._temperature}

    def get_control_mode(self):
        """Get the currently active control mode.

        @return ControlMode: Current control mode
        """
        return self._control_mode

    def get_power_range(self):
        """Get the laser power range.

        @return tuple(float, float): Laser power range in mW
        """
        return 0.0, self._max_power

    def get_current_range(self):
        """Get the laser current range.

        @return tuple(float, float): Laser current range in %
        """
        return 0.0, 100.0

    def get_power_setpoint(self):
        """Get the laser power setpoint.

        @return float: Laser power setpoint in mW
        """
        return self._power_setpoint

    def get_current_setpoint(self):
        """Get the laser current setpoint.

        @return float: Laser current setpoint in %
        """
        return self._current_setpoint

    def get_extra_info(self):
        """Get extra information about the laser.

        @return str: Extra information about laser
        """
        extra = f"NV Simulator Laser at {self._wavelength} nm, max power: {self._max_power} mW"
        return extra

    def set_power(self, power):
        """Set the laser power.

        @param float power: Laser power setpoint in mW

        @return float: Actually set laser power in mW
        """
        with self._thread_lock:
            # Constrain power to valid range
            min_power, max_power = self.get_power_range()
            if power < min_power:
                power = min_power
            elif power > max_power:
                power = max_power
                
            self._power_setpoint = power
            
            # Set control mode to POWER
            self._control_mode = ControlMode.POWER
            
            # Calculate corresponding current for display
            self._current_setpoint = power / max_power * 100.0
            
            # Apply to simulator if laser is on and shutter is open
            if self._laser_state == LaserState.ON and self._shutter_state == ShutterState.OPEN:
                self._qudi_facade.laser_controller.set_power(power)
                
            return self._power_setpoint

    def set_current(self, current):
        """Set the laser current.

        @param float current: Laser current setpoint in %

        @return float: Actually set laser current in %
        """
        with self._thread_lock:
            # Constrain current to valid range
            min_current, max_current = self.get_current_range()
            if current < min_current:
                current = min_current
            elif current > max_current:
                current = max_current
                
            self._current_setpoint = current
            
            # Set control mode to CURRENT
            self._control_mode = ControlMode.CURRENT
            
            # Calculate corresponding power
            self._power_setpoint = current / 100.0 * self._max_power
            
            # Apply to simulator if laser is on and shutter is open
            if self._laser_state == LaserState.ON and self._shutter_state == ShutterState.OPEN:
                self._qudi_facade.laser_controller.set_power(self._power_setpoint)
                
            return self._current_setpoint

    def on(self):
        """Turn the laser on.

        @return LaserState: Actually set laser state
        """
        with self._thread_lock:
            self._laser_state = LaserState.ON
            
            # Activate laser in simulator if shutter is open
            if self._shutter_state == ShutterState.OPEN:
                self._qudi_facade.laser_controller.set_power(self._power_setpoint)
                self._qudi_facade.laser_controller.on()
                
            return self._laser_state

    def off(self):
        """Turn the laser off.

        @return LaserState: Actually set laser state
        """
        with self._thread_lock:
            self._laser_state = LaserState.OFF
            
            # Turn off laser in simulator
            self._qudi_facade.laser_controller.off()
            
            return self._laser_state

    def open_shutter(self):
        """Open the laser shutter.

        @return ShutterState: Actually set shutter state
        """
        with self._thread_lock:
            self._shutter_state = ShutterState.OPEN
            
            # Activate laser in simulator if it's on
            if self._laser_state == LaserState.ON:
                self._qudi_facade.laser_controller.set_power(self._power_setpoint)
                self._qudi_facade.laser_controller.on()
                
            return self._shutter_state

    def close_shutter(self):
        """Close the laser shutter.

        @return ShutterState: Actually set shutter state
        """
        with self._thread_lock:
            self._shutter_state = ShutterState.CLOSED
            
            # Turn off laser in simulator
            self._qudi_facade.laser_controller.off()
            
            return self._shutter_state

    def set_temperatures(self, temps):
        """Set laser temperatures.

        @param dict temps: Dict of temperature name and setpoint value in degrees Celsius

        @return dict: Dict of temperature name and actually set temperature value
        """
        with self._thread_lock:
            result = {}
            
            for name, value in temps.items():
                if name == 'laser':
                    self._temperature = value
                    result[name] = value
                    
            return result

    def get_wavelength(self):
        """Get the laser wavelength.

        @return float: Laser wavelength in nm
        """
        return self._wavelength

    def set_wavelength(self, wavelength):
        """Set the laser wavelength.

        @param float wavelength: Laser wavelength in nm

        @return float: Actually set wavelength in nm
        """
        # We don't actually set the wavelength as it's fixed for this laser
        return self._wavelength