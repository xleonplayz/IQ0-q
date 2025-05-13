# -*- coding: utf-8 -*-

"""
This module acts like a laser with integration to the NV simulator.

Copyright (c) 2021, the qudi developers. See the AUTHORS.md file at the top-level directory of this
distribution and on <https://github.com/Ulm-IQO/qudi-iqo-modules/>

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

import math
import time
import random

from qudi.core.configoption import ConfigOption
from qudi.interface.simple_laser_interface import SimpleLaserInterface
from qudi.interface.simple_laser_interface import LaserState, ShutterState, ControlMode
from qudi.util.mutex import RecursiveMutex


class SimpleLaserDummy(SimpleLaserInterface):
    """ Laser dummy with integrated NV simulator support
    
    This module integrates with the NV center simulator to provide physically
    accurate control of the laser for simulated NV experiments.

    Example config for copy-paste:

    laser_dummy:
        module.Class: 'laser.simple_laser_dummy.SimpleLaserDummy'
        options:
            use_simulator: True  # Whether to use the NV center simulator
    """
    
    # Config options
    _use_simulator = ConfigOption('use_simulator', True)

    def __init__(self, **kwargs):
        """ Initialize the laser dummy """
        super().__init__(**kwargs)
        self._thread_lock = RecursiveMutex()
        self.lstate = LaserState.OFF
        self.shutter = ShutterState.CLOSED
        self.mode = ControlMode.POWER
        self.current_setpoint = 0
        self.power_setpoint = 0
        
        # Simulator state
        self._simulator_available = False
        self._simulator_manager = None

    def on_activate(self):
        """ Activate module.
        """
        # Try to connect to the simulator if configured to use it
        if self._use_simulator:
            try:
                from qudi.hardware.nv_simulator.simulator_manager import SimulatorManager
                self._simulator_manager = SimulatorManager()
                self._simulator_manager.register_module('simple_laser_dummy')
                self._simulator_available = True
                self.log.info("Successfully connected to NV simulator for laser control")
            except Exception as e:
                self.log.warning(f"Could not connect to NV simulator: {str(e)}. "
                                f"Using fallback dummy behavior instead.")
                self._simulator_available = False
        else:
            self._simulator_available = False

    def on_deactivate(self):
        """ Deactivate module.
        """
        # Turn off laser
        try:
            if self.lstate == LaserState.ON:
                self.off()
        except:
            pass
            
        # Unregister from simulator if we were using it
        if self._simulator_available and self._simulator_manager is not None:
            try:
                # Make sure laser is turned off in the simulator
                self._simulator_manager.apply_laser(0.0, False)
                self._simulator_manager.unregister_module('simple_laser_dummy')
            except:
                pass

    def get_power_range(self):
        """ Return optical power range

        @return float[2]: power range (min, max)
        """
        return 0, 0.250

    def get_power(self):
        """ Return laser power

        @return float: Laser power in watts
        """
        with self._thread_lock:
            return self.power_setpoint * random.gauss(1, 0.01)

    def get_power_setpoint(self):
        """ Return optical power setpoint.

        @return float: power setpoint in watts
        """
        with self._thread_lock:
            return self.power_setpoint

    def set_power(self, power):
        """ Set power setpoint.

        @param float power: power to set
        """
        with self._thread_lock:
            self.power_setpoint = power
            self.current_setpoint = math.sqrt(4*self.power_setpoint)*100
            
            # Update simulator if available and laser is on
            if self._simulator_available and self._simulator_manager is not None and self.lstate == LaserState.ON:
                try:
                    # Convert power to normalized value (0-1) for simulator
                    # Assuming max power is 0.25W
                    norm_power = min(1.0, power / 0.25)
                    self._simulator_manager.ping('simple_laser_dummy')
                    self._simulator_manager.apply_laser(norm_power, True)
                except Exception as e:
                    self.log.debug(f"Could not update simulator laser power: {str(e)}")

    def get_current_unit(self):
        """ Get unit for laser current.

        @return str: unit
        """
        return '%'

    def get_current_range(self):
        """ Get laser current range.

        @return float[2]: laser current range
        """
        return 0, 100

    def get_current(self):
        """ Get actual laser current

        @return float: laser current in current units
        """
        with self._thread_lock:
            return self.current_setpoint * random.gauss(1, 0.05)

    def get_current_setpoint(self):
        """ Get laser current setpoint

        @return float: laser current setpoint
        """
        with self._thread_lock:
            return self.current_setpoint

    def set_current(self, current):
        """ Set laser current setpoint

        @param float current: desired laser current setpoint
        """
        with self._thread_lock:
            self.current_setpoint = current
            self.power_setpoint = math.pow(self.current_setpoint/100, 2) / 4
            
            # Update simulator if available and laser is on
            if self._simulator_available and self._simulator_manager is not None and self.lstate == LaserState.ON:
                try:
                    # Convert current to normalized power (0-1) for simulator
                    # Current is 0-100%, power derived from it
                    norm_power = min(1.0, self.power_setpoint / 0.25)
                    self._simulator_manager.ping('simple_laser_dummy')
                    self._simulator_manager.apply_laser(norm_power, True)
                except Exception as e:
                    self.log.debug(f"Could not update simulator laser power: {str(e)}")

    def allowed_control_modes(self):
        """ Get supported control modes

        @return frozenset: set of supported ControlMode enums
        """
        return frozenset({ControlMode.POWER, ControlMode.CURRENT})

    def get_control_mode(self):
        """ Get the currently active control mode

        @return ControlMode: active control mode enum
        """
        with self._thread_lock:
            return self.mode

    def set_control_mode(self, control_mode):
        """ Set the active control mode

        @param ControlMode control_mode: desired control mode enum
        """
        with self._thread_lock:
            self.mode = control_mode

    def on(self):
        """ Turn on laser.

            @return LaserState: actual laser state
        """
        with self._thread_lock:
            time.sleep(1)
            self.lstate = LaserState.ON
            
            # Update simulator if available
            if self._simulator_available and self._simulator_manager is not None:
                try:
                    # Convert power to normalized value (0-1) for simulator
                    norm_power = min(1.0, self.power_setpoint / 0.25)
                    self._simulator_manager.ping('simple_laser_dummy')
                    self._simulator_manager.apply_laser(norm_power, True)
                except Exception as e:
                    self.log.warning(f"Could not turn on simulator laser: {str(e)}")
                    
            return self.lstate

    def off(self):
        """ Turn off laser.

            @return LaserState: actual laser state
        """
        with self._thread_lock:
            time.sleep(1)
            self.lstate = LaserState.OFF
            
            # Update simulator if available
            if self._simulator_available and self._simulator_manager is not None:
                try:
                    self._simulator_manager.ping('simple_laser_dummy')
                    self._simulator_manager.apply_laser(0.0, False)
                except Exception as e:
                    self.log.debug(f"Could not turn off simulator laser: {str(e)}")
                    
            return self.lstate

    def get_laser_state(self):
        """ Get laser state

        @return LaserState: current laser state
        """
        with self._thread_lock:
            return self.lstate

    def set_laser_state(self, state):
        """ Set laser state.

        @param LaserState state: desired laser state enum
        """
        with self._thread_lock:
            time.sleep(1)
            prev_state = self.lstate
            self.lstate = state
            
            # Update simulator if available and state changed
            if self._simulator_available and self._simulator_manager is not None and prev_state != state:
                try:
                    self._simulator_manager.ping('simple_laser_dummy')
                    if state == LaserState.ON:
                        # Convert power to normalized value (0-1) for simulator
                        norm_power = min(1.0, self.power_setpoint / 0.25)
                        self._simulator_manager.apply_laser(norm_power, True)
                    else:
                        self._simulator_manager.apply_laser(0.0, False)
                except Exception as e:
                    self.log.debug(f"Could not update simulator laser state: {str(e)}")
                    
            return self.lstate

    def get_shutter_state(self):
        """ Get laser shutter state

        @return ShutterState: actual laser shutter state
        """
        with self._thread_lock:
            return self.shutter

    def set_shutter_state(self, state):
        """ Set laser shutter state.

        @param ShutterState state: desired laser shutter state
        """
        with self._thread_lock:
            time.sleep(1)
            prev_state = self.shutter
            self.shutter = state
            
            # Update simulator if available, shutter state changed, and laser is on
            if (self._simulator_available and self._simulator_manager is not None and 
                prev_state != state and self.lstate == LaserState.ON):
                try:
                    self._simulator_manager.ping('simple_laser_dummy')
                    if state == ShutterState.OPEN:
                        # Convert power to normalized value (0-1) for simulator
                        norm_power = min(1.0, self.power_setpoint / 0.25)
                        self._simulator_manager.apply_laser(norm_power, True)
                    else:
                        # Closed shutter means no light
                        self._simulator_manager.apply_laser(0.0, False)
                except Exception as e:
                    self.log.debug(f"Could not update simulator shutter state: {str(e)}")
                    
            return self.shutter

    def get_temperatures(self):
        """ Get all available temperatures.

        @return dict: dict of temperature names and value in degrees Celsius
        """
        return {
            'psu': 32.2 * random.gauss(1, 0.1),
            'head': 42.0 * random.gauss(1, 0.2)
        }

    def get_extra_info(self):
        """ Multiple lines of diagnostic information

            @return str: much laser, very useful
        """
        info = "Dummy laser v1.0.0\n"
        if self._simulator_available:
            info += "Integrated with NV simulator\n"
        info += "Price reasonable, quality excellent"
        return info

