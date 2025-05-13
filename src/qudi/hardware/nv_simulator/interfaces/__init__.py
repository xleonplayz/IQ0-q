# -*- coding: utf-8 -*-

"""
This file contains interface definitions for the NV simulator integration.

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

from abc import ABC, abstractmethod

class SimulatorInterface(ABC):
    """
    Interface for accessing the NV simulator core functionality.
    
    This interface defines the methods that the simulator manager must implement
    in order to be compatible with the Qudi dummy hardware modules.
    """
    
    @abstractmethod
    def register_module(self, module_name):
        """
        Register a module as active with the simulator.
        
        @param str module_name: Name of the module to register
        """
        pass
    
    @abstractmethod
    def unregister_module(self, module_name):
        """
        Unregister a module from the simulator.
        
        @param str module_name: Name of the module to unregister
        """
        pass
    
    @abstractmethod
    def ping(self, module_name=None):
        """
        Update the watchdog timer for a module, or check simulator health.
        
        @param str module_name: Optional name of module to update watchdog
        @return bool: True if simulator is healthy
        """
        pass
    
    @abstractmethod
    def apply_magnetic_field(self, field_vector):
        """
        Set the magnetic field vector.
        
        @param field_vector: [Bx, By, Bz] in Gauss
        @return bool: Success or failure
        """
        pass
    
    @abstractmethod
    def apply_laser(self, power, on=True):
        """
        Control the laser for optical excitation.
        
        @param power: Laser power in normalized units (0.0-1.0)
        @param on: Bool whether laser is on/off
        @return bool: Success or failure
        """
        pass
    
    @abstractmethod
    def apply_microwave(self, frequency, power_dbm, on=True):
        """
        Control the microwave excitation.
        
        @param frequency: Microwave frequency in Hz
        @param power_dbm: Microwave power in dBm
        @param on: Bool whether microwave is on/off
        @return bool: Success or failure
        """
        pass
    
    @abstractmethod
    def get_fluorescence(self):
        """
        Get the current fluorescence signal.
        
        @return float: Fluorescence count rate in counts/s
        """
        pass
    
    @abstractmethod
    def simulate_odmr(self, f_min, f_max, n_points, mw_power=-10.0):
        """
        Simulate an ODMR experiment.
        
        @param f_min: Start frequency in Hz
        @param f_max: End frequency in Hz
        @param n_points: Number of frequency points
        @param mw_power: Microwave power in dBm
        
        @return: Dictionary with frequencies and signal
        """
        pass
    
    @abstractmethod
    def set_position(self, x, y, z):
        """
        Set the position for scanning probe.
        
        @param x: X position in meters
        @param y: Y position in meters
        @param z: Z position in meters
        @return bool: Success or failure
        """
        pass
    
    @abstractmethod
    def get_confocal_image(self, x_range, y_range, z_position, resolution):
        """
        Generate a confocal image based on current settings.
        
        @param x_range: (min, max) for x axis in meters
        @param y_range: (min, max) for y axis in meters
        @param z_position: Z position in meters
        @param resolution: Number of pixels per dimension
        
        @return: 2D array of confocal image data
        """
        pass