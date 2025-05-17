"""
This module provides integration adapters for Qudi-IQO modules.

These adapters allow the Qudi-IQO dummy modules to use the NV simulator API
instead of generating their own data.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Union, Any
import time
import os
import sys

# Add package to Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
package_dir = os.path.dirname(os.path.dirname(script_dir))
if package_dir not in sys.path:
    sys.path.insert(0, package_dir)

from nv.api.client import NVSimulatorClient

logger = logging.getLogger(__name__)

# Check if we can import Qudi interfaces
QUDI_AVAILABLE = False
try:
    from qudi.interface.microwave_interface import MicrowaveInterface
    from qudi.interface.microwave_interface import MicrowaveMode
    from qudi.interface.microwave_interface import MicrowaveConstraints
    from qudi.interface.microwave_interface import TriggerEdge
    from qudi.interface.fast_counter_interface import FastCounterInterface
    from qudi.interface.simple_laser_interface import SimpleLaserInterface
    from qudi.interface.simple_laser_interface import ShutterState
    from qudi.interface.simple_laser_interface import ControlMode
    from qudi.interface.simple_laser_interface import LaserState
    QUDI_AVAILABLE = True
except ImportError:
    logger.warning("Qudi interfaces not available. Using mock interfaces.")
    # Create mock classes to allow the module to be imported
    class MicrowaveInterface:
        pass
    class FastCounterInterface:
        pass
    class SimpleLaserInterface:
        pass
    # Enums
    class MicrowaveMode:
        CW = 'cw'
        SWEEP = 'sweep'
    class TriggerEdge:
        RISING = 0
        FALLING = 1
    class ShutterState:
        CLOSED = 'closed'
        OPEN = 'open'
        NOSHUTTER = 'noshutter'
    class ControlMode:
        POWER = 'power'
        CURRENT = 'current'
    class LaserState:
        ON = 'on'
        OFF = 'off'
        LOCKED = 'locked'
    class MicrowaveConstraints:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

# Constants for conversion
MW_API_TO_QUDI_STATE = {
    'idle': 'off',
    'locked': 'on',
}

LASER_API_TO_QUDI_STATE = {
    'OFF': LaserState.OFF,
    'ON': LaserState.ON,
    'LOCKED': LaserState.LOCKED
}

QUDI_TO_API_LASER_STATE = {
    LaserState.OFF: 'OFF',
    LaserState.ON: 'ON',
    LaserState.LOCKED: 'LOCKED'
}

QUDI_TO_API_SHUTTER_STATE = {
    ShutterState.CLOSED: 'CLOSED',
    ShutterState.OPEN: 'OPEN',
    ShutterState.NOSHUTTER: 'NO_SHUTTER'
}

API_TO_QUDI_SHUTTER_STATE = {
    'CLOSED': ShutterState.CLOSED,
    'OPEN': ShutterState.OPEN,
    'NO_SHUTTER': ShutterState.NOSHUTTER
}

QUDI_TO_API_CONTROL_MODE = {
    ControlMode.POWER: 'POWER',
    ControlMode.CURRENT: 'CURRENT'
}

API_TO_QUDI_CONTROL_MODE = {
    'POWER': ControlMode.POWER,
    'CURRENT': ControlMode.CURRENT
}

class MicrowaveSimulatorAdapter(MicrowaveInterface):
    """Adapter for Qudi microwave interface that uses the NV simulator API."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the adapter."""
        super().__init__(*args, **kwargs)
        
        # Initialize client
        self.api_client = NVSimulatorClient()
        
        # Initialize internal state
        self._mode = MicrowaveMode.CW
        self._frequency_sweep_start = 2.8e9
        self._frequency_sweep_stop = 2.9e9
        self._frequency_sweep_points = 101
        self._frequency_sweep_list = None
        self._mw_power = 0.0
        self._mw_frequency = 2.87e9
        self._scan_power = 0.0
        self._use_ext_trigger = False
        self._trigger_in_edge = TriggerEdge.RISING
        self._trigger_out_edge = TriggerEdge.RISING
        self._active = False
        self._scan_active = False
        
        # Get constraints from API
        constraints = self.api_client.get_microwave_constraints()
        self._constraints = MicrowaveConstraints(
            power_limits=constraints['power_limits'],
            frequency_limits=constraints['frequency_limits'], 
            scan_size_limits=constraints['scan_size_limits'],
            sample_rate_limits=constraints['sample_rate_limits'],
            scan_modes=[MicrowaveMode.CW, MicrowaveMode.SWEEP]
        )
    
    def on_activate(self):
        """Activate the module."""
        # Refresh state from API
        self._update_state_from_api()
    
    def on_deactivate(self):
        """Deactivate the module."""
        # Turn off microwave if active
        if self._active:
            self.off()
    
    def _update_state_from_api(self):
        """Update internal state from API."""
        try:
            status = self.api_client.get_microwave_status()
            self._active = status['module_state'] == 'locked'
            self._scan_active = status['is_scanning']
            self._mw_frequency = status['cw_frequency']
            self._mw_power = status['cw_power']
            self._scan_power = status['scan_power']
            
            if status['scan_frequencies'] is not None:
                if isinstance(status['scan_frequencies'], list):
                    if status['scan_mode'] == 'JUMP_LIST':
                        self._frequency_sweep_list = status['scan_frequencies']
                        self._mode = MicrowaveMode.SWEEP
                    elif len(status['scan_frequencies']) >= 2:
                        self._frequency_sweep_start = status['scan_frequencies'][0]
                        self._frequency_sweep_stop = status['scan_frequencies'][-1]
                        self._frequency_sweep_points = len(status['scan_frequencies'])
                        self._mode = MicrowaveMode.SWEEP
            else:
                self._mode = MicrowaveMode.CW
        except Exception as e:
            logger.error(f"Failed to update state from API: {e}")
    
    def get_constraints(self):
        """Return the hardware constraints."""
        return self._constraints
    
    def get_status(self):
        """Return the current status."""
        self._update_state_from_api()
        return self._active, self._scan_active
    
    def off(self):
        """Switch off the microwave source."""
        try:
            self.api_client.microwave_off()
            self._active = False
            self._scan_active = False
            return 0
        except Exception as e:
            logger.error(f"Error turning off microwave: {e}")
            return -1
    
    def get_power(self):
        """Return the current power."""
        return self._mw_power
    
    def get_frequency(self):
        """Return the current frequency."""
        return self._mw_frequency
    
    def cw_on(self):
        """Switch on cw microwave output."""
        try:
            self.api_client.cw_on()
            self._active = True
            self._scan_active = False
            self._mode = MicrowaveMode.CW
            return 0
        except Exception as e:
            logger.error(f"Error turning on CW microwave: {e}")
            return -1
    
    def set_cw(self, frequency, power):
        """Configure the CW microwave output."""
        try:
            self.api_client.set_cw(frequency, power)
            self._mw_frequency = frequency
            self._mw_power = power
            self._mode = MicrowaveMode.CW
            return 0
        except Exception as e:
            logger.error(f"Error setting CW parameters: {e}")
            return -1
    
    def configure_sweep(self, start, stop, points, power):
        """Configure a frequency sweep."""
        try:
            self.api_client.configure_scan(
                power=power,
                frequencies=[start, stop, points],
                mode="EQUIDISTANT_SWEEP"
            )
            self._frequency_sweep_start = start
            self._frequency_sweep_stop = stop
            self._frequency_sweep_points = points
            self._scan_power = power
            self._mode = MicrowaveMode.SWEEP
            self._frequency_sweep_list = None
            return 0
        except Exception as e:
            logger.error(f"Error configuring sweep: {e}")
            return -1
    
    def configure_frequency_sweep(self, freq_list, power):
        """Configure a frequency list sweep."""
        try:
            self.api_client.configure_scan(
                power=power,
                frequencies=freq_list,
                mode="JUMP_LIST"
            )
            self._frequency_sweep_list = freq_list
            self._scan_power = power
            self._mode = MicrowaveMode.SWEEP
            return 0
        except Exception as e:
            logger.error(f"Error configuring frequency sweep: {e}")
            return -1
    
    def set_sweep_parameters(self, start, stop, points):
        """Set sweep parameters."""
        self._frequency_sweep_start = start
        self._frequency_sweep_stop = stop
        self._frequency_sweep_points = points
        self._frequency_sweep_list = None
        return 0
    
    def reset_sweep(self):
        """Reset the sweep."""
        try:
            self.api_client.reset_scan()
            return 0
        except Exception as e:
            logger.error(f"Error resetting sweep: {e}")
            return -1
    
    def sweep_on(self):
        """Turn on the sweep."""
        try:
            self.api_client.start_scan()
            self._active = True
            self._scan_active = True
            return 0
        except Exception as e:
            logger.error(f"Error starting sweep: {e}")
            return -1
    
    def set_ext_trigger(self, trigger_in_edge, trigger_out_edge):
        """Set external trigger."""
        self._trigger_in_edge = trigger_in_edge
        self._trigger_out_edge = trigger_out_edge
        self._use_ext_trigger = True
        return 0
    
    def get_mode(self):
        """Return the current mode (cw or sweep)."""
        return self._mode


class FastCounterSimulatorAdapter(FastCounterInterface):
    """Adapter for Qudi fast counter interface that uses the NV simulator API."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the adapter."""
        super().__init__(*args, **kwargs)
        
        # Initialize client
        self.api_client = NVSimulatorClient()
        
        # Initialize internal state
        self.statusvar = 0  # 0=unconfigured, 1=idle, 2=running, 3=paused, -1=error
        self._binwidth = 1.0
        self._record_length = 3000
        self._number_of_gates = 0
        self._trace_data = None
    
    def on_activate(self):
        """Activate the module."""
        # Get constraints from API
        try:
            constraints = self.api_client.get_fast_counter_constraints()
            self._available_binwidths = constraints['hardware_binwidth_list']
            
            # Set default binwidth to smallest available
            self._binwidth = min(self._available_binwidths)
            self.statusvar = 1  # idle
        except Exception as e:
            logger.error(f"Error during activation: {e}")
            self.statusvar = -1  # error
    
    def on_deactivate(self):
        """Deactivate the module."""
        # Stop counter if running
        if self.statusvar == 2:  # running
            self.stop_measure()
    
    def get_constraints(self):
        """Return the hardware constraints."""
        return {'hardware_binwidth_list': self._available_binwidths}
    
    def configure(self, bin_width_s, record_length_s, number_of_gates=0):
        """Configure the fast counter."""
        try:
            # Configure through API
            response = self.api_client.configure_fast_counter(
                bin_width_s, record_length_s, number_of_gates
            )
            
            # Update internal state
            self._binwidth = response['binwidth_s']
            self._record_length = int(response['record_length_s'] / self._binwidth)
            self._number_of_gates = response['number_of_gates']
            self.statusvar = 1  # idle
            
            return response['binwidth_s'], response['record_length_s'], response['number_of_gates']
        except Exception as e:
            logger.error(f"Error configuring fast counter: {e}")
            self.statusvar = -1  # error
            return -1, -1, -1
    
    def get_status(self):
        """Return the current status."""
        try:
            status = self.api_client.get_fast_counter_status()
            self.statusvar = status['status']
            return self.statusvar
        except Exception as e:
            logger.error(f"Error getting status: {e}")
            self.statusvar = -1  # error
            return self.statusvar
    
    def start_measure(self):
        """Start the measurement."""
        try:
            self.api_client.start_fast_counter()
            self.statusvar = 2  # running
            return 0
        except Exception as e:
            logger.error(f"Error starting measurement: {e}")
            self.statusvar = -1  # error
            return -1
    
    def stop_measure(self):
        """Stop the measurement."""
        try:
            self.api_client.stop_fast_counter()
            self.statusvar = 1  # idle
            return 0
        except Exception as e:
            logger.error(f"Error stopping measurement: {e}")
            self.statusvar = -1  # error
            return -1
    
    def pause_measure(self):
        """Pause the measurement."""
        try:
            self.api_client.pause_fast_counter()
            self.statusvar = 3  # paused
            return 0
        except Exception as e:
            logger.error(f"Error pausing measurement: {e}")
            self.statusvar = -1  # error
            return -1
    
    def continue_measure(self):
        """Continue the paused measurement."""
        try:
            self.api_client.continue_fast_counter()
            self.statusvar = 2  # running
            return 0
        except Exception as e:
            logger.error(f"Error continuing measurement: {e}")
            self.statusvar = -1  # error
            return -1
    
    def get_data_trace(self):
        """Return the current timetrace."""
        try:
            data = self.api_client.get_fast_counter_data()
            
            # Store data
            self._trace_data = data['data']
            
            # Process data for Qudi interface
            if self._number_of_gates > 0:
                # Gated counter
                trace = np.array(data['data'])
            else:
                # Ungated counter
                trace = np.array(data['data'])
            
            # Return data with additional info
            return trace, {'elapsed_sweeps': None, 'elapsed_time': None}
        except Exception as e:
            logger.error(f"Error getting data trace: {e}")
            return np.zeros(self._record_length), {'elapsed_sweeps': None, 'elapsed_time': None}
    
    def get_timetrace(self):
        """Return the current timetrace and timeline."""
        data, info = self.get_data_trace()
        time_axis = np.arange(len(data)) * self._binwidth
        return data, time_axis, info


class SimpleLaserSimulatorAdapter(SimpleLaserInterface):
    """Adapter for Qudi simple laser interface that uses the NV simulator API."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the adapter."""
        super().__init__(*args, **kwargs)
        
        # Initialize client
        self.api_client = NVSimulatorClient()
        
        # Initialize internal state
        self.lstate = LaserState.OFF
        self.shutter = ShutterState.CLOSED
        self.mode = ControlMode.POWER
        self.current_setpoint = 0.0
        self.power_setpoint = 0.0
        self.current = 0.0
        self.power = 0.0
        self.temps = {}
    
    def on_activate(self):
        """Activate the module."""
        # Get constraints from API
        try:
            constraints = self.api_client.get_laser_constraints()
            self._power_range = constraints['power_range']
            self._current_range = constraints['current_range']
            self._current_unit = constraints['current_unit']
            
            # Update state from API
            self._update_state_from_api()
        except Exception as e:
            logger.error(f"Error during activation: {e}")
    
    def on_deactivate(self):
        """Deactivate the module."""
        # Turn off laser if on
        if self.lstate != LaserState.OFF:
            self.off()
    
    def _update_state_from_api(self):
        """Update internal state from API."""
        try:
            status = self.api_client.get_laser_status()
            self.lstate = LASER_API_TO_QUDI_STATE[status['laser_state']]
            self.shutter = API_TO_QUDI_SHUTTER_STATE[status['shutter_state']]
            self.mode = API_TO_QUDI_CONTROL_MODE[status['control_mode']]
            self.current = status['current']
            self.power = status['power']
            self.current_setpoint = status['current_setpoint']
            self.power_setpoint = status['power_setpoint']
            self.temps = status['temperatures']
        except Exception as e:
            logger.error(f"Failed to update state from API: {e}")
    
    def get_power_range(self):
        """Return the laser power range."""
        return self._power_range
    
    def get_current_range(self):
        """Return the laser current range."""
        return self._current_range
    
    def get_current_unit(self):
        """Return the unit for laser current."""
        return self._current_unit
    
    def get_laser_state(self):
        """Return the current laser state."""
        self._update_state_from_api()
        return self.lstate
    
    def get_shutter_state(self):
        """Return the current shutter state."""
        self._update_state_from_api()
        return self.shutter
    
    def get_temps(self):
        """Return the current temperatures."""
        try:
            temps = self.api_client.get_laser_temperatures()
            self.temps = temps['temperatures']
            return self.temps
        except Exception as e:
            logger.error(f"Error getting temperatures: {e}")
            return self.temps
    
    def get_power(self):
        """Return the current laser power."""
        self._update_state_from_api()
        return self.power
    
    def get_power_setpoint(self):
        """Return the current power setpoint."""
        self._update_state_from_api()
        return self.power_setpoint
    
    def get_current(self):
        """Return the current laser current."""
        self._update_state_from_api()
        return self.current
    
    def get_current_setpoint(self):
        """Return the current setpoint."""
        self._update_state_from_api()
        return self.current_setpoint
    
    def get_control_mode(self):
        """Return the current control mode."""
        self._update_state_from_api()
        return self.mode
    
    def set_power(self, power):
        """Set the laser power."""
        try:
            response = self.api_client.set_laser_power(power)
            self.power_setpoint = response['power_setpoint']
            self.current_setpoint = response['current_setpoint']
            return 0
        except Exception as e:
            logger.error(f"Error setting power: {e}")
            return -1
    
    def set_current(self, current):
        """Set the laser current."""
        try:
            response = self.api_client.set_laser_current(current)
            self.current_setpoint = response['current_setpoint']
            self.power_setpoint = response['power_setpoint']
            return 0
        except Exception as e:
            logger.error(f"Error setting current: {e}")
            return -1
    
    def set_control_mode(self, mode):
        """Set the control mode."""
        try:
            self.api_client.set_laser_control_mode(QUDI_TO_API_CONTROL_MODE[mode])
            self.mode = mode
            return 0
        except Exception as e:
            logger.error(f"Error setting control mode: {e}")
            return -1
    
    def on(self):
        """Turn on the laser."""
        try:
            self.api_client.set_laser_state("ON")
            self.lstate = LaserState.ON
            return 0
        except Exception as e:
            logger.error(f"Error turning on laser: {e}")
            return -1
    
    def off(self):
        """Turn off the laser."""
        try:
            self.api_client.set_laser_state("OFF")
            self.lstate = LaserState.OFF
            return 0
        except Exception as e:
            logger.error(f"Error turning off laser: {e}")
            return -1
    
    def open_shutter(self):
        """Open the shutter."""
        try:
            self.api_client.set_shutter_state("OPEN")
            self.shutter = ShutterState.OPEN
            return 0
        except Exception as e:
            logger.error(f"Error opening shutter: {e}")
            return -1
    
    def close_shutter(self):
        """Close the shutter."""
        try:
            self.api_client.set_shutter_state("CLOSED")
            self.shutter = ShutterState.CLOSED
            return 0
        except Exception as e:
            logger.error(f"Error closing shutter: {e}")
            return -1
    
    def allowed_control_modes(self):
        """Return the list of allowed control modes."""
        try:
            constraints = self.api_client.get_laser_constraints()
            return [API_TO_QUDI_CONTROL_MODE[mode] for mode in constraints['allowed_control_modes']]
        except Exception as e:
            logger.error(f"Error getting allowed control modes: {e}")
            return [ControlMode.POWER, ControlMode.CURRENT]  # Default fallback


if __name__ == "__main__":
    # Test code
    if QUDI_AVAILABLE:
        print("Testing Qudi adapters")
        
        # Test microwave adapter
        mw = MicrowaveSimulatorAdapter()
        mw.on_activate()
        print(f"Microwave constraints: {mw.get_constraints()}")
        print(f"Setting CW: {mw.set_cw(2.87e9, 0.0)}")
        print(f"CW on: {mw.cw_on()}")
        print(f"Current frequency: {mw.get_frequency()}")
        print(f"Current power: {mw.get_power()}")
        print(f"Current status: {mw.get_status()}")
        print(f"Turning off: {mw.off()}")
        
        # Test fast counter adapter
        fc = FastCounterSimulatorAdapter()
        fc.on_activate()
        print(f"Fast counter constraints: {fc.get_constraints()}")
        print(f"Configuring: {fc.configure(1e-9, 1e-6)}")
        print(f"Starting measurement: {fc.start_measure()}")
        print(f"Current status: {fc.get_status()}")
        print(f"Getting data: {fc.get_data_trace()[0].shape}")
        print(f"Stopping measurement: {fc.stop_measure()}")
        
        # Test simple laser adapter
        laser = SimpleLaserSimulatorAdapter()
        laser.on_activate()
        print(f"Power range: {laser.get_power_range()}")
        print(f"Current range: {laser.get_current_range()}")
        print(f"Setting power: {laser.set_power(0.1)}")
        print(f"Turning on: {laser.on()}")
        print(f"Current power: {laser.get_power()}")
        print(f"Current status: {laser.get_laser_state()}")
        print(f"Turning off: {laser.off()}")
        
        # Clean up
        mw.on_deactivate()
        fc.on_deactivate()
        laser.on_deactivate()
    else:
        print("Qudi not available, can't test adapters")