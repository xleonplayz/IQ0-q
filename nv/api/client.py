import requests
from typing import Dict, List, Optional, Union, Any
import json
import time
import logging
from enum import Enum
import numpy as np
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SamplingOutputMode(str, Enum):
    """Scan modes for the microwave source."""
    JUMP_LIST = "JUMP_LIST"
    EQUIDISTANT_SWEEP = "EQUIDISTANT_SWEEP"

class LaserState(str, Enum):
    """Possible states for the laser."""
    OFF = "OFF"
    ON = "ON"
    LOCKED = "LOCKED"

class ShutterState(str, Enum):
    """Possible states for the laser shutter."""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    NO_SHUTTER = "NO_SHUTTER"

class ControlMode(str, Enum):
    """Control modes for the laser."""
    POWER = "POWER"
    CURRENT = "CURRENT"

class NVSimulatorClient:
    """Client for the NV Simulator API."""
    
    def __init__(self, base_url="http://localhost:5000/api/v1", api_key=None):
        """
        Initialize the NV Simulator API client.
        
        Parameters
        ----------
        base_url : str, optional
            Base URL for the API
        api_key : str, optional
            API key for authentication. If not provided, it attempts to get it from environment.
        """
        self.base_url = base_url
        self.api_key = api_key or os.environ.get("NV_API_KEY", "dev-key")
        self.headers = {"X-API-Key": self.api_key}
    
    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------
    
    def _get(self, endpoint, params=None):
        """
        Make a GET request to the API.
        
        Parameters
        ----------
        endpoint : str
            API endpoint
        params : dict, optional
            Query parameters
            
        Returns
        -------
        dict
            API response
        """
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"GET request failed: {e}")
            raise
    
    def _post(self, endpoint, data=None):
        """
        Make a POST request to the API.
        
        Parameters
        ----------
        endpoint : str
            API endpoint
        data : dict, optional
            Request body
            
        Returns
        -------
        dict
            API response
        """
        url = f"{self.base_url}/{endpoint}"
        try:
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"POST request failed: {e}")
            raise
    
    # -------------------------------------------------------------------------
    # Microwave API methods
    # -------------------------------------------------------------------------
    
    def get_microwave_constraints(self):
        """Get microwave hardware constraints."""
        return self._get("microwave/constraints")
    
    def get_microwave_status(self):
        """Get current microwave status."""
        return self._get("microwave/status")
    
    def set_cw(self, frequency, power):
        """
        Set CW parameters for the microwave.
        
        Parameters
        ----------
        frequency : float
            Frequency in Hz
        power : float
            Power in dBm
            
        Returns
        -------
        dict
            Updated CW parameters
        """
        data = {"frequency": frequency, "power": power}
        return self._post("microwave/set_cw", data)
    
    def cw_on(self):
        """Turn on CW mode for the microwave."""
        return self._post("microwave/cw_on")
    
    def configure_scan(self, power, frequencies, mode="JUMP_LIST", sample_rate=100.0):
        """
        Configure a frequency scan.
        
        Parameters
        ----------
        power : float
            Power in dBm
        frequencies : list or array
            For JUMP_LIST: List of frequencies in Hz
            For EQUIDISTANT_SWEEP: [start, stop, count]
        mode : str, optional
            Scan mode (JUMP_LIST or EQUIDISTANT_SWEEP)
        sample_rate : float, optional
            Sample rate in Hz
            
        Returns
        -------
        dict
            Scan configuration
        """
        data = {
            "power": power,
            "frequencies": frequencies,
            "mode": mode,
            "sample_rate": sample_rate
        }
        return self._post("microwave/configure_scan", data)
    
    def start_scan(self):
        """Start the configured frequency scan."""
        return self._post("microwave/start_scan")
    
    def reset_scan(self):
        """Reset the current scan."""
        return self._post("microwave/reset_scan")
    
    def microwave_off(self):
        """Turn off the microwave."""
        return self._post("microwave/off")
    
    # -------------------------------------------------------------------------
    # Fast Counter API methods
    # -------------------------------------------------------------------------
    
    def get_fast_counter_constraints(self):
        """Get fast counter hardware constraints."""
        return self._get("fast_counter/constraints")
    
    def get_fast_counter_status(self):
        """Get current fast counter status."""
        return self._get("fast_counter/status")
    
    def configure_fast_counter(self, bin_width_s, record_length_s, number_of_gates=0):
        """
        Configure the fast counter.
        
        Parameters
        ----------
        bin_width_s : float
            Desired bin width in seconds
        record_length_s : float
            Desired record length in seconds
        number_of_gates : int, optional
            Number of gates (0 for ungated)
            
        Returns
        -------
        dict
            Counter configuration
        """
        data = {
            "bin_width_s": bin_width_s,
            "record_length_s": record_length_s,
            "number_of_gates": number_of_gates
        }
        return self._post("fast_counter/configure", data)
    
    def start_fast_counter(self):
        """Start the fast counter measurement."""
        return self._post("fast_counter/start_measure")
    
    def stop_fast_counter(self):
        """Stop the fast counter measurement."""
        return self._post("fast_counter/stop_measure")
    
    def pause_fast_counter(self):
        """Pause the fast counter measurement."""
        return self._post("fast_counter/pause_measure")
    
    def continue_fast_counter(self):
        """Continue the paused fast counter measurement."""
        return self._post("fast_counter/continue_measure")
    
    def get_fast_counter_data(self):
        """Get the current counter data."""
        return self._get("fast_counter/get_data_trace")
    
    # -------------------------------------------------------------------------
    # Laser API methods
    # -------------------------------------------------------------------------
    
    def get_laser_constraints(self):
        """Get laser hardware constraints."""
        return self._get("laser/constraints")
    
    def get_laser_status(self):
        """Get current laser status."""
        return self._get("laser/status")
    
    def set_laser_power(self, power):
        """
        Set the laser power.
        
        Parameters
        ----------
        power : float
            Power in W
            
        Returns
        -------
        dict
            Updated power settings
        """
        data = {"power": power}
        return self._post("laser/set_power", data)
    
    def set_laser_current(self, current):
        """
        Set the laser current.
        
        Parameters
        ----------
        current : float
            Current value
            
        Returns
        -------
        dict
            Updated current settings
        """
        data = {"current": current}
        return self._post("laser/set_current", data)
    
    def set_laser_control_mode(self, mode):
        """
        Set the laser control mode.
        
        Parameters
        ----------
        mode : str
            Control mode (POWER or CURRENT)
            
        Returns
        -------
        dict
            Updated control mode
        """
        data = {"mode": mode}
        return self._post("laser/set_control_mode", data)
    
    def set_laser_state(self, state):
        """
        Set the laser state.
        
        Parameters
        ----------
        state : str
            Laser state (OFF, ON, or LOCKED)
            
        Returns
        -------
        dict
            Updated laser state
        """
        data = {"state": state}
        return self._post("laser/set_laser_state", data)
    
    def set_shutter_state(self, state):
        """
        Set the shutter state.
        
        Parameters
        ----------
        state : str
            Shutter state (CLOSED, OPEN, or NO_SHUTTER)
            
        Returns
        -------
        dict
            Updated shutter state
        """
        data = {"state": state}
        return self._post("laser/set_shutter_state", data)
    
    def get_laser_temperatures(self):
        """Get the laser temperature readings."""
        return self._get("laser/get_temperatures")
    
    # -------------------------------------------------------------------------
    # Simulation Parameters API methods
    # -------------------------------------------------------------------------
    
    def get_simulation_params(self):
        """Get the current simulation parameters."""
        return self._get("simulation/params")
    
    def set_simulation_params(self, **params):
        """
        Set simulation parameters.
        
        Parameters
        ----------
        **params
            Simulation parameters to update
            
        Returns
        -------
        dict
            Updated simulation parameters
        """
        return self._post("simulation/params", params)
    
    def reset_simulation(self, nv_center_density, volume, randomize_orientations=True):
        """
        Reset the simulation and generate new NV centers.
        
        Parameters
        ----------
        nv_center_density : float
            NV center density in 1/m^3
        volume : list
            Volume dimensions [x, y, z] in meters
        randomize_orientations : bool, optional
            Whether to randomize NV orientations
            
        Returns
        -------
        dict
            Reset confirmation
        """
        data = {
            "nv_center_density": nv_center_density,
            "volume": volume,
            "randomize_orientations": randomize_orientations
        }
        return self._post("simulation/reset", data)
    
    # -------------------------------------------------------------------------
    # Helper methods for basic measurements
    # -------------------------------------------------------------------------
    
    def get_fluorescence_counts(self, laser_power=0.01, duration=1.0, 
                       with_microwave=False, mw_freq=2.87e9, mw_power=-10.0):
        """
        Measure fluorescence counts with the given parameters.
        
        Parameters
        ----------
        laser_power : float, optional
            Laser power in W
        duration : float, optional
            Measurement duration in seconds
        with_microwave : bool, optional
            Whether to apply microwaves during measurement
        mw_freq : float, optional
            Microwave frequency in Hz (if with_microwave is True)
        mw_power : float, optional
            Microwave power in dBm (if with_microwave is True)
            
        Returns
        -------
        float
            Measured fluorescence in counts/s
        """
        # Set up laser
        self.set_laser_power(laser_power)
        self.set_laser_state("ON")
        
        # Apply microwave if requested
        if with_microwave:
            self.set_cw(mw_freq, mw_power)
            self.cw_on()
        
        # Configure counter
        self.configure_fast_counter(1e-9, duration)
        
        # Run measurement
        self.start_fast_counter()
        time.sleep(duration)  # Simulate waiting for the measurement
        
        # Get data
        data = self.get_fast_counter_data()
        
        # Process data - average counts
        if isinstance(data["data"], list):
            if isinstance(data["data"][0], list):  # Gated data
                counts = np.mean([np.mean(gate) for gate in data["data"]])
            else:  # Ungated data
                counts = np.mean(data["data"])
        else:
            counts = 0
        
        # Clean up
        self.stop_fast_counter()
        if with_microwave:
            self.microwave_off()
        self.set_laser_state("OFF")
        
        return counts

# Example usage
if __name__ == "__main__":
    client = NVSimulatorClient()
    
    # Get simulator status
    sim_params = client.get_simulation_params()
    print(f"Simulator initialized with {len(sim_params['nv_centers'])} NV centers")
    print(f"Magnetic field: {sim_params['magnetic_field']['magnitude']} T")
    
    # Simple fluorescence measurement with microwave
    print("Measuring fluorescence with microwave on/off...")
    fluorescence_off = client.get_fluorescence_counts(with_microwave=False)
    fluorescence_on = client.get_fluorescence_counts(with_microwave=True)
    contrast = (fluorescence_off - fluorescence_on) / fluorescence_off
    print(f"Fluorescence off: {fluorescence_off:.1f}, on: {fluorescence_on:.1f}, contrast: {contrast:.3f}")