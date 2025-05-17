import json
import logging
import os
from typing import Dict, List, Optional, Union, Any
from fastapi import FastAPI, HTTPException, Depends, Header, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import uvicorn
from enum import Enum

from nv.simulator import NVSimulator
from nv.config_loader import load_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI instance
app = FastAPI(
    title="NV Simulator API",
    description="API for NV-Center simulator with Qudi-IQO compatibility",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API authentication
API_KEY = os.environ.get("NV_API_KEY", "dev-key")

def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return x_api_key

# Global state
simulator = None
microwave_state = {
    "module_state": "idle",  # "idle" or "locked"
    "is_scanning": False,
    "cw_power": 0.0,
    "cw_frequency": 2.87e9,
    "scan_power": 0.0,
    "scan_frequencies": None,  # or Array/Tuple depending on mode
    "scan_mode": "JUMP_LIST",
    "scan_sample_rate": 100.0
}

fast_counter_state = {
    "status": 1,  # 0=unconfigured, 1=idle, 2=running, 3=paused, -1=error
    "is_gated": False,
    "binwidth": 1.0526315789473685e-9,
    "gate_length_bins": 8192,
    "count_data": None
}

laser_state = {
    "laser_state": "OFF",  # OFF, ON, LOCKED
    "shutter_state": "CLOSED",  # CLOSED, OPEN, NO_SHUTTER
    "control_mode": "POWER",  # POWER, CURRENT
    "power": 0.0,
    "power_setpoint": 0.0,
    "current": 0.0,
    "current_setpoint": 0.0,
    "temperatures": {
        "psu": 32.5,
        "head": 41.8
    }
}

simulation_params = {
    "nv_centers": [
        {"position": [0.000053, 0.000068, 0.0], "orientation": [0.57735, 0.57735, 0.57735], "type": "N14"},
        {"position": [0.000087, 0.000022, 0.0], "orientation": [0.57735, 0.57735, -0.57735], "type": "N14"}
    ],
    "magnetic_field": {"magnitude": 0.001, "direction": [0.0, 0.0, 1.0]},
    "temperature": 293.15,
    "noise_level": 0.05,
    "background_counts": 5000.0,
    "collection_efficiency": 0.05,
    "spot_psf_fwhm": 400e-9
}

# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

class SamplingOutputMode(str, Enum):
    JUMP_LIST = "JUMP_LIST"
    EQUIDISTANT_SWEEP = "EQUIDISTANT_SWEEP"

class LaserState(str, Enum):
    OFF = "OFF"
    ON = "ON"
    LOCKED = "LOCKED"

class ShutterState(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    NO_SHUTTER = "NO_SHUTTER"

class ControlMode(str, Enum):
    POWER = "POWER"
    CURRENT = "CURRENT"

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------

# Microwave models
class MicrowaveConstraints(BaseModel):
    power_limits: List[float] = Field([-60.0, 30.0], description="Power limits in dBm [min, max]")
    frequency_limits: List[float] = Field([100e3, 20e9], description="Frequency limits in Hz [min, max]")
    scan_size_limits: List[int] = Field([2, 1001], description="Scan size limits [min, max]")
    sample_rate_limits: List[float] = Field([0.1, 200.0], description="Sample rate limits in Hz [min, max]")
    scan_modes: List[SamplingOutputMode] = Field([SamplingOutputMode.JUMP_LIST, SamplingOutputMode.EQUIDISTANT_SWEEP], 
                                               description="Available scan modes")

class MicrowaveStatus(BaseModel):
    module_state: str = Field(..., description="Module state ('idle' or 'locked')")
    is_scanning: bool = Field(..., description="Whether a frequency scan is running")
    cw_power: float = Field(..., description="CW power setpoint in dBm")
    cw_frequency: float = Field(..., description="CW frequency setpoint in Hz")
    scan_power: float = Field(..., description="Scan power setpoint in dBm")
    scan_frequencies: Optional[Union[List[float], List[List[float]]]] = Field(None, 
                                                                        description="Scan frequencies in Hz")
    scan_mode: SamplingOutputMode = Field(..., description="Scan mode (JUMP_LIST or EQUIDISTANT_SWEEP)")
    scan_sample_rate: float = Field(..., description="Sample rate for scans in Hz")

class CWParams(BaseModel):
    frequency: float = Field(..., description="Frequency in Hz")
    power: float = Field(..., description="Power in dBm")

class ScanParams(BaseModel):
    power: float = Field(..., description="Power in dBm")
    frequencies: Union[List[float], List[List[float]]] = Field(..., 
                                                     description="Frequencies in Hz or [start, stop, count]")
    mode: SamplingOutputMode = Field(..., description="Scan mode")
    sample_rate: float = Field(..., description="Sample rate in Hz")

# Fast Counter models
class FastCounterConstraints(BaseModel):
    hardware_binwidth_list: List[float] = Field(..., description="Available binwidths in seconds")

class FastCounterStatus(BaseModel):
    status: int = Field(..., description="Status code (0=unconfigured, 1=idle, 2=running, 3=paused, -1=error)")
    status_desc: str = Field(..., description="Status description")
    is_gated: bool = Field(..., description="Whether gated mode is enabled")
    binwidth: float = Field(..., description="Binwidth in seconds")
    gate_length_bins: int = Field(..., description="Gate length in bins")

class FastCounterConfig(BaseModel):
    bin_width_s: float = Field(..., description="Desired bin width in seconds")
    record_length_s: float = Field(..., description="Desired record length in seconds")
    number_of_gates: int = Field(0, description="Number of gates (0 for ungated)")

class CounterData(BaseModel):
    data: Union[List[float], List[List[float]]] = Field(..., description="Count data")
    info: dict = Field({}, description="Additional info about the data")

# Laser models
class LaserConstraints(BaseModel):
    power_range: List[float] = Field(..., description="Power range in W [min, max]")
    current_range: List[float] = Field(..., description="Current range [min, max]")
    current_unit: str = Field(..., description="Current unit")
    allowed_control_modes: List[ControlMode] = Field(..., description="Allowed control modes")

class LaserStatus(BaseModel):
    laser_state: LaserState = Field(..., description="Laser state")
    shutter_state: ShutterState = Field(..., description="Shutter state")
    control_mode: ControlMode = Field(..., description="Control mode")
    power: float = Field(..., description="Current power in W")
    power_setpoint: float = Field(..., description="Power setpoint in W")
    current: float = Field(..., description="Current current")
    current_setpoint: float = Field(..., description="Current setpoint")
    temperatures: Dict[str, float] = Field(..., description="Temperature readings")

class LaserPower(BaseModel):
    power: float = Field(..., description="Power in W")

class LaserCurrent(BaseModel):
    current: float = Field(..., description="Current")

class LaserControlMode(BaseModel):
    mode: ControlMode = Field(..., description="Control mode")

class LaserStateParam(BaseModel):
    state: LaserState = Field(..., description="Laser state")

class ShutterStateParam(BaseModel):
    state: ShutterState = Field(..., description="Shutter state")

# Simulation models
class NVCenter(BaseModel):
    position: List[float] = Field(..., description="Position [x, y, z] in meters")
    orientation: List[float] = Field(..., description="Orientation vector [x, y, z]")
    type: str = Field(..., description="Type of NV center (N14 or N15)")

class MagneticField(BaseModel):
    magnitude: float = Field(..., description="Magnitude in Tesla")
    direction: List[float] = Field(..., description="Direction vector [x, y, z]")

class SimulationParams(BaseModel):
    nv_centers: Optional[List[NVCenter]] = None
    magnetic_field: Optional[MagneticField] = None
    temperature: Optional[float] = None
    noise_level: Optional[float] = None
    background_counts: Optional[float] = None
    collection_efficiency: Optional[float] = None
    spot_psf_fwhm: Optional[float] = None

class SimulationReset(BaseModel):
    nv_center_density: float = Field(..., description="NV center density in 1/m^3")
    volume: List[float] = Field(..., description="Volume dimensions [x, y, z] in meters")
    randomize_orientations: bool = Field(True, description="Whether to randomize NV orientations")

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def initialize_simulator():
    """Initialize the simulator if not already initialized."""
    global simulator
    if simulator is None:
        config_loader = load_config()
        config = config_loader.get_nv_system_config()
        simulator = NVSimulator(config)
        logger.info("NV Simulator initialized")
    return simulator

def get_status_description(status_code):
    """Convert status code to description."""
    status_map = {
        0: "unconfigured",
        1: "idle",
        2: "running",
        3: "paused",
        -1: "error"
    }
    return status_map.get(status_code, "unknown")

def generate_count_data(length, is_gated=False, gates=1):
    """Generate simulated count data."""
    sim = initialize_simulator()
    
    # Basic counts from fluorescence
    base_counts = sim.get_fluorescence()
    
    # Add noise
    noise_level = simulation_params["noise_level"]
    background = simulation_params["background_counts"]
    
    if is_gated:
        # Generate gated data
        data = []
        for i in range(gates):
            gate_data = np.random.poisson(base_counts + background, length)
            data.append(gate_data.tolist())
    else:
        # Generate ungated data
        data = np.random.poisson(base_counts + background, length).tolist()
    
    return data

# -----------------------------------------------------------------------------
# Routes: Microwave API
# -----------------------------------------------------------------------------

@app.get("/api/v1/microwave/constraints", response_model=MicrowaveConstraints, 
         dependencies=[Depends(verify_api_key)])
def get_microwave_constraints():
    """Get microwave hardware constraints."""
    return MicrowaveConstraints()

@app.get("/api/v1/microwave/status", response_model=MicrowaveStatus, 
         dependencies=[Depends(verify_api_key)])
def get_microwave_status():
    """Get current microwave status."""
    return microwave_state

@app.post("/api/v1/microwave/set_cw", dependencies=[Depends(verify_api_key)])
def set_cw(params: CWParams):
    """Set CW parameters for the microwave."""
    # Validate parameters against constraints
    constraints = MicrowaveConstraints()
    if not (constraints.frequency_limits[0] <= params.frequency <= constraints.frequency_limits[1]):
        raise HTTPException(status_code=400, detail="Frequency out of range")
    if not (constraints.power_limits[0] <= params.power <= constraints.power_limits[1]):
        raise HTTPException(status_code=400, detail="Power out of range")
    
    # Update state
    microwave_state["cw_frequency"] = params.frequency
    microwave_state["cw_power"] = params.power
    
    # Update simulator
    sim = initialize_simulator()
    sim.set_microwave(params.frequency, params.power)
    
    return {
        "cw_frequency": microwave_state["cw_frequency"],
        "cw_power": microwave_state["cw_power"]
    }

@app.post("/api/v1/microwave/cw_on", dependencies=[Depends(verify_api_key)])
def cw_on():
    """Turn on CW mode for the microwave."""
    # Update state
    microwave_state["module_state"] = "locked"
    microwave_state["is_scanning"] = False
    
    # Update simulator
    sim = initialize_simulator()
    sim.set_microwave(microwave_state["cw_frequency"], microwave_state["cw_power"])
    
    return {
        "module_state": microwave_state["module_state"],
        "is_scanning": microwave_state["is_scanning"]
    }

@app.post("/api/v1/microwave/configure_scan", dependencies=[Depends(verify_api_key)])
def configure_scan(params: ScanParams):
    """Configure a frequency scan."""
    # Validate parameters
    constraints = MicrowaveConstraints()
    if params.power < constraints.power_limits[0] or params.power > constraints.power_limits[1]:
        raise HTTPException(status_code=400, detail="Power out of range")
    
    if params.sample_rate < constraints.sample_rate_limits[0] or params.sample_rate > constraints.sample_rate_limits[1]:
        raise HTTPException(status_code=400, detail="Sample rate out of range")
    
    # Process frequencies based on scan mode
    if params.mode == SamplingOutputMode.JUMP_LIST:
        # Frequencies should be a list
        frequencies = params.frequencies
        if len(frequencies) < constraints.scan_size_limits[0] or len(frequencies) > constraints.scan_size_limits[1]:
            raise HTTPException(status_code=400, detail="Scan size out of range")
        
        # Check each frequency
        for freq in frequencies:
            if freq < constraints.frequency_limits[0] or freq > constraints.frequency_limits[1]:
                raise HTTPException(status_code=400, detail="Frequency out of range")
    
    elif params.mode == SamplingOutputMode.EQUIDISTANT_SWEEP:
        # Frequencies should be [start, stop, count]
        if len(params.frequencies) != 3:
            raise HTTPException(status_code=400, 
                              detail="For EQUIDISTANT_SWEEP, frequencies must be [start, stop, count]")
        
        start, stop, count = params.frequencies
        if start < constraints.frequency_limits[0] or stop > constraints.frequency_limits[1]:
            raise HTTPException(status_code=400, detail="Frequency out of range")
        
        if count < constraints.scan_size_limits[0] or count > constraints.scan_size_limits[1]:
            raise HTTPException(status_code=400, detail="Scan size out of range")
        
        # Generate frequencies
        frequencies = np.linspace(start, stop, int(count)).tolist()
    
    else:
        raise HTTPException(status_code=400, detail=f"Invalid scan mode: {params.mode}")
    
    # Update state
    microwave_state["scan_power"] = params.power
    microwave_state["scan_frequencies"] = frequencies
    microwave_state["scan_mode"] = params.mode
    microwave_state["scan_sample_rate"] = params.sample_rate
    
    return {
        "scan_power": microwave_state["scan_power"],
        "scan_frequencies": microwave_state["scan_frequencies"],
        "scan_mode": microwave_state["scan_mode"],
        "scan_sample_rate": microwave_state["scan_sample_rate"]
    }

@app.post("/api/v1/microwave/start_scan", dependencies=[Depends(verify_api_key)])
def start_scan():
    """Start the configured frequency scan."""
    # Check if scan is configured
    if microwave_state["scan_frequencies"] is None:
        raise HTTPException(status_code=400, detail="Scan not configured")
    
    # Update state
    microwave_state["module_state"] = "locked"
    microwave_state["is_scanning"] = True
    
    # If frequencies is a list, set the first frequency
    if isinstance(microwave_state["scan_frequencies"], list) and len(microwave_state["scan_frequencies"]) > 0:
        sim = initialize_simulator()
        sim.set_microwave(microwave_state["scan_frequencies"][0], microwave_state["scan_power"])
    
    return {
        "module_state": microwave_state["module_state"],
        "is_scanning": microwave_state["is_scanning"]
    }

@app.post("/api/v1/microwave/reset_scan", dependencies=[Depends(verify_api_key)])
def reset_scan():
    """Reset the current scan."""
    # Check if a scan is running
    if not microwave_state["is_scanning"]:
        raise HTTPException(status_code=400, detail="No scan is running")
    
    # Just acknowledge the reset
    return {
        "scan_reset": True
    }

@app.post("/api/v1/microwave/off", dependencies=[Depends(verify_api_key)])
def microwave_off():
    """Turn off the microwave."""
    # Update state
    microwave_state["module_state"] = "idle"
    microwave_state["is_scanning"] = False
    
    # Turn off microwave in simulator
    sim = initialize_simulator()
    sim.set_microwave(microwave_state["cw_frequency"], -100)  # -100 dBm is effectively off
    
    return {
        "module_state": microwave_state["module_state"],
        "is_scanning": microwave_state["is_scanning"]
    }

# -----------------------------------------------------------------------------
# Routes: Fast Counter API
# -----------------------------------------------------------------------------

@app.get("/api/v1/fast_counter/constraints", response_model=FastCounterConstraints, 
         dependencies=[Depends(verify_api_key)])
def get_fast_counter_constraints():
    """Get fast counter hardware constraints."""
    return FastCounterConstraints(
        hardware_binwidth_list=[1.0526315789473685e-09, 2.1052631578947366e-09, 
                                4.2105263157894735e-09, 8.421052631578947e-09]
    )

@app.get("/api/v1/fast_counter/status", dependencies=[Depends(verify_api_key)])
def get_fast_counter_status():
    """Get current fast counter status."""
    return {
        "status": fast_counter_state["status"],
        "status_desc": get_status_description(fast_counter_state["status"]),
        "is_gated": fast_counter_state["is_gated"],
        "binwidth": fast_counter_state["binwidth"],
        "gate_length_bins": fast_counter_state["gate_length_bins"]
    }

@app.post("/api/v1/fast_counter/configure", dependencies=[Depends(verify_api_key)])
def configure_fast_counter(config: FastCounterConfig):
    """Configure the fast counter."""
    # Get constraints
    constraints = FastCounterConstraints(
        hardware_binwidth_list=[1.0526315789473685e-09, 2.1052631578947366e-09, 
                                4.2105263157894735e-09, 8.421052631578947e-09]
    )
    
    # Find closest bin width
    closest_binwidth = min(constraints.hardware_binwidth_list, 
                           key=lambda x: abs(x - config.bin_width_s))
    
    # Calculate gate length in bins
    gate_length_bins = int(config.record_length_s / closest_binwidth)
    
    # Update state
    fast_counter_state["binwidth"] = closest_binwidth
    fast_counter_state["gate_length_bins"] = gate_length_bins
    fast_counter_state["is_gated"] = config.number_of_gates > 0
    fast_counter_state["status"] = 1  # idle
    fast_counter_state["count_data"] = None
    
    return {
        "binwidth_s": closest_binwidth,
        "record_length_s": gate_length_bins * closest_binwidth,
        "number_of_gates": config.number_of_gates
    }

@app.post("/api/v1/fast_counter/start_measure", dependencies=[Depends(verify_api_key)])
def start_fast_counter():
    """Start the fast counter measurement."""
    # Check if counter is configured
    if fast_counter_state["status"] == 0:
        raise HTTPException(status_code=400, detail="Counter not configured")
    
    # Generate data
    gates = 0 if not fast_counter_state["is_gated"] else 1  # Default to 1 gate
    fast_counter_state["count_data"] = generate_count_data(
        fast_counter_state["gate_length_bins"], 
        fast_counter_state["is_gated"],
        gates
    )
    
    # Update state
    fast_counter_state["status"] = 2  # running
    
    return {
        "status": fast_counter_state["status"],
        "status_desc": get_status_description(fast_counter_state["status"])
    }

@app.post("/api/v1/fast_counter/stop_measure", dependencies=[Depends(verify_api_key)])
def stop_fast_counter():
    """Stop the fast counter measurement."""
    # Update state
    fast_counter_state["status"] = 1  # idle
    
    return {
        "status": fast_counter_state["status"],
        "status_desc": get_status_description(fast_counter_state["status"])
    }

@app.post("/api/v1/fast_counter/pause_measure", dependencies=[Depends(verify_api_key)])
def pause_fast_counter():
    """Pause the fast counter measurement."""
    # Check if counter is running
    if fast_counter_state["status"] != 2:
        raise HTTPException(status_code=400, detail="Counter not running")
    
    # Update state
    fast_counter_state["status"] = 3  # paused
    
    return {
        "status": fast_counter_state["status"],
        "status_desc": get_status_description(fast_counter_state["status"])
    }

@app.post("/api/v1/fast_counter/continue_measure", dependencies=[Depends(verify_api_key)])
def continue_fast_counter():
    """Continue the paused fast counter measurement."""
    # Check if counter is paused
    if fast_counter_state["status"] != 3:
        raise HTTPException(status_code=400, detail="Counter not paused")
    
    # Update state
    fast_counter_state["status"] = 2  # running
    
    return {
        "status": fast_counter_state["status"],
        "status_desc": get_status_description(fast_counter_state["status"])
    }

@app.get("/api/v1/fast_counter/get_data_trace", response_model=CounterData, 
         dependencies=[Depends(verify_api_key)])
def get_fast_counter_data():
    """Get the current counter data."""
    # Check if data is available
    if fast_counter_state["count_data"] is None:
        # Generate new data
        gates = 0 if not fast_counter_state["is_gated"] else 1  # Default to 1 gate
        fast_counter_state["count_data"] = generate_count_data(
            fast_counter_state["gate_length_bins"], 
            fast_counter_state["is_gated"],
            gates
        )
    
    return {
        "data": fast_counter_state["count_data"],
        "info": {
            "elapsed_sweeps": None,
            "elapsed_time": None
        }
    }

# -----------------------------------------------------------------------------
# Routes: Laser API
# -----------------------------------------------------------------------------

@app.get("/api/v1/laser/constraints", response_model=LaserConstraints, 
         dependencies=[Depends(verify_api_key)])
def get_laser_constraints():
    """Get laser hardware constraints."""
    return LaserConstraints(
        power_range=[0.0, 0.25],
        current_range=[0.0, 100.0],
        current_unit="%",
        allowed_control_modes=[ControlMode.POWER, ControlMode.CURRENT]
    )

@app.get("/api/v1/laser/status", response_model=LaserStatus, 
         dependencies=[Depends(verify_api_key)])
def get_laser_status():
    """Get current laser status."""
    # Update temperature for realism
    laser_state["temperatures"]["psu"] = 32.5 + np.random.normal(0, 0.2)
    laser_state["temperatures"]["head"] = 41.8 + np.random.normal(0, 0.3)
    
    return laser_state

@app.post("/api/v1/laser/set_power", dependencies=[Depends(verify_api_key)])
def set_laser_power(power_params: LaserPower):
    """Set the laser power."""
    # Get constraints
    constraints = LaserConstraints(
        power_range=[0.0, 0.25],
        current_range=[0.0, 100.0],
        current_unit="%",
        allowed_control_modes=[ControlMode.POWER, ControlMode.CURRENT]
    )
    
    # Validate parameters
    if power_params.power < constraints.power_range[0] or power_params.power > constraints.power_range[1]:
        raise HTTPException(status_code=400, detail="Power out of range")
    
    # Calculate current from power (simple linear model for demonstration)
    current = power_params.power / constraints.power_range[1] * constraints.current_range[1]
    
    # Update state
    laser_state["power_setpoint"] = power_params.power
    laser_state["current_setpoint"] = current
    
    # If laser is on, also update actual values
    if laser_state["laser_state"] == LaserState.ON:
        laser_state["power"] = power_params.power
        laser_state["current"] = current
        
        # Update simulator
        sim = initialize_simulator()
        sim.set_laser(power_params.power * 1000)  # Convert to mW
    
    return {
        "power_setpoint": laser_state["power_setpoint"],
        "current_setpoint": laser_state["current_setpoint"]
    }

@app.post("/api/v1/laser/set_current", dependencies=[Depends(verify_api_key)])
def set_laser_current(current_params: LaserCurrent):
    """Set the laser current."""
    # Get constraints
    constraints = LaserConstraints(
        power_range=[0.0, 0.25],
        current_range=[0.0, 100.0],
        current_unit="%",
        allowed_control_modes=[ControlMode.POWER, ControlMode.CURRENT]
    )
    
    # Validate parameters
    if current_params.current < constraints.current_range[0] or current_params.current > constraints.current_range[1]:
        raise HTTPException(status_code=400, detail="Current out of range")
    
    # Calculate power from current (simple linear model for demonstration)
    power = current_params.current / constraints.current_range[1] * constraints.power_range[1]
    
    # Update state
    laser_state["power_setpoint"] = power
    laser_state["current_setpoint"] = current_params.current
    
    # If laser is on, also update actual values
    if laser_state["laser_state"] == LaserState.ON:
        laser_state["power"] = power
        laser_state["current"] = current_params.current
        
        # Update simulator
        sim = initialize_simulator()
        sim.set_laser(power * 1000)  # Convert to mW
    
    return {
        "current_setpoint": laser_state["current_setpoint"],
        "power_setpoint": laser_state["power_setpoint"]
    }

@app.post("/api/v1/laser/set_control_mode", dependencies=[Depends(verify_api_key)])
def set_laser_control_mode(mode_params: LaserControlMode):
    """Set the laser control mode."""
    # Validate mode
    constraints = LaserConstraints(
        power_range=[0.0, 0.25],
        current_range=[0.0, 100.0],
        current_unit="%",
        allowed_control_modes=[ControlMode.POWER, ControlMode.CURRENT]
    )
    
    if mode_params.mode not in constraints.allowed_control_modes:
        raise HTTPException(status_code=400, detail=f"Invalid control mode: {mode_params.mode}")
    
    # Update state
    laser_state["control_mode"] = mode_params.mode
    
    return {
        "control_mode": laser_state["control_mode"]
    }

@app.post("/api/v1/laser/set_laser_state", dependencies=[Depends(verify_api_key)])
def set_laser_state_endpoint(state_params: LaserStateParam):
    """Set the laser state."""
    # Update state
    laser_state["laser_state"] = state_params.state
    
    # If turning on, set actual values to setpoints
    if state_params.state == LaserState.ON:
        laser_state["power"] = laser_state["power_setpoint"]
        laser_state["current"] = laser_state["current_setpoint"]
        
        # Update simulator
        sim = initialize_simulator()
        sim.set_laser(laser_state["power"] * 1000)  # Convert to mW
    else:
        # If turning off, set actual values to 0
        laser_state["power"] = 0.0
        laser_state["current"] = 0.0
        
        # Update simulator
        sim = initialize_simulator()
        sim.set_laser(0.0)
    
    return {
        "laser_state": laser_state["laser_state"]
    }

@app.post("/api/v1/laser/set_shutter_state", dependencies=[Depends(verify_api_key)])
def set_shutter_state(state_params: ShutterStateParam):
    """Set the shutter state."""
    # Update state
    laser_state["shutter_state"] = state_params.state
    
    return {
        "shutter_state": laser_state["shutter_state"]
    }

@app.get("/api/v1/laser/get_temperatures", dependencies=[Depends(verify_api_key)])
def get_laser_temperatures():
    """Get the laser temperature readings."""
    # Update temperatures for realism
    laser_state["temperatures"]["psu"] = 32.5 + np.random.normal(0, 0.2)
    laser_state["temperatures"]["head"] = 41.8 + np.random.normal(0, 0.3)
    
    return {
        "temperatures": laser_state["temperatures"]
    }

# -----------------------------------------------------------------------------
# Routes: Simulation Parameters API
# -----------------------------------------------------------------------------

@app.get("/api/v1/simulation/params", dependencies=[Depends(verify_api_key)])
def get_simulation_params():
    """Get the current simulation parameters."""
    return simulation_params

@app.post("/api/v1/simulation/params", dependencies=[Depends(verify_api_key)])
def set_simulation_params(params: SimulationParams):
    """Set simulation parameters."""
    global simulation_params
    
    # Update only provided parameters
    if params.nv_centers is not None:
        simulation_params["nv_centers"] = [center.dict() for center in params.nv_centers]
    
    if params.magnetic_field is not None:
        simulation_params["magnetic_field"] = params.magnetic_field.dict()
        
        # Update simulator magnetic field
        sim = initialize_simulator()
        mag = params.magnetic_field.magnitude
        dir_vec = params.magnetic_field.direction
        # Normalize direction vector
        norm = np.sqrt(sum(d*d for d in dir_vec))
        if norm > 0:
            dir_vec = [d/norm for d in dir_vec]
        # Set magnetic field
        b_field = [mag * dir_vec[0], mag * dir_vec[1], mag * dir_vec[2]]
        sim.set_magnetic_field(b_field)
    
    if params.temperature is not None:
        simulation_params["temperature"] = params.temperature
    
    if params.noise_level is not None:
        simulation_params["noise_level"] = params.noise_level
    
    if params.background_counts is not None:
        simulation_params["background_counts"] = params.background_counts
    
    if params.collection_efficiency is not None:
        simulation_params["collection_efficiency"] = params.collection_efficiency
    
    if params.spot_psf_fwhm is not None:
        simulation_params["spot_psf_fwhm"] = params.spot_psf_fwhm
    
    return {
        "params_updated": True,
        "current_params": simulation_params
    }

@app.post("/api/v1/simulation/reset", dependencies=[Depends(verify_api_key)])
def reset_simulation(params: SimulationReset):
    """Reset the simulation and generate new NV centers."""
    global simulation_params
    
    # Calculate expected number of NV centers
    volume_m3 = params.volume[0] * params.volume[1] * params.volume[2]
    expected_count = int(params.nv_center_density * volume_m3)
    
    # Limit to a reasonable number
    count = min(expected_count, 200)
    
    # Generate random positions within the volume
    nv_centers = []
    for _ in range(count):
        pos = [
            np.random.uniform(0, params.volume[0]),
            np.random.uniform(0, params.volume[1]),
            np.random.uniform(0, params.volume[2])
        ]
        
        # Generate orientation - either random or one of the 4 NV orientations
        if params.randomize_orientations:
            # Random unit vector
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            orient = [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.cos(theta)
            ]
        else:
            # One of the 4 NV orientations
            nv_orientations = [
                [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
                [1/np.sqrt(3), 1/np.sqrt(3), -1/np.sqrt(3)],
                [1/np.sqrt(3), -1/np.sqrt(3), 1/np.sqrt(3)],
                [1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)]
            ]
            orient = nv_orientations[np.random.randint(0, 4)]
        
        # Randomly choose between N14 and N15
        nv_type = "N14" if np.random.random() < 0.9 else "N15"
        
        nv_centers.append({
            "position": pos,
            "orientation": orient,
            "type": nv_type
        })
    
    # Update simulation params
    simulation_params["nv_centers"] = nv_centers
    
    # Initialize simulator with new configuration
    sim = initialize_simulator()
    
    return {
        "reset": True,
        "nv_centers_count": len(nv_centers)
    }

# -----------------------------------------------------------------------------
# Main function to run the server
# -----------------------------------------------------------------------------

def run_server(host="0.0.0.0", port=5000):
    # Initialize simulator
    initialize_simulator()
    
    # Run server
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server()