# -*- coding: utf-8 -*-

"""
Confocal microscopy simulator for the NV center simulator.

Copyright (c) 2023
"""

import numpy as np
import time
from typing import Tuple, List, Dict, Any, Optional, Union

from .diamond_lattice import DiamondLattice
from .focused_laser import FocusedLaserBeam


class ConfocalSimulator:
    """
    Simulates a confocal microscope scanning an NV center sample.
    """
    
    def __init__(self, dimensions: Tuple[float, float, float] = (20e-6, 20e-6, 5e-6), 
                nv_density: float = 1e14, random_seed: Optional[int] = None):
        """
        Initialize the confocal microscope simulator.
        
        Parameters
        ----------
        dimensions : tuple
            The dimensions of the diamond sample in meters (x, y, z)
        nv_density : float
            NV center density in centers per cubic meter
        random_seed : int or None
            Random seed for reproducible NV distributions
        """
        self.diamond = DiamondLattice(dimensions, nv_density, random_seed)
        self.laser = FocusedLaserBeam()
        
        # Scanner parameters
        self.position = (dimensions[0]/2, dimensions[1]/2, 0)
        self.background_counts = 200  # counts/s
        self.collection_volume = (2e-6, 2e-6, 5e-6)  # effective collection volume
        
        # Simulation parameters
        self._saturation_power = 1.0  # mW, power at which NV centers saturate
        self._noise_enabled = True
        self._realistic_timing = False  # Whether to simulate realistic timing delays
        
        # Statistics
        self._total_centers_measured = 0
        self._scan_times = []
    
    def measure_fluorescence(self, position: Tuple[float, float, float], 
                           integration_time: float = 0.01) -> float:
        """
        Measure fluorescence at a specific position.
        
        Parameters
        ----------
        position : tuple
            (x, y, z) coordinates of the focal point
        integration_time : float
            Measurement time in seconds
            
        Returns
        -------
        float
            Fluorescence counts
        """
        # Simulate realistic timing if enabled
        if self._realistic_timing:
            time.sleep(integration_time)
        
        # Get NV centers in the collection volume
        nv_centers = self.diamond.get_nv_centers_in_volume(
            position, self.collection_volume
        )
        
        # Update statistics
        self._total_centers_measured += len(nv_centers)
        
        # Calculate total fluorescence
        total_counts = 0
        for nv in nv_centers:
            # Calculate excitation intensity for this NV
            intensity = self.laser.intensity_at_position(position, nv['position'])
            
            # Calculate collection efficiency
            collection_eff = self.laser.collection_efficiency(position, nv['position'])
            
            # Get the actual NV fluorescence based on the quantum state
            nv_model = nv['model']
            
            # Apply laser with appropriate power
            # Account for saturation effects
            effective_power = self.laser.power * intensity
            saturated_power = effective_power / (1 + effective_power / self._saturation_power)
            
            # Apply laser to NV model and get fluorescence
            nv_model.apply_laser(saturated_power, True)
            base_fluorescence = nv_model.get_fluorescence()
            nv_model.apply_laser(0.0, False)  # Turn laser off
            
            # Calculate actual detected fluorescence
            nv_counts = base_fluorescence * collection_eff * integration_time
            
            total_counts += nv_counts
        
        # Add background and shot noise
        total_counts += self.background_counts * integration_time
        
        # Add Poisson noise if enabled
        if self._noise_enabled:
            noisy_counts = np.random.poisson(total_counts)
        else:
            noisy_counts = total_counts
        
        return noisy_counts
    
    def scan_line(self, start: Tuple[float, float, float], 
                end: Tuple[float, float, float], 
                steps: int, integration_time: float = 0.01) -> np.ndarray:
        """
        Scan along a line and measure fluorescence at each point.
        
        Parameters
        ----------
        start : tuple
            (x, y, z) coordinates of the start point
        end : tuple
            (x, y, z) coordinates of the end point
        steps : int
            Number of points along the line
        integration_time : float
            Integration time per point in seconds
            
        Returns
        -------
        ndarray
            Array of fluorescence counts
        """
        start_time = time.time()
        
        # Generate positions along the line
        positions = []
        for i in range(steps):
            t = i / (steps - 1) if steps > 1 else 0
            pos = tuple(s + t * (e - s) for s, e in zip(start, end))
            positions.append(pos)
        
        # Measure fluorescence at each position
        fluorescence = np.zeros(steps)
        for i, pos in enumerate(positions):
            fluorescence[i] = self.measure_fluorescence(pos, integration_time)
        
        # Record scan time
        scan_time = time.time() - start_time
        self._scan_times.append(scan_time)
        
        return fluorescence
    
    def scan_plane(self, center: Tuple[float, float, float], 
                 size: Tuple[float, float], 
                 resolution: Tuple[int, int], 
                 integration_time: float = 0.01) -> np.ndarray:
        """
        Scan a plane and measure fluorescence at each point.
        
        Parameters
        ----------
        center : tuple
            (x, y, z) coordinates of the center of the plane
        size : tuple
            (width, height) of the plane in meters
        resolution : tuple
            (nx, ny) resolution of the scan
        integration_time : float
            Integration time per point in seconds
            
        Returns
        -------
        ndarray
            2D array of fluorescence counts
        """
        start_time = time.time()
        
        # Generate scan coordinates
        x_start = center[0] - size[0]/2
        x_end = center[0] + size[0]/2
        y_start = center[1] - size[1]/2
        y_end = center[1] + size[1]/2
        z = center[2]
        
        x_points = np.linspace(x_start, x_end, resolution[0])
        y_points = np.linspace(y_start, y_end, resolution[1])
        
        # Initialize image array
        image = np.zeros(resolution)
        
        # Scan the plane
        for i in range(resolution[0]):
            for j in range(resolution[1]):
                position = (x_points[i], y_points[j], z)
                image[i, j] = self.measure_fluorescence(position, integration_time)
        
        # Record scan time
        scan_time = time.time() - start_time
        self._scan_times.append(scan_time)
        
        return image
    
    def scan_volume(self, center: Tuple[float, float, float], 
                  size: Tuple[float, float, float], 
                  resolution: Tuple[int, int, int], 
                  integration_time: float = 0.01) -> np.ndarray:
        """
        Scan a volume and measure fluorescence at each point.
        
        Parameters
        ----------
        center : tuple
            (x, y, z) coordinates of the center of the volume
        size : tuple
            (width, height, depth) of the volume in meters
        resolution : tuple
            (nx, ny, nz) resolution of the scan
        integration_time : float
            Integration time per point in seconds
            
        Returns
        -------
        ndarray
            3D array of fluorescence counts
        """
        start_time = time.time()
        
        # Generate scan coordinates
        x_start = center[0] - size[0]/2
        x_end = center[0] + size[0]/2
        y_start = center[1] - size[1]/2
        y_end = center[1] + size[1]/2
        z_start = center[2] - size[2]/2
        z_end = center[2] + size[2]/2
        
        x_points = np.linspace(x_start, x_end, resolution[0])
        y_points = np.linspace(y_start, y_end, resolution[1])
        z_points = np.linspace(z_start, z_end, resolution[2])
        
        # Initialize volume array
        volume = np.zeros(resolution)
        
        # Scan the volume
        for i in range(resolution[0]):
            for j in range(resolution[1]):
                for k in range(resolution[2]):
                    position = (x_points[i], y_points[j], z_points[k])
                    volume[i, j, k] = self.measure_fluorescence(position, integration_time)
        
        # Record scan time
        scan_time = time.time() - start_time
        self._scan_times.append(scan_time)
        
        return volume
    
    def set_laser_power(self, power: float) -> None:
        """
        Set the laser power.
        
        Parameters
        ----------
        power : float
            Laser power in mW
        """
        self.laser.set_power(power)
    
    def set_background_counts(self, counts: float) -> None:
        """
        Set the background count rate.
        
        Parameters
        ----------
        counts : float
            Background counts per second
        """
        self.background_counts = counts
    
    def set_noise_enabled(self, enabled: bool) -> None:
        """
        Enable or disable noise in the measurements.
        
        Parameters
        ----------
        enabled : bool
            Whether to add noise to measurements
        """
        self._noise_enabled = enabled
    
    def set_realistic_timing(self, enabled: bool) -> None:
        """
        Enable or disable realistic timing in simulations.
        
        Parameters
        ----------
        enabled : bool
            Whether to simulate realistic timing delays
        """
        self._realistic_timing = enabled
    
    def set_saturation_power(self, power: float) -> None:
        """
        Set the saturation power for NV centers.
        
        Parameters
        ----------
        power : float
            Saturation power in mW
        """
        self._saturation_power = power
    
    def apply_magnetic_field(self, field: Tuple[float, float, float]) -> None:
        """
        Apply a magnetic field to the sample.
        
        Parameters
        ----------
        field : tuple
            (Bx, By, Bz) magnetic field vector in Tesla
        """
        self.diamond.apply_magnetic_field(field)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the confocal simulations.
        
        Returns
        -------
        dict
            Dictionary with simulation statistics
        """
        return {
            'total_centers_measured': self._total_centers_measured,
            'nv_count': self.diamond.get_nv_count(),
            'nv_density': self.diamond.get_nv_density(),
            'avg_scan_time': np.mean(self._scan_times) if self._scan_times else 0,
            'total_scan_time': np.sum(self._scan_times) if self._scan_times else 0,
            'scan_count': len(self._scan_times),
            'psf_dimensions': self.laser.get_psf_dimensions()
        }