# -*- coding: utf-8 -*-

"""
Diamond lattice model with NV centers for confocal microscopy simulation.

Copyright (c) 2023
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union
import sys
import os

# Add parent directory to Python path for importing model
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, parent_dir)

# Import PhysicalNVModel
from src.model import PhysicalNVModel


class DiamondLattice:
    """
    Represents a diamond lattice with NV centers at realistic positions.
    """
    
    def __init__(self, dimensions: Tuple[float, float, float] = (20e-6, 20e-6, 5e-6), 
                density: float = 1e14, random_seed: Optional[int] = None):
        """
        Initialize a diamond lattice with NV centers.
        
        Parameters
        ----------
        dimensions : tuple
            The dimensions of the diamond sample in meters (x, y, z)
        density : float
            NV center density in centers per cubic meter
        random_seed : int or None
            Random seed for reproducible NV distributions
        """
        self.dimensions = dimensions
        self.density = density
        self._rng = np.random.RandomState(random_seed)
        self.nv_centers = []
        self._generate_nv_centers()
    
    def _generate_nv_centers(self) -> None:
        """Generate NV centers at random positions in the diamond lattice"""
        # Calculate volume and expected number of NV centers
        volume = self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
        center_count = int(volume * self.density)
        
        # Generate random positions for all centers
        for i in range(center_count):
            pos = (
                self._rng.uniform(0, self.dimensions[0]),
                self._rng.uniform(0, self.dimensions[1]),
                self._rng.uniform(0, self.dimensions[2])
            )
            
            # Randomly assign an NV orientation from the 4 possible crystallographic axes
            # See: https://doi.org/10.1038/s41467-019-09429-x
            orientation_idx = self._rng.randint(0, 4)
            if orientation_idx == 0:
                orientation = np.array([1, 1, 1]) / np.sqrt(3)
            elif orientation_idx == 1:
                orientation = np.array([1, -1, -1]) / np.sqrt(3)
            elif orientation_idx == 2:
                orientation = np.array([-1, 1, -1]) / np.sqrt(3)
            else:  # idx == 3
                orientation = np.array([-1, -1, 1]) / np.sqrt(3)
            
            # Create an NV center with these properties
            strain = self._rng.normal(0, 5e6)  # Random strain in Hz
            zfs_variation = self._rng.normal(0, 1e6)  # Variation in zero-field splitting
            
            # Create an NV model with individual parameters
            nv_model = PhysicalNVModel(
                strain=strain,
                zero_field_splitting=2.87e9 + zfs_variation
            )
            
            nv = {
                'position': pos,
                'orientation': orientation,
                'strain': strain,
                'model': nv_model
            }
            self.nv_centers.append(nv)
    
    def get_nv_centers_in_volume(self, center: Tuple[float, float, float], 
                               dimensions: Tuple[float, float, float]) -> List[Dict[str, Any]]:
        """
        Get all NV centers in a specified volume around a point.
        
        Parameters
        ----------
        center : tuple
            (x, y, z) coordinates of the center of the volume
        dimensions : tuple
            (width, height, depth) of the volume in meters
            
        Returns
        -------
        list
            List of NV centers in the specified volume
        """
        x_min, x_max = center[0] - dimensions[0]/2, center[0] + dimensions[0]/2
        y_min, y_max = center[1] - dimensions[1]/2, center[1] + dimensions[1]/2
        z_min, z_max = center[2] - dimensions[2]/2, center[2] + dimensions[2]/2
        
        centers = []
        for nv in self.nv_centers:
            pos = nv['position']
            if (x_min <= pos[0] <= x_max and 
                y_min <= pos[1] <= y_max and 
                z_min <= pos[2] <= z_max):
                centers.append(nv)
        
        return centers
    
    def get_nv_count(self) -> int:
        """
        Get the total number of NV centers in the lattice.
        
        Returns
        -------
        int
            Number of NV centers
        """
        return len(self.nv_centers)
    
    def get_nv_density(self) -> float:
        """
        Get the actual NV center density.
        
        Returns
        -------
        float
            Actual NV center density in centers per cubic meter
        """
        volume = self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
        return len(self.nv_centers) / volume
    
    def regenerate(self, random_seed: Optional[int] = None) -> None:
        """
        Regenerate the NV center distribution with a new random seed.
        
        Parameters
        ----------
        random_seed : int or None
            New random seed for NV distribution
        """
        if random_seed is not None:
            self._rng = np.random.RandomState(random_seed)
        
        self.nv_centers = []
        self._generate_nv_centers()
    
    def apply_magnetic_field(self, field_vector: Tuple[float, float, float]) -> None:
        """
        Apply a magnetic field to all NV centers in the lattice.
        
        Parameters
        ----------
        field_vector : tuple
            (Bx, By, Bz) magnetic field vector in Tesla
        """
        # Convert to NumPy array for vector operations
        field = np.array(field_vector)
        
        for nv in self.nv_centers:
            # Calculate projection of field onto NV axis
            nv_axis = nv['orientation']
            field_projection = np.dot(field, nv_axis)
            
            # Apply the field to the NV model
            nv_model = nv['model']
            nv_model.set_magnetic_field(field_vector)

    def get_avg_fluorescence(self) -> float:
        """
        Calculate the average base fluorescence of all NV centers.
        
        Returns
        -------
        float
            Average fluorescence in counts per second
        """
        if not self.nv_centers:
            return 0.0
        
        total_fluorescence = 0.0
        for nv in self.nv_centers:
            model = nv['model']
            # Apply laser to get fluorescence
            model.apply_laser(1.0, True)
            fluorescence = model.get_fluorescence()
            model.apply_laser(0.0, False)  # Turn laser off
            total_fluorescence += fluorescence
        
        return total_fluorescence / len(self.nv_centers)