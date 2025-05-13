"""
Nuclear spin bath module for simulating spin environments around an NV center.

This module provides classes for creating and managing nuclear spin environments,
including positioning of nuclear spins, species configuration, and integration
with the SimOS quantum simulation framework.
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union, NamedTuple
import sys
import os

# Configure logging
logger = logging.getLogger(__name__)

# Diamond lattice constants
LATTICE_CONSTANT = 3.57e-10  # meters
CARBON_CARBON_DISTANCE = 1.54e-10  # meters
UNIT_CELL_VOLUME = (LATTICE_CONSTANT**3) / 8  # 8 carbon atoms per unit cell

# Nuclear magnetic properties
GYROMAGNETIC_RATIOS = {
    '13C': 10.705e6,    # Hz/T
    '14N': 3.077e6,     # Hz/T
    '15N': -4.316e6,    # Hz/T
    '1H': 42.577e6      # Hz/T (for surface hydrogen, if needed)
}

# Nuclear spin quantums
SPIN_QUANTUM = {
    '13C': 0.5,
    '14N': 1.0,
    '15N': 0.5,
    '1H': 0.5
}


class SpinConfig(NamedTuple):
    """Configuration for a single nuclear spin."""
    position: Tuple[float, float, float]
    species: str = '13C'
    index: int = 0
    name: Optional[str] = None
    
    @property
    def gyromagnetic_ratio(self) -> float:
        """Get the gyromagnetic ratio for this nuclear species."""
        return GYROMAGNETIC_RATIOS.get(self.species, 0.0)
    
    @property
    def spin_quantum(self) -> float:
        """Get the spin quantum number for this nuclear species."""
        return SPIN_QUANTUM.get(self.species, 0.0)
    
    def to_simos_dict(self) -> Dict[str, Any]:
        """Convert to SimOS-compatible dictionary format."""
        return {
            'val': self.spin_quantum,
            'name': self.name or f"{self.species}_{self.index}",
            'type': self.species,
            'pos': self.position
        }


class NuclearSpinBath:
    """
    Class representing a nuclear spin bath environment around an NV center.
    
    This class manages the nuclear spins surrounding an NV center, including
    their positions, species, and interactions. It provides methods for generating
    realistic spin configurations based on diamond lattice structure and for
    integrating with the SimOS quantum simulation framework.
    """
    
    def __init__(self, concentration: float = 0.011, bath_size: int = 10, 
                 random_seed: Optional[int] = None, include_nitrogen: bool = True,
                 nitrogen_species: str = '14N'):
        """
        Initialize a nuclear spin bath environment.
        
        Parameters
        ----------
        concentration : float
            Natural abundance or chosen concentration of 13C (0-1)
        bath_size : int
            Number of nuclear spins to include
        random_seed : int, optional
            Random seed for reproducible positions
        include_nitrogen : bool
            Whether to include the host nitrogen nuclear spin
        nitrogen_species : str
            Nitrogen isotope to use ('14N' or '15N')
        """
        self._concentration = concentration
        self._bath_size = bath_size
        self._rng = np.random.RandomState(random_seed)
        self._spins = []
        self._nv_position = (0.0, 0.0, 0.0)  # NV center at origin
        
        # Generate random spin positions according to diamond lattice
        self._generate_spin_positions()
        
        # Add host nitrogen if requested
        if include_nitrogen:
            if nitrogen_species not in ('14N', '15N'):
                raise ValueError(f"Invalid nitrogen species: {nitrogen_species}. Must be '14N' or '15N'")
            
            # Nitrogen position is along the NV axis (z-axis) at the vacancy site
            nitrogen_position = (0.0, 0.0, -CARBON_CARBON_DISTANCE)
            self.add_custom_spin(nitrogen_position, species=nitrogen_species)
    
    def _generate_spin_positions(self):
        """
        Generate random positions for nuclear spins in diamond lattice.
        
        This method creates a realistic distribution of 13C nuclear spins based on
        the diamond lattice structure and the specified concentration.
        """
        # Calculate volume needed to contain the requested number of spins
        # Diamond has 8 carbon atoms per unit cell
        carbon_density = 8 / (LATTICE_CONSTANT**3)
        
        # Total volume needed for the bath_size at natural abundance
        volume_needed = self._bath_size / (carbon_density * self._concentration)
        
        # Radius of a sphere with this volume
        radius = (3 * volume_needed / (4 * np.pi))**(1/3)
        
        # Determine number of lattice sites to generate (we'll filter later)
        # Over-generate to ensure we have enough after filtering
        safety_factor = 4
        num_sites = int(safety_factor * self._bath_size / self._concentration)
        
        # Generate positions in a diamond lattice structure within the radius
        positions = []
        
        # Generate positions on a cubic grid
        grid_size = int(np.ceil(2 * radius / LATTICE_CONSTANT))
        
        # Diamond lattice basis vectors (FCC lattice + basis)
        basis = [
            (0, 0, 0),
            (0, 0.5, 0.5),
            (0.5, 0, 0.5),
            (0.5, 0.5, 0),
            (0.25, 0.25, 0.25),
            (0.25, 0.75, 0.75),
            (0.75, 0.25, 0.75),
            (0.75, 0.75, 0.25)
        ]
        
        # Generate positions using diamond lattice structure
        for i in range(-grid_size, grid_size + 1):
            for j in range(-grid_size, grid_size + 1):
                for k in range(-grid_size, grid_size + 1):
                    for dx, dy, dz in basis:
                        x = (i + dx) * LATTICE_CONSTANT
                        y = (j + dy) * LATTICE_CONSTANT
                        z = (k + dz) * LATTICE_CONSTANT
                        
                        # Calculate distance from NV center
                        dist = np.sqrt(x**2 + y**2 + z**2)
                        
                        # Add if within radius and not too close to NV
                        min_distance = 2 * CARBON_CARBON_DISTANCE  # Minimum distance from NV
                        if dist <= radius and dist >= min_distance:
                            positions.append((x, y, z))
        
        # Randomly select positions based on concentration
        selected_indices = self._rng.choice(
            len(positions),
            size=min(self._bath_size, len(positions)),
            replace=False
        )
        
        # Create spin configurations
        for i, idx in enumerate(selected_indices):
            position = positions[idx]
            self.add_custom_spin(position, species='13C', index=i)
        
        logger.info(f"Generated {len(self._spins)} nuclear spins in bath")
    
    def add_custom_spin(self, position: Tuple[float, float, float], 
                         species: str = '13C', index: Optional[int] = None, 
                         name: Optional[str] = None):
        """
        Add a specific nuclear spin at a given position.
        
        Parameters
        ----------
        position : tuple
            (x, y, z) coordinates in meters
        species : str
            Nuclear species ('13C', '14N', '15N', etc.)
        index : int, optional
            Custom index for the spin
        name : str, optional
            Custom name for the spin
        """
        if index is None:
            index = len(self._spins)
            
        if name is None:
            name = f"{species}_{index}"
        
        spin_config = SpinConfig(
            position=position,
            species=species,
            index=index,
            name=name
        )
        
        self._spins.append(spin_config)
        return spin_config
    
    def get_spins(self, species: Optional[str] = None) -> List[SpinConfig]:
        """
        Get all spins or filter by species.
        
        Parameters
        ----------
        species : str, optional
            Filter by nuclear species
            
        Returns
        -------
        list
            List of SpinConfig objects
        """
        if species is None:
            return self._spins
        
        return [spin for spin in self._spins if spin.species == species]
    
    def get_spin_positions(self, species: Optional[str] = None) -> List[Tuple[float, float, float]]:
        """
        Get positions of all spins or filter by species.
        
        Parameters
        ----------
        species : str, optional
            Filter by nuclear species
            
        Returns
        -------
        list
            List of (x, y, z) position tuples
        """
        spins = self.get_spins(species)
        return [spin.position for spin in spins]
    
    def create_simos_system(self, method: str = 'qutip'):
        """
        Create a SimOS system with the nuclear spin bath.
        
        Parameters
        ----------
        method : str
            Numerical backend to use ('qutip', 'numpy', or 'sparse')
            
        Returns
        -------
        NVSystem
            SimOS NV system with nuclear spins included
        """
        # Import here to avoid circular imports
        from simos.systems.NV import NVSystem
        
        # Create additional spins for SimOS
        further_spins = [spin.to_simos_dict() for spin in self._spins]
        
        # Determine if nitrogen is already included in the bath
        has_nitrogen = any(spin.species in ('14N', '15N') for spin in self._spins)
        
        # Create NV system with nuclear spins
        nv_system = NVSystem(
            optics=True, 
            orbital=False,
            nitrogen=has_nitrogen,  # We add nitrogen explicitly in the bath if needed
            method=method
        )
        
        # Add further spins to the system
        if further_spins:
            # Note: Actual implementation depends on how SimOS handles adding spins
            # This is a simplified version
            nv_system.add_additional_spins(further_spins)
        
        return nv_system
    
    def __len__(self):
        """Get the number of spins in the bath."""
        return len(self._spins)
    
    def __repr__(self):
        """String representation of the nuclear spin bath."""
        species_count = {}
        for spin in self._spins:
            species_count[spin.species] = species_count.get(spin.species, 0) + 1
        
        species_str = ", ".join(f"{count} {species}" for species, count in species_count.items())
        return f"NuclearSpinBath(total_spins={len(self._spins)}, {species_str})"