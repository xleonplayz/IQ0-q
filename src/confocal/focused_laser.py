# -*- coding: utf-8 -*-

"""
Focused laser beam model for confocal microscopy simulation.

Copyright (c) 2023
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional, Union


class FocusedLaserBeam:
    """
    Model of a focused laser beam for confocal microscopy.
    Implements realistic 3D Gaussian beam profile and depth-dependent effects.
    """
    
    def __init__(self, wavelength: float = 532e-9, numerical_aperture: float = 0.95,
                power: float = 1.0, pinhole_size: float = 50e-6):
        """
        Initialize focused laser beam model.
        
        Parameters
        ----------
        wavelength : float
            Laser wavelength in meters
        numerical_aperture : float
            Numerical aperture of the objective
        power : float
            Laser power in mW
        pinhole_size : float
            Confocal pinhole size in meters
        """
        self.wavelength = wavelength
        self.na = numerical_aperture
        self.power = power
        self.pinhole_size = pinhole_size
        
        # Calculate beam properties
        self.beam_waist = 0.61 * wavelength / numerical_aperture  # Abbe diffraction limit
        self.rayleigh_range = np.pi * self.beam_waist**2 / wavelength  # Depth of focus
        
        # Calculate optical point spread function (PSF) parameters
        self.psf_lateral_fwhm = 0.51 * wavelength / numerical_aperture
        self.psf_axial_fwhm = 0.88 * wavelength / (numerical_aperture**2)
        
        # Convert FWHM to Gaussian sigma
        self.psf_lateral_sigma = self.psf_lateral_fwhm / (2 * np.sqrt(2 * np.log(2)))
        self.psf_axial_sigma = self.psf_axial_fwhm / (2 * np.sqrt(2 * np.log(2)))
        
        # Calculate emission PSF (larger due to longer wavelength, typically)
        self.emission_wavelength = 637e-9  # NV center zero-phonon line
        self.emission_psf_lateral_sigma = 0.51 * self.emission_wavelength / numerical_aperture / (2 * np.sqrt(2 * np.log(2)))
        
        # Calculate confocal parameter (effective detection PSF)
        self.confocal_lateral_sigma = np.sqrt(self.psf_lateral_sigma**2 + self.emission_psf_lateral_sigma**2) / np.sqrt(2)
        self.confocal_axial_sigma = np.sqrt(self.psf_axial_sigma**2 + (0.88 * self.emission_wavelength / (numerical_aperture**2))**2 / (4 * np.log(2)))
        
        # Effective collection volume
        self.effective_volume = (2*np.pi)**(3/2) * self.confocal_lateral_sigma**2 * self.confocal_axial_sigma
    
    def intensity_at_position(self, beam_center: Tuple[float, float, float],
                            target_position: Tuple[float, float, float]) -> float:
        """
        Calculate the laser intensity at a target position.
        
        Parameters
        ----------
        beam_center : tuple
            (x, y, z) coordinates of the beam focus
        target_position : tuple
            (x, y, z) coordinates of the target point
            
        Returns
        -------
        float
            Relative intensity between 0 and 1
        """
        # Calculate distance from center in transverse plane
        r_trans = np.sqrt(
            (target_position[0] - beam_center[0])**2 + 
            (target_position[1] - beam_center[1])**2
        )
        
        # Calculate axial distance
        z = target_position[2] - beam_center[2]
        
        # Calculate beam width at this axial position
        w_z = self.beam_waist * np.sqrt(1 + (z / self.rayleigh_range)**2)
        
        # Calculate intensity using Gaussian beam profile
        intensity = self.power * np.exp(-2 * r_trans**2 / w_z**2) / (1 + (z / self.rayleigh_range)**2)
        
        # Add refractive index mismatch aberration if needed
        # This model assumes perfect index matching for simplicity
        
        return intensity
    
    def collection_efficiency(self, beam_center: Tuple[float, float, float],
                            emitter_position: Tuple[float, float, float]) -> float:
        """
        Calculate the collection efficiency for fluorescence.
        
        Parameters
        ----------
        beam_center : tuple
            (x, y, z) coordinates of the beam focus
        emitter_position : tuple
            (x, y, z) coordinates of the emitter
            
        Returns
        -------
        float
            Collection efficiency between 0 and 1
        """
        # Calculate distance from focus
        r_trans = np.sqrt(
            (emitter_position[0] - beam_center[0])**2 + 
            (emitter_position[1] - beam_center[1])**2
        )
        z = emitter_position[2] - beam_center[2]
        
        # Model confocal response function (approximately Gaussian)
        trans_response = np.exp(-2 * r_trans**2 / (2 * self.confocal_lateral_sigma**2))
        axial_response = np.exp(-2 * z**2 / (2 * self.confocal_axial_sigma**2))
        
        # The pinhole limits the detection volume
        # Calculate effect of pinhole (simplified model)
        pinhole_factor = 1 - np.exp(-2 * self.pinhole_size**2 / (8 * (self.emission_psf_lateral_sigma * (1 + z / self.rayleigh_range))**2))
        
        # Account for refractive index mismatch (simplified)
        # In reality, this depends on the depth and medium properties
        z_abs = abs(z)
        depth_attenuation = np.exp(-0.1 * z_abs)  # Simple exponential attenuation with depth
        
        return trans_response * axial_response * pinhole_factor * depth_attenuation
    
    def set_power(self, power: float) -> None:
        """
        Set the laser power.
        
        Parameters
        ----------
        power : float
            Laser power in mW
        """
        self.power = power
    
    def set_pinhole_size(self, pinhole_size: float) -> None:
        """
        Set the confocal pinhole size.
        
        Parameters
        ----------
        pinhole_size : float
            Pinhole size in meters
        """
        self.pinhole_size = pinhole_size
    
    def get_psf_volume(self) -> float:
        """
        Get the effective PSF volume.
        
        Returns
        -------
        float
            Effective volume in cubic meters
        """
        return self.effective_volume
    
    def get_psf_dimensions(self) -> Dict[str, float]:
        """
        Get the PSF dimensions.
        
        Returns
        -------
        dict
            Dictionary with PSF dimensions
        """
        return {
            'lateral_fwhm': self.psf_lateral_fwhm,
            'axial_fwhm': self.psf_axial_fwhm,
            'confocal_lateral_sigma': self.confocal_lateral_sigma,
            'confocal_axial_sigma': self.confocal_axial_sigma
        }
    
    def calculate_psf_3d(self, size: Tuple[float, float, float] = (5e-6, 5e-6, 10e-6),
                       resolution: Tuple[int, int, int] = (101, 101, 101)) -> np.ndarray:
        """
        Calculate the 3D PSF over a volume.
        
        Parameters
        ----------
        size : tuple
            (x, y, z) dimensions of the volume in meters
        resolution : tuple
            (nx, ny, nz) resolution of the volume
            
        Returns
        -------
        ndarray
            3D array with PSF values
        """
        # Create coordinate grids
        x = np.linspace(-size[0]/2, size[0]/2, resolution[0])
        y = np.linspace(-size[1]/2, size[1]/2, resolution[1])
        z = np.linspace(-size[2]/2, size[2]/2, resolution[2])
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Calculate distance from center in transverse plane
        R_trans = np.sqrt(X**2 + Y**2)
        
        # Calculate beam width at each axial position
        W_z = self.beam_waist * np.sqrt(1 + (Z / self.rayleigh_range)**2)
        
        # Calculate 3D PSF
        psf = np.exp(-2 * R_trans**2 / (W_z**2)) / (1 + (Z / self.rayleigh_range)**2)
        
        return psf