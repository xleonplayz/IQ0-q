#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Example script for confocal microscopy simulation with the NV center simulator.

This demonstrates how to use the confocal simulator to create virtual NV center
diamond samples and simulate confocal scans.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add the root directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, root_dir)

# Import the confocal simulator
from sim.src.confocal import DiamondLattice, FocusedLaserBeam, ConfocalSimulator


def plot_diamond_lattice(diamond, figsize=(10, 8)):
    """Plot the NV centers in the diamond lattice."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract NV center positions and orientations
    positions = np.array([nv['position'] for nv in diamond.nv_centers])
    orientations = np.array([nv['orientation'] for nv in diamond.nv_centers])
    
    # Plot each NV center as a point
    ax.scatter(
        positions[:, 0], 
        positions[:, 1], 
        positions[:, 2], 
        c='red', 
        marker='o', 
        alpha=0.8,
        label=f'NV Centers ({len(diamond.nv_centers)})'
    )
    
    # Plot orientation vectors (scaled)
    scale = 1e-6  # Scale factor for the orientation vectors
    for pos, orient in zip(positions, orientations):
        ax.quiver(
            pos[0], pos[1], pos[2],
            orient[0] * scale, orient[1] * scale, orient[2] * scale,
            color='blue', alpha=0.5
        )
    
    # Set labels and title
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Diamond Lattice with {len(diamond.nv_centers)} NV Centers')
    
    # Set axis limits
    ax.set_xlim(0, diamond.dimensions[0])
    ax.set_ylim(0, diamond.dimensions[1])
    ax.set_zlim(0, diamond.dimensions[2])
    
    plt.legend()
    plt.tight_layout()
    
    return fig, ax


def plot_laser_psf(laser, size=(5e-6, 5e-6, 10e-6), resolution=(51, 51, 51), figsize=(15, 5)):
    """Plot the laser point spread function (PSF)."""
    # Calculate the 3D PSF
    psf = laser.calculate_psf_3d(size, resolution)
    
    # Create coordinate grids
    x = np.linspace(-size[0]/2, size[0]/2, resolution[0])
    y = np.linspace(-size[1]/2, size[1]/2, resolution[1])
    z = np.linspace(-size[2]/2, size[2]/2, resolution[2])
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot XY plane at z=0
    z_mid = resolution[2] // 2
    xy_slice = psf[:, :, z_mid]
    im0 = axes[0].imshow(xy_slice.T, extent=[-size[0]/2*1e6, size[0]/2*1e6, -size[1]/2*1e6, size[1]/2*1e6], 
                        origin='lower', cmap='viridis')
    axes[0].set_title('XY Plane (z=0)')
    axes[0].set_xlabel('X (µm)')
    axes[0].set_ylabel('Y (µm)')
    plt.colorbar(im0, ax=axes[0])
    
    # Plot XZ plane at y=0
    y_mid = resolution[1] // 2
    xz_slice = psf[:, y_mid, :]
    im1 = axes[1].imshow(xz_slice.T, extent=[-size[0]/2*1e6, size[0]/2*1e6, -size[2]/2*1e6, size[2]/2*1e6], 
                        origin='lower', cmap='viridis')
    axes[1].set_title('XZ Plane (y=0)')
    axes[1].set_xlabel('X (µm)')
    axes[1].set_ylabel('Z (µm)')
    plt.colorbar(im1, ax=axes[1])
    
    # Plot YZ plane at x=0
    x_mid = resolution[0] // 2
    yz_slice = psf[x_mid, :, :]
    im2 = axes[2].imshow(yz_slice.T, extent=[-size[1]/2*1e6, size[1]/2*1e6, -size[2]/2*1e6, size[2]/2*1e6], 
                        origin='lower', cmap='viridis')
    axes[2].set_title('YZ Plane (x=0)')
    axes[2].set_xlabel('Y (µm)')
    axes[2].set_ylabel('Z (µm)')
    plt.colorbar(im2, ax=axes[2])
    
    plt.suptitle(f'Laser PSF (λ={laser.wavelength*1e9:.1f}nm, NA={laser.na:.2f})', fontsize=16)
    plt.tight_layout()
    
    return fig, axes


def plot_confocal_scan(image, extent, title='Confocal Scan', figsize=(10, 8)):
    """Plot a confocal scan image."""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot the image
    im = ax.imshow(image.T, extent=extent, origin='lower', cmap='inferno')
    
    # Add labels and title
    ax.set_xlabel('X (µm)')
    ax.set_ylabel('Y (µm)')
    ax.set_title(title)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Counts')
    
    plt.tight_layout()
    
    return fig, ax


if __name__ == "__main__":
    # Create a diamond lattice with NV centers
    print("Creating diamond lattice...")
    dimensions = (20e-6, 20e-6, 5e-6)  # 20µm x 20µm x 5µm
    nv_density = 5e13  # NV centers per cubic meter (5e13 = 0.05 centers per cubic micron)
    diamond = DiamondLattice(dimensions, nv_density, random_seed=42)
    
    print(f"Created diamond with {len(diamond.nv_centers)} NV centers")
    print(f"Actual NV density: {diamond.get_nv_density()/1e12:.2f} × 10¹² centers/m³")
    
    # Plot the diamond lattice
    fig_diamond, ax_diamond = plot_diamond_lattice(diamond)
    plt.savefig(os.path.join(script_dir, "diamond_lattice.png"), dpi=150)
    
    # Create a focused laser beam
    print("\nCreating focused laser beam...")
    laser = FocusedLaserBeam(wavelength=532e-9, numerical_aperture=0.95, power=1.0)
    
    # Display PSF dimensions
    psf_dimensions = laser.get_psf_dimensions()
    print(f"PSF dimensions:")
    print(f"  Lateral FWHM: {psf_dimensions['lateral_fwhm']*1e9:.2f} nm")
    print(f"  Axial FWHM: {psf_dimensions['axial_fwhm']*1e9:.2f} nm")
    print(f"  Effective volume: {laser.get_psf_volume()*1e18:.2f} µm³")
    
    # Plot the laser PSF
    fig_psf, ax_psf = plot_laser_psf(laser)
    plt.savefig(os.path.join(script_dir, "laser_psf.png"), dpi=150)
    
    # Create a confocal simulator
    print("\nInitializing confocal simulator...")
    confocal = ConfocalSimulator(dimensions, nv_density, random_seed=42)
    
    # Set parameters
    confocal.set_laser_power(1.0)  # 1 mW
    confocal.set_background_counts(200)  # 200 counts/s background
    
    # Measure at a single point
    print("\nMeasuring fluorescence at center point...")
    center_point = (dimensions[0]/2, dimensions[1]/2, 0)
    counts = confocal.measure_fluorescence(center_point, integration_time=0.1)
    print(f"Fluorescence at center: {counts:.1f} counts (0.1s integration)")
    
    # Perform a 2D scan
    print("\nPerforming 2D scan...")
    scan_center = (dimensions[0]/2, dimensions[1]/2, 0)  # Center of scan
    scan_size = (10e-6, 10e-6)  # 10µm x 10µm scan
    scan_resolution = (50, 50)  # 50x50 pixels
    
    image = confocal.scan_plane(scan_center, scan_size, scan_resolution, integration_time=0.01)
    
    # Plot the confocal scan
    # Convert extent to microns
    extent = [
        (scan_center[0] - scan_size[0]/2) * 1e6,
        (scan_center[0] + scan_size[0]/2) * 1e6,
        (scan_center[1] - scan_size[1]/2) * 1e6,
        (scan_center[1] + scan_size[1]/2) * 1e6
    ]
    
    fig_scan, ax_scan = plot_confocal_scan(
        image, 
        extent, 
        title=f'Confocal Scan at z={scan_center[2]*1e6:.1f}µm'
    )
    plt.savefig(os.path.join(script_dir, "confocal_scan.png"), dpi=150)
    
    # Perform a depth scan (xz plane)
    print("\nPerforming depth scan (XZ plane)...")
    depth_center = (dimensions[0]/2, dimensions[1]/2, dimensions[2]/2)
    depth_size = (10e-6, 5e-6)  # 10µm x 5µm (x and z)
    depth_resolution = (50, 25)  # 50x25 pixels
    
    # Create a series of 1D scans at different depths
    depth_image = np.zeros(depth_resolution)
    
    z_points = np.linspace(
        depth_center[2] - depth_size[1]/2,
        depth_center[2] + depth_size[1]/2,
        depth_resolution[1]
    )
    
    for i, z in enumerate(z_points):
        # Create start and end points for this depth
        start = (depth_center[0] - depth_size[0]/2, depth_center[1], z)
        end = (depth_center[0] + depth_size[0]/2, depth_center[1], z)
        
        # Scan a line
        line = confocal.scan_line(start, end, depth_resolution[0], integration_time=0.01)
        
        # Add to image
        depth_image[:, i] = line
    
    # Plot the depth scan
    depth_extent = [
        (depth_center[0] - depth_size[0]/2) * 1e6,
        (depth_center[0] + depth_size[0]/2) * 1e6,
        (depth_center[2] - depth_size[1]/2) * 1e6,
        (depth_center[2] + depth_size[1]/2) * 1e6
    ]
    
    fig_depth, ax_depth = plot_confocal_scan(
        depth_image, 
        depth_extent, 
        title=f'Depth Scan at y={depth_center[1]*1e6:.1f}µm'
    )
    ax_depth.set_ylabel('Z (µm)')
    plt.savefig(os.path.join(script_dir, "depth_scan.png"), dpi=150)
    
    # Show all plots
    print("\nDone! Saved plots to examples/confocal directory.")
    plt.show()