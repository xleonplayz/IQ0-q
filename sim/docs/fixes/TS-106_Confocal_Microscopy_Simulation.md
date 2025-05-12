# TS-106: Confocal Microscopy Simulation

## Summary
Implement confocal microscopy simulation capabilities for the NV center simulator, enabling spatial scanning and 3D confocal image generation with realistic NV center distributions in a diamond lattice.

## Motivation
While the current simulator accurately models the quantum behavior of a single NV center, it lacks the spatial component necessary for simulating confocal microscopy experiments. In real experiments, researchers scan across diamond samples containing multiple NV centers at various positions and depths. Adding this capability would allow the simulator to generate realistic confocal images, depth-dependent measurements, and optimize experimental parameters before real hardware experiments.

## Description
This technical story focuses on extending the simulator with spatial scanning capabilities to simulate confocal microscopy experiments. It will implement a virtual diamond sample with multiple NV centers, a focused laser beam model, and integration with Qudi's scanning probe interface.

### Requirements
1. Create a virtual diamond sample model:
   - Generate NV centers at realistic positions in diamond lattice
   - Configure NV center density and distribution
   - Set up multiple NV centers with individual properties
   - Support different NV orientations

2. Implement a focused laser beam model:
   - Model 3D Gaussian beam profile
   - Simulate position-dependent excitation
   - Calculate collection efficiency based on focus position
   - Support scanning beam across the sample

3. Simulate realistic confocal signals:
   - Generate fluorescence counts based on excitation intensity
   - Add realistic noise and background
   - Model depth-dependent signal strength
   - Support optical saturation effects

4. Integrate with Qudi scanning interface:
   - Implement ScanningProbeInterface for the simulator
   - Support 1D, 2D, and 3D scanning
   - Enable position feedback
   - Allow coordinate transformations

### Implementation Details
1. Create a 3D diamond lattice with NV centers:
   ```python
   class DiamondLattice:
       def __init__(self, dimensions=(20e-6, 20e-6, 5e-6), density=1e14, random_seed=None):
           """
           Initialize a diamond lattice with NV centers.
           
           Parameters:
           -----------
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
           
       def _generate_nv_centers(self):
           """Generate NV centers at random positions in the diamond lattice"""
           # Calculate volume and expected number of NV centers
           volume = self.dimensions[0] * self.dimensions[1] * self.dimensions[2]
           center_count = int(volume * self.density)
           
           # Generate random positions for all centers
           positions = []
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
                   orientation = (1, 1, 1) / np.sqrt(3)
               elif orientation_idx == 1:
                   orientation = (1, -1, -1) / np.sqrt(3)
               elif orientation_idx == 2:
                   orientation = (-1, 1, -1) / np.sqrt(3)
               else:  # idx == 3
                   orientation = (-1, -1, 1) / np.sqrt(3)
               
               # Create an NV center with these properties
               strain = self._rng.normal(0, 5e6)  # Random strain in Hz
               nv = {
                   'position': pos,
                   'orientation': orientation,
                   'strain': strain,
                   'model': PhysicalNVModel(
                       strain=strain,
                       zero_field_splitting=2.87e9 + self._rng.normal(0, 1e6)
                   )
               }
               self.nv_centers.append(nv)
           
       def get_nv_centers_in_volume(self, center, dimensions):
           """Get all NV centers in a specified volume around a point"""
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
   ```

2. Implement the laser beam model:
   ```python
   class FocusedLaserBeam:
       def __init__(self, wavelength=532e-9, numerical_aperture=0.95):
           """
           Model a focused laser beam for confocal microscopy.
           
           Parameters:
           -----------
           wavelength : float
               Laser wavelength in meters
           numerical_aperture : float
               Numerical aperture of the objective
           """
           self.wavelength = wavelength
           self.na = numerical_aperture
           
           # Calculate beam properties
           self.beam_waist = 0.61 * wavelength / numerical_aperture
           self.rayleigh_range = np.pi * self.beam_waist**2 / wavelength
           
       def intensity_at_position(self, beam_center, target_position):
           """
           Calculate the laser intensity at a target position.
           
           Parameters:
           -----------
           beam_center : tuple
               (x, y, z) coordinates of the beam focus
           target_position : tuple
               (x, y, z) coordinates of the target point
               
           Returns:
           --------
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
           intensity = np.exp(-2 * r_trans**2 / w_z**2) / (1 + (z / self.rayleigh_range)**2)
           
           return intensity
       
       def collection_efficiency(self, beam_center, emitter_position):
           """
           Calculate the collection efficiency for fluorescence.
           
           Parameters:
           -----------
           beam_center : tuple
               (x, y, z) coordinates of the beam focus
           emitter_position : tuple
               (x, y, z) coordinates of the emitter
               
           Returns:
           --------
           float
               Collection efficiency between 0 and 1
           """
           # In a real confocal microscope, collection efficiency depends on:
           # - Distance from focus
           # - Pinhole size
           # - Wavelength
           # - Scattering in the diamond
           
           # Calculate distance from focus
           r_trans = np.sqrt(
               (emitter_position[0] - beam_center[0])**2 + 
               (emitter_position[1] - beam_center[1])**2
           )
           z = emitter_position[2] - beam_center[2]
           
           # Model confocal response function (approximately Gaussian)
           trans_response = np.exp(-2 * r_trans**2 / self.beam_waist**2)
           axial_response = np.exp(-2 * z**2 / (2 * self.rayleigh_range)**2)
           
           return trans_response * axial_response
   ```

3. Create the confocal scanner simulator:
   ```python
   class ConfocalSimulator:
       def __init__(self, dimensions=(20e-6, 20e-6, 5e-6), nv_density=1e14, random_seed=None):
           """
           Simulate a confocal microscope scanning an NV center sample.
           
           Parameters:
           -----------
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
           
       def measure_fluorescence(self, position, integration_time=0.01):
           """
           Measure fluorescence at a specific position.
           
           Parameters:
           -----------
           position : tuple
               (x, y, z) coordinates of the focal point
           integration_time : float
               Measurement time in seconds
               
           Returns:
           --------
           float
               Fluorescence counts
           """
           # Get NV centers in the collection volume
           nv_centers = self.diamond.get_nv_centers_in_volume(
               position, self.collection_volume
           )
           
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
               nv_model.apply_laser(1.0, True)  # 1 mW laser power, on
               
               # Get fluorescence from the NV model
               base_fluorescence = nv_model.get_fluorescence()
               
               # Calculate actual detected fluorescence
               nv_counts = base_fluorescence * intensity * collection_eff * integration_time
               
               # Turn laser off
               nv_model.apply_laser(0.0, False)
               
               total_counts += nv_counts
           
           # Add background and shot noise
           total_counts += self.background_counts * integration_time
           
           # Add Poisson noise
           noisy_counts = np.random.poisson(total_counts)
           
           return noisy_counts
       
       def scan_line(self, start, end, steps, integration_time=0.01):
           """Scan along a line and measure fluorescence"""
           positions = np.linspace(start, end, steps)
           fluorescence = np.zeros(steps)
           
           for i in range(steps):
               fluorescence[i] = self.measure_fluorescence(positions[i], integration_time)
               
           return fluorescence
       
       def scan_plane(self, center, size, resolution, integration_time=0.01):
           """Scan a plane and measure fluorescence at each point"""
           x_start = center[0] - size[0]/2
           x_end = center[0] + size[0]/2
           y_start = center[1] - size[1]/2
           y_end = center[1] + size[1]/2
           z = center[2]
           
           x_points = np.linspace(x_start, x_end, resolution[0])
           y_points = np.linspace(y_start, y_end, resolution[1])
           
           image = np.zeros(resolution)
           
           for i in range(resolution[0]):
               for j in range(resolution[1]):
                   position = (x_points[i], y_points[j], z)
                   image[i, j] = self.measure_fluorescence(position, integration_time)
                   
           return image
   ```

4. Implement the scanning probe interface for Qudi:
   ```python
   class ConfocalSimulatorScanner(ScanningProbeInterface):
       """
       A QDI scanning probe interface implementation for the confocal simulator.
       """
       
       def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self._thread_lock = RecursiveMutex()
           self._scan_settings = None
           self._back_scan_settings = None
           self._scan_data = None
           self._back_scan_data = None
           self._current_position = {}
           self._confocal_simulator = None
           
       def on_activate(self):
           """Initialize the confocal simulator."""
           # Create confocal simulator
           self._confocal_simulator = ConfocalSimulator()
           
           # Set up constraints for scanner axes
           axes = []
           x_range = (0, self._confocal_simulator.diamond.dimensions[0])
           y_range = (0, self._confocal_simulator.diamond.dimensions[1])
           z_range = (0, self._confocal_simulator.diamond.dimensions[2])
           
           # Position, step, resolution, and frequency constraints for each axis
           axes.append(
               ScannerAxis(
                   name='x', 
                   unit='m', 
                   position=ScalarConstraint(default=x_range[0], bounds=x_range),
                   step=ScalarConstraint(default=0.1e-6, bounds=(0.01e-6, 1e-6)),
                   resolution=ScalarConstraint(default=100, bounds=(2, 1000), enforce_int=True),
                   frequency=ScalarConstraint(default=10, bounds=(0.1, 100))
               )
           )
           axes.append(
               ScannerAxis(
                   name='y', 
                   unit='m', 
                   position=ScalarConstraint(default=y_range[0], bounds=y_range),
                   step=ScalarConstraint(default=0.1e-6, bounds=(0.01e-6, 1e-6)),
                   resolution=ScalarConstraint(default=100, bounds=(2, 1000), enforce_int=True),
                   frequency=ScalarConstraint(default=10, bounds=(0.1, 100))
               )
           )
           axes.append(
               ScannerAxis(
                   name='z', 
                   unit='m', 
                   position=ScalarConstraint(default=z_range[0], bounds=z_range),
                   step=ScalarConstraint(default=0.1e-6, bounds=(0.01e-6, 1e-6)),
                   resolution=ScalarConstraint(default=100, bounds=(2, 1000), enforce_int=True),
                   frequency=ScalarConstraint(default=1, bounds=(0.1, 10))
               )
           )
           
           # Define scanner channels
           channels = [
               ScannerChannel(name='fluorescence', unit='c/s', dtype='float64')
           ]
           
           # Create constraints
           self._constraints = ScanConstraints(
               axis_objects=tuple(axes),
               channel_objects=tuple(channels),
               back_scan_capability=BackScanCapability.AVAILABLE,
               has_position_feedback=False,
               square_px_only=False
           )
           
           # Initialize position
           self._current_position = {
               'x': self._confocal_simulator.diamond.dimensions[0]/2,
               'y': self._confocal_simulator.diamond.dimensions[1]/2,
               'z': 0
           }
           
       def on_deactivate(self):
           """Deinitialize hardware."""
           self.reset()
           
       def reset(self):
           """Hard reset of the hardware."""
           with self._thread_lock:
               if self.module_state() == 'locked':
                   self.stop_scan()
                   self.module_state.unlock()
               
       @property
       def constraints(self):
           """Return scanner constraints."""
           return self._constraints
           
       @property
       def scan_settings(self):
           """Return current scan settings."""
           return self._scan_settings
           
       @property
       def back_scan_settings(self):
           """Return back scan settings."""
           return self._back_scan_settings
           
       def configure_scan(self, settings):
           """Configure the scan."""
           with self._thread_lock:
               if self.module_state() != 'idle':
                   raise RuntimeError('Cannot configure scan while scanning is in progress.')
               
               # Check settings against constraints
               self.constraints.check_settings(settings)
               
               # Store settings
               self._scan_settings = settings
               self._back_scan_settings = None
               
       def configure_back_scan(self, settings):
           """Configure the back scan."""
           with self._thread_lock:
               if self.module_state() != 'idle':
                   raise RuntimeError('Cannot configure back scan while scanning is in progress.')
               if self._scan_settings is None:
                   raise RuntimeError('Configure forward scan first.')
                   
               # Check settings
               self.constraints.check_back_scan_settings(
                   backward_settings=settings, 
                   forward_settings=self._scan_settings
               )
               
               # Store settings
               self._back_scan_settings = settings
               
       def move_absolute(self, position, velocity=None, blocking=False):
           """Move to absolute position."""
           with self._thread_lock:
               if self.module_state() != 'idle':
                   raise RuntimeError('Cannot move while scanning is in progress.')
               
               # Check position is valid
               for axis, pos in position.items():
                   self.constraints.axes[axis].position.check(pos)
               
               # Update position
               self._current_position.update(position)
               
               return self._current_position.copy()
               
       def move_relative(self, distance, velocity=None, blocking=False):
           """Move by relative distance."""
           with self._thread_lock:
               if self.module_state() != 'idle':
                   raise RuntimeError('Cannot move while scanning is in progress.')
               
               # Calculate new position
               new_pos = {}
               for axis, dist in distance.items():
                   new_pos[axis] = self._current_position[axis] + dist
                   
                   # Check position is valid
                   self.constraints.axes[axis].position.check(new_pos[axis])
               
               # Update position
               self._current_position.update(new_pos)
               
               return self._current_position.copy()
               
       def get_target(self):
           """Get current target position."""
           return self._current_position.copy()
           
       def get_position(self):
           """Get current actual position."""
           return self._current_position.copy()
           
       def start_scan(self):
           """Start a configured scan."""
           with self._thread_lock:
               if self.module_state() != 'idle':
                   raise RuntimeError('Cannot start scan while scanning is in progress.')
               if self._scan_settings is None:
                   raise RuntimeError('No scan configured.')
               
               # Lock module state
               self.module_state.lock()
               
               # Initialize scan data
               self._scan_data = ScanData.from_constraints(
                   settings=self._scan_settings,
                   constraints=self.constraints,
                   scanner_target_at_start=self.get_target()
               )
               self._scan_data.new_scan()
               
               # Initialize back scan data if configured
               if self._back_scan_settings is not None:
                   self._back_scan_data = ScanData.from_constraints(
                       settings=self._back_scan_settings,
                       constraints=self.constraints,
                       scanner_target_at_start=self.get_target()
                   )
                   self._back_scan_data.new_scan()
               
               # Calculate scan parameters from settings
               axes = self._scan_settings.axes
               ranges = self._scan_settings.range
               resolutions = self._scan_settings.resolution
               
               # Perform the scan
               if len(axes) == 1:
                   # 1D scan
                   axis = axes[0]
                   start = ranges[0][0]
                   end = ranges[0][1]
                   steps = resolutions[0]
                   
                   # Other position components
                   position = self._current_position.copy()
                   
                   # Scan along line
                   scan_positions = np.linspace(start, end, steps)
                   for i, pos in enumerate(scan_positions):
                       position[axis] = pos
                       counts = self._confocal_simulator.measure_fluorescence(
                           (position['x'], position['y'], position['z'])
                       )
                       
                       # Update scan data
                       data_dict = {'fluorescence': np.array([counts])}
                       for ch, arr in data_dict.items():
                           self._scan_data.data[ch][i] = arr
                       
               elif len(axes) == 2:
                   # 2D scan
                   x_axis, y_axis = axes
                   x_start, x_end = ranges[0]
                   y_start, y_end = ranges[1]
                   x_steps, y_steps = resolutions
                   
                   # Other position components
                   position = self._current_position.copy()
                   
                   # Scan plane
                   x_positions = np.linspace(x_start, x_end, x_steps)
                   y_positions = np.linspace(y_start, y_end, y_steps)
                   
                   for i, x_pos in enumerate(x_positions):
                       for j, y_pos in enumerate(y_positions):
                           position[x_axis] = x_pos
                           position[y_axis] = y_pos
                           
                           counts = self._confocal_simulator.measure_fluorescence(
                               (position['x'], position['y'], position['z'])
                           )
                           
                           # Update scan data
                           data_dict = {'fluorescence': np.array([[counts]])}
                           for ch, arr in data_dict.items():
                               self._scan_data.data[ch][i, j] = arr
               
               # Back scan if configured
               if self._back_scan_settings is not None:
                   # Implement back scan similarly to forward scan
                   pass
               
               # Unlock module state
               self.module_state.unlock()
               
       def stop_scan(self):
           """Stop the current scan."""
           with self._thread_lock:
               if self.module_state() != 'locked':
                   raise RuntimeError('No scan in progress.')
               
               self.module_state.unlock()
               
       def get_scan_data(self):
           """Return the scan data."""
           with self._thread_lock:
               if self._scan_data is None:
                   return None
               return self._scan_data.copy()
               
       def get_back_scan_data(self):
           """Return the back scan data."""
           with self._thread_lock:
               if self._back_scan_data is None:
                   return None
               return self._back_scan_data.copy()
               
       def emergency_stop(self):
           """Emergency stop the scanner."""
           with self._thread_lock:
               if self.module_state() == 'locked':
                   self.module_state.unlock()
   ```

5. Add the confocal scanner to the NV simulator device:
   ```python
   class NVSimulatorDevice:
       """Main simulator device integrating all interfaces."""
       
       def __init__(self):
           self._simulator = PhysicalNVModel()
           self.microwave = NVSimulatorMicrowave(self._simulator)
           self.scanner = NVSimulatorScanner(self._simulator)
           self.confocal = ConfocalSimulatorScanner()  # New confocal interface
           
           # Configuration
           self._experiment_modes = {
               'odmr': ODMRMode(self._simulator),
               'rabi': RabiMode(self._simulator),
               'confocal': ConfocalMode(self.confocal)  # New confocal mode
           }
   ```

### Testing Strategy
1. Test NV center spatial distribution in diamond lattice
2. Verify focused laser beam model against theoretical profiles
3. Test confocal point spread function and optical resolution
4. Validate integration with Qudi by performing test scans
5. Compare simulated confocal images with experimental data

## Technical Risks
1. Computational performance with large numbers of NV centers
2. Accuracy of the confocal PSF model
3. Integration complexity with the Qudi scanning probe interface
4. Memory usage for high-resolution 3D scans

## Effort Estimation
- DiamondLattice implementation: 2 days
- FocusedLaserBeam model: 1.5 days
- ConfocalSimulator: 3 days
- Qudi ScanningProbeInterface implementation: 3 days
- Integration and testing: 2.5 days
- Total: 12 days