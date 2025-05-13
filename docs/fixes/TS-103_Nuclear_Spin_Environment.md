# TS-103: Nuclear Spin Environment Implementation

## Summary
Enhance the NV center simulator to include interactions with surrounding nuclear spins, primarily carbon-13 and nitrogen-14/15 nuclei, enabling quantum-accurate simulation of hyperfine interactions, coherence effects, and spin bath dynamics.

## Motivation
Real NV centers in diamond interact strongly with nearby nuclear spins, which significantly impact quantum coherence properties, sensing capabilities, and quantum information processing. Currently, the simulator lacks a realistic spin bath environment, which limits its ability to model practical NV center experiments like DEER, NMR sensing, and coherence time measurements.

## Description
This technical story adds the capability to create and simulate realistic nuclear spin environments around the NV center using the SimOS backend, enabling quantum-accurate modeling of hyperfine interactions, decoherence effects, and advanced quantum sensing protocols.

### Requirements
1. Implement configurable nuclear spin bath environment
   - 13C nuclear spins in random or predefined positions
   - Nitrogen nuclear spin with appropriate hyperfine coupling
   - Customizable spin bath parameters

2. Model accurate interaction Hamiltonians
   - Hyperfine coupling tensors
   - Nuclear-nuclear dipolar coupling
   - Quadrupolar interactions for I > 1/2 nuclei

3. Support simulation of nuclear spin manipulation
   - RF pulses for nuclear spin control
   - Dynamic nuclear polarization
   - DEER (Double Electron-Electron Resonance) experiments

4. Provide realistic decoherence modeling due to spin bath dynamics
   - T2* effects from quasi-static nuclear fields
   - T2 decay due to nuclear spin flip-flops
   - Spectral diffusion

### Implementation Details
1. Create a dedicated module for nuclear spin environment:
   ```
   src/
     nuclear_environment/
       __init__.py
       spin_bath.py
       hyperfine.py
       nuclear_control.py
       decoherence_models.py
   ```

2. Implement the nuclear spin bath class:
   ```python
   class NuclearSpinBath:
       def __init__(self, concentration=0.011, bath_size=10, random_seed=None):
           """
           Initialize a nuclear spin bath environment.
           
           Parameters:
           -----------
           concentration : float
               Natural abundance or chosen concentration of 13C (0-1)
           bath_size : int
               Number of nuclear spins to include
           random_seed : int
               Random seed for reproducible positions
           """
           self._concentration = concentration
           self._bath_size = bath_size
           self._rng = np.random.RandomState(random_seed)
           self._spins = []
           self._generate_spin_positions()
           
       def _generate_spin_positions(self):
           """Generate random positions for nuclear spins in diamond lattice"""
           # Diamond lattice constants and structure
           lattice_constant = 3.57e-10  # meters
           # Generate positions using diamond lattice structure
           # ...
           
       def add_custom_spin(self, position, species='13C'):
           """Add a specific nuclear spin at a given position"""
           self._spins.append({
               'position': position,
               'species': species,
               'index': len(self._spins)
           })
           
       def create_simos_system(self):
           """Create a SimOS system with the nuclear spin bath"""
           # Create an NV system with included nuclear spins
           from simos.systems.NV import NVSystem
           
           # Create additional spins for SimOS
           further_spins = []
           for spin in self._spins:
               if spin['species'] == '13C':
                   further_spins.append({
                       'val': 1/2, 
                       'name': f'C{spin["index"]}',
                       'type': '13C',
                       'pos': spin['position']
                   })
               elif spin['species'] == '14N':
                   further_spins.append({
                       'val': 1, 
                       'name': f'N{spin["index"]}',
                       'type': '14N',
                       'pos': spin['position']
                   })
               # Add other nuclear species as needed
           
           # Create NV system with nuclear spins
           nv_system = NVSystem(
               optics=True, 
               orbital=False,
               nitrogen=True,  # Default N spin
               natural=False,  # Use 15N by default
               further_spins=further_spins,
               method='qutip'
           )
           
           return nv_system
           
       def calculate_hyperfine_hamiltonian(self, nv_system):
           """Calculate the hyperfine interaction Hamiltonian"""
           # Use SimOS functions to calculate hyperfine couplings
           from simos.systems.NV import auto_pairwise_coupling
           
           # Get pairwise couplings
           h_hyperfine = auto_pairwise_coupling(
               nv_system, 
               approx=False,
               only_to_NV=False
           )
           
           return h_hyperfine
   ```

3. Extend PhysicalNVModel to support nuclear spin environment:
   ```python
   class PhysicalNVModel:
       # ... existing code ...
       
       def __init__(self, **config):
           # ... existing code ...
           
           # Nuclear environment configuration
           self.nuclear_bath = None
           self._nuclear_enabled = self.config.get("nuclear_spins", False)
           
           if self._nuclear_enabled:
               conc = self.config.get("c13_concentration", 0.011)
               bath_size = self.config.get("bath_size", 10)
               self.nuclear_bath = NuclearSpinBath(
                   concentration=conc,
                   bath_size=bath_size
               )
               
               # Override the default NV system with one including nuclear spins
               if self.nuclear_bath is not None:
                   self._nv_system = self.nuclear_bath.create_simos_system()
                   
                   # Update Hamiltonian to include hyperfine interactions
                   self._update_hamiltonian()
       
       def _update_hamiltonian(self):
           """Update the system Hamiltonian with nuclear spin interactions"""
           # ... existing code ...
           
           # Add hyperfine interactions if nuclear bath exists
           if self.nuclear_bath is not None:
               h_hyperfine = self.nuclear_bath.calculate_hyperfine_hamiltonian(self._nv_system)
               self._H_free = self._H_free + h_hyperfine
           
           # ... rest of existing code ...
       
       def apply_rf_pulse(self, frequency, power, duration, target_nuclear='13C'):
           """
           Apply an RF pulse to manipulate nuclear spins
           
           Parameters:
           -----------
           frequency : float
               RF frequency in Hz
           power : float
               RF power in W
           duration : float
               Pulse duration in seconds
           target_nuclear : str
               Target nuclear species ('13C', '14N', '15N')
           """
           if not self._nuclear_enabled:
               raise ValueError("Nuclear spin environment not enabled")
               
           # Calculate RF Hamiltonian for nuclear spin manipulation
           # ...
           
           # Apply RF pulse using SimOS evolution
           # ...
       
       def simulate_deer(self, tau_values, target_nuclear='13C'):
           """
           Simulate DEER (Double Electron-Electron Resonance) experiment
           
           Parameters:
           -----------
           tau_values : array
               Array of tau delay times to simulate
           target_nuclear : str
               Target nuclear species for DEER
               
           Returns:
           --------
           SimulationResult
               Results including DEER signal and analysis
           """
           if not self._nuclear_enabled:
               raise ValueError("Nuclear spin environment not enabled")
               
           # Initialize results
           deer_signal = np.zeros_like(tau_values)
           
           # For each tau value, run the DEER sequence
           for i, tau in enumerate(tau_values):
               # Create DEER sequence
               # ... sequence code ...
               
               # Simulate and get signal
               result = self._simulate_sequence(sequence)
               deer_signal[i] = result.get_population('ms0')
           
           # Return results
           return SimulationResult(
               type="DEER",
               tau_values=tau_values,
               signal=deer_signal,
               target_nuclear=target_nuclear
           )
   ```

4. Implement decoherence models due to spin bath dynamics:
   ```python
   class SpinBathDecoherence:
       def __init__(self, spin_bath):
           self.spin_bath = spin_bath
           
       def calculate_t2_star(self, magnetic_field):
           """Calculate T2* due to random nuclear fields"""
           # Implement CCE (cluster correlation expansion) or other methods
           # ...
           
       def calculate_spectral_diffusion(self, tau_values):
           """Calculate spectral diffusion contribution to decoherence"""
           # ...
   ```

### API Integration
The nuclear spin capabilities will be accessible through configuration parameters when creating the NV model:

```python
# Create model with nuclear spin environment
model = PhysicalNVModel(
    nuclear_spins=True,
    c13_concentration=0.011,  # Natural abundance
    bath_size=20,             # Number of nuclear spins to include
    include_n14=True          # Include host nitrogen
)

# Simulate DEER experiment
tau_values = np.linspace(0, 50e-6, 51)  # 0 to 50 µs
deer_result = model.simulate_deer(tau_values, target_nuclear='13C')

# Visualize results
plt.figure(figsize=(10, 6))
plt.plot(tau_values * 1e6, deer_result.signal)
plt.xlabel('Tau (µs)')
plt.ylabel('DEER Signal')
plt.grid(True)
```

### Testing Strategy
1. Verify nuclear spin coupling calculations against analytical expectations
2. Compare simulated DEER, ESEEM, and NMR signals with experimental data
3. Validate decoherence curves and T2* estimates against literature values
4. Test scalability with increasing bath size

## Technical Risks
1. Computational complexity scaling with nuclear spin bath size
2. Numerical stability with many-body interactions
3. Memory requirements for large nuclear spin systems
4. Trade-off between accuracy and performance

## Effort Estimation
- Nuclear spin bath implementation: 2.5 days
- Hyperfine interaction modeling: 2 days
- Nuclear control pulse implementation: 1.5 days
- Advanced experiment protocols: 2 days
- Decoherence models: 2 days
- Testing and validation: 2 days
- Total: 12 days