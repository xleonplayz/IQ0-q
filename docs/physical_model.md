# NV Center Physical Model
*Authored by Leon Kaiser*

## Quantum Physics of NV Centers

The Nitrogen-Vacancy (NV) center in diamond is a point defect consisting of a substitutional nitrogen atom adjacent to a vacancy in the carbon lattice. NV centers exist in different charge states, with the negatively charged state (NV-) being particularly interesting for quantum applications due to its electronic structure and optical properties.

### Electronic Structure

The NV- center has a spin-1 ground state (³A₂) with three spin sublevels (ms = 0, ±1) that can be manipulated using microwave radiation. The key properties that make NV centers valuable for quantum applications include:

1. **Zero-Field Splitting (ZFS)**: Even in the absence of an external magnetic field, the ms = ±1 states are separated from the ms = 0 state by approximately 2.87 GHz due to the electronic structure and crystal field.

2. **Zeeman Effect**: When an external magnetic field is applied, the energy levels of the ms = ±1 states split according to their interaction with the field. The splitting is proportional to the field strength.

3. **Optical Properties**: The NV center can be optically excited using green light (typically 532 nm), causing a transition to the excited state (³E). The subsequent relaxation back to the ground state is spin-dependent, allowing for optical initialization and readout of the spin state.

4. **Long Coherence Times**: The spin states of the NV center can have long coherence times (up to milliseconds at room temperature), making them suitable for quantum information processing.

### Key Physical Interactions

The full Hamiltonian of the NV center includes several terms:

- **Zero-Field Splitting**: D·S²z term (~2.87 GHz)
- **Zeeman Interaction**: γe·B·S term (interaction with magnetic field)
- **Strain Effects**: E·(S²x - S²y) term (crystal strain)
- **Hyperfine Interaction**: A·I·S term (interaction with nitrogen nuclear spin)
- **Electric Field Effects**: d_⊥·E_⊥·S term (susceptibility to electric fields)
- **Microwave Driving**: H_MW·cos(ωt) term (controlled manipulation)

## Implementation Details

Our `PhysicalNVModel` class provides a comprehensive simulation of NV center physics, including all these interactions and the dynamics they induce.

```
@file physical_model.py
@brief Quantum physical model of NV center in diamond

This module implements a comprehensive physical simulation of the 
Nitrogen-Vacancy (NV) center in diamond, including its quantum mechanical
properties, interactions with external fields, and optical dynamics.
```

## PhysicalNVModel Class

The `PhysicalNVModel` class is the core implementation of the NV center physics simulation.

```
@class PhysicalNVModel
@brief Comprehensive physical model of NV centers in diamond

This class implements a quantum mechanical simulation of an NV center,
including its interaction with magnetic fields, microwave radiation,
and laser excitation. It handles:

- Ground and excited state dynamics
- Zeeman splitting in magnetic fields 
- Zero-field splitting and strain effects
- Microwave-driven Rabi oscillations
- Optical cycling and spin polarization
- Quantum coherence and decoherence processes
- Hyperfine interaction with nitrogen nucleus
```

### Key Physical Methods

#### Hamiltonian Construction

```
@method _update_hamiltonian
@brief Update Hamiltonian based on current physical parameters
       
This method constructs the quantum Hamiltonian operator that governs
the NV center's energy levels and dynamics. The Hamiltonian includes:

1. Zero-field splitting (D term ~2.87 GHz)
2. Zeeman interaction with magnetic field
3. Strain effects in the crystal
4. Hyperfine coupling to nitrogen nucleus
5. Microwave driving terms with proper RWA

@note The rotating wave approximation (RWA) is used for the microwave
      driving terms to maintain accuracy while improving computational
      efficiency.
```

```python
def _update_hamiltonian(self):
    """Update Hamiltonian based on control parameters with complete physics.
    
    The Hamiltonian of an NV center contains several terms:
    
    H = D*Sz^2 + E*(Sx^2-Sy^2) + γe*(Bx*Sx + By*Sy + Bz*Sz) + A*I·S + Hmw
    
    Where:
    - D is the zero-field splitting (~2.87 GHz)
    - E is the strain splitting parameter
    - γe is the electron gyromagnetic ratio
    - Bx,By,Bz are the magnetic field components
    - A is the hyperfine coupling tensor
    - I is the nitrogen nuclear spin operator
    - S is the electron spin operator
    - Hmw is the microwave driving term
    """
    if self.is_mock:
        # Mock implementation remains for testing
        self._hamiltonian = np.eye(6)
        if self.mw_on:
            self._hamiltonian = self._hamiltonian + np.eye(6) * 0.1
        return
        
    # Get strain parameters from config
    strain_e = self.model.config.get('strain_e', 0.0)  # E strain component
    strain_d = self.model.config.get('strain_d', 0.0)  # D strain component
    
    # Create Hamiltonian with current fields AND strain
    self._hamiltonian = self.simos_nv.field_hamiltonian(
        Bvec=self.magnetic_field,
        strain=[strain_e, strain_d]  # Add strain parameters
    )
    
    # Add hyperfine interactions if N14/N15 nucleus is enabled
    if self.model.config.get('include_nitrogen_nucleus', False):
        # Get nuclear species and hyperfine parameters
        n_species = self.model.config.get('nitrogen_isotope', 14)
        a_parallel = self.model.config.get('hyperfine_coupling_parallel', 2.16e6)  # Hz
        a_perp = self.model.config.get('hyperfine_coupling_perpendicular', 2.7e6)  # Hz
        
        # Add hyperfine interaction term using SimOS API
        self._hamiltonian += self.simos_nv.hyperfine_hamiltonian(
            n_species=n_species,
            hyperfine_parallel=a_parallel,
            hyperfine_perpendicular=a_perp
        )
    
    # Add microwave Hamiltonian if active - using proper RWA
    if self.mw_on:
        # Convert from dBm to amplitude
        mw_amplitude = 10**(self.mw_power/20) * 1e-3  # Approximation
        
        # Calculate detuning from resonance
        zfs = self.model.config['zero_field_splitting']
        gamma = self.model.config['gyromagnetic_ratio']
        b_z = self.magnetic_field[2]
        
        # Resonance frequencies
        f_0_to_minus1 = zfs - gamma * b_z
        f_0_to_plus1 = zfs + gamma * b_z
        
        # Determine which transition is being driven
        if abs(self.mw_freq - f_0_to_minus1) < abs(self.mw_freq - f_0_to_plus1):
            detuning = self.mw_freq - f_0_to_minus1
            transition = "0-m"
        else:
            detuning = self.mw_freq - f_0_to_plus1
            transition = "0-p"
        
        # Apply proper RWA with detuning term
        if transition == "0-m":
            # Create RWA Hamiltonian for ms=0 to ms=-1 transition
            mw_H = (
                0.5 * mw_amplitude * self.simos_nv.Sx_0m +  # σx term
                2 * np.pi * detuning * self.simos_nv.Sz_0m   # Detuning term
            )
        else:
            # Create RWA Hamiltonian for ms=0 to ms=+1 transition
            mw_H = (
                0.5 * mw_amplitude * self.simos_nv.Sx_0p +  # σx term
                2 * np.pi * detuning * self.simos_nv.Sz_0p   # Detuning term
            )
            
        # Add the microwave driving term
        self._hamiltonian += mw_H
```

#### Quantum Decoherence

```
@method _get_c_ops
@brief Get collapse operators for quantum decoherence processes
       
This method constructs Lindblad collapse operators to model:

1. T1 relaxation (longitudinal/energy relaxation)
2. T2 dephasing (transverse/phase relaxation)
3. T2* inhomogeneous dephasing (ensemble effects)
4. Optical excitation and emission processes
5. Intersystem crossing (ISC) between singlet and triplet states

@note The Lindblad master equation provides a physically accurate
      description of open quantum system dynamics with decoherence.
```

```python
def _get_c_ops(self):
    """Get collapse operators based on current control parameters.
    
    In quantum mechanics, decoherence and relaxation processes are modeled using
    the Lindblad master equation:
    
    dρ/dt = -i[H,ρ] + Σ(j) γj(Lj·ρ·Lj† - 1/2{Lj†·Lj,ρ})
    
    Where:
    - ρ is the density matrix
    - H is the system Hamiltonian
    - γj are decoherence rates
    - Lj are Lindblad operators (collapse operators)
    
    For NV centers, these operators represent:
    1. T1 processes (energy relaxation): ms=±1 → ms=0
    2. T2 processes (pure dephasing): loss of phase coherence
    3. Optical transitions: ground ↔ excited state
    4. Intersystem crossing: spin-dependent transitions via singlet states
    """
    # Get collapse operators from the SimOS NV model
    T = self.model.config.get('temperature', 298)  # Room temperature in Kelvin by default
    
    if self.is_mock:
        # Mock implementation remains for testing
        if self.laser_on and self.laser_power > 0:
            # Higher number of operators with laser on
            return [np.eye(6) * 0.1 for _ in range(5)]
        else:
            # Fewer operators without laser
            return self._c_ops_laser_off if self._c_ops_laser_off else [np.eye(6) * 0.05 for _ in range(3)]
    
    # Get T1 and T2 parameters from config
    t1 = self.model.config.get('T1', 1e-3)  # Default 1 ms
    t2 = self.model.config.get('T2', 1e-6)  # Default 1 µs
    t2_star = self.model.config.get('T2_star', 0.5e-6)  # Default 500 ns
    
    # Initialize collapse operators list
    c_ops = []
    
    # Actual SimOS implementation
    if self.laser_on and self.laser_power > 0:
        # Normalize laser power to saturation value for SimOS beta parameter (0-1)
        saturation_power = self.model.config['excitation_saturation_power']
        beta = min(1.0, self.laser_power / saturation_power)
        
        # Get optical transition operators from SimOS
        optical_ops, relaxation_ops = self.simos_nv.transition_operators(
            T=T, beta=beta, Bvec=self.magnetic_field)
        
        # Add all optical operators
        c_ops.extend(optical_ops)
        
        # Cache relaxation operators for use when laser is off
        self._c_ops_laser_off = relaxation_ops
    else:
        # Relaxation operators (if cached, use those)
        if not self._c_ops_laser_off:
            # Get from SimOS
            _, self._c_ops_laser_off = self.simos_nv.transition_operators(
                T=T, beta=0, Bvec=self.magnetic_field)
    
    # Add T1 and T2 processes
    c_ops.extend(self._c_ops_laser_off)
    
    # Create additional pure dephasing operators based on T2 and T2*
    # For pure dephasing, we need the difference between total and population decay
    if t2 > 0:
        # Calculate pure dephasing rate
        # 1/T2 = 1/T2' + 1/(2*T1), thus 1/T2' = 1/T2 - 1/(2*T1)
        pure_dephasing_rate = 1/t2 - 1/(2*t1) if t1 > 0 else 1/t2
        
        if pure_dephasing_rate > 0:
            # Add pure dephasing collapse operators
            dephasing_ops = self.simos_nv.pure_dephasing_operators(rate=pure_dephasing_rate)
            c_ops.extend(dephasing_ops)
    
    # Handle inhomogeneous dephasing from T2* (using custom implementation)
    if hasattr(self.simos_nv, 'inhomogeneous_dephasing_operators') and t2_star > 0:
        # 1/T2* = 1/T2 + 1/Tinhom, thus 1/Tinhom = 1/T2* - 1/T2
        inhom_rate = 1/t2_star - 1/t2 if t2 > 0 else 1/t2_star
        
        if inhom_rate > 0:
            # Add inhomogeneous dephasing operators 
            inhom_ops = self.simos_nv.inhomogeneous_dephasing_operators(rate=inhom_rate)
            c_ops.extend(inhom_ops)
    
    return c_ops
```

#### Adaptive Timestep Algorithm

```
@method _get_optimal_timestep
@brief Calculate the optimal timestep for accurate quantum evolution
       
This method determines an appropriate timestep for the numerical
integration of the quantum master equation. It:

1. Analyzes the Hamiltonian's energy spectrum
2. Estimates the required time resolution based on the highest energy scale
3. Applies the Trotter error bound to ensure numerical accuracy
4. Adjusts the timestep to balance computational efficiency and precision

@note The timestep is crucial for accurate simulation of quantum dynamics,
      especially for systems with multiple energy scales.
```

```python
def _get_optimal_timestep(self, desired_accuracy=1e-6):
    """Calculate the optimal timestep for the current Hamiltonian.
    
    In quantum evolution simulation, the accuracy of numerical integration
    depends critically on the timestep relative to the system's energy scales.
    
    The Trotter error bound for evolution operators shows that the error scales as:
        Error ~ O(dt² * ||[H₁,H₂]||)
    
    where H₁ and H₂ are parts of the Hamiltonian and [H₁,H₂] is their commutator.
    
    For a general Hamiltonian, a conservative estimate is:
        Error ~ O(dt² * (max_energy)²)
    
    Therefore, to maintain accuracy ε, we need:
        dt ~ sqrt(ε) / max_energy
    
    Args:
        desired_accuracy: Target accuracy for the quantum evolution
        
    Returns:
        float: Optimal timestep in seconds
    """
    # Convert Hamiltonian to matrix form
    if hasattr(self._hamiltonian, 'to_matrix'):
        H_matrix = self._hamiltonian.to_matrix()
    else:
        # If it's already a matrix
        H_matrix = self._hamiltonian
    
    # Check if H_matrix is a sparse matrix
    if hasattr(H_matrix, 'toarray'):
        is_sparse = True
        nnz = H_matrix.nnz
        dimension = H_matrix.shape[0]
        
        # For large sparse matrices, use an iterative approach
        if nnz > 10000:
            from scipy.sparse.linalg import eigsh
            try:
                # Calculate the largest few eigenvalues
                largest_eigs = eigsh(H_matrix, k=min(6, dimension-1), which='LM', 
                                    return_eigenvectors=False)
                max_energy = max(abs(largest_eigs))
            except Exception:
                # Fallback if eigsh fails
                max_energy = np.max(np.abs(H_matrix.diagonal())) * 2
        else:
            # For smaller matrices, convert to dense
            H_matrix = H_matrix.toarray()
            is_sparse = False
    else:
        is_sparse = False
        dimension = H_matrix.shape[0]
    
    # For dense matrices, use appropriate algorithm based on size
    if not is_sparse:
        if dimension <= 100:
            # For small matrices, direct eigenvalue computation is efficient
            from scipy.linalg import eigvalsh
            try:
                eigs = eigvalsh(H_matrix)
                max_energy = max(abs(eigs))
            except Exception:
                # Fallback if eigenvalue computation fails
                max_energy = np.max(np.abs(np.diag(H_matrix))) * 2
        else:
            # For larger matrices, use partial eigenvalue calculation
            from scipy.linalg import eigh
            try:
                # Calculate the largest few eigenvalues
                largest_eigs = eigh(H_matrix, eigvals_only=True, 
                                  eigvals=(dimension-3, dimension-1))
                max_energy = max(abs(largest_eigs))
            except Exception:
                # Fallback if eigh fails
                max_energy = np.max(np.abs(np.diag(H_matrix))) * 2
    
    # Calculate the optimal timestep using a more accurate formula
    # Based on the Trotter error bound for evolution operator splitting
    if max_energy > 0:
        # Use sqrt relationship for better accuracy
        timestep = np.sqrt(desired_accuracy) / max_energy
        
        # Apply additional safety factor for numerical stability
        safety_factor = 0.8
        timestep *= safety_factor
        
        # Clamp to reasonable limits
        min_dt = 1e-12  # 1 ps
        max_dt = 1e-6   # 1 µs
        timestep = max(min_dt, min(max_dt, timestep))
    else:
        # If max_energy is zero (rare), use a reasonable default
        timestep = 1e-9  # 1 ns
    
    return timestep
```

#### State Evolution

```
@method evolve
@brief Evolve the quantum state for a specified duration
       
This method advances the quantum state by integrating the master
equation over a specified time interval. It:

1. Constructs the full Hamiltonian and collapse operators
2. Solves the Lindblad master equation for open quantum systems
3. Handles both unitary (coherent) and non-unitary (decoherent) evolution
4. Implements error handling and recovery strategies

@note The core of the quantum simulation, implementing the actual
      time propagation of the quantum state.
```

```python
def evolve(self, dt):
    """Evolve the quantum state for a specified duration.
    
    This method integrates the Lindblad master equation:
    
    dρ/dt = -i[H,ρ] + Σ(j) γj(Lj·ρ·Lj† - 1/2{Lj†·Lj,ρ})
    
    Where the first term represents coherent evolution governed by the
    Hamiltonian, and the second term accounts for decoherence processes.
    
    For closed quantum systems (no decoherence), this reduces to the
    Liouville-von Neumann equation:
    
    dρ/dt = -i[H,ρ]
    
    With formal solution:
    ρ(t) = e^(-iHt) ρ(0) e^(iHt)
    
    Args:
        dt: Time interval in seconds
        
    Raises:
        QuantumEvolutionError: If evolution fails
    """
    if dt <= 0:
        return

    try:
        # Get collapse operators
        c_ops = self._get_c_ops()
        
        # Evolve the state
        if len(c_ops) > 0:
            # With decoherence (Lindblad master equation)
            if hasattr(simos.propagation, 'mesolve'):
                try:
                    result = simos.propagation.mesolve(self._hamiltonian, self._rho, dt, c_ops)
                    self._rho = result.states[-1] if hasattr(result, 'states') else self._rho
                except Exception as e:
                    raise QuantumEvolutionError(f"Lindblad evolution failed: {str(e)}")
            else:
                # Simplified evolution if mesolve not available
                self._handle_microwave_transitions(dt)
                if self.laser_on:
                    self._handle_optical_processes(dt)
                self._handle_t1_relaxation(dt)
        else:
            # Without decoherence (unitary evolution)
            try:
                propagator = simos.propagation.evol(self._hamiltonian, dt)
                if hasattr(propagator, 'dag'):  # Check if dag method exists
                    self._rho = propagator * self._rho * propagator.dag()
                else:
                    # Simplified evolution if proper operators not available
                    self._handle_microwave_transitions(dt)
            except Exception as e:
                raise QuantumEvolutionError(f"Unitary evolution failed: {str(e)}")
        
        # Update populations and time
        self._update_populations()
        self.simulation_time += dt
    except Exception as e:
        # Log the error with detailed context
        logger.error(f"Error in evolve: {e}")
        logger.debug(f"State: mw_on={self.mw_on}, laser_on={self.laser_on}, B={self.magnetic_field}")
        
        # Try to recover to a valid state
        if isinstance(e, QuantumEvolutionError):
            # For known evolution errors, fall back to simplified evolution
            try:
                # Fallback to simplified evolution
                self._handle_microwave_transitions(dt)
                if self.laser_on:
                    self._handle_optical_processes(dt)
                self._handle_t1_relaxation(dt)
                self.simulation_time += dt
                logger.info("Recovered using simplified evolution")
            except Exception as recovery_error:
                # If even recovery fails, reset state and raise detailed error
                logger.critical(f"Recovery failed: {recovery_error}")
                self.reset_state()
                raise QuantumEvolutionError(f"Evolution failed and recovery failed. System reset. Original error: {str(e)}")
        else:
            # For unknown errors, reset state and raise
            self.reset_state()
            raise QuantumEvolutionError(f"Unhandled error during evolution: {str(e)}")
```

## Error Handling

```
@class SimOSImportError
@brief Error raised when SimOS cannot be imported or initialized properly

@class PhysicalModelError
@brief Base class for all physical model errors

@class InvalidConfigurationError
@brief Error raised when configuration is invalid 

@class QuantumEvolutionError
@brief Error raised during quantum state evolution

@class HamiltonianError
@brief Error raised when Hamiltonian cannot be constructed
```

```python
class SimOSImportError(Exception):
    """Error raised when SimOS cannot be imported or initialized properly.
    
    This error indicates that the SimOS library could not be imported or
    properly initialized. SimOS (Simulation of Optically-addressable Spins)
    is a critical dependency for accurate physical simulation of NV centers.
    """
    pass

class PhysicalModelError(Exception):
    """Base class for all physical model errors.
    
    This is the parent class for all exceptions related to the physical
    modeling of the NV center. Specific errors inherit from this class
    to maintain a consistent error hierarchy.
    """
    pass

class InvalidConfigurationError(PhysicalModelError):
    """Error raised when configuration is invalid.
    
    This error indicates that the model configuration contains invalid
    or incompatible parameters. For example, negative relaxation times
    or physically impossible field values.
    """
    pass

class QuantumEvolutionError(PhysicalModelError):
    """Error raised during quantum state evolution.
    
    This error indicates a problem during the time evolution of the
    quantum state, such as numerical instability, invalid state,
    or failure in the ODE integration.
    """
    pass

class HamiltonianError(PhysicalModelError):
    """Error raised when Hamiltonian cannot be constructed.
    
    This error indicates a problem with constructing or manipulating
    the system Hamiltonian, such as incompatible operators or
    incorrect parameter ranges.
    """
    pass
```

## Experiments

The simulator provides implementations of standard NV center experiments:

- Rabi oscillations (coherent spin control)
- Ramsey interferometry (measure T2*)
- Spin echo (measure T2)
- ODMR spectroscopy (optically detected magnetic resonance)
- T1 relaxometry
- Advanced pulse sequences (CPMG, XY-N)

These implementations follow the standard protocols used in experimental quantum physics and can be used to validate the simulator against real experimental data.