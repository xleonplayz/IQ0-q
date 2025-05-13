# TS-111: SimOS Time Evolution Implementation

## Summary
This technical story describes how SimOS implements time evolution of quantum systems - one of the core features used by the NV simulator. SimOS provides robust algorithms for evolving quantum states under static and time-dependent Hamiltonians with both unitary and non-unitary (dissipative) dynamics.

## Time Evolution Implementation

### Core Evolution Function
The primary time evolution method in SimOS is the `evol()` function in `propagation.py`:

```python
def evol(H, t, *rho, c_ops=[], proj=[], wallclock='global'):
    """Evolve state rho under static conditions for time t and return rho'."""
```

This function implements:
1. **Unitary evolution** under a time-independent Hamiltonian
2. **Lindblad master equation** evolution when collapse operators are provided
3. **Time tracking** via a global or local wallclock to manage system time

### Key Implementation Details

#### 1. Unitary Evolution
For Hilbert space evolution (no collapse operators), SimOS calculates:

```python
# Calculate propagator U(t) = exp(-i*H*t)
Ut = (-I*t*H)
if hasattr(H,'expm'):
    Ut = Ut.expm()
else:
    Ut = expm(Ut)

# Apply to state: either ψ' = U ψ (for pure states)
# or ρ' = U ρ U† (for mixed states)
```

The matrix exponential is computed using a highly optimized implementation:
- For small matrices: Custom algorithm with Schur decomposition (`fast_expm.py`)
- For larger matrices: Backend-specific algorithms (QuTiP, NumPy, etc.)

#### 2. Dissipative Evolution
For open quantum systems (with collapse operators), SimOS solves the Lindblad master equation:

```python
# Construct Liouvillian superoperator L
liouvillian = getattr(getattr(backends,method), 'liouvillian')
L = liouvillian(H,c_ops)

# Calculate the propagator in superoperator form: exp(Lt)
UL = (L*t)
UL = UL.expm()  # or expm(UL)

# Apply to density matrix
return applySuperoperator(UL,rho)
```

#### 3. Time-Dependent Evolution
For time-dependent Hamiltonians, the `prop()` function implements more advanced algorithms:

```python
def prop(H0,dt,*rho,c_ops=[],H1=None,carr1=None,c_ops2=[],carr2=None,engine='cpu')
```

This function supports:
- Static background Hamiltonian `H0`
- Time-dependent terms `H1` with coefficient arrays `carr1`
- Time-dependent collapse operators `c_ops2` with coefficient arrays `carr2`
- Multiple computation engines (CPU, QuTiP, Parament for GPU acceleration)

Time-dependent propagation uses the following strategies:
1. **Piecewise constant approximation**: Divide time into small steps where Hamiltonian is approximately constant
2. **Magnus expansion**: Higher-order integration to reduce errors for rapidly varying Hamiltonians
3. **Differential equation solvers**: RK45 integration for Lindblad dynamics with time-dependent terms

### Core Numerical Implementation

The core propagation for time-dependent Hamiltonians uses a sophisticated approach:

```python
def propagate_cpu(H0, H1, carr, dt, maxsize=1, **kwargs):
    # Optionally apply Magnus expansion for higher accuracy
    if magnus:
        H1 = generate_magnus_commutators(H0, H1, magnus, second_order)
        carr = generate_simpson(carr, dt, magnus, second_order)
        dt = dt*2

    # Initialize propagator as identity
    U = _np.eye(H0.shape[0], dtype=_np.complex128)
    
    # Process in batches to manage memory usage
    for i in range(0, pts, batchmax):
        # Calculate effective Hamiltonian for each time point
        Hc = _np.tensordot(carr[i:i+currlen], H1, axes=(0,0))
        Hc = Hc + H0
        
        # Calculate propagator for this batch via matrix exponential
        Uc = la.expm(-1j*dt*Hc)
        
        # Multiply propagators (time-ordered product)
        U = functools.reduce(_np.dot, Uc) @ U
        
    return U
```

### Tracking System Time

SimOS includes a `WallClock` to track the global time of a simulation:

```python
class WallClock:
    def __init__(self, start_time=0.0):
        self.time = start_time
    
    def inc(self, time):
        self.time += time
    
    def phase(self, freq, is_angular=False):
        if is_angular:
            return 2*_np.pi*freq*self.time
        else:
            return freq*self.time
```

This allows:
1. Maintaining phase coherence across multiple operations
2. Tracking total simulation time
3. Supporting realistic time-dependent control sequences

## Advanced Features

### Specialized Pulse Sequences
SimOS implements efficient specialized functions for common pulse sequences:

```python
def square_pulse(H0, Hrf, f, phase, amplitude, dur, *rho, pts_per_cycle=100):
    """Apply a square pulse with given frequency, phase, amplitude and duration."""
```

### Coordinate Transformations
The library supports rotating frames and coordinate transformations:

```python
def rotate_operator(system, dm, *args):
    """Global system rotation/coordinate transformation."""
```

### Optimized Matrix Exponential
SimOS uses a highly optimized matrix exponential implementation:

```python
def expm(a):
    """Optimized matrix exponential using Schur decomposition."""
    exp = expm2(a)  # Call compiled numba implementation
    if use_qutip:
        return qu.Qobj(exp, dims=qutip_a.dims)
    else:
        return exp
```

## Integration with NV Simulator

The NV simulator leverages SimOS time evolution capabilities for:

1. **ODMR Simulations**: Evolving under microwave drives of varying frequencies
2. **Rabi Oscillations**: Time evolution under resonant microwave fields
3. **Relaxation Measurements**: T1 and T2 dynamics with decoherence
4. **Pulse Sequences**: Implementing complex control sequences for advanced experiments

Key integration points:
- `PhysicalNVModel.evolve()` method directly calls SimOS `evol()` function
- `PhysicalNVModel.simulate_*` methods use SimOS for quantum-accurate simulations
- Fallback mechanisms when SimOS is unavailable

## Performance Considerations

The SimOS time evolution algorithms include several optimizations:

1. **Memory management**: Batched processing for large systems
2. **Backend selection**: Automatic choice of optimal matrix operations
3. **Compiled code**: Fast matrix operations with Numba
4. **GPU acceleration**: Support for GPU computation via Parament
5. **Selective Magnus expansion**: Higher-order integration when accuracy is critical

## Conclusion

SimOS provides a comprehensive, physically accurate, and computationally efficient framework for quantum time evolution. Its implementation supports both unitary and non-unitary dynamics, with specialized functions for common pulse sequences in NV center experiments. This allows the NV simulator to model realistic quantum dynamics with high fidelity, enabling accurate predictions of experimental results.