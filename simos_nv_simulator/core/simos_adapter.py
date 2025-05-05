"""
SimOS adapter module for the NV simulator.

This module provides a robust adapter layer between the SimOS quantum simulation
library and the NV-center specific simulator. It ensures clean handling of
dependencies, proper error propagation, and efficient integration of the
underlying quantum mechanical models.
"""

import os
import sys
import importlib
import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union, Callable

from .exceptions import (
    SimOSImportError, ConfigurationError, HamiltonianError, 
    QuantumStateError, EvolutionError
)

# Configure logging
logger = logging.getLogger(__name__)


class SimOSVersionValidator:
    """
    Validates the installed SimOS version for compatibility.
    
    This class checks if the installed SimOS version is compatible with
    the NV simulator and provides detailed information about compatibility
    issues when they arise.
    """
    
    # Minimum required version
    MIN_VERSION = (0, 5, 0)
    
    # Features required from SimOS
    REQUIRED_MODULES = [
        "simos.nv",
        "simos.propagation",
        "simos.operators",
        "simos.tools"
    ]
    
    REQUIRED_ATTRIBUTES = {
        "simos.nv": ["create_nv_system", "NVSystemConfig"],
        "simos.propagation": ["mesolve", "evol", "sesolve"],
        "simos.operators": ["sigmaz", "sigmax", "sigmay"]
    }
    
    @classmethod
    def get_version(cls, simos) -> Tuple[int, int, int]:
        """
        Get the version of the installed SimOS package.
        
        Parameters
        ----------
        simos : module
            The imported SimOS module
            
        Returns
        -------
        tuple
            Version number as (major, minor, patch)
        """
        if not hasattr(simos, "__version__"):
            # If no version attribute, assume old version
            return (0, 0, 0)
        
        # Parse version string (expected format: "0.5.0")
        try:
            version_str = simos.__version__
            parts = version_str.split(".")
            return tuple(int(part) for part in parts[:3])
        except (ValueError, AttributeError, IndexError):
            logger.warning(f"Could not parse SimOS version: {getattr(simos, '__version__', 'unknown')}")
            return (0, 0, 0)
    
    @classmethod
    def check_imports(cls, simos) -> List[str]:
        """
        Check if all required modules and attributes are available.
        
        Parameters
        ----------
        simos : module
            The imported SimOS module
            
        Returns
        -------
        list
            List of missing modules/attributes, empty if all requirements are met
        """
        missing = []
        
        # Check required modules
        for module_name in cls.REQUIRED_MODULES:
            try:
                # Try to import the module
                if "." in module_name:
                    parent, child = module_name.split(".", 1)
                    if not hasattr(simos, child):
                        missing.append(module_name)
                else:
                    importlib.import_module(module_name)
            except ImportError:
                missing.append(module_name)
        
        # Check required attributes
        for module_name, attributes in cls.REQUIRED_ATTRIBUTES.items():
            try:
                if "." in module_name:
                    parent, child = module_name.split(".", 1)
                    module = getattr(simos, child, None)
                else:
                    module = importlib.import_module(module_name)
                
                if module:
                    for attr in attributes:
                        if not hasattr(module, attr):
                            missing.append(f"{module_name}.{attr}")
            except ImportError:
                # Module already reported as missing
                pass
        
        return missing
    
    @classmethod
    def validate(cls, simos) -> Tuple[bool, List[str]]:
        """
        Validate the SimOS installation.
        
        Parameters
        ----------
        simos : module
            The imported SimOS module
            
        Returns
        -------
        tuple
            (is_valid, issues) where is_valid is a boolean and issues is a list of strings
        """
        issues = []
        
        # Check version
        version = cls.get_version(simos)
        if version < cls.MIN_VERSION:
            version_str = ".".join(str(v) for v in version)
            min_version_str = ".".join(str(v) for v in cls.MIN_VERSION)
            issues.append(f"SimOS version {version_str} is older than required minimum {min_version_str}")
        
        # Check imports
        missing = cls.check_imports(simos)
        if missing:
            issues.append(f"Missing required SimOS components: {', '.join(missing)}")
        
        return len(issues) == 0, issues


class SimOSFinder:
    """
    Locates and imports the SimOS package.
    
    This class handles finding and importing the SimOS package from various
    potential locations, ensuring proper error handling and useful debugging
    information when the package cannot be found.
    """
    
    @staticmethod
    def find_simos_repo() -> Optional[Path]:
        """
        Find the SimOS repository directory by checking various possible locations.
        
        Returns
        -------
        Path or None
            Path to SimOS repository if found, None otherwise
        """
        # List of potential locations to search for simos_repo
        potential_paths = [
            # Direct relative path from this file
            Path(__file__).parent.parent.parent / "simos_repo",
            # Current working directory
            Path.cwd() / "simos_repo",
            # Parent of current working directory (for CI environments)
            Path.cwd().parent / "simos_repo",
            # Sibling directory to the project (common setup)
            Path(__file__).parent.parent.parent.parent / "simos_repo",
        ]
        
        # Check each path
        for path in potential_paths:
            if path.exists() and (path / "simos").exists() and (path / "simos" / "__init__.py").exists():
                return path
        
        return None
    
    @staticmethod
    def add_simos_to_path(simos_path: Path) -> None:
        """
        Add the SimOS package to the Python path.
        
        Parameters
        ----------
        simos_path : Path
            Path to the SimOS repository
        """
        if str(simos_path) not in sys.path:
            logger.debug(f"Adding SimOS path to sys.path: {simos_path}")
            sys.path.insert(0, str(simos_path))
    
    @classmethod
    def import_simos(cls) -> Any:
        """
        Import the SimOS package.
        
        Returns
        -------
        module
            The imported SimOS module
            
        Raises
        ------
        SimOSImportError
            If SimOS cannot be imported or is not compatible
        """
        # First try to import directly (in case it's installed)
        try:
            import simos
            logger.debug("SimOS imported from installed package")
            return simos
        except ImportError:
            logger.debug("SimOS not installed as package, searching in local directories")
        
        # Try to find and import from repository
        simos_path = cls.find_simos_repo()
        if simos_path:
            cls.add_simos_to_path(simos_path)
            try:
                import simos
                logger.debug(f"SimOS imported from repository: {simos_path}")
                return simos
            except ImportError as e:
                error_msg = f"Found SimOS repository at {simos_path} but failed to import: {str(e)}"
                logger.error(error_msg)
                raise SimOSImportError(
                    error_msg, 
                    import_path=str(simos_path),
                    recommendations=[
                        "Check that the SimOS repository is complete and initialized",
                        "Ensure all SimOS dependencies are installed",
                        "Try installing SimOS directly: pip install -e /path/to/simos_repo"
                    ]
                )
        
        # Failed to find or import SimOS
        error_msg = "Could not find or import SimOS"
        logger.error(error_msg)
        raise SimOSImportError(
            error_msg, 
            recommendations=[
                "Clone the SimOS repository to one of the expected locations",
                "Install SimOS using pip: pip install simos",
                "Set up a symlink to the SimOS repository"
            ]
        )


class SimOSNVAdapter:
    """
    Adapter for the SimOS NV center quantum simulation functionality.
    
    This class provides a consistent interface to the SimOS quantum functionality
    required for NV center simulations, handling differences in SimOS versions
    and providing graceful error handling.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the SimOS NV adapter.
        
        Parameters
        ----------
        config : dict, optional
            Configuration parameters for the NV system
            
        Raises
        ------
        SimOSImportError
            If SimOS cannot be imported or is not compatible
        ConfigurationError
            If the provided configuration is invalid
        """
        self.config = config or {}
        
        # Import SimOS
        try:
            self.simos = SimOSFinder.import_simos()
        except SimOSImportError as e:
            logger.critical("Failed to import SimOS - simulator cannot function properly")
            raise e
        
        # Validate SimOS version
        is_valid, issues = SimOSVersionValidator.validate(self.simos)
        if not is_valid:
            error_msg = "SimOS installation is not compatible with this simulator"
            logger.error(f"{error_msg}: {', '.join(issues)}")
            raise SimOSImportError(error_msg, recommendations=issues)
        
        # Create NV system
        try:
            self._initialize_nv_system()
        except Exception as e:
            logger.error(f"Failed to initialize NV system: {str(e)}")
            if isinstance(e, (SimOSImportError, ConfigurationError)):
                raise
            raise ConfigurationError(
                f"Failed to initialize NV system: {str(e)}",
                parameter="config", 
                value=self.config
            ) from e
            
        logger.info("SimOS NV adapter initialized successfully")
    
    def _initialize_nv_system(self) -> None:
        """
        Initialize the NV system using SimOS.
        
        This method sets up the NV system with the appropriate configuration,
        initializing all necessary operators and states.
        
        Raises
        ------
        ConfigurationError
            If the provided configuration is invalid
        """
        # Get the NV configuration class from SimOS
        try:
            NVSystemConfig = self.simos.nv.NVSystemConfig
        except AttributeError:
            raise SimOSImportError(
                "Missing NVSystemConfig class in SimOS",
                recommendations=["Update to a newer version of SimOS"]
            )
        
        # Set up default configuration parameters if not provided
        default_config = {
            "zero_field_splitting": 2.87e9,  # Hz
            "gyromagnetic_ratio": 28.0e9,    # Hz/T
            "strain_e": 0.0,                 # Hz (E strain)
            "strain_d": 0.0,                 # Hz (D strain)
            "temperature": 298,              # K
            "include_excited_state": True,
            "include_nitrogen_nucleus": False,
            "nitrogen_isotope": 14           # 14N or 15N
        }
        
        # Update defaults with provided config
        for key, value in self.config.items():
            default_config[key] = value
        
        # Clean and validate configuration
        nv_config = self._prepare_nv_config(default_config)
        
        # Create the NV system
        try:
            self.nv_system = self.simos.nv.create_nv_system(nv_config)
        except Exception as e:
            raise ConfigurationError(
                f"Failed to create NV system with SimOS: {str(e)}",
                parameter="nv_config",
                value=nv_config
            ) from e
        
        # Initialize quantum operators
        self._initialize_operators()
    
    def _prepare_nv_config(self, config: Dict[str, Any]) -> Any:
        """
        Prepare and validate the NV system configuration.
        
        Parameters
        ----------
        config : dict
            Raw configuration parameters
            
        Returns
        -------
        NVSystemConfig
            SimOS NV system configuration object
            
        Raises
        ------
        ConfigurationError
            If the configuration is invalid
        """
        # Validate key parameters
        for key, value in config.items():
            if key == "zero_field_splitting" and (value <= 0 or value > 1e11):
                raise ConfigurationError(
                    f"Invalid zero_field_splitting value: {value}",
                    parameter=key,
                    value=value,
                    valid_range="(0, 1e11) Hz"
                )
            elif key == "gyromagnetic_ratio" and (value <= 0 or value > 1e12):
                raise ConfigurationError(
                    f"Invalid gyromagnetic_ratio value: {value}",
                    parameter=key,
                    value=value,
                    valid_range="(0, 1e12) Hz/T"
                )
            elif key == "temperature" and (value <= 0 or value > 1000):
                raise ConfigurationError(
                    f"Invalid temperature value: {value}",
                    parameter=key,
                    value=value,
                    valid_range="(0, 1000) K"
                )
            elif key == "nitrogen_isotope" and value not in (14, 15):
                raise ConfigurationError(
                    f"Invalid nitrogen_isotope value: {value}",
                    parameter=key,
                    value=value,
                    valid_range="14 or 15"
                )
        
        # Create SimOS configuration object
        try:
            NVSystemConfig = self.simos.nv.NVSystemConfig
            nv_config = NVSystemConfig(
                D=config["zero_field_splitting"],
                gamma_e=config["gyromagnetic_ratio"],
                temperature=config["temperature"],
                include_es=config["include_excited_state"],
                include_nitrogen=config["include_nitrogen_nucleus"],
                nitrogen_isotope=config["nitrogen_isotope"],
                strain=[config["strain_e"], config["strain_d"]]
            )
            return nv_config
        except (AttributeError, TypeError) as e:
            raise SimOSImportError(
                f"Error creating NVSystemConfig: {str(e)}",
                recommendations=["Check SimOS version compatibility"]
            ) from e
        except ValueError as e:
            raise ConfigurationError(
                f"Invalid NV configuration: {str(e)}",
                parameter="config",
                value=config
            ) from e
    
    def _initialize_operators(self) -> None:
        """
        Initialize quantum operators for the NV system.
        
        This method sets up all necessary quantum operators for simulations,
        including spin operators, Hamiltonian components, and measurement operators.
        
        Raises
        ------
        HamiltonianError
            If operators cannot be initialized
        """
        try:
            # Basic spin operators
            self.Sx = self.nv_system.Sx
            self.Sy = self.nv_system.Sy
            self.Sz = self.nv_system.Sz
            
            # Transition operators
            self.Sx_0m = self.nv_system.get_transition_operator("0", "-1", "x")  # |0⟩⟨-1| + |-1⟩⟨0|
            self.Sx_0p = self.nv_system.get_transition_operator("0", "+1", "x")  # |0⟩⟨+1| + |+1⟩⟨0|
            
            self.Sz_0m = self.nv_system.get_transition_operator("0", "-1", "z")  # |0⟩⟨0| - |-1⟩⟨-1|
            self.Sz_0p = self.nv_system.get_transition_operator("0", "+1", "z")  # |0⟩⟨0| - |+1⟩⟨+1|
            
            # State projectors
            self.P_m = self.nv_system.get_projector("-1")  # |-1⟩⟨-1|
            self.P_0 = self.nv_system.get_projector("0")   # |0⟩⟨0|
            self.P_p = self.nv_system.get_projector("+1")  # |+1⟩⟨+1|
            
            # Identity operator
            self.I = self.nv_system.identity
            
            # Basic states
            self.psi_0 = self.nv_system.get_state("0")
            self.psi_m = self.nv_system.get_state("-1")
            self.psi_p = self.nv_system.get_state("+1")
            
        except AttributeError as e:
            raise SimOSImportError(
                f"Missing required operator methods in SimOS: {str(e)}",
                recommendations=["Update to a newer version of SimOS"]
            ) from e
        except Exception as e:
            raise HamiltonianError(
                f"Failed to initialize quantum operators: {str(e)}"
            ) from e
    
    def field_hamiltonian(self, Bvec: List[float], strain: Optional[List[float]] = None) -> Any:
        """
        Create the Hamiltonian for a given magnetic field and strain.
        
        Parameters
        ----------
        Bvec : list of float
            Magnetic field vector [Bx, By, Bz] in Tesla
        strain : list of float, optional
            Strain parameters [E, D] in Hz
            
        Returns
        -------
        object
            SimOS Hamiltonian operator
            
        Raises
        ------
        HamiltonianError
            If the Hamiltonian cannot be constructed
        """
        if len(Bvec) != 3:
            raise HamiltonianError(
                f"Magnetic field vector must have 3 components, got {len(Bvec)}",
                hamiltonian_type="field_hamiltonian",
                params={"Bvec": Bvec}
            )
        
        try:
            # If strain provided, use it
            if strain is not None:
                if len(strain) != 2:
                    raise HamiltonianError(
                        f"Strain parameter must have 2 components [E, D], got {len(strain)}",
                        hamiltonian_type="field_hamiltonian",
                        params={"strain": strain}
                    )
                return self.nv_system.get_hamiltonian(Bvec=Bvec, strain=strain)
            else:
                return self.nv_system.get_hamiltonian(Bvec=Bvec)
        except Exception as e:
            raise HamiltonianError(
                f"Failed to create field Hamiltonian: {str(e)}",
                hamiltonian_type="field_hamiltonian",
                params={"Bvec": Bvec, "strain": strain}
            ) from e
    
    def hyperfine_hamiltonian(self, n_species: int = 14, 
                              hyperfine_parallel: float = 2.16e6,
                              hyperfine_perpendicular: float = 2.7e6) -> Any:
        """
        Create the hyperfine interaction Hamiltonian.
        
        Parameters
        ----------
        n_species : int
            Nitrogen isotope (14 or 15)
        hyperfine_parallel : float
            Parallel hyperfine coupling in Hz
        hyperfine_perpendicular : float
            Perpendicular hyperfine coupling in Hz
            
        Returns
        -------
        object
            SimOS Hamiltonian operator
            
        Raises
        ------
        HamiltonianError
            If the Hamiltonian cannot be constructed
        """
        if n_species not in (14, 15):
            raise HamiltonianError(
                f"Nitrogen isotope must be 14 or 15, got {n_species}",
                hamiltonian_type="hyperfine_hamiltonian",
                params={"n_species": n_species}
            )
        
        try:
            return self.nv_system.get_hyperfine_hamiltonian(
                n_species=n_species,
                A_parallel=hyperfine_parallel,
                A_perpendicular=hyperfine_perpendicular
            )
        except Exception as e:
            raise HamiltonianError(
                f"Failed to create hyperfine Hamiltonian: {str(e)}",
                hamiltonian_type="hyperfine_hamiltonian",
                params={
                    "n_species": n_species,
                    "hyperfine_parallel": hyperfine_parallel,
                    "hyperfine_perpendicular": hyperfine_perpendicular
                }
            ) from e
    
    def transition_operators(self, T: float, beta: float, Bvec: List[float]) -> Tuple[List[Any], List[Any]]:
        """
        Get collapse operators for optical and relaxation processes.
        
        Parameters
        ----------
        T : float
            Temperature in Kelvin
        beta : float
            Optical driving parameter (0-1)
        Bvec : list of float
            Magnetic field vector [Bx, By, Bz] in Tesla
            
        Returns
        -------
        tuple
            (optical_operators, relaxation_operators)
            
        Raises
        ------
        QuantumStateError
            If the operators cannot be constructed
        """
        if T <= 0:
            raise QuantumStateError(
                f"Temperature must be positive, got {T}",
                state_info={"T": T}
            )
        
        if beta < 0 or beta > 1:
            raise QuantumStateError(
                f"Optical driving parameter beta must be between 0 and 1, got {beta}",
                state_info={"beta": beta}
            )
        
        if len(Bvec) != 3:
            raise QuantumStateError(
                f"Magnetic field vector must have 3 components, got {len(Bvec)}",
                state_info={"Bvec": Bvec}
            )
        
        try:
            return self.nv_system.get_collapse_operators(T=T, beta=beta, Bvec=Bvec)
        except Exception as e:
            raise QuantumStateError(
                f"Failed to create transition operators: {str(e)}",
                state_info={"T": T, "beta": beta, "Bvec": Bvec}
            ) from e
    
    def pure_dephasing_operators(self, rate: float) -> List[Any]:
        """
        Get collapse operators for pure dephasing processes.
        
        Parameters
        ----------
        rate : float
            Dephasing rate in Hz
            
        Returns
        -------
        list
            List of dephasing operators
            
        Raises
        ------
        QuantumStateError
            If the operators cannot be constructed
        """
        if rate < 0:
            raise QuantumStateError(
                f"Dephasing rate must be non-negative, got {rate}",
                state_info={"rate": rate}
            )
        
        try:
            return self.nv_system.get_dephasing_operators(rate=rate)
        except Exception as e:
            raise QuantumStateError(
                f"Failed to create dephasing operators: {str(e)}",
                state_info={"rate": rate}
            ) from e
    
    def inhomogeneous_dephasing_operators(self, rate: float) -> List[Any]:
        """
        Get collapse operators for inhomogeneous dephasing (T2*).
        
        Parameters
        ----------
        rate : float
            Inhomogeneous dephasing rate in Hz
            
        Returns
        -------
        list
            List of inhomogeneous dephasing operators
            
        Raises
        ------
        QuantumStateError
            If the operators cannot be constructed
        """
        if rate < 0:
            raise QuantumStateError(
                f"Inhomogeneous dephasing rate must be non-negative, got {rate}",
                state_info={"rate": rate}
            )
        
        try:
            # Use Sz operators with appropriate rates for inhomogeneous dephasing
            # This is a common approach for T2* processes
            gamma = np.sqrt(rate)
            return [
                gamma * self.Sz  # Global dephasing
            ]
        except Exception as e:
            raise QuantumStateError(
                f"Failed to create inhomogeneous dephasing operators: {str(e)}",
                state_info={"rate": rate}
            ) from e
    
    def evolve_density_matrix(self, rho: Any, H: Any, dt: float, 
                             c_ops: Optional[List[Any]] = None) -> Any:
        """
        Evolve a density matrix according to the Lindblad master equation.
        
        Parameters
        ----------
        rho : object
            Initial density matrix
        H : object
            Hamiltonian operator
        dt : float
            Time step in seconds
        c_ops : list, optional
            Collapse operators for decoherence
            
        Returns
        -------
        object
            Evolved density matrix
            
        Raises
        ------
        EvolutionError
            If the evolution fails
        """
        if dt <= 0:
            raise EvolutionError(
                f"Time step must be positive, got {dt}",
                duration=dt
            )
        
        c_ops = c_ops or []
        
        try:
            if c_ops:
                # Use mesolve for open quantum systems
                result = self.simos.propagation.mesolve(H, rho, dt, c_ops)
                if hasattr(result, 'states') and len(result.states) > 0:
                    return result.states[-1]
                else:
                    raise EvolutionError(
                        "mesolve did not return any states",
                        duration=dt,
                        method="mesolve"
                    )
            else:
                # Use unitary evolution for closed systems
                propagator = self.simos.propagation.evol(H, dt)
                if hasattr(propagator, 'dag'):
                    return propagator * rho * propagator.dag()
                else:
                    # Fallback for older SimOS versions
                    prop_conj = propagator.conj().transpose()
                    return propagator @ rho @ prop_conj
        except Exception as e:
            if isinstance(e, EvolutionError):
                raise
            raise EvolutionError(
                f"Failed to evolve density matrix: {str(e)}",
                duration=dt,
                method="mesolve" if c_ops else "unitary"
            ) from e
            
    def evolve_adaptive(self, rho, H, dt, c_ops=None, tol=1e-6, method='adams'):
        """
        Adaptively evolve a density matrix with variable step size.
        
        Parameters
        ----------
        rho : object
            Initial density matrix
        H : object
            Hamiltonian operator
        dt : float
            Maximum time step in seconds
        c_ops : list, optional
            Collapse operators for decoherence
        tol : float
            Error tolerance for adaptive stepping
        method : str
            ODE method ('adams', 'bdf', 'rk45', etc.)
            
        Returns
        -------
        object
            Evolved density matrix
            
        Raises
        ------
        EvolutionError
            If the evolution fails
        """
        try:
            # Use advanced integration method from SimOS
            solver_options = {
                'method': method,
                'atol': tol,
                'rtol': tol,
                'max_step': dt / 10
            }
            
            result = self.simos.propagation.mesolve(
                H, rho, dt, c_ops or [], 
                options=solver_options,
                progress_bar=False
            )
            
            if hasattr(result, 'states') and len(result.states) > 0:
                return result.states[-1]
            else:
                raise EvolutionError(
                    "Adaptive solver did not return any states",
                    duration=dt,
                    method=method
                )
        except Exception as e:
            if isinstance(e, EvolutionError):
                raise
            raise EvolutionError(
                f"Failed to evolve density matrix adaptively: {str(e)}",
                duration=dt,
                method=method
            ) from e

    def evolve_non_markovian(self, rho, H, dt, bath_spectrum, temperature=298):
        """
        Evolve a density matrix with non-Markovian dynamics.
        
        Parameters
        ----------
        rho : object
            Initial density matrix
        H : object
            Hamiltonian operator
        dt : float
            Time step in seconds
        bath_spectrum : callable
            Spectral density function of the bath
        temperature : float
            Temperature in Kelvin
            
        Returns
        -------
        object
            Evolved density matrix
            
        Raises
        ------
        EvolutionError
            If the evolution fails
        """
        try:
            # Use SimOS non-Markovian solver
            memory_cutoff = 5 * self.nv_system.t2_star()  # Reasonable memory cutoff
            
            result = self.simos.propagation.ttm_solve(
                H, rho, dt, 
                bath_spectrum=bath_spectrum,
                temperature=temperature,
                memory_cutoff=memory_cutoff
            )
            
            if hasattr(result, 'states') and len(result.states) > 0:
                return result.states[-1]
            else:
                raise EvolutionError(
                    "Non-Markovian solver did not return any states",
                    duration=dt,
                    method="ttm_solve"
                )
        except Exception as e:
            if isinstance(e, EvolutionError):
                raise
            raise EvolutionError(
                f"Failed to evolve density matrix with non-Markovian dynamics: {str(e)}",
                duration=dt,
                method="ttm_solve"
            ) from e
    
    def evolve_state_vector(self, psi: Any, H: Any, dt: float) -> Any:
        """
        Evolve a state vector according to the Schrödinger equation.
        
        Parameters
        ----------
        psi : object
            Initial state vector
        H : object
            Hamiltonian operator
        dt : float
            Time step in seconds
            
        Returns
        -------
        object
            Evolved state vector
            
        Raises
        ------
        EvolutionError
            If the evolution fails
        """
        if dt <= 0:
            raise EvolutionError(
                f"Time step must be positive, got {dt}",
                duration=dt
            )
        
        try:
            # Use sesolve for state vector evolution
            result = self.simos.propagation.sesolve(H, psi, dt)
            if hasattr(result, 'states') and len(result.states) > 0:
                return result.states[-1]
            else:
                raise EvolutionError(
                    "sesolve did not return any states",
                    duration=dt,
                    method="sesolve"
                )
        except Exception as e:
            if isinstance(e, EvolutionError):
                raise
            raise EvolutionError(
                f"Failed to evolve state vector: {str(e)}",
                duration=dt,
                method="sesolve"
            ) from e
    
    def get_equilibrium_state(self, T: float = 298) -> Any:
        """
        Get thermal equilibrium state at a given temperature.
        
        Parameters
        ----------
        T : float
            Temperature in Kelvin
            
        Returns
        -------
        object
            Equilibrium density matrix
            
        Raises
        ------
        QuantumStateError
            If the state cannot be constructed
        """
        if T <= 0:
            raise QuantumStateError(
                f"Temperature must be positive, got {T}",
                state_info={"T": T}
            )
        
        try:
            return self.nv_system.get_thermal_state(T=T)
        except Exception as e:
            raise QuantumStateError(
                f"Failed to create equilibrium state: {str(e)}",
                state_info={"T": T}
            ) from e
    
    def get_excited_state_probability(self, rho: Any) -> float:
        """
        Calculate the probability of being in the excited state.
        
        Parameters
        ----------
        rho : object
            Density matrix
            
        Returns
        -------
        float
            Excited state probability
            
        Raises
        ------
        QuantumStateError
            If the probability cannot be calculated
        """
        try:
            return self.nv_system.excited_state_probability(rho)
        except Exception as e:
            raise QuantumStateError(
                f"Failed to calculate excited state probability: {str(e)}"
            ) from e
    
    def get_ground_state_populations(self, rho: Any) -> Dict[str, float]:
        """
        Calculate ground state populations (ms=0, ms=±1).
        
        Parameters
        ----------
        rho : object
            Density matrix
            
        Returns
        -------
        dict
            Dictionary with keys 'ms0', 'ms_minus', 'ms_plus' and their probabilities
            
        Raises
        ------
        QuantumStateError
            If the populations cannot be calculated
        """
        try:
            return {
                'ms0': float(np.real(self.nv_system.expect(self.P_0, rho))),
                'ms_minus': float(np.real(self.nv_system.expect(self.P_m, rho))),
                'ms_plus': float(np.real(self.nv_system.expect(self.P_p, rho)))
            }
        except Exception as e:
            raise QuantumStateError(
                f"Failed to calculate ground state populations: {str(e)}"
            ) from e
    
    def get_coherences(self, rho: Any) -> Dict[str, complex]:
        """
        Calculate coherences between ground state levels.
        
        Parameters
        ----------
        rho : object
            Density matrix
            
        Returns
        -------
        dict
            Dictionary with coherence values between different states
            
        Raises
        ------
        QuantumStateError
            If the coherences cannot be calculated
        """
        try:
            # Get coherences between ms=0 and ms=±1, and between ms=-1 and ms=+1
            ms0_msm = self.nv_system.get_coherence(rho, "0", "-1")
            ms0_msp = self.nv_system.get_coherence(rho, "0", "+1")
            msm_msp = self.nv_system.get_coherence(rho, "-1", "+1")
            
            return {
                'ms0_ms_minus': complex(ms0_msm),
                'ms0_ms_plus': complex(ms0_msp),
                'ms_minus_ms_plus': complex(msm_msp)
            }
        except Exception as e:
            raise QuantumStateError(
                f"Failed to calculate coherences: {str(e)}"
            ) from e
            
    def dipolar_coupling_hamiltonian(self, target_spin, coupling_strength, orientation=None):
        """
        Create Hamiltonian for dipolar coupling to a target spin.
        
        Parameters
        ----------
        target_spin : str
            Type of target spin ("13C", "14N", "15N", "1H")
        coupling_strength : float
            Dipolar coupling strength in Hz
        orientation : list, optional
            Unit vector of the dipolar coupling axis
            
        Returns
        -------
        object
            Dipolar coupling Hamiltonian
            
        Raises
        ------
        HamiltonianError
            If the Hamiltonian cannot be constructed
        """
        try:
            # Map spin types to gyromagnetic ratios
            gyro_ratios = {
                "13C": 6.728e7,    # Hz/T
                "14N": 1.933e7,    # Hz/T
                "15N": -2.712e7,   # Hz/T
                "1H": 2.675e8      # Hz/T
            }
            
            if target_spin not in gyro_ratios:
                raise HamiltonianError(
                    f"Unknown target spin type: {target_spin}",
                    hamiltonian_type="dipolar_coupling",
                    params={"target_spin": target_spin}
                )
            
            # Use default orientation if not provided
            if orientation is None:
                orientation = [0, 0, 1]  # Z-axis
                
            # Normalize orientation
            orientation = np.array(orientation) / np.linalg.norm(orientation)
            
            # Create dipolar Hamiltonian with SimOS
            H_dipolar = self.nv_system.get_dipolar_hamiltonian(
                spin_type=target_spin,
                coupling_strength=coupling_strength,
                orientation=orientation.tolist(),
                gyro_ratio=gyro_ratios[target_spin]
            )
            
            return H_dipolar
        except Exception as e:
            if isinstance(e, HamiltonianError):
                raise
            raise HamiltonianError(
                f"Failed to create dipolar coupling Hamiltonian: {str(e)}",
                hamiltonian_type="dipolar_coupling",
                params={
                    "target_spin": target_spin,
                    "coupling_strength": coupling_strength,
                    "orientation": orientation
                }
            ) from e
            
    def get_spin_bath_hamiltonian(self, bath_concentration, bath_coupling=None):
        """
        Create a Hamiltonian representing coupling to a spin bath.
        
        Parameters
        ----------
        bath_concentration : float
            Concentration of bath spins in spins/cm³
        bath_coupling : float, optional
            Overall coupling strength. If None, calculated from concentration
            
        Returns
        -------
        object
            Effective spin bath Hamiltonian
            
        Raises
        ------
        HamiltonianError
            If the Hamiltonian cannot be constructed
        """
        try:
            # Calculate coupling if not provided
            if bath_coupling is None:
                # Empirical relationship between concentration and coupling
                bath_coupling = 2e6 * (bath_concentration / 1e19)**(2/3)  # Hz
            
            # Create bath Hamiltonian using SimOS
            H_bath = self.nv_system.get_spin_bath_hamiltonian(
                concentration=bath_concentration,
                coupling_strength=bath_coupling,
                bath_type="electronic",  # Can be "electronic", "nuclear", or "mixed"
                random_seed=42  # For reproducibility
            )
            
            return H_bath
        except Exception as e:
            if isinstance(e, HamiltonianError):
                raise
            raise HamiltonianError(
                f"Failed to create spin bath Hamiltonian: {str(e)}",
                hamiltonian_type="spin_bath",
                params={
                    "bath_concentration": bath_concentration,
                    "bath_coupling": bath_coupling
                }
            ) from e
            
    def lorentzian_bath(self, strength, cutoff):
        """
        Create a Lorentzian bath spectral density function.
        
        Parameters
        ----------
        strength : float
            Overall coupling strength in Hz
        cutoff : float
            Spectral cutoff frequency in Hz
            
        Returns
        -------
        callable
            Spectral density function J(ω)
        """
        def J_lorentzian(omega):
            return 2 * strength * cutoff / (omega**2 + cutoff**2)
        
        return J_lorentzian

    def ohmic_bath(self, strength, s_param, cutoff):
        """
        Create an Ohmic bath spectral density function.
        
        Parameters
        ----------
        strength : float
            Overall coupling strength in Hz
        s_param : float
            Ohmicity parameter (s=1 is Ohmic, s<1 sub-Ohmic, s>1 super-Ohmic)
        cutoff : float
            Spectral cutoff frequency in Hz
            
        Returns
        -------
        callable
            Spectral density function J(ω)
        """
        def J_ohmic(omega):
            return strength * (omega/cutoff)**s_param * np.exp(-abs(omega)/cutoff)
        
        return J_ohmic

    def white_noise_bath(self, strength):
        """
        Create a white noise bath spectral density function.
        
        Parameters
        ----------
        strength : float
            Overall noise strength in Hz
            
        Returns
        -------
        callable
            Spectral density function J(ω)
        """
        def J_white(omega):
            return strength * np.ones_like(omega)
        
        return J_white
        
    def get_superposition_state(self, electron_states, nuclear_state=None):
        """
        Create a superposition state of multiple electron and nuclear states.
        
        Parameters
        ----------
        electron_states : list
            List of electron states to include in superposition (e.g., ["0", "+1"])
        nuclear_state : str, optional
            Nuclear spin state if relevant ("up" or "down")
            
        Returns
        -------
        object
            Density matrix representing the superposition state
            
        Raises
        ------
        QuantumStateError
            If the state cannot be constructed
        """
        try:
            # Validate states
            valid_electron = ["0", "+1", "-1"]
            for state in electron_states:
                if state not in valid_electron:
                    raise QuantumStateError(
                        f"Invalid electron state: {state}",
                        state_info={"electron_states": electron_states}
                    )
            
            # Get individual states
            states = []
            for e_state in electron_states:
                if nuclear_state:
                    state = self.nv_system.get_state(e_state, nuclear_state)
                else:
                    state = self.nv_system.get_state(e_state)
                states.append(state)
            
            # Create equal superposition
            psi = sum(states) / np.sqrt(len(states))
            
            # Convert to density matrix
            if hasattr(self.nv_system, 'state_to_dm'):
                rho = self.nv_system.state_to_dm(psi)
            else:
                # Manual conversion
                rho = psi * psi.dag()
            
            return rho
        except Exception as e:
            if isinstance(e, QuantumStateError):
                raise
            raise QuantumStateError(
                f"Failed to create superposition state: {str(e)}",
                state_info={
                    "electron_states": electron_states,
                    "nuclear_state": nuclear_state
                }
            ) from e