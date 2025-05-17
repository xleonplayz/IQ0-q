import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional, Union, Callable

logger = logging.getLogger(__name__)

class Hamiltonian:
    """Hamiltonian-Builder für NV-Simulationen mit Unterstützung für zeit(un)abhängige Terme."""
    
    def __init__(self, method="qutip"):
        """
        Initialisiere einen Hamiltonian-Builder.
        
        Parameters
        ----------
        method : str, optional
            Numerische Methode (qutip, numpy, sparse)
        """
        self.method = method
        self._static_terms = []
        self._time_dependent_terms = []
        self._cached_operators = {}
    
    def add_term(self, operator, coefficient=1.0):
        """
        Füge einen statischen Term zum Hamiltonian hinzu.
        
        Parameters
        ----------
        operator : SimOS operator
            Operator-Term
        coefficient : float or complex, optional
            Koeffizient für den Term
        
        Returns
        -------
        Hamiltonian
            Der Hamiltonian-Builder für Verkettung
        """
        self._static_terms.append((coefficient, operator))
        return self
    
    def add_time_dependent_term(self, operator, time_function, coefficient=1.0):
        """
        Füge einen zeitabhängigen Term zum Hamiltonian hinzu.
        
        Parameters
        ----------
        operator : SimOS operator
            Operator-Term
        time_function : callable
            Zeitfunktion mit Signatur f(t, args) -> float
        coefficient : float or complex, optional
            Koeffizient für den Term
        
        Returns
        -------
        Hamiltonian
            Der Hamiltonian-Builder für Verkettung
        """
        if not callable(time_function):
            raise ValueError("time_function muss eine aufrufbare Funktion sein")
        
        self._time_dependent_terms.append((coefficient, operator, time_function))
        return self
    
    def add_hyperfine_term(self, nv_system, hyperfine_tensor=None):
        """
        Füge einen Hyperfein-Kopplungsterm zwischen Elektronen- und Kernspins hinzu.
        
        Parameters
        ----------
        nv_system : NVSystem
            Das NV-System, für das der Hyperfein-Term erstellt werden soll
        hyperfine_tensor : ndarray, optional
            3x3 Hyperfein-Tensor für die Kopplung, falls nicht spezifiziert,
            wird ein Standard-Tensor für 14N verwendet
            
        Returns
        -------
        Hamiltonian
            Der Hamiltonian-Builder für Verkettung
        """
        if not hasattr(nv_system, 'simos_nv'):
            raise ValueError("NV-System muss SimOS NV-System haben")
        
        try:
            # Standardtensor für 14N falls nicht angegeben (A_parallel = -2.16 MHz, A_perp = -2.7 MHz)
            if hyperfine_tensor is None:
                A_parallel = -2.16e6  # Hz
                A_perp = -2.7e6       # Hz
                hyperfine_tensor = np.diag([A_perp, A_perp, A_parallel])
            
            # Elektronenspin-Operatoren
            S = [nv_system.simos_nv.Sx, nv_system.simos_nv.Sy, nv_system.simos_nv.Sz]
            
            # Kernspin-Operatoren (falls vorhanden)
            if not hasattr(nv_system.simos_nv, 'Ix'):
                raise ValueError("NV-System hat keine Kernspin-Operatoren, setze nitrogen=True in der Konfiguration")
            
            I = [nv_system.simos_nv.Ix, nv_system.simos_nv.Iy, nv_system.simos_nv.Iz]
            
            # Hyperfein-Term berechnen
            H_hyperfine = 0
            for i in range(3):
                for j in range(3):
                    if hyperfine_tensor[i, j] != 0:
                        H_hyperfine += hyperfine_tensor[i, j] * S[i] * I[j]
            
            # Als statischen Term hinzufügen
            self.add_term(H_hyperfine, 1.0)
            return self
            
        except Exception as e:
            logger.error(f"Fehler bei Hyperfein-Kopplung: {str(e)}")
            raise
    
    def add_strain_term(self, nv_system, strain_amplitude, strain_angle=0.0):
        """
        Füge einen Hamiltonian für Gitterverzerrungen (strain) im NV-System hinzu.
        
        Parameters
        ----------
        nv_system : NVSystem
            Das NV-System, für das der Strain-Term erstellt werden soll
        strain_amplitude : float
            Stärke der Gitterverzerrung in Hz
        strain_angle : float, optional
            Winkel der Gitterverzerrung in der xy-Ebene in Radiant
            
        Returns
        -------
        Hamiltonian
            Der Hamiltonian-Builder für Verkettung
        """
        if not hasattr(nv_system, 'simos_nv'):
            raise ValueError("NV-System muss SimOS NV-System haben")
        
        try:
            # Strain als transversale Komponenten anwenden
            Ex = strain_amplitude * np.cos(strain_angle)
            Ey = strain_amplitude * np.sin(strain_angle)
            
            # SimOS Operatoren verwenden
            Sx = nv_system.simos_nv.Sx
            Sy = nv_system.simos_nv.Sy
            
            # Als statische Terme hinzufügen
            self.add_term(Sx, Ex)
            self.add_term(Sy, Ey)
            
            return self
            
        except Exception as e:
            logger.error(f"Fehler bei Strain-Term: {str(e)}")
            raise
    
    def build(self, is_time_dependent=None):
        """
        Erzeuge den vollständigen Hamiltonian.
        
        Parameters
        ----------
        is_time_dependent : bool, optional
            Erzwinge zeitabhängiges oder zeitunabhängiges Format.
            Wenn None, wird automatisch ermittelt.
        
        Returns
        -------
        SimOS operator or list
            Der konstruierte Hamiltonian, zeitunabhängig oder zeitabhängig
            im Format [H0, [H1, f1], [H2, f2], ...]
        """
        # Prüfen, ob überhaupt Terme vorhanden sind
        if not self._static_terms and not self._time_dependent_terms:
            raise ValueError("Keine Terme im Hamiltonian")
        
        try:
            # Statischen Hamiltonian aufbauen
            H0 = 0
            for coef, op in self._static_terms:
                H0 += coef * op
            
            # Entscheiden, ob zeitabhängig oder nicht
            has_time_terms = len(self._time_dependent_terms) > 0
            if is_time_dependent is None:
                is_time_dependent = has_time_terms
            
            # Zeitunabhängigen Hamiltonian zurückgeben
            if not is_time_dependent:
                return H0
                
            # Zeitabhängigen Hamiltonian erstellen: [H0, [H1, f1], [H2, f2], ...]
            H = [H0]
            
            # Zeitabhängige Terme hinzufügen
            for coef, op, func in self._time_dependent_terms:
                # Skalierte Zeitfunktion erstellen
                scaled_func = lambda t, args, coef=coef, func=func: coef * func(t, args)
                
                # Term hinzufügen
                H.append([op, scaled_func])
            
            return H
            
        except Exception as e:
            logger.error(f"Fehler beim Aufbau des Hamiltonians: {str(e)}")
            raise
    
    def reset(self):
        """Setze alle Terme zurück."""
        self._static_terms = []
        self._time_dependent_terms = []
        self._cached_operators = {}
        return self