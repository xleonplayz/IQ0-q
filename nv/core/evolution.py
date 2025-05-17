#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import logging
import os
import sys

# SimOS-Pfad zur Python-Umgebung hinzufügen
simos_paths = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../trash/ui/simos')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../../simos')),
]

for simos_path in simos_paths:
    if os.path.exists(simos_path) and simos_path not in sys.path:
        sys.path.insert(0, simos_path)

# Logging einrichten
logger = logging.getLogger(__name__)

class QuantumEvolution:
    """Quantenzeitentwicklungs-Engine basierend auf SimOS."""
    
    def __init__(self, method="qutip"):
        """
        Initialisiere die Zeitentwicklungs-Engine.
        
        Parameters
        ----------
        method : str, optional
            Numerische Methode (qutip, numpy, sparse)
        """
        self.method = method
        self._initialized = False
        self._cached_operators = {}
        
        # SimOS importieren - wenn Fehler, dann ausdrückliche Fehlermeldung
        try:
            from simos.simos import propagation
            self.propagation = propagation
        except ImportError:
            logger.error("SimOS-Bibliothek konnte nicht importiert werden!")
            logger.error("Bitte installieren Sie SimOS mit: cd <simos_path> && pip install -e .")
            logger.error(f"SimOS sollte in einem dieser Pfade sein: {simos_paths}")
            raise ImportError("SimOS-Bibliothek fehlt!")
        
        # Speichermanagement einrichten
        try:
            import gc
            # Automatische Garbage Collection konfigurieren
            gc.enable()
            # Einstellungen für aggressivere Speicherfreigabe
            gc.set_threshold(700, 10, 10)
        except ImportError:
            logger.warning("Garbage Collection (gc) konnte nicht konfiguriert werden")
        
    def initialize(self):
        """Initialisiere interne Zustände und Cache, falls noch nicht geschehen."""
        if not self._initialized:
            # Ressourcenverwaltung initialisieren
            self._initialized = True
    
    def __del__(self):
        """Aufräumen beim Löschen des Objekts."""
        try:
            self._cached_operators.clear()
            import gc
            gc.collect()
        except:
            pass
    
    def unitary_evolution(self, state, hamiltonian, time):
        """
        Führe unitäre Zeitentwicklung unter zeit(un)abhängigem Hamiltonian durch.
        
        Parameters
        ----------
        state : QuantumState
            Anfangszustand
        hamiltonian : SimOS operator or list
            Hamiltonoperator (zeitunabhängig oder zeitabhängig)
        time : float
            Entwicklungszeit in Sekunden
        
        Returns
        -------
        QuantumState
            Entwickelter Quantenzustand
            
        Raises
        ------
        ValueError
            Bei ungültigen Parametern
        """
        if time < 0:
            raise ValueError("Entwicklungszeit muss positiv sein")
            
        if time == 0:
            return state.copy()
        
        try:
            # SimOS-Zustand extrahieren
            simos_state = state.get_simos_state()
            
            # Unterscheide zwischen zeitabhängigem und zeitunabhängigem Hamiltonian
            # Importiere hier lokal, damit Fehler besser behandelt werden können
            from simos.simos import coherent
            
            # Zeitabhängiger oder zeitunabhängiger Hamiltonian?
            is_time_dependent = isinstance(hamiltonian, list) and len(hamiltonian) > 1
            
            if is_time_dependent:
                # Zeitabhängige Evolution mit effizientem Integrator
                options = {'method': 'magnus', 'order': 4, 'nsteps': 100}
                evolved_state = self.propagation.propagate_array(simos_state, hamiltonian, 0, time, **options)
            else:
                # Zeitunabhängige Evolution mit exaktem Propagator
                U = self.propagation.evol(hamiltonian, time)
                
                # Zustand entwickeln
                if coherent.is_ket(simos_state):
                    evolved_state = U * simos_state
                else:
                    evolved_state = U * simos_state * U.dag()
            
            # Speicher freigeben
            import gc
            gc.collect()
            
            # Als QuantumState zurückgeben
            from nv.core.quantum_state import QuantumState
            return QuantumState(evolved_state, method=self.method)
            
        except Exception as e:
            logger.error(f"Fehler in unitärer Evolution: {str(e)}")
            raise ValueError(f"Unitäre Evolution fehlgeschlagen: {str(e)}")
    
    def open_evolution(self, state, hamiltonian, collapse_operators, time, steps=100):
        """
        Führe offene Systemevolution mit Lindblad-Mastergleichung durch.
        
        Parameters
        ----------
        state : QuantumState
            Anfangszustand
        hamiltonian : SimOS operator or list
            Hamiltonoperator (zeitunabhängig oder zeitabhängig)
        collapse_operators : list
            Liste von Collapse-Operatoren für Dissipation
        time : float
            Entwicklungszeit in Sekunden
        steps : int, optional
            Anzahl der Zeitschritte
        
        Returns
        -------
        QuantumState
            Entwickelter Quantenzustand
            
        Raises
        ------
        ValueError
            Bei ungültigen Parametern
        """
        if time < 0:
            raise ValueError("Entwicklungszeit muss positiv sein")
            
        if time == 0:
            return state.copy()
            
        try:
            # Muss QuTiP für Mastergleichung verwenden
            import qutip as qt
            
            # In Dichtematrix konvertieren falls nötig
            dm_state = state.to_density_matrix().get_simos_state()
            
            # Zeitabhängiger oder zeitunabhängiger Hamiltonian?
            is_time_dependent = isinstance(hamiltonian, list) and len(hamiltonian) > 1
            
            # Zeitpunkte erstellen
            tlist = np.linspace(0, time, steps+1)
            
            # Fortschrittsanzeige deaktivieren für bessere Performance
            options = qt.Options(nsteps=5000, store_states=True, store_final_state=True, progress_bar=None)
            
            if is_time_dependent:
                # SimOS Hamiltonian zu QuTiP-Format konvertieren
                from simos.simos import coherent, util
                qutip_ham = []
                
                # Statischen Teil hinzufügen
                qutip_ham.append(hamiltonian[0])
                
                # Zeitabhängige Terme hinzufügen
                for i in range(1, len(hamiltonian)):
                    h_op = hamiltonian[i][0]  # Operator
                    h_func = hamiltonian[i][1]  # Zeitfunktion
                    qutip_ham.append([h_op, h_func])
                    
                result = qt.mesolve(qutip_ham, dm_state, tlist, collapse_operators, [], options=options)
            else:
                # Standardlöser für zeitunabhängige Hamiltonians verwenden
                result = qt.mesolve(hamiltonian, dm_state, tlist, collapse_operators, [], options=options)
            
            # Speicher freigeben
            import gc
            for i in range(len(result.states)-1):
                result.states[i] = None
            gc.collect()
            
            # Endzustand zurückgeben
            from nv.core.quantum_state import QuantumState
            return QuantumState(result.states[-1], method="qutip")
            
        except Exception as e:
            logger.error(f"Fehler in offener Systemevolution: {str(e)}")
            raise ValueError(f"Offene Evolution fehlgeschlagen: {str(e)}")