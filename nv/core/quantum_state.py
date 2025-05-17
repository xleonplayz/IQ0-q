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

# Logger einrichten
logger = logging.getLogger(__name__)

# Versuche SimOS zu importieren
try:
    from simos.simos import core, states, coherent
except ImportError:
    logger.error("Konnte SimOS-Module nicht importieren!")
    logger.error("Bitte installiere SimOS mit: pip install -e /pfad/zu/simos")
    logger.error(f"SimOS sollte in einem dieser Pfade sein: {simos_paths}")
    raise ImportError("SimOS-Module nicht gefunden.")

class QuantumState:
    """Abstraktion eines Quantenzustands für NV-Simulationen."""
    
    def __init__(self, simos_state=None, dimension=None, method="qutip"):
        """
        Initialisiere einen Quantenzustand.
        
        Parameters
        ----------
        simos_state : SimOS state, optional
            Existierender SimOS-Zustand
        dimension : int, optional
            Dimension des Hilbertraums
        method : str, optional
            Numerische Methode (qutip, numpy, sparse)
        """
        self.method = method
        
        if simos_state is not None:
            self._state = simos_state
        elif dimension is not None:
            # Erzeuge Identitätszustand mit gegebener Dimension
            self._state = core.id(dimension, method=method)
        else:
            raise ValueError("Entweder simos_state oder dimension muss angegeben werden")
    
    def to_density_matrix(self):
        """Konvertiere in Dichtematrix-Darstellung."""
        if coherent.is_ket(self._state):
            return QuantumState(coherent.dm(self._state), method=self.method)
        return self
    
    def expect(self, operator):
        """Berechne Erwartungswert eines Operators."""
        return coherent.expect(operator, self._state)
    
    def get_simos_state(self):
        """Gibt den zugrundeliegenden SimOS-Zustand zurück."""
        return self._state
        
    def copy(self):
        """Erstellt eine Kopie des Quantenzustands."""
        return QuantumState(self._state, method=self.method)