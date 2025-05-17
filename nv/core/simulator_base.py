import logging
from nv.core.evolution import QuantumEvolution

logger = logging.getLogger(__name__)

class SimulatorBase:
    """Basisklasse für Quantensimulatoren."""
    
    def __init__(self, config=None):
        """
        Initialisiere Simulator mit Konfiguration.
        
        Parameters
        ----------
        config : dict, optional
            Konfigurationsparameter
        """
        # Standard-Konfiguration
        self.config = {
            "method": "qutip",
            "temperature": 300.0,  # K
            "magnetic_field": [0.0, 0.0, 0.0],  # T
            "zero_field_splitting": 2.87e9,  # Hz
            "gyromagnetic_ratio": 28.025e9,  # Hz/T
            "t1": 5.0e-3,  # s
            "t2": 1.0e-5,  # s
        }
        
        # Mit angegebener Konfiguration aktualisieren
        if config:
            self.config.update(config)
        
        # Kernkomponenten initialisieren
        self.evolution_engine = QuantumEvolution(method=self.config["method"])
        
        # Zustand wird von Unterklassen initialisiert
        self.state = None
    
    def reset(self):
        """Simulator-Zustand zurücksetzen. Muss von Unterklassen implementiert werden."""
        raise NotImplementedError("Unterklassen müssen reset() implementieren")
    
    def evolve(self, duration):
        """
        Quantenzustand für angegebene Dauer entwickeln.
        
        Parameters
        ----------
        duration : float
            Entwicklungszeit in Sekunden
        """
        raise NotImplementedError("Unterklassen müssen evolve() implementieren")
    
    def get_populations(self):
        """
        Zustandspopulationen erhalten.
        
        Returns
        -------
        dict
            Dictionary der Zustandspopulationen
        """
        raise NotImplementedError("Unterklassen müssen get_populations() implementieren")
    
    def get_fluorescence(self):
        """
        Fluoreszenzsignal für aktuellen Zustand erhalten.
        
        Returns
        -------
        float
            Fluoreszenzsignalwert
        """
        raise NotImplementedError("Unterklassen müssen get_fluorescence() implementieren")