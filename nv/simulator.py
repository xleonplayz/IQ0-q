import logging
from nv.core.simulator_base import SimulatorBase
from nv.physics.nv_center import NVSystem
from nv.utils.logging import get_logger

logger = get_logger(__name__)

class NVSimulator(SimulatorBase):
    """
    Hauptsimulator für NV-Zentrum-Physik.
    
    Dieser Simulator integriert die Physik-Module für eine NV-Simulation
    und bietet grundlegende Hardwarefunktionen: Mikrowelle, Laser und Fluoreszenz.
    """
    
    def __init__(self, config=None):
        """
        Den NV-Simulator initialisieren.
        
        Parameters
        ----------
        config : dict, optional
            Konfigurationsparameter
        """
        super().__init__(config)
        
        # NV-System initialisieren
        self.nv_system = NVSystem(self.config)
        
        # Zustandsreferenz setzen
        self.state = self.nv_system.state
        
        logger.info("NV-Simulator initialisiert")
    
    def reset(self):
        """Simulator auf Grundzustand zurücksetzen."""
        self.nv_system.reset_state()
        self.state = self.nv_system.state
        logger.debug("Simulator zurückgesetzt")
    
    def evolve(self, duration):
        """
        Quantenzustand entwickeln.
        
        Parameters
        ----------
        duration : float
            Entwicklungszeit in Sekunden
        """
        self.nv_system.evolve(duration)
        self.state = self.nv_system.state
        logger.debug(f"Zustand für {duration} s entwickelt")
    
    def get_populations(self):
        """
        Zustandspopulationen erhalten.
        
        Returns
        -------
        dict
            Zustandspopulationen
        """
        return self.nv_system.get_populations()
    
    def get_fluorescence(self):
        """
        Fluoreszenzsignal erhalten.
        
        Returns
        -------
        float
            Fluoreszenz-Zählrate pro Sekunde
        """
        return self.nv_system.get_fluorescence()
    
    def set_magnetic_field(self, field):
        """
        Magnetfeld setzen.
        
        Parameters
        ----------
        field : list or ndarray
            Magnetfeldvektor [Bx, By, Bz] in Tesla
        """
        self.nv_system.set_magnetic_field(field)
        logger.info(f"Magnetfeld auf {field} T gesetzt")
    
    def set_microwave(self, frequency, power=0.0):
        """
        Mikrowellenparameter setzen.
        
        Parameters
        ----------
        frequency : float
            Mikrowellenfrequenz in Hz
        power : float, optional
            Mikrowellenleistung in dBm
        """
        self.nv_system.set_microwave(frequency, power)
        logger.info(f"Mikrowelle auf {frequency} Hz, {power} dBm gesetzt")
    
    def set_laser(self, power=0.0):
        """
        Laserleistung setzen.
        
        Parameters
        ----------
        power : float, optional
            Laserleistung in mW
        """
        self.nv_system.set_laser(power)
        logger.info(f"Laser auf {power} mW gesetzt")
    
    def run_sequence(self, sequence, measure_points=None):
        """
        Eine Pulssequenz ausführen.
        
        Parameters
        ----------
        sequence : PulseSequence
            Auszuführende Sequenz
        measure_points : list of int, optional
            Indizes, an denen der Zustand gemessen werden soll
            
        Returns
        -------
        dict
            Sequenzresultate
        """
        return sequence.run(self, measure_points)