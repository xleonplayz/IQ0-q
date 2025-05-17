import numpy as np
import logging
from nv.sequences.pulse import Pulse
from nv.utils.logging import get_logger

logger = get_logger(__name__)

class PulseSequence:
    """
    Sequenz von Kontrollpulsen für Quantenkontrolle.
    
    Diese Klasse repräsentiert eine Sequenz von Pulsen für ein komplettes Experiment.
    """
    
    def __init__(self, name="custom"):
        """
        Eine Pulssequenz initialisieren.
        
        Parameters
        ----------
        name : str, optional
            Name der Sequenz
        """
        self.name = name
        self.pulses = []
        self.metadata = {}
        
        logger.debug(f"Pulssequenz '{name}' erstellt")
    
    def add_pulse(self, pulse):
        """
        Einen Puls zur Sequenz hinzufügen.
        
        Parameters
        ----------
        pulse : Pulse
            Hinzuzufügender Puls
        """
        self.pulses.append(pulse)
        logger.debug(f"Puls hinzugefügt zu Sequenz '{self.name}': {pulse}")
        return self
    
    def add_wait(self, duration):
        """
        Eine Warteperiode zur Sequenz hinzufügen.
        
        Parameters
        ----------
        duration : float
            Wartedauer in Sekunden
        """
        self.pulses.append(Pulse("wait", duration))
        logger.debug(f"Wartepuls ({duration*1e9:.2f} ns) hinzugefügt zu Sequenz '{self.name}'")
        return self
    
    def add_mw_pulse(self, duration, frequency=None, power=0.0, phase=0.0):
        """
        Einen Mikrowellenpuls zur Sequenz hinzufügen.
        
        Parameters
        ----------
        duration : float
            Pulsdauer in Sekunden
        frequency : float, optional
            Mikrowellenfrequenz in Hz
        power : float, optional
            Mikrowellenleistung in dBm
        phase : float, optional
            Mikrowellenphase in Radiant
        """
        params = {"power": power, "phase": phase}
        if frequency is not None:
            params["frequency"] = frequency
        
        self.pulses.append(Pulse("mw", duration, **params))
        logger.debug(f"MW-Puls ({duration*1e9:.2f} ns, " + 
                    f"{frequency/1e9 if frequency else 'default'} GHz, {power} dBm) " +
                    f"hinzugefügt zu Sequenz '{self.name}'")
        return self
    
    def add_laser_pulse(self, duration, power=1.0):
        """
        Einen Laserpuls zur Sequenz hinzufügen.
        
        Parameters
        ----------
        duration : float
            Pulsdauer in Sekunden
        power : float, optional
            Laserleistung in mW
        """
        self.pulses.append(Pulse("laser", duration, power=power))
        logger.debug(f"Laser-Puls ({duration*1e9:.2f} ns, {power} mW) hinzugefügt zu Sequenz '{self.name}'")
        return self
    
    def add_metadata(self, key, value):
        """
        Metadaten zur Sequenz hinzufügen.
        
        Parameters
        ----------
        key : str
            Metadatenschlüssel
        value : any
            Metadatenwert
        """
        self.metadata[key] = value
        logger.debug(f"Metadaten hinzugefügt zu Sequenz '{self.name}': {key}={value}")
        return self
    
    def get_duration(self):
        """
        Gesamtdauer der Sequenz berechnen.
        
        Returns
        -------
        float
            Gesamtdauer in Sekunden
        """
        return sum(p.duration for p in self.pulses)
    
    def get_info(self):
        """
        Informationen über die Sequenz erhalten.
        
        Returns
        -------
        dict
            Sequenzinformationen
        """
        return {
            "name": self.name,
            "n_pulses": len(self.pulses),
            "duration": self.get_duration(),
            "pulses": [p.get_info() for p in self.pulses],
            "metadata": self.metadata
        }
    
    def run(self, simulator, measure_points=None):
        """
        Die Pulssequenz auf einem Simulator ausführen.
        
        Parameters
        ----------
        simulator : NVSimulator
            Simulator, auf dem die Sequenz ausgeführt werden soll
        measure_points : list of int, optional
            Indizes, an denen der Systemzustand gemessen werden soll
        
        Returns
        -------
        dict
            Sequenzresultate
        """
        logger.info(f"Ausführen der Pulssequenz '{self.name}' mit {len(self.pulses)} Pulsen")
        
        # Originalzustand und -parameter speichern
        original_state = simulator.state
        original_mw_freq = simulator.nv_system.mw_frequency
        original_mw_power = simulator.nv_system.mw_power
        original_laser_power = simulator.nv_system.laser_power
        
        try:
            # Simulatorzustand zurücksetzen
            simulator.reset()
            
            # Resultatbehälter vorbereiten
            results = {
                "sequence_name": self.name,
                "total_duration": self.get_duration(),
                "n_pulses": len(self.pulses),
                "metadata": self.metadata,
                "measurements": []
            }
            
            # Sequenz ausführen
            for i, pulse in enumerate(self.pulses):
                # Puls anwenden
                pulse.apply(simulator)
                
                # Messen falls angefordert
                if measure_points and i in measure_points:
                    results["measurements"].append({
                        "pulse_index": i,
                        "pulse_info": pulse.get_info(),
                        "time": sum(p.duration for p in self.pulses[:i+1]),
                        "populations": simulator.get_populations(),
                        "fluorescence": simulator.get_fluorescence()
                    })
            
            # Abschließende Messung
            results["final_populations"] = simulator.get_populations()
            results["final_fluorescence"] = simulator.get_fluorescence()
            
            logger.info(f"Pulssequenz '{self.name}' abgeschlossen, Gesamtdauer: {self.get_duration()*1e9:.2f} ns")
            return results
        
        finally:
            # Originalzustand und -parameter wiederherstellen
            simulator.state = original_state
            simulator.nv_system.set_microwave(original_mw_freq, original_mw_power)
            simulator.nv_system.set_laser(original_laser_power)
            
    def __str__(self):
        """String-Repräsentation der Sequenz."""
        return f"PulseSequence('{self.name}', {len(self.pulses)} Pulse, {self.get_duration()*1e9:.2f} ns)"