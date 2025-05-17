import numpy as np
import logging
from nv.utils.logging import get_logger
from typing import Dict, Any, Callable, Optional, Union

logger = get_logger(__name__)

class Pulse:
    """
    Einzelner Kontrollpuls für Quantenkontrolle.
    
    Diese Klasse repräsentiert eine einzelne Kontrolloperatione mit Zeitinformationen und
    unterstützt realistische Pulsformen.
    """
    
    # Verfügbare Pulsformen
    PULSE_SHAPES = {
        'rectangular': lambda t, t_start, t_end, amplitude: 
            amplitude if t_start <= t <= t_end else 0,
        
        'gaussian': lambda t, t_start, t_end, amplitude, sigma=None:
            amplitude * np.exp(-((t - (t_start + t_end)/2)**2) / 
                              (2 * (sigma or (t_end - t_start)/6)**2)) 
            if t_start <= t <= t_end else 0,
        
        'sine': lambda t, t_start, t_end, amplitude:
            amplitude * np.sin(np.pi * (t - t_start) / (t_end - t_start)) 
            if t_start <= t <= t_end else 0,
        
        'trapezoidal': lambda t, t_start, t_end, amplitude, rise_time=None:
            Pulse._trapezoidal(t, t_start, t_end, amplitude, rise_time or (t_end - t_start) * 0.2)
    }
    
    @staticmethod
    def _trapezoidal(t, t_start, t_end, amplitude, rise_time):
        """Trapezförmiger Puls mit Anstiegs- und Abfallzeit."""
        if t < t_start or t > t_end:
            return 0
        
        # Anstiegsphase
        if t < t_start + rise_time:
            return amplitude * (t - t_start) / rise_time
        
        # Abfallphase
        if t > t_end - rise_time:
            return amplitude * (t_end - t) / rise_time
        
        # Plateauphase
        return amplitude
    
    def __init__(self, pulse_type, duration, shape='rectangular', shape_params=None, **params):
        """
        Einen Kontrollpuls initialisieren.
        
        Parameters
        ----------
        pulse_type : str
            Pulstyp ("mw", "laser", "wait")
        duration : float
            Pulsdauer in Sekunden
        shape : str, optional
            Pulsform ('rectangular', 'gaussian', 'sine', 'trapezoidal', 'custom')
        shape_params : dict, optional
            Parameter für spezifische Pulsformen
        **params
            Zusätzliche Pulsparameter:
            - frequency: Frequenz in Hz
            - power: Leistung in passenden Einheiten
            - phase: Phase in Radiant
            - custom_shape_func: Funktion für benutzerdefinierte Pulsform
        """
        self.pulse_type = pulse_type
        self.duration = duration
        self.params = params
        self.shape = shape
        self.shape_params = shape_params or {}
        
        # Zeitabhängige Pulsform
        self.t_start = 0.0  # Wird während der Anwendung gesetzt
        self.custom_shape_func = params.get('custom_shape_func', None)
        
        # Validieren der Pulsparameter
        self._validate()
        
        logger.debug(f"Puls erstellt: {pulse_type}, {shape}, {duration*1e9:.2f} ns")
    
    def _validate(self):
        """Validiert die Pulsparameter."""
        if self.pulse_type not in ["mw", "laser", "wait"]:
            raise ValueError(f"Ungültiger Pulstyp: {self.pulse_type}")
        
        if self.duration < 0:
            raise ValueError(f"Pulsdauer kann nicht negativ sein: {self.duration}")
        
        # Validiere Pulsform
        if self.shape not in self.PULSE_SHAPES and self.shape != 'custom':
            valid_shapes = list(self.PULSE_SHAPES.keys()) + ['custom']
            raise ValueError(f"Ungültige Pulsform: {self.shape}. Erlaubt sind: {', '.join(valid_shapes)}")
            
        if self.shape == 'custom' and not callable(self.custom_shape_func):
            raise ValueError("Bei shape='custom' muss custom_shape_func als Parameter übergeben werden")
        
        # Validiere spezifische Parameter je nach Pulstyp
        if self.pulse_type == "mw":
            if "power" in self.params and not isinstance(self.params["power"], (int, float)):
                raise ValueError(f"Mikrowellenleistung muss eine Zahl sein: {self.params['power']}")
            
            if "frequency" in self.params and not isinstance(self.params["frequency"], (int, float)):
                raise ValueError(f"Mikrowellenfrequenz muss eine Zahl sein: {self.params['frequency']}")
                
            if "phase" in self.params and not isinstance(self.params["phase"], (int, float)):
                raise ValueError(f"Mikrowellenphase muss eine Zahl sein: {self.params['phase']}")
        
        elif self.pulse_type == "laser":
            if "power" in self.params and not isinstance(self.params["power"], (int, float)):
                raise ValueError(f"Laserleistung muss eine Zahl sein: {self.params['power']}")
    
    def get_amplitude_at_time(self, t, relative_time=True):
        """
        Gibt die Amplitude des Pulses zu einem bestimmten Zeitpunkt zurück.
        
        Parameters
        ----------
        t : float
            Zeitpunkt in Sekunden
        relative_time : bool, optional
            Wenn True, wird t als relativ zum Pulsstart interpretiert
            
        Returns
        -------
        float
            Amplitude zum Zeitpunkt t
        """
        # Absolute Zeit berechnen
        t_abs = t + self.t_start if relative_time else t
        t_end = self.t_start + self.duration
        
        # Prüfen, ob Zeit im Pulsbereich liegt
        if t_abs < self.t_start or t_abs > t_end:
            return 0.0
        
        # Amplitude basierend auf Pulsform berechnen
        amplitude = self.params.get("power", 1.0)
        
        if self.shape == 'custom':
            if not self.custom_shape_func:
                return amplitude  # Fallback
            return self.custom_shape_func(t_abs, self.t_start, t_end, amplitude, **self.shape_params)
        else:
            shape_func = self.PULSE_SHAPES.get(self.shape, self.PULSE_SHAPES['rectangular'])
            return shape_func(t_abs, self.t_start, t_end, amplitude, **self.shape_params)
    
    def apply(self, simulator, current_time=0.0):
        """
        Diesen Puls auf einen Simulator anwenden.
        
        Parameters
        ----------
        simulator : NVSimulator
            Simulator, auf den der Puls angewendet werden soll
        current_time : float, optional
            Aktuelle Zeit in der Sequenz, wird für zeitabhängige Pulse verwendet
        """
        try:
            # Pulsstart-Zeit für zeitabhängige Anwendung setzen
            self.t_start = current_time
            
            # Zeitintervalle für komplexe Pulsformen
            steps = 1  # Rechteckpuls braucht nur einen Schritt
            
            # Bei komplexen Pulsformen die Zeitentwicklung in Schritten durchführen
            if self.shape != 'rectangular' and self.duration > 0:
                steps = max(10, int(self.duration * 1e9 / 10))  # ~10 ns Auflösung
                dt = self.duration / steps
                
                for i in range(steps):
                    t = current_time + i * dt
                    amp = self.get_amplitude_at_time(i * dt, relative_time=True)
                    
                    # Puls mit aktueller Amplitude anwenden
                    self._apply_step(simulator, amp)
                    
                    # Für einen Zeitschritt entwickeln
                    simulator.evolve(dt)
                
                return  # Nach schrittweiser Entwicklung hier beenden
            
            # Standardfall: Rechteckpuls oder Wartepuls
            if self.pulse_type == "mw":
                # Mikrowelle anwenden
                frequency = self.params.get("frequency", simulator.nv_system.mw_frequency)
                power = self.params.get("power", 0.0)
                phase = self.params.get("phase", 0.0)  # Phase berücksichtigen
                
                # Mikrowelle mit Phase
                if phase != 0.0:
                    # Phase in zeitabhängigen Hamiltonian einbauen
                    # Hier müsste ein komplexerer Aufruf erfolgen
                    pass
                
                simulator.nv_system.set_microwave(frequency, power)
                logger.debug(f"Mikrowellenpuls angewendet: {frequency/1e9:.4f} GHz, {power} dBm, Form: {self.shape}")
            
            elif self.pulse_type == "laser":
                # Laser anwenden
                power = self.params.get("power", 1.0)
                simulator.nv_system.set_laser(power)
                logger.debug(f"Laserpuls angewendet: {power} mW, Form: {self.shape}")
            
            elif self.pulse_type == "wait":
                # Keine Kontrollfelder während des Wartens
                simulator.nv_system.set_microwave(0, -100)
                simulator.nv_system.set_laser(0)
                logger.debug(f"Wartepuls angewendet: {self.duration*1e9:.1f} ns")
            
            else:
                # Sollte durch Validierung abgefangen werden
                raise ValueError(f"Unbekannter Pulstyp: {self.pulse_type}")
            
            # Für Pulsdauer entwickeln
            if self.duration > 0:
                simulator.evolve(self.duration)
                
        except Exception as e:
            logger.error(f"Fehler bei Pulsanwendung ({self.pulse_type}): {str(e)}")
            raise
            
    def _apply_step(self, simulator, amplitude):
        """Wendet einen einzelnen Schritt eines Pulses mit angegebener Amplitude an."""
        if self.pulse_type == "mw":
            frequency = self.params.get("frequency", simulator.nv_system.mw_frequency)
            # Amplitude aus Pulsform anstatt voller Leistung verwenden
            effective_power = 20 * np.log10(amplitude * 100) if amplitude > 0 else -100
            simulator.nv_system.set_microwave(frequency, effective_power)
            
        elif self.pulse_type == "laser":
            # Amplitude aus Pulsform verwenden
            simulator.nv_system.set_laser(amplitude)
            
        elif self.pulse_type == "wait":
            # Beim Warten keine Felder anlegen
            simulator.nv_system.set_microwave(0, -100)
            simulator.nv_system.set_laser(0)
    
    def get_info(self):
        """
        Gibt Informationen über den Puls zurück.
        
        Returns
        -------
        dict
            Pulsinformationen
        """
        return {
            "type": self.pulse_type,
            "duration": self.duration,
            "parameters": self.params
        }
    
    def __str__(self):
        """String-Repräsentation des Pulses."""
        if self.pulse_type == "mw":
            frequency = self.params.get("frequency", "default")
            power = self.params.get("power", 0)
            return f"MW-Puls({self.duration*1e9:.2f} ns, {frequency/1e9 if isinstance(frequency, (int, float)) else frequency} GHz, {power} dBm)"
        
        elif self.pulse_type == "laser":
            power = self.params.get("power", 1.0)
            return f"Laser-Puls({self.duration*1e9:.2f} ns, {power} mW)"
        
        elif self.pulse_type == "wait":
            return f"Warte-Puls({self.duration*1e9:.2f} ns)"
        
        return f"Puls({self.pulse_type}, {self.duration*1e9:.2f} ns)"