import json
import os
import logging
import numpy as np
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigLoader:
    """
    Lädt und verwaltet Konfigurationen für den digitalen Twin eines NV-Zentrums.
    """
    
    DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialisiert den Konfigurations-Loader.
        
        Parameters
        ----------
        config_path : str, optional
            Pfad zur Konfigurationsdatei. Wenn None, wird die Standard-Konfiguration verwendet.
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Lädt die Konfiguration aus der JSON-Datei."""
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Konfiguration geladen aus: {self.config_path}")
        except FileNotFoundError:
            logger.error(f"Konfigurationsdatei nicht gefunden: {self.config_path}")
            raise
        except json.JSONDecodeError:
            logger.error(f"Fehler beim Parsen der Konfigurationsdatei: {self.config_path}")
            raise
        
    def save_config(self, config_path: Optional[str] = None) -> None:
        """
        Speichert die aktuelle Konfiguration in eine JSON-Datei.
        
        Parameters
        ----------
        config_path : str, optional
            Pfad, unter dem die Konfiguration gespeichert werden soll. 
            Wenn None, wird der aktuelle Konfigurations-Pfad verwendet.
        """
        path = config_path or self.config_path
        try:
            with open(path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logger.info(f"Konfiguration gespeichert in: {path}")
        except Exception as e:
            logger.error(f"Fehler beim Speichern der Konfiguration: {str(e)}")
            raise
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Holt einen Wert aus der Konfiguration mit Unterstützung für verschachtelte Pfade.
        
        Parameters
        ----------
        key_path : str
            Pfad zum gewünschten Wert, mit Punkten als Trennzeichen 
            (z.B. "physical_parameters.t2")
        default : Any, optional
            Rückgabewert, falls der Schlüssel nicht existiert
            
        Returns
        -------
        Any
            Der Wert aus der Konfiguration oder der Default-Wert
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            logger.debug(f"Konfigurationsschlüssel nicht gefunden: {key_path}, Verwende Standard: {default}")
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Setzt einen Wert in der Konfiguration mit Unterstützung für verschachtelte Pfade.
        
        Parameters
        ----------
        key_path : str
            Pfad zum zu setzenden Wert, mit Punkten als Trennzeichen
        value : Any
            Der zu setzende Wert
        """
        keys = key_path.split('.')
        config = self.config
        
        # Bis zum vorletzten Schlüssel navigieren
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Letzten Schlüssel setzen
        config[keys[-1]] = value
        logger.debug(f"Konfigurationswert gesetzt: {key_path} = {value}")
    
    def get_nv_system_config(self) -> Dict[str, Any]:
        """
        Erzeugt eine Konfiguration für das NV-System aus der geladenen Konfiguration.
        
        Returns
        -------
        Dict[str, Any]
            Konfigurationswörterbuch für das NV-System
        """
        nv_config = {
            # Systemkonfiguration
            "method": self.get("system.method", "qutip"),
            "optics": self.get("system.optics", True),
            "nitrogen": self.get("system.nitrogen", False),
            
            # Physikalische Parameter
            "zero_field_splitting": self.get("physical_parameters.zero_field_splitting", 2.87e9),
            "gyromagnetic_ratio": self.get("physical_parameters.gyromagnetic_ratio", 2.8025e10),
            "strain": self.get("physical_parameters.transverse_strain", 0.0),
            "t1": self.get("physical_parameters.t1", 1.0e-3),
            "t2": self.get("physical_parameters.t2", 1.0e-6),
            "temperature": self.get("physical_parameters.temperature", 298.0),
            
            # Fluoreszenzeigenschaften
            "fluorescence_contrast": self.get("optical_properties.fluorescence_contrast", 0.3),
            
            # Experimentelle Parameter
            "power_to_rabi_factor": self.get("experimental.microwave.power_to_rabi_factor", 1.0e5),
        }
        
        return nv_config
    
    def get_hyperfine_tensor(self) -> np.ndarray:
        """
        Erstellt den Hyperfein-Tensor für Stickstoff basierend auf der Konfiguration.
        
        Returns
        -------
        np.ndarray
            3x3 Hyperfein-Tensor
        """
        A_parallel = self.get("hyperfine.nitrogen.A_parallel", -2.16e6)
        A_perp = self.get("hyperfine.nitrogen.A_perpendicular", -2.7e6)
        
        # 3x3 Tensor für Axial-Symmetrische Hyperfein-Kopplung
        hf_tensor = np.diag([A_perp, A_perp, A_parallel])
        return hf_tensor
    
    def get_pulse_shape_config(self, pulse_type: str) -> Dict[str, Any]:
        """
        Holt die Konfiguration für eine bestimmte Pulsform.
        
        Parameters
        ----------
        pulse_type : str
            Art des Pulses ('mw_pi', 'mw_pi2', 'laser_init', 'laser_readout')
            
        Returns
        -------
        Dict[str, Any]
            Pulsform-Konfiguration mit 'shape' und 'shape_params'
        """
        key = f"pulse_shapes.{pulse_type}"
        default = {"shape": "rectangular", "shape_params": {}}
        
        return self.get(key, default)


# Hilfsfunktion für einfachen Zugriff auf die Konfiguration
def load_config(config_path: Optional[str] = None) -> ConfigLoader:
    """
    Lädt und gibt einen Konfigurationsloader zurück.
    
    Parameters
    ----------
    config_path : str, optional
        Pfad zur Konfigurationsdatei
        
    Returns
    -------
    ConfigLoader
        Der Konfigurationsloader mit der geladenen Konfiguration
    """
    return ConfigLoader(config_path)