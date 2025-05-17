import numpy as np
import logging
from nv.core.quantum_state import QuantumState
from nv.physics import constants

# SimOS-Imports
from sim.simos.simos import systems
from sim.simos.simos.systems import NV

logger = logging.getLogger(__name__)

class NVSystem:
    """NV-Zentrum-Quantensystem mit SimOS."""
    
    def __init__(self, config=None):
        """
        Initialisiere NV-Zentrum-System.
        
        Parameters
        ----------
        config : dict, optional
            Konfigurationsparameter
        """
        # Standard-Konfiguration
        self.config = {
            "method": "qutip",
            "optics": True,  # Optische Niveaus einbeziehen
            "nitrogen": False,  # Stickstoff-Kernspin einbeziehen
            "zero_field_splitting": constants.NV_ZERO_FIELD_SPLITTING,  # Hz
            "gyromagnetic_ratio": constants.NV_GYROMAGNETIC_RATIO,  # Hz/T
            "temperature": 300.0,  # K
            "strain": 0.0,  # Hz
            "t1": constants.NV_T1_ROOM_TEMP,  # s
            "t2": constants.NV_T2_ROOM_TEMP,  # s
        }
        
        # Mit angegebener Konfiguration aktualisieren
        if config:
            self.config.update(config)
        
        # Evolutionsmaschine erstellen (wiederverwendbar)
        from nv.core.evolution import QuantumEvolution
        self.evolution = QuantumEvolution(method=self.config["method"])
        
        try:
            # SimOS NV-System initialisieren
            self.simos_nv = NV.NVSystem(
                optics=self.config["optics"],
                nitrogen=self.config["nitrogen"],
                method=self.config["method"]
            )
            
            # Zustand und Operatoren initialisieren
            self.reset_state()
            
            # Steuerparameter
            self.mw_frequency = self.config["zero_field_splitting"]
            self.mw_power = 0.0
            self.laser_power = 0.0
            self.magnetic_field = [0.0, 0.0, 0.0]
            
            # Hamiltonian und Kollaps-Operatoren aktualisieren
            self.update_hamiltonian()
            
        except ImportError as e:
            logger.error(f"SimOS Modul konnte nicht importiert werden: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Fehler bei NV-System-Initialisierung: {str(e)}")
            raise
    
    def reset_state(self):
        """Auf Grundzustand |ms=0⟩ zurücksetzen."""
        # Projektor verwenden, um ms=0-Zustand zu erzeugen
        ms0_projector = self.simos_nv.Sp[0]
        self.state = QuantumState(ms0_projector.unit(), method=self.config["method"])
    
    def update_hamiltonian(self):
        """System-Hamiltonian basierend auf aktuellen Feldern aktualisieren."""
        # Magnetfeld in NumPy-Array konvertieren
        b_field = np.array(self.magnetic_field)
        
        # Temperaturabhängige T1 und T2 Zeiten
        self._update_temperature_effects()
        
        try:
            # NV-Hamiltonian mit SimOS erzeugen
            params = {
                'Bvec': b_field,
                'EGS_vec': np.zeros(3)  # Grundzustand E-Feld
            }
            
            # Strain-Effekte über E-Feld berücksichtigen wenn konfiguriert
            if self.config["strain"] > 0:
                strain_vec = np.array([self.config["strain"], 0, 0])  # X-Richtung
                params['EGS_vec'] = strain_vec
            
            # Optische Übergänge mit E-Feld im angeregten Zustand
            if self.config["optics"]:
                params['EES_vec'] = np.zeros(3)
            
            self.H_free = self.simos_nv.field_hamiltonian(**params)
        except Exception as e:
            logger.error(f"Fehler bei Hamiltonian-Aktualisierung: {str(e)}")
            raise
    
    def _update_temperature_effects(self):
        """Aktualisiert temperaturabhängige Parameter."""
        try:
            # T1-Zeit temperaturabhängig aktualisieren
            temp_K = self.config["temperature"]
            
            # Einfaches Modell: T1 sinkt mit höherer Temperatur
            # Physikalisches Modell: T1 ~ T^-5 für direkte phononische Prozesse
            if temp_K > 0:
                # Referenztemperatur 300K
                t1_room = constants.NV_T1_ROOM_TEMP
                scaling = (300.0 / temp_K)**5
                
                # Begrenzen auf sinnvollen Bereich
                if scaling > 100:
                    scaling = 100  # Limit für sehr niedrige Temperaturen
                
                self.config["t1"] = t1_room * scaling
                
                # T2 kann nie länger als 2*T1 sein (fundamentales Limit)
                if self.config["t2"] > 2*self.config["t1"]:
                    self.config["t2"] = 2*self.config["t1"]
                    
                logger.debug(f"Temperatureffekte aktualisiert: T={temp_K}K, T1={self.config['t1']}s, T2={self.config['t2']}s")
        except Exception as e:
            logger.warning(f"Fehler bei Temperatureffekt-Berechnung: {str(e)}")
            # Standardwerte beibehalten bei Fehler
        
        # Mikrowellen-Antrieb hinzufügen falls nötig
        if self.mw_power > 0:
            # dBm in Amplitude konvertieren
            amplitude = 10**(self.mw_power/20) * 0.01
            
            # Antriebsterm erzeugen
            omega_r = 2 * np.pi * amplitude * 10e6  # Rabi-Frequenz
            # Zeitabhängigen Antriebsterm erzeugen
            from sim.simos.simos import coherent
            def mw_drive_x(t, args):
                return np.cos(2*np.pi*self.mw_frequency*t)
                
            def mw_drive_y(t, args):
                return np.sin(2*np.pi*self.mw_frequency*t)
                
            # Zeitabhängigen Hamiltonian im Format [H0, [H1, f1], [H2, f2], ...] erstellen
            self.H = [self.H_free, [omega_r/2 * self.simos_nv.Sx, mw_drive_x], 
                     [omega_r/2 * self.simos_nv.Sy, mw_drive_y]]
        else:
            self.H = self.H_free
        
        # Kollaps-Operatoren für Dekohärenz erstellen
        self.c_ops = []
        
        # T1-Relaxation
        if self.config["t1"] > 0:
            gamma1 = 1.0 / self.config["t1"]
            self.c_ops.append(np.sqrt(gamma1) * self.simos_nv.Splus)
            self.c_ops.append(np.sqrt(gamma1) * self.simos_nv.Sminus)
        
        # T2-Dephasierung
        if self.config["t2"] > 0:
            gamma2 = 1.0 / self.config["t2"] - 1.0 / (2 * self.config["t1"])
            if gamma2 > 0:
                self.c_ops.append(np.sqrt(gamma2) * self.simos_nv.Sz)
    
    def set_magnetic_field(self, field):
        """
        Magnetfeldvektor setzen.
        
        Parameters
        ----------
        field : list or ndarray
            Magnetfeldvektor [Bx, By, Bz] in Tesla
        """
        self.magnetic_field = field
        self.update_hamiltonian()
    
    def set_microwave(self, frequency, power):
        """
        Mikrowellenparameter setzen.
        
        Parameters
        ----------
        frequency : float
            Mikrowellenfrequenz in Hz
        power : float
            Mikrowellenleistung in dBm
        """
        self.mw_frequency = frequency
        self.mw_power = power
        self.update_hamiltonian()
    
    def set_laser(self, power):
        """
        Laserleistung setzen.
        
        Parameters
        ----------
        power : float
            Laserleistung in mW
        """
        self.laser_power = power
        # Lasereffekte werden während der Evolution angewendet
    
    def __init__(self, config=None):
        # ... vorheriger Code wird beibehalten ...
        
        # Evolutionsmaschine erstellen (wiederverwendbar)
        from nv.core.evolution import QuantumEvolution
        self.evolution = QuantumEvolution(method=self.config["method"])
    
    def evolve(self, duration):
        """
        NV-Zustand für angegebene Dauer entwickeln.
        
        Parameters
        ----------
        duration : float
            Entwicklungszeit in Sekunden
            
        Raises
        ------
        ValueError
            Bei ungültigen Parametern
        RuntimeError
            Bei Fehlern in der Zeitentwicklung
        """
        try:
            # Offene Systemevolution verwenden, wenn Kollaps-Operatoren existieren
            if self.c_ops:
                self.state = self.evolution.open_evolution(
                    self.state, 
                    self.H, 
                    self.c_ops, 
                    duration
                )
            else:
                self.state = self.evolution.unitary_evolution(
                    self.state,
                    self.H,
                    duration
                )
            
            # Optisches Pumpen anwenden, wenn Laser an ist
            if self.laser_power > 0:
                self._apply_optical_pumping()
                
        except Exception as e:
            logger.error(f"Fehler während Zeitevolution: {str(e)}")
            raise RuntimeError(f"Evolution fehlgeschlagen: {str(e)}")
    
    def _apply_optical_pumping(self):
        """Optische Pumpeffekte vom Laser anwenden."""
        # In echtem NV pumpt Laser in Richtung ms=0-Zustand
        # Aktuelle Populationen erhalten
        pops = self.get_populations()
        
        # Pumpstärke basierend auf Laserleistung berechnen
        pump_rate = min(1.0, self.laser_power / 0.5)  # Sättigt bei ~0.5 mW
        
        # Neuen Zustand mit mehr ms=0-Population erzeugen
        ms0_projector = self.simos_nv.Sp[0]
        thermal_dm = QuantumState(ms0_projector.unit(), method=self.config["method"])
        
        # Aktuellen Zustand mit ms=0-Zustand mischen
        from sim.simos.simos import core
        mixed_state = (1-pump_rate) * self.state.get_simos_state() + pump_rate * thermal_dm.get_simos_state()
        
        # Normalisieren und Zustand aktualisieren
        self.state = QuantumState(mixed_state / core.trace(mixed_state), method=self.config["method"])
    
    def get_populations(self):
        """
        Populationen verschiedener Spinzustände erhalten.
        
        Returns
        -------
        dict
            Dictionary mit Schlüsseln 'ms0', 'ms+1', 'ms-1' und ihren Wahrscheinlichkeiten
        """
        # Erwartungswerte für Projektoren berechnen
        p0 = self.state.expect(self.simos_nv.Sp[0])
        pp = self.state.expect(self.simos_nv.Sp[1])
        pm = self.state.expect(self.simos_nv.Sp[2])
        
        return {
            "ms0": float(np.real(p0)),
            "ms+1": float(np.real(pp)),
            "ms-1": float(np.real(pm))
        }
    
    def get_fluorescence(self):
        """
        Fluoreszenzsignal für aktuellen Zustand erhalten.
        
        Returns
        -------
        float
            Fluoreszenzsignal in Zählungen/s
        """
        # NV-Fluoreszenz hängt hauptsächlich von ms=0-Population ab
        ms0_pop = self.get_populations()["ms0"]
        
        # Einfaches Fluoreszenzmodell
        base_rate = 100000.0  # Zählungen/s
        contrast = constants.NV_FLUORESCENCE_CONTRAST  # Kontrast zwischen ms=0 und ms=±1
        
        # Höhere Fluoreszenz für ms=0, niedrigere für ms=±1
        return base_rate * (1.0 - contrast * (1.0 - ms0_pop))