# Qudi-Quantum Hardware Simulator Spezifikation

Diese Dokumentation definiert alle notwendigen Schnittstellen, Datenstrukturen und Kommunikationsprotokolle, um einen vollständigen Hardware-Simulator für Qudi zu implementieren, der sich wie ein echter Quantencomputer oder NV-Zentren-Setup verhält.

## 1. Interface-Implementierungen

### 1.1 Fast Counter Interface

**Klassenreferenz:** `qudi-iqo-modules/src/qudi/interface/fast_counter_interface.py`

**Zu implementierende Methoden:**
```python
def get_constraints(self) -> FastCounterConstraints:
    """Liefert die Hardware-Einschränkungen des Fast Counters"""
    pass

def configure(self, bin_width_s: float, record_length_s: float, number_of_gates: int = 0) -> None:
    """Konfiguriert die Zählparameter"""
    pass
    
def start_measure(self) -> None:
    """Startet die Zählung"""
    pass
    
def stop_measure(self) -> None:
    """Stoppt die Zählung"""
    pass
    
def pause_measure(self) -> None:
    """Pausiert die Zählung"""
    pass
    
def continue_measure(self) -> None:
    """Setzt die Zählung fort"""
    pass
    
def get_data_trace(self) -> numpy.ndarray:
    """Liefert die erfassten Zähldaten
    @return numpy.ndarray: 1D-Array mit Zählraten pro Bin
    """
    pass
```

**Datenstrukturen:**
```python
class FastCounterConstraints:
    """Hardware-Einschränkungen für den Fast Counter"""
    # Minimale/maximale Binbreite in Sekunden
    min_bin_width: float
    max_bin_width: float
    
    # Minimale/maximale Aufnahmelänge in Sekunden  
    min_record_length: float
    max_record_length: float
    
    # Maximale Anzahl von Gates (0 = unbeschränkt)
    max_number_of_gates: int
```

**Statusvariablen:**
- `is_running`: Bool - Aktueller Betriebszustand des Counters
- `bin_width`: Float - Aktuelle Binbreite in Sekunden
- `record_length`: Float - Aktuelle Aufzeichnungslänge in Sekunden
- `number_of_gates`: Int - Aktuelle Anzahl von Gates

**Physikalisches NV-Modell für Simulator:**
- Photonenzählung folgt Poisson-Verteilung
- Grundzustand: ~250-300 kcounts/s bei 1mW Laseranregung
- Angeregter Zustand: ~150-180 kcounts/s bei 1mW Laseranregung
- Sättigungsverhalten bei hoher Laserleistung
- Zeit- und zustandsabhängige Fluoreszenz nach gepulster Anregung

### 1.2 Microwave Interface

**Klassenreferenz:** `qudi-iqo-modules/src/qudi/interface/microwave_interface.py`

**Zu implementierende Methoden:**
```python
@property
def constraints(self) -> MicrowaveConstraints:
    """Liefert die Hardware-Einschränkungen"""
    pass

@property
def is_scanning(self) -> bool:
    """Gibt an, ob ein Frequenzscan läuft"""
    pass
    
@property
def cw_power(self) -> float:
    """Aktuelle CW-Leistung in dBm"""
    pass
    
@property
def cw_frequency(self) -> float:
    """Aktuelle CW-Frequenz in Hz"""
    pass
    
@property
def scan_power(self) -> float:
    """Leistung für den Frequenzscan in dBm"""
    pass
    
@property
def scan_frequencies(self) -> Union[numpy.ndarray, Tuple[float, float, int]]:
    """Frequenzen für den Scan in Hz"""
    pass
    
@property
def scan_mode(self) -> SamplingOutputMode:
    """Modus des Frequenzscans"""
    pass
    
@property
def scan_sample_rate(self) -> float:
    """Abtastrate des Scans in Hz"""
    pass
    
def off(self) -> None:
    """Schaltet den Mikrowellenausgang aus"""
    pass
    
def set_cw(self, frequency: float, power: float) -> None:
    """Konfiguriert den CW-Modus"""
    pass
    
def cw_on(self) -> None:
    """Aktiviert den CW-Ausgang"""
    pass
    
def configure_scan(self, power: float, frequencies: Union[numpy.ndarray, Tuple[float, float, int]], 
                   mode: SamplingOutputMode, sample_rate: float) -> None:
    """Konfiguriert einen Frequenzscan"""
    pass
    
def start_scan(self) -> None:
    """Startet den konfigurierten Frequenzscan"""
    pass
    
def reset_scan(self) -> None:
    """Setzt den Scan zurück zur Startfrequenz"""
    pass
```

**Datenstrukturen:**
```python
class MicrowaveConstraints:
    """Hardware-Einschränkungen für Mikrowellenquelle"""
    # Leistungsgrenzen in dBm
    power_limits: Tuple[float, float]
    
    # Frequenzgrenzen in Hz
    frequency_limits: Tuple[float, float]
    
    # Grenzen für Scangröße (Anzahl von Punkten)
    scan_size_limits: Tuple[int, int]
    
    # Grenzen für Abtastrate in Hz
    sample_rate_limits: Tuple[float, float]
    
    # Unterstützte Scanmodi
    scan_modes: Tuple[SamplingOutputMode, ...]

class SamplingOutputMode(Enum):
    """Scanmodi für die Mikrowellenquelle"""
    JUMP_LIST = 0  # Direkte Sprünge zwischen diskreten Frequenzen
    EQUIDISTANT_SWEEP = 1  # Gleichmäßiges Durchstimmen über Frequenzbereich
```

**Statusvariablen:**
- `module_state`: String - Status des Moduls ('idle', 'running')
- `cw_power`: Float - Aktuelle CW-Leistung in dBm
- `cw_frequency`: Float - Aktuelle CW-Frequenz in Hz
- `scan_power`: Float - Scan-Leistung in dBm
- `scan_frequencies`: Array/Tuple - Scan-Frequenzen
- `scan_mode`: Enum - Aktueller Scan-Modus
- `scan_sample_rate`: Float - Abtastrate in Hz
- `is_scanning`: Bool - Status des Frequenzscans

**Physikalisches NV-Modell für Simulator:**
- Resonanz bei ~2.87 GHz für NV-Zentren bei null-Magnetfeld
- Zeeman-Aufspaltung: ~2.8 MHz/Gauss
- Hyperfein-Aufspaltung für N14: Triplett mit ~2.2 MHz Abstand
- Hyperfein-Aufspaltung für N15: Dublett mit ~3.1 MHz Abstand
- ODMR-Linienbreite: ~5-15 MHz abhängig von T2*
- Rabi-Oszillationen mit leistungsabhängiger Frequenz (~10-20 MHz bei 30 dBm)

### 1.3 Pulser Interface

**Klassenreferenz:** `qudi-iqo-modules/src/qudi/interface/pulser_interface.py`

**Zu implementierende Methoden:**
```python
def get_constraints(self) -> PulserConstraints:
    """Liefert die Hardware-Einschränkungen des Pulsers"""
    pass

def pulser_on(self) -> int:
    """Aktiviert den Pulsgenerator
    @return int: Fehlercode (0:OK, -1:Fehler)
    """
    pass
    
def pulser_off(self) -> int:
    """Deaktiviert den Pulsgenerator
    @return int: Fehlercode (0:OK, -1:Fehler)
    """
    pass
    
def load_waveform(self, load_dict: Union[Dict[int, str], List[str]]) -> Dict[int, str]:
    """Lädt eine Wellenform in die spezifizierten Kanäle
    @return dict: Tatsächlich geladene Wellenformen pro Kanal
    """
    pass
    
def load_sequence(self, sequence_name: Union[Dict[int, str], List[str]]) -> Dict[int, str]:
    """Lädt eine Sequenz in die Kanäle
    @return dict: Tatsächlich geladene Sequenzen pro Kanal
    """
    pass
    
def get_loaded_assets(self) -> Tuple[Dict[int, str], str]:
    """Gibt die aktuell geladenen Assets zurück
    @return tuple: Dictionary mit Kanalnummern und Assets, Asset-Typ
    """
    pass
    
def clear_all(self) -> int:
    """Löscht alle geladenen Wellenformen
    @return int: Fehlercode (0:OK, -1:Fehler)
    """
    pass
    
def get_status(self) -> Tuple[int, Dict[str, str]]:
    """Gibt den Gerätestatus zurück
    @return tuple: Statuscode, Dictionary mit Statusbeschreibungen
    """
    pass
    
def get_sample_rate(self) -> float:
    """Gibt die aktuelle Abtastrate zurück
    @return float: Abtastrate in Hz
    """
    pass
    
def set_sample_rate(self, sample_rate: float) -> float:
    """Setzt die Abtastrate
    @return float: Tatsächlich gesetzte Abtastrate in Hz
    """
    pass
    
def get_analog_level(self, amplitude: List[str] = None, offset: List[str] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Gibt Amplitude und Offset der analogen Kanäle zurück
    @return tuple: Amplituden-Dict, Offset-Dict (jeweils in Volt)
    """
    pass
    
def set_analog_level(self, amplitude: Dict[str, float] = None, offset: Dict[str, float] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Setzt Amplitude und Offset der analogen Kanäle
    @return tuple: Tatsächlich gesetzte Amplituden und Offsets
    """
    pass
    
def get_digital_level(self, low: List[str] = None, high: List[str] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Gibt Low- und High-Pegel der digitalen Kanäle zurück
    @return tuple: Low-Dict, High-Dict (jeweils in Volt)
    """
    pass
    
def set_digital_level(self, low: Dict[str, float] = None, high: Dict[str, float] = None) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Setzt Low- und High-Pegel der digitalen Kanäle
    @return tuple: Tatsächlich gesetzte Low- und High-Pegel
    """
    pass
    
def get_active_channels(self, ch: List[str] = None) -> Dict[str, bool]:
    """Gibt die aktiven Kanäle zurück
    @return dict: Kanal-Aktivierungszustand {Kanalname: bool}
    """
    pass
    
def set_active_channels(self, ch: Dict[str, bool] = None) -> Dict[str, bool]:
    """Aktiviert/deaktiviert Kanäle
    @return dict: Tatsächlicher Aktivierungszustand aller Kanäle
    """
    pass
    
def write_waveform(self, name: str, analog_samples: Dict[str, numpy.ndarray], 
                  digital_samples: Dict[str, numpy.ndarray], is_first_chunk: bool, 
                  is_last_chunk: bool, total_number_of_samples: int) -> Tuple[int, List[str]]:
    """Schreibt eine Wellenform in den Gerätespeicher
    @return tuple: Anzahl geschriebener Samples, Liste erstellter Wellenformen
    """
    pass
    
def write_sequence(self, name: str, sequence_parameters: List[Tuple[List[str], SequenceStep]]) -> int:
    """Schreibt eine Sequenz in den Gerätespeicher
    @return int: Anzahl der geschriebenen Sequenzschritte
    """
    pass
    
def get_waveform_names(self) -> List[str]:
    """Gibt die Namen aller hochgeladenen Wellenformen zurück
    @return list: Liste aller hochgeladenen Wellenformnamen
    """
    pass
    
def get_sequence_names(self) -> List[str]:
    """Gibt die Namen aller hochgeladenen Sequenzen zurück
    @return list: Liste aller hochgeladenen Sequenznamen
    """
    pass
    
def delete_waveform(self, waveform_name: Union[str, List[str]]) -> List[str]:
    """Löscht die angegebene Wellenform
    @return list: Liste der gelöschten Wellenformnamen
    """
    pass
    
def delete_sequence(self, sequence_name: Union[str, List[str]]) -> List[str]:
    """Löscht die angegebene Sequenz
    @return list: Liste der gelöschten Sequenznamen
    """
    pass
    
def get_interleave(self) -> bool:
    """Prüft, ob Interleave aktiviert ist
    @return bool: True: AN, False: AUS
    """
    pass
    
def set_interleave(self, state: bool = False) -> bool:
    """Schaltet Interleave um
    @return bool: Tatsächlicher Interleave-Status
    """
    pass
    
def reset(self) -> int:
    """Setzt das Gerät zurück
    @return int: Fehlercode (0:OK, -1:Fehler)
    """
    pass
```

**Datenstrukturen:**
```python
class PulserConstraints:
    """Hardware-Einschränkungen für den Pulsgenerator"""
    # Abtastrate (Zeitbasis des Pulsers)
    sample_rate: ScalarConstraint
    
    # Amplitude und Offset der analogen Kanäle
    a_ch_amplitude: ScalarConstraint
    a_ch_offset: ScalarConstraint
    
    # Low- und High-Pegel der digitalen Kanäle
    d_ch_low: ScalarConstraint
    d_ch_high: ScalarConstraint
    
    # Länge der erzeugten Wellenform in Samples
    waveform_length: ScalarConstraint
    
    # Anzahl von Wellenformen/Sequenzen pro Asset
    waveform_num: ScalarConstraint
    sequence_num: ScalarConstraint
    subsequence_num: ScalarConstraint
    
    # Sequenzparameter
    sequence_steps: ScalarConstraint
    repetitions: ScalarConstraint
    
    # Ereignistrigger und Flags
    event_triggers: List[str]
    flags: List[str]
    
    # Aktivierungskonfigurationen
    activation_config: Dict[str, frozenset]
    
    # Sequenzoptionen
    sequence_option: SequenceOption

class SequenceOption(Enum):
    """Optionen für den Sequenzmodus im Pulser"""
    NON = 0  # Kein Sequenzmodus, nur Wellenformen
    OPTIONAL = 1  # Kann mit Wellenformen oder Sequenzen arbeiten
    FORCED = 2  # Ausgabe nur für Sequenzen erlaubt

class SequenceStep:
    """Repräsentiert einen Schritt in einer Pulssequenz"""
    # Wiederholungen dieses Schritts
    repetitions: int
    
    # Bedingte Verzweigungsoptionen
    go_to: int
    event_jump_to: Dict[str, int]
    event_trigger: str
    
    # Ausgabeflags für diesen Schritt
    flags: List[str]
```

**Statusvariablen:**
- `is_on`: Bool - Ob der Pulser aktiviert ist
- `loaded_waveforms`: Dict - Aktuell geladene Wellenformen pro Kanal
- `loaded_sequences`: Dict - Aktuell geladene Sequenzen pro Kanal
- `sample_rate`: Float - Aktuelle Abtastrate in Hz
- `interleave`: Bool - Interleave-Status
- `active_channels`: Dict - Aktivierte Kanäle

**Physikalisches NV-Modell für Simulator:**
- π-Puls für NV: Typischerweise 40-100 ns bei 30 dBm
- Spinevolution gemäß Blochgleichungen
- T1-Relaxation: ~1-5 ms bei Raumtemperatur
- T2*-Dekohärenz: ~1-5 µs je nach Umgebung
- T2-Dekohärenz: ~100-500 µs mit Spin-Echo
- Dynamische Entkopplungssequenzen (XY8, CPMG) verlängern Kohärenzzeit

### 1.4 Scanning Probe Interface

**Klassenreferenz:** `qudi-iqo-modules/src/qudi/interface/scanning_probe_interface.py`

**Zu implementierende Methoden:**
```python
@property
def constraints(self) -> ScanConstraints:
    """Liefert die Hardware-Einschränkungen des Scanners"""
    pass

def reset(self) -> None:
    """Setzt die Hardware zurück"""
    pass
    
@property
def scan_settings(self) -> Optional[ScanSettings]:
    """Gibt alle für einen 1D- oder 2D-Scan nötigen Parameter zurück"""
    pass
    
@property
def back_scan_settings(self) -> Optional[ScanSettings]:
    """Gibt die Parameter des Rückwärtsscans zurück"""
    pass
    
def configure_scan(self, settings: ScanSettings) -> None:
    """Konfiguriert alle für einen 1D- oder 2D-Scan nötigen Parameter"""
    pass
    
def configure_back_scan(self, settings: ScanSettings) -> None:
    """Konfiguriert die Parameter des Rückwärtsscans"""
    pass
    
def move_absolute(self, position: Dict[str, float], velocity: Optional[float] = None, 
                  blocking: bool = False) -> Dict[str, float]:
    """Bewegt den Scanner auf eine absolute Position
    @return dict: Neue Position aller Achsen
    """
    pass
    
def move_relative(self, distance: Dict[str, float], velocity: Optional[float] = None, 
                  blocking: bool = False) -> Dict[str, float]:
    """Bewegt den Scanner um einen relativen Abstand
    @return dict: Neue Position aller Achsen
    """
    pass
    
def get_target(self) -> Dict[str, float]:
    """Gibt die aktuelle Zielposition zurück
    @return dict: Aktuelle Zielposition pro Achse
    """
    pass
    
def get_position(self) -> Dict[str, float]:
    """Gibt die tatsächliche Position zurück
    @return dict: Aktuelle Position pro Achse
    """
    pass
    
def start_scan(self) -> None:
    """Startet einen zuvor konfigurierten Scan"""
    pass
    
def stop_scan(self) -> None:
    """Stoppt den laufenden Scan"""
    pass
    
def get_scan_data(self) -> Optional[ScanData]:
    """Gibt die ScanData-Instanz zurück, die im Scan verwendet wird"""
    pass
    
def get_back_scan_data(self) -> Optional[ScanData]:
    """Gibt die ScanData-Instanz für den Rückwärtsscan zurück"""
    pass
    
def emergency_stop(self) -> None:
    """Führt einen Notfall-Stopp durch"""
    pass
```

**Datenstrukturen:**
```python
@dataclass(frozen=True)
class ScannerChannel:
    """Repräsentiert einen Scanner-Kanal und seine Einschränkungen"""
    name: str
    unit: str = ''
    dtype: str = 'float64'

@dataclass(frozen=True)
class ScannerAxis:
    """Repräsentiert eine Scanachse und ihre Einschränkungen"""
    name: str
    unit: str
    position: ScalarConstraint  # Positionsbeschränkungen
    step: ScalarConstraint      # Schrittgrößenbeschränkungen
    resolution: ScalarConstraint  # Auflösungsbeschränkungen
    frequency: ScalarConstraint   # Frequenzbeschränkungen

@dataclass(frozen=True)
class ScanSettings:
    """Enthält alle für einen Scan notwendigen Einstellungen"""
    channels: Tuple[str, ...]  # Namen der Scannerkanäle
    axes: Tuple[str, ...]      # Namen der Scannerachsen
    range: Tuple[Tuple[float, float], ...]  # Bereich für jede Scanachse
    resolution: Tuple[int, ...]  # Anzahl der Punkte für jede Scanachse
    frequency: float           # Scan-Pixelfrequenz der schnellen Achse
    position_feedback_axes: Tuple[str, ...] = field(default_factory=tuple)

@dataclass(frozen=True)
class ScanConstraints:
    """Enthält die vollständigen Einschränkungen einer Scanmessung"""
    channel_objects: Tuple[ScannerChannel, ...]
    axis_objects: Tuple[ScannerAxis, ...]
    back_scan_capability: BackScanCapability
    has_position_feedback: bool
    square_px_only: bool
    
    @property
    def channels(self) -> Dict[str, ScannerChannel]:
        """Liefert ein Dictionary mit Kanalname -> Kanalobjekt"""
        pass
        
    @property
    def axes(self) -> Dict[str, ScannerAxis]:
        """Liefert ein Dictionary mit Achsname -> Achsobjekt"""
        pass

@dataclass
class ScanData:
    """Enthält Einstellungen und Ergebnisse einer Scanmessung"""
    settings: ScanSettings  # Scan-Einstellungen
    _channel_units: Tuple[str, ...]  # Einheiten für alle Kanäle
    _channel_dtypes: Tuple[str, ...]  # Datentypen für alle Kanäle
    _axis_units: Tuple[str, ...]  # Einheiten für alle Achsen
    scanner_target_at_start: Dict[str, float] = field(default_factory=dict)
    timestamp: Optional[datetime.datetime] = None
    _data: Optional[Tuple[np.ndarray, ...]] = None  # Scandaten
    _position_data: Optional[Tuple[np.ndarray, ...]] = None  # Positionsdaten
    coord_transform_info: Dict[str, Any] = field(default_factory=dict)

class BackScanCapability(Flag):
    """Verfügbarkeit und Konfigurierbarkeit des Rückwärtsscans"""
    AVAILABLE = auto()
    FREQUENCY_CONFIGURABLE = auto()
    RESOLUTION_CONFIGURABLE = auto()
    FULLY_CONFIGURABLE = FREQUENCY_CONFIGURABLE | RESOLUTION_CONFIGURABLE
```

**Statusvariablen:**
- `is_scanning`: Bool - Ob ein Scan läuft
- `target_position`: Dict - Aktuelle Zielposition pro Achse
- `current_position`: Dict - Tatsächliche Position pro Achse
- `scan_progress`: Float - Fortschritt des aktuellen Scans (0-100%)

**Physikalisches NV-Modell für Simulator:**
- NV-Zentren haben typische Größe von ~5-10 nm
- Räumliche Auflösung durch Beugungsbegrenzung: ~250-300 nm für 532 nm Anregung
- PSF (Point Spread Function) folgt Gauß-Verteilung
- Rauschen in Positionierung: ~5-10 nm Jitter
- Laterale Scanbereiche: ~100 µm x 100 µm
- Tiefenbereich: ~10-20 µm

### 1.5 Simple Laser Interface

**Klassenreferenz:** `qudi-iqo-modules/src/qudi/interface/simple_laser_interface.py`

**Zu implementierende Methoden:**
```python
@property
def constraints(self) -> LaserConstraints:
    """Liefert die Hardware-Einschränkungen des Lasers"""
    pass

def on(self) -> None:
    """Schaltet den Laserausgang ein"""
    pass
    
def off(self) -> None:
    """Schaltet den Laserausgang aus"""
    pass
    
def get_power(self) -> float:
    """Gibt die aktuelle Laserleistung zurück
    @return float: Leistung in Watt
    """
    pass
    
def set_power(self, power: float) -> None:
    """Setzt die Laserleistung
    @param float power: Leistung in Watt
    """
    pass
    
def get_power_range(self) -> Tuple[float, float]:
    """Gibt den erlaubten Leistungsbereich zurück
    @return tuple(float, float): min, max Ausgangsleistung in Watt
    """
    pass
    
def get_current_unit(self) -> str:
    """Gibt die Einheit der Stromstärke zurück
    @return str: Stromeinheit (z.B. 'A')
    """
    pass
    
def get_current(self) -> float:
    """Gibt die aktuelle Laserstromstärke zurück
    @return float: Strom in der Einheit von get_current_unit
    """
    pass
    
def get_current_range(self) -> Tuple[float, float]:
    """Gibt den erlaubten Strombereich zurück
    @return tuple(float, float): min, max Strom
    """
    pass
    
def get_power_setpoint(self) -> float:
    """Gibt den Leistungs-Sollwert zurück
    @return float: Sollwert in Watt
    """
    pass
    
def get_shutter_state(self) -> bool:
    """Gibt den Status des Shutters zurück
    @return bool: True=geöffnet, False=geschlossen
    """
    pass
    
def set_shutter_state(self, state: bool) -> None:
    """Setzt den Status des Shutters
    @param bool state: True=öffnen, False=schließen
    """
    pass
```

**Datenstrukturen:**
```python
class LaserConstraints:
    """Enthält die Hardware-Einschränkungen des Lasers"""
    wavelength_range: Tuple[float, float]  # in m
    power_range: Tuple[float, float]       # in W
    power_setpoint_range: Tuple[float, float]  # in W
    current_range: Tuple[float, float]     # in A (oder andere Einheit)
    has_shutter: bool
    shutter_minimum_state_duration: float  # in s
```

**Statusvariablen:**
- `is_on`: Bool - Ob der Laser eingeschaltet ist
- `power`: Float - Aktuelle Leistung in Watt
- `current`: Float - Aktueller Strom
- `shutter_open`: Bool - Status des Shutters

**Physikalisches NV-Modell für Simulator:**
- Typische Anregungswellenlänge: 532 nm (grüner Laser)
- Leistungsbereich: 0-100 mW typisch für Einzelemitter
- Sättigungsverhalten bei ~1-5 mW fokussierter Leistung
- Photobleaching bei hoher Leistung über lange Zeit
- Emissionsbereich: 637-750 nm (Nullphononenlinie bei 637 nm)

### 1.6 Switch Interface

**Klassenreferenz:** `qudi-iqo-modules/src/qudi/interface/switch_interface.py`

**Zu implementierende Methoden:**
```python
@property
def constraints(self) -> SwitchConstraints:
    """Liefert die Hardware-Einschränkungen des Switches"""
    pass

def get_state(self, switch: Optional[Union[str, List[str]]] = None) -> Dict[str, int]:
    """Gibt den Zustand des Switches zurück
    @return dict: Aktuelle Zustände {Switchname: Zustandsnummer}
    """
    pass
    
def set_state(self, state: Dict[str, int]) -> Dict[str, int]:
    """Setzt den Zustand des Switches
    @return dict: Tatsächlich gesetzte Zustände
    """
    pass
    
def get_state_names(self, switch: Optional[str] = None) -> Dict[str, Dict[int, str]]:
    """Gibt die Namen der Zustände zurück
    @return dict: Name jedes Zustands {Switchname: {Zustandsnummer: Zustandsname}}
    """
    pass
    
def get_switch_names(self) -> List[str]:
    """Gibt die Namen aller Switches zurück
    @return list: Liste aller Switchnamen
    """
    pass
```

**Datenstrukturen:**
```python
class SwitchConstraints:
    """Enthält die Hardware-Einschränkungen des Switches"""
    # Mögliche Switch-Zustände {Name: [Zustände]}
    states: Dict[str, List[int]]
    
    # Name jedes Zustands {Name: {Zustandsnummer: Zustandsname}}
    state_names: Dict[str, Dict[int, str]]
```

**Statusvariablen:**
- `states`: Dict - Aktuelle Zustände aller Switches
- `switch_names`: List - Namen aller verfügbaren Switches

**Physikalisches NV-Modell für Simulator:**
- RF-Switches für Signaleroutung (Mikrowave/APD)
- Optische Pfad-Switches (Einzelphotonendetektion/Spektroskopie)
- Polarisationskontrolle
- Strahlendichtemodulation

## 2. Datenflussmodelle

### 2.1 ODMR-Messung

```
Ablauf:
1. Laser ON (simple_laser_interface)
2. Für jede Frequenz im Frequenzbereich:
   a. Mikrowell-Frequenz setzen (microwave_interface)
   b. Mikrowell ON (microwave_interface)
   c. Zählung integrieren (fast_counter_interface)
   d. Mikrowell OFF (microwave_interface)
3. Laser OFF (simple_laser_interface)

Datenstruktur:
{
  'frequency': numpy.ndarray,  # Frequenzen in Hz
  'counts': numpy.ndarray,     # Zählraten in counts/s
  'fit_parameters': {          # Parameter nach Fit
    'baseline': float,         # Grundlinie
    'contrast': float,         # Kontrast (in %)
    'center_frequency': float, # Resonanzfrequenz in Hz
    'linewidth': float         # Linienbreite in Hz
  }
}
```

### 2.2 Rabi-Oszillationen

```
Ablauf:
1. Für jede Pulsdauer im Bereich:
   a. Pulssequenz programmieren (pulser_interface):
      - Laser-Initialisierung (532 nm, ~3 µs)
      - MW-Puls mit variabler Dauer
      - Laser-Auslese (532 nm, ~300 ns)
   b. Sequenz starten (pulser_interface)
   c. Photonen zählen während Auslesefenster (fast_counter_interface)
   d. Sequenz wiederholen für Signal-Mittelung

Datenstruktur:
{
  'pulse_duration': numpy.ndarray,  # Pulsdauern in s
  'normalized_counts': numpy.ndarray,  # Normierte Photonenzahlen
  'fit_parameters': {
    'amplitude': float,        # Rabi-Amplitude
    'frequency': float,        # Rabi-Frequenz in Hz
    'phase': float,            # Phasenverschiebung in rad
    'offset': float,           # Offset
    'decay_time': float        # T2* Zeit in s
  }
}
```

### 2.3 Konfokales Scanning

```
Ablauf:
1. Scanner konfigurieren (scanning_probe_interface)
2. Laser ON (simple_laser_interface)
3. Scan starten (scanning_probe_interface)
4. Für jeden Pixel:
   a. Position anfahren
   b. Zählung integrieren (fast_counter_interface)
5. Scan beenden (scanning_probe_interface)
6. Laser OFF (simple_laser_interface)

Datenstruktur:
{
  'x_range': (float, float),   # Scanbereich X-Achse in m
  'y_range': (float, float),   # Scanbereich Y-Achse in m
  'z_position': float,         # Z-Position in m
  'resolution': (int, int),    # Pixel-Auflösung (X, Y)
  'image_data': numpy.ndarray, # 2D-Array mit Zählraten
  'channel_units': str,        # Einheit der Zählraten
  'axis_units': (str, str)     # Einheiten der Achsen
}
```

## 3. NV-Zentrum Quantenmodell

### 3.1 Grundzustandshamiltonian

```
H = D*S_z^2 + E*(S_x^2 - S_y^2) + gamma_e*B_z*S_z + gamma_e*B_x*S_x + gamma_e*B_y*S_y + A_z*I_z*S_z

Dabei ist:
- D ≈ 2.87 GHz: Nullfeld-Aufspaltung
- E ≈ 0-5 MHz: Strain-Parameter
- gamma_e ≈ 28 MHz/mT: Gyromagnetisches Verhältnis des Elektrons
- B_z, B_x, B_y: Magnetfeldkomponenten
- A_z ≈ 2.2 MHz: Hyperfeinaufspaltungskonstante (N14)
- S_z, S_x, S_y: Elektronenspin-Operatoren
- I_z: Kernspin-Operator
```

### 3.2 Optischer Zyklus

```
Raten für die Simulation:
- Anregungsrate: k_ex ≈ 10^7 s^-1 bei 1 mW Laser
- Emissionsrate: k_em ≈ 7*10^7 s^-1
- Nicht-strahlender Übergang: k_nr ≈ 3*10^7 s^-1
- Intersystem Crossing (spinabhängig):
  - m_s=0: k_isc,0 ≈ 5*10^6 s^-1
  - m_s=±1: k_isc,±1 ≈ 5*10^7 s^-1
- Rückkehr aus Singulett: k_s ≈ 3*10^6 s^-1
```

### 3.3 Kohärenzzeiten

```
Typische Werte:
- T1 (Spin-Gitter-Relaxation): 1-5 ms bei Raumtemperatur
- T2* (freier Induktionszerfall): 1-5 µs
- T2 (Spin-Echo): 100-500 µs
- T2 (XY8/CPMG): bis zu 2 ms

Rauschmodell:
- Langsame Magnetfeldschwankungen: 1/f-Rauschen
- Kernspinbad: Gaußsches Rauschen
```

### 3.4 ODMR-Linienform

```
Für jede Resonanz:
PL(f) = PL_0 * (1 - C * L(f, f_0, Γ))

Dabei ist:
- PL_0: Photolumineszenz ohne Mikrowell
- C: Kontrast (typisch 5-20%)
- L(f, f_0, Γ): Lorentz-Funktion
- f_0: Resonanzfrequenz
- Γ: Linienbreite (FWHM, typisch 5-15 MHz)
```

## 4. Simulator-Implementierungsdetails

### 4.1 Zustandsrepräsentation

Der Simulator sollte folgende Zustände verwalten:

```python
class NVSimulatorState:
    """Interner Zustand des NV-Simulators"""
    
    # Physikalische Parameter
    magnetic_field: numpy.ndarray  # [B_x, B_y, B_z] in Tesla
    temperature: float             # Temperatur in Kelvin
    strain: float                  # Strain-Parameter E in Hz
    
    # Quantenzustand
    electron_spin_state: numpy.ndarray  # Dichtematrix des Elektronenspins
    nuclear_spin_state: numpy.ndarray   # Dichtematrix des Kernspins
    optical_state: str                  # 'ground', 'excited', 'singlet'
    
    # Gerätezustände
    laser_power: float             # Aktuelle Laserleistung in W
    laser_on: bool                 # Laserstatus
    mw_frequency: float           # Aktuelle Mikrowellenfrequenz in Hz
    mw_power: float               # Aktuelle Mikrowellenleistung in dBm
    mw_on: bool                    # Mikrowellenstatus
    scanner_position: Dict[str, float]  # Aktuelle Scannerposition
    
    # Zeitliche Entwicklung
    coherence_time_T1: float       # T1-Zeit in s
    coherence_time_T2_star: float  # T2*-Zeit in s
    coherence_time_T2: float       # T2-Zeit in s
    last_update_time: float        # Letzte Aktualisierungszeit
```

### 4.2 Konfigurationsdatei-Format

```yaml
nv_simulator:
  # Physikalische Parameter
  physical_model:
    d_constant: 2.87e9  # Zero-field splitting in Hz
    e_strain: 5.0e6     # Strain parameter in Hz
    hyperfine_coupling: 2.2e6  # Hyperfine coupling in Hz
    nitrogen_isotope: "N14"    # N14 or N15
    
  # Kohärenzeigenschaften
  coherence:
    t1_time: 2.0e-3     # T1 time in seconds
    t2_star_time: 3.0e-6  # T2* time in seconds
    t2_time: 300.0e-6   # T2 time in seconds
    
  # Optische Eigenschaften
  optical:
    contrast: 0.15      # ODMR contrast 0-1
    base_counts: 250000  # Base count rate in counts/s
    saturation_power: 2.0e-3  # Saturation power in W
    
  # Umgebungsparameter
  environment:
    temperature: 295.0  # Temperature in K
    base_magnetic_field: [0.0, 0.0, 0.0]  # [Bx, By, Bz] in Tesla
    field_noise: 1.0e-7  # Magnetic noise amplitude in Tesla
    
  # Scanner-Konfiguration
  scanner:
    nv_positions:
      - position: [2.5e-6, 2.5e-6, 5.0e-6]  # [x, y, z] in meters
        contrast: 0.18
        t2_star_time: 2.8e-6
      - position: [7.5e-6, 5.0e-6, 5.0e-6]
        contrast: 0.15
        t2_star_time: 3.5e-6
```

### 4.3 Kommunikationsschnittstellen

```
- Hardwaremodul-Registration über Qudi Modulmanager
- Jedes Simulatormodul implementiert die entsprechende Hardware-Schnittstelle
- Module teilen sich einen gemeinsamen Quantenzustandsspeicher
- Statusvariablen-Updates bei jedem Methodenaufruf
- Realistische Zeitverzögerungen durch Timeouts
```

### 4.4 Logging

Der Simulator sollte detaillierte Logs für die Analyse und Fehlersuche bereitstellen:

```python
# Beispiel für Logging
def set_cw(self, frequency: float, power: float) -> None:
    """Implementierung für microwave_interface.set_cw"""
    self.log.info(f"Setting CW with frequency {frequency:.6e} Hz and power {power} dBm")
    
    # Parametrisierte Verzögerung für realistische Hardware-Kommunikationszeit
    time.sleep(self._get_communication_delay())
    
    # Parameter innerhalb der Hardware-Grenzen halten
    if self._check_frequency_in_range(frequency) and self._check_power_in_range(power):
        self._internal_state.mw_frequency = frequency
        self._internal_state.mw_power = power
        self.log.debug("CW parameters set successfully")
    else:
        error_msg = f"Parameters out of range: freq={frequency}, power={power}"
        self.log.error(error_msg)
        raise ValueError(error_msg)
```

## 5. Integrationstests

### 5.1 ODMR-Testsequenz

```python
# Testet die Integration zwischen Laser, Mikrowell und Fast Counter
def test_odmr_sequence():
    # 1. Initialisierung der Module
    laser = SimpleLaserDummy()
    microwave = MicrowaveDummy()
    counter = FastCounterDummy()
    
    # 2. Konfiguration
    laser.set_power(0.001)  # 1 mW
    freq_start = 2.85e9
    freq_stop = 2.89e9
    freq_step = 1e6
    frequencies = np.arange(freq_start, freq_stop, freq_step)
    
    # 3. Messung
    counts = []
    laser.on()
    for freq in frequencies:
        microwave.set_cw(freq, 0.0)
        microwave.cw_on()
        counter.start_measure()
        time.sleep(0.1)  # Integrationszeit
        data = counter.get_data_trace()
        counts.append(np.mean(data))
        counter.stop_measure()
        microwave.off()
    laser.off()
    
    # 4. Prüfen des Ergebnisses
    # Die Zählraten sollten eine Dip-Struktur um 2.87 GHz aufweisen
    central_idx = np.abs(frequencies - 2.87e9).argmin()
    assert counts[central_idx] < np.mean(counts)
```

### 5.2 Rabi-Testsequenz

```python
# Testet die Integration zwischen Pulser und Fast Counter
def test_rabi_sequence():
    # 1. Initialisierung der Module
    pulser = PulserDummy()
    counter = FastCounterDummy()
    
    # 2. Konfiguration
    pulser.set_sample_rate(1e9)  # 1 GSample/s
    pulse_durations = np.linspace(10e-9, 500e-9, 20)  # 10-500 ns
    
    # 3. Messung
    counts = []
    for duration in pulse_durations:
        # Pulssequenz erstellen
        samples = pulser.get_waveform_template(duration)
        pulser.write_waveform("rabi", samples, {}, True, True, len(samples))
        pulser.load_waveform({"a_ch1": "rabi"})
        
        # Messung durchführen
        pulser.pulser_on()
        counter.start_measure()
        time.sleep(1.0)  # Messzeit
        data = counter.get_data_trace()
        counts.append(np.sum(data))
        counter.stop_measure()
        pulser.pulser_off()
    
    # 4. Prüfen des Ergebnisses
    # Die Zählraten sollten eine kosinusförmige Oszillation zeigen
    from scipy.optimize import curve_fit
    
    def cos_model(x, amp, freq, phase, offset):
        return amp * np.cos(2*np.pi*freq*x + phase) + offset
    
    params, _ = curve_fit(cos_model, pulse_durations, counts)
    assert 20e6 < params[1] < 40e6  # Rabi-Frequenz sollte ~30 MHz sein
```

### 5.3 Konfokaler Scan-Test

```python
# Testet die Integration zwischen Scanner, Laser und Fast Counter
def test_confocal_scan():
    # 1. Initialisierung der Module
    scanner = ScanningProbeDummy()
    laser = SimpleLaserDummy()
    counter = FastCounterDummy()
    
    # 2. Konfiguration
    laser.set_power(0.001)  # 1 mW
    
    settings = ScanSettings(
        channels=('APD',),
        axes=('x', 'y'),
        range=((0, 10e-6), (0, 10e-6)),  # 10 µm x 10 µm
        resolution=(21, 21),             # 21x21 Pixel
        frequency=10.0,                  # 10 Hz Pixelrate
        position_feedback_axes=()
    )
    
    scanner.configure_scan(settings)
    
    # 3. Messung
    laser.on()
    scanner.start_scan()
    
    # Simuliere den Scan manuell für den Test
    while scanner.is_scanning:
        time.sleep(0.1)
    
    # Scan-Daten abrufen
    scan_data = scanner.get_scan_data()
    laser.off()
    
    # 4. Prüfen des Ergebnisses
    # Das Bild sollte NV-Zentren als helle Punkte zeigen
    image = scan_data.data['APD']
    max_positions = np.where(image > 0.8 * np.max(image))
    
    # Sollte mindestens einen hellen Punkt geben
    assert len(max_positions[0]) > 0
```

## 6. Erweiterungsmöglichkeiten

### 6.1 Mehrere NV-Zentren

```python
def add_nv_center(position, properties):
    """Fügt dem Simulator ein weiteres NV-Zentrum hinzu
    
    @param position: (x,y,z)-Position in Metern
    @param properties: Eigenschaften des NV-Zentrums (Orientierung, Kohärenzzeiten etc.)
    """
    self._nv_centers.append({
        'position': np.array(position),
        'properties': properties,
        'state': self._create_initial_state()
    })
```

### 6.2 Magnetfeldgradienten

```python
def set_magnetic_field_gradient(self, gradient_tensor):
    """Setzt einen Magnetfeldgradienten
    
    @param gradient_tensor: 3x3 Tensor für den Gradienten dB_i/dx_j in T/m
    """
    self._field_gradient = np.array(gradient_tensor)
    
    # Aktualisiere das Feld an jeder NV-Position
    for nv in self._nv_centers:
        position = nv['position']
        base_field = self._base_magnetic_field
        local_field = base_field + np.dot(self._field_gradient, position)
        nv['local_field'] = local_field
```

### 6.3 Dynamische Simulationen

```python
def start_continuous_simulation(self, timestep=1e-9):
    """Startet eine kontinuierliche zeitliche Simulation
    
    @param timestep: Zeitschritt in Sekunden
    """
    if self._simulation_running:
        self.log.warn("Simulation bereits aktiv")
        return
        
    self._simulation_running = True
    self._simulation_thread = threading.Thread(
        target=self._simulation_loop,
        args=(timestep,)
    )
    self._simulation_thread.daemon = True
    self._simulation_thread.start()
    
def _simulation_loop(self, timestep):
    """Interne Simulationsschleife"""
    last_time = time.time()
    
    while self._simulation_running:
        current_time = time.time()
        elapsed = current_time - last_time
        
        # Physikalische Zeitentwicklung berechnen
        self._evolve_quantum_state(elapsed)
        
        # Hardware-Status aktualisieren
        self._update_hardware_response()
        
        # Warte bis zum nächsten Zeitschritt
        time.sleep(max(0, timestep - (time.time() - current_time)))
        last_time = current_time
```

### 6.4 Quantum Circuit Interface

```python
def apply_quantum_gate(self, gate_name, parameters=None, target_nv_indices=None):
    """Wendet ein Quantengatter auf ausgewählte NV-Zentren an
    
    @param gate_name: Name des Gatters (X, Y, Z, H, CNOT, etc.)
    @param parameters: Gatter-Parameter (z.B. Rotationswinkel)
    @param target_nv_indices: Indizes der Ziel-NVs
    """
    if target_nv_indices is None:
        target_nv_indices = range(len(self._nv_centers))
        
    gate_matrix = self._get_gate_matrix(gate_name, parameters)
    
    for idx in target_nv_indices:
        if 0 <= idx < len(self._nv_centers):
            nv = self._nv_centers[idx]
            # Wende das Gatter auf den Quantenzustand an
            if gate_name in ['X', 'Y', 'Z', 'H']:
                # Einzel-Qubit-Gatter
                nv['state'] = self._apply_single_qubit_gate(gate_matrix, nv['state'])
            elif gate_name == 'CNOT':
                # Zwei-Qubit-Gatter (erfordert zwei Indizes)
                if len(target_nv_indices) >= 2:
                    control_idx = target_nv_indices[0]
                    target_idx = target_nv_indices[1]
                    self._apply_two_qubit_gate(gate_name, self._nv_centers[control_idx]['state'],
                                              self._nv_centers[target_idx]['state'])
```

Dieses vollständige Anforderungsdokument gibt Ihnen alle notwendigen technischen Details, um einen funktionierenden NV-Zentren-Simulator zu implementieren, der mit der Qudi-Software kommunizieren kann. Es deckt alle Schnittstellen, Datenstrukturen, Kommunikationsprotokolle und das physikalische Modell vollständig ab.