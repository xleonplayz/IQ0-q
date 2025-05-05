# Systemarchitektur des NV-Zentren Simulators

## 1. Architekturübersicht

Der NV-Zentren Simulator ist nach einer mehrschichtigen Architektur konzipiert, die eine klare Trennung von Verantwortlichkeiten ermöglicht und gleichzeitig die nahtlose Integration mit dem Qudi-Framework gewährleistet.

### 1.1 Architekturprinzipien

- **Schichtenbasierter Aufbau**: Klare Trennung zwischen physikalischer Simulation, Zustandsverwaltung, Interface-Implementierungen und Netzwerkkommunikation
- **Modulare Struktur**: Unabhängige Module mit klar definierten Schnittstellen
- **Singleton-basierte Zustandsverwaltung**: Einheitliche Verwaltung des Quantensystemzustands
- **Threadsicherheit**: Vollständige Threadsicherheit durch konsequenten Einsatz von Mutex-Locks
- **Erweiterbarkeit**: Einfache Möglichkeit zur Erweiterung um neue Funktionen und Schnittstellen

### 1.2 Systemschichten

```
┌───────────────────────────────────────────────────────┐
│ Qudi Framework                                         │
└───────────────▲───────────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────────┐
│ Netzwerkschicht (Network Manager)                     │
└───────────────▲───────────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────────┐
│ Interface-Schicht (Qudi Hardware Interface Impl.)     │
└───────────────▲───────────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────────┐
│ Zustandsverwaltung (Quantum Simulator State)          │
└───────────────▲───────────────────────────────────────┘
                │
┌───────────────▼───────────────────────────────────────┐
│ Physikalische Modellierung (Physical NV Model)        │
└───────────────────────────────────────────────────────┘
```

## 2. Kernkomponenten

### 2.1 PhysicalNVModel

Das physikalische Modell bildet das Herzstück des Simulators und ist für die realistische Simulation der quantenmechanischen Eigenschaften von NV-Zentren verantwortlich.

#### Hauptverantwortlichkeiten:

- Simulation des NV-Zentrum Hamiltonians unter externen Feldern
- Berechnung von Energieniveaus und Übergängen
- Simulation von ODMR-Spektren
- Simulation von Rabi-Oszillationen
- Berechnung von Photonenzählraten
- Simulation von Dekohärenz und Relaxation

#### Klassendiagramm:

```
┌───────────────────────────────────────────┐
│ PhysicalNVModel                           │
├───────────────────────────────────────────┤
│ - simulator: simos.Simulator              │
│ - d_constant: float                       │
│ - e_strain: float                         │
│ - hyperfine: float                        │
│ - nitrogen_isotope: str                   │
│ - t1_time: float                          │
│ - t2_star_time: float                     │
│ - t2_time: float                          │
│ - contrast: float                         │
│ - base_counts: float                      │
│ - saturation_power: float                 │
│ - magnetic_field: np.ndarray              │
│ - laser_power: float                      │
│ - microwave_frequency: float              │
│ - microwave_power: float                  │
│ - electron_state: np.ndarray              │
│ - nuclear_state: np.ndarray               │
│ - optical_state: str                      │
│ - last_update_time: float                 │
├───────────────────────────────────────────┤
│ + __init__(config: Dict)                  │
│ + _initialize_quantum_state(): void       │
│ + apply_magnetic_field(field): void       │
│ + set_laser_power(power): void            │
│ + apply_microwave(frequency, power): void │
│ + stop_microwave(): void                  │
│ + calculate_odmr_spectrum(...): np.ndarray│
│ + evolve_under_pulse(...): void           │
│ + get_photon_count_rate(): float          │
│ + evolve_quantum_state(time): void        │
└───────────────────────────────────────────┘
```

#### SimOS-Integration:

Die Klasse nutzt SimOS als Kernmodul für die physikalische Simulation:

```python
import simos as sos
from simos.systems.NV import NVSystem

class PhysicalNVModel:
    def __init__(self, config: Dict):
        # SimOS Integration
        self.simulator = sos.Simulator()
        
        # NV-System initialisieren
        self.nv_system = NVSystem(
            optics=True,
            orbital=config.get('temperature', 298) < 200,
            nitrogen=True,
            natural=config.get('nitrogen_isotope', 'N14') == 'N14'
        )
        
        # Physikalische Parameter
        self.d_constant = config.get('d_constant', 2.87e9)  # Zero-field splitting in Hz
        self.e_strain = config.get('e_strain', 5.0e6)       # Strain parameter in Hz
        # ... weitere Parameter ...
```

### 2.2 QuantumSimulatorState

Die Zustandsverwaltung ist als Singleton implementiert und sorgt für einen konsistenten Zustand über alle Komponenten hinweg.

#### Hauptverantwortlichkeiten:

- Zentrale Speicherung des Quantensystemzustands
- Koordination zwischen verschiedenen Hardware-Interfaces
- Threadsichere Zustandsmodifikationen
- Zustandsperistenz (Speichern/Laden)

#### Klassendiagramm:

```
┌───────────────────────────────────────────┐
│ QuantumSimulatorState                     │
├───────────────────────────────────────────┤
│ - _instance: QuantumSimulatorState        │
│ - nv_models: Dict                         │
│ - laser_state: bool                       │
│ - laser_power: float                      │
│ - laser_wavelength: float                 │
│ - laser_shutter_state: bool               │
│ - microwave_state: bool                   │
│ - microwave_frequency: float              │
│ - microwave_power: float                  │
│ - microwave_scan_params: Dict             │
│ - scanner_position: Dict                  │
│ - scanner_target: Dict                    │
│ - scanner_settings: ScanSettings          │
│ - counter_params: Dict                    │
│ - counter_is_running: bool                │
│ - pulser_is_on: bool                      │
│ - pulser_sample_rate: float               │
│ - pulser_loaded_assets: Dict              │
│ - switch_states: Dict                     │
│ - _lock: threading.RLock                  │
├───────────────────────────────────────────┤
│ + __new__(cls): QuantumSimulatorState     │
│ + _initialize(): void                     │
│ + add_nv_center(position, properties): void│
│ + get_nv_at_position(position): PhysicalNV│
│ + update_laser_state(is_on, power): void  │
│ + update_microwave_state(...): void       │
│ + update_scanner_position(position): void │
│ + save_state(filepath): void              │
│ + load_state(filepath): void              │
└───────────────────────────────────────────┘
```

#### Verwendungsbeispiel:

```python
class QuantumSimulatorState:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(QuantumSimulatorState, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
        
    def _initialize(self):
        """Initialisiert den Zustand des Simulators"""
        self.nv_models = {}  # Dictionary von NV-Modellen an verschiedenen Positionen
        
        # Hardware-Zustandsvariablen
        self.laser_state = False
        self.laser_power = 0.0
        # ... weitere Zustandsvariablen ...
        
        # Thread-Sicherheit
        self._lock = threading.RLock()
    
    def update_laser_state(self, is_on, power=None):
        """Aktualisiert den Laserzustand"""
        with self._lock:
            self.laser_state = is_on
            if power is not None:
                self.laser_power = power
                
            # Aktualisiere alle NV-Modelle mit dem neuen Laserzustand
            for nv_model in self.nv_models.values():
                nv_model.set_laser_power(self.laser_power if is_on else 0.0)
```

### 2.3 Interface-Implementierungen

Die Interface-Schicht besteht aus einer Reihe von Klassen, die die Qudi-Hardware-Interfaces implementieren und intern mit dem PhysicalNVModel und QuantumSimulatorState kommunizieren.

#### Hauptkomponenten:

1. **MicrowaveSimulator**: Implementiert das MicrowaveInterface
2. **FastCounterSimulator**: Implementiert das FastCounterInterface
3. **PulserSimulator**: Implementiert das PulserInterface
4. **ScanningProbeSimulator**: Implementiert das ScanningProbeInterface
5. **SimpleLaserSimulator**: Implementiert das SimpleLaserInterface
6. **SwitchSimulator**: Implementiert das SwitchInterface

#### Beispielklassendiagramm für MicrowaveSimulator:

```
┌────────────────────────────────────────────┐
│ MicrowaveSimulator                         │
├────────────────────────────────────────────┤
│ - _thread_lock: Mutex                      │
│ - _simulator_state: QuantumSimulatorState  │
│ - _constraints: MicrowaveConstraints       │
│ - _cw_frequency: float                     │
│ - _cw_power: float                         │
│ - _scan_frequencies: np.ndarray            │
│ - _scan_power: float                       │
│ - _scan_mode: SamplingOutputMode           │
│ - _scan_sample_rate: float                 │
│ - _is_scanning: bool                       │
├────────────────────────────────────────────┤
│ + constraints: MicrowaveConstraints        │
│ + is_scanning: bool                        │
│ + cw_power: float                          │
│ + cw_frequency: float                      │
│ + scan_power: float                        │
│ + scan_frequencies: Union[np.ndarray, Tuple│
│ + scan_mode: SamplingOutputMode            │
│ + scan_sample_rate: float                  │
│ + off(): void                              │
│ + set_cw(frequency, power): void           │
│ + cw_on(): void                            │
│ + configure_scan(...): void                │
│ + start_scan(): void                       │
│ + reset_scan(): void                       │
│ - _run_scan(): void                        │
└────────────────────────────────────────────┘
```

### 2.4 NetworkManager

Der NetworkManager implementiert verschiedene Netzwerkprotokolle für die Kommunikation mit Qudi.

#### Hauptverantwortlichkeiten:

- Implementierung von TCP/IP-basierten Protokollen
- Optional: ZeroMQ-Unterstützung
- Optional: gRPC-Unterstützung
- Verbindungsmanagement
- Fehlerbehandlung und Wiederherstellung

#### Klassendiagramm:

```
┌────────────────────────────────────────────┐
│ NetworkManager                             │
├────────────────────────────────────────────┤
│ - config: Dict                             │
│ - servers: Dict                            │
│ - simulator_state: QuantumSimulatorState   │
├────────────────────────────────────────────┤
│ + __init__(config): void                   │
│ + _start_tcp_server(): void                │
│ + _start_zmq_server(): void                │
│ + _start_grpc_server(): void               │
│ + shutdown(): void                         │
└────────────────────────────────────────────┘

┌────────────────────────────────────────────┐
│ TcpServer                                  │
├────────────────────────────────────────────┤
│ - port: int                                │
│ - simulator_state: QuantumSimulatorState   │
│ - server_socket: socket.socket             │
│ - running: bool                            │
│ - thread: threading.Thread                 │
├────────────────────────────────────────────┤
│ + start(): void                            │
│ + _run_server(): void                      │
│ + _handle_client(socket, addr): void       │
│ + _process_command(command): Dict          │
│ + stop(): void                             │
└────────────────────────────────────────────┘
```

### 2.5 WebUIServer

Der WebUIServer bietet eine benutzerfreundliche Weboberfläche zur Konfiguration und Überwachung des Simulators.

#### Hauptverantwortlichkeiten:

- Webserver für die Konfigurationsoberfläche
- REST-API für Simulator-Steuerung
- Frontend für Benutzerinteraktion
- Statusvisualisierung

#### Klassendiagramm:

```
┌────────────────────────────────────────────┐
│ WebUIServer                                │
├────────────────────────────────────────────┤
│ - config: Dict                             │
│ - simulator_state: QuantumSimulatorState   │
│ - host: str                                │
│ - port: int                                │
│ - app: Flask                               │
│ - server: threading.Thread                 │
├────────────────────────────────────────────┤
│ + start(): void                            │
│ + _register_routes(): void                 │
│ + _run_server(): void                      │
│ + _get_state_data(): Dict                  │
│ + _update_config(new_config): void         │
│ + stop(): void                             │
└────────────────────────────────────────────┘
```

## 3. Datenflussbeschreibungen

### 3.1 ODMR-Messung

```
┌──────────────┐      ┌───────────────┐      ┌─────────────────┐
│              │      │               │      │                 │
│ Qudi Logic   │─────►│ Microwave     │─────►│ Quantum         │
│ Module       │◄─────│ Simulator     │◄─────│ Simulator State │
│              │      │               │      │                 │
└──────┬───────┘      └───────────────┘      └────────┬────────┘
       │                                              │
       │                                              │
       │            ┌───────────────┐                 │
       │            │               │                 │
       └───────────►│ FastCounter   │◄────────────────┘
                    │ Simulator     │
                    │               │
                    └───────┬───────┘
                            │
                            │
                            │
                            ▼
                    ┌───────────────┐
                    │               │
                    │ Physical NV   │
                    │ Model         │
                    │               │
                    └───────────────┘
```

Prozessablauf:
1. **Qudi Logic Module** sendet Befehle an **MicrowaveSimulator**:
   - `set_cw(frequency, power)` - Setzt Mikrowellenfrequenz
   - `cw_on()` - Aktiviert Mikrowellenausgang

2. **MicrowaveSimulator** aktualisiert den **QuantumSimulatorState**:
   - `update_microwave_state(True, frequency, power)`

3. **QuantumSimulatorState** aktualisiert das **PhysicalNVModel**:
   - `apply_microwave(frequency, power)`

4. **Qudi Logic Module** aktiviert den **FastCounterSimulator**:
   - `start_measure()` - Startet Photonenzählung

5. **FastCounterSimulator** fragt den **QuantumSimulatorState** ab:
   - `get_nv_at_position(scanner_position)`

6. **QuantumSimulatorState** liefert das aktive **PhysicalNVModel**

7. **FastCounterSimulator** fragt das **PhysicalNVModel** ab:
   - `get_photon_count_rate()` - Erhält Zählrate basierend auf aktuellem Zustand

8. **Qudi Logic Module** liest Daten vom **FastCounterSimulator**:
   - `get_data_trace()` - Erhält Photonenzähldaten

### 3.2 Rabi-Oszillations-Messung

```
┌──────────────┐      ┌───────────────┐      ┌─────────────────┐
│              │      │               │      │                 │
│ Qudi Logic   │─────►│ Pulser        │─────►│ Quantum         │
│ Module       │◄─────│ Simulator     │◄─────│ Simulator State │
│              │      │               │      │                 │
└──────┬───────┘      └───────────────┘      └────────┬────────┘
       │                                              │
       │                                              │
       │            ┌───────────────┐                 │
       │            │               │                 │
       └───────────►│ FastCounter   │◄────────────────┘
                    │ Simulator     │
                    │               │
                    └───────┬───────┘
                            │
                            │
                            │
                            ▼
                    ┌───────────────┐
                    │               │
                    │ Physical NV   │
                    │ Model         │
                    │               │
                    └───────────────┘
```

Prozessablauf:
1. **Qudi Logic Module** konfiguriert den **PulserSimulator**:
   - `write_waveform()` - Definiert die Pulssequenz (Initialisierung, MW-Puls, Auslese)
   - `load_waveform()` - Lädt die Pulssequenz

2. **PulserSimulator** aktualisiert den **QuantumSimulatorState**:
   - Speichert Pulsinformationen im Zustand

3. **Qudi Logic Module** aktiviert den **PulserSimulator**:
   - `pulser_on()` - Startet die Pulssequenz

4. **PulserSimulator** führt die Pulssequenz auf dem **PhysicalNVModel** aus:
   - `evolve_under_pulse(duration, frequency, power)` - Für Mikrowellenpuls
   - `set_laser_power(power)` - Für Laser-Initialisierung und -Auslese

5. **FastCounterSimulator** sammelt Photonen während des Auslesefensters:
   - `get_photon_count_rate()` - Abfrage des aktuellen PhysicalNVModel

6. **Qudi Logic Module** liest die Ergebnisse:
   - `get_data_trace()` - Erhält die Zähldaten vom FastCounterSimulator

## 4. Thread-Sicherheit und Synchronisation

### 4.1 Mutex-Strategie

Alle Zustandsänderungen werden durch Mutex-Locks geschützt, um Race-Conditions zu vermeiden:

```python
def update_laser_state(self, is_on, power=None):
    """Aktualisiert den Laserzustand"""
    with self._thread_lock:
        self.laser_state = is_on
        if power is not None:
            self.laser_power = power
        # Weitere Operationen...
```

### 4.2 Kommunikationsmuster

- **Thread-safe Singleton**: Das QuantumSimulatorState-Objekt ist thread-safe implementiert
- **Read-Copy-Update (RCU)**: Bei Datenabfragen werden Kopien zurückgegeben, nicht direkte Referenzen
- **Atomar-Operationen**: Zustandsänderungen werden atomar durchgeführt

## 5. Fehlerbehandlungsstrategie

### 5.1 Fehlertypen

- **Netzwerkfehler**: Verbindungsabbrüche, Timeouts
- **Parameterfehler**: Ungültige Parameter für Hardware-Operationen
- **Zustandsfehler**: Ungültige Zustandsübergänge (z.B. Start eines Scans während bereits ein Scan läuft)
- **Simulationsfehler**: Probleme in der physikalischen Simulation

### 5.2 Fehlerbehandlung

```python
def set_cw(self, frequency, power):
    """Konfiguriert den CW-Mikrowellenausgang."""
    with self._thread_lock:
        if self.module_state() != 'idle':
            raise RuntimeError('Unable to set CW: Microwave is active.')
            
        # Parameter-Prüfung
        if not (self._constraints.frequency_limits[0] <= frequency <= self._constraints.frequency_limits[1]):
            raise ValueError(f"Frequency {frequency} out of bounds ({self._constraints.frequency_limits})")
            
        # ... weitere Validierungen ...
        
        try:
            # Update durchführen
            # ...
        except Exception as e:
            self.log.error(f"Error setting CW parameters: {str(e)}")
            # Zustand wiederherstellen wenn nötig
            raise
```

## 6. Konfigurationsmanagement

### 6.1 Konfigurationsstruktur

```yaml
simulator:
  # Grundlegende Einstellungen
  nv_count: 1
  temperature: 298.0
  
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
    
  # Netzwerkeinstellungen
  network:
    enable_tcp_server: true
    tcp_port: 5555
    enable_zmq: false
    zmq_pub_port: 5556
    zmq_sub_port: 5557
    enable_grpc: false
    grpc_port: 50051
    
  # Web-UI-Einstellungen
  webui:
    enabled: true
    host: "0.0.0.0"
    port: 8080
    
  # Timing und Delays
  timing:
    realistic_delays: true
    microwave_delay: 50.0e-3  # 50ms
    laser_delay: 10.0e-3      # 10ms
    counter_delay: 5.0e-3     # 5ms
```

### 6.2 Konfigurationsverwaltung

- Konfigurationen werden bei Programmstart eingelesen
- Webinterface ermöglicht dynamische Änderungen
- Konfigurationsänderungen werden validiert
- Konfigurationen können als Profile gespeichert werden

## 7. Netzwerkimplementierung

### 7.1 TCP/IP-Protokollspezifikation

**Allgemeines Format**:
```
<COMMAND>:<PARAMETER>\n
```

**Beispiele**:
```
SET:MICROWAVE:FREQUENCY:2.87E9\n
SET:MICROWAVE:POWER:-20\n
MICROWAVE:ON\n
```

**Antwortformat**:
```
OK\n
ERROR:Fehlermeldung\n
DATA:JSON-Daten\n
```

### 7.2 ZeroMQ-Protokollspezifikation

**PUB/SUB-Muster**:
- Publisher (Simulator) sendet Updates an Subscriber (Qudi)
- Daten im MessagePack-Format für Effizienz

**REQ/REP-Muster**:
- Request (Qudi) sendet Befehle an Reply (Simulator)
- JSON-formatierte Nachrichten

## 8. Web-UI Implementierung

### 8.1 Backend-Implementierung

- **Flask-basierter Server**: Implementiert REST-API
- **API-Endpunkte**: Für Konfiguration und Statusabfragen
- **Echtzeit-Updates**: Über WebSockets für Statusinformationen

### 8.2 Frontend-Implementierung

- **HTML/CSS/JavaScript**: Moderne Weboberfläche
- **Responsive Design**: Funktioniert auf verschiedenen Geräten
- **Konfigurations-UI**: Intuitive Benutzeroberfläche für Parameteranpassungen
- **Status-Monitor**: Visualisierung des aktuellen Simulatorzustands

## 9. Erweiterbarkeit

### 9.1 Erweiterungsmechanismen

- **Modulares Design**: Einfaches Hinzufügen neuer Interface-Implementierungen
- **Austauschbare Komponenten**: Kernkomponenten können ausgetauscht werden
- **Plugin-System**: Für zukünftige Erweiterungen

### 9.2 Anpassungsmöglichkeiten

- **Benutzerdefinierte Rauschmodelle**: Eigene Rauschprofile
- **Eigene Quantenmodelle**: Erweiterung der physikalischen Simulation
- **Benutzerdefinierte Hardware-Parameter**: Anpassung an spezifische Hardware

## 10. Zusammenfassung

Die Systemarchitektur des NV-Zentren Simulators bietet:

- **Modularen Aufbau**: Klare Trennung von Verantwortlichkeiten
- **Realistische Simulation**: Basierend auf SimOS für physikalische Genauigkeit
- **Vollständige Qudi-Integration**: Implementierung aller relevanten Hardware-Interfaces
- **Netzwerkfähigkeit**: Flexible Kommunikationsprotokolle
- **Konfigurierbarkeit**: Anpassbare Parameter über Web-UI
- **Erweiterbarkeit**: Modulares Design für zukünftige Erweiterungen

Diese Architektur ermöglicht die Entwicklung eines hochrealistischen NV-Zentren Simulators, der nahtlos mit dem Qudi-Framework zusammenarbeitet und für verschiedene Anwendungsfälle konfiguriert werden kann.