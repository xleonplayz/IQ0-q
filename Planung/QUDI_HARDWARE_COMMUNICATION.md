# Qudi-Hardware Kommunikationsarchitektur

Dieses Dokument beschreibt, wie ein Quantencomputer oder andere Quantenhardware mit dem Qudi-Framework kommuniziert.

## Grundlegende Architektur

Qudi verwendet ein modulares, interface-basiertes System für die Kommunikation mit Hardware:

```
┌────────────┐      ┌────────────┐     ┌──────────────────┐
│  GUI       │─────▶│  Logic     │────▶│ Hardware-Module  │──────▶ Physische Hardware
└────────────┘      └────────────┘     └──────────────────┘
```

Die Hardwarekommunikation erfolgt in drei Schichten:

1. **Interface-Ebene**: Abstrakte Basisklassen definieren Hardware-APIs
2. **Hardware-Implementierungsebene**: Gerätespezifische Klassen, die Interfaces implementieren
3. **Connector-System**: Verbindet Logik-Module mit den Hardware-Modulen

## Hardware-Modul Implementierung

Jedes Hardware-Modul in Qudi muss:

1. Ein entsprechendes Interface implementieren (z.B. `MicrowaveInterface`)
2. Von `qudi.core.module.Base` erben
3. `on_activate()` und `on_deactivate()` Methoden definieren
4. Thread-sichere Kommunikation durch Mutex-Locks gewährleisten

### Beispiel: Grundgerüst eines Hardware-Moduls

```python
from qudi.core.module import Base
from qudi.interface.microwave_interface import MicrowaveInterface
from qudi.core.statusvariable import StatusVar
from qudi.util.mutex import Mutex

class MyQuantumProcessorInterface(MicrowaveInterface):
    """
    Hardware-Modul zur Kommunikation mit meinem Quantenprozessor.
    """
    # Status-Variablen, die beim Neustart des Programms erhalten bleiben
    _cw_frequency = StatusVar('cw_frequency', 2.87e9)
    _cw_power = StatusVar('cw_power', -20)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread_lock = Mutex()
        self._device = None
        # Weitere Initialisierungen...
    
    def on_activate(self):
        """Wird aufgerufen, wenn das Modul aktiviert wird."""
        # Verbindung zur Hardware herstellen
        # z.B. über VISA, Sockets, serielle Schnittstelle etc.
        self._connect_device()
        
    def on_deactivate(self):
        """Wird aufgerufen, wenn das Modul deaktiviert wird."""
        # Hardware-Verbindung trennen, Ressourcen freigeben
        self._disconnect_device()
        
    # Interface-Methoden implementieren
    @property
    def constraints(self):
        """Gibt Hardware-Beschränkungen zurück."""
        return self._constraints
        
    def set_cw(self, frequency, power):
        """Setzt CW-Parameter für den Quantenprozessor."""
        with self._thread_lock:
            # Hardware-Kommando senden
            self._device.send_command(f"SET:FREQ {frequency}")
            self._device.send_command(f"SET:POW {power}")
            # Status aktualisieren
            self._cw_frequency = frequency
            self._cw_power = power
```

## Kommunikationsprotokolle

Qudi unterstützt verschiedene Kommunikationswege für Quantenhardware:

1. **VISA/GPIB**: Standard für wissenschaftliche Instrumente
   ```python
   import visa
   self._rm = visa.ResourceManager()
   self._device = self._rm.open_resource(self._visa_address)
   ```

2. **TCP/IP**: Für netzwerkfähige Quantencomputer
   ```python
   import socket
   self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
   self._socket.connect((self._ip_address, self._port))
   ```

3. **REST API**: Für cloud-basierte Quantencomputer
   ```python
   import requests
   response = requests.post(
       f"{self._api_endpoint}/execute_circuit",
       headers={"Authorization": f"Bearer {self._api_key}"},
       json={"circuit": circuit_data}
   )
   ```

4. **Herstellerspezifische SDKs**: Integration über Python-Bibliotheken
   ```python
   import qiskit
   # Oder andere Quantum SDK-Bibliotheken wie cirq, pyquil, etc.
   ```

## Registrierung in der Konfiguration

Damit Qudi ein Hardware-Modul laden kann, muss es in der Konfigurationsdatei registriert werden:

```yaml
hardware:
  quantum_processor:
    module.Class: 'quantum.my_quantum_processor.MyQuantumProcessorInterface'
    ip_address: '192.168.1.100'
    port: 5000
    api_key: 'your_secret_key'
```

## Datenfluss und Zustandsverwaltung

Der Datenfluss zwischen Qudi und der Quantenhardware folgt diesem Muster:

1. **Konfiguration**: Parameter werden an die Hardware gesendet
2. **Ausführung**: Befehle zur Ausführung von Operationen werden gesendet
3. **Statusabfrage**: Status und Fortschritt werden abgefragt
4. **Datenerfassung**: Messergebnisse werden zurückgelesen

Beispiel für einen typischen Kommunikationsablauf:

```python
def execute_quantum_circuit(self, circuit):
    """Führt einen Quantenschaltkreis aus und gibt die Messergebnisse zurück."""
    with self._thread_lock:
        # 1. Konfiguration
        self._device.send_command("RESET")
        
        # 2. Schaltkreis senden und ausführen
        circuit_data = self._serialize_circuit(circuit)
        self._device.send_command(f"LOAD:CIRCUIT {circuit_data}")
        self._device.send_command("RUN")
        
        # 3. Auf Abschluss warten (Polling)
        while True:
            status = self._device.query("STATUS?")
            if status == "COMPLETE":
                break
            time.sleep(0.1)
        
        # 4. Ergebnisse lesen
        results = self._device.query("RESULTS?")
        return self._parse_results(results)
```

## Status-Aktualisierung und Rückmeldung

Statusänderungen werden in Qudi durch mehrere Mechanismen behandelt:

1. **StatusVariable**: Speichert persistente Zustände
2. **SignalInterface**: Aktualisiert die GUI über Hardware-Änderungen
3. **Logging**: Erfasst Statusänderungen, Fehler und Debugging-Informationen

```python
# Status-Signale
class MyQuantumProcessorInterface(MicrowaveInterface):
    # Signale definieren
    sigStateChanged = QtCore.Signal(str)
    
    def _update_state(self, new_state):
        """Aktualisiert den internen Zustand und informiert andere Module."""
        self._state = new_state
        # Signal emittieren
        self.sigStateChanged.emit(new_state)
        # Loggen
        self.log.info(f"Quantum processor state changed to {new_state}")
```

## Fehlerbehandlung

Qudi-Hardware-Module implementieren robust Fehlerbehandlung:

```python
def _send_command(self, command):
    """Sendet ein Kommando und behandelt häufige Fehler."""
    try:
        self._device.write(command)
    except visa.VisaIOError as e:
        if e.error_code == visa.constants.VI_ERROR_TMO:
            self.log.error("Communication timeout. Check device connection.")
            raise TimeoutError("Device communication timeout")
        else:
            self.log.exception("VISA communication error")
            raise
    except Exception as e:
        self.log.exception("Unexpected error during command sending")
        raise
```

## Zusammenfassung

Die Kommunikation zwischen Qudi und Quantenhardware erfolgt über:

1. **Abstrakte Interfaces**: Definieren einheitliche Schnittstellen
2. **Hardware-spezifische Implementierungen**: Übersetzen Interface-Aufrufe in gerätespezifische Befehle
3. **Verschiedene Kommunikationsprotokolle**: VISA, TCP/IP, REST APIs oder spezialisierte SDKs
4. **Thread-sichere Synchronisation**: Verhindert konkurrierende Hardwarezugriffe
5. **Status- und Fehlerverwaltung**: Robuste Fehlerbehandlung und Statusaktualisierung

Um einen Quantencomputer in Qudi zu integrieren, muss ein entsprechendes Hardware-Modul implementiert werden, das die benötigten Interfaces implementiert und mit der Quantenhardware über deren bevorzugtes Kommunikationsprotokoll interagiert.