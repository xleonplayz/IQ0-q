# Kommunikationsprotokolle für den NV-Zentren Simulator

## 1. Einführung

Dieses Dokument spezifiziert im Detail die Netzwerkprotokolle und Kommunikationsverfahren, die der NV-Zentren Simulator zur Kommunikation mit dem Qudi-Framework und anderen Clients verwendet. Diese Protokolle sind entscheidend für die nahtlose Integration des Simulators in bestehende Experimentumgebungen und ermöglichen die verteilte Ausführung von Quantenexperimenten.

## 2. Protokoll-Übersicht

Der NV-Zentren Simulator unterstützt mehrere Kommunikationsprotokolle, um unterschiedliche Anwendungsfälle und Clients zu bedienen:

1. **TCP/IP Socket-basierte Kommunikation**: Primäres Protokoll für direkte Verbindungen
2. **ZeroMQ-basierte Kommunikation**: Für fortgeschrittene Nachrichtenübermittlungsmuster
3. **Optional: gRPC-basierte Kommunikation**: Für strukturierte Kommunikation mit starker Typisierung

## 3. TCP/IP Socket-Kommunikation

### 3.1 Verbindungsaufbau

TCP/IP ist das Basisprotokoll für die Kommunikation mit dem Simulator. Der Simulator öffnet mehrere Ports für verschiedene Funktionalitäten:

- **Kontrollkanal (Port 5555)**: Für Befehle und Gerätekonfiguration
- **Datenkanal (Port 5556)**: Für die Übertragung von Messdaten und Ergebnissen
- **Statuskanal (Port 5557)**: Für Statusinformationen und Benachrichtigungen

```python
def start_tcp_server(self):
    """Startet den TCP/IP-Server für die Simulator-Kommunikation."""
    # Kontrollkanal
    self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.control_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self.control_socket.bind(('0.0.0.0', self.control_port))
    self.control_socket.listen(5)
    
    # Datenkanal
    self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.data_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self.data_socket.bind(('0.0.0.0', self.data_port))
    self.data_socket.listen(5)
    
    # Statuskanal
    self.status_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.status_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self.status_socket.bind(('0.0.0.0', self.status_port))
    self.status_socket.listen(5)
    
    # Starte Listener-Threads für jeden Kanal
    threading.Thread(target=self._handle_control_connections, daemon=True).start()
    threading.Thread(target=self._handle_data_connections, daemon=True).start()
    threading.Thread(target=self._handle_status_connections, daemon=True).start()
```

### 3.2 Befehlsprotokoll

Das Befehlsprotokoll über den Kontrollkanal folgt einem einfachen textbasierten Format:

```
<COMMAND>:<PARAMETER1>:<PARAMETER2>:...\n
```

**Beispiele für Befehle:**

```
SET:MICROWAVE:FREQUENCY:2.87E9\n
SET:MICROWAVE:POWER:-20\n
MICROWAVE:ON\n
COUNTER:START\n
LASER:ON\n
GET:SCANNER:POSITION\n
```

**Antwortformat:**

```
OK\n                        # Erfolgreiche Ausführung ohne Rückgabewert
OK:<RETURN_VALUE>\n         # Erfolgreiche Ausführung mit Rückgabewert
ERROR:<ERROR_MESSAGE>\n     # Fehler bei der Ausführung
```

### 3.3 Datenprotokoll

Der Datenkanal wird für die Übertragung größerer Datenmengen wie Messergebnisse verwendet. Das Format ist binär und besteht aus einem Header und den eigentlichen Daten:

**Header (24 Bytes):**
```c
struct DataHeader {
    uint32_t magic;          // Magic number (0x51444154, 'QDAT')
    uint32_t type;           // Datentyp (z.B. 1=ODMR, 2=Rabi, 3=Scan)
    uint32_t format;         // Datenformat (z.B. 1=Float32, 2=Int32)
    uint32_t flags;          // Flags für zusätzliche Optionen
    uint64_t data_size;      // Größe der Nutzdaten in Bytes
    uint32_t checksum;       // CRC32-Prüfsumme der Daten
};
```

**Datenübertragung in Python:**

```python
def send_data(sock, data, data_type, data_format):
    """Sendet Daten im binären Protokoll über den Datenkanal."""
    # Header erstellen
    header = struct.pack('!IIIIQII', 
                         0x51444154,  # Magic
                         data_type,   # Datentyp
                         data_format, # Format
                         0,           # Flags
                         len(data),   # Datengröße
                         zlib.crc32(data))  # Prüfsumme
    
    # Header und Daten senden
    sock.sendall(header)
    sock.sendall(data)
```

**Beispiele für Datentypen:**

| Typ-ID | Beschreibung | Format |
|--------|--------------|--------|
| 1 | ODMR-Spektrum | Array von (Frequenz, Counts)-Paaren |
| 2 | Rabi-Oszillationen | Array von (Zeit, Counts)-Paaren |
| 3 | Konfokales Scan-Bild | 2D-Array mit Zählraten |
| 4 | Zeitaufgelöste Zählung | Array von Zählraten pro Bin |

### 3.4 Statusprotokoll

Der Statuskanal sendet regelmäßig Statusupdates in JSON-Format, um den aktuellen Zustand des Simulators zu melden:

```json
{
    "timestamp": 1635789600.123,
    "state": {
        "running": true,
        "module_states": {
            "microwave": "running",
            "laser": "on",
            "scanner": "idle",
            "counter": "running"
        },
        "hardware": {
            "microwave_frequency": 2.87e9,
            "microwave_power": -20.0,
            "laser_power": 1.0,
            "scanner_position": {"x": 10.0, "y": 15.0, "z": 5.0}
        },
        "measurements": {
            "current_counts": 250000,
            "scan_progress": 45.5
        }
    }
}
```

## 4. ZeroMQ-Kommunikation

ZeroMQ bietet fortgeschrittenere Nachrichtenübermittlungsmuster als reine Sockets und wird für komplexere Kommunikationsanforderungen verwendet.

### 4.1 Nachrichtenmuster

Der Simulator implementiert verschiedene ZeroMQ-Muster:

- **REQ/REP (Request/Reply)**: Für Befehle und Konfiguration (Port 5555)
- **PUB/SUB (Publisher/Subscriber)**: Für Broadcast von Daten und Status (Port 5556)
- **PUSH/PULL**: Für Streamen großer Datenmengen (Port 5557)

```python
import zmq

def setup_zmq(self):
    """Initialisiert ZeroMQ-Sockets für die verschiedenen Kommunikationsmuster."""
    self.zmq_context = zmq.Context()
    
    # REQ/REP für Befehle
    self.zmq_rep_socket = self.zmq_context.socket(zmq.REP)
    self.zmq_rep_socket.bind(f"tcp://*:{self.zmq_command_port}")
    
    # PUB für Statusupdates und Daten
    self.zmq_pub_socket = self.zmq_context.socket(zmq.PUB)
    self.zmq_pub_socket.bind(f"tcp://*:{self.zmq_pub_port}")
    
    # PUSH für Streaming großer Datenmengen
    self.zmq_push_socket = self.zmq_context.socket(zmq.PUSH)
    self.zmq_push_socket.bind(f"tcp://*:{self.zmq_push_port}")
    
    # Starte Worker-Threads für die Nachrichtenverarbeitung
    threading.Thread(target=self._handle_zmq_commands, daemon=True).start()
```

### 4.2 Nachrichtenformate

Die ZeroMQ-Nachrichtenformate sind auf Effizienz optimiert und verwenden MessagePack oder Protocol Buffers für die Serialisierung:

**Befehlsformat (REQ/REP):**

```python
import msgpack

# Client-Seite (Qudi)
def send_command(socket, command, params=None):
    """Sendet einen Befehl über ZeroMQ REQ/REP."""
    msg = {
        'cmd': command,
        'params': params or {},
        'id': str(uuid.uuid4())
    }
    socket.send(msgpack.packb(msg))
    return msgpack.unpackb(socket.recv())

# Server-Seite (Simulator)
def _handle_zmq_commands(self):
    """Verarbeitet Befehle über ZeroMQ REQ/REP."""
    while self.running:
        try:
            message = msgpack.unpackb(self.zmq_rep_socket.recv())
            command = message.get('cmd')
            params = message.get('params', {})
            
            # Befehl ausführen
            result = self._execute_command(command, params)
            
            # Antwort senden
            self.zmq_rep_socket.send(msgpack.packb({
                'success': True,
                'result': result,
                'id': message.get('id')
            }))
        except Exception as e:
            # Fehlerantwort senden
            self.zmq_rep_socket.send(msgpack.packb({
                'success': False,
                'error': str(e),
                'id': message.get('id', '')
            }))
```

**Statusupdates (PUB/SUB):**

```python
def publish_status(self):
    """Veröffentlicht Statusupdates über ZeroMQ PUB/SUB."""
    while self.running:
        status = self._get_current_status()
        # Formatieren als Thema + Nachricht
        self.zmq_pub_socket.send_multipart([
            b'status',  # Thema (Topic)
            msgpack.packb(status)  # Daten
        ])
        time.sleep(0.2)  # Aktualisierungsrate begrenzen
```

**Datenpublikation (PUB/SUB):**

```python
def publish_measurement_data(self, data_type, data):
    """Veröffentlicht Messdaten über ZeroMQ PUB/SUB."""
    # Thema basierend auf Datentyp
    topic = f"data.{data_type}".encode()
    
    # Daten packen und senden
    self.zmq_pub_socket.send_multipart([
        topic,
        msgpack.packb(data)
    ])
```

### 4.3 Optimierte Datenübertragung

Für große Datenmengen (z.B. Konfokale Bilder) wird eine optimierte Streaming-Strategie verwendet:

```python
def stream_large_data(self, data_type, data_generator):
    """Streamt große Datenmengen in Chunks über ZeroMQ PUSH."""
    # Metadaten senden
    metadata = {
        'type': data_type,
        'timestamp': time.time(),
        'chunks': 0  # wird aktualisiert
    }
    
    # Daten in Chunks streamen
    chunk_size = 1024 * 1024  # 1 MB Chunks
    chunks = 0
    
    for data_chunk in data_generator(chunk_size):
        # Chunk mit Header senden
        self.zmq_push_socket.send_multipart([
            b'chunk',
            msgpack.packb({
                'type': data_type,
                'chunk_number': chunks,
                'final': False
            }),
            data_chunk
        ])
        chunks += 1
    
    # Abschluss-Chunk senden
    metadata['chunks'] = chunks
    self.zmq_push_socket.send_multipart([
        b'chunk',
        msgpack.packb({
            'type': data_type,
            'chunk_number': chunks,
            'final': True,
            'metadata': metadata
        }),
        b''
    ])
```

## 5. gRPC-Kommunikation (Optional)

Für strukturierte Kommunikation mit starker Typisierung unterstützt der Simulator optional gRPC.

### 5.1 Service-Definition

Die gRPC-Dienste werden in .proto-Dateien definiert:

```protobuf
syntax = "proto3";

package nv_simulator;

service SimulatorService {
  // Gerätekontrolle
  rpc ConfigureMicrowave(MicrowaveParams) returns (CommandResult);
  rpc SetMicrowaveOutput(MicrowaveOutputState) returns (CommandResult);
  rpc ConfigureCounter(CounterParams) returns (CommandResult);
  rpc SetCounterState(CounterState) returns (CommandResult);
  rpc SetLaserState(LaserState) returns (CommandResult);
  
  // Messungen
  rpc PerformODMR(ODMRParams) returns (stream ODMRData);
  rpc PerformRabi(RabiParams) returns (stream RabiData);
  rpc PerformConfocalScan(ScanParams) returns (stream ScanData);
  
  // Systemverwaltung
  rpc GetSystemStatus(StatusRequest) returns (SystemStatus);
  rpc ConfigureSystem(SystemConfig) returns (CommandResult);
}

// Nachrichtentypen für Microwave
message MicrowaveParams {
  double frequency = 1; // in Hz
  double power = 2; // in dBm
}

message MicrowaveOutputState {
  bool enabled = 1;
}

// Weitere Nachrichtentypen...
```

### 5.2 gRPC-Server-Implementierung

```python
import grpc
import nv_simulator_pb2
import nv_simulator_pb2_grpc
from concurrent import futures

class SimulatorServicer(nv_simulator_pb2_grpc.SimulatorServiceServicer):
    """Implementiert die gRPC-Dienste für den NV-Simulator."""
    
    def __init__(self, simulator_state):
        self.simulator_state = simulator_state
    
    def ConfigureMicrowave(self, request, context):
        """Konfiguriert die Mikrowellenparameter."""
        try:
            # Parameter aus der Anfrage extrahieren
            frequency = request.frequency
            power = request.power
            
            # Simulator-Zustand aktualisieren
            self.simulator_state.update_microwave_state(False, frequency, power)
            
            return nv_simulator_pb2.CommandResult(
                success=True,
                message="Microwave parameters configured successfully"
            )
        except Exception as e:
            return nv_simulator_pb2.CommandResult(
                success=False,
                message=f"Error: {str(e)}"
            )

    def PerformODMR(self, request, context):
        """Führt eine ODMR-Messung durch und streamt die Ergebnisse."""
        try:
            # ODMR-Parameter konfigurieren
            start_freq = request.start_frequency
            stop_freq = request.stop_frequency
            steps = request.steps
            power = request.power
            
            # Frequenzliste erstellen
            frequencies = np.linspace(start_freq, stop_freq, steps)
            
            # ODMR-Messung simulieren
            for i, freq in enumerate(frequencies):
                # Mikrowell-Frequenz aktualisieren
                self.simulator_state.update_microwave_state(True, freq, power)
                
                # Photonenzählung simulieren
                count_rate = self._simulate_odmr_count_rate(freq)
                
                # Ergebnis zurückgeben
                yield nv_simulator_pb2.ODMRData(
                    frequency=freq,
                    counts=count_rate,
                    progress=100.0 * i / steps
                )
                
                # Prüfen, ob der Client die Verbindung getrennt hat
                if context.is_active() is False:
                    break
                    
            # Mikrowelle ausschalten
            self.simulator_state.update_microwave_state(False)
            
        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INTERNAL)
            
    def _simulate_odmr_count_rate(self, frequency):
        """Simuliert die Zählrate für eine gegebene Mikrowellenfrequenz."""
        # NV-Modell abrufen
        nv_model = self.simulator_state.get_active_nv_model()
        if nv_model:
            return nv_model.calculate_odmr_count_rate(frequency)
        return 0

def serve_grpc(simulator_state, port=50051):
    """Startet den gRPC-Server für den Simulator."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    nv_simulator_pb2_grpc.add_SimulatorServiceServicer_to_server(
        SimulatorServicer(simulator_state), server
    )
    server.add_insecure_port(f'[::]:{port}')
    server.start()
    return server
```

## 6. Kommunikationsablauf für typische Experimente

### 6.1 ODMR-Messung

Typischer Kommunikationsablauf für eine ODMR-Messung:

```
1. Client sendet: "SET:LASER:POWER:0.001\n"  (Laser-Leistung setzen, 1mW)
   Server antwortet: "OK\n"

2. Client sendet: "LASER:ON\n"  (Laser einschalten)
   Server antwortet: "OK\n"

3. Client sendet: "SET:MICROWAVE:SCAN:START:2.85E9\n"  (Scan-Startfrequenz)
   Server antwortet: "OK\n"

4. Client sendet: "SET:MICROWAVE:SCAN:STOP:2.89E9\n"  (Scan-Endfrequenz)
   Server antwortet: "OK\n"

5. Client sendet: "SET:MICROWAVE:SCAN:STEPS:100\n"  (Anzahl Scan-Punkte)
   Server antwortet: "OK\n"

6. Client sendet: "SET:MICROWAVE:POWER:-20\n"  (Mikrowell-Leistung, -20dBm)
   Server antwortet: "OK\n"

7. Client sendet: "MICROWAVE:SCAN:START\n"  (Scan starten)
   Server antwortet: "OK\n"

8. Server sendet Statusupdates über den Statuskanal:
   {"timestamp": 1635789601.123, "state": {"measurements": {"scan_progress": 0.0}}}
   {"timestamp": 1635789602.234, "state": {"measurements": {"scan_progress": 1.0}}}
   ...
   {"timestamp": 1635789700.345, "state": {"measurements": {"scan_progress": 100.0}}}

9. Client sendet: "GET:MICROWAVE:SCAN:DATA\n"  (Ergebnisse abrufen)
   Server sendet Daten über den Datenkanal (Binärformat)

10. Client sendet: "MICROWAVE:OFF\n"  (Mikrowelle ausschalten)
    Server antwortet: "OK\n"

11. Client sendet: "LASER:OFF\n"  (Laser ausschalten)
    Server antwortet: "OK\n"
```

### 6.2 Rabi-Oszillations-Messung

```
1. Client sendet: "SET:PULSER:SAMPLE_RATE:1E9\n"  (1 GSample/s)
   Server antwortet: "OK\n"

2. Client sendet Pulssequenz-Definition (über Datenkanal oder ZeroMQ)
   - Initialisierungspuls (Laser)
   - Mikrowellenpuls mit variierender Länge
   - Auslesepuls (Laser)
   
3. Client sendet: "PULSER:ON\n"  (Pulser starten)
   Server antwortet: "OK\n"

4. Client sendet: "COUNTER:START\n"  (Zählung starten)
   Server antwortet: "OK\n"

5. Server schickt regelmäßig Statusinformationen und Daten

6. Client sendet: "COUNTER:STOP\n"
   Server antwortet: "OK\n"

7. Client sendet: "PULSER:OFF\n"
   Server antwortet: "OK\n"

8. Client sendet: "GET:COUNTER:DATA\n"  (Daten abrufen)
   Server sendet Daten über den Datenkanal
```

## 7. Fehlercodes und -meldungen

Der Simulator verwendet standardisierte Fehlercodes und -meldungen für eine konsistente Fehlerbehandlung:

| Fehlercode | Kategorie | Beschreibung |
|------------|-----------|--------------|
| 100-199 | Verbindungsfehler | Probleme mit der Netzwerkverbindung |
| 200-299 | Befehlsfehler | Ungültige oder nicht unterstützte Befehle |
| 300-399 | Parameterfehler | Ungültige Parameter für gültige Befehle |
| 400-499 | Zustandsfehler | Befehle, die im aktuellen Zustand nicht ausgeführt werden können |
| 500-599 | Hardwarefehler | Simulierte Hardwarefehler oder -ausfälle |
| 600-699 | Systemfehler | Interne Fehler im Simulator |

Beispiel für eine Fehlerantwort:

```
ERROR:302:Frequency out of range (2.87e10 exceeds maximum 6e9)\n
```

## 8. Sicherheit und Authentifizierung

Für Produktionsumgebungen unterstützt der Simulator verschiedene Sicherheitsmaßnahmen:

### 8.1 TLS/SSL-Verschlüsselung

```python
def start_secure_server(self):
    """Startet einen TLS-verschlüsselten Server."""
    # Zertifikate laden
    context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    context.load_cert_chain(certfile=self.cert_file, keyfile=self.key_file)
    
    # Socket erstellen und binden
    self.secure_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.secure_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    self.secure_socket.bind(('0.0.0.0', self.secure_port))
    self.secure_socket.listen(5)
    
    # Akzeptiere sichere Verbindungen
    while self.running:
        client_sock, addr = self.secure_socket.accept()
        secure_sock = context.wrap_socket(client_sock, server_side=True)
        threading.Thread(target=self._handle_secure_client, 
                        args=(secure_sock, addr), 
                        daemon=True).start()
```

### 8.2 API-Schlüssel-Authentifizierung

```python
def authenticate_request(self, request_data):
    """Authentifiziert eine Anfrage anhand des API-Schlüssels."""
    # API-Schlüssel aus der Anfrage extrahieren
    api_key = request_data.get('api_key', '')
    
    # Mit konfigurierten Schlüsseln vergleichen
    if api_key in self.authorized_keys:
        return True
    
    return False
```

## 9. Performance-Optimierung

Für die effiziente Übertragung großer Datenmengen werden verschiedene Optimierungen eingesetzt:

### 9.1 Datenkompression

```python
def send_compressed_data(self, sock, data):
    """Sendet komprimierte Daten über einen Socket."""
    # Daten mit zlib komprimieren
    compressed_data = zlib.compress(data, level=6)
    
    # Header mit Informationen zur Kompression
    header = struct.pack('!IIQQ', 
                        0x51444154,  # Magic
                        1,           # Komprimierungstyp (1=zlib)
                        len(compressed_data),  # Größe der komprimierten Daten
                        len(data))   # Originalgröße
    
    # Header und komprimierte Daten senden
    sock.sendall(header)
    sock.sendall(compressed_data)
```

### 9.2 Datenmultiplexing

```python
def multiplex_data_streams(self, streams):
    """Multiplexed mehrere Datenströme über einen Socket."""
    # Stream-IDs und Daten zusammenstellen
    multiplexed_data = []
    
    for stream_id, data in streams.items():
        # Chunk-Header
        chunk_header = struct.pack('!II', stream_id, len(data))
        multiplexed_data.append(chunk_header)
        multiplexed_data.append(data)
    
    # Alle Chunks als ein großes Paket senden
    return b''.join(multiplexed_data)
```

## 10. Erweiterbarkeit

Das Kommunikationsprotokoll ist erweiterbar gestaltet, um zukünftige Anforderungen zu erfüllen:

### 10.1 Protokollversionierung

```python
def handle_command(self, command_str):
    """Verarbeitet Befehle mit Versionsunterstützung."""
    # Versionsinformationen extrahieren (falls vorhanden)
    if command_str.startswith('v1:') or command_str.startswith('v2:'):
        version, command = command_str.split(':', 1)
        version = int(version[1:])
    else:
        # Standardversion
        version = 1
        command = command_str
    
    # Befehl basierend auf Version verarbeiten
    if version == 1:
        return self._handle_v1_command(command)
    elif version == 2:
        return self._handle_v2_command(command)
    else:
        return f"ERROR:Unsupported protocol version: {version}"
```

### 10.2 Erweiterbare Datenformate

Alle Datenformate enthalten Felder für zukünftige Erweiterungen und Flags:

```python
def create_extensible_message(self, data, message_type):
    """Erstellt eine erweiterbare Nachricht."""
    message = {
        'version': self.protocol_version,
        'type': message_type,
        'timestamp': time.time(),
        'data': data,
        'extensions': {}  # Reserviert für zukünftige Erweiterungen
    }
    
    return json.dumps(message)
```

## 11. Clients und Bibliotheken

Der Simulator stellt Client-Bibliotheken für die einfache Integration zur Verfügung:

### 11.1 Python-Client

```python
class NVSimulatorClient:
    """Python-Client für den NV-Zentren Simulator."""
    
    def __init__(self, host='localhost', control_port=5555, 
                data_port=5556, status_port=5557):
        """Initialisiert die Verbindung zum Simulator."""
        self.host = host
        self.control_port = control_port
        self.data_port = data_port
        self.status_port = status_port
        
        # Verbindungen erstellen
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_socket.connect((self.host, self.control_port))
        
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_socket.connect((self.host, self.data_port))
        
        self.status_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.status_socket.connect((self.host, self.status_port))
        
        # Status-Thread starten
        self.status_callback = None
        self._status_thread = threading.Thread(target=self._monitor_status, daemon=True)
        self._status_thread.start()
    
    def set_microwave_params(self, frequency, power):
        """Setzt die Mikrowellenparameter."""
        command = f"SET:MICROWAVE:FREQUENCY:{frequency}\n"
        self._send_command(command)
        
        command = f"SET:MICROWAVE:POWER:{power}\n"
        self._send_command(command)
    
    def microwave_on(self):
        """Schaltet den Mikrowellenausgang ein."""
        command = "MICROWAVE:ON\n"
        self._send_command(command)
    
    def _send_command(self, command):
        """Sendet einen Befehl und verarbeitet die Antwort."""
        self.control_socket.sendall(command.encode('utf-8'))
        response = self._receive_response()
        
        if response.startswith('ERROR:'):
            error_msg = response[6:]
            raise NVSimulatorError(error_msg)
        
        return response[3:] if response.startswith('OK:') else None
    
    def _receive_response(self):
        """Empfängt eine Antwort vom Kontrollkanal."""
        response = b""
        while b'\n' not in response:
            chunk = self.control_socket.recv(4096)
            if not chunk:
                raise ConnectionError("Connection closed by server")
            response += chunk
        
        return response.decode('utf-8').strip()
    
    def _monitor_status(self):
        """Überwacht den Statuskanal und ruft Callbacks auf."""
        while True:
            try:
                status_data = b""
                while b'\n' not in status_data:
                    chunk = self.status_socket.recv(4096)
                    if not chunk:
                        time.sleep(0.1)
                        continue
                    status_data += chunk
                
                status_json = status_data.decode('utf-8').strip()
                status = json.loads(status_json)
                
                if self.status_callback:
                    self.status_callback(status)
                    
            except Exception as e:
                print(f"Status monitoring error: {e}")
                time.sleep(1.0)  # Kurze Pause bei Fehlern
    
    def register_status_callback(self, callback):
        """Registriert einen Callback für Statusänderungen."""
        self.status_callback = callback
    
    def close(self):
        """Schließt alle Verbindungen."""
        self.control_socket.close()
        self.data_socket.close()
        self.status_socket.close()
```

### 11.2 Integration mit Qudi

Ein spezieller Adapter ermöglicht die transparente Integration des Simulators in das Qudi-Framework:

```python
class NVSimulatorAdapter:
    """Adaptiert den NV-Simulator für die Verwendung mit Qudi."""
    
    @classmethod
    def create_hardware_modules(cls, config):
        """Erstellt Qudi-konforme Hardware-Module basierend auf der Konfiguration."""
        # Extrahiere Simulator-Konfiguration
        simulator_config = config.get('simulator', {})
        host = simulator_config.get('host', 'localhost')
        
        # Erstelle Client-Instanz
        client = NVSimulatorClient(host=host)
        
        # Erstelle und konfiguriere Hardwaremodule
        modules = {}
        
        if 'microwave' in simulator_config.get('modules', []):
            from qudi_nv_simulator.microwave import RemoteMicrowaveInterface
            modules['microwave'] = RemoteMicrowaveInterface(client)
            
        if 'counter' in simulator_config.get('modules', []):
            from qudi_nv_simulator.counter import RemoteCounterInterface
            modules['fast_counter'] = RemoteCounterInterface(client)
            
        # Weitere Module...
        
        return modules
```

## 12. Zusammenfassung

Die Kommunikationsprotokolle des NV-Zentren Simulators bieten eine robuste, erweiterbare und effiziente Grundlage für die Integration mit dem Qudi-Framework und anderen Clients. Durch die Unterstützung mehrerer Protokolle (TCP/IP, ZeroMQ, gRPC) wird eine hohe Flexibilität gewährleistet, während standardisierte Formate und Konventionen die Entwicklung von Clients und die Fehlersuche erleichtern.

Die wichtigsten Merkmale sind:

1. **Mehrere Kommunikationskanäle**: Separate Verbindungen für Kontrolle, Daten und Status
2. **Einfaches textbasiertes Protokoll**: Für Steuerbefehle und einfache Antworten
3. **Effiziente binäre Datenübertragung**: Für Messdaten und große Datenmengen
4. **Fortgeschrittene Musterfür komplexe Szenarien**: Mit ZeroMQ und gRPC
5. **Sicherheitsmechanismen**: TLS-Verschlüsselung und API-Schlüssel-Authentifizierung
6. **Erweiterbarkeit**: Versionierung und erweiterbare Formate für zukünftige Anforderungen
7. **Clientbibliotheken**: Für die einfache Integration in verschiedene Anwendungen