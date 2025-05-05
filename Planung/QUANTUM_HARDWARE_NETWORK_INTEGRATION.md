# Netzwerkbasierte Quantum-Hardware-Integration mit Qudi

Diese technische Dokumentation beschreibt im Detail, wie ein netzwerkbasierter Quantencomputer mit dem Qudi-Framework kommuniziert. Die Dokumentation konzentriert sich auf die Low-Level-Kommunikationsprotokolle, Portbelegungen, Datenformate und Netzwerkinteraktionen.

## 1. Netzwerkarchitektur und Topologie

### 1.1 Physische Netzwerkarchitektur

Ein typisches Setup für einen netzwerkbasierten Quantencomputer mit Qudi-Integration sieht folgendermaßen aus:

```
┌───────────────┐     ┌────────────────┐     ┌─────────────────┐     ┌───────────────┐
│               │     │                │     │                 │     │               │
│ Qudi-System   │◄───►│ Kontroll-PC /  │◄───►│ Quantum Control │◄───►│ Quantum       │
│ (Messrechner) │     │ Gateway        │     │ Hardware        │     │ Prozessor     │
│               │     │                │     │                 │     │               │
└───────────────┘     └────────────────┘     └─────────────────┘     └───────────────┘
   Ethernet/TCP          Ethernet/TCP            Spezialisierte         Quantenhardware
   (1 Gbps+)             (10 Gbps+)          Hardware-Schnittstellen
```

### 1.2 Netzwerkschichten

| Schicht | Komponenten | Beschreibung |
|---------|-------------|--------------|
| Anwendungsschicht | Qudi-Software, Hardware-Module | Benutzerinteraktion, Messprotokolle, Datenverarbeitung |
| Kommunikationsschicht | TCP/IP, UDP, ZeroMQ, gRPC | Protokolle für den Datenaustausch |
| Transportschicht | Ethernet, InfiniBand | Physikalische Verbindung und Datentransport |

### 1.3 IP-Adressierung und Subnetting

Typische Netzwerkkonfiguration:
- **Qudi-System**: 192.168.1.10/24
- **Kontroll-PC/Gateway**: 192.168.1.2/24 und 10.0.0.1/24
- **Quantum-Hardware**: 10.0.0.2-10.0.0.10/24

Empfohlene Konfiguration:
- Dediziertes VLAN für Quantenhardware
- Firewall-Regeln zur Isolation des Quantennetzwerks
- Statische IP-Adressen für alle Komponenten

## 2. Kommunikationsprotokolle im Detail

### 2.1 TCP/IP-basierte Protokolle

#### 2.1.1 Raw Socket-Kommunikation

**Portbelegungen**:
- **Standard-Kontrollport**: 5555
- **Daten-Streaming-Port**: 5556
- **Status-Monitoring-Port**: 5557

**Verbindungsaufbau**:
```python
import socket

class QuantumTCPInterface:
    def __init__(self, ip_address='10.0.0.2', control_port=5555, 
                 data_port=5556, status_port=5557):
        self.ip_address = ip_address
        self.control_port = control_port
        self.data_port = data_port
        self.status_port = status_port
        
        # Kontrollverbindung
        self.control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.control_socket.connect((self.ip_address, self.control_port))
        
        # Datenverbindung
        self.data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.data_socket.connect((self.ip_address, self.data_port))
        
        # Statusverbindung
        self.status_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.status_socket.connect((self.ip_address, self.status_port))
```

**Befehlsformat**:
Typisches Befehlsformat über den Kontrollkanal:
```
<COMMAND>:<PARAMETER>\n
```

Beispiele:
```
INIT:ALL\n
SET:FREQUENCY:2.87E9\n
RUN:SEQUENCE:RABI\n
```

**Datenaustauschformat**:
Binäre Daten mit Header (24 Bytes) + Datenpaket:
```
struct DataPacket {
    uint32_t magic;           // Magic number (0xQC)
    uint32_t packet_type;     // Type of data
    uint32_t packet_size;     // Size of data in bytes
    uint64_t timestamp;       // Microseconds since epoch
    uint32_t sequence_number; // Packet sequence number
    byte[]   data;            // Actual data
};
```

**Statusnachrichten**:
JSON-formatierte Statusaktualisierungen:
```json
{
    "status": "RUNNING",
    "progress": 42.5,
    "subsystems": {
        "qubit_control": "OK",
        "readout": "OK",
        "microwave": "OK",
        "laser": "OK"
    },
    "timestamp": 1635789600,
    "error_code": null
}
```

**Vollständige Socket-Kommunikation**:
```python
def send_command(self, command, parameters=None):
    """Sendet einen Befehl an den Quantencomputer."""
    if parameters:
        cmd_str = f"{command}:{parameters}\n"
    else:
        cmd_str = f"{command}\n"
        
    self.control_socket.sendall(cmd_str.encode('utf-8'))
    
    # Warte auf Antwort (Bestätigung)
    response = b""
    while b'\n' not in response:
        chunk = self.control_socket.recv(4096)
        if not chunk:
            raise ConnectionError("Connection closed by remote host")
        response += chunk
        
    return response.decode('utf-8').strip()

def receive_data(self, expected_size=None, timeout=10.0):
    """Empfängt Daten vom Quantencomputer."""
    self.data_socket.settimeout(timeout)
    
    # Header empfangen (24 Bytes)
    header = b""
    while len(header) < 24:
        chunk = self.data_socket.recv(24 - len(header))
        if not chunk:
            raise ConnectionError("Connection closed during header reception")
        header += chunk
    
    # Header parsen
    magic, packet_type, packet_size, timestamp, seq_num = struct.unpack('!IIIQQ', header)
    
    if magic != 0x5143: # 'QC' in hex
        raise ValueError(f"Invalid magic number: {magic:x}")
    
    # Daten empfangen
    data = b""
    while len(data) < packet_size:
        chunk = self.data_socket.recv(min(4096, packet_size - len(data)))
        if not chunk:
            raise ConnectionError("Connection closed during data reception")
        data += chunk
    
    return {
        'type': packet_type,
        'timestamp': timestamp,
        'sequence': seq_num,
        'data': data
    }
```

#### 2.1.2 ZeroMQ-basierte Kommunikation

ZeroMQ bietet eine höhere Abstraktionsebene und ist besonders geeignet für Pub/Sub-Muster bei Quantenmessungen.

**Portbelegungen**:
- **REQ/REP-Port (Befehle)**: 5555
- **PUB/SUB-Port (Daten)**: 5556
- **PUB/SUB-Port (Status)**: 5557

**Verbindungsaufbau**:
```python
import zmq

class QuantumZMQInterface:
    def __init__(self, ip_address='10.0.0.2'):
        self.ip_address = ip_address
        
        # ZeroMQ Kontext erstellen
        self.context = zmq.Context()
        
        # Request-Reply Socket für Befehle
        self.command_socket = self.context.socket(zmq.REQ)
        self.command_socket.connect(f"tcp://{self.ip_address}:5555")
        
        # Subscriber Socket für Daten
        self.data_socket = self.context.socket(zmq.SUB)
        self.data_socket.connect(f"tcp://{self.ip_address}:5556")
        self.data_socket.setsockopt_string(zmq.SUBSCRIBE, "DATA")
        
        # Subscriber Socket für Status
        self.status_socket = self.context.socket(zmq.SUB)
        self.status_socket.connect(f"tcp://{self.ip_address}:5557")
        self.status_socket.setsockopt_string(zmq.SUBSCRIBE, "STATUS")
        
        # Poller für asynchrones Empfangen
        self.poller = zmq.Poller()
        self.poller.register(self.data_socket, zmq.POLLIN)
        self.poller.register(self.status_socket, zmq.POLLIN)
```

**Befehlsformat (JSON)**:
```json
{
    "command": "SET_PARAMETERS",
    "parameters": {
        "frequency": 2.87e9,
        "power": -10.0,
        "pulse_duration": 100e-9
    },
    "id": "cmd-1234"
}
```

**Datenaustausch via ZeroMQ**:
```python
def send_command(self, command, parameters=None, timeout=5000):
    """Sendet einen Befehl an den Quantencomputer via ZeroMQ."""
    cmd_obj = {
        "command": command,
        "id": f"cmd-{uuid.uuid4().hex[:8]}",
        "timestamp": time.time()
    }
    
    if parameters:
        cmd_obj["parameters"] = parameters
    
    # Befehl senden
    self.command_socket.send_json(cmd_obj)
    
    # Mit Timeout auf Antwort warten
    if self.command_socket.poll(timeout):
        response = self.command_socket.recv_json()
        return response
    else:
        raise TimeoutError(f"Command {command} timed out after {timeout}ms")

def receive_data(self, timeout=1000):
    """Empfängt Daten und Statusupdates asynchron."""
    events = dict(self.poller.poll(timeout))
    
    results = {}
    
    if self.data_socket in events:
        topic, data = self.data_socket.recv_multipart()
        results['data'] = msgpack.unpackb(data)
    
    if self.status_socket in events:
        topic, status = self.status_socket.recv_multipart()
        results['status'] = json.loads(status)
    
    return results
```

### 2.2 SCPI-basierte Instrumentenkommunikation

Scientific Command Protocol for Instrumentation (SCPI) ist ein Standard für Messgerätekommunikation.

**Portbelegung**:
- **Standard SCPI Port**: 5025

**Beispiel für SCPI-Kommunikation**:
```python
import socket

class QuantumSCPIInterface:
    def __init__(self, ip_address='10.0.0.2', port=5025):
        self.ip_address = ip_address
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.ip_address, self.port))
        self.socket.settimeout(5.0)
        
    def send_command(self, command):
        """Sendet SCPI-Befehl ohne Rückgabewert."""
        self.socket.sendall(f"{command}\n".encode('ascii'))
        
    def query(self, command):
        """Sendet SCPI-Befehl und empfängt Antwort."""
        self.socket.sendall(f"{command}\n".encode('ascii'))
        
        response = b""
        while b'\n' not in response:
            chunk = self.socket.recv(4096)
            if not chunk:
                break
            response += chunk
            
        return response.decode('ascii').strip()
        
    def initialize_quantum_system(self):
        """Initialisiert das Quantensystem mit SCPI."""
        self.send_command("*RST")  # Reset
        self.send_command("*CLS")  # Clear status registers
        idn = self.query("*IDN?")  # Identifikation
        self.log.info(f"Connected to quantum system: {idn}")
        
        # System-spezifische Konfiguration
        self.send_command("QSYStem:INITialize")
        self.send_command("QSYStem:TEMPerature:SET 3.0")  # Setze Temperatur (Kelvin)
        
        # Qubit-Initialisierung
        self.send_command("QBITs:COUNt 5")  # 5-Qubit-System
        self.send_command("QBITs:INITialize:ALL")
```

**SCPI-Befehlssatz für Quantenhardware**:

| Befehlsgruppe | Beispielbefehle | Beschreibung |
|---------------|-----------------|--------------|
| System | `*RST`, `*IDN?`, `SYSTem:ERRor?` | Allgemeine Gerätebefehle |
| QSystem | `QSYStem:INITialize`, `QSYStem:STATus?` | Quantensystem-Management |
| QBits | `QBITs:COUNt?`, `QBITs:MEAS:ALL?` | Qubitsteuerung und -messung |
| QPulse | `QPULse:LOAD "X90"`, `QPULse:RUN` | Pulsprogrammierung |
| QSequence | `QSEQuence:LOAD "BELL"`, `QSEQuence:RUN` | Sequenzausführung |

### 2.3 gRPC-basierte Kommunikation

gRPC bietet eine moderne, effiziente RPC-Lösung mit starker Typisierung.

**Portbelegung**:
- **gRPC Port**: 50051

**Protocol Buffer Definition (quantum.proto)**:
```protobuf
syntax = "proto3";

package quantum;

service QuantumProcessor {
  // Systemkontrolle
  rpc Initialize(InitializeRequest) returns (InitializeResponse);
  rpc GetStatus(StatusRequest) returns (StatusResponse);
  
  // Quantenoperationen
  rpc RunCircuit(Circuit) returns (MeasurementResults);
  rpc RunCalibration(CalibrationType) returns (CalibrationResults);
  
  // Streaming-Messergebnisse
  rpc MonitorResults(CircuitHandle) returns (stream MeasurementResults);
}

message InitializeRequest {
  bool reset_qubits = 1;
  bool run_calibration = 2;
}

message InitializeResponse {
  bool success = 1;
  string message = 2;
  SystemInformation system_info = 3;
}

message SystemInformation {
  string processor_id = 1;
  int32 num_qubits = 2;
  float temperature_k = 3;
  map<string, float> coherence_times = 4;
  repeated QubitConnectivity connectivity = 5;
}

message QubitConnectivity {
  int32 qubit1 = 1;
  int32 qubit2 = 2;
  float coupling_strength = 3;
}

message StatusRequest {
  bool include_temperatures = 1;
  bool include_error_rates = 2;
}

message StatusResponse {
  enum SystemStatus {
    UNKNOWN = 0;
    INITIALIZING = 1;
    IDLE = 2;
    RUNNING = 3;
    ERROR = 4;
    CALIBRATING = 5;
  }
  
  SystemStatus status = 1;
  float progress = 2;
  string message = 3;
  map<string, float> subsystem_status = 4;
}

message Circuit {
  repeated Instruction instructions = 1;
  repeated int32 qubits_to_measure = 2;
  int32 repetitions = 3;
}

message Instruction {
  enum GateType {
    IDENTITY = 0;
    X = 1;
    Y = 2;
    Z = 3;
    H = 4;
    CNOT = 5;
    // Weitere Gates...
  }
  
  GateType gate = 1;
  repeated int32 qubits = 2;
  repeated float parameters = 3;  // Für parametrisierte Gates
}

message MeasurementResults {
  string circuit_id = 1;
  int32 repetitions = 2;
  repeated Measurement measurements = 3;
}

message Measurement {
  repeated bool qubit_states = 1;
  int32 count = 2;
}

message CircuitHandle {
  string circuit_id = 1;
}

message CalibrationType {
  enum Type {
    FULL = 0;
    SINGLE_QUBIT = 1;
    TWO_QUBIT = 2;
    READOUT = 3;
  }
  
  Type type = 1;
  repeated int32 target_qubits = 2;
}

message CalibrationResults {
  bool success = 1;
  string message = 2;
  map<string, float> parameters = 3;
  map<string, float> error_rates = 4;
}
```

**gRPC-Client-Implementierung**:
```python
import grpc
import quantum_pb2
import quantum_pb2_grpc

class QuantumGRPCInterface:
    def __init__(self, ip_address='10.0.0.2', port=50051):
        """Initialisiert die gRPC-Verbindung zur Quantenhardware."""
        self.channel = grpc.insecure_channel(f"{ip_address}:{port}")
        self.stub = quantum_pb2_grpc.QuantumProcessorStub(self.channel)
        
    def initialize_system(self, reset_qubits=True, run_calibration=False):
        """Initialisiert das Quantensystem."""
        request = quantum_pb2.InitializeRequest(
            reset_qubits=reset_qubits,
            run_calibration=run_calibration
        )
        
        try:
            response = self.stub.Initialize(request)
            if response.success:
                system_info = response.system_info
                return {
                    "processor_id": system_info.processor_id,
                    "num_qubits": system_info.num_qubits,
                    "temperature_k": system_info.temperature_k,
                    "coherence_times": dict(system_info.coherence_times)
                }
            else:
                raise RuntimeError(f"Initialization failed: {response.message}")
        except grpc.RpcError as e:
            raise ConnectionError(f"gRPC error: {e.details()}")
            
    def get_status(self, include_temperatures=True, include_error_rates=True):
        """Fragt den aktuellen Status des Quantenprozessors ab."""
        request = quantum_pb2.StatusRequest(
            include_temperatures=include_temperatures,
            include_error_rates=include_error_rates
        )
        
        try:
            response = self.stub.GetStatus(request)
            return {
                "status": quantum_pb2.StatusResponse.SystemStatus.Name(response.status),
                "progress": response.progress,
                "message": response.message,
                "subsystem_status": dict(response.subsystem_status)
            }
        except grpc.RpcError as e:
            raise ConnectionError(f"gRPC error: {e.details()}")
            
    def run_circuit(self, circuit_definition, qubits_to_measure, repetitions=1024):
        """Führt einen Quantenschaltkreis aus und gibt Messergebnisse zurück."""
        # Konvertiert die Circuit-Definition in das gRPC-Format
        instructions = []
        for gate_def in circuit_definition:
            gate_type = quantum_pb2.Instruction.GateType.Value(gate_def["gate"])
            qubits = gate_def.get("qubits", [])
            parameters = gate_def.get("parameters", [])
            
            instruction = quantum_pb2.Instruction(
                gate=gate_type,
                qubits=qubits,
                parameters=parameters
            )
            instructions.append(instruction)
            
        # Erstelle die Circuit-Nachricht
        circuit = quantum_pb2.Circuit(
            instructions=instructions,
            qubits_to_measure=qubits_to_measure,
            repetitions=repetitions
        )
        
        try:
            # Führe den Schaltkreis aus
            results = self.stub.RunCircuit(circuit)
            
            # Konvertiere die Ergebnisse in ein Python-Dictionary
            processed_results = {
                "circuit_id": results.circuit_id,
                "repetitions": results.repetitions,
                "measurements": []
            }
            
            for measurement in results.measurements:
                processed_results["measurements"].append({
                    "qubit_states": list(measurement.qubit_states),
                    "count": measurement.count
                })
                
            return processed_results
            
        except grpc.RpcError as e:
            raise RuntimeError(f"Circuit execution failed: {e.details()}")
            
    def monitor_results(self, circuit_id):
        """Startet einen Stream, um Messergebnisse kontinuierlich zu empfangen."""
        circuit_handle = quantum_pb2.CircuitHandle(circuit_id=circuit_id)
        
        try:
            for result in self.stub.MonitorResults(circuit_handle):
                yield {
                    "circuit_id": result.circuit_id,
                    "repetitions": result.repetitions,
                    "measurements": [
                        {
                            "qubit_states": list(m.qubit_states),
                            "count": m.count
                        } for m in result.measurements
                    ]
                }
        except grpc.RpcError as e:
            raise ConnectionError(f"Monitoring error: {e.details()}")
```

## 3. Datenformate und Serialisierung

### 3.1 QASM für Schaltkreisbeschreibung

Das Quantum Assembly Language (QASM) Format wird für die Beschreibung von Quantenschaltkreisen verwendet:

```qasm
OPENQASM 2.0;
include "qelib1.inc";

qreg q[5];
creg c[5];

// Hadamard auf Qubit 0
h q[0];

// CNOT mit Qubit 0 als Control und Qubit 1 als Target
cx q[0],q[1];

// Messe alle Qubits
measure q -> c;
```

**Implementierung in Qudi**:
```python
def send_qasm_circuit(self, qasm_string):
    """Sendet einen QASM-Schaltkreis zur Ausführung."""
    qasm_encoded = qasm_string.encode('utf-8')
    
    # Sende Befehlsheader
    cmd = "LOAD:QASM\n"
    self.control_socket.sendall(cmd.encode('utf-8'))
    
    # Sende Größe des QASM-Codes
    size_header = struct.pack('!I', len(qasm_encoded))
    self.control_socket.sendall(size_header)
    
    # Sende QASM-Code
    self.control_socket.sendall(qasm_encoded)
    
    # Warte auf Bestätigung
    response = self.control_socket.recv(1024)
    
    if response.startswith(b'OK'):
        # Führe den Schaltkreis aus
        self.control_socket.sendall(b'RUN\n')
        run_response = self.control_socket.recv(1024)
        
        if run_response.startswith(b'RUNNING'):
            return True
        else:
            raise RuntimeError(f"Failed to run circuit: {run_response.decode('utf-8')}")
    else:
        raise ValueError(f"Failed to load QASM: {response.decode('utf-8')}")
```

### 3.2 JSON für Konfiguration und Ergebnisse

Beispiel für ein JSON-Konfigurationsdokument:

```json
{
    "quantum_processor": {
        "name": "QProc-5Q",
        "qubits": 5,
        "topology": [
            [0, 1], [1, 2], [2, 3], [3, 4], [4, 0]
        ],
        "coherence_times": {
            "T1": [45.6e-6, 39.8e-6, 42.1e-6, 47.2e-6, 41.5e-6],
            "T2": [12.3e-6, 10.8e-6, 11.4e-6, 12.7e-6, 11.1e-6]
        },
        "gate_times": {
            "X": 20e-9,
            "H": 40e-9,
            "CNOT": 120e-9
        }
    },
    "control_parameters": {
        "pulse_amplitude": 0.8,
        "readout_duration": 100e-9,
        "reset_time": 500e-9
    },
    "network": {
        "address": "10.0.0.2",
        "control_port": 5555,
        "data_port": 5556,
        "status_port": 5557
    }
}
```

Beispiel für ein JSON-Ergebnisdokument:

```json
{
    "experiment_id": "exp-20221030-001",
    "timestamp": "2022-10-30T14:32:45.123Z",
    "circuit_type": "bell_state",
    "num_shots": 1024,
    "results": {
        "counts": {
            "00": 502,
            "11": 498,
            "01": 12,
            "10": 12
        },
        "probabilities": {
            "00": 0.490234375,
            "11": 0.486328125,
            "01": 0.01171875,
            "10": 0.01171875
        }
    },
    "metadata": {
        "temperature": 0.015,
        "readout_errors": [0.023, 0.019],
        "experiment_duration": 5.234
    }
}
```

### 3.3 Binäre Datenformate für Messergebnisse

Für große Datensätze (z.B. Raw-Readout, Tomographie-Daten) werden binäre Formate verwendet:

**Beispiel für ein Binärformat**:
```
[Header: 32 Bytes]
- Magic Number (4 Bytes): 0x51444154 ('QDAT')
- Version (4 Bytes): 0x00000001
- Data Type (4 Bytes): 0x00000001 (Readout Data)
- Total Size (8 Bytes): Gesamtgröße in Bytes
- Num Records (8 Bytes): Anzahl der Datensätze
- Reserved (4 Bytes): 0x00000000

[Datensatz 1]
- Timestamp (8 Bytes): Mikrosekunden seit Epoch
- Metadata Size (4 Bytes): Größe der Metadaten
- Data Size (4 Bytes): Größe der Daten
- Metadata (variabel): JSON-formatierte Metadaten
- Data (variabel): Binäre Daten

[Datensatz 2]
...
```

**Binärlese-Funktion**:
```python
def read_binary_results(self, file_path):
    """Liest binäre Ergebnisse aus einer Datei."""
    with open(file_path, 'rb') as f:
        # Header lesen
        magic = struct.unpack('!I', f.read(4))[0]
        if magic != 0x51444154:  # 'QDAT'
            raise ValueError("Invalid file format")
            
        version = struct.unpack('!I', f.read(4))[0]
        data_type = struct.unpack('!I', f.read(4))[0]
        total_size = struct.unpack('!Q', f.read(8))[0]
        num_records = struct.unpack('!Q', f.read(8))[0]
        _ = f.read(4)  # Reserved bytes
        
        records = []
        for _ in range(num_records):
            # Datensatzheader lesen
            timestamp = struct.unpack('!Q', f.read(8))[0]
            metadata_size = struct.unpack('!I', f.read(4))[0]
            data_size = struct.unpack('!I', f.read(4))[0]
            
            # Metadaten lesen und parsen
            metadata_bytes = f.read(metadata_size)
            metadata = json.loads(metadata_bytes.decode('utf-8'))
            
            # Daten lesen
            data = f.read(data_size)
            
            # Daten je nach Typ verarbeiten
            if data_type == 1:  # Readout-Daten
                # Beispiel: Konvertiere 8-Bit-Readout-Werte in Python-Liste
                readout_values = [b for b in data]
            else:
                readout_values = data
                
            records.append({
                'timestamp': timestamp,
                'metadata': metadata,
                'data': readout_values
            })
            
        return {
            'version': version,
            'data_type': data_type,
            'num_records': num_records,
            'records': records
        }
```

## 4. Fehlerbehandlung und Wiederherstellung

### 4.1 Verbindungswiederherstellung

```python
class RobustQuantumConnection:
    def __init__(self, ip_address, port, max_retries=5, retry_delay=2.0):
        self.ip_address = ip_address
        self.port = port
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.socket = None
        
    def connect(self):
        """Stellt eine Verbindung her mit Wiederholungsversuchen."""
        retries = 0
        last_error = None
        
        while retries < self.max_retries:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(10.0)
                self.socket.connect((self.ip_address, self.port))
                self.log.info(f"Connected to quantum hardware at {self.ip_address}:{self.port}")
                return True
            except (socket.timeout, ConnectionRefusedError, socket.error) as e:
                last_error = e
                retries += 1
                self.log.warning(f"Connection attempt {retries}/{self.max_retries} failed: {e}")
                
                if self.socket:
                    self.socket.close()
                    self.socket = None
                
                time.sleep(self.retry_delay)
        
        self.log.error(f"Failed to connect after {self.max_retries} attempts: {last_error}")
        raise ConnectionError(f"Could not connect to quantum hardware: {last_error}")
        
    def send_with_retry(self, data):
        """Sendet Daten mit automatischer Wiederverbindung."""
        retries = 0
        
        while retries < self.max_retries:
            try:
                if not self.socket:
                    self.connect()
                
                self.socket.sendall(data)
                return True
            except (socket.timeout, BrokenPipeError, ConnectionResetError) as e:
                retries += 1
                self.log.warning(f"Send failed (attempt {retries}/{self.max_retries}): {e}")
                
                if self.socket:
                    self.socket.close()
                    self.socket = None
                
                time.sleep(self.retry_delay)
        
        raise ConnectionError(f"Failed to send data after {self.max_retries} attempts")
```

### 4.2 Heartbeat-Mechanismus

```python
import threading
import time

class QuantumHeartbeat:
    def __init__(self, quantum_interface, interval=5.0):
        self.quantum_interface = quantum_interface
        self.interval = interval
        self.running = False
        self.heartbeat_thread = None
        
    def start(self):
        """Startet den Heartbeat-Thread."""
        if self.running:
            return
            
        self.running = True
        self.heartbeat_thread = threading.Thread(target=self._heartbeat_loop)
        self.heartbeat_thread.daemon = True
        self.heartbeat_thread.start()
        
    def stop(self):
        """Stoppt den Heartbeat-Thread."""
        self.running = False
        if self.heartbeat_thread:
            self.heartbeat_thread.join(timeout=self.interval*2)
            
    def _heartbeat_loop(self):
        """Hauptschleife für Heartbeat-Nachrichten."""
        while self.running:
            try:
                # Sende Heartbeat und überprüfe Antwort
                response = self.quantum_interface.send_command("PING")
                
                if response != "PONG":
                    self.quantum_interface.log.warning(
                        f"Unexpected heartbeat response: {response}"
                    )
                    # Optional: Verbindung neu aufbauen
                    self.quantum_interface.reconnect()
            except Exception as e:
                self.quantum_interface.log.error(f"Heartbeat failed: {e}")
                try:
                    self.quantum_interface.reconnect()
                except Exception as reconnect_error:
                    self.quantum_interface.log.error(
                        f"Reconnection failed: {reconnect_error}"
                    )
                    
            # Warte bis zum nächsten Heartbeat
            time.sleep(self.interval)
```

## 5. Sicherheitsaspekte

### 5.1 Authentifizierung

**TLS-Zertifikatsbasierte Authentifizierung**:
```python
import ssl
import socket

class SecureQuantumConnection:
    def __init__(self, ip_address, port, cert_file, key_file, ca_file=None):
        self.ip_address = ip_address
        self.port = port
        self.cert_file = cert_file
        self.key_file = key_file
        self.ca_file = ca_file
        self.socket = None
        self.ssl_socket = None
        
    def connect(self):
        """Stellt eine sichere TLS-Verbindung zur Quantenhardware her."""
        # SSL-Kontext erstellen
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        
        if self.ca_file:
            context.load_verify_locations(self.ca_file)
            
        # Client-Zertifikat konfigurieren für Mutual TLS
        context.load_cert_chain(certfile=self.cert_file, keyfile=self.key_file)
        
        # Verbindungsaufbau
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.ssl_socket = context.wrap_socket(
            self.socket, server_hostname=self.ip_address
        )
        
        try:
            self.ssl_socket.connect((self.ip_address, self.port))
            self.log.info(f"Secure connection established to {self.ip_address}:{self.port}")
            self.log.debug(f"Using cipher: {self.ssl_socket.cipher()}")
            return True
        except ssl.SSLError as e:
            self.log.error(f"SSL error: {e}")
            if self.ssl_socket:
                self.ssl_socket.close()
            if self.socket:
                self.socket.close()
            self.ssl_socket = None
            self.socket = None
            raise
```

### 5.2 Verschlüsselung von Nutzlasten

```python
from cryptography.fernet import Fernet
import base64

class EncryptedQuantumPayload:
    def __init__(self, encryption_key):
        """Initialisiert die Verschlüsselung mit einem gegebenen Schlüssel."""
        if isinstance(encryption_key, str):
            # Konvertiere String-Schlüssel zu Bytes
            encryption_key = encryption_key.encode('utf-8')
            # Stelle sicher, dass der Schlüssel URL-sicher base64-kodiert ist
            encryption_key = base64.urlsafe_b64encode(
                encryption_key.ljust(32)[:32]
            )
            
        self.cipher = Fernet(encryption_key)
        
    def encrypt_payload(self, payload):
        """Verschlüsselt eine Payload (dict oder str)."""
        if isinstance(payload, dict):
            payload = json.dumps(payload)
            
        if isinstance(payload, str):
            payload = payload.encode('utf-8')
            
        encrypted_data = self.cipher.encrypt(payload)
        return base64.b64encode(encrypted_data).decode('utf-8')
        
    def decrypt_payload(self, encrypted_payload):
        """Entschlüsselt eine verschlüsselte Payload."""
        if isinstance(encrypted_payload, str):
            encrypted_payload = base64.b64decode(encrypted_payload)
            
        decrypted_data = self.cipher.decrypt(encrypted_payload)
        
        try:
            # Versuche, als JSON zu parsen
            return json.loads(decrypted_data)
        except json.JSONDecodeError:
            # Wenn kein JSON, gib Bytestring zurück
            return decrypted_data.decode('utf-8')
```

## 6. Performance-Optimierung

### 6.1 Parallele Kommunikation

```python
import threading
import queue
import time

class ParallelQuantumCommunication:
    def __init__(self, quantum_interface, max_workers=4):
        self.quantum_interface = quantum_interface
        self.max_workers = max_workers
        self.command_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.workers = []
        self.running = False
        
    def start_workers(self):
        """Startet Worker-Threads für parallele Kommunikation."""
        self.running = True
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop, 
                name=f"QComm-Worker-{i}"
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
            
    def stop_workers(self):
        """Stoppt alle Worker-Threads."""
        self.running = False
        
        # Leere Aufgaben in die Queue stellen, um blockierte Worker zu wecken
        for _ in range(len(self.workers)):
            self.command_queue.put(None)
            
        # Warte auf Beendigung aller Worker
        for worker in self.workers:
            worker.join(timeout=2.0)
            
        self.workers = []
        
    def _worker_loop(self):
        """Worker-Thread-Hauptschleife."""
        while self.running:
            try:
                task = self.command_queue.get(timeout=1.0)
                
                if task is None:
                    self.command_queue.task_done()
                    continue
                    
                command, args, kwargs, task_id = task
                
                try:
                    # Führe die Kommandofunktion aus
                    method = getattr(self.quantum_interface, command)
                    result = method(*args, **kwargs)
                    
                    # Speichere das Ergebnis
                    self.result_queue.put((task_id, True, result))
                except Exception as e:
                    # Bei Fehler, speichere die Exception
                    self.result_queue.put((task_id, False, e))
                    
                self.command_queue.task_done()
                
            except queue.Empty:
                # Timeout bei leerer Queue, prüfe ob noch läuft
                continue
                
            except Exception as e:
                self.quantum_interface.log.error(
                    f"Worker error: {e}", exc_info=True
                )
                
    def execute_parallel(self, commands):
        """
        Führt mehrere Kommandos parallel aus.
        
        Args:
            commands: Liste von (command, args, kwargs)-Tupeln
            
        Returns:
            Dictionary mit Task-ID -> Ergebnis
        """
        if not self.running:
            self.start_workers()
            
        # Task-IDs erstellen und Kommandos in die Queue stellen
        task_ids = []
        for i, (command, args, kwargs) in enumerate(commands):
            task_id = f"task-{time.time()}-{i}"
            task_ids.append(task_id)
            self.command_queue.put((command, args, kwargs, task_id))
            
        # Warte auf Abschluss aller Aufgaben
        self.command_queue.join()
        
        # Sammle alle Ergebnisse
        results = {}
        while not self.result_queue.empty():
            task_id, success, result = self.result_queue.get()
            if task_id in task_ids:
                if success:
                    results[task_id] = result
                else:
                    # Bei Fehler, Exception weitergeben
                    raise result
                    
        # Prüfe, ob alle Task-IDs Ergebnisse haben
        missing_tasks = set(task_ids) - set(results.keys())
        if missing_tasks:
            raise RuntimeError(
                f"Missing results for tasks: {missing_tasks}"
            )
            
        return results
```

### 6.2 Pipelining von Operationen

```python
class QuantumOperationPipeline:
    def __init__(self, quantum_interface):
        self.quantum_interface = quantum_interface
        self.pipeline = []
        
    def add_operation(self, operation, args=None, kwargs=None):
        """Fügt eine Operation zur Pipeline hinzu."""
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}
            
        self.pipeline.append((operation, args, kwargs))
        return self
        
    def add_init(self):
        """Initialisierung als erste Operation."""
        return self.add_operation("initialize_system")
        
    def add_x_gate(self, qubit):
        """X-Gate zur Pipeline hinzufügen."""
        return self.add_operation(
            "apply_gate", 
            args=["X", [qubit]]
        )
        
    def add_h_gate(self, qubit):
        """H-Gate zur Pipeline hinzufügen."""
        return self.add_operation(
            "apply_gate", 
            args=["H", [qubit]]
        )
        
    def add_cnot_gate(self, control, target):
        """CNOT-Gate zur Pipeline hinzufügen."""
        return self.add_operation(
            "apply_gate", 
            args=["CNOT", [control, target]]
        )
        
    def add_measure(self, qubits=None):
        """Messung zur Pipeline hinzufügen."""
        return self.add_operation(
            "measure_qubits", 
            args=[qubits]
        )
        
    def execute(self):
        """Führt alle Operationen in der Pipeline aus."""
        # Kombiniere Operationen zu einem Batch, wenn möglich
        batched_operations = self._optimize_pipeline()
        
        results = []
        for operation in batched_operations:
            if isinstance(operation, tuple):
                # Einzelne Operation
                op_name, args, kwargs = operation
                method = getattr(self.quantum_interface, op_name)
                result = method(*args, **kwargs)
                results.append(result)
            elif isinstance(operation, list):
                # Batch von Operationen
                batch_results = self.quantum_interface.execute_batch([
                    (op_name, args, kwargs)
                    for op_name, args, kwargs in operation
                ])
                results.extend(batch_results)
                
        # Pipeline zurücksetzen
        self.pipeline = []
        
        return results
        
    def _optimize_pipeline(self):
        """Optimiert die Pipeline für effiziente Ausführung."""
        # Hier könnte eine komplexere Optimierungslogik stehen
        # Beispiel: Gruppiere aufeinanderfolgende Gate-Operationen
        
        optimized = []
        current_batch = []
        
        for op in self.pipeline:
            op_name = op[0]
            
            if op_name.startswith("apply_gate"):
                # Gate-Operation zum aktuellen Batch hinzufügen
                current_batch.append(op)
            else:
                # Wenn nicht Gate-Operation, vorherigen Batch abschließen
                if current_batch:
                    optimized.append(current_batch)
                    current_batch = []
                optimized.append(op)
                
        # Letzten Batch hinzufügen, falls vorhanden
        if current_batch:
            optimized.append(current_batch)
            
        return optimized
```

## 7. Qudi-Integration und Beispielimplementierungen

### 7.1 Qudi Hardware-Modul für Netzwerk-Quantenhardware

```python
from qudi.core.module import Base
from qudi.core.configoption import ConfigOption
from qudi.interface.microwave_interface import MicrowaveInterface
from qudi.util.mutex import Mutex

import socket
import json
import time
import numpy as np

class NetworkQuantumHardware(MicrowaveInterface, Base):
    """
    Qudi Hardware-Modul für netzwerkbasierte Quantenhardware.
    
    Beispiel-Konfiguration:
    
    network_quantum:
        module.Class: 'quantum.network_quantum_hardware.NetworkQuantumHardware'
        ip_address: '10.0.0.2'
        control_port: 5555
        data_port: 5556
        status_port: 5557
        connection_timeout: 5.0
    """
    
    # Konfigurations-Optionen
    _ip_address = ConfigOption('ip_address', default='127.0.0.1')
    _control_port = ConfigOption('control_port', default=5555)
    _data_port = ConfigOption('data_port', default=5556)
    _status_port = ConfigOption('status_port', default=5557)
    _connection_timeout = ConfigOption('connection_timeout', default=5.0)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Thread-Sicherheit
        self._thread_lock = Mutex()
        
        # Socket-Verbindungen
        self._control_socket = None
        self._data_socket = None
        self._status_socket = None
        
        # Zustands-Tracking
        self._cw_frequency = 2.87e9  # Standard-Wert
        self._cw_power = -20         # Standard-Wert
        self._scan_frequencies = None
        self._is_scanning = False
        
        # Constraints initialisieren (werden in on_activate gefüllt)
        self._constraints = None
        
    def on_activate(self):
        """Wird beim Aktivieren des Moduls aufgerufen."""
        try:
            # Verbindungen aufbauen
            self._establish_connections()
            
            # Hardware-Informationen abfragen
            hw_info = self._query_hardware_info()
            
            # Constraints basierend auf Hardware-Informationen erzeugen
            self._constraints = self._create_constraints(hw_info)
            
            self.log.info(f"Successfully connected to quantum hardware at {self._ip_address}")
            
        except Exception as e:
            self.log.error(f"Failed to activate quantum hardware module: {e}")
            raise
            
    def on_deactivate(self):
        """Wird beim Deaktivieren des Moduls aufgerufen."""
        try:
            # Hardware ausschalten
            self.off()
            
            # Verbindungen schließen
            self._close_connections()
            
            self.log.info("Quantum hardware module deactivated")
            
        except Exception as e:
            self.log.error(f"Error during deactivation: {e}")
            
    def _establish_connections(self):
        """Stellt alle Socket-Verbindungen her."""
        with self._thread_lock:
            # Kontrollverbindung
            self._control_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._control_socket.settimeout(self._connection_timeout)
            self._control_socket.connect((self._ip_address, self._control_port))
            
            # Datenverbindung
            self._data_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._data_socket.settimeout(self._connection_timeout)
            self._data_socket.connect((self._ip_address, self._data_port))
            
            # Statusverbindung
            self._status_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._status_socket.settimeout(self._connection_timeout)
            self._status_socket.connect((self._ip_address, self._status_port))
            
    def _close_connections(self):
        """Schließt alle Socket-Verbindungen."""
        with self._thread_lock:
            for socket_name in ['_control_socket', '_data_socket', '_status_socket']:
                socket_obj = getattr(self, socket_name)
                if socket_obj is not None:
                    try:
                        socket_obj.close()
                    except Exception as e:
                        self.log.warning(f"Error closing {socket_name}: {e}")
                    finally:
                        setattr(self, socket_name, None)
                        
    def _send_command(self, command, expect_response=True):
        """Sendet einen Befehl über die Kontrollverbindung."""
        with self._thread_lock:
            if self._control_socket is None:
                raise ConnectionError("Control socket not connected")
                
            try:
                # Befehl mit Zeilenumbruch senden
                cmd_str = f"{command}\n"
                self._control_socket.sendall(cmd_str.encode('utf-8'))
                
                if expect_response:
                    # Auf Antwort warten
                    response = b""
                    while b'\n' not in response:
                        chunk = self._control_socket.recv(4096)
                        if not chunk:
                            raise ConnectionError("Connection closed by remote host")
                        response += chunk
                        
                    return response.decode('utf-8').strip()
                    
            except socket.timeout:
                raise TimeoutError(f"Command '{command}' timed out")
                
            except Exception as e:
                self.log.error(f"Error sending command '{command}': {e}")
                raise
                
    def _query_hardware_info(self):
        """Fragt Hardware-Informationen ab."""
        response = self._send_command("GET:INFO")
        return json.loads(response)
        
    def _create_constraints(self, hw_info):
        """Erstellt MicrowaveConstraints basierend auf Hardware-Informationen."""
        from qudi.interface.microwave_interface import MicrowaveConstraints
        from qudi.util.enums import SamplingOutputMode
        
        # Hardware-Limits extrahieren
        frequency_min = hw_info.get('frequency_min', 100e3)
        frequency_max = hw_info.get('frequency_max', 20e9)
        power_min = hw_info.get('power_min', -60)
        power_max = hw_info.get('power_max', 25)
        
        # Constraints erstellen
        constraints = MicrowaveConstraints(
            power_limits=(power_min, power_max),
            frequency_limits=(frequency_min, frequency_max),
            scan_size_limits=(2, 1001),
            sample_rate_limits=(0.1, 1000.0),
            scan_modes=(SamplingOutputMode.JUMP_LIST, SamplingOutputMode.EQUIDISTANT_SWEEP)
        )
        
        return constraints
        
    #
    # Interface-Methoden
    #
    
    @property
    def constraints(self):
        """Die Mikrowellen-Constraints für dieses Gerät."""
        return self._constraints
        
    @property
    def is_scanning(self):
        """Flag, ob ein Frequenzscan läuft."""
        with self._thread_lock:
            return self._is_scanning
            
    @property
    def cw_power(self):
        """Die aktuell konfigurierte CW-Leistung in dBm."""
        with self._thread_lock:
            return self._cw_power
            
    @property
    def cw_frequency(self):
        """Die aktuell konfigurierte CW-Frequenz in Hz."""
        with self._thread_lock:
            return self._cw_frequency
            
    @property
    def scan_power(self):
        """Die aktuell konfigurierte Scan-Leistung in dBm."""
        with self._thread_lock:
            return self._scan_power
            
    @property
    def scan_frequencies(self):
        """Die aktuell konfigurierten Scan-Frequenzen."""
        with self._thread_lock:
            return self._scan_frequencies
            
    @property
    def scan_mode(self):
        """Der aktuell konfigurierte Scan-Modus."""
        with self._thread_lock:
            return self._scan_mode
            
    @property
    def scan_sample_rate(self):
        """Die aktuell konfigurierte Scan-Abtastrate in Hz."""
        with self._thread_lock:
            return self._scan_sample_rate
            
    def off(self):
        """Schaltet jeden Mikrowellenausgang aus."""
        with self._thread_lock:
            if self.module_state() == 'idle':
                self.log.debug('Microwave output was not active')
                return
                
            self.log.debug("Stopping microwave output")
            response = self._send_command("RF:OFF")
            
            if response != "OK":
                self.log.warning(f"Unexpected response to off command: {response}")
                
            self._is_scanning = False
            self.module_state.unlock()
            
    def set_cw(self, frequency, power):
        """Konfiguriert den CW-Mikrowellenausgang."""
        with self._thread_lock:
            if self.module_state() != 'idle':
                raise RuntimeError('Unable to set CW: Microwave is active.')
                
            # Prüfe, ob Parameter innerhalb der Grenzen liegen
            if not (self._constraints.frequency_limits[0] <= frequency <= self._constraints.frequency_limits[1]):
                raise ValueError(f"Frequency {frequency} out of bounds ({self._constraints.frequency_limits})")
                
            if not (self._constraints.power_limits[0] <= power <= self._constraints.power_limits[1]):
                raise ValueError(f"Power {power} out of bounds ({self._constraints.power_limits})")
                
            # Parameter an Hardware senden
            command = f"RF:FREQ {frequency}"
            response1 = self._send_command(command)
            
            command = f"RF:POW {power}"
            response2 = self._send_command(command)
            
            if response1 != "OK" or response2 != "OK":
                raise RuntimeError(f"Error setting CW parameters: {response1}, {response2}")
                
            # Parameter speichern
            self._cw_frequency = frequency
            self._cw_power = power
            
            self.log.debug(f"Set CW: {frequency} Hz, {power} dBm")
            
    def cw_on(self):
        """Schaltet den CW-Mikrowellenausgang ein."""
        with self._thread_lock:
            if self.module_state() == 'idle':
                self.log.debug(f"Starting CW: {self._cw_frequency} Hz, {self._cw_power} dBm")
                
                # CW-Modus aktivieren
                command = "RF:CW:ON"
                response = self._send_command(command)
                
                if response != "OK":
                    raise RuntimeError(f"Error enabling CW mode: {response}")
                    
                self._is_scanning = False
                self.module_state.lock()
                
            elif self._is_scanning:
                raise RuntimeError('Unable to start CW: frequency scanning in progress.')
                
            else:
                self.log.debug('CW microwave output already running')
                
    def configure_scan(self, power, frequencies, mode, sample_rate):
        """Konfiguriert einen Frequenzscan."""
        with self._thread_lock:
            if self.module_state() != 'idle':
                raise RuntimeError('Unable to configure scan: Microwave is active.')
                
            # Prüfe, ob Parameter innerhalb der Grenzen liegen
            if not (self._constraints.power_limits[0] <= power <= self._constraints.power_limits[1]):
                raise ValueError(f"Scan power {power} out of bounds ({self._constraints.power_limits})")
                
            if not (self._constraints.sample_rate_limits[0] <= sample_rate <= self._constraints.sample_rate_limits[1]):
                raise ValueError(f"Sample rate {sample_rate} out of bounds ({self._constraints.sample_rate_limits})")
                
            # Frequenzliste vorbereiten
            if mode == SamplingOutputMode.JUMP_LIST:
                freq_list = frequencies
                if not all(self._constraints.frequency_limits[0] <= f <= self._constraints.frequency_limits[1] for f in freq_list):
                    raise ValueError(f"Some frequencies out of bounds ({self._constraints.frequency_limits})")
                    
                # Frequenzliste an Hardware senden
                freq_json = json.dumps(list(freq_list))
                command = f"SCAN:FREQ:LIST {freq_json}"
                response = self._send_command(command)
                
                if response != "OK":
                    raise RuntimeError(f"Error setting frequency list: {response}")
                    
            elif mode == SamplingOutputMode.EQUIDISTANT_SWEEP:
                start, stop, num_points = frequencies
                
                if not (self._constraints.frequency_limits[0] <= start <= self._constraints.frequency_limits[1]):
                    raise ValueError(f"Start frequency {start} out of bounds ({self._constraints.frequency_limits})")
                    
                if not (self._constraints.frequency_limits[0] <= stop <= self._constraints.frequency_limits[1]):
                    raise ValueError(f"Stop frequency {stop} out of bounds ({self._constraints.frequency_limits})")
                    
                # Sweep-Parameter an Hardware senden
                command = f"SCAN:FREQ:SWEEP {start} {stop} {num_points}"
                response = self._send_command(command)
                
                if response != "OK":
                    raise RuntimeError(f"Error setting frequency sweep: {response}")
                    
            else:
                raise ValueError(f"Unsupported scan mode: {mode}")
                
            # Weitere Scan-Parameter konfigurieren
            command = f"SCAN:POW {power}"
            response1 = self._send_command(command)
            
            command = f"SCAN:RATE {sample_rate}"
            response2 = self._send_command(command)
            
            command = f"SCAN:MODE {mode.name}"
            response3 = self._send_command(command)
            
            if not all(response == "OK" for response in [response1, response2, response3]):
                raise RuntimeError(f"Error configuring scan: {response1}, {response2}, {response3}")
                
            # Parameter speichern
            self._scan_power = power
            self._scan_mode = mode
            self._scan_sample_rate = sample_rate
            
            if mode == SamplingOutputMode.JUMP_LIST:
                self._scan_frequencies = np.array(frequencies)
            else:
                self._scan_frequencies = frequencies
                
            self.log.debug(f"Scan configured: Power={power}dBm, Mode={mode.name}, Rate={sample_rate}Hz")
            
    def start_scan(self):
        """Startet den konfigurierten Frequenzscan."""
        with self._thread_lock:
            if self.module_state() != 'idle':
                raise RuntimeError('Unable to start scan: Microwave is active.')
                
            if self._scan_frequencies is None:
                raise RuntimeError('Unable to start scan: No scan frequencies configured.')
                
            # Scan starten
            command = "SCAN:START"
            response = self._send_command(command)
            
            if response != "OK":
                raise RuntimeError(f"Error starting scan: {response}")
                
            self._is_scanning = True
            self.module_state.lock()
            
            self.log.debug(f"Started frequency scan in {self._scan_mode.name} mode")
            
    def reset_scan(self):
        """Setzt den Scan zurück zur Startfrequenz."""
        with self._thread_lock:
            if not self._is_scanning:
                self.log.warning("Cannot reset scan: Not scanning")
                return
                
            # Scan zurücksetzen
            command = "SCAN:RESET"
            response = self._send_command(command)
            
            if response != "OK":
                raise RuntimeError(f"Error resetting scan: {response}")
                
            self.log.debug("Frequency scan reset")
```

### 7.2 Qudi-Konfiguration für Netzwerk-Quantenhardware

```yaml
# Beispiel-Konfiguration für Qudi mit Netzwerk-Quantenhardware

global:
    startup:
        gui: true

hardware:
    network_quantum:
        module.Class: 'quantum.network_quantum_hardware.NetworkQuantumHardware'
        ip_address: '10.0.0.2'
        control_port: 5555
        data_port: 5556
        status_port: 5557
        connection_timeout: 5.0

    fast_counter:
        module.Class: 'quantum.network_quantum_counter.NetworkQuantumCounter'
        ip_address: '10.0.0.2'
        port: 5560

    scanner:
        module.Class: 'quantum.network_quantum_scanner.NetworkQuantumScanner'
        ip_address: '10.0.0.2'
        port: 5565

logic:
    odmr_logic:
        module.Class: 'odmr_logic.ODMRLogic'
        connect:
            microwave: network_quantum
            fast_counter: fast_counter

    scanner_logic:
        module.Class: 'scanning_probe_logic.ScanningProbeLogic'
        connect:
            scanner: scanner
            data_scanner: scanner

gui:
    odmr_gui:
        module.Class: 'odmr.odmrgui.ODMRGui'
        connect:
            odmrlogic: odmr_logic

    scanner_gui:
        module.Class: 'scanning.scannergui.ScannerGui'
        connect:
            scannerlogic: scanner_logic
```

## 8. Netzwerk-Diagnose und Fehlerbehebung

### 8.1 Verbindungstest und Diagnose-Werkzeuge

```python
import socket
import subprocess
import platform
import time

class QuantumNetworkDiagnostics:
    def __init__(self, ip_address, ports):
        self.ip_address = ip_address
        self.ports = ports
        
    def run_ping_test(self, count=4):
        """Führt einen Ping-Test zur Quantenhardware durch."""
        param = '-n' if platform.system().lower() == 'windows' else '-c'
        command = ['ping', param, str(count), self.ip_address]
        
        try:
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            
            if result.returncode == 0:
                print(f"Ping to {self.ip_address} successful")
                
                # Extrahiere durchschnittliche Latenz
                if platform.system().lower() == 'windows':
                    # Windows-Format parsen
                    for line in result.stdout.split('\n'):
                        if "Average" in line:
                            latency = line.split('=')[1].strip().split('ms')[0].strip()
                            print(f"Average latency: {latency} ms")
                else:
                    # Unix-Format parsen
                    for line in result.stdout.split('\n'):
                        if "min/avg/max" in line:
                            avg = line.split('=')[1].split('/')[1].strip()
                            print(f"Average latency: {avg} ms")
                            
                return True
            else:
                print(f"Ping to {self.ip_address} failed")
                print(result.stderr)
                return False
                
        except subprocess.TimeoutExpired:
            print(f"Ping to {self.ip_address} timed out")
            return False
            
        except Exception as e:
            print(f"Error during ping test: {e}")
            return False
            
    def check_port_connectivity(self):
        """Überprüft die Verbindbarkeit aller Ports."""
        results = {}
        
        for port_name, port in self.ports.items():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            
            start_time = time.time()
            result = sock.connect_ex((self.ip_address, port))
            connection_time = time.time() - start_time
            
            sock.close()
            
            if result == 0:
                print(f"Port {port} ({port_name}) is open, connection time: {connection_time*1000:.1f} ms")
                results[port_name] = {
                    'status': 'open',
                    'connection_time': connection_time
                }
            else:
                print(f"Port {port} ({port_name}) is closed or filtered, error code: {result}")
                results[port_name] = {
                    'status': 'closed',
                    'error_code': result
                }
                
        return results
        
    def measure_throughput(self, port, data_size=1024*1024, timeout=10.0):
        """Misst den Durchsatz auf einem bestimmten Port."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((self.ip_address, port))
            
            # Sende Durchsatztest-Kommando
            command = f"DIAG:THROUGHPUT {data_size}\n"
            sock.sendall(command.encode('utf-8'))
            
            # Empfange Daten und messe Zeit
            received = 0
            start_time = time.time()
            
            while received < data_size:
                chunk = sock.recv(min(64*1024, data_size - received))
                if not chunk:
                    break
                received += len(chunk)
                
            end_time = time.time()
            
            if received == data_size:
                duration = end_time - start_time
                throughput = (data_size / duration) / (1024 * 1024)  # MB/s
                
                print(f"Throughput test on port {port}: {throughput:.2f} MB/s")
                return throughput
            else:
                print(f"Throughput test failed: Received only {received} of {data_size} bytes")
                return None
                
        except Exception as e:
            print(f"Throughput test error: {e}")
            return None
        finally:
            sock.close()
            
    def run_diagnostics(self):
        """Führt alle Diagnosetests durch."""
        print(f"=== Quantum Network Diagnostics for {self.ip_address} ===")
        
        # Ping-Test
        ping_success = self.run_ping_test()
        
        # Port-Konnektivität
        port_results = self.check_port_connectivity()
        
        # Durchsatz-Tests für offene Ports
        throughput_results = {}
        for port_name, result in port_results.items():
            if result['status'] == 'open':
                port = self.ports[port_name]
                if port_name == 'data':  # Nur Datenport testen
                    throughput = self.measure_throughput(port)
                    if throughput is not None:
                        throughput_results[port_name] = throughput
                        
        # Zusammenfassung
        print("\n=== Diagnostic Summary ===")
        print(f"Ping status: {'Success' if ping_success else 'Failed'}")
        
        print("\nPort connectivity:")
        for port_name, result in port_results.items():
            status = result['status']
            if status == 'open':
                time_ms = result.get('connection_time', 0) * 1000
                print(f"  - {port_name}: {status} ({time_ms:.1f} ms)")
            else:
                error = result.get('error_code', 'unknown')
                print(f"  - {port_name}: {status} (error {error})")
                
        if throughput_results:
            print("\nThroughput tests:")
            for port_name, throughput in throughput_results.items():
                print(f"  - {port_name}: {throughput:.2f} MB/s")
                
        print("\nDiagnostic complete")
        
# Verwendungsbeispiel
if __name__ == "__main__":
    # Ports für die Quantenhardware
    ports = {
        'control': 5555,
        'data': 5556,
        'status': 5557,
        'counter': 5560,
        'scanner': 5565
    }
    
    diagnostics = QuantumNetworkDiagnostics('10.0.0.2', ports)
    diagnostics.run_diagnostics()
```

### 8.2 Wireshark-Capture-Filter für Quantenhardware-Kommunikation

```
# Capture-Filter für alle Kommunikation mit der Quantenhardware
host 10.0.0.2

# Capture-Filter für spezifische Ports
host 10.0.0.2 and (port 5555 or port 5556 or port 5557)

# Display-Filter für Kontrollkommandos
tcp.port == 5555 and tcp.payload contains "RF:"

# Display-Filter für ODMR-Daten
tcp.port == 5556
```

## 9. Sicherheitsaspekte

### 9.1 Beispiel für Firewall-Konfiguration

```bash
# IP-Tables-Regeln für Quantenhardware-Kommunikation

# Nur spezifische Quell-IPs zulassen (Qudi-System)
iptables -A INPUT -p tcp --dport 5555 -s 192.168.1.10 -j ACCEPT
iptables -A INPUT -p tcp --dport 5556 -s 192.168.1.10 -j ACCEPT
iptables -A INPUT -p tcp --dport 5557 -s 192.168.1.10 -j ACCEPT
iptables -A INPUT -p tcp --dport 5560 -s 192.168.1.10 -j ACCEPT
iptables -A INPUT -p tcp --dport 5565 -s 192.168.1.10 -j ACCEPT

# Alle anderen Verbindungen zu diesen Ports ablehnen
iptables -A INPUT -p tcp --dport 5555 -j DROP
iptables -A INPUT -p tcp --dport 5556 -j DROP
iptables -A INPUT -p tcp --dport 5557 -j DROP
iptables -A INPUT -p tcp --dport 5560 -j DROP
iptables -A INPUT -p tcp --dport 5565 -j DROP
```

## 10. Zusammenfassung der Kommunikationswege

| Schnittstelle | Protokoll | Port | Datenformat | Typische Latenz | Bandbreite |
|---------------|-----------|------|-------------|-----------------|------------|
| Kontrollkanal | TCP/IP    | 5555 | Text/JSON   | 1-10 ms         | < 1 MB/s   |
| Datenkanal    | TCP/IP    | 5556 | Binär       | 0.5-5 ms        | 10-100 MB/s |
| Statuskanal   | TCP/IP    | 5557 | JSON        | 1-10 ms         | < 1 MB/s   |
| ZeroMQ-REQ/REP | TCP/IP   | 5558 | MsgPack     | 0.5-5 ms        | 1-10 MB/s  |
| ZeroMQ-PUB/SUB | TCP/IP   | 5559 | MsgPack     | 0.5-5 ms        | 10-50 MB/s |
| gRPC          | TCP/IP    | 50051| ProtoBuf    | 1-10 ms         | 5-50 MB/s  |

### Fazit:

Diese umfassende technische Dokumentation beschreibt alle notwendigen Kommunikationsprotokolle, Datenformate und Netzwerkinteraktionen, um einen netzwerkbasierten Quantencomputer vollständig in das Qudi-Framework zu integrieren. Die beschriebenen Schnittstellen unterstützen verschiedene Kommunikationsmodelle, von einfachen Socket-Verbindungen bis hin zu modernen RPC-Frameworks wie gRPC, und bieten damit sowohl Flexibilität als auch Leistungsfähigkeit für die anspruchsvollen Anforderungen der Quantenhardware-Steuerung.

Die hier dargestellten Best Practices und Code-Beispiele können als Grundlage für die Implementierung eigener Quantenhardware-Schnittstellen dienen und gewährleisten eine zuverlässige, sichere und effiziente Kommunikation zwischen der Qudi-Software und der Quantenhardware.