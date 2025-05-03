# Web-UI-Spezifikation für den NV-Zentren Simulator

## 1. Einführung

Diese Spezifikation definiert die Web-Benutzeroberfläche (Web-UI) für den NV-Zentren Simulator. Das Web-UI dient ausschließlich zur Konfiguration, Überwachung und Steuerung des Simulators selbst und bildet nicht die in Qudi sichtbaren Experimentoberflächen nach. Es ermöglicht den Zugriff auf simulatorspezifische Einstellungen und Parameter, die normalerweise nicht über die Qudi-Schnittstelle zugänglich sind.

## 2. Anforderungen

### 2.1 Funktionale Anforderungen

1. **Simulator-Konfiguration**: Anpassung aller Simulationsparameter:
   - Physikalische Parameter (z.B. D-Konstante, E-Strain, Hyperfeine Kopplung)
   - Kohärenzeigenschaften (T1, T2*, T2)
   - Netzwerkeinstellungen (Ports, Protokolle)
   - Verzögerungs- und Timing-Einstellungen

2. **NV-Zentren-Verwaltung**:
   - Hinzufügen/Entfernen virtueller NV-Zentren
   - Positionierung der NV-Zentren im simulierten Raum
   - Konfiguration individueller Eigenschaften für jedes NV-Zentrum

3. **Fehlersimulation**:
   - Konfiguration von Rauschprofilen
   - Definition von simulierten Hardwarefehlern und -ausfällen
   - Einstellung realistischer Verzögerungen für verschiedene Operationen

4. **Monitoring**:
   - Echtzeitüberwachung des Simulatorzustands
   - Anzeige aktiver Verbindungen und Operationen
   - Protokollierung aller Aktivitäten

5. **Diagnose**:
   - Analyse der Netzwerkverbindungen
   - Fehlerdiagnose und -reporting
   - Performance-Monitoring

### 2.2 Nicht-funktionale Anforderungen

1. **Benutzbarkeit**:
   - Intuitive Benutzeroberfläche
   - Responsive Design für verschiedene Geräte
   - Einheitliche Designsprache

2. **Performance**:
   - Minimale Latenz bei Interaktionen
   - Effiziente Datenübertragung
   - Geringe Serverlast

3. **Sicherheit**:
   - Authentifizierungsmechanismen
   - Zugriffskontrollen
   - Sichere Datenübertragung

4. **Zuverlässigkeit**:
   - Stabile Funktionalität
   - Fehlertolerante Implementierung
   - Automatische Wiederverbindung

## 3. Architektur

### 3.1 Gesamtarchitektur

```
┌────────────────┐     ┌─────────────────┐     ┌────────────────┐
│                │     │                 │     │                │
│  Web-Browser   │◄───►│  Web UI Server  │◄───►│  NV-Simulator  │
│                │     │                 │     │                │
└────────────────┘     └─────────────────┘     └────────────────┘
        HTTP                  API               Direkter Zugriff
```

### 3.2 Technologiestack

1. **Backend**:
   - Flask (Python-Webframework)
   - Flask-RESTful für API-Endpunkte
   - Flask-SocketIO für Echtzeit-Updates
   - SQLite für Konfigurationsspeicherung

2. **Frontend**:
   - HTML5, CSS3, JavaScript
   - Bootstrap für responsive Layouts
   - Vue.js für reaktive Komponenten
   - Chart.js für Visualisierungen

### 3.3 Kommunikationsprotokolle

1. **Browser ↔ Web UI Server**:
   - HTTP/HTTPS für Standardanfragen
   - WebSockets für Echtzeit-Updates
   - REST API für Datenzugriff

2. **Web UI Server ↔ NV-Simulator**:
   - Direkter Zugriff auf Simulator-Objekte
   - Interprozesskommunikation

## 4. Benutzeroberfläche

### 4.1 Hauptkomponenten

1. **Dashboard**: Überblick über den Simulator-Status
2. **Konfiguration**: Parameter und Einstellungen
3. **NV-Management**: Verwaltung virtueller NV-Zentren
4. **Fehlersimulation**: Steuerung von Rauschen und Fehlern
5. **Monitoring**: Echtzeitüberwachung
6. **Logs**: Ereignisprotokolle

### 4.2 Seitenstruktur

```
├── Dashboard
├── Konfiguration
│   ├── Physikalisches Modell
│   ├── Netzwerk
│   ├── Timing & Verzögerungen
│   └── Speichern/Laden
├── NV-Zentren
│   ├── Übersicht
│   ├── Hinzufügen/Bearbeiten
│   └── Positionierung
├── Fehlersimulation
│   ├── Rauschprofile
│   ├── Hardware-Fehler
│   └── Timing-Variationen
├── Monitoring
│   ├── Status
│   ├── Verbindungen
│   └── Performanz
└── Logs
    ├── System
    ├── Anfragen
    └── Fehler
```

### 4.3 Mockups

#### 4.3.1 Dashboard

```
+-------------------------------------------------------+
| NV-Zentren Simulator                               [X] |
+---------------+---------------------------------------+
| Dashboard     | Simulator-Status: AKTIV               |
| Konfiguration |                                       |
| NV-Zentren    | +-----------------------------------+ |
| Fehler-       | | Aktive Verbindungen:           2 | |
| simulation    | +-----------------------------------+ |
| Monitoring    |                                       |
| Logs          | +-----------------------------------+ |
|               | | Aktive NV-Zentren:              3 | |
|               | +-----------------------------------+ |
|               |                                       |
|               | +-----------------------------------+ |
|               | | CPU-Auslastung:               25% | |
|               | +-----------------------------------+ |
|               |                                       |
|               | +-----------------------------------+ |
|               | | Letzter Fehler:               N/A | |
|               | +-----------------------------------+ |
|               |                                       |
|               | [Neustart] [Einstellungen] [Hilfe]   |
+---------------+---------------------------------------+
```

#### 4.3.2 Konfiguration - Physikalisches Modell

```
+-------------------------------------------------------+
| NV-Zentren Simulator - Konfiguration              [X] |
+---------------+---------------------------------------+
| Dashboard     | Physikalisches Modell                 |
| Konfiguration |                                       |
| > Physik      | Zero-Field Splitting (D):             |
| > Netzwerk    | [      2.87      ] GHz                |
| > Timing      |                                       |
| > Speichern   | Strain (E):                           |
| NV-Zentren    | [      0.005     ] GHz                |
| Fehler-       |                                       |
| simulation    | Hyperfeine Kopplung:                  |
| Monitoring    | [      2.2       ] MHz                |
| Logs          |                                       |
|               | Stickstoff-Isotop:                    |
|               | (O) N14   ( ) N15                     |
|               |                                       |
|               | Kohärenzzeiten:                       |
|               | T1:  [      2.0      ] ms             |
|               | T2*: [      3.0      ] µs             |
|               | T2:  [     300.0     ] µs             |
|               |                                       |
|               | [Zurücksetzen] [Anwenden]             |
+---------------+---------------------------------------+
```

#### 4.3.3 NV-Zentren Management

```
+-------------------------------------------------------+
| NV-Zentren Simulator - NV-Zentren                 [X] |
+---------------+---------------------------------------+
| Dashboard     | NV-Zentren Verwaltung                 |
| Konfiguration |                                       |
| NV-Zentren    | +-----------------------------------+ |
| > Übersicht   | | ID | Position (µm)  | Kontrast   | |
| > Hinzufügen  | +-----------------------------------+ |
| > Position    | | 1  | 2.5, 2.5, 5.0  | 18%        | |
| Fehler-       | | 2  | 7.5, 5.0, 5.0  | 15%        | |
| simulation    | | 3  | 15.0, 15.0, 5.0| 20%        | |
| Monitoring    | +-----------------------------------+ |
| Logs          |                                       |
|               | [Hinzufügen] [Bearbeiten] [Entfernen] |
|               |                                       |
|               | +-----------------------------------+ |
|               | |                                   | |
|               | |      [Konfokale Ansicht]         | |
|               | |                                   | |
|               | +-----------------------------------+ |
|               |                                       |
+---------------+---------------------------------------+
```

#### 4.3.4 Fehlersimulation

```
+-------------------------------------------------------+
| NV-Zentren Simulator - Fehlersimulation           [X] |
+---------------+---------------------------------------+
| Dashboard     | Rauschprofile und Fehler              |
| Konfiguration |                                       |
| NV-Zentren    | Rauschprofile:                        |
| Fehler-       | Magnetisches Rauschen: [####------] 40%|
| simulation    | Positions-Jitter:     [##--------] 20%|
| > Rauschen    | Laser-Intensität:     [###-------] 30%|
| > Fehler      | MW-Phasenrauschen:    [#---------] 10%|
| > Timing      |                                       |
| Monitoring    | Hardware-Fehler:                      |
| Logs          | [ ] Zufällige Verbindungsabbrüche (5%)|
|               | [ ] Mikrowellen-Aussetzer       (2%)  |
|               | [ ] Counter-Überläufe           (1%)  |
|               | [X] Laserleistungs-Schwankungen (10%) |
|               |                                       |
|               | Verzögerungsvariationen:              |
|               | Mikrowell-Schaltzeit: [####------] 40%|
|               | Laser-Schaltzeit:     [##--------] 20%|
|               |                                       |
|               | [Zurücksetzen] [Anwenden]             |
+---------------+---------------------------------------+
```

## 5. API-Endpunkte

Die Web-UI-Server-API bietet folgende REST-Endpunkte:

### 5.1 Konfigurationsendpunkte

| Endpunkt | Methode | Beschreibung | Parameter |
|----------|---------|--------------|-----------|
| `/api/config` | GET | Gibt die aktuelle Konfiguration zurück | - |
| `/api/config` | POST | Aktualisiert die Konfiguration | JSON-Konfigurationsobjekt |
| `/api/config/physics` | GET | Gibt physikalische Parameter zurück | - |
| `/api/config/physics` | POST | Aktualisiert physikalische Parameter | JSON-Objekt mit Parametern |
| `/api/config/network` | GET | Gibt Netzwerkeinstellungen zurück | - |
| `/api/config/network` | POST | Aktualisiert Netzwerkeinstellungen | JSON-Objekt mit Einstellungen |

### 5.2 NV-Zentren-Endpunkte

| Endpunkt | Methode | Beschreibung | Parameter |
|----------|---------|--------------|-----------|
| `/api/nv` | GET | Listet alle NV-Zentren auf | - |
| `/api/nv` | POST | Fügt ein neues NV-Zentrum hinzu | JSON-Objekt mit NV-Eigenschaften |
| `/api/nv/{id}` | GET | Gibt Details zu einem NV-Zentrum zurück | ID des NV-Zentrums |
| `/api/nv/{id}` | PUT | Aktualisiert ein NV-Zentrum | ID und JSON-Objekt mit Eigenschaften |
| `/api/nv/{id}` | DELETE | Entfernt ein NV-Zentrum | ID des NV-Zentrums |

### 5.3 Fehlersimulations-Endpunkte

| Endpunkt | Methode | Beschreibung | Parameter |
|----------|---------|--------------|-----------|
| `/api/noise` | GET | Gibt aktuelle Rauschprofile zurück | - |
| `/api/noise` | POST | Aktualisiert Rauschprofile | JSON-Objekt mit Rauschparametern |
| `/api/faults` | GET | Gibt aktuelle Fehlereinstellungen zurück | - |
| `/api/faults` | POST | Aktualisiert Fehlereinstellungen | JSON-Objekt mit Fehlerparametern |
| `/api/timing` | GET | Gibt Timing-Variationen zurück | - |
| `/api/timing` | POST | Aktualisiert Timing-Variationen | JSON-Objekt mit Timing-Parametern |

### 5.4 Monitoring-Endpunkte

| Endpunkt | Methode | Beschreibung | Parameter |
|----------|---------|--------------|-----------|
| `/api/status` | GET | Gibt den aktuellen Simulatorstatus zurück | - |
| `/api/connections` | GET | Listet aktive Verbindungen auf | - |
| `/api/performance` | GET | Gibt Performancedaten zurück | - |
| `/api/logs` | GET | Gibt Protokolleinträge zurück | type (optional), count (optional) |
| `/api/simulator/restart` | POST | Startet den Simulator neu | - |

## 6. WebSocket-Ereignisse

Für Echtzeitkommunikation verwendet das Web-UI WebSockets, die folgende Ereignisse senden:

| Ereignis | Beschreibung | Datenstruktur |
|----------|--------------|---------------|
| `status_update` | Aktualisierung des Simulatorstatus | `{running: bool, modules: {}}` |
| `connection_changed` | Neue Verbindung oder Verbindungsverlust | `{action: "connected"/"disconnected", client_id: string}` |
| `error_occurred` | Fehler im Simulator | `{level: "warning"/"error", message: string, module: string}` |
| `performance_update` | Performancedaten | `{cpu: float, memory: float, latency: float}` |
| `nv_change` | Änderung der NV-Zentren | `{action: "added"/"modified"/"removed", nv_id: string}` |

## 7. Datenmodelle

### 7.1 Konfigurationsobjekt

```json
{
  "physics": {
    "d_constant": 2.87e9,
    "e_strain": 5.0e6,
    "hyperfine_coupling": 2.2e6,
    "nitrogen_isotope": "N14",
    "coherence": {
      "t1_time": 2.0e-3,
      "t2_star_time": 3.0e-6,
      "t2_time": 300.0e-6
    }
  },
  "network": {
    "tcp_enabled": true,
    "tcp_port": 5555,
    "zmq_enabled": false,
    "zmq_pub_port": 5556,
    "zmq_sub_port": 5557,
    "grpc_enabled": false,
    "grpc_port": 50051
  },
  "timing": {
    "realistic_delays": true,
    "microwave_delay": 50.0e-3,
    "laser_delay": 10.0e-3,
    "counter_delay": 5.0e-3
  }
}
```

### 7.2 NV-Zentren-Objekt

```json
{
  "id": "nv1",
  "position": [2.5e-6, 2.5e-6, 5.0e-6],
  "properties": {
    "contrast": 0.18,
    "t2_star_time": 2.8e-6,
    "strain": 4.5e6
  }
}
```

### 7.3 Rausch- und Fehlerobjekt

```json
{
  "noise": {
    "magnetic_noise": 0.4,
    "position_jitter": 0.2,
    "laser_intensity_noise": 0.3,
    "microwave_phase_noise": 0.1
  },
  "faults": {
    "connection_drops": {
      "enabled": false,
      "probability": 0.05
    },
    "microwave_glitches": {
      "enabled": false,
      "probability": 0.02
    },
    "counter_overflows": {
      "enabled": false,
      "probability": 0.01
    },
    "laser_fluctuations": {
      "enabled": true,
      "probability": 0.1
    }
  },
  "timing_variations": {
    "microwave_switching": 0.4,
    "laser_switching": 0.2
  }
}
```

## 8. Implementierungsdetails

### 8.1 Backend-Implementierung

Der Web-UI-Server wird mit Flask implementiert:

```python
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
import threading
import time

app = Flask(__name__)
socketio = SocketIO(app)

# Zugriff auf den Simulator-Zustand
from nv_simulator.core.state import QuantumSimulatorState
simulator_state = QuantumSimulatorState()

@app.route('/')
def index():
    """Rendert die Hauptseite."""
    return render_template('index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    """Gibt die aktuelle Konfiguration zurück."""
    return jsonify(simulator_state.config)

@app.route('/api/config', methods=['POST'])
def update_config():
    """Aktualisiert die Konfiguration."""
    new_config = request.json
    
    try:
        # Konfiguration validieren
        # ...
        
        # Konfiguration aktualisieren
        simulator_state.update_config(new_config)
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/api/nv', methods=['GET'])
def get_nv_centers():
    """Listet alle NV-Zentren auf."""
    nv_centers = []
    
    for position, nv_model in simulator_state.nv_models.items():
        nv_centers.append({
            'id': f"nv{len(nv_centers)+1}",
            'position': position,
            'properties': {
                'contrast': nv_model.contrast,
                't2_star_time': nv_model.t2_star_time
            }
        })
    
    return jsonify(nv_centers)

@app.route('/api/nv', methods=['POST'])
def add_nv_center():
    """Fügt ein neues NV-Zentrum hinzu."""
    data = request.json
    
    try:
        position = tuple(data['position'])
        properties = data.get('properties', {})
        
        simulator_state.add_nv_center(position, properties)
        
        socketio.emit('nv_change', {
            'action': 'added',
            'nv_id': f"nv{len(simulator_state.nv_models)}"
        })
        
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

# Weitere API-Routen...

def status_update_thread():
    """Thread für regelmäßige Statusaktualisierungen über WebSockets."""
    while True:
        status = {
            'running': simulator_state.is_running(),
            'modules': {
                'microwave': simulator_state.microwave_state,
                'laser': simulator_state.laser_state,
                'scanner': simulator_state.scanner_position,
                'counter': simulator_state.counter_is_running
            }
        }
        
        socketio.emit('status_update', status)
        time.sleep(1.0)

# Starte den Statusaktualisierungs-Thread
threading.Thread(target=status_update_thread, daemon=True).start()

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8080, debug=True)
```

### 8.2 Frontend-Implementierung

Die Frontend-Implementierung verwendet Vue.js für reaktive Komponenten:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NV-Zentren Simulator</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <script src="https://cdn.jsdelivr.net/npm/vue@2.6.12/dist/vue.js"></script>
    <script src="https://cdn.socket.io/socket.io-3.0.1.min.js"></script>
</head>
<body>
    <div id="app" class="container-fluid">
        <div class="row">
            <!-- Sidebar -->
            <div class="col-md-3 bg-light p-0">
                <div class="list-group list-group-flush">
                    <a href="#" 
                       @click="currentPage = 'dashboard'" 
                       :class="['list-group-item', 'list-group-item-action', currentPage === 'dashboard' ? 'active' : '']">
                        Dashboard
                    </a>
                    <a href="#" 
                       @click="currentPage = 'config'" 
                       :class="['list-group-item', 'list-group-item-action', currentPage === 'config' ? 'active' : '']">
                        Konfiguration
                    </a>
                    <a href="#" 
                       @click="currentPage = 'nv'" 
                       :class="['list-group-item', 'list-group-item-action', currentPage === 'nv' ? 'active' : '']">
                        NV-Zentren
                    </a>
                    <a href="#" 
                       @click="currentPage = 'faults'" 
                       :class="['list-group-item', 'list-group-item-action', currentPage === 'faults' ? 'active' : '']">
                        Fehlersimulation
                    </a>
                    <a href="#" 
                       @click="currentPage = 'monitoring'" 
                       :class="['list-group-item', 'list-group-item-action', currentPage === 'monitoring' ? 'active' : '']">
                        Monitoring
                    </a>
                    <a href="#" 
                       @click="currentPage = 'logs'" 
                       :class="['list-group-item', 'list-group-item-action', currentPage === 'logs' ? 'active' : '']">
                        Logs
                    </a>
                </div>
            </div>
            
            <!-- Main content -->
            <div class="col-md-9 p-3">
                <!-- Dashboard -->
                <div v-if="currentPage === 'dashboard'">
                    <h2>Dashboard</h2>
                    <div class="alert" :class="simulatorRunning ? 'alert-success' : 'alert-danger'">
                        Simulator-Status: {{ simulatorRunning ? 'AKTIV' : 'INAKTIV' }}
                    </div>
                    
                    <div class="card mb-3">
                        <div class="card-body">
                            <h5 class="card-title">Aktive Verbindungen</h5>
                            <h3>{{ connectionCount }}</h3>
                        </div>
                    </div>
                    
                    <div class="card mb-3">
                        <div class="card-body">
                            <h5 class="card-title">Aktive NV-Zentren</h5>
                            <h3>{{ nvCount }}</h3>
                        </div>
                    </div>
                    
                    <div class="row">
                        <div class="col-md-6">
                            <button class="btn btn-primary" @click="restartSimulator">Neustart</button>
                        </div>
                        <div class="col-md-6 text-right">
                            <button class="btn btn-secondary" @click="showSettings">Einstellungen</button>
                            <button class="btn btn-info ml-2" @click="showHelp">Hilfe</button>
                        </div>
                    </div>
                </div>
                
                <!-- Konfiguration -->
                <div v-if="currentPage === 'config'">
                    <h2>Physikalisches Modell</h2>
                    <form @submit.prevent="savePhysicsConfig">
                        <div class="form-group">
                            <label>Zero-Field Splitting (D):</label>
                            <div class="input-group">
                                <input type="number" v-model="config.physics.d_constant" 
                                       class="form-control" step="0.01" />
                                <div class="input-group-append">
                                    <span class="input-group-text">GHz</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="form-group">
                            <label>Strain (E):</label>
                            <div class="input-group">
                                <input type="number" v-model="config.physics.e_strain" 
                                       class="form-control" step="0.001" />
                                <div class="input-group-append">
                                    <span class="input-group-text">GHz</span>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Weitere Felder... -->
                        
                        <div class="form-group">
                            <button type="button" class="btn btn-secondary" @click="resetPhysicsConfig">
                                Zurücksetzen
                            </button>
                            <button type="submit" class="btn btn-primary ml-2">
                                Anwenden
                            </button>
                        </div>
                    </form>
                </div>
                
                <!-- NV-Zentren -->
                <div v-if="currentPage === 'nv'">
                    <h2>NV-Zentren Verwaltung</h2>
                    <div class="table-responsive">
                        <table class="table table-bordered">
                            <thead>
                                <tr>
                                    <th>ID</th>
                                    <th>Position (µm)</th>
                                    <th>Kontrast</th>
                                    <th>Aktionen</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr v-for="(nv, index) in nvCenters" :key="nv.id">
                                    <td>{{ nv.id }}</td>
                                    <td>{{ formatPosition(nv.position) }}</td>
                                    <td>{{ (nv.properties.contrast * 100).toFixed(0) }}%</td>
                                    <td>
                                        <button class="btn btn-sm btn-info" @click="editNV(nv.id)">
                                            Bearbeiten
                                        </button>
                                        <button class="btn btn-sm btn-danger ml-1" @click="removeNV(nv.id)">
                                            Entfernen
                                        </button>
                                    </td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                    
                    <button class="btn btn-success" @click="showAddNVDialog">
                        Hinzufügen
                    </button>
                </div>
                
                <!-- Weitere Seiten... -->
            </div>
        </div>
    </div>
    
    <script>
        new Vue({
            el: '#app',
            data: {
                currentPage: 'dashboard',
                simulatorRunning: false,
                connectionCount: 0,
                nvCount: 0,
                nvCenters: [],
                config: {
                    physics: {
                        d_constant: 2.87,
                        e_strain: 0.005,
                        hyperfine_coupling: 2.2,
                        nitrogen_isotope: 'N14',
                        coherence: {
                            t1_time: 2.0,
                            t2_star_time: 3.0,
                            t2_time: 300.0
                        }
                    },
                    network: {},
                    timing: {}
                },
                logs: []
            },
            mounted() {
                this.initSocket();
                this.fetchData();
            },
            methods: {
                initSocket() {
                    // Socket.io initialisieren
                    this.socket = io();
                    
                    // Statusaktualisierungen
                    this.socket.on('status_update', (data) => {
                        this.simulatorRunning = data.running;
                        // Module-Status aktualisieren
                    });
                    
                    // Verbindungsaktualisierungen
                    this.socket.on('connection_changed', (data) => {
                        // Verbindungsliste aktualisieren
                        this.fetchConnections();
                    });
                    
                    // NV-Zentren-Aktualisierungen
                    this.socket.on('nv_change', (data) => {
                        // NV-Zentren aktualisieren
                        this.fetchNVCenters();
                    });
                },
                fetchData() {
                    // Konfiguration laden
                    fetch('/api/config')
                        .then(response => response.json())
                        .then(data => {
                            this.config = data;
                        });
                    
                    // NV-Zentren laden
                    this.fetchNVCenters();
                    
                    // Verbindungen laden
                    this.fetchConnections();
                },
                fetchNVCenters() {
                    fetch('/api/nv')
                        .then(response => response.json())
                        .then(data => {
                            this.nvCenters = data;
                            this.nvCount = data.length;
                        });
                },
                fetchConnections() {
                    fetch('/api/connections')
                        .then(response => response.json())
                        .then(data => {
                            this.connectionCount = data.length;
                        });
                },
                restartSimulator() {
                    if (confirm('Sind Sie sicher, dass Sie den Simulator neu starten möchten?')) {
                        fetch('/api/simulator/restart', {
                            method: 'POST'
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.status === 'success') {
                                alert('Simulator wurde neu gestartet.');
                            } else {
                                alert('Fehler beim Neustart: ' + data.message);
                            }
                        });
                    }
                },
                savePhysicsConfig() {
                    fetch('/api/config/physics', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify(this.config.physics)
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.status === 'success') {
                            alert('Physikalische Parameter wurden aktualisiert.');
                        } else {
                            alert('Fehler beim Speichern: ' + data.message);
                        }
                    });
                },
                resetPhysicsConfig() {
                    if (confirm('Sind Sie sicher, dass Sie alle Änderungen verwerfen möchten?')) {
                        this.fetchData();
                    }
                },
                formatPosition(position) {
                    return position
                        .map(p => (p * 1e6).toFixed(1))
                        .join(', ');
                },
                showAddNVDialog() {
                    // Dialog zum Hinzufügen eines NV-Zentrums anzeigen
                    // ...
                },
                editNV(id) {
                    // Dialog zum Bearbeiten eines NV-Zentrums anzeigen
                    // ...
                },
                removeNV(id) {
                    if (confirm(`Sind Sie sicher, dass Sie das NV-Zentrum ${id} entfernen möchten?`)) {
                        fetch(`/api/nv/${id}`, {
                            method: 'DELETE'
                        })
                        .then(response => response.json())
                        .then(data => {
                            if (data.status === 'success') {
                                this.fetchNVCenters();
                            } else {
                                alert('Fehler beim Entfernen: ' + data.message);
                            }
                        });
                    }
                }
                // Weitere Methoden...
            }
        });
    </script>
</body>
</html>
```

## 9. Sicherheitsaspekte

Die Web-UI implementiert folgende Sicherheitsmaßnahmen:

1. **Authentifizierung**:
   - Optionale Basis-Authentifizierung mit Benutzername/Passwort
   - Token-basierte Authentifizierung für API-Zugriffe

2. **Autorisierung**:
   - Rollenbasierte Zugriffskontrollen (Admin, Benutzer, Beobachter)
   - Einschränkung kritischer Operationen auf Admin-Rechte

3. **Datensicherheit**:
   - HTTPS-Verschlüsselung für alle Übertragungen
   - Validierung aller Eingabeparameter
   - Schutz vor Cross-Site-Scripting (XSS) und Cross-Site-Request-Forgery (CSRF)

4. **Netzwerksicherheit**:
   - Beschränkung des Web-UI-Zugriffs auf lokale Netzwerke
   - Firewallregeln zur Beschränkung des Zugriffs

## 10. Testplan

Der Testplan für die Web-UI umfasst:

1. **Komponententests**:
   - API-Endpunkt-Tests
   - UI-Komponententests

2. **Integrationstests**:
   - Tests der Interaktion zwischen Web-UI und Simulator
   - WebSocket-Kommunikationstests

3. **Systemtests**:
   - End-to-End-Tests für komplette Arbeitsabläufe
   - Browser-Kompatibilitätstests

4. **Sicherheitstests**:
   - Penetrationstests
   - Zugriffskontrollen-Tests

## 11. Zusammenfassung

Die Web-UI für den NV-Zentren Simulator bietet eine umfassende Benutzeroberfläche zur Konfiguration und Überwachung des Simulators. Mit einem modernen, reaktiven Frontend und einer robusten Backend-Implementierung ermöglicht sie die vollständige Kontrolle über alle simulatorspezifischen Parameter und Einstellungen. Die Benutzeroberfläche ist intuitiv gestaltet und bietet sowohl Anfängern als auch erfahrenen Benutzern alle notwendigen Werkzeuge zur Anpassung des Simulators an ihre spezifischen Anforderungen.