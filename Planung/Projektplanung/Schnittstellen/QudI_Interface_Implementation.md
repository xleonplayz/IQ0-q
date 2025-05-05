# Qudi-Interface-Implementierung für den NV-Zentren Simulator

## 1. Einführung

Dieses Dokument beschreibt im Detail, wie die verschiedenen Qudi-Hardware-Interfaces im NV-Zentren Simulator implementiert werden. Jedes Interface muss die spezifischen Anforderungen von Qudi erfüllen und gleichzeitig mit dem physikalischen Modell des Simulators interagieren.

## 2. Allgemeine Design-Prinzipien

Bei der Implementierung der Hardware-Interfaces werden folgende Prinzipien befolgt:

- **Vollständige Interface-Kompatibilität**: Jede Implementierung folgt exakt der Qudi-Interface-Definition
- **Realistische Simulationsparameter**: Hardware-Einschränkungen basieren auf realen Geräten
- **Gemeinsamer Zustandsspeicher**: Alle Interface-Implementierungen teilen sich den gemeinsamen QuantumSimulatorState
- **Realistische Verzögerungen**: Operationen simulieren realistische Hardware-Verzögerungen
- **Fehlertoleranz**: Robuste Fehlerbehandlung und Validierung
- **Thread-Sicherheit**: Alle Operationen sind thread-safe

## 3. Gemeinsame Basisklasse

Alle Hardware-Interface-Implementierungen erben von einer gemeinsamen Basisklasse `SimulatedHardwareBase`:

```python
from qudi.core.module import Base
from qudi.util.mutex import Mutex

class SimulatedHardwareBase(Base):
    """Basisklasse für alle simulierten Hardware-Module."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._thread_lock = Mutex()
        self._simulator_state = None
        
    def on_activate(self):
        """Gemeinsame Aktivierungslogik für alle simulierten Hardware-Module."""
        # Zugriff auf Simulator-Zustand
        from nv_simulator.core.state import QuantumSimulatorState
        self._simulator_state = QuantumSimulatorState()
        
        # Logging
        self.log.info(f"{self.__class__.__name__} activated")
        
    def on_deactivate(self):
        """Gemeinsame Deaktivierungslogik für alle simulierten Hardware-Module."""
        self.log.info(f"{self.__class__.__name__} deactivated")
    
    def _simulate_delay(self, operation_type):
        """Simuliert eine realistische Verzögerung für eine Hardware-Operation."""
        delay_config = self._simulator_state.config.get('timing', {})
        
        # Wenn realistische Verzögerungen aktiviert sind
        if delay_config.get('realistic_delays', True):
            delay = delay_config.get(f'{operation_type}_delay', 0.05)  # Default: 50ms
            if delay > 0:
                import time
                time.sleep(delay)
```

## 4. Microwave Interface Implementation

### 4.1 Klassenübersicht

```python
from qudi.interface.microwave_interface import MicrowaveInterface, MicrowaveConstraints
from qudi.util.enums import SamplingOutputMode
import numpy as np
import threading
import time

class MicrowaveSimulator(SimulatedHardwareBase, MicrowaveInterface):
    """Simulierte Mikrowellenquelle für NV-Zentren-Experimente."""
    
    _modclass = 'microwave'
    _modtype = 'hardware'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Gerätezustände
        self._cw_frequency = 2.87e9  # Default: NV-Zentren Zero-Field Splitting
        self._cw_power = -20         # Default: -20 dBm
        self._scan_frequencies = None
        self._scan_power = -20
        self._scan_mode = SamplingOutputMode.JUMP_LIST
        self._scan_sample_rate = 100
        self._is_scanning = False
        self._scan_thread = None
        
        # Constraints werden in on_activate initialisiert
        self._constraints = None
```

### 4.2 Interface-Properties und Methoden

```python
@property
def constraints(self) -> MicrowaveConstraints:
    """Gibt die Hardware-Einschränkungen zurück."""
    return self._constraints

@property
def is_scanning(self) -> bool:
    """Gibt an, ob ein Frequenzscan läuft."""
    with self._thread_lock:
        return self._is_scanning

@property
def cw_power(self) -> float:
    """Gibt die aktuelle CW-Leistung in dBm zurück."""
    with self._thread_lock:
        return self._cw_power

@property
def cw_frequency(self) -> float:
    """Gibt die aktuelle CW-Frequenz in Hz zurück."""
    with self._thread_lock:
        return self._cw_frequency

# Weitere Properties...

def off(self) -> None:
    """Schaltet den Mikrowellenausgang aus."""
    with self._thread_lock:
        if self.module_state() == 'idle':
            self.log.debug('Microwave output was not active')
            return
        
        # Realistischen Hardware-Delay simulieren
        self._simulate_delay('microwave_off')
        
        # Aktualisiere den Simulator-Zustand
        self._simulator_state.update_microwave_state(False)
        
        # Aktualisiere den Modulzustand
        self._is_scanning = False
        self.module_state.unlock()
        
        self.log.debug("Microwave output switched OFF")

def set_cw(self, frequency: float, power: float) -> None:
    """Konfiguriert den CW-Mikrowellenausgang."""
    with self._thread_lock:
        if self.module_state() != 'idle':
            raise RuntimeError('Unable to set CW: Microwave is active.')
            
        # Parameter-Validierung
        if not (self._constraints.frequency_limits[0] <= frequency <= self._constraints.frequency_limits[1]):
            raise ValueError(f"Frequency {frequency} out of bounds")
            
        if not (self._constraints.power_limits[0] <= power <= self._constraints.power_limits[1]):
            raise ValueError(f"Power {power} out of bounds")
        
        # Realistischen Hardware-Delay simulieren
        self._simulate_delay('microwave_set')
        
        # Parameter lokal speichern
        self._cw_frequency = frequency
        self._cw_power = power
        
        # Aktualisiere den Simulator-Zustand
        self._simulator_state.update_microwave_state(False, frequency, power)
        
        self.log.debug(f"CW parameters set: frequency={frequency}Hz, power={power}dBm")

def cw_on(self) -> None:
    """Aktiviert den CW-Mikrowellenausgang."""
    with self._thread_lock:
        if self.module_state() == 'idle':
            # Realistischen Hardware-Delay simulieren
            self._simulate_delay('microwave_on')
            
            # Aktualisiere den Simulator-Zustand
            self._simulator_state.update_microwave_state(True)
            
            # Aktualisiere den Modulzustand
            self._is_scanning = False
            self.module_state.lock()
            
            self.log.debug(f"CW output enabled at {self._cw_frequency}Hz with {self._cw_power}dBm")
        elif self._is_scanning:
            raise RuntimeError('Unable to start CW: frequency scanning in progress.')
        else:
            self.log.debug('CW microwave output already running')

# Weitere Methoden für Frequency Scanning...
```

### 4.3 Scan-Methoden-Implementierung

```python
def configure_scan(self, power: float, frequencies, mode: SamplingOutputMode, sample_rate: float) -> None:
    """Konfiguriert einen Frequenzscan."""
    with self._thread_lock:
        if self.module_state() != 'idle':
            raise RuntimeError('Unable to configure scan: Microwave is active.')
            
        # Parameter-Validierung
        if not (self._constraints.power_limits[0] <= power <= self._constraints.power_limits[1]):
            raise ValueError(f"Power {power} out of bounds")
            
        if not (self._constraints.sample_rate_limits[0] <= sample_rate <= self._constraints.sample_rate_limits[1]):
            raise ValueError(f"Sample rate {sample_rate} out of bounds")
        
        # Frequenzliste vorbereiten
        if mode == SamplingOutputMode.JUMP_LIST:
            if not all(self._constraints.frequency_limits[0] <= f <= self._constraints.frequency_limits[1] for f in frequencies):
                raise ValueError(f"Some frequencies out of bounds")
            self._scan_frequencies = np.array(frequencies)
            
        elif mode == SamplingOutputMode.EQUIDISTANT_SWEEP:
            start, stop, num_points = frequencies
            if not (self._constraints.frequency_limits[0] <= start <= self._constraints.frequency_limits[1]):
                raise ValueError(f"Start frequency {start} out of bounds")
            if not (self._constraints.frequency_limits[0] <= stop <= self._constraints.frequency_limits[1]):
                raise ValueError(f"Stop frequency {stop} out of bounds")
            self._scan_frequencies = (start, stop, num_points)
        
        # Realistischen Hardware-Delay simulieren
        self._simulate_delay('microwave_configure')
        
        # Parameter lokal speichern
        self._scan_power = power
        self._scan_mode = mode
        self._scan_sample_rate = sample_rate
        
        # Aktualisiere den Simulator-Zustand
        self._simulator_state.microwave_scan_params = {
            'power': power,
            'frequencies': self._scan_frequencies,
            'mode': mode,
            'sample_rate': sample_rate
        }
        
        self.log.debug(f"Scan configured: power={power}dBm, mode={mode.name}, rate={sample_rate}Hz")

def start_scan(self) -> None:
    """Startet den konfigurierten Frequenzscan."""
    with self._thread_lock:
        if self.module_state() != 'idle':
            raise RuntimeError('Unable to start scan: Microwave is active.')
            
        if self._scan_frequencies is None:
            raise RuntimeError('Unable to start scan: No frequencies configured.')
        
        # Realistischen Hardware-Delay simulieren
        self._simulate_delay('microwave_start')
        
        # Aktualisiere den Modulzustand
        self._is_scanning = True
        self.module_state.lock()
        
        # Starte den Scan-Thread
        if self._scan_thread is not None and self._scan_thread.is_alive():
            self._scan_thread.join(timeout=1.0)
            
        self._scan_thread = threading.Thread(target=self._run_scan)
        self._scan_thread.daemon = True
        self._scan_thread.start()
        
        self.log.debug(f"Frequency scan started in {self._scan_mode.name} mode")
        
def _run_scan(self):
    """Führt den Frequenzscan im Hintergrund aus."""
    try:
        # Je nach Scan-Modus, extrahiere die Frequenzliste
        if self._scan_mode == SamplingOutputMode.JUMP_LIST:
            freq_list = self._scan_frequencies
        else:  # EQUIDISTANT_SWEEP
            start, stop, num_points = self._scan_frequencies
            freq_list = np.linspace(start, stop, num_points)
        
        # Setze die erste Frequenz
        self._simulator_state.update_microwave_state(True, freq_list[0], self._scan_power)
        
        # Durchlaufe alle Frequenzen
        for freq in freq_list:
            # Prüfe, ob der Scan unterbrochen wurde
            if not self._is_scanning:
                break
                
            # Aktualisiere die Frequenz
            self._simulator_state.update_microwave_state(True, freq, self._scan_power)
            
            # Warte entsprechend der Sample-Rate
            time.sleep(1.0 / self._scan_sample_rate)
            
        # Scan beenden, falls noch aktiv
        if self._is_scanning:
            # Muss über off() gehen, um den Modulzustand korrekt zu aktualisieren
            self.off()
            
    except Exception as e:
        self.log.error(f"Error during scan: {str(e)}")
        # Versuche, sauber aufzuräumen
        try:
            if self._is_scanning:
                self.off()
        except:
            pass
```

### 4.4 Initialisierung der Hardware-Constraints

```python
def on_activate(self):
    """Initialisierung bei Aktivierung des Moduls."""
    # Übergeordnete Klasse initialisieren
    super().on_activate()
    
    # Hardware-Constraints erstellen
    self._constraints = MicrowaveConstraints()
    self._constraints.power_limits = (-120, 30)  # -120 bis +30 dBm
    self._constraints.frequency_limits = (100e3, 20e9)  # 100 kHz bis 20 GHz
    self._constraints.scan_size_limits = (2, 10001)  # Mindestens 2, maximal 10001 Punkte
    self._constraints.sample_rate_limits = (0.1, 1000)  # 0.1 Hz bis 1 kHz
    self._constraints.scan_modes = (SamplingOutputMode.JUMP_LIST, SamplingOutputMode.EQUIDISTANT_SWEEP)
    
    # Parameter aus der Konfiguration übernehmen, falls vorhanden
    config = self._simulator_state.config.get('microwave_constraints', {})
    if 'power_limits' in config:
        self._constraints.power_limits = tuple(config['power_limits'])
    if 'frequency_limits' in config:
        self._constraints.frequency_limits = tuple(config['frequency_limits'])
    
    self.log.info("MicrowaveSimulator initialized with constraints:")
    self.log.info(f"- Power: {self._constraints.power_limits[0]} to {self._constraints.power_limits[1]} dBm")
    self.log.info(f"- Frequency: {self._constraints.frequency_limits[0]/1e6} to {self._constraints.frequency_limits[1]/1e6} MHz")
```

## 5. Fast Counter Interface Implementation

### 5.1 Klassenübersicht

```python
from qudi.interface.fast_counter_interface import FastCounterInterface, FastCounterConstraints
import numpy as np
import threading
import time

class FastCounterSimulator(SimulatedHardwareBase, FastCounterInterface):
    """Simulierter Fast Counter für Photonenzählung in NV-Zentren-Experimenten."""
    
    _modclass = 'fast_counter'
    _modtype = 'hardware'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Zustände und Parameter
        self._is_running = False
        self._bin_width = 1e-9  # 1 ns Bins
        self._record_length = 1e-6  # 1 µs Record
        self._number_of_gates = 0
        self._data_buffer = None
        self._counting_thread = None
```

### 5.2 Interface-Methoden

```python
def get_constraints(self) -> FastCounterConstraints:
    """Gibt die Hardware-Constraints zurück."""
    constraints = FastCounterConstraints()
    constraints.min_bin_width = 1e-12  # 1 ps
    constraints.max_bin_width = 1e-6   # 1 µs
    constraints.min_record_length = 1e-9  # 1 ns
    constraints.max_record_length = 1.0  # 1 s
    constraints.max_number_of_gates = 1000000  # Maximal 1M Gates
    
    # Parameter aus der Konfiguration übernehmen, falls vorhanden
    config = self._simulator_state.config.get('fast_counter_constraints', {})
    if 'min_bin_width' in config:
        constraints.min_bin_width = config['min_bin_width']
    # Weitere Parameter...
    
    return constraints

def configure(self, bin_width_s: float, record_length_s: float, number_of_gates: int = 0) -> None:
    """Konfiguriert die Zählparameter."""
    with self._thread_lock:
        # Parameter-Validierung
        constraints = self.get_constraints()
        
        if not constraints.min_bin_width <= bin_width_s <= constraints.max_bin_width:
            raise ValueError(f"Bin width {bin_width_s} out of bounds")
            
        if not constraints.min_record_length <= record_length_s <= constraints.max_record_length:
            raise ValueError(f"Record length {record_length_s} out of bounds")
            
        if not 0 <= number_of_gates <= constraints.max_number_of_gates:
            raise ValueError(f"Number of gates {number_of_gates} out of bounds")
        
        # Falls aktuell eine Messung läuft, diese stoppen
        if self._is_running:
            self.stop_measure()
        
        # Realistischen Hardware-Delay simulieren
        self._simulate_delay('counter_configure')
        
        # Parameter speichern
        self._bin_width = bin_width_s
        self._record_length = record_length_s
        self._number_of_gates = number_of_gates
        
        # Daten-Buffer initialisieren
        num_bins = int(record_length_s / bin_width_s)
        if self._number_of_gates > 0:
            self._data_buffer = np.zeros((self._number_of_gates, num_bins), dtype=np.uint32)
        else:
            self._data_buffer = np.zeros(num_bins, dtype=np.uint32)
        
        # Aktualisiere den Simulator-Zustand
        self._simulator_state.counter_params = {
            'bin_width': bin_width_s,
            'record_length': record_length_s,
            'number_of_gates': number_of_gates
        }
        
        self.log.debug(f"Fast counter configured: bin_width={bin_width_s}s, length={record_length_s}s, gates={number_of_gates}")

def start_measure(self) -> None:
    """Startet die Photonenzählung."""
    with self._thread_lock:
        if self._is_running:
            self.log.warning("Measurement already running")
            return
            
        if self._data_buffer is None:
            raise RuntimeError("Counter not configured")
        
        # Realistischen Hardware-Delay simulieren
        self._simulate_delay('counter_start')
        
        # Status aktualisieren
        self._is_running = True
        self._simulator_state.counter_is_running = True
        
        # Zähl-Thread starten
        if self._counting_thread is not None and self._counting_thread.is_alive():
            self._counting_thread.join(timeout=1.0)
            
        self._counting_thread = threading.Thread(target=self._count_photons)
        self._counting_thread.daemon = True
        self._counting_thread.start()
        
        self.log.debug("Fast counter measurement started")

def _count_photons(self):
    """Simuliert die Photonenzählung im Hintergrund."""
    try:
        while self._is_running:
            # Aktives NV-Zentrum an der aktuellen Position finden
            nv_model = self._simulator_state.get_nv_at_position(self._simulator_state.scanner_position)
            
            if nv_model is not None:
                # Zählrate vom NV-Modell abrufen
                count_rate = nv_model.get_photon_count_rate()
                
                # Poisson-verteilte Zählung simulieren
                mean_counts_per_bin = count_rate * self._bin_width
                
                if self._number_of_gates > 0:
                    # Multi-Gate Messung
                    for gate in range(self._number_of_gates):
                        if not self._is_running:
                            break
                        self._data_buffer[gate, :] = np.random.poisson(mean_counts_per_bin, self._data_buffer.shape[1])
                else:
                    # Einfache Messung
                    self._data_buffer[:] = np.random.poisson(mean_counts_per_bin, self._data_buffer.shape[0])
            
            # Kurze Pause für CPU-Entlastung
            time.sleep(0.01)
            
    except Exception as e:
        self.log.error(f"Error in counting thread: {str(e)}")
        self._is_running = False
        self._simulator_state.counter_is_running = False

def stop_measure(self) -> None:
    """Stoppt die Photonenzählung."""
    with self._thread_lock:
        if not self._is_running:
            self.log.warning("Measurement not running")
            return
        
        # Realistischen Hardware-Delay simulieren
        self._simulate_delay('counter_stop')
        
        # Status aktualisieren
        self._is_running = False
        self._simulator_state.counter_is_running = False
        
        # Auf Thread-Ende warten
        if self._counting_thread is not None and self._counting_thread.is_alive():
            self._counting_thread.join(timeout=1.0)
            
        self.log.debug("Fast counter measurement stopped")

def pause_measure(self) -> None:
    """Pausiert die Photonenzählung."""
    with self._thread_lock:
        if not self._is_running:
            self.log.warning("Measurement not running")
            return
        
        # Realistischen Hardware-Delay simulieren
        self._simulate_delay('counter_pause')
        
        # Status aktualisieren
        self._is_running = False
        self._simulator_state.counter_is_running = False
        
        self.log.debug("Fast counter measurement paused")

def continue_measure(self) -> None:
    """Setzt eine pausierte Photonenzählung fort."""
    with self._thread_lock:
        if self._is_running:
            self.log.warning("Measurement already running")
            return
            
        if self._data_buffer is None:
            raise RuntimeError("Counter not configured")
        
        # Realistischen Hardware-Delay simulieren
        self._simulate_delay('counter_continue')
        
        # Status aktualisieren
        self._is_running = True
        self._simulator_state.counter_is_running = True
        
        # Zähl-Thread starten
        if self._counting_thread is not None and self._counting_thread.is_alive():
            self._counting_thread.join(timeout=1.0)
            
        self._counting_thread = threading.Thread(target=self._count_photons)
        self._counting_thread.daemon = True
        self._counting_thread.start()
        
        self.log.debug("Fast counter measurement continued")

def get_data_trace(self) -> np.ndarray:
    """Gibt die erfassten Zähldaten zurück."""
    with self._thread_lock:
        if self._data_buffer is None:
            raise RuntimeError("No data available")
        
        # Kopie zurückgeben, um Race-Conditions zu vermeiden
        return self._data_buffer.copy()
```

## 6. Pulser Interface Implementation

### 6.1 Klassenübersicht

```python
from qudi.interface.pulser_interface import PulserInterface, PulserConstraints
from qudi.util.helpers import natural_sort
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
import os
import time

class PulserSimulator(SimulatedHardwareBase, PulserInterface):
    """Simulierter Pulsgenerator für Quantenoperationen an NV-Zentren."""
    
    _modclass = 'pulser'
    _modtype = 'hardware'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Zustände und Parameter
        self._is_on = False
        self._sample_rate = 1.0e9  # 1 GHz Abtastrate
        self._loaded_assets = {}
        self._asset_type = None  # 'waveform' oder 'sequence'
        
        # Speicherung für Wellenformen und Sequenzen
        self._saved_waveforms = {}  # Name -> {Analog-Daten, Digital-Daten}
        self._saved_sequences = {}  # Name -> Liste von Sequenzparametern
        
        # Kanalkonfiguration
        self._active_channels = {}
        self._analog_levels = {}
        self._digital_levels = {}
```

### 6.2 Interface-Methoden

```python
def get_constraints(self) -> PulserConstraints:
    """Gibt die Hardware-Constraints zurück."""
    from qudi.util.constraints import ScalarConstraint
    from qudi.interface.pulser_interface import SequenceOption
    
    constraints = PulserConstraints()
    
    # Abtastrate
    constraints.sample_rate = ScalarConstraint(min=100e6, max=5e9, step=1e6, default=1e9)
    
    # Analoge Kanäle
    constraints.a_ch_amplitude = ScalarConstraint(min=0.0, max=2.0, step=0.001, default=1.0)
    constraints.a_ch_offset = ScalarConstraint(min=-1.0, max=1.0, step=0.001, default=0.0)
    
    # Digitale Kanäle
    constraints.d_ch_low = ScalarConstraint(min=0.0, max=0.0, step=0.0, default=0.0)
    constraints.d_ch_high = ScalarConstraint(min=5.0, max=5.0, step=0.0, default=5.0)
    
    # Wellenformparameter
    constraints.waveform_length = ScalarConstraint(min=1, max=1e9, step=1, default=1000)
    constraints.waveform_num = ScalarConstraint(min=1, max=1e6, step=1, default=1)
    
    # Sequenzparameter
    constraints.sequence_num = ScalarConstraint(min=1, max=10000, step=1, default=1)
    constraints.subsequence_num = ScalarConstraint(min=1, max=10000, step=1, default=1)
    constraints.sequence_steps = ScalarConstraint(min=1, max=1000, step=1, default=1)
    constraints.repetitions = ScalarConstraint(min=0, max=65536, step=1, default=0)
    
    # Weitere Konfigurationen
    constraints.event_triggers = ['A', 'B', 'C', 'D']
    constraints.flags = ['X', 'Y', 'Z']
    
    # Aktivierungskonfigurationen
    constraints.activation_config = {
        'all': frozenset({'a_ch1', 'a_ch2', 'd_ch1', 'd_ch2', 'd_ch3', 'd_ch4'}),
        'analog_only': frozenset({'a_ch1', 'a_ch2'}),
        'digital_only': frozenset({'d_ch1', 'd_ch2', 'd_ch3', 'd_ch4'})
    }
    
    # Sequenzunterstützung
    constraints.sequence_option = SequenceOption.OPTIONAL
    
    return constraints

def pulser_on(self) -> int:
    """Aktiviert den Pulsgenerator."""
    with self._thread_lock:
        if self._is_on:
            self.log.warning("Pulser is already on")
            return 0
            
        if not self._loaded_assets:
            self.log.error("No assets loaded")
            return -1
        
        # Realistischen Hardware-Delay simulieren
        self._simulate_delay('pulser_on')
        
        # Status aktualisieren
        self._is_on = True
        self._simulator_state.pulser_is_on = True
        
        # Jetzt sollten wir die geladenen Assets an das physikalische Modell übergeben
        # und dort die entsprechenden Pulse ausführen
        if self._asset_type == 'waveform':
            self._execute_waveform()
        elif self._asset_type == 'sequence':
            self._execute_sequence()
        
        self.log.debug(f"Pulser activated with {self._asset_type}: {list(self._loaded_assets.values())}")
        return 0

def _execute_waveform(self):
    """Führt die geladene Wellenform im physikalischen Modell aus."""
    # Diese Methode muss mit dem physikalischen Modell interagieren, um Pulse auszuführen
    
    # Vereinfachtes Beispiel für eine Waveform-Ausführung:
    # 1. Wellenformdaten holen
    first_channel = list(self._loaded_assets.keys())[0]
    waveform_name = self._loaded_assets[first_channel]
    waveform_data = self._saved_waveforms[waveform_name]
    
    # 2. Durch die Samples der Wellenform iterieren
    sample_time = 1.0 / self._sample_rate
    total_duration = len(waveform_data['analog_samples'][first_channel]) * sample_time
    
    # 3. Benachrichtigungsthread starten, der die Wellenform schrittweise verarbeitet
    execution_thread = threading.Thread(
        target=self._process_waveform,
        args=(waveform_data, sample_time, total_duration)
    )
    execution_thread.daemon = True
    execution_thread.start()

def _process_waveform(self, waveform_data, sample_time, total_duration):
    """Verarbeitet eine Wellenform schrittweise im Hintergrund."""
    try:
        # Beispielimplementierung (stark vereinfacht)
        # In einer echten Implementierung würden wir hier die einzelnen Samples
        # an das physikalische Modell weiterleiten
        
        # Simuliere den zeitlichen Ablauf der Wellenform
        time.sleep(total_duration)
        
        # Nach Abschluss den Pulser automatisch ausschalten
        self._is_on = False
        self._simulator_state.pulser_is_on = False
        
        self.log.debug(f"Waveform execution completed after {total_duration:.6f}s")
        
    except Exception as e:
        self.log.error(f"Error during waveform execution: {str(e)}")
        self._is_on = False
        self._simulator_state.pulser_is_on = False

def pulser_off(self) -> int:
    """Deaktiviert den Pulsgenerator."""
    with self._thread_lock:
        if not self._is_on:
            self.log.warning("Pulser is already off")
            return 0
        
        # Realistischen Hardware-Delay simulieren
        self._simulate_delay('pulser_off')
        
        # Status aktualisieren
        self._is_on = False
        self._simulator_state.pulser_is_on = False
        
        self.log.debug("Pulser deactivated")
        return 0

def load_waveform(self, load_dict: Union[Dict[int, str], List[str]]) -> Dict[int, str]:
    """Lädt eine Wellenform in die spezifizierten Kanäle."""
    with self._thread_lock:
        if self._is_on:
            self.log.error("Cannot load waveform while pulser is active")
            return {}
            
        # Wenn ein anderer Asset-Typ geladen ist, diesen löschen
        if self._asset_type == 'sequence':
            self.clear_all()
            
        # Format des load_dict vereinheitlichen
        if isinstance(load_dict, list):
            channels = list(range(1, len(load_dict) + 1))
            load_dict = dict(zip(channels, load_dict))
            
        # Validierung: Prüfen, ob alle Wellenformen existieren
        if not all(wfm_name in self._saved_waveforms for wfm_name in load_dict.values()):
            missing = [wfm for wfm in load_dict.values() if wfm not in self._saved_waveforms]
            self.log.error(f"Cannot load missing waveforms: {missing}")
            return {}
        
        # Realistischen Hardware-Delay simulieren
        self._simulate_delay('pulser_load')
        
        # Wellenformen laden
        for channel, waveform_name in load_dict.items():
            ch_name = f'a_ch{channel}' if channel <= 2 else f'd_ch{channel-2}'
            self._loaded_assets[ch_name] = waveform_name
            
        self._asset_type = 'waveform'
        self._simulator_state.pulser_loaded_assets = self._loaded_assets.copy()
        
        self.log.debug(f"Loaded waveforms: {load_dict}")
        return load_dict

def write_waveform(self, name: str, analog_samples: Dict[str, np.ndarray], 
                  digital_samples: Dict[str, np.ndarray], is_first_chunk: bool, 
                  is_last_chunk: bool, total_number_of_samples: int) -> Tuple[int, List[str]]:
    """Schreibt eine Wellenform in den Gerätespeicher."""
    with self._thread_lock:
        # Validierung
        constraints = self.get_constraints()
        
        # Prüfen, ob die Sampleanzahl in den erlaubten Grenzen liegt
        if total_number_of_samples < constraints.waveform_length.min or total_number_of_samples > constraints.waveform_length.max:
            self.log.error(f"Invalid number of samples: {total_number_of_samples}")
            return 0, []
            
        # Prüfen, ob analoge Werte im erlaubten Bereich liegen (-1.0 bis 1.0)
        for ch_name, samples in analog_samples.items():
            if np.any(samples < -1.0) or np.any(samples > 1.0):
                self.log.error(f"Analog samples for {ch_name} outside allowed range [-1.0, 1.0]")
                return 0, []
        
        # Realistischen Hardware-Delay simulieren
        self._simulate_delay('pulser_write')
        
        # Die erste Chunk-Übertragung oder Einzelübertragung
        if is_first_chunk:
            # Bei mehreren Chunks, temporäre Speicherung vorbereiten
            if not is_last_chunk:
                temp_data = {
                    'analog_samples': {ch: samples.copy() for ch, samples in analog_samples.items()},
                    'digital_samples': {ch: samples.copy() for ch, samples in digital_samples.items()},
                    'current_size': sum(len(samples) for samples in analog_samples.values())
                }
                self._temp_waveform_data = temp_data
                return len(next(iter(analog_samples.values()))), []
            
            # Einzelübertragung: Direkt speichern
            waveform_data = {
                'analog_samples': {ch: samples.copy() for ch, samples in analog_samples.items()},
                'digital_samples': {ch: samples.copy() for ch, samples in digital_samples.items()}
            }
            
            self._saved_waveforms[name] = waveform_data
            return total_number_of_samples, [name]
            
        # Fortsetzung einer mehrteiligen Übertragung
        else:
            # Daten anhängen
            for ch, samples in analog_samples.items():
                self._temp_waveform_data['analog_samples'][ch] = np.append(
                    self._temp_waveform_data['analog_samples'][ch], samples)
                    
            for ch, samples in digital_samples.items():
                self._temp_waveform_data['digital_samples'][ch] = np.append(
                    self._temp_waveform_data['digital_samples'][ch], samples)
                    
            self._temp_waveform_data['current_size'] += len(next(iter(analog_samples.values())))
            
            # Wenn letzter Chunk, dann speichern
            if is_last_chunk:
                waveform_data = {
                    'analog_samples': self._temp_waveform_data['analog_samples'],
                    'digital_samples': self._temp_waveform_data['digital_samples']
                }
                
                self._saved_waveforms[name] = waveform_data
                total_size = self._temp_waveform_data['current_size']
                self._temp_waveform_data = None
                
                return total_size, [name]
                
            # Noch nicht fertig mit den Chunks
            return self._temp_waveform_data['current_size'], []
```

## 7. Weitere Interface-Implementierungen

Ähnliche detaillierte Implementierungen werden für die folgenden Interfaces erstellt:

- **ScanningProbeSimulator**: Implementiert das ScanningProbeInterface
- **SimpleLaserSimulator**: Implementiert das SimpleLaserInterface 
- **SwitchSimulator**: Implementiert das SwitchInterface

Jede dieser Implementierungen folgt denselben Design-Prinzipien:

1. Erben von SimulatedHardwareBase und dem entsprechenden Qudi-Interface
2. Vollständige Implementierung aller Interface-Methoden
3. Thread-sichere Operationen mit Mutex-Locks
4. Konsistente Zustandsverwaltung über das gemeinsame QuantumSimulatorState-Objekt
5. Realistische Verzögerungen für Hardware-Operationen
6. Umfassende Fehlerbehandlung und Parameter-Validierung

## 8. Integration und Interaktionen

### 8.1 Zustandsynchronisation

Die verschiedenen Interface-Implementierungen interagieren über den gemeinsamen QuantumSimulatorState, der sicherstellt, dass alle Komponenten konsistent arbeiten:

```python
# In MicrowaveSimulator
def cw_on(self):
    with self._thread_lock:
        # ... Validierung ...
        self._simulator_state.update_microwave_state(True)
        # ... weitere Operationen ...

# In FastCounterSimulator
def _count_photons(self):
    while self._is_running:
        # NV-Zentrum an der aktuellen Position abfragen
        nv_model = self._simulator_state.get_nv_at_position(self._simulator_state.scanner_position)
        
        # Zählrate abhängig vom aktuellen Zustand des NV (beeinflusst durch Mikrowell etc.)
        count_rate = nv_model.get_photon_count_rate()
        # ... weitere Operationen ...
```

### 8.2 Realistische NV-Physik

Die Interface-Implementierungen interagieren mit dem physikalischen Modell, um realistische NV-Physik zu simulieren:

```python
# In MicrowaveSimulator
def _run_scan(self):
    for freq in freq_list:
        # Frequenz setzen
        self._simulator_state.update_microwave_state(True, freq, self._scan_power)
        
        # Dies führt zu:
        # 1. Aktualisierung der Mikrowellenzustände im QuantumSimulatorState
        # 2. Aktualisierung des Hamiltonians im PhysicalNVModel
        # 3. Änderung der Besetzungswahrscheinlichkeiten des NV-Zentrums
        # 4. Änderung der Fluoreszenzrate, die vom FastCounterSimulator gemessen wird
```

### 8.3 Simulierte Experiment-Abläufe

Die Interfaces arbeiten zusammen, um komplette experimentelle Abläufe zu simulieren:

```
ODMR-Messung:
1. SimpleLaserSimulator.on()
2. MicrowaveSimulator.configure_scan()
3. MicrowaveSimulator.start_scan()
   ↓
   FastCounterSimulator.get_data_trace() liefert Zählraten,
   die die ODMR-Dips an den Resonanzfrequenzen zeigen
   ↓
4. MicrowaveSimulator.off()
5. SimpleLaserSimulator.off()

Rabi-Messung:
1. PulserSimulator.write_waveform() (definiert die Pulssequenz)
2. PulserSimulator.load_waveform()
3. PulserSimulator.pulser_on()
   ↓
   PhysicalNVModel simuliert die Zeitentwicklung unter der Pulssequenz
   FastCounterSimulator.get_data_trace() liefert oszillierende Zählraten
   ↓
4. PulserSimulator.pulser_off()
```

## 9. Zusammenfassung

Die Interface-Implementierungen für den NV-Zentren Simulator bieten:

1. **Vollständige Qudi-Kompatibilität**: Alle Interface-Methoden werden gemäß der Qudi-Spezifikation implementiert
2. **Realistische Simulation**: Simuliert das tatsächliche Verhalten von NV-Zentren unter verschiedenen Experimentbedingungen
3. **Konsistenter Zustand**: Gemeinsame Zustandsverwaltung über alle Komponenten hinweg
4. **Robuste Fehlerbehandlung**: Umfassende Validierung und Fehlerbehandlung
5. **Erweiterbarkeit**: Modularer Aufbau für zukünftige Erweiterungen

Diese Implementierungen ermöglichen es dem NV-Zentren Simulator, als vollständiger Ersatz für physische Hardware in der Qudi-Umgebung zu fungieren und realistische Simulationen von Quantenexperimenten durchzuführen.