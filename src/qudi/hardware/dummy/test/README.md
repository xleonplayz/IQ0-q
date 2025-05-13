# NV Simulator ODMR Test Suite

Diese Test-Suite enthält Werkzeuge zum Testen und Debuggen der NV-Simulator-Komponenten, insbesondere für ODMR-Messungen.

## Übersicht der Testskripte

### 1. `test_mw_sampler_sync.py`

Testet die direkte Kommunikation zwischen dem Microwave-Controller und dem Finite Sampler über den Shared-State-Mechanismus, ohne die ODMR-Logik.

**Enthält zwei Tests:**
- `test_direct_frequency_setting()`: Testet die direkte Frequenzeinstellung und Synchronisation
- `test_scan_mode_synchronization()`: Testet die Scan-Modus-Frequenzsynchronisation

**Ausführung:**
```
python test_mw_sampler_sync.py
```

### 2. `test_odmr_flow.py`

Testet den vollständigen Informationsfluss zwischen ODMR-Logik, Microwave-Controller und Finite Sampler.

**Ausführung:**
```
python run_odmr_test.py
```

### 3. `run_odmr_test.py`

Runner-Skript für den ODMR-Flow-Test mit Visualisierung der Ergebnisse.

## Fehlerbehebung

### Typische Probleme und Lösungsansätze

1. **Keine ODMR-Dips sichtbar**
   - Überprüfen Sie den konfigurierten Frequenzbereich (sollte 1.4-4.4 GHz für ein 500-Gauss-Feld sein)
   - Überprüfen Sie, ob die Microwave-Frequenz korrekt im Shared State aktualisiert wird
   - Überprüfen Sie, ob der Finite Sampler die aktuelle Frequenz aus dem Shared State liest

2. **Falsche Frequenzen**
   - Überprüfen Sie die Kommunikation zwischen Microwave-Controller und Finite Sampler
   - Stellen Sie sicher, dass scan_next() korrekt aufgerufen wird und den Scan-Index aktualisiert

3. **Keine Kommunikation zwischen Modulen**
   - Stellen Sie sicher, dass alle Module auf dieselbe QudiFacade-Instanz zugreifen
   - Überprüfen Sie, ob der Shared-State-Mechanismus in allen relevanten Methoden verwendet wird

## Erwartete ODMR-Resonanzen

Bei einem magnetischen Feld von 500 Gauss entlang der NV-Achse:
- Erste Resonanz: etwa 1.47 GHz (2.87 GHz - 1.4 GHz)
- Zweite Resonanz: etwa 4.27 GHz (2.87 GHz + 1.4 GHz)

## Logdateien

Die Tests erzeugen detaillierte Logdateien im Test-Verzeichnis:
- `odmr_flow_test.log`
- `mw_sampler_sync_test.log`

## Visualisierungen

Die Testskripte erzeugen automatisch Visualisierungen der Ergebnisse im Unterverzeichnis `results/`.