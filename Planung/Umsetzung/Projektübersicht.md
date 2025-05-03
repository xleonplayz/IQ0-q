# NV-Simulator Projektübersicht

## Projektstatus (Stand 03.05.2025)

### Implementierte Komponenten

- **Kernphysik-Modul**: Grundlegende Implementierung des NV-Zentren-Quantenmodells (TS-101 bis TS-103)
  - Implementierung der PhysicalNVModel-Klasse mit grundlegender Struktur
  - Magnetfeldinteraktion
  - Mikrowellen- und Lasersteuerung
  - Implementierung des Hamiltonians
  - Zeitsimulation und Quantenzustandsentwicklung

### Laufende Entwicklung

- **Kernphysik-Modul**: Vollständige Implementierung (TS-104 bis TS-109)
  - Zeitentwicklung des Quantenzustands
  - Implementierung optischer Prozesse
  - Relaxation und Dekohärenzprozesse
  - Experimentprotokolle
  - Performance-Optimierung
  - Validierung und Kalibrierung

### Nächste Schritte

1. **Epic E-CORE-02**: Simulatorzustandsverwaltung 
   - Thread-sichere zentrale Zustandsverwaltung
   - Netzwerkkommunikation

2. **Epic E-QDI-01**: Mikrowellen-Interface
   - Qudi-kompatible Mikrowave-Schnittstelle
   - Integration mit dem physikalischen Modell

3. **Epic E-WEB-01**: Web-UI Grundgerüst
   - Basis-Webschnittstelle für Konfiguration
   - Visualisierung von Simulationsparametern

## Projektorganisation

- **Entwicklungsmethodik**: Agil mit Epics und User Stories
- **Versionskontrolle**: GitHub (https://github.com/xleonplayz/IQ0-q)
- **CI/CD**: GitHub Actions mit automatisierten Tests
- **Dokumentation**: Umfassende Planung im Verzeichnis "Planung"

## Projektstruktur

```
simos_nv_simulator/
├── core/                # Kernmodule für Quantensimulation
├── qudi_interfaces/     # Qudi-Hardware-Schnittstellen (geplant)
├── network/             # Netzwerkkommunikation (geplant)
├── web_ui/              # Web-Benutzeroberfläche (geplant)
tests/
├── core/                # Tests für Kernmodule
├── qudi_interfaces/     # Tests für Qudi-Schnittstellen (geplant)
```

## Testabdeckung

- Umfassende Testsuiten für das physikalische Modell
- Automatisierte Tests via GitHub Actions
- Kontinuierliche Integration mit Qualitätssicherung