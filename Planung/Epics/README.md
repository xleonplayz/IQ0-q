# NV-Center Simulator - Entwicklungs-Epics

In diesem Verzeichnis sind die Entwicklungs-Epics für das NV-Center Simulator-Projekt organisiert. Jedes Epic beschreibt einen größeren Funktionsbereich, der aus mehreren technischen Stories besteht und einen signifikanten Geschäftswert liefert.

## Was sind Epics?

Epics sind große Arbeitspakete, die in kleinere Stories unterteilt werden können. Sie stellen eine umfangreiche Funktion oder Komponente dar, die mehrere Wochen Entwicklungszeit erfordert. Jedes Epic umfasst:

- Eine klare Beschreibung des Umfangs
- Den erwarteten Geschäftswert
- Akzeptanzkriterien für den Abschluss
- Eine Aufschlüsselung in technische Stories mit konkreten Aufgaben
- Eine Aufwandsschätzung
- Risiken und Abhängigkeiten

## Epic-Struktur

Die Epics sind in den folgenden Hauptkategorien organisiert:

1. **CORE** - Kern-Epics (E-CORE-xx)
   - Fundamentale Komponenten des Simulators
   - Quantenphysikalisches Modell und Zustandsverwaltung

2. **QDI** - Qudi-Interface-Epics (E-QDI-xx)
   - Implementierungen der Qudi-Hardware-Interfaces
   - Simulation von Hardware-Komponenten

3. **NET** - Netzwerk-Epics (E-NET-xx)
   - Netzwerkkommunikationsschicht
   - Protokolle und Konnektivität

4. **WEB** - Web-UI-Epics (E-WEB-xx)
   - Webbasierte Benutzeroberfläche
   - Konfiguration und Visualisierung

5. **INT** - Integrations-Epics (E-INT-xx)
   - Integration mit externen Systemen
   - Gesamtsystem-Integration

6. **TST** - Test-Epics (E-TST-xx)
   - Komponenten- und Integrationstests
   - Validierung und Verifikation

7. **DOC** - Dokumentations-Epics (E-DOC-xx)
   - Entwickler- und Benutzerdokumentation
   - Tutorials und Beispiele

## Implementierungsreihenfolge

Die empfohlene Implementierungsreihenfolge ist in der [Epic-Übersicht](EPIC_OVERVIEW.md) beschrieben. Die Implementierung sollte mit den Kern-Epics beginnen und dann zu den abhängigen Komponenten übergehen.

## Verfügbare Epics

Hier sind die verfügbaren Epic-Beschreibungen:

| Epic-ID | Name | Beschreibung | Datei |
|---------|------|-------------|-------|
| E-CORE-01 | Physikalisches Modell | Quantenphysikalisches NV-Center-Modell | [E-CORE-01_Physikalisches_Modell.md](E-CORE-01_Physikalisches_Modell.md) |
| E-CORE-02 | Simulatorzustandsverwaltung | Zentraler Thread-sicherer Zustandsmanager | [E-CORE-02_Simulatorzustandsverwaltung.md](E-CORE-02_Simulatorzustandsverwaltung.md) |
| E-QDI-01 | Mikrowellen-Interface | Implementierung des MicrowaveInterface | [E-QDI-01_Mikrowellen_Interface.md](E-QDI-01_Mikrowellen_Interface.md) |
| E-WEB-01 | Web-UI Grundgerüst | Web-UI-Framework für die Konfiguration | [E-WEB-01_Web_UI_Grundgeruest.md](E-WEB-01_Web_UI_Grundgeruest.md) |

Weitere Epics werden im Laufe des Projekts hinzugefügt und spezifiziert.

## Zeitplanung

Die gesamte Implementierungsdauer wird auf etwa 24 Wochen (6 Monate) geschätzt, aufgeteilt in 5 Phasen:

1. **Phase 1: Kernmodell** (4 Wochen)
2. **Phase 2: Basisschnittstellen** (6 Wochen)
3. **Phase 3: Erweiterte Features** (8 Wochen)
4. **Phase 4: Integration und Tests** (4 Wochen)
5. **Phase 5: Finalisierung** (2 Wochen)

Details zu den Phasen finden sich in der [Epic-Übersicht](EPIC_OVERVIEW.md).

## Verwendung dieser Epics

Diese Epic-Beschreibungen dienen als Grundlage für die Projektplanung und -implementierung. Sie sollten in ein Projektmanagement-Tool (wie JIRA, GitHub Projects, etc.) übertragen und dort verwaltet werden. Jede technische Story kann als eigenständiges Ticket erstellt und zugewiesen werden.

Die Epics sollten regelmäßig überprüft und bei Bedarf aktualisiert werden, um Änderungen in den Anforderungen oder im Projektumfang zu berücksichtigen.