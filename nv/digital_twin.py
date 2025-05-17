import logging
import numpy as np
import os
import json
import scipy.optimize as opt
from typing import Dict, Any, List, Optional, Union, Tuple

from nv.simulator import NVSimulator
from nv.config_loader import load_config

logger = logging.getLogger(__name__)

class DigitalTwin(NVSimulator):
    """
    Digitaler Twin eines realen NV-Zentrums, der experimentelle Daten nachbildet.
    """
    
    def __init__(self, config=None, config_path=None):
        """
        Initialisiert den digitalen Twin.
        
        Parameters
        ----------
        config : dict, optional
            Direkte Konfigurationsparameter
        config_path : str, optional
            Pfad zur Konfigurationsdatei für den digitalen Twin
        """
        # Konfiguration laden
        self.config_loader = load_config(config_path)
        
        # Standardkonfiguration mit digitalen Twin-Einstellungen
        twin_config = self.config_loader.get_nv_system_config()
        
        # Direkte Konfigurationsparameter haben Vorrang
        if config:
            twin_config.update(config)
        
        # NV-Simulator initialisieren
        super().__init__(twin_config)
        
        # Referenzdaten und Kalibrierungsinformationen
        self.reference_data = {}
        self.calibration_info = self.config_loader.get("digital_twin_calibration", {})
        
        # Referenzdaten laden, falls vorhanden
        self._load_reference_data()
        
        logger.info(f"Digitaler Twin initialisiert: {self.calibration_info.get('device_id', 'unbekannt')}")
    
    def _load_reference_data(self):
        """Lädt Referenzdaten aus den in der Konfiguration angegebenen Dateien."""
        try:
            fitted_params = self.calibration_info.get("fitted_parameters", [])
            for param in fitted_params:
                data_path = param.get("reference_data")
                if not data_path:
                    continue
                
                # Vollständigen Pfad erstellen
                full_path = os.path.join(os.path.dirname(self.config_loader.config_path), data_path)
                
                # Überprüfen, ob die Datei existiert
                if not os.path.exists(full_path):
                    logger.warning(f"Referenzdaten nicht gefunden: {full_path}")
                    continue
                
                # Daten laden basierend auf Dateityp
                if full_path.endswith('.csv'):
                    self._load_csv_data(full_path, param)
                elif full_path.endswith('.json'):
                    self._load_json_data(full_path, param)
                else:
                    logger.warning(f"Unbekanntes Dateiformat für Referenzdaten: {full_path}")
        except Exception as e:
            logger.error(f"Fehler beim Laden der Referenzdaten: {str(e)}")
    
    def _load_csv_data(self, file_path, param_info):
        """Lädt CSV-Daten für ein bestimmtes Experiment."""
        try:
            import pandas as pd
            data = pd.read_csv(file_path)
            
            # Experiment-Typ identifizieren
            exp_type = param_info.get("reference_experiment", "unknown")
            
            # Daten je nach Experimenttyp extrahieren
            if exp_type == "odmr":
                # Erwartete Spalten: frequency, fluorescence
                self.reference_data[exp_type] = {
                    'frequencies': data['frequency'].values,
                    'fluorescence': data['fluorescence'].values
                }
            elif exp_type in ["rabi", "ramsey", "t1", "t2"]:
                # Erwartete Spalten: time, fluorescence
                self.reference_data[exp_type] = {
                    'durations': data['time'].values,
                    'fluorescence': data['fluorescence'].values
                }
            
            logger.info(f"Referenzdaten geladen für {exp_type}: {file_path}")
        except Exception as e:
            logger.error(f"Fehler beim Laden der CSV-Daten {file_path}: {str(e)}")
    
    def _load_json_data(self, file_path, param_info):
        """Lädt JSON-Daten für ein bestimmtes Experiment."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Experiment-Typ identifizieren
            exp_type = param_info.get("reference_experiment", "unknown")
            
            # Daten zum entsprechenden Experiment speichern
            self.reference_data[exp_type] = data
            
            logger.info(f"Referenzdaten geladen für {exp_type}: {file_path}")
        except Exception as e:
            logger.error(f"Fehler beim Laden der JSON-Daten {file_path}: {str(e)}")
    
    def calibrate_from_reference_data(self):
        """
        Kalibriert den digitalen Twin anhand aller verfügbaren Referenzdaten.
        
        Returns
        -------
        dict
            Ergebnisse der Kalibrierung
        """
        results = {}
        
        # Parameter aus der Konfiguration holen
        fitted_params = self.calibration_info.get("fitted_parameters", [])
        
        for param in fitted_params:
            param_name = param.get("name")
            exp_type = param.get("reference_experiment")
            
            if not (param_name and exp_type):
                continue
                
            # Überprüfen, ob Referenzdaten für dieses Experiment vorhanden sind
            if exp_type not in self.reference_data:
                logger.warning(f"Keine Referenzdaten für {exp_type} gefunden")
                continue
                
            # Parameter kalibrieren
            try:
                value = self.calibrate_parameter(param_name, exp_type)
                if value is not None:
                    results[param_name] = value
                    logger.info(f"Parameter '{param_name}' kalibriert: {value}")
            except Exception as e:
                logger.error(f"Fehler bei Kalibrierung von {param_name}: {str(e)}")
        
        # Konfiguration speichern nach Kalibrierung
        self.config_loader.save_config()
        
        return results
    
    def calibrate_parameter(self, param_name, exp_type):
        """
        Kalibriert einen einzelnen Parameter anhand von Referenzdaten.
        
        Parameters
        ----------
        param_name : str
            Name des zu kalibrierenden Parameters
        exp_type : str
            Typ des Referenzexperiments (odmr, rabi, usw.)
            
        Returns
        -------
        float or None
            Kalibrierter Parameterwert, oder None bei Fehler
        """
        # Überprüfen, ob Referenzdaten vorhanden
        if exp_type not in self.reference_data:
            logger.warning(f"Keine Referenzdaten für {exp_type}")
            return None
        
        data = self.reference_data[exp_type]
        
        # Parameter je nach Experimenttyp kalibrieren
        if exp_type == "odmr":
            return self._calibrate_from_odmr(param_name, data)
        elif exp_type == "rabi":
            return self._calibrate_from_rabi(param_name, data)
        elif exp_type == "ramsey":
            return self._calibrate_from_ramsey(param_name, data)
        elif exp_type == "t1":
            return self._calibrate_from_t1(param_name, data)
        elif exp_type == "t2":
            return self._calibrate_from_t2(param_name, data)
        else:
            logger.warning(f"Kalibrierung für {exp_type} nicht implementiert")
            return None
    
    def _calibrate_from_odmr(self, param_name, data):
        """Kalibriert Parameter aus ODMR-Daten."""
        # Extrahiere Daten
        x_data = np.array(data['frequencies'])
        y_data = np.array(data['fluorescence'])
        
        # Lorentz-Modell für ODMR
        def lorentz_model(x, amplitude, center, width, offset):
            return offset - amplitude / (1 + ((x - center) / (width/2))**2)
        
        # Startparameter schätzen
        est_offset = np.max(y_data)
        est_center = x_data[np.argmin(y_data)]
        est_amplitude = est_offset - np.min(y_data)
        est_width = (x_data[-1] - x_data[0]) / 10
        
        # Anpassen
        p0 = [est_amplitude, est_center, est_width, est_offset]
        try:
            popt, _ = opt.curve_fit(lorentz_model, x_data, y_data, p0=p0)
            
            if param_name == "zero_field_splitting":
                # D-Wert extrahieren
                d_value = popt[1]  # Resonanzposition
                self.config_loader.set("physical_parameters.zero_field_splitting", d_value)
                return d_value
                
            elif param_name == "fluorescence_contrast":
                # Kontrast berechnen
                contrast = popt[0] / popt[3]  # Amplitude/Offset
                self.config_loader.set("optical_properties.fluorescence_contrast", contrast)
                return contrast
                
            elif param_name == "strain":
                # Strain aus ODMR-Peak-Breite abschätzen
                # Einfache Näherung: überschüssige Breite als Strain interpretieren
                natural_width = 0.1e6  # 100 kHz natürliche Linienbreite
                strain = max(0, popt[2] - natural_width)
                self.config_loader.set("physical_parameters.transverse_strain", strain)
                return strain
        except Exception as e:
            logger.error(f"Fehler bei ODMR-Kalibrierung: {str(e)}")
        
        return None
    
    def _calibrate_from_rabi(self, param_name, data):
        """Kalibriert Parameter aus Rabi-Daten."""
        # Extrahiere Daten
        x_data = np.array(data['durations'])
        y_data = np.array(data['fluorescence'])
        
        # Rabi-Modell mit Dämpfung
        def rabi_model(x, amplitude, frequency, phase, offset, decay):
            return amplitude * np.cos(2*np.pi*frequency*x + phase) * np.exp(-x/decay) + offset
        
        # Startparameter schätzen
        est_offset = np.mean(y_data)
        est_amplitude = (np.max(y_data) - np.min(y_data)) / 2
        
        # Frequenz mit FFT schätzen
        from scipy.fft import fft
        y_centered = y_data - est_offset
        fft_values = fft(y_centered)
        freqs = np.fft.fftfreq(len(x_data), x_data[1] - x_data[0])
        est_freq = abs(freqs[np.argmax(np.abs(fft_values[1:]))+1])
        
        # Anpassen
        p0 = [est_amplitude, est_freq, 0, est_offset, max(x_data)/3]
        try:
            popt, _ = opt.curve_fit(rabi_model, x_data, y_data, p0=p0)
            
            if param_name == "rabi_frequency":
                rabi_freq = abs(popt[1])
                # Mikrowellenleistung in den Daten berücksichtigen
                power = data.get('power', 0)
                power_to_rabi = rabi_freq / (10**(power/20) if power != -np.inf else 1e-10)
                self.config_loader.set("experimental.microwave.power_to_rabi_factor", power_to_rabi)
                return rabi_freq
                
            elif param_name == "t2_star":
                # T2* aus Rabi-Oszillationsdämpfung
                t2_star = popt[4]
                self.config_loader.set("physical_parameters.t2", t2_star)
                return t2_star
                
            elif param_name == "fluorescence_contrast":
                # Kontrast aus Rabi-Amplitude
                contrast = 2 * abs(popt[0]) / popt[3]
                self.config_loader.set("optical_properties.fluorescence_contrast", contrast)
                return contrast
        except Exception as e:
            logger.error(f"Fehler bei Rabi-Kalibrierung: {str(e)}")
        
        return None
    
    def _calibrate_from_ramsey(self, param_name, data):
        """Kalibriert Parameter aus Ramsey-Daten."""
        # Extrahiere Daten
        x_data = np.array(data['durations'])
        y_data = np.array(data['fluorescence'])
        
        # Ramsey-Modell mit Dämpfung
        def ramsey_model(x, amplitude, frequency, phase, offset, decay):
            return amplitude * np.cos(2*np.pi*frequency*x + phase) * np.exp(-x/decay) + offset
        
        # Startparameter schätzen
        est_offset = np.mean(y_data)
        est_amplitude = (np.max(y_data) - np.min(y_data)) / 2
        
        # Frequenz mit FFT schätzen
        from scipy.fft import fft
        y_centered = y_data - est_offset
        fft_values = fft(y_centered)
        freqs = np.fft.fftfreq(len(x_data), x_data[1] - x_data[0])
        est_freq = abs(freqs[np.argmax(np.abs(fft_values[1:]))+1])
        
        # Anpassen
        p0 = [est_amplitude, est_freq, 0, est_offset, max(x_data)/3]
        try:
            popt, _ = opt.curve_fit(ramsey_model, x_data, y_data, p0=p0)
            
            if param_name == "t2_star":
                # T2* direkt aus Ramsey-Dämpfung
                t2_star = popt[4]
                self.config_loader.set("physical_parameters.t2", t2_star)
                return t2_star
                
            elif param_name == "detuning":
                # Verstimmung aus Ramsey-Frequenz
                detuning = abs(popt[1])
                # Hier keine direkte Konfigurationsänderung, da dies experimentabhängig ist
                return detuning
                
            elif param_name == "fluorescence_contrast":
                # Kontrast aus Ramsey-Amplitude
                contrast = 2 * abs(popt[0]) / popt[3]
                self.config_loader.set("optical_properties.fluorescence_contrast", contrast)
                return contrast
        except Exception as e:
            logger.error(f"Fehler bei Ramsey-Kalibrierung: {str(e)}")
        
        return None
    
    def _calibrate_from_t1(self, param_name, data):
        """Kalibriert Parameter aus T1-Daten."""
        # Extrahiere Daten
        x_data = np.array(data['durations'])
        y_data = np.array(data['fluorescence'])
        
        # T1-Modell: Exponentieller Zerfall
        def t1_model(x, amplitude, offset, decay):
            return amplitude * np.exp(-x/decay) + offset
        
        # Startparameter schätzen
        est_offset = np.min(y_data)
        est_amplitude = np.max(y_data) - est_offset
        est_decay = max(x_data) / 3
        
        # Anpassen
        p0 = [est_amplitude, est_offset, est_decay]
        try:
            popt, _ = opt.curve_fit(t1_model, x_data, y_data, p0=p0)
            
            if param_name == "t1":
                t1_value = popt[2]
                self.config_loader.set("physical_parameters.t1", t1_value)
                return t1_value
        except Exception as e:
            logger.error(f"Fehler bei T1-Kalibrierung: {str(e)}")
        
        return None
    
    def _calibrate_from_t2(self, param_name, data):
        """Kalibriert Parameter aus T2-Daten."""
        # Extrahiere Daten
        x_data = np.array(data['durations'])
        y_data = np.array(data['fluorescence'])
        
        # T2-Modell: Gedämpfte Oszillation
        def t2_model(x, amplitude, frequency, phase, offset, decay):
            return amplitude * np.cos(2*np.pi*frequency*x + phase) * np.exp(-x/decay) + offset
        
        # Startparameter schätzen
        est_offset = np.mean(y_data)
        est_amplitude = (np.max(y_data) - np.min(y_data)) / 2
        
        # Frequenz mit FFT schätzen
        from scipy.fft import fft
        y_centered = y_data - est_offset
        fft_values = fft(y_centered)
        freqs = np.fft.fftfreq(len(x_data), x_data[1] - x_data[0])
        est_freq = abs(freqs[np.argmax(np.abs(fft_values[1:]))+1])
        
        # Anpassen
        p0 = [est_amplitude, est_freq, 0, est_offset, max(x_data)/3]
        try:
            popt, _ = opt.curve_fit(t2_model, x_data, y_data, p0=p0)
            
            if param_name == "t2":
                t2_value = popt[4]
                self.config_loader.set("physical_parameters.t2", t2_value)
                return t2_value
        except Exception as e:
            logger.error(f"Fehler bei T2-Kalibrierung: {str(e)}")
        
        return None
    
    def compare_to_reference(self, experiment_type, sim_data):
        """
        Vergleicht Simulationsdaten mit Referenzdaten.
        
        Parameters
        ----------
        experiment_type : str
            Typ des Experiments (odmr, rabi, usw.)
        sim_data : dict
            Simulationsdaten
            
        Returns
        -------
        dict
            Vergleichsstatistiken
        """
        if experiment_type not in self.reference_data:
            logger.warning(f"Keine Referenzdaten für {experiment_type}")
            return None
            
        ref_data = self.reference_data[experiment_type]
        
        # X-Achse je nach Experiment extrahieren
        if experiment_type == "odmr":
            sim_x = sim_data['frequencies']
            sim_y = sim_data['fluorescence']
            ref_x = ref_data['frequencies']
            ref_y = ref_data['fluorescence']
            x_label = "Frequenz (Hz)"
        else:
            sim_x = sim_data['durations']
            sim_y = sim_data['fluorescence']
            ref_x = ref_data['durations']
            ref_y = ref_data['fluorescence']
            x_label = "Zeit (s)"
        
        # Interpoliere Simulationsdaten auf Referenzdaten-X-Werte
        from scipy.interpolate import interp1d
        try:
            sim_interp = interp1d(sim_x, sim_y, bounds_error=False, fill_value="extrapolate")
            sim_y_interp = sim_interp(ref_x)
            
            # Berechne Fehlermetriken
            residuals = ref_y - sim_y_interp
            mse = np.mean(residuals**2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(residuals))
            
            # Normalisierter Fehler
            max_range = np.max(ref_y) - np.min(ref_y)
            if max_range > 0:
                nrmse = rmse / max_range
            else:
                nrmse = np.nan
                
            # Korrelationskoeffizient
            from scipy.stats import pearsonr
            corr, _ = pearsonr(ref_y, sim_y_interp)
            
            results = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "nrmse": nrmse,
                "correlation": corr,
                "sim_data": {
                    "x": sim_x,
                    "y": sim_y,
                    "label": "Simulation"
                },
                "ref_data": {
                    "x": ref_x,
                    "y": ref_y,
                    "label": "Referenz"
                },
                "x_label": x_label,
                "y_label": "Fluoreszenz"
            }
            
            return results
        except Exception as e:
            logger.error(f"Fehler beim Vergleich mit Referenzdaten: {str(e)}")
            return None
    
    def plot_comparison(self, experiment_type, sim_data=None):
        """
        Erstellt einen Vergleichsplot zwischen Simulation und Referenzdaten.
        
        Parameters
        ----------
        experiment_type : str
            Typ des Experiments (odmr, rabi, usw.)
        sim_data : dict, optional
            Simulationsdaten. Falls None, wird das Experiment automatisch ausgeführt.
            
        Returns
        -------
        tuple
            (fig, ax) - Matplotlib-Figur und Achsen
        """
        try:
            import matplotlib.pyplot as plt
            
            # Referenzdaten prüfen
            if experiment_type not in self.reference_data:
                logger.warning(f"Keine Referenzdaten für {experiment_type}")
                return None
                
            # Experiment ausführen, falls keine Simulationsdaten übergeben wurden
            if sim_data is None:
                if experiment_type == "odmr":
                    # ODMR-Parameter aus Referenzdaten extrahieren
                    ref_freqs = self.reference_data[experiment_type]['frequencies']
                    freq_range = (np.min(ref_freqs), np.max(ref_freqs))
                    num_points = len(ref_freqs)
                    sim_data = self.odmr.run(freq_range, num_points=num_points)
                elif experiment_type == "rabi":
                    # Rabi-Parameter aus Referenzdaten extrahieren
                    ref_times = self.reference_data[experiment_type]['durations']
                    time_range = (np.min(ref_times), np.max(ref_times))
                    num_points = len(ref_times)
                    sim_data = self.rabi.run(time_range, num_points=num_points)
                # Weitere Experimente hier hinzufügen...
            
            # Vergleichsdaten berechnen
            comparison = self.compare_to_reference(experiment_type, sim_data)
            if comparison is None:
                return None
                
            # Plot erstellen
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Referenzdaten plotten
            ax.plot(
                comparison["ref_data"]["x"], 
                comparison["ref_data"]["y"], 
                'o', 
                label=comparison["ref_data"]["label"]
            )
            
            # Simulationsdaten plotten
            ax.plot(
                comparison["sim_data"]["x"], 
                comparison["sim_data"]["y"], 
                '-', 
                label=f"{comparison['sim_data']['label']} (NRMSE: {comparison['nrmse']:.4f})"
            )
            
            # Achsenbeschriftungen
            ax.set_xlabel(comparison["x_label"])
            ax.set_ylabel(comparison["y_label"])
            ax.set_title(f"Vergleich: {experiment_type.upper()}")
            ax.legend()
            ax.grid(True)
            
            return fig, ax
        except Exception as e:
            logger.error(f"Fehler beim Erstellen des Vergleichsplots: {str(e)}")
            return None
    
    def optimize_twin(self, parameters=None, experiment_type=None, method='Powell'):
        """
        Optimiert automatisch mehrere Parameter, um die Übereinstimmung mit Referenzdaten zu maximieren.
        
        Parameters
        ----------
        parameters : list, optional
            Liste von zu optimierenden Parametern. Falls None, werden alle für das
            angegebene Experiment relevanten Parameter optimiert.
        experiment_type : str, optional
            Zu optimierendes Experiment. Falls None, wird versucht, für alle
            verfügbaren Referenzdaten zu optimieren.
        method : str, optional
            Optimierungsmethode für scipy.optimize.minimize
            
        Returns
        -------
        dict
            Optimierungsergebnisse
        """
        from scipy.optimize import minimize
        
        # Parameter und Experimente vorbereiten
        if parameters is None:
            if experiment_type == "odmr":
                parameters = ["zero_field_splitting", "fluorescence_contrast", "strain"]
            elif experiment_type == "rabi":
                parameters = ["rabi_frequency", "t2_star", "fluorescence_contrast"]
            elif experiment_type == "ramsey":
                parameters = ["t2_star", "fluorescence_contrast"]
            elif experiment_type == "t1":
                parameters = ["t1"]
            elif experiment_type == "t2":
                parameters = ["t2"]
            else:
                # Standardparameter für alle Experimente
                parameters = ["t1", "t2", "fluorescence_contrast", "zero_field_splitting"]
        
        if experiment_type is None and len(self.reference_data) > 0:
            # Verwende alle verfügbaren Referenzdaten
            experiments = list(self.reference_data.keys())
        elif experiment_type in self.reference_data:
            experiments = [experiment_type]
        else:
            logger.error(f"Keine gültigen Referenzdaten für {experiment_type}")
            return None
        
        # Ursprüngliche Werte sichern
        original_values = {}
        for param in parameters:
            if param == "zero_field_splitting":
                original_values[param] = self.config_loader.get("physical_parameters.zero_field_splitting")
            elif param == "t1":
                original_values[param] = self.config_loader.get("physical_parameters.t1")
            elif param == "t2" or param == "t2_star":
                original_values[param] = self.config_loader.get("physical_parameters.t2")
            elif param == "fluorescence_contrast":
                original_values[param] = self.config_loader.get("optical_properties.fluorescence_contrast")
            elif param == "strain":
                original_values[param] = self.config_loader.get("physical_parameters.transverse_strain")
            elif param == "rabi_frequency":
                original_values[param] = self.config_loader.get("experimental.microwave.power_to_rabi_factor")
        
        # Optimierungsfunktion
        def objective(x):
            # Parameter setzen
            for i, param in enumerate(parameters):
                if param == "zero_field_splitting":
                    self.config_loader.set("physical_parameters.zero_field_splitting", x[i])
                    self.nv_system.config["zero_field_splitting"] = x[i]
                elif param == "t1":
                    self.config_loader.set("physical_parameters.t1", x[i])
                    self.nv_system.config["t1"] = x[i]
                elif param == "t2" or param == "t2_star":
                    self.config_loader.set("physical_parameters.t2", x[i])
                    self.nv_system.config["t2"] = x[i]
                elif param == "fluorescence_contrast":
                    self.config_loader.set("optical_properties.fluorescence_contrast", x[i])
                    self.nv_system.config["fluorescence_contrast"] = x[i]
                elif param == "strain":
                    self.config_loader.set("physical_parameters.transverse_strain", x[i])
                    self.nv_system.config["strain"] = x[i]
                elif param == "rabi_frequency":
                    self.config_loader.set("experimental.microwave.power_to_rabi_factor", x[i])
            
            # Hamiltonian aktualisieren
            self.nv_system.update_hamiltonian()
            
            # Experimente ausführen und Fehler berechnen
            total_error = 0
            for exp in experiments:
                # Experiment ausführen
                if exp == "odmr":
                    ref_freqs = self.reference_data[exp]['frequencies']
                    freq_range = (np.min(ref_freqs), np.max(ref_freqs))
                    num_points = min(51, len(ref_freqs))  # Reduzierte Punktzahl für Optimierung
                    sim_data = self.odmr.run(freq_range, num_points=num_points)
                elif exp == "rabi":
                    ref_times = self.reference_data[exp]['durations']
                    time_range = (np.min(ref_times), np.max(ref_times))
                    num_points = min(31, len(ref_times))  # Reduzierte Punktzahl für Optimierung
                    sim_data = self.rabi.run(time_range, num_points=num_points)
                elif exp == "ramsey":
                    # Ramsey-Experiment
                    continue
                elif exp == "t1":
                    # T1-Experiment
                    continue
                elif exp == "t2":
                    # T2-Experiment
                    continue
                else:
                    continue
                
                # Vergleich berechnen
                comparison = self.compare_to_reference(exp, sim_data)
                if comparison is not None:
                    # NRMSE als Fehlermetrik verwenden
                    total_error += comparison["nrmse"]
            
            return total_error
        
        # Startparameter
        x0 = [original_values[param] for param in parameters]
        
        # Grenzen festlegen
        bounds = []
        for param in parameters:
            if param == "zero_field_splitting":
                bounds.append((2.8e9, 2.9e9))  # 2.8-2.9 GHz
            elif param == "t1":
                bounds.append((1e-6, 10e-3))  # 1µs - 10ms
            elif param == "t2" or param == "t2_star":
                bounds.append((100e-9, 1e-3))  # 100ns - 1ms
            elif param == "fluorescence_contrast":
                bounds.append((0.05, 0.5))  # 5-50% Kontrast
            elif param == "strain":
                bounds.append((0, 10e6))  # 0-10 MHz
            elif param == "rabi_frequency":
                bounds.append((1e4, 1e6))  # 10kHz-1MHz bei 0dBm
        
        # Optimierung durchführen
        logger.info(f"Starte Optimierung für Parameter: {parameters}")
        
        try:
            result = minimize(objective, x0, method=method, bounds=bounds)
            
            # Optimierte Werte zurückgeben
            optimized = {}
            for i, param in enumerate(parameters):
                optimized[param] = result.x[i]
                
                # Werte in die Konfiguration übernehmen
                if param == "zero_field_splitting":
                    self.config_loader.set("physical_parameters.zero_field_splitting", result.x[i])
                elif param == "t1":
                    self.config_loader.set("physical_parameters.t1", result.x[i])
                elif param == "t2" or param == "t2_star":
                    self.config_loader.set("physical_parameters.t2", result.x[i])
                elif param == "fluorescence_contrast":
                    self.config_loader.set("optical_properties.fluorescence_contrast", result.x[i])
                elif param == "strain":
                    self.config_loader.set("physical_parameters.transverse_strain", result.x[i])
                elif param == "rabi_frequency":
                    self.config_loader.set("experimental.microwave.power_to_rabi_factor", result.x[i])
            
            # Hamiltonian mit optimierten Werten aktualisieren
            self.nv_system.update_hamiltonian()
            
            # Konfiguration speichern
            self.config_loader.save_config()
            
            logger.info(f"Optimierung abgeschlossen: {optimized}")
            
            return {
                "optimized_parameters": optimized,
                "original_values": original_values,
                "final_error": result.fun,
                "success": result.success,
                "message": result.message
            }
            
        except Exception as e:
            logger.error(f"Fehler bei Optimierung: {str(e)}")
            
            # Auf ursprüngliche Werte zurücksetzen
            for param, value in original_values.items():
                if param == "zero_field_splitting":
                    self.config_loader.set("physical_parameters.zero_field_splitting", value)
                elif param == "t1":
                    self.config_loader.set("physical_parameters.t1", value)
                elif param == "t2" or param == "t2_star":
                    self.config_loader.set("physical_parameters.t2", value)
                elif param == "fluorescence_contrast":
                    self.config_loader.set("optical_properties.fluorescence_contrast", value)
                elif param == "strain":
                    self.config_loader.set("physical_parameters.transverse_strain", value)
                elif param == "rabi_frequency":
                    self.config_loader.set("experimental.microwave.power_to_rabi_factor", value)
            
            return None