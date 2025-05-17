#!/usr/bin/env python
"""
Kalibrierungsskript für den digitalen Twin eines NV-Zentrums.
Lädt Referenzdaten, optimiert die Parameter und speichert die kalibrierte Konfiguration.
"""

import os
import sys
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Pfad zur Projektbasis hinzufügen
script_dir = Path(__file__).resolve().parent
project_dir = script_dir
sys.path.insert(0, str(project_dir))

from nv.digital_twin import DigitalTwin

# Logging konfigurieren
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Kommandozeilenargumente parsen."""
    parser = argparse.ArgumentParser(description="Kalibrierungsskript für den digitalen Twin eines NV-Zentrums")
    parser.add_argument("-c", "--config", type=str, default="config.json",
                        help="Pfad zur Konfigurationsdatei (relativ zum NV-Paket)")
    parser.add_argument("-e", "--experiment", type=str, choices=["odmr", "rabi", "ramsey", "t1", "t2", "all"],
                        default="all", help="Zu kalibrierendes Experiment")
    parser.add_argument("-p", "--parameters", type=str, nargs="+",
                        help="Zu kalibrierende Parameter (kommagetrennt)")
    parser.add_argument("-o", "--output", type=str,
                        help="Ausgabepfad für kalibrierte Konfiguration")
    parser.add_argument("--plot", action="store_true", 
                        help="Vergleichsplots erstellen")
    parser.add_argument("--optimize", action="store_true",
                        help="Parameteroptimierung durchführen")
    parser.add_argument("--method", type=str, default="Powell",
                        help="Optimierungsmethode für scipy.optimize.minimize")
    return parser.parse_args()

def main():
    """Hauptfunktion für die Kalibrierung."""
    args = parse_args()
    
    # Vollständigen Pfad zur Konfigurationsdatei erstellen
    config_path = os.path.join(project_dir, "nv", args.config)
    
    # Twin initialisieren
    logger.info(f"Initialisiere digitalen Twin mit Konfiguration: {config_path}")
    twin = DigitalTwin(config_path=config_path)
    
    # Verfügbare Referenzdaten anzeigen
    logger.info(f"Verfügbare Referenzdaten: {list(twin.reference_data.keys())}")
    
    # Parameter aus Argumenten oder aus Konfiguration bestimmen
    parameters = args.parameters
    if parameters is None:
        # Parameter aus fitted_parameters in Konfiguration verwenden
        parameters = [p.get("name") for p in twin.calibration_info.get("fitted_parameters", [])]
    logger.info(f"Zu kalibrierende Parameter: {parameters}")
    
    # Experimente bestimmen
    if args.experiment == "all":
        experiments = list(twin.reference_data.keys())
    else:
        experiments = [args.experiment]
    logger.info(f"Zu kalibrierende Experimente: {experiments}")
    
    # Kalibrierung durchführen
    results = {}
    for exp_type in experiments:
        if exp_type not in twin.reference_data:
            logger.warning(f"Keine Referenzdaten für Experiment: {exp_type}")
            continue
            
        logger.info(f"Kalibriere Parameter für Experiment: {exp_type}")
        
        # Parameter für dieses Experiment filtern
        exp_params = [p for p in parameters if any(
            param.get("name") == p and param.get("reference_experiment") == exp_type
            for param in twin.calibration_info.get("fitted_parameters", [])
        )]
        
        if not exp_params:
            logger.warning(f"Keine Parameter für Experiment {exp_type} definiert")
            continue
            
        # Kalibrierung durchführen
        for param in exp_params:
            value = twin.calibrate_parameter(param, exp_type)
            if value is not None:
                results.setdefault(exp_type, {})[param] = value
                logger.info(f"Parameter {param} kalibriert: {value}")
                
        # Optimierung durchführen, falls gewünscht
        if args.optimize:
            logger.info(f"Optimiere Parameter für Experiment {exp_type}")
            opt_result = twin.optimize_twin(parameters=exp_params, experiment_type=exp_type, method=args.method)
            if opt_result:
                results.setdefault(exp_type, {})["optimization"] = opt_result
                logger.info(f"Optimierte Parameter: {opt_result['optimized_parameters']}")
                logger.info(f"Fehler nach Optimierung: {opt_result['final_error']}")
        
        # Plot erstellen, falls gewünscht
        if args.plot:
            logger.info(f"Erstelle Vergleichsplot für Experiment {exp_type}")
            try:
                fig, ax = twin.plot_comparison(exp_type)
                if fig:
                    # Ausgabeverzeichnis für Plots erstellen
                    plot_dir = os.path.join(project_dir, "plots")
                    os.makedirs(plot_dir, exist_ok=True)
                    
                    # Plot speichern
                    plot_path = os.path.join(plot_dir, f"{exp_type}_comparison.png")
                    fig.savefig(plot_path)
                    logger.info(f"Plot gespeichert unter: {plot_path}")
                    plt.close(fig)
            except Exception as e:
                logger.error(f"Fehler beim Erstellen des Plots: {str(e)}")
    
    # Konfiguration speichern
    if args.output:
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(project_dir, output_path)
        twin.config_loader.save_config(output_path)
        logger.info(f"Kalibrierte Konfiguration gespeichert unter: {output_path}")
    else:
        twin.config_loader.save_config()
        logger.info(f"Kalibrierte Konfiguration gespeichert unter: {twin.config_loader.config_path}")
    
    # Zusammenfassung ausgeben
    logger.info("Kalibrierung abgeschlossen:")
    for exp_type, exp_results in results.items():
        logger.info(f"  {exp_type}:")
        for param, value in exp_results.items():
            if param != "optimization":
                logger.info(f"    {param}: {value}")

if __name__ == "__main__":
    main()