import logging
import sys
import os
from datetime import datetime

def setup_logger(log_level=logging.INFO, log_file=None):
    """
    Richtet das Logging-System ein.
    
    Parameters
    ----------
    log_level : int, optional
        Logging-Level (z.B. logging.DEBUG, logging.INFO)
    log_file : str, optional
        Dateiname für Log-Ausgabe. Wenn None, nur Konsolen-Logging.
    
    Returns
    -------
    logging.Logger
        Konfigurierter Logger
    """
    # Root-Logger konfigurieren
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Formatter für ausführliche Informationen
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Konsolen-Handler hinzufügen
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Datei-Handler hinzufügen falls angefordert
    if log_file:
        # Sicherstellen, dass Verzeichnis existiert
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # SimOS-Logger konfigurieren
    simos_logger = logging.getLogger('sim.simos')
    simos_logger.setLevel(log_level)
    
    # Simulator-Logger konfigurieren
    simulator_logger = logging.getLogger('nv')
    simulator_logger.setLevel(log_level)
    
    return simulator_logger

def get_logger(name):
    """
    Erhalte einen benannten Logger.
    
    Parameters
    ----------
    name : str
        Logger-Name
    
    Returns
    -------
    logging.Logger
        Konfigurierter Logger
    """
    return logging.getLogger(name)