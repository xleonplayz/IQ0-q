"""
NV-Zentrum-Simulator basierend auf SimOS.

Dieses Paket implementiert einen physikalisch akkuraten Simulator für
NV-Zentren in Diamant, mit voller Unterstützung für Quantendynamik.
"""

# SimOS-Pfad zur Python-Umgebung hinzufügen
import os
import sys
simos_paths = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../trash/ui/simos')),
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../simos')),
]
for simos_path in simos_paths:
    if os.path.exists(simos_path) and simos_path not in sys.path:
        sys.path.insert(0, simos_path)

__version__ = "0.1.0"

# Standardkonfiguration des Loggings
from nv.utils.logging import setup_logger

# Logger einrichten
setup_logger()