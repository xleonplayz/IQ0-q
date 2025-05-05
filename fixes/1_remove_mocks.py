"""
Implementierung zur Entfernung der Mock-Implementation und Sicherstellung
einer direkten SimOS-Abhängigkeit.
"""

import os
import sys
import re

# Pfad zur Originaldatei
PHYSICAL_MODEL_PATH = "../simos_nv_simulator/core/physical_model.py"
FIXED_MODEL_PATH = "../simos_nv_simulator/core/physical_model_fixed.py"

def remove_mock_implementation():
    """Entfernt alle Mock-Implementierungen und macht SimOS zur harten Abhängigkeit"""
    
    with open(PHYSICAL_MODEL_PATH, 'r') as file:
        content = file.read()
    
    # 1. Entferne die SIMOS_AVAILABLE Variable und mache es zur festen Annahme
    content = re.sub(r'SIMOS_AVAILABLE\s*=\s*False', 'SIMOS_AVAILABLE = True', content)
    
    # 2. Entferne den Fallback-Importversuch
    import_pattern = re.compile(r'# Try to import SimOS.*?except ImportError.*?SIMOS_AVAILABLE = False', 
                               re.DOTALL)
    content = re.sub(import_pattern, 
                    '# Import SimOS directly - it\'s a hard dependency\n'
                    'import simos\n'
                    'import simos.propagation\n'
                    'import simos.systems.NV\n'
                    'SIMOS_AVAILABLE = True', content)
    
    # 3. Entferne Mock-Klassen (NumericMock, ArrayMock)
    mock_classes_pattern = re.compile(r'# Create a special class for numeric mocks.*?# 3. Create a wrapper', 
                                    re.DOTALL)
    content = re.sub(mock_classes_pattern, '# 3. Create a wrapper', content)
    
    # 4. Entferne alle bedingten Prüfungen auf Mock
    mock_check_pattern = r'if hasattr\(simos, "_mock_id"\) or "pytest" in sys\.modules:.*?else:'
    content = re.sub(mock_check_pattern, '', content, flags=re.DOTALL)
    
    # 5. Entferne die is_mock Variable in SimOSNVWrapper
    mock_var_pattern = r'self\.is_mock = hasattr\(simos, "_mock_id"\) or "pytest" in sys\.modules'
    content = re.sub(mock_var_pattern, 'self.is_mock = False', content)
    
    # 6. Entferne alle bedingten if self.is_mock Abfragen
    mock_condition_pattern = r'if self\.is_mock:.*?(?=(\s+else:|$))'
    content = re.sub(mock_condition_pattern, '', content, flags=re.DOTALL)
    
    # 7. Entferne SIMOS_AVAILABLE Überprüfungen vor Funktionsaufrufen
    simos_available_check = r'if SIMOS_AVAILABLE and hasattr\(self\.nv_system, \'_rho\'\):.*?(?=(\s+# Create result|$))'
    content = re.sub(simos_available_check, '', content, flags=re.DOTALL)
    
    # 8. Entferne spezifische Mock-Anpassungen aus Testfunktionen
    test_specific_mocks = r'# For the Zeeman tests with mocks, we need to explicitly calculate.*?center_frequency = zfs'
    content = re.sub(test_specific_mocks, 'center_frequency = zfs', content, flags=re.DOTALL)
    
    # 9. Entferne die spezielle Behandlung für den test_simos_unavailable_behavior
    test_unavailable_pattern = r'# For test_simos_unavailable_behavior in test_error_handling.*?raise ImportError\([^)]*\)'
    content = re.sub(test_unavailable_pattern, '# Import error is propagated naturally', content, flags=re.DOTALL)
    
    # Schreibe die bereinigten Inhalte in eine neue Datei
    with open(FIXED_MODEL_PATH, 'w') as file:
        file.write(content)
    
    print(f"Mock-Implementierungen entfernt und in {FIXED_MODEL_PATH} gespeichert.")

if __name__ == "__main__":
    remove_mock_implementation()