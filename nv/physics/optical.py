import numpy as np
from nv.physics import constants
from nv.core.quantum_state import QuantumState

def calculate_fluorescence(ms0_population, count_rate=100000.0):
    """
    Berechnet die Fluoreszenzrate basierend auf der ms=0-Population.
    
    NV-Zentren haben eine höhere Fluoreszenz im ms=0-Zustand als in ms=±1.
    
    Parameters
    ----------
    ms0_population : float
        Population im ms=0-Zustand (0 bis 1)
    count_rate : float, optional
        Basis-Zählrate in Zählungen/s für ms=0
    
    Returns
    -------
    float
        Fluoreszenzrate in Zählungen/s
    """
    # Fluoreszenzkontrast zwischen ms=0 und ms=±1
    contrast = constants.NV_FLUORESCENCE_CONTRAST
    
    # Fluoreszenzrate berechnen
    # Höhere Fluoreszenz für ms=0, niedrigere für ms=±1
    return count_rate * (1.0 - contrast * (1.0 - ms0_population))

def apply_optical_pumping(state, nv_system, laser_power, method="qutip"):
    """
    Wendet optische Pumping-Effekte auf einen Quantenzustand an.
    
    Laserlicht pumpt NV-Zentren in den ms=0-Zustand aufgrund des
    Intersystem-Crossing-Prozesses.
    
    Parameters
    ----------
    state : QuantumState
        Aktueller Quantenzustand
    nv_system : SimOS NVSystem
        NV-System-Objekt aus SimOS
    laser_power : float
        Laserleistung in mW
    method : str, optional
        Numerische Methode (qutip, numpy, sparse)
    
    Returns
    -------
    QuantumState
        Neuer Quantenzustand nach optischem Pumpen
    """
    # Pumpstärke basierend auf Laserleistung berechnen
    pump_rate = min(1.0, laser_power / 0.5)  # Sättigt bei ~0.5 mW
    
    # Wenn keine Leistung, keine Änderung
    if pump_rate <= 0:
        return state
    
    # Neuen Zustand mit mehr ms=0-Population erzeugen
    ms0_projector = nv_system.Sp[0]
    thermal_dm = QuantumState(ms0_projector.unit(), method=method)
    
    # Aktuellen Zustand mit ms=0-Zustand mischen
    from sim.simos.simos import core
    simos_state = state.get_simos_state()
    mixed_state = (1-pump_rate) * simos_state + pump_rate * thermal_dm.get_simos_state()
    
    # Normalisieren und Zustand aktualisieren
    return QuantumState(mixed_state / core.trace(mixed_state), method=method)

def get_fluorescence_photons(fluorescence_rate, measurement_time, poisson_noise=True):
    """
    Berechnet die Anzahl der Fluoreszenzphotonen für eine gegebene Messzeit.
    
    Parameters
    ----------
    fluorescence_rate : float
        Fluoreszenzrate in Photonen/s
    measurement_time : float
        Messzeit in Sekunden
    poisson_noise : bool, optional
        Ob Poisson-Rauschen hinzugefügt werden soll
    
    Returns
    -------
    int or float
        Anzahl der detektierten Photonen
    """
    # Mittlere Anzahl von Photonen
    mean_counts = fluorescence_rate * measurement_time
    
    if poisson_noise:
        # Poisson-Rauschen für Photonenzählung hinzufügen
        return np.random.poisson(mean_counts)
    else:
        return mean_counts