import numpy as np
from nv.physics import constants

def calculate_t1_relaxation_rate(temperature=300.0):
    """
    Berechnet die T1-Relaxationsrate für eine gegebene Temperatur.
    
    Bei höheren Temperaturen wird T1 typischerweise kürzer.
    
    Parameters
    ----------
    temperature : float, optional
        Temperatur in Kelvin
    
    Returns
    -------
    float
        T1-Relaxationsrate in Hz (1/s)
    """
    # Einfaches Temperaturmodell, typische NV-Zentren folgen
    # einer Arrhenius-Kurve mit mehr Relaxation bei höherer Temperatur
    T1_room_temp = constants.NV_T1_ROOM_TEMP
    activation_energy = 0.012  # Typische Aktivierungsenergie in eV
    
    # Boltzmann-Faktor für Temperaturabhängigkeit
    kB_eV = 8.617333262e-5  # Boltzmann-Konstante in eV/K
    
    # Rate bei Raumtemperatur
    rate_room_temp = 1.0 / T1_room_temp
    
    # Rate bei gegebener Temperatur
    rate = rate_room_temp * np.exp((temperature - 300) * activation_energy / (kB_eV * 300 * temperature))
    
    return rate

def calculate_t2_dephasing_rate(c13_concentration=0.011):
    """
    Berechnet die T2-Dephasierungsrate basierend auf der 13C-Konzentration.
    
    T2 wird stark von der Kernspin-Umgebung beeinflusst, insbesondere von 13C-Spins.
    
    Parameters
    ----------
    c13_concentration : float, optional
        13C-Konzentration (natürliche Häufigkeit ist 0.011)
    
    Returns
    -------
    float
        T2-Dephasierungsrate in Hz (1/s)
    """
    # Basisrate bei natürlicher 13C-Häufigkeit (0.011)
    base_rate = 1.0 / constants.NV_T2_ROOM_TEMP
    
    # Skalieren mit der 13C-Konzentration relativ zur natürlichen Häufigkeit
    # Die Skalierung ist ungefähr linear für geringe Konzentrationen
    rate = base_rate * (c13_concentration / 0.011)
    
    return rate

def get_collapse_operators(nv_system, t1=None, t2=None):
    """
    Erstellt Collapse-Operatoren für T1-Relaxation und T2-Dephasierung.
    
    Diese Operatoren können in der Master-Gleichung für offene Quantensysteme verwendet werden.
    
    Parameters
    ----------
    nv_system : SimOS NVSystem
        NV-System-Objekt aus SimOS
    t1 : float, optional
        T1-Relaxationszeit in Sekunden
    t2 : float, optional
        T2-Dephasierungszeit in Sekunden
    
    Returns
    -------
    list
        Liste von Collapse-Operatoren
    """
    c_ops = []
    
    # T1-Relaxation hinzufügen
    if t1 is not None and t1 > 0:
        gamma1 = 1.0 / t1
        c_ops.append(np.sqrt(gamma1) * nv_system.Splus)
        c_ops.append(np.sqrt(gamma1) * nv_system.Sminus)
    
    # T2-Dephasierung hinzufügen (reine Dephasierung, keine Relaxation)
    if t2 is not None and t2 > 0 and t1 is not None and t1 > 0:
        # Berechne reine Dephasierungsrate: 1/T2' = 1/T2 - 1/(2*T1)
        gamma_phi = 1.0/t2 - 1.0/(2*t1)
        if gamma_phi > 0:
            c_ops.append(np.sqrt(gamma_phi) * nv_system.Sz)
    
    return c_ops