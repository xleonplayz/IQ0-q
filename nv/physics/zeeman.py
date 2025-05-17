import numpy as np
from nv.physics import constants

def calculate_zeeman_shift(magnetic_field):
    """
    Berechnet Zeeman-Verschiebung für ein Magnetfeld.
    
    Parameters
    ----------
    magnetic_field : list or ndarray
        Magnetfeldvektor [Bx, By, Bz] in Tesla
    
    Returns
    -------
    tuple
        (f1, f2) Resonanzfrequenzen für ms=0↔ms=-1 und ms=0↔ms=+1 in Hz
    """
    # Magnetfeldstärke berechnen
    field_magnitude = np.linalg.norm(magnetic_field)
    
    # Gyromagnetisches Verhältnis des NV-Zentrums
    gyro = constants.NV_GYROMAGNETIC_RATIO
    
    # Zeeman-Verschiebung berechnen
    zeeman_shift = gyro * field_magnitude
    
    # Nullfeldaufspaltung
    zfs = constants.NV_ZERO_FIELD_SPLITTING
    
    # Resonanzfrequenzen berechnen
    f_minus = zfs - zeeman_shift  # ms=0 ↔ ms=-1
    f_plus = zfs + zeeman_shift   # ms=0 ↔ ms=+1
    
    return (f_minus, f_plus)

def calculate_magnetic_field_from_resonances(f_minus, f_plus):
    """
    Berechnet Magnetfeld aus den Resonanzfrequenzen.
    
    Parameters
    ----------
    f_minus : float
        Resonanzfrequenz für ms=0↔ms=-1 in Hz
    f_plus : float
        Resonanzfrequenz für ms=0↔ms=+1 in Hz
    
    Returns
    -------
    float
        Magnetfeldstärke in Tesla
    """
    # Zeeman-Aufspaltung berechnen
    splitting = f_plus - f_minus
    
    # Magnetfeld aus Aufspaltung berechnen
    field_magnitude = splitting / (2 * constants.NV_GYROMAGNETIC_RATIO)
    
    return field_magnitude