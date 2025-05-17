"""Physikalische Konstanten für NV-Simulationen."""

# Fundamentale Konstanten
PLANCK_CONSTANT = 6.62607015e-34  # J·s
HBAR = 1.0545718e-34  # reduziertes Plancksches Wirkungsquantum in J·s
BOLTZMANN_CONSTANT = 1.380649e-23  # Boltzmann-Konstante in J/K

# Elektronische Konstanten
ELECTRON_CHARGE = 1.602176634e-19  # Elektronenladung in C
ELECTRON_MASS = 9.1093837015e-31  # Elektronenmasse in kg
BOHR_MAGNETON = 9.2740100783e-24  # Bohrsches Magneton in J/T

# NV-spezifische Konstanten
NV_ZERO_FIELD_SPLITTING = 2.87e9  # D in Hz
NV_GYROMAGNETIC_RATIO = 28.025e9  # γ in Hz/T
NV_TRANSVERSE_STRAIN = 0.0  # E in Hz (kann je nach NV variieren)

# Typische NV-Zeitskalen
NV_T1_ROOM_TEMP = 5.0e-3  # T1 bei Raumtemperatur in s
NV_T2_ROOM_TEMP = 1.0e-5  # T2 bei Raumtemperatur in s

# Optische Eigenschaften
NV_ZERO_PHONON_LINE = 637e-9  # Null-Phononen-Linie in m
NV_FLUORESCENCE_CONTRAST = 0.3  # Typischer Fluoreszenzkontrast zwischen ms=0 und ms=±1