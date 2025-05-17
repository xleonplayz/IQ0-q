import numpy as np
import logging
from nv.sequences.sequence import PulseSequence
from nv.sequences.pulse import Pulse
from nv.utils.logging import get_logger

logger = get_logger(__name__)

class CustomSequenceBuilder:
    """
    Builder für benutzerdefinierte Pulssequenzen.
    
    Stellt erweiterte Methoden zum Erstellen komplexer Pulssequenzen bereit.
    """
    
    def __init__(self, name="custom"):
        """
        Initialisiert einen neuen SequenceBuilder.
        
        Parameters
        ----------
        name : str, optional
            Name der Sequenz
        """
        self.sequence = PulseSequence(name)
        logger.debug(f"CustomSequenceBuilder für Sequenz '{name}' erstellt")
    
    def add_pi_pulse(self, mw_frequency=None, mw_power=0.0, phase=0.0):
        """
        Fügt einen π-Puls zur Sequenz hinzu.
        
        Parameters
        ----------
        mw_frequency : float, optional
            Mikrowellenfrequenz in Hz
        mw_power : float, optional
            Mikrowellenleistung in dBm
        phase : float, optional
            Phase des Pulses in Radiant
        
        Returns
        -------
        CustomSequenceBuilder
            self für Method-Chaining
        """
        # Pulsdauer berechnen
        power_factor = 10**(mw_power/20)
        rabi_freq = 10e6 * power_factor  # Hz
        pi_time = 1.0 / (2 * rabi_freq)  # s
        
        # π-Puls hinzufügen
        params = {"power": mw_power, "phase": phase}
        if mw_frequency is not None:
            params["frequency"] = mw_frequency
            
        self.sequence.add_pulse(Pulse("mw", pi_time, **params))
        logger.debug(f"π-Puls ({pi_time*1e9:.2f} ns) hinzugefügt")
        return self
    
    def add_pi_half_pulse(self, mw_frequency=None, mw_power=0.0, phase=0.0):
        """
        Fügt einen π/2-Puls zur Sequenz hinzu.
        
        Parameters
        ----------
        mw_frequency : float, optional
            Mikrowellenfrequenz in Hz
        mw_power : float, optional
            Mikrowellenleistung in dBm
        phase : float, optional
            Phase des Pulses in Radiant
        
        Returns
        -------
        CustomSequenceBuilder
            self für Method-Chaining
        """
        # Pulsdauer berechnen
        power_factor = 10**(mw_power/20)
        rabi_freq = 10e6 * power_factor  # Hz
        pi_half_time = 1.0 / (4 * rabi_freq)  # s
        
        # π/2-Puls hinzufügen
        params = {"power": mw_power, "phase": phase}
        if mw_frequency is not None:
            params["frequency"] = mw_frequency
            
        self.sequence.add_pulse(Pulse("mw", pi_half_time, **params))
        logger.debug(f"π/2-Puls ({pi_half_time*1e9:.2f} ns) hinzugefügt")
        return self
    
    def add_initialization(self, duration=3e-6, power=1.0):
        """
        Fügt eine Standard-Initialisierungssequenz hinzu.
        
        Parameters
        ----------
        duration : float, optional
            Dauer des Laserpulses in Sekunden
        power : float, optional
            Laserleistung in mW
        
        Returns
        -------
        CustomSequenceBuilder
            self für Method-Chaining
        """
        # Initialisierungspulse hinzufügen
        self.sequence.add_laser_pulse(duration, power)
        self.sequence.add_wait(1e-6)  # Standard-Wartezeit nach Initialisierung
        logger.debug(f"Initialisierung hinzugefügt (Laser: {duration*1e6:.2f} µs, {power} mW)")
        return self
    
    def add_readout(self, duration=3e-6, power=1.0):
        """
        Fügt eine Standard-Auslesesequenz hinzu.
        
        Parameters
        ----------
        duration : float, optional
            Dauer des Laserpulses in Sekunden
        power : float, optional
            Laserleistung in mW
        
        Returns
        -------
        CustomSequenceBuilder
            self für Method-Chaining
        """
        # Auslesepulse hinzufügen
        self.sequence.add_laser_pulse(duration, power)
        logger.debug(f"Auslese hinzugefügt (Laser: {duration*1e6:.2f} µs, {power} mW)")
        return self
    
    def add_dynamical_decoupling(self, tau, n_pulses, mode="xy", 
                                mw_frequency=None, mw_power=0.0):
        """
        Fügt eine dynamische Entkopplungssequenz hinzu.
        
        Parameters
        ----------
        tau : float
            Zeit zwischen π-Pulsen in Sekunden
        n_pulses : int
            Anzahl der π-Pulse
        mode : str, optional
            Modus der Pulssequenz: "xy" (alternierend) oder "x" (alle gleich)
        mw_frequency : float, optional
            Mikrowellenfrequenz in Hz
        mw_power : float, optional
            Mikrowellenleistung in dBm
        
        Returns
        -------
        CustomSequenceBuilder
            self für Method-Chaining
        """
        # Parameter validieren
        if n_pulses <= 0:
            raise ValueError(f"Anzahl der Pulse muss positiv sein: {n_pulses}")
        
        if mode not in ["xy", "x"]:
            raise ValueError(f"Ungültiger Modus: {mode}, erlaubt sind 'xy' oder 'x'")
        
        # π-Pulsdauer berechnen
        power_factor = 10**(mw_power/20)
        rabi_freq = 10e6 * power_factor  # Hz
        pi_time = 1.0 / (2 * rabi_freq)  # s
        
        # Parameter für MW-Pulse
        base_params = {"power": mw_power}
        if mw_frequency is not None:
            base_params["frequency"] = mw_frequency
        
        # Warten vor erstem Puls
        self.sequence.add_wait(tau/2)
        
        # Pulse hinzufügen
        for i in range(n_pulses):
            # Phase berechnen
            if mode == "xy":
                phase = 0.0 if i % 2 == 0 else np.pi/2  # Abwechselnd X und Y
            else:
                phase = 0.0  # Immer X
                
            # π-Puls mit berechneter Phase
            pulse_params = base_params.copy()
            pulse_params["phase"] = phase
            self.sequence.add_pulse(Pulse("mw", pi_time, **pulse_params))
            
            # Warten zwischen Pulsen (außer nach dem letzten)
            if i < n_pulses - 1:
                self.sequence.add_wait(tau)
        
        # Warten nach letztem Puls
        self.sequence.add_wait(tau/2)
        
        logger.debug(f"Dynamische Entkopplung hinzugefügt: {n_pulses} {mode}-Pulse, τ={tau*1e9:.2f} ns")
        return self
    
    def build(self):
        """
        Erstellt die fertige Pulssequenz.
        
        Returns
        -------
        PulseSequence
            Die erstellte Pulssequenz
        """
        logger.info(f"Pulssequenz '{self.sequence.name}' erstellt mit {len(self.sequence.pulses)} Pulsen")
        return self.sequence


def create_cpmg_sequence(tau, n_pulses, mw_frequency=None, mw_power=0.0):
    """
    Erstellt eine CPMG-Dynamical-Decoupling-Sequenz.
    
    Die CPMG-Sequenz besteht aus π/2-Puls(X), n π-Pulse(Y), π/2-Puls(X).
    
    Parameters
    ----------
    tau : float
        Zeitintervall zwischen π-Pulsen in Sekunden
    n_pulses : int
        Anzahl der π-Pulse
    mw_frequency : float, optional
        Mikrowellenfrequenz in Hz
    mw_power : float, optional
        Mikrowellenleistung in dBm
    
    Returns
    -------
    PulseSequence
        CPMG-Sequenz
    """
    builder = CustomSequenceBuilder("CPMG")
    
    # Metadaten hinzufügen
    builder.sequence.add_metadata("experiment", "cpmg")
    builder.sequence.add_metadata("tau", tau)
    builder.sequence.add_metadata("n_pulses", n_pulses)
    builder.sequence.add_metadata("mw_power", mw_power)
    if mw_frequency is not None:
        builder.sequence.add_metadata("mw_frequency", mw_frequency)
    
    # Initialisierung
    builder.add_initialization()
    
    # Erster π/2-Puls (X-Phase)
    builder.add_pi_half_pulse(mw_frequency, mw_power, phase=0)
    
    # Dynamische Entkopplung mit Y-Pulsen
    builder.add_dynamical_decoupling(tau, n_pulses, mode="x", 
                                    mw_frequency=mw_frequency,
                                    mw_power=mw_power)
    
    # Finaler π/2-Puls (X-Phase)
    builder.add_pi_half_pulse(mw_frequency, mw_power, phase=0)
    
    # Auslese
    builder.add_readout()
    
    logger.info(f"CPMG-Sequenz erstellt: {n_pulses} Pulse, τ={tau*1e9:.2f} ns")
    return builder.build()


def create_correlation_spectroscopy_sequence(tau_1, tau_2, mw_frequency=None, mw_power=0.0):
    """
    Erstellt eine Korrelationsspektroskopie-Sequenz.
    
    Die Sequenz besteht aus:
    π/2 - τ₁ - π - τ₁ - π/2 - τ₂ - π/2 - τ₁ - π - τ₁ - π/2
    
    Parameters
    ----------
    tau_1 : float
        Erste Verzögerungszeit in Sekunden
    tau_2 : float
        Zweite Verzögerungszeit in Sekunden
    mw_frequency : float, optional
        Mikrowellenfrequenz in Hz
    mw_power : float, optional
        Mikrowellenleistung in dBm
    
    Returns
    -------
    PulseSequence
        Korrelationsspektroskopie-Sequenz
    """
    builder = CustomSequenceBuilder("Correlation")
    
    # Metadaten hinzufügen
    builder.sequence.add_metadata("experiment", "correlation_spectroscopy")
    builder.sequence.add_metadata("tau_1", tau_1)
    builder.sequence.add_metadata("tau_2", tau_2)
    builder.sequence.add_metadata("mw_power", mw_power)
    if mw_frequency is not None:
        builder.sequence.add_metadata("mw_frequency", mw_frequency)
    
    # Initialisierung
    builder.add_initialization()
    
    # Erste Spin-Echo-Sequenz
    builder.add_pi_half_pulse(mw_frequency, mw_power)  # π/2
    builder.sequence.add_wait(tau_1)                  # τ₁
    builder.add_pi_pulse(mw_frequency, mw_power)       # π
    builder.sequence.add_wait(tau_1)                  # τ₁
    builder.add_pi_half_pulse(mw_frequency, mw_power)  # π/2
    
    # Korrelationszeit
    builder.sequence.add_wait(tau_2)                  # τ₂
    
    # Zweite Spin-Echo-Sequenz
    builder.add_pi_half_pulse(mw_frequency, mw_power)  # π/2
    builder.sequence.add_wait(tau_1)                  # τ₁
    builder.add_pi_pulse(mw_frequency, mw_power)       # π
    builder.sequence.add_wait(tau_1)                  # τ₁
    builder.add_pi_half_pulse(mw_frequency, mw_power)  # π/2
    
    # Auslese
    builder.add_readout()
    
    logger.info(f"Korrelationsspektroskopie-Sequenz erstellt: τ₁={tau_1*1e9:.2f} ns, τ₂={tau_2*1e9:.2f} ns")
    return builder.build()