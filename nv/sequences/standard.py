import numpy as np
import logging
from nv.sequences.sequence import PulseSequence
from nv.physics import constants
from nv.utils.logging import get_logger

logger = get_logger(__name__)

class StandardSequences:
    """
    Factory für Standard-Pulssequenzen in NV-Experimenten.
    
    Stellt Methoden zum Erstellen häufig verwendeter Pulssequenzen bereit.
    """
    
    @staticmethod
    def rabi_sequence(duration, mw_frequency=None, mw_power=0.0, 
                      laser_init_duration=3e-6, readout_duration=3e-6):
        """
        Erstellt eine Rabi-Oszillationssequenz.
        
        Parameters
        ----------
        duration : float
            Mikrowellenpulsdauer in Sekunden
        mw_frequency : float, optional
            Mikrowellenfrequenz in Hz
        mw_power : float, optional
            Mikrowellenleistung in dBm
        laser_init_duration : float, optional
            Dauer des Initialisierungslaserpulses in Sekunden
        readout_duration : float, optional
            Dauer des Ausleselaserpulses in Sekunden
        
        Returns
        -------
        PulseSequence
            Rabi-Sequenz
        """
        sequence = PulseSequence("Rabi")
        
        # Metadaten
        sequence.add_metadata("experiment", "rabi")
        sequence.add_metadata("mw_power", mw_power)
        if mw_frequency is not None:
            sequence.add_metadata("mw_frequency", mw_frequency)
        
        # Einfache Rabi-Sequenz: Initialisierung, MW-Puls, Auslese
        sequence.add_laser_pulse(laser_init_duration)  # Initialisierung zu ms=0
        sequence.add_wait(1e-6)  # Warten auf Relaxation
        sequence.add_mw_pulse(duration, mw_frequency, mw_power)  # Mikrowellenpuls
        sequence.add_laser_pulse(readout_duration)  # Auslese
        
        logger.info(f"Rabi-Sequenz erstellt: MW-Dauer {duration*1e9:.2f} ns, {mw_power} dBm")
        return sequence
        
    @staticmethod
    def ramsey_sequence(free_time, mw_frequency=None, mw_power=0.0,
                        laser_init_duration=3e-6, readout_duration=3e-6):
        """
        Erstellt eine Ramsey-Sequenz.
        
        Parameters
        ----------
        free_time : float
            Freie Evolutionszeit in Sekunden
        mw_frequency : float, optional
            Mikrowellenfrequenz in Hz
        mw_power : float, optional
            Mikrowellenleistung in dBm
        laser_init_duration : float, optional
            Dauer des Initialisierungslaserpulses in Sekunden
        readout_duration : float, optional
            Dauer des Ausleselaserpulses in Sekunden
        
        Returns
        -------
        PulseSequence
            Ramsey-Sequenz
        """
        sequence = PulseSequence("Ramsey")
        
        # Metadaten
        sequence.add_metadata("experiment", "ramsey")
        sequence.add_metadata("free_time", free_time)
        sequence.add_metadata("mw_power", mw_power)
        if mw_frequency is not None:
            sequence.add_metadata("mw_frequency", mw_frequency)
        
        # π/2-Pulsdauer berechnen
        power_factor = 10**(mw_power/20)
        rabi_freq = 10e6 * power_factor  # Hz
        pi2_time = 1.0 / (4 * rabi_freq)  # s
        
        # Ramsey-Sequenz: Initialisierung, π/2, freie Evolution, π/2, Auslese
        sequence.add_laser_pulse(laser_init_duration)  # Initialisierung zu ms=0
        sequence.add_wait(1e-6)  # Warten auf Relaxation
        sequence.add_mw_pulse(pi2_time, mw_frequency, mw_power)  # Erster π/2-Puls
        sequence.add_wait(free_time)  # Freie Evolution
        sequence.add_mw_pulse(pi2_time, mw_frequency, mw_power)  # Zweiter π/2-Puls
        sequence.add_laser_pulse(readout_duration)  # Auslese
        
        logger.info(f"Ramsey-Sequenz erstellt: Freie Zeit {free_time*1e9:.2f} ns, π/2-Pulse {pi2_time*1e9:.2f} ns")
        return sequence
        
    @staticmethod
    def spin_echo_sequence(free_time, mw_frequency=None, mw_power=0.0,
                           laser_init_duration=3e-6, readout_duration=3e-6):
        """
        Erstellt eine Hahn-Echo-Sequenz.
        
        Parameters
        ----------
        free_time : float
            Gesamte freie Evolutionszeit in Sekunden
        mw_frequency : float, optional
            Mikrowellenfrequenz in Hz
        mw_power : float, optional
            Mikrowellenleistung in dBm
        laser_init_duration : float, optional
            Dauer des Initialisierungslaserpulses in Sekunden
        readout_duration : float, optional
            Dauer des Ausleselaserpulses in Sekunden
        
        Returns
        -------
        PulseSequence
            Spin-Echo-Sequenz
        """
        sequence = PulseSequence("SpinEcho")
        
        # Metadaten
        sequence.add_metadata("experiment", "spin_echo")
        sequence.add_metadata("free_time", free_time)
        sequence.add_metadata("mw_power", mw_power)
        if mw_frequency is not None:
            sequence.add_metadata("mw_frequency", mw_frequency)
        
        # Pulsdauern berechnen
        power_factor = 10**(mw_power/20)
        rabi_freq = 10e6 * power_factor  # Hz
        pi2_time = 1.0 / (4 * rabi_freq)  # s
        pi_time = 1.0 / (2 * rabi_freq)  # s
        
        # Spin-Echo-Sequenz: π/2, τ/2, π, τ/2, π/2, Auslese
        sequence.add_laser_pulse(laser_init_duration)  # Initialisierung zu ms=0
        sequence.add_wait(1e-6)  # Warten auf Relaxation
        sequence.add_mw_pulse(pi2_time, mw_frequency, mw_power)  # Erster π/2-Puls
        sequence.add_wait(free_time/2)  # Erste freie Evolution
        sequence.add_mw_pulse(pi_time, mw_frequency, mw_power)  # π-Puls
        sequence.add_wait(free_time/2)  # Zweite freie Evolution
        sequence.add_mw_pulse(pi2_time, mw_frequency, mw_power)  # Finaler π/2-Puls
        sequence.add_laser_pulse(readout_duration)  # Auslese
        
        logger.info(f"Spin-Echo-Sequenz erstellt: Freie Zeit {free_time*1e9:.2f} ns, π-Puls {pi_time*1e9:.2f} ns")
        return sequence
    
    @staticmethod
    def t1_sequence(wait_time, use_pi_pulse=True, mw_frequency=None, mw_power=0.0,
                    laser_init_duration=3e-6, readout_duration=3e-6):
        """
        Erstellt eine T1-Messungssequenz.
        
        Parameters
        ----------
        wait_time : float
            Wartezeit in Sekunden
        use_pi_pulse : bool, optional
            Ob ein π-Puls verwendet werden soll
        mw_frequency : float, optional
            Mikrowellenfrequenz in Hz
        mw_power : float, optional
            Mikrowellenleistung in dBm
        laser_init_duration : float, optional
            Dauer des Initialisierungslaserpulses in Sekunden
        readout_duration : float, optional
            Dauer des Ausleselaserpulses in Sekunden
        
        Returns
        -------
        PulseSequence
            T1-Sequenz
        """
        sequence = PulseSequence("T1")
        
        # Metadaten
        sequence.add_metadata("experiment", "t1")
        sequence.add_metadata("wait_time", wait_time)
        sequence.add_metadata("use_pi_pulse", use_pi_pulse)
        if use_pi_pulse:
            sequence.add_metadata("mw_power", mw_power)
            if mw_frequency is not None:
                sequence.add_metadata("mw_frequency", mw_frequency)
        
        # Initialisierung
        sequence.add_laser_pulse(laser_init_duration)  # Initialisierung zu ms=0
        sequence.add_wait(1e-6)  # Warten auf Relaxation
        
        # π-Puls (optional)
        if use_pi_pulse:
            # Pulsdauer berechnen
            power_factor = 10**(mw_power/20)
            rabi_freq = 10e6 * power_factor  # Hz
            pi_time = 1.0 / (2 * rabi_freq)  # s
            
            sequence.add_mw_pulse(pi_time, mw_frequency, mw_power)  # π-Puls
        
        # Wartezeit für T1-Relaxation
        sequence.add_wait(wait_time)  # T1-Relaxation
        
        # Auslese
        sequence.add_laser_pulse(readout_duration)  # Auslese
        
        logger.info(f"T1-Sequenz erstellt: Wartezeit {wait_time*1e6:.3f} µs, {'mit' if use_pi_pulse else 'ohne'} π-Puls")
        return sequence
    
    @staticmethod
    def odmr_sequence(mw_frequency, mw_power=0.0, mw_duration=None,
                      laser_init_duration=3e-6, readout_duration=3e-6):
        """
        Erstellt eine einzelne ODMR-Messungssequenz für eine Frequenz.
        
        Parameters
        ----------
        mw_frequency : float
            Mikrowellenfrequenz in Hz
        mw_power : float, optional
            Mikrowellenleistung in dBm
        mw_duration : float, optional
            Mikrowellenpulsdauer in Sekunden. Wenn None, wird π-Pulsdauer berechnet.
        laser_init_duration : float, optional
            Dauer des Initialisierungslaserpulses in Sekunden
        readout_duration : float, optional
            Dauer des Ausleselaserpulses in Sekunden
        
        Returns
        -------
        PulseSequence
            ODMR-Sequenz
        """
        sequence = PulseSequence("ODMR")
        
        # Metadaten
        sequence.add_metadata("experiment", "odmr")
        sequence.add_metadata("mw_frequency", mw_frequency)
        sequence.add_metadata("mw_power", mw_power)
        
        # Wenn keine Dauer angegeben, π-Pulsdauer berechnen
        if mw_duration is None:
            power_factor = 10**(mw_power/20)
            rabi_freq = 10e6 * power_factor  # Hz
            mw_duration = 1.0 / (2 * rabi_freq)  # s
            sequence.add_metadata("mw_duration", mw_duration)
        
        # ODMR-Sequenz: Initialisierung, MW-Puls, Auslese
        sequence.add_laser_pulse(laser_init_duration)  # Initialisierung zu ms=0
        sequence.add_wait(1e-6)  # Warten auf Relaxation
        sequence.add_mw_pulse(mw_duration, mw_frequency, mw_power)  # Mikrowellenpuls
        sequence.add_laser_pulse(readout_duration)  # Auslese
        
        logger.info(f"ODMR-Sequenz erstellt: {mw_frequency/1e9:.6f} GHz, {mw_power} dBm, {mw_duration*1e9:.2f} ns")
        return sequence
    
    @staticmethod
    def xy8_sequence(tau, n_repetitions=1, mw_frequency=None, mw_power=0.0,
                     laser_init_duration=3e-6, readout_duration=3e-6):
        """
        Erstellt eine XY8-Dynamical-Decoupling-Sequenz.
        
        Die XY8-Sequenz besteht aus 8 π-Pulsen mit alternierenden X und Y Phasen.
        
        Parameters
        ----------
        tau : float
            Zeitintervall zwischen Pulsen in Sekunden
        n_repetitions : int, optional
            Anzahl der Wiederholungen des XY8-Blocks
        mw_frequency : float, optional
            Mikrowellenfrequenz in Hz
        mw_power : float, optional
            Mikrowellenleistung in dBm
        laser_init_duration : float, optional
            Dauer des Initialisierungslaserpulses in Sekunden
        readout_duration : float, optional
            Dauer des Ausleselaserpulses in Sekunden
        
        Returns
        -------
        PulseSequence
            XY8-Sequenz
        """
        sequence = PulseSequence("XY8")
        
        # Metadaten
        sequence.add_metadata("experiment", "xy8")
        sequence.add_metadata("tau", tau)
        sequence.add_metadata("n_repetitions", n_repetitions)
        sequence.add_metadata("mw_power", mw_power)
        if mw_frequency is not None:
            sequence.add_metadata("mw_frequency", mw_frequency)
        
        # Pulsdauern berechnen
        power_factor = 10**(mw_power/20)
        rabi_freq = 10e6 * power_factor  # Hz
        pi2_time = 1.0 / (4 * rabi_freq)  # s
        pi_time = 1.0 / (2 * rabi_freq)  # s
        
        # Initialisierung
        sequence.add_laser_pulse(laser_init_duration)  # Initialisierung zu ms=0
        sequence.add_wait(1e-6)  # Warten auf Relaxation
        
        # Erster π/2-Puls (X-Achse)
        sequence.add_mw_pulse(pi2_time, mw_frequency, mw_power, phase=0)
        
        # XY8-Block wiederholen
        for rep in range(n_repetitions):
            # Warten vor erstem Puls
            sequence.add_wait(tau - pi_time/2)
            
            # XY8 Pulssequenz (8 π-Pulse mit alternierenden Phasen)
            # X-Y-X-Y-Y-X-Y-X
            phases = [0, np.pi/2, 0, np.pi/2, np.pi/2, 0, np.pi/2, 0]
            
            for i, phase in enumerate(phases):
                # π-Puls mit gegebener Phase
                sequence.add_mw_pulse(pi_time, mw_frequency, mw_power, phase=phase)
                
                # Warten zwischen Pulsen (oder nach letztem Puls)
                if i < len(phases) - 1:
                    sequence.add_wait(2*tau - pi_time)
                else:
                    sequence.add_wait(tau - pi_time/2)
        
        # Finaler π/2-Puls (X-Achse)
        sequence.add_mw_pulse(pi2_time, mw_frequency, mw_power, phase=0)
        
        # Auslese
        sequence.add_laser_pulse(readout_duration)
        
        logger.info(f"XY8-Sequenz erstellt: τ={tau*1e9:.2f} ns, {n_repetitions} Wiederholungen")
        return sequence