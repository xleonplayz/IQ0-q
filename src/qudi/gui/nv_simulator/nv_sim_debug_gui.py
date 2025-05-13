# -*- coding: utf-8 -*-

"""
This file contains a GUI for debugging the NV simulator functionality.

Copyright (c) 2023, IQO

This file is part of qudi.

Qudi is free software: you can redistribute it and/or modify it under the terms of
the GNU Lesser General Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

Qudi is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along with qudi.
If not, see <https://www.gnu.org/licenses/>.
"""

import os
import numpy as np
import time
import datetime
from PySide2 import QtCore, QtWidgets, QtGui

from qudi.core.connector import Connector
from qudi.core.configoption import ConfigOption
from qudi.util.colordefs import QudiPalettePale as palette
from qudi.core.module import GuiBase
from qudi.util.paths import get_artwork_dir
from qudi.hardware.nv_simulator.qudi_facade import QudiFacade

class NVSimDebugLogTableModel(QtCore.QAbstractTableModel):
    """Table model for displaying log entries"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.log_entries = []
        self.headers = ['Timestamp', 'Type', 'Message']
        
    def rowCount(self, parent=None):
        return len(self.log_entries)
        
    def columnCount(self, parent=None):
        return len(self.headers)
        
    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid():
            return None
            
        if role == QtCore.Qt.DisplayRole:
            row, col = index.row(), index.column()
            entry = self.log_entries[row]
            
            if col == 0:
                return entry['timestamp']
            elif col == 1:
                return entry['type']
            elif col == 2:
                return entry['message']
                
        elif role == QtCore.Qt.BackgroundRole:
            entry_type = self.log_entries[index.row()]['type']
            if entry_type == 'ERROR':
                return QtGui.QColor(255, 200, 200)
            elif entry_type == 'WARNING':
                return QtGui.QColor(255, 255, 200)
            elif entry_type == 'INFO':
                return QtGui.QColor(200, 255, 200)
                
        return None
        
    def headerData(self, section, orientation, role):
        if role == QtCore.Qt.DisplayRole and orientation == QtCore.Qt.Horizontal:
            return self.headers[section]
        return None
        
    def add_log_entry(self, entry_type, message):
        timestamp = datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]
        self.beginInsertRows(QtCore.QModelIndex(), len(self.log_entries), len(self.log_entries))
        self.log_entries.append({
            'timestamp': timestamp,
            'type': entry_type,
            'message': message
        })
        self.endInsertRows()
        
        # Keep only the last 1000 entries to avoid performance issues
        if len(self.log_entries) > 1000:
            self.beginRemoveRows(QtCore.QModelIndex(), 0, 0)
            self.log_entries.pop(0)
            self.endRemoveRows()
            
    def clear(self):
        self.beginResetModel()
        self.log_entries = []
        self.endResetModel()

class NVSimDebugMainWindow(QtWidgets.QMainWindow):
    """Main window for the NV simulator debugging GUI"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setWindowTitle('qudi: NV Simulator Debug')
        self.resize(1000, 800)
        
        # Create central widget and layout
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout()
        central_widget.setLayout(layout)
        
        # Create menu bar
        menu_bar = QtWidgets.QMenuBar(self)
        self.setMenuBar(menu_bar)
        
        menu = menu_bar.addMenu('File')
        self.action_close = QtWidgets.QAction('Close')
        path = os.path.join(get_artwork_dir(), 'icons', 'application-exit')
        self.action_close.setIcon(QtGui.QIcon(path))
        self.action_close.triggered.connect(self.close)
        menu.addAction(self.action_close)
        
        # Create status bar
        status_bar = QtWidgets.QStatusBar(self)
        status_bar.setStyleSheet('QStatusBar::item { border: 0px}')
        self.setStatusBar(status_bar)
        
        # Add tabs for different debug views
        self.tab_widget = QtWidgets.QTabWidget()
        layout.addWidget(self.tab_widget)
        
        # Add log tab
        self.log_tab = QtWidgets.QWidget()
        self.tab_widget.addTab(self.log_tab, "Simulator Log")
        log_layout = QtWidgets.QVBoxLayout()
        self.log_tab.setLayout(log_layout)
        
        # Controls for log tab
        log_controls = QtWidgets.QHBoxLayout()
        self.clear_log_button = QtWidgets.QPushButton("Clear Log")
        log_controls.addWidget(self.clear_log_button)
        log_controls.addStretch(1)
        log_layout.addLayout(log_controls)
        
        # Log table
        self.log_table = QtWidgets.QTableView()
        self.log_table_model = NVSimDebugLogTableModel()
        self.log_table.setModel(self.log_table_model)
        self.log_table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.log_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.log_table.setAlternatingRowColors(True)
        self.log_table.setSortingEnabled(False)
        log_layout.addWidget(self.log_table)
        
        # Add simulator state tab
        self.state_tab = QtWidgets.QWidget()
        self.tab_widget.addTab(self.state_tab, "Simulator State")
        state_layout = QtWidgets.QVBoxLayout()
        self.state_tab.setLayout(state_layout)
        
        # Group for microwave state
        mw_group = QtWidgets.QGroupBox("Microwave State")
        mw_layout = QtWidgets.QFormLayout()
        mw_group.setLayout(mw_layout)
        self.mw_frequency_label = QtWidgets.QLabel("0 Hz")
        self.mw_power_label = QtWidgets.QLabel("0 dBm")
        self.mw_state_label = QtWidgets.QLabel("OFF")
        mw_layout.addRow("Frequency:", self.mw_frequency_label)
        mw_layout.addRow("Power:", self.mw_power_label)
        mw_layout.addRow("State:", self.mw_state_label)
        state_layout.addWidget(mw_group)
        
        # Group for scan state
        scan_group = QtWidgets.QGroupBox("Scan State")
        scan_layout = QtWidgets.QFormLayout()
        scan_group.setLayout(scan_layout)
        self.scan_status_label = QtWidgets.QLabel("INACTIVE")
        self.scan_index_label = QtWidgets.QLabel("0")
        self.scan_frequencies_label = QtWidgets.QLabel("None")
        scan_layout.addRow("Status:", self.scan_status_label)
        scan_layout.addRow("Current Index:", self.scan_index_label)
        scan_layout.addRow("Frequencies:", self.scan_frequencies_label)
        state_layout.addWidget(scan_group)
        
        # Group for laser state
        laser_group = QtWidgets.QGroupBox("Laser State")
        laser_layout = QtWidgets.QFormLayout()
        laser_group.setLayout(laser_layout)
        self.laser_power_label = QtWidgets.QLabel("0 mW")
        self.laser_state_label = QtWidgets.QLabel("OFF")
        laser_layout.addRow("Power:", self.laser_power_label)
        laser_layout.addRow("State:", self.laser_state_label)
        state_layout.addWidget(laser_group)
        
        # Group for ODMR signal simulation
        odmr_group = QtWidgets.QGroupBox("ODMR Signal")
        odmr_layout = QtWidgets.QFormLayout()
        odmr_group.setLayout(odmr_layout)
        self.resonance_label = QtWidgets.QLabel("2.87 GHz ± 0 MHz")
        self.zeeman_shift_label = QtWidgets.QLabel("0 MHz")
        self.magnetic_field_label = QtWidgets.QLabel("0 Gauss")
        self.contrast_label = QtWidgets.QLabel("0%")
        odmr_layout.addRow("Resonance:", self.resonance_label)
        odmr_layout.addRow("Zeeman Shift:", self.zeeman_shift_label)
        odmr_layout.addRow("Magnetic Field:", self.magnetic_field_label)
        odmr_layout.addRow("Contrast:", self.contrast_label)
        state_layout.addWidget(odmr_group)
        
        state_layout.addStretch(1)
        
        # Connect signals
        self.clear_log_button.clicked.connect(self.log_table_model.clear)

class NVSimDebugLogic(QtCore.QObject):
    """Logic class for the NV simulator debug GUI"""
    
    sigStateUpdated = QtCore.Signal(dict)
    sigLogEntry = QtCore.Signal(str, str)
    
    def __init__(self, parent=None, nv_simulator=None):
        super().__init__(parent)
        self._nv_simulator = nv_simulator
        self._timer = None
        self._update_interval = 200  # ms
        
    def start_monitoring(self):
        """Start monitoring the NV simulator state"""
        if self._timer is None:
            self._timer = QtCore.QTimer()
            self._timer.timeout.connect(self._update_state)
            
        if not self._timer.isActive():
            self._timer.start(self._update_interval)
            self.log_info("Started monitoring NV simulator")
            
    def stop_monitoring(self):
        """Stop monitoring the NV simulator state"""
        if self._timer is not None and self._timer.isActive():
            self._timer.stop()
            self.log_info("Stopped monitoring NV simulator")
            
    def log_info(self, message):
        """Log an info message"""
        self.sigLogEntry.emit("INFO", message)
        
    def log_warning(self, message):
        """Log a warning message"""
        self.sigLogEntry.emit("WARNING", message)
        
    def log_error(self, message):
        """Log an error message"""
        self.sigLogEntry.emit("ERROR", message)
        
    def _update_state(self):
        """Update the state information from the NV simulator"""
        try:
            if self._nv_simulator is None:
                return
                
            # Get values from shared state
            state = {}
            
            try:
                # Microwave state
                state['mw_frequency'] = self._nv_simulator._shared_state.get('current_mw_frequency', 0)
                state['mw_power'] = self._nv_simulator._shared_state.get('current_mw_power', 0)
                state['mw_on'] = self._nv_simulator._shared_state.get('current_mw_on', False)
                
                # Laser state
                state['laser_power'] = self._nv_simulator._shared_state.get('current_laser_power', 0)
                state['laser_on'] = self._nv_simulator._shared_state.get('current_laser_on', False)
                
                # Scan state
                state['scanning_active'] = self._nv_simulator._shared_state.get('scanning_active', False)
                state['scan_index'] = self._nv_simulator._shared_state.get('current_scan_index', 0)
                state['scan_frequencies'] = self._nv_simulator._shared_state.get('scan_frequencies', None)
            except Exception as e:
                self.log_error(f"Error getting values from shared state: {str(e)}")
            
            try:
                # Physics parameters
                if hasattr(self._nv_simulator, 'nv_model') and self._nv_simulator.nv_model is not None:
                    b_field = self._nv_simulator.nv_model.b_field
                    # Convert Tesla to Gauss (1 T = 10,000 G)
                    field_strength_gauss = np.linalg.norm(b_field) * 10000.0
                    state['magnetic_field'] = field_strength_gauss
                    
                    # Zeeman splitting (~2.8 MHz/G)
                    zeeman_shift = 2.8e6 * field_strength_gauss  # field in G, shift in Hz
                    state['zeeman_shift'] = zeeman_shift
                    
                    # ODMR resonances
                    resonance_freq = 2.87e9  # Zero-field splitting (Hz)
                    state['resonance'] = resonance_freq
                    state['dip1_center'] = resonance_freq - zeeman_shift
                    state['dip2_center'] = resonance_freq + zeeman_shift
            except Exception as e:
                self.log_error(f"Error calculating physics parameters: {str(e)}")
                
            self.sigStateUpdated.emit(state)
            
        except Exception as e:
            self.log_error(f"Error updating state: {str(e)}")

class NVSimDebugGui(GuiBase):
    """Main GUI class for NV simulator debugging.
    
    Example config for copy-paste:
    
    nv_sim_debug_gui:
        module.Class: 'nv_simulator.nv_sim_debug_gui.NVSimDebugGui'
        connect:
            simulator: nv_simulator
    """
    
    # declare connectors
    _simulator = Connector(name='simulator', interface='MicrowaveInterface')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._mw = None  # type: NVSimDebugMainWindow
        self._logic = None  # type: NVSimDebugLogic
        
    def on_activate(self):
        """Initialization performed during activation of the module."""
        simulator = self._simulator()
        
        # Create debug logic
        self._logic = NVSimDebugLogic(parent=self, nv_simulator=simulator)
        
        # Create main window
        self._mw = NVSimDebugMainWindow()
        
        # Connect signals
        self._logic.sigStateUpdated.connect(self._state_updated)
        self._logic.sigLogEntry.connect(self._mw.log_table_model.add_log_entry)
        
        # Add initial log entry
        self._logic.log_info("NV Simulator Debug GUI activated")
        
        # Start monitoring
        self._logic.start_monitoring()
        
        self.show()
        
    def on_deactivate(self):
        """Deinitialisation performed during deactivation of the module."""
        # Stop monitoring
        self._logic.stop_monitoring()
        
        # Disconnect signals
        self._logic.sigStateUpdated.disconnect(self._state_updated)
        
        # Close window
        self._mw.close()
        
    def show(self):
        """Make window visible and put it above all other windows."""
        self._mw.show()
        self._mw.raise_()
        self._mw.activateWindow()
        
    def _state_updated(self, state):
        """Update the GUI with the current state"""
        # Format values with SI prefixes
        if 'mw_frequency' in state:
            freq_hz = state['mw_frequency']
            if freq_hz > 1e9:
                self._mw.mw_frequency_label.setText(f"{freq_hz/1e9:.6f} GHz")
            elif freq_hz > 1e6:
                self._mw.mw_frequency_label.setText(f"{freq_hz/1e6:.6f} MHz")
            else:
                self._mw.mw_frequency_label.setText(f"{freq_hz:.1f} Hz")
                
        if 'mw_power' in state:
            self._mw.mw_power_label.setText(f"{state['mw_power']:.2f} dBm")
            
        if 'mw_on' in state:
            self._mw.mw_state_label.setText("ON" if state['mw_on'] else "OFF")
            
        if 'laser_power' in state:
            self._mw.laser_power_label.setText(f"{state['laser_power']:.2f} mW")
            
        if 'laser_on' in state:
            self._mw.laser_state_label.setText("ON" if state['laser_on'] else "OFF")
            
        if 'scanning_active' in state:
            self._mw.scan_status_label.setText("ACTIVE" if state['scanning_active'] else "INACTIVE")
            
        if 'scan_index' in state:
            self._mw.scan_index_label.setText(str(state['scan_index']))
            
        if 'scan_frequencies' in state and state['scan_frequencies'] is not None:
            try:
                if hasattr(state['scan_frequencies'], '__len__') and len(state['scan_frequencies']) > 10:
                    freq_text = f"{len(state['scan_frequencies'])} points"
                    if state['scanning_active'] and 'scan_index' in state:
                        try:
                            scan_index = state['scan_index']
                            if 0 <= scan_index < len(state['scan_frequencies']):
                                current_freq = state['scan_frequencies'][scan_index]
                                freq_text += f", current: {current_freq/1e9:.6f} GHz"
                        except (IndexError, TypeError) as e:
                            self._logic.log_error(f"Error accessing scan frequency: {str(e)}")
                    self._mw.scan_frequencies_label.setText(freq_text)
                else:
                    self._mw.scan_frequencies_label.setText(str(state['scan_frequencies']))
            except Exception as e:
                self._logic.log_error(f"Error processing scan frequencies: {str(e)}")
                self._mw.scan_frequencies_label.setText("Error")
        else:
            self._mw.scan_frequencies_label.setText("None")
            
        # Update ODMR physics parameters
        if 'zeeman_shift' in state:
            self._mw.zeeman_shift_label.setText(f"{state['zeeman_shift']/1e6:.2f} MHz")
            
        if 'resonance' in state and 'zeeman_shift' in state:
            central_freq = state['resonance'] / 1e9  # GHz
            shift = state['zeeman_shift'] / 1e6  # MHz
            self._mw.resonance_label.setText(f"{central_freq:.2f} GHz ± {shift:.2f} MHz")
            
        if 'magnetic_field' in state:
            self._mw.magnetic_field_label.setText(f"{state['magnetic_field']:.2f} Gauss")
            
        # Calculate and display contrast based on current frequency
        if 'mw_frequency' in state and 'dip1_center' in state and 'dip2_center' in state:
            linewidth = 20e6  # 20 MHz linewidth (realistic for ODMR in diamond)
            max_contrast = 0.3  # 30% contrast (typical for NV centers)
            
            freq = state['mw_frequency']
            dip1 = max_contrast * linewidth**2 / ((freq - state['dip1_center'])**2 + linewidth**2)
            dip2 = max_contrast * linewidth**2 / ((freq - state['dip2_center'])**2 + linewidth**2)
            
            # Convert to percent
            contrast_percent = (dip1 + dip2) * 100
            self._mw.contrast_label.setText(f"{contrast_percent:.2f}%")
            
            # Log if we're close to a resonance
            if (abs(freq - state['dip1_center']) < 20e6 or 
                abs(freq - state['dip2_center']) < 20e6) and state['mw_on']:
                self._logic.log_info(f"Near resonance at {freq/1e9:.6f} GHz (contrast: {contrast_percent:.2f}%)")