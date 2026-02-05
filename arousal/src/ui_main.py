from PySide6.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QWidget,
    QPushButton, QMessageBox, QLineEdit, QFileDialog, QHBoxLayout, QSlider, QProgressBar
)
from PySide6.QtCore import Signal, Qt
from plotter import Plotter
import asyncio
import numpy as np
from data_logger import DataLogger
from collections import deque
import paho.mqtt.client as mqtt
import json
import time

class MainWindow(QMainWindow):
    data_signal = Signal(int, list)

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Arousal Monitor')

        # Stato baseline/calibrazione
        self.calibrating        = False
        self.raw_rr_values      = []      # raw arousal from RR (baseline)
        self.hr_values          = []      # HR values for baseline
        self.baseline_hr        = None    # mean HR during calibration
        self.baseline_rr_min    = None    # fisso: solo baseline
        self.baseline_rr_max    = None    # aggiornato dinamico
        self.baseline_hr_min    = None    # fisso: solo baseline
        self.baseline_hr_max    = None    # aggiornato dinamico

        # Smoothing setup
        self.smoothing_window   = deque(maxlen=10)

        # Recording setup
        self.logger = DataLogger()
        self.recording = False

        # MQTT setup
        self.mqtt_url_edit = QLineEdit()
        self.mqtt_url_edit.setPlaceholderText("mqtt://broker.hivemq.com:1883")
        self.mqtt_btn = QPushButton("Start MQTT")
        self.mqtt_btn.setCheckable(True)
        self.mqtt_btn.setChecked(False)
        self.mqtt_btn.clicked.connect(self.on_mqtt_toggle)
        self.mqtt_client = None
        self.mqtt_enabled = False
        self.mqtt_topic = "sensor/driver/arousal"

        # Widgets
        self.label_status  = QLabel('Status: Disconnected')
        self.label_hr      = QLabel('HR: -- bpm')
        self.label_arousal = QLabel('Arousal: --')
        self.plotter       = Plotter()

        self.slider_label  = QLabel('Smooth window size: 10')
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(1, 10)
        self.smooth_slider.setValue(10)
        self.smooth_slider.valueChanged.connect(self.on_slider_changed)

        self.dir_edit    = QLineEdit()
        self.browse_btn  = QPushButton('Browse...')
        self.browse_btn.clicked.connect(self.on_browse)

        self.record_btn  = QPushButton('Start Recording')
        self.record_btn.clicked.connect(self.on_record)
        self.record_btn.setEnabled(False)

        self.start_btn   = QPushButton('Connect to Coospo')
        self.start_btn.clicked.connect(self.on_start)
        self.reconnect_btn = QPushButton('Reconnect')
        self.reconnect_btn.clicked.connect(self.on_start)
        self.reconnect_btn.setVisible(False)
        self.calibrate_btn = QPushButton('Start Baseline (30s)')
        self.calibrate_btn.clicked.connect(self.on_calibrate)
        self.calibrate_btn.setEnabled(False)

        # Progress bar per la baseline
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(30)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)

        # Layout
        main_layout = QVBoxLayout()
        main_layout.addWidget(self.label_status)
        main_layout.addWidget(self.label_hr)
        main_layout.addWidget(self.label_arousal)
        main_layout.addWidget(self.progress_bar)
        main_layout.addWidget(self.plotter.canvas)

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.slider_label)
        slider_layout.addWidget(self.smooth_slider)
        main_layout.addLayout(slider_layout)

        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.dir_edit)
        dir_layout.addWidget(self.browse_btn)
        main_layout.addLayout(dir_layout)

        # MQTT layout
        mqtt_layout = QHBoxLayout()
        mqtt_layout.addWidget(QLabel("MQTT broker URL:"))
        mqtt_layout.addWidget(self.mqtt_url_edit)
        mqtt_layout.addWidget(self.mqtt_btn)
        main_layout.addLayout(mqtt_layout)

        main_layout.addWidget(self.record_btn)
        main_layout.addWidget(self.start_btn)
        main_layout.addWidget(self.reconnect_btn)
        main_layout.addWidget(self.calibrate_btn)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        self.data_signal.connect(self.handle_new_data)

    def on_slider_changed(self, value: int):
        self.slider_label.setText(f'Smooth window size: {value}')
        recent = list(self.smoothing_window)[-value:]
        self.smoothing_window = deque(recent, maxlen=value)

    def set_monitor(self, monitor):
        self.monitor = monitor

    async def try_connect(self):
        self.start_btn.setEnabled(False)
        self.reconnect_btn.setVisible(False)
        self.label_status.setText('Status: Connecting…')
        try:
            await self.monitor.connect_and_notify()
            self.label_status.setText('Status: Connected')
            self.calibrate_btn.setEnabled(True)
            self.record_btn.setEnabled(True)
        except Exception as e:
            self.label_status.setText('Status: Disconnected')
            self.show_error(str(e))
            self.reconnect_btn.setVisible(True)
            self.start_btn.setEnabled(True)

    def on_start(self):
        asyncio.create_task(self.try_connect())

    def on_browse(self):
        path = QFileDialog.getExistingDirectory(self, 'Select Directory')
        if path:
            self.dir_edit.setText(path)

    def on_record(self):
        if not self.recording:
            directory = self.dir_edit.text().strip()
            if not directory:
                QMessageBox.warning(self, 'Warning', 'Select a directory first')
                return
            try:
                self.logger.start(directory)
                self.recording = True
                self.record_btn.setText('Stop Recording')
            except Exception as e:
                QMessageBox.critical(self, 'Error', str(e))
        else:
            self.logger.stop()
            self.recording = False
            self.record_btn.setText('Start Recording')

    def on_calibrate(self):
        self.calibrate_btn.setEnabled(False)
        self.start_btn.setEnabled(False)
        self.reconnect_btn.setVisible(False)
        self.label_status.setText('Status: Calibrating…')
        self.raw_rr_values = []
        self.hr_values     = []
        self.calibrating   = True
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        asyncio.create_task(self._run_baseline())

    async def _run_baseline(self):
        for t in range(31):  # da 0 a 30
            self.progress_bar.setValue(t)
            await asyncio.sleep(1)

        if self.hr_values:
            self.baseline_hr = float(np.mean(self.hr_values))

        rr_valids = [v for v in self.raw_rr_values if v > 0]
        hr_arousal_vals = [abs(h - self.baseline_hr) / self.baseline_hr for h in self.hr_values] if self.baseline_hr else []

        if rr_valids:
            self.baseline_rr_min, self.baseline_rr_max = min(rr_valids), max(rr_valids)
            if self.baseline_rr_min == self.baseline_rr_max:
                self.baseline_rr_max += 0.02
        else:
            self.baseline_rr_min = self.baseline_rr_max = None

        if hr_arousal_vals:
            self.baseline_hr_min, self.baseline_hr_max = min(hr_arousal_vals), max(hr_arousal_vals)
            if self.baseline_hr_min == self.baseline_hr_max:
                self.baseline_hr_max += 0.02
        else:
            self.baseline_hr_min = self.baseline_hr_max = None

        self.calibrating = False
        self.progress_bar.setVisible(False)

        status = 'Connected' if getattr(self.monitor, 'client', None) and self.monitor.client.is_connected else 'Disconnected'
        self.label_status.setText(f'Status: {status}')
        self.start_btn.setEnabled(status == 'Disconnected')
        self.calibrate_btn.setEnabled(status == 'Connected')

    def on_mqtt_toggle(self):
        if self.mqtt_btn.isChecked():
            url = self.mqtt_url_edit.text().strip()
            if not url:
                QMessageBox.warning(self, "Warning", "Insert a MQTT broker URL first")
                self.mqtt_btn.setChecked(False)
                return
            try:
                # Parsing mqtt://host:port/path o host:port/path
                if url.startswith("mqtt://"):
                    url = url[len("mqtt://"):]
                if '/' in url:
                    host_port, base_topic = url.split('/', 1)
                    base_topic = '/' + base_topic
                else:
                    host_port = url
                    base_topic = ''
                if ':' in host_port:
                    host, port = host_port.split(':', 1)
                    port = int(port)
                else:
                    host = host_port
                    port = 1883
                self.mqtt_topic = f"{base_topic}/sensor/driver/arousal".replace("//","/") if base_topic else "sensor/driver/arousal"
                self.mqtt_client = mqtt.Client()
                self.mqtt_client.connect(host, port, 60)
                self.mqtt_enabled = True
                self.mqtt_btn.setText("Stop MQTT")
            except Exception as e:
                QMessageBox.critical(self, "MQTT Error", str(e))
                self.mqtt_btn.setChecked(False)
                self.mqtt_enabled = False
                self.mqtt_client = None
        else:
            if self.mqtt_client:
                try:
                    self.mqtt_client.disconnect()
                except Exception:
                    pass
            self.mqtt_client = None
            self.mqtt_enabled = False
            self.mqtt_btn.setText("Start MQTT")

    def handle_new_data(self, hr: int, rr_list: list):
        from_rr = isinstance(rr_list, list) and len(rr_list) >= 2
        rmssd = raw_rr = raw_hr_arousal = None

        if from_rr:
            diffs = np.diff(rr_list)
            rmssd = float(np.sqrt(np.mean(diffs ** 2)))
            raw_rr = 1.0 / rmssd if rmssd > 0 else 0.0
            if self.calibrating:
                self.raw_rr_values.append(raw_rr)
            if not self.calibrating and raw_rr is not None:
                if self.baseline_rr_max is None or raw_rr > self.baseline_rr_max:
                    self.baseline_rr_max = raw_rr

        # Popola SEMPRE hr_values durante calibrazione
        if self.calibrating:
            self.hr_values.append(hr)

        if self.baseline_hr:
            raw_hr_arousal = abs(hr - self.baseline_hr) / self.baseline_hr
            if not self.calibrating and raw_hr_arousal is not None:
                if self.baseline_hr_max is None or raw_hr_arousal > self.baseline_hr_max:
                    self.baseline_hr_max = raw_hr_arousal

        if self.calibrating:
            return

        norm = raw_rr if from_rr else raw_hr_arousal
        baseline_min = self.baseline_rr_min if from_rr else self.baseline_hr_min
        baseline_max = self.baseline_rr_max if from_rr else self.baseline_hr_max
        if baseline_min is not None and baseline_max is not None:
            denom = baseline_max - baseline_min
            norm = (norm - baseline_min) / denom if denom > 0 else 0.0
        arousal_norm = float(np.clip(norm, 0.0, 1.0)) if norm else 0.0

        self.smoothing_window.append(arousal_norm)
        arousal_smooth = float(sum(self.smoothing_window) / len(self.smoothing_window))

        self.label_hr.setText(f'HR: {hr} bpm')
        color = 'green' if from_rr else 'orange'
        self.label_arousal.setStyleSheet(f'color: {color};')
        self.label_arousal.setText(f'Arousal: {arousal_smooth:.2f}')
        self.plotter.update(arousal_smooth)

        if self.recording:
            method = 'actual' if from_rr else 'estimated'
            self.logger.log(hr, arousal_smooth, method, rmssd)

        # --- MQTT publish ---
        if self.mqtt_enabled and self.mqtt_client is not None:
            try:
                millis = int(time.time() * 1000)
                rr_value = rr_list[-1] if (isinstance(rr_list, list) and len(rr_list) >= 2) else ""
                msg = json.dumps({"msg":{
                    "arousal": float(arousal_smooth), "timestamp": millis, "hr": int(hr) if hr is not None else "", "rr": rr_value}})
                # Chiama loop_start solo se non già chiamato
                self.mqtt_client.loop_start()
                self.mqtt_client.publish(self.mqtt_topic, msg)
                self.mqtt_client.loop_stop()
            except Exception as e:
                print("MQTT publish error:", e)

    def show_error(self, message: str):
        QMessageBox.critical(self, 'Errore', message)

    def closeEvent(self, event):
        asyncio.create_task(self.monitor.disconnect_and_cleanup())
        if self.recording:
            self.logger.stop()
        if self.mqtt_client:
            try:
                self.mqtt_client.disconnect()
            except Exception:
                pass
        super().closeEvent(event)
