import asyncio
import time
import json
import numpy as np
from collections import deque

from PySide6.QtWidgets import (
    QMainWindow, QLabel, QVBoxLayout, QWidget,
    QPushButton, QMessageBox, QLineEdit, QFileDialog,
    QHBoxLayout, QSlider, QProgressBar, QComboBox
)
from PySide6.QtCore import Signal, Qt

import paho.mqtt.client as mqtt

from plotter import Plotter
from data_logger import DataLogger


# -----------------------------
# Physiological constants
# -----------------------------
RR_WINDOW_SEC = 60.0
MIN_RR_SAMPLES = 20
MAX_RR_JUMP   = 0.25
AROUSAL_ZMAX  = 2.5
MIN_RR_MS     = 300
MAX_RR_MS     = 2000

MIN_LOG_RMSSD_STD = 0.15   # physiological floor
MIN_HR_STD        = 1.0    # bpm floor


class MainWindow(QMainWindow):
    data_signal = Signal(int, list)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Arousal Monitor")

        # -----------------------------
        # Baseline state
        # -----------------------------
        self.calibrating = False
        self.hr_baseline_vals = []
        self.rmssd_baseline_vals = []

        self.hr_mean = None
        self.hr_std  = None
        self.rmssd_mean = None
        self.rmssd_std  = None

        # -----------------------------
        # RR buffering
        # -----------------------------
        self.rr_buffer = deque()
        self.last_rr_quality = "N/A"

        # -----------------------------
        # Smoothing
        # -----------------------------
        self.smoothing_window = deque(maxlen=10)
        self.smooth_method = "mean"

        # -----------------------------
        # Recording
        # -----------------------------
        self.logger = DataLogger()
        self.recording = False

        # -----------------------------
        # MQTT
        # -----------------------------
        self.mqtt_client = None
        self.mqtt_enabled = False
        self.mqtt_topic = "sensor/driver/arousal"

        # -----------------------------
        # UI
        # -----------------------------
        self.label_status  = QLabel("Status: Disconnected")
        self.label_hr      = QLabel("HR: -- bpm")
        self.label_arousal = QLabel("Arousal: --")
        self.label_quality = QLabel("Signal quality: N/A")

        self.plotter = Plotter()

        self.slider_label  = QLabel("Smooth window: 10")
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(3, 20)
        self.smooth_slider.setValue(10)
        self.smooth_slider.valueChanged.connect(self.on_slider_changed)

        self.smooth_combo = QComboBox()
        self.smooth_combo.addItems(["Mean", "Median"])
        self.smooth_combo.currentTextChanged.connect(self.on_smooth_method_changed)

        self.dir_edit   = QLineEdit()
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self.on_browse)

        self.record_btn = QPushButton("Start Recording")
        self.record_btn.clicked.connect(self.on_record)
        self.record_btn.setEnabled(False)

        self.start_btn = QPushButton("Connect to Coospo")
        self.start_btn.clicked.connect(self.on_start)

        self.reconnect_btn = QPushButton("Reconnect")
        self.reconnect_btn.clicked.connect(self.on_start)
        self.reconnect_btn.setVisible(False)

        self.calibrate_btn = QPushButton("Start Baseline (60s)")
        self.calibrate_btn.clicked.connect(self.on_calibrate)
        self.calibrate_btn.setEnabled(False)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 60)
        self.progress_bar.setVisible(False)

        self.mqtt_url_edit = QLineEdit()
        self.mqtt_url_edit.setPlaceholderText("mqtt://broker.hivemq.com:1883")
        self.mqtt_btn = QPushButton("Start MQTT")
        self.mqtt_btn.setCheckable(True)
        self.mqtt_btn.clicked.connect(self.on_mqtt_toggle)

        main = QVBoxLayout()
        main.addWidget(self.label_status)
        main.addWidget(self.label_hr)
        main.addWidget(self.label_arousal)
        main.addWidget(self.label_quality)
        main.addWidget(self.progress_bar)
        main.addWidget(self.plotter.canvas)

        srow = QHBoxLayout()
        srow.addWidget(self.slider_label)
        srow.addWidget(self.smooth_slider)
        srow.addWidget(QLabel("Method:"))
        srow.addWidget(self.smooth_combo)
        main.addLayout(srow)

        drow = QHBoxLayout()
        drow.addWidget(QLabel("Save to:"))
        drow.addWidget(self.dir_edit)
        drow.addWidget(self.browse_btn)
        main.addLayout(drow)

        mqtt_row = QHBoxLayout()
        mqtt_row.addWidget(QLabel("MQTT:"))
        mqtt_row.addWidget(self.mqtt_url_edit)
        mqtt_row.addWidget(self.mqtt_btn)
        main.addLayout(mqtt_row)

        main.addWidget(self.record_btn)
        main.addWidget(self.start_btn)
        main.addWidget(self.reconnect_btn)
        main.addWidget(self.calibrate_btn)

        container = QWidget()
        container.setLayout(main)
        self.setCentralWidget(container)

        self.data_signal.connect(self.handle_new_data)

    # --------------------------------------------------
    # UI handlers
    # --------------------------------------------------
    def on_slider_changed(self, value: int):
        self.slider_label.setText(f"Smooth window: {value}")
        recent = list(self.smoothing_window)[-value:]
        self.smoothing_window = deque(recent, maxlen=value)

    def on_smooth_method_changed(self, text: str):
        self.smooth_method = text.lower()

    def set_monitor(self, monitor):
        self.monitor = monitor

    async def try_connect(self):
        self.start_btn.setEnabled(False)
        self.reconnect_btn.setVisible(False)
        self.label_status.setText("Status: Connecting…")
        try:
            await self.monitor.connect_and_notify()
            self.label_status.setText("Status: Connected")
            self.calibrate_btn.setEnabled(True)
            self.record_btn.setEnabled(True)
        except Exception as e:
            self.label_status.setText("Status: Disconnected")
            QMessageBox.critical(self, "Error", str(e))
            self.reconnect_btn.setVisible(True)
            self.start_btn.setEnabled(True)

    def on_start(self):
        asyncio.create_task(self.try_connect())

    def on_browse(self):
        path = QFileDialog.getExistingDirectory(self, "Select Directory")
        if path:
            self.dir_edit.setText(path)

    def on_record(self):
        if not self.recording:
            if not self.dir_edit.text():
                QMessageBox.warning(self, "Warning", "Select a directory first")
                return
            self.logger.start(self.dir_edit.text())
            self.recording = True
            self.record_btn.setText("Stop Recording")
        else:
            self.logger.stop()
            self.recording = False
            self.record_btn.setText("Start Recording")

    def on_mqtt_toggle(self):
        if self.mqtt_btn.isChecked():
            url = self.mqtt_url_edit.text().strip()
            if not url:
                QMessageBox.warning(self, "Warning", "Enter MQTT broker URL")
                self.mqtt_btn.setChecked(False)
                return
            
            try:
                # Parse mqtt://host:port/topic
                if url.startswith("mqtt://"):
                    url = url[7:]
                
                if '/' in url:
                    host_port, base_topic = url.split('/', 1)
                else:
                    host_port, base_topic = url, ""
                
                if ':' in host_port:
                    host, port = host_port.rsplit(':', 1)
                    port = int(port)
                else:
                    host, port = host_port, 1883

                self.mqtt_topic = f"{base_topic}/sensor/driver/arousal".strip('/') if base_topic else "sensor/driver/arousal"
                
                self.mqtt_client = mqtt.Client()
                self.mqtt_client.connect(host, port, 60)
                self.mqtt_client.loop_start()
                self.mqtt_enabled = True
                self.mqtt_btn.setText("Stop MQTT")
                
            except Exception as e:
                QMessageBox.critical(self, "MQTT Error", str(e))
                self.mqtt_btn.setChecked(False)
                self.mqtt_enabled = False
        else:
            if self.mqtt_client:
                try:
                    self.mqtt_client.loop_stop()
                    self.mqtt_client.disconnect()
                except Exception:
                    pass
            self.mqtt_client = None
            self.mqtt_enabled = False
            self.mqtt_btn.setText("Start MQTT")

    # --------------------------------------------------
    # Calibration
    # --------------------------------------------------
    def on_calibrate(self):
        self.calibrating = True
        self.hr_baseline_vals.clear()
        self.rmssd_baseline_vals.clear()
        self.rr_buffer.clear()

        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.label_status.setText("Status: Calibrating… (sit still, breathe normally)")

        asyncio.create_task(self._baseline_task())

    async def _baseline_task(self):
        for t in range(61):
            self.progress_bar.setValue(t)
            await asyncio.sleep(1)

        self.calibrating = False
        self.progress_bar.setVisible(False)

        # HR baseline: median + MAD-based robust std
        if len(self.hr_baseline_vals) > 10:
            self.hr_mean = float(np.median(self.hr_baseline_vals))
            hr_vals = np.asarray(self.hr_baseline_vals, dtype=float)
            mad_hr = np.median(np.abs(hr_vals - self.hr_mean))
            self.hr_std = max(1.4826 * mad_hr, MIN_HR_STD)

        # RMSSD baseline: log-space, median + MAD-based robust std
        if len(self.rmssd_baseline_vals) > 10:
            self.rmssd_mean = float(np.median(self.rmssd_baseline_vals))
            rmssd_vals = np.asarray(self.rmssd_baseline_vals, dtype=float)
            mad_rmssd = np.median(np.abs(rmssd_vals - self.rmssd_mean))
            self.rmssd_std = max(1.4826 * mad_rmssd, MIN_LOG_RMSSD_STD)

        parts = []
        if self.hr_mean is not None:
            parts.append("HR")
        if self.rmssd_mean is not None:
            parts.append("HRV")

        status = "Baseline: " + ", ".join(parts) if parts else "Baseline incomplete"

        self.label_status.setText(f"Status: Connected ({status})")

    # --------------------------------------------------
    # Signal processing
    # --------------------------------------------------
    def _update_rr_buffer(self, rr_list):
        """Update RR buffer with physiological bounds checking"""
        now = time.time()
        for rr in rr_list:
            if MIN_RR_MS <= rr <= MAX_RR_MS:
                self.rr_buffer.append((now, rr))

        # Sliding window: keep last RR_WINDOW_SEC seconds
        cutoff = now - RR_WINDOW_SEC
        while self.rr_buffer and self.rr_buffer[0][0] < cutoff:
            self.rr_buffer.popleft()

    def _compute_rmssd(self):
        if len(self.rr_buffer) < MIN_RR_SAMPLES:
            self.last_rr_quality = f"Insufficient ({len(self.rr_buffer)}/{MIN_RR_SAMPLES})"
            return None
        rr = np.array([v for _, v in self.rr_buffer], dtype=float)

        diffs = np.diff(rr)
        rel_jumps = np.abs(diffs) / rr[:-1]
        valid = rel_jumps < MAX_RR_JUMP

        clean_count = int(valid.sum())
        if clean_count < 10:
            self.last_rr_quality = f"Too noisy ({clean_count} clean)"
            return None

        rmssd = np.sqrt(np.mean(diffs[valid] ** 2))

        quality_pct = 100 * clean_count / len(diffs)
        self.last_rr_quality = f"{quality_pct:.0f}% clean ({len(rr)} samples)"

        return float(rmssd)


    def _smooth_arousal(self, value):
        """Apply smoothing (mean or median)"""
        self.smoothing_window.append(value)
        if self.smooth_method == "median":
            return float(np.median(self.smoothing_window))
        return float(np.mean(self.smoothing_window))

    # --------------------------------------------------
    # Main data handler
    # --------------------------------------------------
    def handle_new_data(self, hr: int, rr_list: list):
        """Process incoming HR and RR intervals"""
        self.label_hr.setText(f"HR: {hr} bpm")

        # Update RR buffer if available
        if rr_list:
            self._update_rr_buffer(rr_list)
        
        print(rr_list)

        # Compute RMSSD (raw value in ms)
        rmssd_raw = self._compute_rmssd()
        
        # Log-transform for processing (RMSSD is log-normal)
        rmssd_log = None
        if rmssd_raw is not None and rmssd_raw > 0:
            rmssd_log = np.log(rmssd_raw)

        self.label_quality.setText(f"Signal quality: {self.last_rr_quality}")

        # Collect baseline samples (log-transformed RMSSD)
        if self.calibrating:
            self.hr_baseline_vals.append(hr)
            if rmssd_log is not None:
                self.rmssd_baseline_vals.append(rmssd_log)
            return

        # Compute arousal z-score
        arousal = None
        method = "estimated"

        # Primary: HRV-based (inverted: low HRV → high arousal)
        if rmssd_log is not None and self.rmssd_mean is not None:
            z = (rmssd_log - self.rmssd_mean) / self.rmssd_std
            arousal = -z  # Invert relationship
            method = "actual"
        
        # Fallback: HR-based
        elif self.hr_mean is not None and self.hr_std is not None:
            z = (hr - self.hr_mean) / self.hr_std
            arousal = z  # High HR → high arousal
            method = "estimated"

        if arousal is None:
            self.label_arousal.setText("Arousal: -- (no baseline)")
            return

        # Smooth then normalize to [0, 1]
        smooth = self._smooth_arousal(arousal)
        normalized = np.clip((smooth + AROUSAL_ZMAX) / (2 * AROUSAL_ZMAX), 0.0, 1.0)

        # Update UI
        color = "green" if method == "actual" else "orange"
        self.label_arousal.setStyleSheet(f"color: {color}; font-weight: bold;")
        self.label_arousal.setText(f"Arousal: {normalized:.2f} ({method})")

        self.plotter.update(normalized)

        # Log data (use raw RMSSD for interpretability)
        if self.recording:
            self.logger.log(hr, normalized, method, rmssd_raw)

        # Publish to MQTT
        if self.mqtt_enabled and self.mqtt_client:
            try:
                payload = {
                    "arousal": float(normalized),
                    "timestamp": int(time.time() * 1000),
                    "hr": int(hr),
                    "rmssd": float(rmssd_raw) if rmssd_raw else None,
                    "method": method,
                    "quality": self.last_rr_quality
                }
                self.mqtt_client.publish(self.mqtt_topic, json.dumps(payload))
            except Exception as e:
                print(f"MQTT publish error: {e}")

    def closeEvent(self, event):
        """Cleanup on window close"""
        if hasattr(self, 'monitor'):
            asyncio.create_task(self.monitor.disconnect_and_cleanup())
        if self.recording:
            self.logger.stop()
        if self.mqtt_client:
            try:
                self.mqtt_client.loop_stop()
                self.mqtt_client.disconnect()
            except Exception:
                pass
        super().closeEvent(event)