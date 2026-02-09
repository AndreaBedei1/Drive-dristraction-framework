from __future__ import annotations

"""BLE heart-rate -> arousal MQTT publisher (headless)."""

import asyncio
import json
import math
import threading
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

try:
    from bleak import BleakScanner, BleakClient
except Exception:  # pragma: no cover - handled at runtime
    BleakScanner = None
    BleakClient = None

try:
    import paho.mqtt.client as mqtt
except Exception:  # pragma: no cover - handled at runtime
    mqtt = None

from src.arousal_provider import _parse_mqtt_url, _build_arousal_topic


SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

RR_WINDOW_SEC = 60.0
MIN_RR_SAMPLES = 20
MAX_RR_JUMP = 0.25
AROUSAL_ZMAX = 2.5
MIN_RR_MS = 300
MAX_RR_MS = 2000
MIN_LOG_RMSSD_STD = 0.15
MIN_HR_STD = 1.0


def _robust_stats(values: List[float], min_std: float) -> Optional[Tuple[float, float]]:
    if len(values) < 10:
        return None
    vals = sorted(float(v) for v in values)
    mid = len(vals) // 2
    if len(vals) % 2 == 0:
        median = 0.5 * (vals[mid - 1] + vals[mid])
    else:
        median = vals[mid]
    abs_dev = [abs(v - median) for v in vals]
    abs_dev.sort()
    if len(abs_dev) % 2 == 0:
        mad = 0.5 * (abs_dev[mid - 1] + abs_dev[mid])
    else:
        mad = abs_dev[mid]
    std = max(1.4826 * mad, float(min_std))
    return float(median), float(std)


def _parse_hrm_packet(data: bytearray) -> Tuple[Optional[int], List[float]]:
    """Parse HR Measurement packet. Returns (hr_bpm, rr_ms_list)."""
    if not data or len(data) < 2:
        return None, []
    flags = data[0]
    hr_16 = bool(flags & 0x01)
    rr_present = bool(flags & 0x10)

    idx = 1
    if hr_16:
        if len(data) < 3:
            return None, []
        hr = int.from_bytes(data[idx:idx + 2], "little")
        idx += 2
    else:
        hr = int(data[idx])
        idx += 1

    rr_list: List[float] = []
    if rr_present:
        while idx + 1 < len(data):
            rr_raw = int.from_bytes(data[idx:idx + 2], "little")
            rr_ms = (rr_raw / 1024.0) * 1000.0
            rr_list.append(rr_ms)
            idx += 2
    return hr, rr_list


class BleArousalPublisher(threading.Thread):
    """Connect to BLE HR sensor, compute arousal, publish to MQTT."""

    def __init__(
        self,
        device_name: str,
        mqtt_url: str,
        mqtt_topic: str = "",
        baseline_seconds: int = 60,
        smoothing_window: int = 10,
        reconnect_seconds: float = 5.0,
        arousal_zmax: float = AROUSAL_ZMAX,
    ) -> None:
        super().__init__(daemon=True)
        self._device_name = str(device_name or "Coospo")
        self._mqtt_url = str(mqtt_url or "").strip()
        self._mqtt_topic_override = str(mqtt_topic or "").strip()
        self._baseline_seconds = max(10, int(baseline_seconds))
        self._smoothing_window = max(3, int(smoothing_window))
        self._reconnect_seconds = max(1.0, float(reconnect_seconds))
        self._arousal_zmax = float(arousal_zmax)

        self._stop_event = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()
        self._last_error: Optional[str] = None

        self._mqtt_client = None
        self._mqtt_topic = ""

        self._rr_buffer: Deque[Tuple[float, float]] = deque()
        self._last_rr_quality = "N/A"
        self._smoothing: Deque[float] = deque(maxlen=self._smoothing_window)

        self._calibrating = True
        self._baseline_start = time.monotonic()
        self._hr_baseline_vals: List[float] = []
        self._rmssd_baseline_vals: List[float] = []
        self._hr_mean: Optional[float] = None
        self._hr_std: Optional[float] = None
        self._rmssd_mean: Optional[float] = None
        self._rmssd_std: Optional[float] = None

    def last_error(self) -> Optional[str]:
        return self._last_error

    def stop(self) -> None:
        self._stop_event.set()
        if self._loop is not None:
            try:
                self._loop.call_soon_threadsafe(lambda: None)
            except Exception:
                pass

    def run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_async())
        finally:
            self._loop.close()

    async def _run_async(self) -> None:
        if BleakScanner is None or BleakClient is None:
            self._last_error = "bleak_not_available"
            return
        if mqtt is None:
            self._last_error = "paho-mqtt not available"
            return
        if not self._mqtt_url:
            self._last_error = "empty_mqtt_url"
            return

        if not self._setup_mqtt():
            return

        while not self._stop_event.is_set():
            device = None
            try:
                device = await BleakScanner.find_device_by_filter(self._device_filter, timeout=10.0)
            except Exception as exc:
                self._last_error = f"ble_scan_failed: {exc}"

            if device is None:
                await asyncio.sleep(self._reconnect_seconds)
                continue

            try:
                async with BleakClient(device.address) as client:
                    await client.start_notify(CHAR_UUID, self._on_hrm)
                    await self._wait_stop()
                    try:
                        await client.stop_notify(CHAR_UUID)
                    except Exception:
                        pass
            except Exception as exc:
                self._last_error = f"ble_connect_failed: {exc}"

            if not self._stop_event.is_set():
                await asyncio.sleep(self._reconnect_seconds)

        self._teardown_mqtt()

    def _device_filter(self, device, _adv) -> bool:
        try:
            name = device.name or ""
        except Exception:
            name = ""
        return self._device_name.lower() in name.lower()

    async def _wait_stop(self) -> None:
        while not self._stop_event.is_set():
            await asyncio.sleep(0.5)

    def _setup_mqtt(self) -> bool:
        try:
            host, port, base_topic = _parse_mqtt_url(self._mqtt_url)
            self._mqtt_topic = self._mqtt_topic_override or _build_arousal_topic(base_topic)
            self._mqtt_client = mqtt.Client()
            self._mqtt_client.connect(host, port, 60)
            self._mqtt_client.loop_start()
            return True
        except Exception as exc:
            self._last_error = f"mqtt_connect_failed: {exc}"
            self._mqtt_client = None
            return False

    def _teardown_mqtt(self) -> None:
        if self._mqtt_client is None:
            return
        try:
            self._mqtt_client.loop_stop()
        except Exception:
            pass
        try:
            self._mqtt_client.disconnect()
        except Exception:
            pass
        self._mqtt_client = None

    def _on_hrm(self, _sender, data: bytearray) -> None:
        hr, rr_list = _parse_hrm_packet(data)
        if hr is None:
            return
        with self._lock:
            self._handle_sample(int(hr), rr_list)

    def _handle_sample(self, hr: int, rr_list: List[float]) -> None:
        now = time.time()
        if rr_list:
            self._update_rr_buffer(rr_list)

        rmssd_raw = self._compute_rmssd()
        rmssd_log = math.log(rmssd_raw) if rmssd_raw and rmssd_raw > 0 else None

        if self._calibrating:
            self._hr_baseline_vals.append(float(hr))
            if rmssd_log is not None:
                self._rmssd_baseline_vals.append(float(rmssd_log))
            if time.monotonic() - self._baseline_start >= self._baseline_seconds:
                self._finish_baseline()
            return

        arousal = None
        method = "estimated"
        if rmssd_log is not None and self._rmssd_mean is not None and self._rmssd_std is not None:
            z = (rmssd_log - self._rmssd_mean) / self._rmssd_std
            arousal = -z
            method = "actual"
        elif self._hr_mean is not None and self._hr_std is not None:
            z = (float(hr) - self._hr_mean) / self._hr_std
            arousal = z
            method = "estimated"

        if arousal is None:
            return

        smooth = self._smooth_arousal(float(arousal))
        normalized = max(0.0, min(1.0, (smooth + self._arousal_zmax) / (2.0 * self._arousal_zmax)))

        payload = {
            "arousal": float(normalized),
            "timestamp": int(now * 1000),
            "hr": int(hr),
            "rmssd": float(rmssd_raw) if rmssd_raw else None,
            "method": method,
            "quality": self._last_rr_quality,
        }
        self._publish(payload)

    def _finish_baseline(self) -> None:
        self._calibrating = False
        stats_hr = _robust_stats(self._hr_baseline_vals, MIN_HR_STD)
        if stats_hr is not None:
            self._hr_mean, self._hr_std = stats_hr
        stats_rmssd = _robust_stats(self._rmssd_baseline_vals, MIN_LOG_RMSSD_STD)
        if stats_rmssd is not None:
            self._rmssd_mean, self._rmssd_std = stats_rmssd

    def _update_rr_buffer(self, rr_list: List[float]) -> None:
        now = time.time()
        for rr in rr_list:
            if MIN_RR_MS <= rr <= MAX_RR_MS:
                self._rr_buffer.append((now, rr))
        cutoff = now - RR_WINDOW_SEC
        while self._rr_buffer and self._rr_buffer[0][0] < cutoff:
            self._rr_buffer.popleft()

    def _compute_rmssd(self) -> Optional[float]:
        if len(self._rr_buffer) < MIN_RR_SAMPLES:
            self._last_rr_quality = f"Insufficient ({len(self._rr_buffer)}/{MIN_RR_SAMPLES})"
            return None
        rr = [v for _, v in self._rr_buffer]
        diffs = [rr[i + 1] - rr[i] for i in range(len(rr) - 1)]
        rel_jumps = [abs(diffs[i]) / rr[i] for i in range(len(diffs)) if rr[i] > 0]
        valid = [j < MAX_RR_JUMP for j in rel_jumps]
        clean_count = sum(1 for v in valid if v)
        if clean_count < 10:
            self._last_rr_quality = f"Too noisy ({clean_count} clean)"
            return None
        clean_diffs = [diffs[i] for i in range(len(diffs)) if valid[i]]
        if not clean_diffs:
            self._last_rr_quality = "Too noisy (0 clean)"
            return None
        rmssd = math.sqrt(sum(d * d for d in clean_diffs) / len(clean_diffs))
        quality_pct = 100.0 * clean_count / max(1, len(diffs))
        self._last_rr_quality = f"{quality_pct:.0f}% clean ({len(rr)} samples)"
        return float(rmssd)

    def _smooth_arousal(self, value: float) -> float:
        self._smoothing.append(value)
        if not self._smoothing:
            return value
        return float(sum(self._smoothing) / len(self._smoothing))

    def _publish(self, payload: dict) -> None:
        if self._mqtt_client is None or not self._mqtt_topic:
            return
        try:
            self._mqtt_client.publish(self._mqtt_topic, json.dumps(payload))
        except Exception:
            pass
