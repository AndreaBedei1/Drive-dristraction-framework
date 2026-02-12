from __future__ import annotations

"""BLE heart-rate -> arousal provider (headless)."""

import asyncio
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

from src.arousal_provider import ArousalSnapshot


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
    elif len(data) > idx + 1:
        # Fallback: some devices omit the RR flag but still append RR intervals.
        while idx + 1 < len(data):
            rr_raw = int.from_bytes(data[idx:idx + 2], "little")
            rr_ms = (rr_raw / 1024.0) * 1000.0
            rr_list.append(rr_ms)
            idx += 2
    return hr, rr_list


class BleArousalProvider(threading.Thread):
    """Connect to BLE HR sensor and compute arousal in-process."""

    def __init__(
        self,
        device_name: str,
        baseline_seconds: int = 60,
        smoothing_window: int = 10,
        reconnect_seconds: float = 5.0,
        no_sample_timeout_seconds: float = 12.0,
        arousal_zmax: float = AROUSAL_ZMAX,
        debug: bool = False,
        debug_interval_seconds: float = 1.0,
    ) -> None:
        super().__init__(daemon=True)
        self._device_name = str(device_name or "Coospo")
        self._device_tokens = [t.strip() for t in self._device_name.replace(";", ",").split(",") if t.strip()]
        if not self._device_tokens:
            self._device_tokens = ["HW9", "Coospo"]
        self._baseline_seconds = max(10, int(baseline_seconds))
        self._smoothing_window = max(3, int(smoothing_window))
        self._reconnect_seconds = max(1.0, float(reconnect_seconds))
        self._no_sample_timeout = max(2.0, float(no_sample_timeout_seconds))
        self._arousal_zmax = float(arousal_zmax)
        self._debug = bool(debug)
        self._debug_interval = max(0.1, float(debug_interval_seconds))
        self._last_debug_ts = 0.0
        self._last_sample_ts = 0.0
        self._baseline_done = threading.Event()

        self._stop_event = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._lock = threading.Lock()
        self._last_error: Optional[str] = None
        self._snapshot = ArousalSnapshot(None, None, None, None, None)

        self._rr_buffer: Deque[Tuple[float, float]] = deque()
        self._last_rr_quality = "N/A"
        self._smoothing: Deque[float] = deque(maxlen=self._smoothing_window)

        self._calibrating = True
        self._baseline_start: Optional[float] = None
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

    def get_snapshot(self) -> ArousalSnapshot:
        with self._lock:
            return self._snapshot

    def wait_for_baseline(self, timeout: Optional[float] = None) -> bool:
        return self._baseline_done.wait(timeout=timeout)

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
            if self._debug:
                print("[Arousal] BLE not available (bleak missing)")
            return

        while not self._stop_event.is_set():
            device = None
            try:
                if self._debug:
                    joined = ", ".join(self._device_tokens)
                    print(f"[Arousal] Scanning for BLE devices: {joined}")
                devices = await BleakScanner.discover(timeout=10.0)
                device = self._select_device(devices)
            except Exception as exc:
                self._last_error = f"ble_scan_failed: {exc}"
                if self._debug:
                    print(f"[Arousal] BLE scan failed: {exc}")

            if device is None:
                if self._debug:
                    names = sorted({d.name for d in (devices or []) if d.name})
                    if names:
                        print(f"[Arousal] BLE devices seen: {', '.join(names)}")
                    print(f"[Arousal] BLE device not found. Retrying in {self._reconnect_seconds:.1f}s")
                await asyncio.sleep(self._reconnect_seconds)
                continue

            try:
                if self._debug:
                    print(f"[Arousal] Connecting to BLE device: {device.name}")
                async with BleakClient(device.address) as client:
                    await client.start_notify(CHAR_UUID, self._on_hrm)
                    # Start no-data watchdog from connection time.
                    self._last_sample_ts = time.monotonic()
                    timed_out = await self._wait_stop()
                    try:
                        await client.stop_notify(CHAR_UUID)
                    except Exception:
                        pass
                    if timed_out:
                        print(
                            f"[Arousal] No HR data for {self._no_sample_timeout:.1f}s, forcing reconnect."
                        )
            except Exception as exc:
                self._last_error = f"ble_connect_failed: {exc}"
                if self._debug:
                    print(f"[Arousal] BLE connection failed: {exc}")

            if not self._stop_event.is_set():
                await asyncio.sleep(self._reconnect_seconds)

    def _device_filter(self, device, _adv) -> bool:
        # Unused with discover(), kept for compatibility.
        try:
            name = device.name or ""
        except Exception:
            name = ""
        return any(token.lower() in name.lower() for token in self._device_tokens)

    def _select_device(self, devices) -> Optional[object]:
        if not devices:
            return None
        for d in devices:
            try:
                name = d.name or ""
            except Exception:
                name = ""
            if any(token.lower() in name.lower() for token in self._device_tokens):
                return d
        # Fallback to common names if tokens didn't match.
        fallback = ["HW9", "Coospo"]
        for d in devices:
            try:
                name = d.name or ""
            except Exception:
                name = ""
            if any(token.lower() in name.lower() for token in fallback):
                return d
        return None

    async def _wait_stop(self) -> bool:
        while not self._stop_event.is_set():
            elapsed_no_data = time.monotonic() - self._last_sample_ts
            if elapsed_no_data >= self._no_sample_timeout:
                self._last_error = f"ble_no_data_timeout: {elapsed_no_data:.1f}s"
                return True
            await asyncio.sleep(0.5)
        return False

    def _on_hrm(self, _sender, data: bytearray) -> None:
        hr, rr_list = _parse_hrm_packet(data)
        if hr is None:
            return
        with self._lock:
            self._handle_sample(int(hr), rr_list)

    def _handle_sample(self, hr: int, rr_list: List[float]) -> None:
        now = time.time()
        mono_now = time.monotonic()
        self._last_sample_ts = mono_now
        if rr_list:
            self._update_rr_buffer(rr_list)

        rmssd_raw = self._compute_rmssd()
        rmssd_log = math.log(rmssd_raw) if rmssd_raw and rmssd_raw > 0 else None

        if self._calibrating:
            if self._baseline_start is None:
                self._baseline_start = mono_now
                if self._debug:
                    print("[Arousal] Baseline calibration started")
            self._hr_baseline_vals.append(float(hr))
            if rmssd_log is not None:
                self._rmssd_baseline_vals.append(float(rmssd_log))
            if self._baseline_start is not None and time.monotonic() - self._baseline_start >= self._baseline_seconds:
                self._finish_baseline()
            self._update_snapshot(
                value=None,
                method="calibrating",
                timestamp_ms=int(now * 1000),
                quality=self._last_rr_quality,
                hr_bpm=hr,
            )
            self._maybe_debug_print(
                mono_now=mono_now,
                hr_bpm=hr,
                arousal=None,
                method="calibrating",
                quality=self._last_rr_quality,
            )
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
        self._update_snapshot(
            value=float(normalized),
            method=method,
            timestamp_ms=int(now * 1000),
            quality=self._last_rr_quality,
            hr_bpm=hr,
        )
        self._maybe_debug_print(
            mono_now=mono_now,
            hr_bpm=hr,
            arousal=float(normalized),
            method=method,
            quality=self._last_rr_quality,
        )

    def _finish_baseline(self) -> None:
        self._calibrating = False
        stats_hr = _robust_stats(self._hr_baseline_vals, MIN_HR_STD)
        if stats_hr is not None:
            self._hr_mean, self._hr_std = stats_hr
        stats_rmssd = _robust_stats(self._rmssd_baseline_vals, MIN_LOG_RMSSD_STD)
        if stats_rmssd is not None:
            self._rmssd_mean, self._rmssd_std = stats_rmssd
        self._baseline_done.set()
        if self._debug:
            print("[Arousal] Baseline calibration completed")

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

    def _update_snapshot(
        self,
        value: Optional[float],
        method: Optional[str],
        timestamp_ms: Optional[int],
        quality: Optional[str],
        hr_bpm: Optional[int],
    ) -> None:
        self._snapshot = ArousalSnapshot(
            value=value,
            method=method,
            timestamp_ms=timestamp_ms,
            quality=quality,
            hr_bpm=hr_bpm,
        )

    def _maybe_debug_print(
        self,
        mono_now: float,
        hr_bpm: int,
        arousal: Optional[float],
        method: Optional[str],
        quality: str,
    ) -> None:
        if not self._debug:
            return
        if self._calibrating:
            if mono_now - self._last_debug_ts < self._debug_interval:
                return
            self._last_debug_ts = mono_now
            remaining = ""
            if self._baseline_start is not None:
                remaining_s = max(0.0, self._baseline_seconds - (mono_now - self._baseline_start))
                remaining = f" baseline_remaining={remaining_s:.0f}s"
            print(f"[Arousal] calibrating hr={hr_bpm} bpm quality={quality}{remaining}")
            return
        # After baseline: print every sample (no throttling) to show all sensor data.
        if arousal is None:
            print(f"[Arousal] hr={hr_bpm} bpm quality={quality}")
        else:
            print(f"[Arousal] hr={hr_bpm} bpm arousal={arousal:.3f} method={method} quality={quality}")


BleArousalPublisher = BleArousalProvider
