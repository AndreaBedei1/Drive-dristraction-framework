from __future__ import annotations

"""Synchronous world ticking helper."""

import threading
import time
from typing import Optional

import carla


class WorldTicker(threading.Thread):
    """Tick the CARLA world at a target frequency."""

    def __init__(
        self,
        world: carla.World,
        target_hz: float = 20.0,
        stats_interval_seconds: float = 0.0,
    ) -> None:
        """Create a ticker for a target tick rate."""
        super().__init__(daemon=True)
        self._world = world
        self._target_hz = max(1e-6, float(target_hz))
        self._period = 1.0 / self._target_hz
        self._stats_interval = max(0.0, float(stats_interval_seconds))
        self._stop_event = threading.Event()
        self._tick_cond = threading.Condition()
        self._last_snapshot: Optional[carla.WorldSnapshot] = None
        self._last_frame: Optional[int] = None
        self._stats_window_start = time.perf_counter()
        self._stats_ticks = 0
        self._stats_overruns = 0
        self._stats_tick_elapsed_sum = 0.0

    def stop(self) -> None:
        """Stop the ticker loop."""
        self._stop_event.set()
        with self._tick_cond:
            self._tick_cond.notify_all()

    def get_snapshot(self) -> Optional[carla.WorldSnapshot]:
        """Return the latest world snapshot."""
        with self._tick_cond:
            return self._last_snapshot

    def wait_for_tick(self, timeout: Optional[float] = None) -> Optional[carla.WorldSnapshot]:
        """Block until a new tick is available and return its snapshot."""
        with self._tick_cond:
            last_frame = self._last_frame
            if not self._tick_cond.wait_for(lambda: self._last_frame != last_frame, timeout=timeout):
                return None
            return self._last_snapshot

    def run(self) -> None:
        """Main loop that calls world.tick at the target rate."""
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()
            try:
                self._world.tick()
                try:
                    snapshot = self._world.get_snapshot()
                except Exception:
                    snapshot = None
            except Exception:
                time.sleep(0.05)
                continue
            frame = None
            if snapshot is not None:
                try:
                    frame = int(snapshot.frame)
                except Exception:
                    frame = None
            if frame is None:
                frame = (self._last_frame or 0) + 1
            with self._tick_cond:
                self._last_snapshot = snapshot
                self._last_frame = frame
                self._tick_cond.notify_all()
            elapsed = time.perf_counter() - loop_start
            self._stats_ticks += 1
            self._stats_tick_elapsed_sum += elapsed
            if elapsed > self._period:
                self._stats_overruns += 1
            self._maybe_log_stats()
            remaining = self._period - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _maybe_log_stats(self) -> None:
        """Periodically print ticker health metrics."""
        if self._stats_interval <= 0.0:
            return
        now = time.perf_counter()
        span = now - self._stats_window_start
        if span < self._stats_interval:
            return
        ticks = max(1, self._stats_ticks)
        effective_hz = ticks / max(1e-6, span)
        avg_tick_ms = 1000.0 * (self._stats_tick_elapsed_sum / ticks)
        overrun_pct = 100.0 * (self._stats_overruns / ticks)
        print(
            "[Ticker] "
            f"hz={effective_hz:.2f}/{self._target_hz:.2f} "
            f"avg_tick_ms={avg_tick_ms:.2f} overruns={overrun_pct:.1f}%"
        )
        self._stats_window_start = now
        self._stats_ticks = 0
        self._stats_overruns = 0
        self._stats_tick_elapsed_sum = 0.0
