from __future__ import annotations

"""Synchronous world ticking helper."""

import threading
import time
from typing import Optional

import carla


class WorldTicker(threading.Thread):
    """Tick the CARLA world at a target frequency."""

    def __init__(self, world: carla.World, target_hz: float = 20.0) -> None:
        """Create a ticker for a target tick rate."""
        super().__init__(daemon=True)
        self._world = world
        self._period = 1.0 / max(1e-6, float(target_hz))
        self._stop_event = threading.Event()
        self._tick_cond = threading.Condition()
        self._last_snapshot: Optional[carla.WorldSnapshot] = None
        self._last_frame: Optional[int] = None

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
            remaining = self._period - elapsed
            if remaining > 0:
                time.sleep(remaining)
