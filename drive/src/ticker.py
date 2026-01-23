from __future__ import annotations

import threading
import time
import carla


class WorldTicker(threading.Thread):
    def __init__(self, world: carla.World, target_hz: float = 20.0) -> None:
        super().__init__(daemon=True)
        self._world = world
        self._period = 1.0 / max(1e-6, float(target_hz))
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()
            try:
                self._world.tick()
            except Exception:
                time.sleep(0.05)
                continue
            # Sleep only the remaining time to avoid double-waiting when tick is slow.
            elapsed = time.perf_counter() - loop_start
            remaining = self._period - elapsed
            if remaining > 0:
                time.sleep(remaining)
