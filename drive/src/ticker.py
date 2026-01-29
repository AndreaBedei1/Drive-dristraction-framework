from __future__ import annotations

"""Synchronous world ticking helper."""

import threading
import time
import carla


class WorldTicker(threading.Thread):
    """Tick the CARLA world at a target frequency."""

    def __init__(self, world: carla.World, target_hz: float = 20.0) -> None:
        """Create a ticker for a target tick rate."""
        super().__init__(daemon=True)
        self._world = world
        self._period = 1.0 / max(1e-6, float(target_hz))
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Stop the ticker loop."""
        self._stop_event.set()

    def run(self) -> None:
        """Main loop that calls world.tick at the target rate."""
        while not self._stop_event.is_set():
            loop_start = time.perf_counter()
            try:
                self._world.tick()
            except Exception:
                time.sleep(0.05)
                continue
            elapsed = time.perf_counter() - loop_start
            remaining = self._period - elapsed
            if remaining > 0:
                time.sleep(remaining)
