from __future__ import annotations

"""Helpers for connecting to CARLA and configuring the world."""

from typing import Optional, List
import time
import carla


class CarlaSession:
    """Wrap CARLA client operations used by the runner."""

    def __init__(self, host: str, port: int, timeout: float) -> None:
        """Create a client with the requested host, port, and timeout."""
        self._client = carla.Client(host, port)
        self._timeout = float(timeout)
        self._client.set_timeout(self._timeout)

    @property
    def client(self) -> carla.Client:
        """Return the underlying CARLA client."""
        return self._client

    def get_world(self) -> carla.World:
        """Return the current world from the server."""
        return self._client.get_world()

    def load_preferred_map(self, preference: List[str]) -> carla.World:
        """Load the first available map from a preference list."""
        available = self._client.get_available_maps()
        short_to_full = {}
        for m in available:
            short = m.split("/")[-1]
            short_to_full[short] = m
        current_world = None
        current_short = None
        try:
            current_world = self._client.get_world()
            current_short = current_world.get_map().name.split("/")[-1]
        except Exception:
            current_world = None
            current_short = None

        for short in preference:
            if short in short_to_full:
                if current_short == short and current_world is not None:
                    return current_world
                return self._load_world_with_retries(short_to_full[short])

        if current_world is not None:
            return current_world
        return self._client.get_world()

    def _load_world_with_retries(self, map_name: str) -> carla.World:
        """Load a map with retries and a longer timeout."""
        load_timeout = max(self._timeout, 30.0)
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                self._client.set_timeout(load_timeout)
                world = self._client.load_world(map_name)
                self._client.set_timeout(self._timeout)
                return world
            except Exception as exc:
                last_exc = exc
                self._client.set_timeout(self._timeout)
                time.sleep(1.0 + 0.5 * attempt)
        try:
            return self._client.get_world()
        except Exception:
            if last_exc is not None:
                raise last_exc
            raise

    def configure_sync(self, world: carla.World, sync: bool, fixed_delta_seconds: float) -> None:
        """Enable or disable synchronous mode and apply fixed delta."""
        settings = world.get_settings()
        settings.synchronous_mode = sync
        settings.fixed_delta_seconds = fixed_delta_seconds if sync else None
        world.apply_settings(settings)

    def set_weather_preset(self, world: carla.World, preset_name: str) -> None:
        """Apply a weather preset by name."""
        if not hasattr(carla.WeatherParameters, preset_name):
            raise ValueError(f"Unknown weather preset: {preset_name}")
        weather = getattr(carla.WeatherParameters, preset_name)
        world.set_weather(weather)

    def get_traffic_manager(self, tm_port: int) -> carla.TrafficManager:
        """Return the traffic manager instance for the given port."""
        return self._client.get_trafficmanager(tm_port)
