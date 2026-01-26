from __future__ import annotations

from typing import Optional, List
import time
import carla


class CarlaSession:
    def __init__(self, host: str, port: int, timeout: float) -> None:
        self._client = carla.Client(host, port)
        self._timeout = float(timeout)
        self._client.set_timeout(self._timeout)

    @property
    def client(self) -> carla.Client:
        return self._client

    def get_world(self) -> carla.World:
        return self._client.get_world()

    def load_preferred_map(self, preference: List[str]) -> carla.World:
        available = self._client.get_available_maps()
        # available contains full paths like "/Game/Carla/Maps/Town04"
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

        # Fallback: keep current world
        if current_world is not None:
            return current_world
        return self._client.get_world()

    def _load_world_with_retries(self, map_name: str) -> carla.World:
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
        settings = world.get_settings()
        settings.synchronous_mode = sync
        settings.fixed_delta_seconds = fixed_delta_seconds if sync else None
        world.apply_settings(settings)

    def set_weather_preset(self, world: carla.World, preset_name: str) -> None:
        if not hasattr(carla.WeatherParameters, preset_name):
            raise ValueError(f"Unknown weather preset: {preset_name}")
        weather = getattr(carla.WeatherParameters, preset_name)
        world.set_weather(weather)

    def get_traffic_manager(self, tm_port: int) -> carla.TrafficManager:
        return self._client.get_trafficmanager(tm_port)
