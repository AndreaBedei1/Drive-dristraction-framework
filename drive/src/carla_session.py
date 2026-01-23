from __future__ import annotations

from typing import Optional, List
import carla


class CarlaSession:
    def __init__(self, host: str, port: int, timeout: float) -> None:
        self._client = carla.Client(host, port)
        self._client.set_timeout(timeout)

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

        for short in preference:
            if short in short_to_full:
                return self._client.load_world(short_to_full[short])

        # Fallback: keep current world
        return self._client.get_world()

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
