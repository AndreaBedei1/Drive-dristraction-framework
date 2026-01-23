from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import yaml


@dataclass(frozen=True)
class TrafficManagerConfig:
    port: int
    hybrid_physics_radius: float
    global_speed_percentage_difference: float
    min_distance_to_leading_vehicle: float
    ignore_lights_percentage: int
    ignore_signs_percentage: int
    random_left_lanechange_percentage: int
    random_right_lanechange_percentage: int


@dataclass(frozen=True)
class CarlaConfig:
    host: str
    port: int
    timeout: float
    seed: int
    map_preference: List[str]
    sync: bool
    fixed_delta_seconds: float
    traffic_manager: TrafficManagerConfig


@dataclass(frozen=True)
class WeatherConfig:
    preset: str


@dataclass(frozen=True)
class TrafficConfig:
    vehicles: int
    safe_radius_from_route_start: float


@dataclass(frozen=True)
class PedestriansConfig:
    walkers: int
    running_percentage: int
    crossing_percentage: int
    max_speed_walking: float
    max_speed_running: float


@dataclass(frozen=True)
class RouteConfig:
    start_spawn_point: Union[str, int]
    end_spawn_point: Union[str, int]
    sampling_resolution: float
    waypoint_reached_threshold: float
    draw: bool
    draw_step: int
    draw_life_time: float
    auto_reset_on_finish: bool
    reset_cooldown_seconds: float


@dataclass(frozen=True)
class ManualControlConfig:
    path: str
    extra_args: List[str]


@dataclass(frozen=True)
class ScenarioConfig:
    carla: CarlaConfig
    weather: WeatherConfig
    traffic: TrafficConfig
    pedestrians: PedestriansConfig
    route: RouteConfig
    manual_control: ManualControlConfig


def _get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    return d[key] if key in d else default


def load_config(path: str) -> ScenarioConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    tm_raw = raw["carla"]["traffic_manager"]
    tm = TrafficManagerConfig(
        port=int(tm_raw["port"]),
        hybrid_physics_radius=float(tm_raw["hybrid_physics_radius"]),
        global_speed_percentage_difference=float(tm_raw["global_speed_percentage_difference"]),
        min_distance_to_leading_vehicle=float(tm_raw["min_distance_to_leading_vehicle"]),
        ignore_lights_percentage=int(tm_raw["ignore_lights_percentage"]),
        ignore_signs_percentage=int(tm_raw["ignore_signs_percentage"]),
        random_left_lanechange_percentage=int(tm_raw["random_left_lanechange_percentage"]),
        random_right_lanechange_percentage=int(tm_raw["random_right_lanechange_percentage"]),
    )

    carla_cfg = CarlaConfig(
        host=str(raw["carla"]["host"]),
        port=int(raw["carla"]["port"]),
        timeout=float(raw["carla"]["timeout"]),
        seed=int(raw["carla"]["seed"]),
        map_preference=list(raw["carla"]["map_preference"]),
        sync=bool(raw["carla"]["sync"]),
        fixed_delta_seconds=float(raw["carla"]["fixed_delta_seconds"]),
        traffic_manager=tm,
    )

    weather_cfg = WeatherConfig(preset=str(raw["weather"]["preset"]))

    traffic_cfg = TrafficConfig(
        vehicles=int(raw["traffic"]["vehicles"]),
        safe_radius_from_route_start=float(raw["traffic"]["safe_radius_from_route_start"]),
    )

    ped_cfg = PedestriansConfig(
        walkers=int(raw["pedestrians"]["walkers"]),
        running_percentage=int(raw["pedestrians"]["running_percentage"]),
        crossing_percentage=int(raw["pedestrians"]["crossing_percentage"]),
        max_speed_walking=float(raw["pedestrians"]["max_speed_walking"]),
        max_speed_running=float(raw["pedestrians"]["max_speed_running"]),
    )

    route_raw = raw["route"]
    route_cfg = RouteConfig(
        start_spawn_point=_get(route_raw, "start_spawn_point", "auto"),
        end_spawn_point=_get(route_raw, "end_spawn_point", "auto_far"),
        sampling_resolution=float(route_raw["sampling_resolution"]),
        waypoint_reached_threshold=float(route_raw["waypoint_reached_threshold"]),
        draw=bool(route_raw["draw"]),
        draw_step=int(route_raw["draw_step"]),
        draw_life_time=float(route_raw["draw_life_time"]),
        auto_reset_on_finish=bool(route_raw["auto_reset_on_finish"]),
        reset_cooldown_seconds=float(route_raw["reset_cooldown_seconds"]),
    )

    mc_raw = raw["manual_control"]
    mc_cfg = ManualControlConfig(
        path=str(mc_raw["path"]),
        extra_args=list(mc_raw.get("extra_args", [])),
    )

    return ScenarioConfig(
        carla=carla_cfg,
        weather=weather_cfg,
        traffic=traffic_cfg,
        pedestrians=ped_cfg,
        route=route_cfg,
        manual_control=mc_cfg,
    )
