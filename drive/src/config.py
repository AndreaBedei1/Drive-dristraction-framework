from __future__ import annotations

"""Configuration loader and typed config models."""

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import yaml


@dataclass(frozen=True)
class TrafficManagerConfig:
    """Traffic manager tuning parameters."""

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
    """Connection, map, and simulation settings."""
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
    """Weather preset configuration."""
    preset: str


@dataclass(frozen=True)
class TrafficConfig:
    """Traffic spawning settings."""
    vehicles: int
    safe_radius_from_route_start: float


@dataclass(frozen=True)
class PedestriansConfig:
    """Pedestrian spawning settings."""
    walkers: int
    running_percentage: int
    crossing_percentage: int
    max_speed_walking: float
    max_speed_running: float


@dataclass(frozen=True)
class RouteConfig:
    """Route generation parameters."""
    start_spawn_point: Union[str, int]
    end_spawn_point: Union[str, int]
    sampling_resolution: float
    waypoint_reached_threshold: float
    draw: bool
    draw_step: int
    draw_life_time: float
    auto_reset_on_finish: bool
    reset_cooldown_seconds: float
    test_spawn_points: List[int]
    test_city_radius_ratio: float
    test_highway_radius_ratio: float
    test_city_density_radius_m: float
    test_city_density_ratio: float
    test_highway_distance_ratio: float


@dataclass(frozen=True)
class ManualControlConfig:
    """Manual control script location and arguments."""
    path: str
    extra_args: List[str]


@dataclass(frozen=True)
class ExperimentConfig:
    """Experiment metadata and test overrides."""
    user_id: str
    run_id: int
    weather_label: str
    output_dir: str
    mode: str
    test_map_preference: List[str]
    test_vehicles: int
    test_walkers: int
    test_dataset_suffix: str


@dataclass(frozen=True)
class ErrorConfig:
    """Error detection thresholds and timing settings."""
    harsh_brake_threshold_mps2: float
    harsh_brake_min_speed_kmh: float
    harsh_brake_min_brake: float
    harsh_brake_cooldown_seconds: float
    red_light_min_speed_kmh: float
    red_light_distance_m: float
    red_light_pass_distance_m: float
    red_light_pass_buffer_m: float
    red_light_track_distance_m: float
    red_light_min_interval_seconds: float
    red_light_cooldown_seconds: float
    stop_sign_min_speed_kmh: float
    stop_sign_zone_half_width_m: float
    stop_sign_zone_length_m: float
    stop_sign_dedupe_distance_m: float
    stop_sign_dedupe_time_s: float
    debug_stop_visualization: bool
    debug_stop_life_time: float
    solid_line_cooldown_seconds: float
    collision_cooldown_seconds: float


@dataclass(frozen=True)
class DistractionConfig:
    """Distraction window timing and audio settings."""
    min_interval_seconds: float
    max_interval_seconds: float
    min_gap_between_windows_seconds: float
    flash_duration_seconds: float
    flash_start_interval_seconds: float
    flash_min_interval_seconds: float
    beep_frequency_hz: int
    beep_duration_ms: int
    window_titles: List[str]
    simulation_window_title: str


@dataclass(frozen=True)
class TrafficLightsConfig:
    """Traffic light cycle timing."""
    green_time: float
    yellow_time: float
    red_time: float


@dataclass(frozen=True)
class ScenarioConfig:
    """Root configuration for the scenario runner."""
    carla: CarlaConfig
    weather: WeatherConfig
    traffic: TrafficConfig
    pedestrians: PedestriansConfig
    route: RouteConfig
    manual_control: ManualControlConfig
    experiment: ExperimentConfig
    errors: ErrorConfig
    distractions: DistractionConfig
    traffic_lights: TrafficLightsConfig


def _get(d: Dict[str, Any], key: str, default: Any = None) -> Any:
    """Return a key from a dict with a default."""
    return d[key] if key in d else default


def _derive_weather_label(preset: str) -> str:
    """Derive a short weather label from a preset name."""
    p = preset.lower()
    if "rain" in p or "wet" in p or "storm" in p:
        return "rain"
    if "night" in p or "sunset" in p:
        return "night"
    return "day"


def load_config(path: str) -> ScenarioConfig:
    """Load a YAML file into typed configuration objects."""
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
        test_spawn_points=[int(x) for x in _get(route_raw, "test_spawn_points", []) or []],
        test_city_radius_ratio=float(_get(route_raw, "test_city_radius_ratio", 0.35)),
        test_highway_radius_ratio=float(_get(route_raw, "test_highway_radius_ratio", 0.7)),
        test_city_density_radius_m=float(_get(route_raw, "test_city_density_radius_m", 60.0)),
        test_city_density_ratio=float(_get(route_raw, "test_city_density_ratio", 0.7)),
        test_highway_distance_ratio=float(_get(route_raw, "test_highway_distance_ratio", 0.6)),
    )

    mc_raw = raw["manual_control"]
    mc_cfg = ManualControlConfig(
        path=str(mc_raw["path"]),
        extra_args=list(mc_raw.get("extra_args", [])),
    )

    exp_raw = _get(raw, "experiment", {})
    exp_cfg = ExperimentConfig(
        user_id=str(_get(exp_raw, "user_id", "unknown")),
        run_id=int(_get(exp_raw, "run_id", 1)),
        weather_label=str(_get(exp_raw, "weather_label", _derive_weather_label(weather_cfg.preset))),
        output_dir=str(_get(exp_raw, "output_dir", "data")),
        mode=str(_get(exp_raw, "mode", "train")),
        test_map_preference=list(_get(exp_raw, "test_map_preference", ["Town04"])),
        test_vehicles=int(_get(exp_raw, "test_vehicles", 0)),
        test_walkers=int(_get(exp_raw, "test_walkers", 0)),
        test_dataset_suffix=str(_get(exp_raw, "test_dataset_suffix", "_test")),
    )

    err_raw = _get(raw, "errors", {})
    err_cfg = ErrorConfig(
        harsh_brake_threshold_mps2=float(_get(err_raw, "harsh_brake_threshold_mps2", 3.5)),
        harsh_brake_min_speed_kmh=float(_get(err_raw, "harsh_brake_min_speed_kmh", 10.0)),
        harsh_brake_min_brake=float(_get(err_raw, "harsh_brake_min_brake", 0.7)),
        harsh_brake_cooldown_seconds=float(_get(err_raw, "harsh_brake_cooldown_seconds", 1.5)),
        red_light_min_speed_kmh=float(_get(err_raw, "red_light_min_speed_kmh", 5.0)),
        red_light_distance_m=float(_get(err_raw, "red_light_distance_m", 12.0)),
        red_light_pass_distance_m=float(_get(err_raw, "red_light_pass_distance_m", 1.5)),
        red_light_pass_buffer_m=float(_get(err_raw, "red_light_pass_buffer_m", 0.8)),
        red_light_track_distance_m=float(_get(err_raw, "red_light_track_distance_m", 12.0)),
        red_light_min_interval_seconds=float(_get(err_raw, "red_light_min_interval_seconds", 2.0)),
        red_light_cooldown_seconds=float(_get(err_raw, "red_light_cooldown_seconds", 5.0)),
        stop_sign_min_speed_kmh=float(_get(err_raw, "stop_sign_min_speed_kmh", 10.0)),
        stop_sign_zone_half_width_m=float(_get(err_raw, "stop_sign_zone_half_width_m", 1.0)),
        stop_sign_zone_length_m=float(_get(err_raw, "stop_sign_zone_length_m", 3.0)),
        stop_sign_dedupe_distance_m=float(_get(err_raw, "stop_sign_dedupe_distance_m", 1.5)),
        stop_sign_dedupe_time_s=float(_get(err_raw, "stop_sign_dedupe_time_s", 0.6)),
        debug_stop_visualization=bool(_get(err_raw, "debug_stop_visualization", True)),
        debug_stop_life_time=float(_get(err_raw, "debug_stop_life_time", 6.0)),
        solid_line_cooldown_seconds=float(_get(err_raw, "solid_line_cooldown_seconds", 2.0)),
        collision_cooldown_seconds=float(_get(err_raw, "collision_cooldown_seconds", 2.0)),
    )

    dis_raw = _get(raw, "distractions", {})
    dis_cfg = DistractionConfig(
        min_interval_seconds=float(_get(dis_raw, "min_interval_seconds", 25.0)),
        max_interval_seconds=float(_get(dis_raw, "max_interval_seconds", 35.0)),
        min_gap_between_windows_seconds=float(_get(dis_raw, "min_gap_between_windows_seconds", 5.0)),
        flash_duration_seconds=float(_get(dis_raw, "flash_duration_seconds", 5.0)),
        flash_start_interval_seconds=float(_get(dis_raw, "flash_start_interval_seconds", 0.5)),
        flash_min_interval_seconds=float(_get(dis_raw, "flash_min_interval_seconds", 0.05)),
        beep_frequency_hz=int(_get(dis_raw, "beep_frequency_hz", 880)),
        beep_duration_ms=int(_get(dis_raw, "beep_duration_ms", 200)),
        window_titles=list(_get(dis_raw, "window_titles", ["Distraction 1", "Distraction 2"])),
        simulation_window_title=str(_get(dis_raw, "simulation_window_title", "CARLA Manual Control")),
    )

    tl_raw = _get(raw, "traffic_lights", {})
    tl_cfg = TrafficLightsConfig(
        green_time=float(_get(tl_raw, "green_time", 20.0)),
        yellow_time=float(_get(tl_raw, "yellow_time", 3.0)),
        red_time=float(_get(tl_raw, "red_time", 8.0)),
    )

    return ScenarioConfig(
        carla=carla_cfg,
        weather=weather_cfg,
        traffic=traffic_cfg,
        pedestrians=ped_cfg,
        route=route_cfg,
        manual_control=mc_cfg,
        experiment=exp_cfg,
        errors=err_cfg,
        distractions=dis_cfg,
        traffic_lights=tl_cfg,
    )
