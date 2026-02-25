from __future__ import annotations

"""Scenario runner for the driving distraction framework."""

import argparse
import os
import subprocess
import sys
import time
from typing import List, Optional, Tuple

import carla

from src.config import load_config
from src.carla_session import CarlaSession
from src.utils import seed_everything, pick_spawn_point_indices, dist, find_hero_vehicle
from src.traffic import TrafficSpawner
from src.pedestrians import PedestrianSpawner
from src.route import RoutePlanner, draw_route
from src.monitor import LapMonitor
from src.ticker import WorldTicker
from src.datasets import (
    DatasetContext,
    ErrorDatasetLogger,
    DistractionDatasetLogger,
    BaselineDrivingTimeLogger,
)
from src.error_monitor import ErrorMonitor
from src.arousal_provider import ArousalSnapshot
from src.distraction_windows import DistractionCoordinator, DistractionWindow, focus_simulation_window
from src.camera_preview import CameraPreviewWindow, preview_available


def _resolve_manual_control_path(path: str) -> str:
    """Return an absolute manual control path or raise if missing."""
    p = os.path.abspath(path)
    if os.path.isfile(p):
        return p
    raise FileNotFoundError(f"manual_control path not found: {p}")


def _launch_manual_control(
    script_path: str,
    host: str,
    port: int,
    extra_args: List[str],
    env: Optional[dict] = None,
) -> subprocess.Popen:
    """Launch the manual control script as a subprocess."""
    cmd = [sys.executable, script_path, f"--host={host}", f"--port={port}"]
    cmd.extend(extra_args)
    print(f"[Runner] Launching manual control: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env)


def _resolve_output_dir(path: str) -> str:
    """Return an absolute output directory path."""
    return path if os.path.isabs(path) else os.path.abspath(path)


def _get_monitor_rects() -> List[Tuple[int, int, int, int]]:
    """Return monitor rectangles ordered from left to right."""
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass

        rects: List[Tuple[int, int, int, int]] = []

        @ctypes.WINFUNCTYPE(ctypes.c_int, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM)
        def _enum_proc(_hmonitor, _hdc, lprc, _lparam):
            r = lprc.contents
            rects.append((r.left, r.top, r.right - r.left, r.bottom - r.top))
            return 1

        user32.EnumDisplayMonitors(0, 0, _enum_proc, 0)
        rects.sort(key=lambda r: r[0])
        return rects
    except Exception:
        return []


def _pick_monitor_layout(rects: List[Tuple[int, int, int, int]]) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]], Optional[Tuple[int, int, int, int]]]:
    """Choose left, center, and right monitor rectangles."""
    if not rects:
        return None, None, None
    if len(rects) == 1:
        return None, rects[0], None
    if len(rects) == 2:
        return rects[0], rects[0], rects[1]
    mid = rects[len(rects) // 2]
    return rects[0], mid, rects[-1]


def _compute_next_run_id(
    output_dir: str,
    user_id: str,
    suffix: str,
    dataset_profile: str,
    mode: str,
) -> int:
    """Compute the next run id for a participant from profile-specific dataset files."""
    import csv

    profile = str(dataset_profile or "").strip().lower()
    run_mode = str(mode or "").strip().lower()
    participant = str(user_id or "").strip()

    filenames = [f"Dataset Errors{suffix}.csv"]
    if run_mode == "test":
        filenames.append(f"Dataset Distractions{suffix}.csv")
    elif profile == "baseline":
        filenames.append(f"Dataset Driving Time{suffix}.csv")
    else:
        filenames.append(f"Dataset Distractions{suffix}.csv")

    max_run_id = 0
    for filename in dict.fromkeys(filenames):
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_user_id = str(row.get("user_id", "")).strip()
                    if participant and row_user_id != participant:
                        continue
                    try:
                        rid = int(str(row.get("run_id", "")).strip())
                    except Exception:
                        continue
                    if rid > max_run_id:
                        max_run_id = rid
        except Exception:
            pass
    return max_run_id + 1 if max_run_id > 0 else 1


def _safe_destroy_ids(world: carla.World, actor_ids: List[int]) -> None:
    """Attempt to stop and destroy actors by id."""
    for actor_id in actor_ids:
        try:
            actor = world.get_actor(actor_id)
        except Exception:
            actor = None
        if actor is None:
            continue
        try:
            if hasattr(actor, "is_alive") and not actor.is_alive:
                continue
        except Exception:
            pass
        try:
            if hasattr(actor, "stop"):
                actor.stop()
        except Exception:
            pass
        try:
            actor.destroy()
        except Exception:
            pass


def _destroy_stale_hero_vehicles(world: carla.World, preferred_role_name: str = "hero") -> int:
    """Destroy existing hero vehicles to avoid stale monitors tracking old actors."""
    destroyed = 0
    try:
        vehicles = world.get_actors().filter("vehicle.*")
    except Exception:
        return 0
    for veh in vehicles:
        try:
            role = veh.attributes.get("role_name", "")
        except Exception:
            role = ""
        if role != preferred_role_name:
            continue
        try:
            if hasattr(veh, "is_alive") and not veh.is_alive:
                continue
        except Exception:
            pass
        try:
            veh.destroy()
            destroyed += 1
        except Exception:
            pass
    return destroyed


def _configure_traffic_lights(world: carla.World, cfg) -> None:
    """Apply timing configuration to all traffic lights."""
    try:
        lights = world.get_actors().filter("*traffic_light*")
    except Exception:
        return
    for tl in lights:
        try:
            tl.set_green_time(float(cfg.green_time))
            tl.set_yellow_time(float(cfg.yellow_time))
            tl.set_red_time(float(cfg.red_time))
        except Exception:
            pass


def _pick_far_spawn_point_index(
    spawn_points: List[carla.Transform],
    avoid_indices: set,
    anchor_locations: List[carla.Location],
) -> int:
    """Pick the spawn point farthest from multiple anchor locations."""
    best_i = None
    best_score = -1.0
    for i, sp in enumerate(spawn_points):
        if i in avoid_indices:
            continue
        score = 0.0
        for loc in anchor_locations:
            score += dist(sp.location, loc)
        if score > best_score:
            best_score = score
            best_i = i
    if best_i is None:
        for i in range(len(spawn_points)):
            if i not in avoid_indices:
                return i
        return 0
    return int(best_i)


def _concat_routes(routes: List[List[carla.Location]]) -> List[carla.Location]:
    """Merge multiple route legs while avoiding duplicate join points."""
    combined: List[carla.Location] = []
    for r in routes:
        if not r:
            continue
        if combined:
            combined.extend(r[1:])
        else:
            combined.extend(r)
    return combined


def _compute_spawn_points_center(spawn_points: List[carla.Transform]) -> carla.Location:
    """Compute the centroid of all spawn point locations."""
    if not spawn_points:
        return carla.Location(x=0.0, y=0.0, z=0.0)
    avg_x = sum(sp.location.x for sp in spawn_points) / len(spawn_points)
    avg_y = sum(sp.location.y for sp in spawn_points) / len(spawn_points)
    avg_z = sum(sp.location.z for sp in spawn_points) / len(spawn_points)
    return carla.Location(x=avg_x, y=avg_y, z=avg_z)


def _compute_spawn_point_densities(spawn_points: List[carla.Transform], radius_m: float) -> List[int]:
    """Count nearby spawn points within a given radius for each point."""
    if not spawn_points or radius_m <= 0:
        return [0 for _ in spawn_points]
    r2 = float(radius_m) * float(radius_m)
    densities: List[int] = []
    for sp in spawn_points:
        loc = sp.location
        count = 0
        for other in spawn_points:
            dx = loc.x - other.location.x
            dy = loc.y - other.location.y
            dz = loc.z - other.location.z
            if dx * dx + dy * dy + dz * dz <= r2:
                count += 1
        densities.append(count)
    return densities


def _pick_center_spawn_index(spawn_points: List[carla.Transform]) -> int:
    """Pick the spawn point closest to the centroid."""
    if not spawn_points:
        return 0
    center = _compute_spawn_points_center(spawn_points)
    best_i = 0
    best_d = 1e18
    for i, sp in enumerate(spawn_points):
        d = dist(sp.location, center)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _pick_farthest_index(spawn_points: List[carla.Transform], from_loc: carla.Location, avoid_indices: set) -> int:
    """Pick the farthest spawn point from a reference location."""
    best_i = None
    best_d = -1.0
    for i, sp in enumerate(spawn_points):
        if i in avoid_indices:
            continue
        d = dist(sp.location, from_loc)
        if d > best_d:
            best_d = d
            best_i = i
    if best_i is None:
        return 0
    return int(best_i)


def _pick_farthest_index_filtered(
    spawn_points: List[carla.Transform],
    from_loc: carla.Location,
    avoid_indices: set,
    predicate,
) -> int:
    """Pick the farthest spawn point that satisfies a predicate."""
    best_i = None
    best_d = -1.0
    for i, sp in enumerate(spawn_points):
        if i in avoid_indices:
            continue
        try:
            if not predicate(i, sp):
                continue
        except Exception:
            continue
        d = dist(sp.location, from_loc)
        if d > best_d:
            best_d = d
            best_i = i
    if best_i is None:
        return _pick_farthest_index(spawn_points, from_loc, avoid_indices)
    return int(best_i)


def main() -> int:
    """Entry point for running a driving scenario."""
    ap = argparse.ArgumentParser()
    default_config = os.path.abspath(os.path.join(os.path.dirname(__file__), "config", "scenario.yaml"))
    ap.add_argument(
        "--config",
        default=default_config,
        help=f"Path to scenario YAML config. (default: {default_config})",
    )
    ap.add_argument("--no-launch-manual", action="store_true", help="Do not launch manual_control_steeringwheel.py.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.carla.seed)

    session = CarlaSession(cfg.carla.host, cfg.carla.port, cfg.carla.timeout)
    shutdown_flag_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "shutdown.flag"))

    mode = cfg.experiment.mode.strip().lower()
    map_preference = cfg.carla.map_preference
    if mode == "test":
        map_preference = cfg.experiment.test_map_preference

    world = session.load_preferred_map(map_preference)
    original_world_settings = world.get_settings()

    session.configure_sync(world, cfg.carla.sync, cfg.carla.fixed_delta_seconds)
    session.set_weather_preset(world, cfg.weather.preset)
    _configure_traffic_lights(world, cfg.traffic_lights)
    removed_heroes = _destroy_stale_hero_vehicles(world, preferred_role_name="hero")
    if removed_heroes > 0:
        print(f"[Runner] Removed stale hero vehicles at startup: {removed_heroes}")

    ticker = None
    if cfg.carla.sync:
        hz = 1.0 / float(cfg.carla.fixed_delta_seconds)
        ticker = WorldTicker(
            world,
            target_hz=hz,
            stats_interval_seconds=cfg.carla.ticker_stats_interval_seconds,
        )
        ticker.start()

    tm = session.get_traffic_manager(cfg.carla.traffic_manager.port)
    traffic_spawner = TrafficSpawner(session.client, world, tm, seed=cfg.carla.seed)
    traffic_spawner.configure_tm(
        synchronous_mode=cfg.carla.sync,
        hybrid_physics_radius=cfg.carla.traffic_manager.hybrid_physics_radius,
        global_speed_percentage_difference=cfg.carla.traffic_manager.global_speed_percentage_difference,
    )

    spawn_points = list(world.get_map().get_spawn_points())
    planner = RoutePlanner(world.get_map(), sampling_resolution=cfg.route.sampling_resolution)

    if mode == "test":
        route = None
        test_indices: List[int] = []
        if getattr(cfg.route, "test_spawn_points", None):
            seen = set()
            for raw_idx in cfg.route.test_spawn_points:
                try:
                    idx = int(raw_idx)
                except Exception:
                    continue
                if idx < 0 or idx >= len(spawn_points):
                    continue
                if idx in seen:
                    continue
                seen.add(idx)
                test_indices.append(idx)

        if len(test_indices) >= 2:
            start_idx = test_indices[0]
            start_tr = spawn_points[start_idx]
            print(f"[Runner] Test route (manual) spawn points: {test_indices}")
            legs: List[List[carla.Location]] = []
            for i, idx in enumerate(test_indices):
                nxt = test_indices[(i + 1) % len(test_indices)]
                legs.append(planner.build_route(spawn_points[idx].location, spawn_points[nxt].location))
            route = _concat_routes(legs)
        else:
            city_density_radius = float(getattr(cfg.route, "test_city_density_radius_m", 60.0))
            city_density_ratio = float(getattr(cfg.route, "test_city_density_ratio", 0.7))
            highway_distance_ratio = float(getattr(cfg.route, "test_highway_distance_ratio", 0.6))
            target_city_points = 4

            center_loc = _compute_spawn_points_center(spawn_points)
            max_center_dist = max((dist(sp.location, center_loc) for sp in spawn_points), default=0.0)

            densities = _compute_spawn_point_densities(spawn_points, city_density_radius)
            max_density = max(densities) if densities else 0
            city_candidates: List[int] = []
            if max_density > 0:
                city_candidates = [i for i, d in enumerate(densities) if d >= max_density * city_density_ratio]

            if not city_candidates:
                city_ratio = float(getattr(cfg.route, "test_city_radius_ratio", 0.35))
                city_radius = max_center_dist * max(0.0, min(1.0, city_ratio))
                city_candidates = [
                    i for i, sp in enumerate(spawn_points) if dist(sp.location, center_loc) <= city_radius
                ]
                if not city_candidates:
                    city_candidates = list(range(len(spawn_points)))

            city_indices: List[int] = []
            if densities and max_density > 0:
                first_city_idx = max(city_candidates, key=lambda i: densities[i])
            else:
                first_city_idx = min(
                    city_candidates, key=lambda i: dist(spawn_points[i].location, center_loc)
                )
            city_indices.append(first_city_idx)
            remaining = [i for i in city_candidates if i != first_city_idx]
            while remaining and len(city_indices) < target_city_points:
                def _min_dist_to_cities(idx: int) -> float:
                    loc = spawn_points[idx].location
                    return min(dist(loc, spawn_points[c].location) for c in city_indices)

                next_idx = max(remaining, key=_min_dist_to_cities)
                city_indices.append(next_idx)
                remaining.remove(next_idx)

            city_trs = [spawn_points[i] for i in city_indices]
            start_tr = city_trs[0]

            highway_min_dist = max_center_dist * max(0.0, min(1.0, highway_distance_ratio))
            highway_candidates = [
                i
                for i, sp in enumerate(spawn_points)
                if dist(sp.location, center_loc) >= highway_min_dist and i not in set(city_indices)
            ]
            if highway_candidates:
                highway1_idx = max(highway_candidates, key=lambda i: dist(spawn_points[i].location, center_loc))
                highway1_tr = spawn_points[highway1_idx]
            else:
                highway1_idx = None
                highway1_tr = None

            print(f"[Runner] Test route city spawn points: {city_indices}")
            if highway1_idx is not None:
                print(f"[Runner] Test route highway spawn point: {highway1_idx}")

            legs: List[List[carla.Location]] = []
            if highway1_tr is not None and len(city_trs) >= 2:
                legs.append(planner.build_route(city_trs[0].location, highway1_tr.location))
                legs.append(planner.build_route(highway1_tr.location, city_trs[1].location))
                for i in range(1, len(city_trs) - 1):
                    legs.append(planner.build_route(city_trs[i].location, city_trs[i + 1].location))
                legs.append(planner.build_route(city_trs[-1].location, city_trs[0].location))
            else:
                for i in range(len(city_trs)):
                    nxt = city_trs[(i + 1) % len(city_trs)]
                    legs.append(planner.build_route(city_trs[i].location, nxt.location))

            route = _concat_routes(legs)
    else:
        start_idx, end_idx = pick_spawn_point_indices(
            spawn_points=spawn_points,
            seed=cfg.carla.seed,
            start_spec=cfg.route.start_spawn_point,
            end_spec=cfg.route.end_spawn_point,
        )
        start_tr = spawn_points[start_idx]
        end_tr = spawn_points[end_idx]
        print(f"[Runner] Route start spawn point: {start_idx}")
        print(f"[Runner] Route end   spawn point: {end_idx}")

        mid_idx = _pick_far_spawn_point_index(
            spawn_points=spawn_points,
            avoid_indices={start_idx, end_idx},
            anchor_locations=[start_tr.location, end_tr.location],
        )
        mid_tr = spawn_points[mid_idx]
        print(f"[Runner] Route mid   spawn point: {mid_idx}")

        route = _concat_routes(
            [
                planner.build_route(start_tr.location, end_tr.location),
                planner.build_route(end_tr.location, mid_tr.location),
                planner.build_route(mid_tr.location, start_tr.location),
            ]
        )
    if not route:
        raise RuntimeError("RoutePlanner returned an empty route. Try different spawn points or a different map.")

    if cfg.route.draw:
        draw_route(world, route, step=cfg.route.draw_step, life_time=cfg.route.draw_life_time)

    vehicles_n = cfg.traffic.vehicles
    walkers_n = cfg.pedestrians.walkers
    if mode == "test":
        if cfg.experiment.test_vehicles > 0:
            vehicles_n = cfg.experiment.test_vehicles
        if cfg.experiment.test_walkers > 0:
            walkers_n = cfg.experiment.test_walkers

    vehicle_ids = traffic_spawner.spawn_vehicles(
        n=vehicles_n,
        tm_port=cfg.carla.traffic_manager.port,
        safe_radius_from_start=cfg.traffic.safe_radius_from_route_start,
        route_start=start_tr,
        min_distance_to_leading_vehicle=cfg.carla.traffic_manager.min_distance_to_leading_vehicle,
        ignore_lights_percentage=cfg.carla.traffic_manager.ignore_lights_percentage,
        ignore_signs_percentage=cfg.carla.traffic_manager.ignore_signs_percentage,
        random_left_lanechange_percentage=cfg.carla.traffic_manager.random_left_lanechange_percentage,
        random_right_lanechange_percentage=cfg.carla.traffic_manager.random_right_lanechange_percentage,
    )
    print(f"[Runner] Spawned vehicles: {len(vehicle_ids)}")

    ped_spawner = PedestrianSpawner(session.client, world, seed=cfg.carla.seed)
    walker_ids, controller_ids = ped_spawner.spawn_walkers(
        n=walkers_n,
        running_percentage=cfg.pedestrians.running_percentage,
        crossing_percentage=cfg.pedestrians.crossing_percentage,
        max_speed_walking=cfg.pedestrians.max_speed_walking,
        max_speed_running=cfg.pedestrians.max_speed_running,
    )
    print(f"[Runner] Spawned walkers: {len(walker_ids)} (controllers: {len(controller_ids)})")

    monitor = LapMonitor(
        world=world,
        route=route,
        start_transform=start_tr,
        waypoint_reached_threshold=cfg.route.waypoint_reached_threshold,
        auto_reset_on_finish=cfg.route.auto_reset_on_finish,
        reset_cooldown_seconds=cfg.route.reset_cooldown_seconds,
        preferred_role_name="hero",
        tick_source=ticker,
    )
    monitor.start()

    map_name = world.get_map().name.split("/")[-1]
    output_dir = _resolve_output_dir(cfg.experiment.output_dir)
    dataset_profile = (cfg.experiment.dataset_profile or "distraction").strip().lower()
    dataset_suffix = ""
    if mode == "test":
        dataset_suffix = cfg.experiment.test_dataset_suffix or "_test"
    elif dataset_profile == "baseline":
        dataset_suffix = cfg.experiment.baseline_dataset_suffix or "_baseline"
    elif dataset_profile == "distraction":
        dataset_suffix = cfg.experiment.distraction_dataset_suffix or "_distraction"
    print(f"[Runner] Dataset profile: {dataset_profile} (suffix='{dataset_suffix}')")
    max_duration_seconds = max(0.0, float(cfg.experiment.max_duration_seconds))
    if max_duration_seconds > 0.0:
        print(f"[Runner] Max scenario duration: {max_duration_seconds:.1f}s")
    else:
        print("[Runner] Max scenario duration disabled (<= 0).")

    run_id = _compute_next_run_id(
        output_dir=output_dir,
        user_id=cfg.experiment.user_id,
        suffix=dataset_suffix,
        dataset_profile=dataset_profile,
        mode=mode,
    )
    print(f"[Runner] Using run_id={run_id} (auto-increment per user_id).")
    context = DatasetContext(
        user_id=cfg.experiment.user_id,
        run_id=run_id,
        weather_label=cfg.experiment.weather_label,
        map_name=map_name,
    )
    model_inference = None
    if cfg.inference.enable_model:
        try:
            if cfg.inference.use_process:
                from src.model_inference import ModelInferenceProcess as ModelInferenceProvider
                model_inference = ModelInferenceProvider()
                model_inference.start()
                print("[Runner] Model inference started (process, 2 Hz, window=3).")
            else:
                from src.model_inference import ModelInferenceService as ModelInferenceProvider
                model_inference = ModelInferenceProvider()
                model_inference.start()
                print("[Runner] Model inference started (thread, 2 Hz, window=3).")
        except Exception as exc:
            print(f"[Runner] Model inference unavailable: {exc}")
            model_inference = None
    else:
        print("[Runner] Model inference disabled by config.")

    emotion_inference = None
    try:
        from src.emotion_inference import EmotionInferenceService

        if not cfg.inference.enable_emotion:
            print("[Runner] Emotion inference disabled by config.")
        elif model_inference is not None:
            emotion_inference = EmotionInferenceService(frame_provider=model_inference)
            emotion_inference.start()
            print("[Runner] Emotion inference started (1 Hz).")
        else:
            print("[Runner] Emotion inference disabled (model inference unavailable).")
    except Exception as exc:
        print(f"[Runner] Emotion inference unavailable: {exc}")
        emotion_inference = None

    arousal_client = None
    baseline_requires_sensor = dataset_profile == "baseline"
    if cfg.arousal_sensor.enabled:
        try:
            from src.arousal_ble import BleArousalProvider

            arousal_client = BleArousalProvider(
                device_name=cfg.arousal_sensor.device_name,
                baseline_seconds=cfg.arousal_sensor.baseline_seconds,
                smoothing_window=cfg.arousal_sensor.smoothing_window,
                reconnect_seconds=cfg.arousal_sensor.reconnect_seconds,
                no_sample_timeout_seconds=cfg.arousal_sensor.no_sample_timeout_seconds,
                debug=cfg.arousal_sensor.debug,
                debug_interval_seconds=cfg.arousal_sensor.debug_interval_seconds,
            )
            arousal_client.start()
            print("[Runner] Arousal BLE provider started.")
        except Exception as exc:
            print(f"[Runner] Arousal BLE provider unavailable: {exc}")
            arousal_client = None
    else:
        from src.arousal_provider import StaticArousalProvider

        arousal_client = StaticArousalProvider(
            value=cfg.arousal_sensor.placeholder_value,
            method="disabled",
            quality="disabled",
        )
        print("[Runner] Arousal pipeline disabled by config.")

    if baseline_requires_sensor and not cfg.arousal_sensor.enabled:
        raise RuntimeError(
            "Baseline run requires arousal sensor: set arousal_sensor.enabled=true."
        )
    if baseline_requires_sensor and arousal_client is None:
        raise RuntimeError(
            "Baseline run requires arousal sensor data, but the BLE provider is unavailable."
        )

    if cfg.arousal_sensor.enabled and arousal_client is not None:
        baseline_wait = float(cfg.arousal_sensor.baseline_seconds)
        if baseline_wait > 0:
            print(f"[Runner] Waiting {baseline_wait:.0f}s for arousal baseline calibration...")
            try:
                baseline_ready = bool(arousal_client.wait_for_baseline(timeout=baseline_wait))
            except Exception as exc:
                if baseline_requires_sensor:
                    raise RuntimeError(
                        f"Baseline run aborted: failed while waiting for sensor calibration ({exc})."
                    ) from exc
                time.sleep(baseline_wait)
            else:
                if not baseline_ready:
                    msg = "Baseline not completed yet (no data or still calibrating)."
                    if baseline_requires_sensor:
                        raise RuntimeError(f"Baseline run aborted: {msg}")
                    print(f"[Runner] {msg}")

    distractions_enabled = bool(cfg.distractions.enabled)
    if dataset_profile == "baseline":
        distractions_enabled = False

    error_logger = ErrorDatasetLogger(
        output_dir=output_dir,
        context=context,
        suffix=dataset_suffix,
        model_provider=model_inference,
        emotion_provider=emotion_inference,
    )
    baseline_time_logger = None
    if dataset_profile == "baseline":
        baseline_time_logger = BaselineDrivingTimeLogger(
            output_dir=output_dir,
            context=context,
            suffix=dataset_suffix,
        )
    distraction_logger = None
    if distractions_enabled:
        distraction_logger = DistractionDatasetLogger(
            output_dir=output_dir,
            context=context,
            suffix=dataset_suffix,
            model_provider=model_inference,
            arousal_provider=arousal_client,
            emotion_provider=emotion_inference,
        )

    error_monitor = ErrorMonitor(
        world=world,
        logger=error_logger,
        config=cfg.errors,
        preferred_role_name="hero",
        tick_source=ticker,
    )
    print(
        "[Runner] Error monitor frequencies: "
        f"red_light={cfg.errors.red_light_check_hz:.1f} Hz, "
        f"stop_sign={cfg.errors.stop_sign_check_hz:.1f} Hz."
    )
    error_monitor.start()

    monitor_rects = _get_monitor_rects()
    left_rect, center_rect, right_rect = _pick_monitor_layout(monitor_rects)
    preview_window = None
    if cfg.inference.enable_preview and model_inference is not None:
        if preview_available():
            try:
                preview_window = CameraPreviewWindow(
                    model_provider=model_inference,
                    arousal_provider=arousal_client,
                    emotion_provider=emotion_inference,
                    target_hz=cfg.inference.preview_hz,
                )
                preview_window.start()
            except Exception as exc:
                print(f"[Runner] Camera preview unavailable: {exc}")
        else:
            print("[Runner] Camera preview unavailable (opencv-python not installed).")
    elif not cfg.inference.enable_preview:
        print("[Runner] Camera preview disabled by config.")
    distraction_windows = []

    if distractions_enabled:
        if distraction_logger is None:
            raise RuntimeError("Distraction logger is required when distractions are enabled.")
        coord = DistractionCoordinator(cfg.distractions.min_gap_between_windows_seconds)
        window_titles = list(cfg.distractions.window_titles)
        while len(window_titles) < 2:
            window_titles.append(f"Distrazione {len(window_titles) + 1}")

        def _with_hero(fn) -> None:
            hero = find_hero_vehicle(world, preferred_role="hero")
            if hero is None:
                return
            fn(hero)

        def _on_distraction_start(window_id: str) -> None:
            _with_hero(lambda hero: distraction_logger.start(window_id, world, hero))

        def _on_distraction_finish(window_id: str) -> None:
            _with_hero(lambda hero: distraction_logger.finish(window_id, world, hero))

        def _refocus_sim() -> None:
            focus_simulation_window(cfg.distractions.simulation_window_title)

        distraction_windows = [
            DistractionWindow(
                window_id="window_1",
                title=window_titles[0],
                coordinator=coord,
                min_interval_seconds=cfg.distractions.min_interval_seconds,
                max_interval_seconds=cfg.distractions.max_interval_seconds,
                flash_duration_seconds=cfg.distractions.flash_duration_seconds,
                flash_start_interval_seconds=cfg.distractions.flash_start_interval_seconds,
                flash_min_interval_seconds=cfg.distractions.flash_min_interval_seconds,
                beep_frequency_hz=cfg.distractions.beep_frequency_hz,
                beep_duration_ms=cfg.distractions.beep_duration_ms,
                on_start=_on_distraction_start,
                on_finish=_on_distraction_finish,
                focus_callback=_refocus_sim,
                monitor_rect=left_rect,
                fullscreen=cfg.distractions.fullscreen,
                fill_monitor=cfg.distractions.fill_monitor,
                steal_focus=cfg.distractions.steal_focus,
                anchor="top_left",
                excluded_letters=tuple(cfg.distractions.excluded_letters),
            ),
            DistractionWindow(
                window_id="window_2",
                title=window_titles[1],
                coordinator=coord,
                min_interval_seconds=cfg.distractions.min_interval_seconds,
                max_interval_seconds=cfg.distractions.max_interval_seconds,
                flash_duration_seconds=cfg.distractions.flash_duration_seconds,
                flash_start_interval_seconds=cfg.distractions.flash_start_interval_seconds,
                flash_min_interval_seconds=cfg.distractions.flash_min_interval_seconds,
                beep_frequency_hz=cfg.distractions.beep_frequency_hz,
                beep_duration_ms=cfg.distractions.beep_duration_ms,
                on_start=_on_distraction_start,
                on_finish=_on_distraction_finish,
                focus_callback=_refocus_sim,
                monitor_rect=right_rect,
                fullscreen=cfg.distractions.fullscreen,
                fill_monitor=cfg.distractions.fill_monitor,
                steal_focus=cfg.distractions.steal_focus,
                anchor="top_right",
                excluded_letters=tuple(cfg.distractions.excluded_letters),
            ),
        ]
        for w in distraction_windows:
            w.start()
    else:
        if dataset_profile == "baseline":
            print("[Runner] Distraction windows disabled (baseline dataset profile).")
        else:
            print("[Runner] Distraction windows disabled by config.")

    proc = None
    drive_start_monotonic: Optional[float] = None
    baseline_pre_drive_snapshot: Optional[ArousalSnapshot] = None

    def _capture_pre_drive_baseline_snapshot() -> Optional[ArousalSnapshot]:
        if baseline_time_logger is None:
            return None
        if arousal_client is None:
            raise RuntimeError("Baseline run aborted: arousal sensor provider unavailable at start.")

        try:
            snapshot = arousal_client.get_snapshot()
        except Exception as exc:
            raise RuntimeError(
                f"Baseline run aborted: failed to read arousal sensor at start ({exc})."
            ) from exc

        try:
            hr_value = int(snapshot.hr_bpm) if snapshot.hr_bpm is not None else None
        except Exception:
            hr_value = None
        try:
            arousal_value = float(snapshot.value) if snapshot.value is not None else None
        except Exception:
            arousal_value = None

        if hr_value is None or hr_value < 35 or hr_value > 220:
            raise RuntimeError(
                "Baseline run aborted: invalid heart-rate sample at start."
            )
        if arousal_value is None or arousal_value < 0.0 or arousal_value > 1.0:
            raise RuntimeError(
                "Baseline run aborted: invalid arousal sample at start."
            )

        return snapshot

    mc_env = os.environ.copy()
    mc_env["SIM_SHUTDOWN_FILE"] = shutdown_flag_path
    if center_rect is not None:
        x, y, w, h = center_rect
        mc_env["SIM_WINDOW_X"] = str(x)
        mc_env["SIM_WINDOW_Y"] = str(y)
        mc_env["SIM_WINDOW_W"] = str(w)
        mc_env["SIM_WINDOW_H"] = str(h)
        mc_env["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"
    try:
        if not args.no_launch_manual:
            mc_path = _resolve_manual_control_path(cfg.manual_control.path)
            mc_args = list(cfg.manual_control.extra_args)
            max_speed_kmh = cfg.manual_control.max_speed_kmh
            if max_speed_kmh is not None and max_speed_kmh > 0:
                if not any(str(arg).startswith("--max-speed-kmh") for arg in mc_args):
                    mc_args.append(f"--max-speed-kmh={max_speed_kmh}")
            baseline_pre_drive_snapshot = _capture_pre_drive_baseline_snapshot()
            proc = _launch_manual_control(
                mc_path,
                cfg.carla.host,
                cfg.carla.port,
                mc_args,
                env=mc_env,
            )
            drive_start_monotonic = time.monotonic()

            while True:
                if max_duration_seconds > 0.0 and drive_start_monotonic is not None:
                    elapsed = time.monotonic() - drive_start_monotonic
                    if elapsed >= max_duration_seconds:
                        print(
                            f"[Runner] Scenario timeout reached at {elapsed:.1f}s "
                            f"(limit={max_duration_seconds:.1f}s). Stopping scenario."
                        )
                        try:
                            with open(shutdown_flag_path, "w", encoding="utf-8") as f:
                                f.write("shutdown\n")
                        except Exception:
                            pass
                        break
                if os.path.exists(shutdown_flag_path):
                    print("[Runner] Shutdown flag detected. Stopping scenario.")
                    break
                rc = proc.poll()
                if rc is not None:
                    print(f"[Runner] manual_control exited with code: {rc}")
                    break
                time.sleep(0.5)
        else:
            print("[Runner] --no-launch-manual set. Scenario running; you can start manual_control separately.")
            baseline_pre_drive_snapshot = _capture_pre_drive_baseline_snapshot()
            drive_start_monotonic = time.monotonic()
            while True:
                if max_duration_seconds > 0.0 and drive_start_monotonic is not None:
                    elapsed = time.monotonic() - drive_start_monotonic
                    if elapsed >= max_duration_seconds:
                        print(
                            f"[Runner] Scenario timeout reached at {elapsed:.1f}s "
                            f"(limit={max_duration_seconds:.1f}s). Stopping scenario."
                        )
                        try:
                            with open(shutdown_flag_path, "w", encoding="utf-8") as f:
                                f.write("shutdown\n")
                        except Exception:
                            pass
                        break
                if os.path.exists(shutdown_flag_path):
                    print("[Runner] Shutdown flag detected. Stopping scenario.")
                    break
                time.sleep(1.0)

    except KeyboardInterrupt:
        print("[Runner] KeyboardInterrupt received, shutting down.")
    finally:
        if baseline_time_logger is not None and drive_start_monotonic is not None:
            try:
                run_duration_seconds = max(0.0, time.monotonic() - drive_start_monotonic)
                total_seconds = baseline_time_logger.log_run_duration(
                    run_duration_seconds,
                    pre_drive_snapshot=baseline_pre_drive_snapshot,
                )
                print(
                    "[Runner] Baseline driving time saved "
                    f"(run={run_duration_seconds:.1f}s, user_total={total_seconds:.1f}s)."
                )
            except Exception as exc:
                print(f"[Runner] Failed to save baseline driving time: {exc}")

        monitor.stop()
        error_monitor.stop()
        for w in distraction_windows:
            w.stop()
        for w in distraction_windows:
            try:
                w.join(timeout=1.0)
            except Exception:
                pass
        if preview_window is not None:
            preview_window.stop()
        if arousal_client is not None:
            try:
                arousal_client.stop()
            except Exception:
                pass
            try:
                arousal_client.join(timeout=2.0)
            except Exception:
                pass

        if emotion_inference is not None:
            emotion_inference.stop()
            try:
                emotion_inference.join(timeout=2.0)
            except Exception:
                pass

        if model_inference is not None:
            model_inference.stop()
            try:
                model_inference.join(timeout=2.0)
            except Exception:
                pass

        if ticker is not None:
            ticker.stop()

        try:
            tm.set_synchronous_mode(False)
        except Exception:
            pass
        try:
            world.apply_settings(original_world_settings)
        except Exception:
            pass

        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

        try:
            if os.path.exists(shutdown_flag_path):
                os.remove(shutdown_flag_path)
        except Exception:
            pass

        _safe_destroy_ids(world, vehicle_ids + controller_ids + walker_ids)
        removed_heroes_end = _destroy_stale_hero_vehicles(world, preferred_role_name="hero")
        if removed_heroes_end > 0:
            print(f"[Runner] Removed stale hero vehicles at shutdown: {removed_heroes_end}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
