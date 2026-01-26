from __future__ import annotations

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
from src.ticker import WorldTicker  # <<< ADD
from src.datasets import DatasetContext, ErrorDatasetLogger, DistractionDatasetLogger
from src.error_monitor import ErrorMonitor
from src.distraction_windows import DistractionCoordinator, DistractionWindow, focus_simulation_window


def _resolve_manual_control_path(path: str) -> str:
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
    cmd = [sys.executable, script_path, f"--host={host}", f"--port={port}"]
    cmd.extend(extra_args)
    print(f"[Runner] Launching manual control: {' '.join(cmd)}")
    return subprocess.Popen(cmd, env=env)


def _resolve_output_dir(path: str) -> str:
    return path if os.path.isabs(path) else os.path.abspath(path)


def _get_monitor_rects() -> List[Tuple[int, int, int, int]]:
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
    if not rects:
        return None, None, None
    if len(rects) == 1:
        return None, rects[0], None
    if len(rects) == 2:
        return rects[0], rects[0], rects[1]
    mid = rects[len(rects) // 2]
    return rects[0], mid, rects[-1]


def _compute_next_run_id(output_dir: str, user_id: str, suffix: str) -> int:
    import csv

    max_run_id = 0
    for filename in (f"Dataset Errors{suffix}.csv", f"Dataset Distractions{suffix}.csv"):
        path = os.path.join(output_dir, filename)
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if user_id and row.get("user_id") and row.get("user_id") != user_id:
                        continue
                    try:
                        rid = int(row.get("run_id", 0))
                    except Exception:
                        continue
                    if rid > max_run_id:
                        max_run_id = rid
        except Exception:
            pass
    return max_run_id + 1 if max_run_id > 0 else 1


def _safe_destroy_ids(world: carla.World, actor_ids: List[int]) -> None:
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


def _configure_traffic_lights(world: carla.World, cfg) -> None:
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
    # Pick the spawn point far from multiple anchors to create a longer loop.
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
    # Merge multiple route legs, avoiding duplicate join points.
    combined: List[carla.Location] = []
    for r in routes:
        if not r:
            continue
        if combined:
            combined.extend(r[1:])
        else:
            combined.extend(r)
    return combined


def _pick_center_spawn_index(spawn_points: List[carla.Transform]) -> int:
    # Pick the spawn point closest to the average of all spawn locations.
    if not spawn_points:
        return 0
    avg_x = sum(sp.location.x for sp in spawn_points) / len(spawn_points)
    avg_y = sum(sp.location.y for sp in spawn_points) / len(spawn_points)
    avg_z = sum(sp.location.z for sp in spawn_points) / len(spawn_points)
    center = carla.Location(x=avg_x, y=avg_y, z=avg_z)
    best_i = 0
    best_d = 1e18
    for i, sp in enumerate(spawn_points):
        d = dist(sp.location, center)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _pick_farthest_index(spawn_points: List[carla.Transform], from_loc: carla.Location, avoid_indices: set) -> int:
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to scenario YAML config.")
    ap.add_argument("--no-launch-manual", action="store_true", help="Do not launch manual_control_steeringwheel.py.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.carla.seed)

    session = CarlaSession(cfg.carla.host, cfg.carla.port, cfg.carla.timeout)

    mode = cfg.experiment.mode.strip().lower()
    map_preference = cfg.carla.map_preference
    if mode == "test":
        map_preference = cfg.experiment.test_map_preference

    world = session.load_preferred_map(map_preference)
    original_world_settings = world.get_settings()

    # Configure sync + weather.
    session.configure_sync(world, cfg.carla.sync, cfg.carla.fixed_delta_seconds)
    session.set_weather_preset(world, cfg.weather.preset)
    _configure_traffic_lights(world, cfg.traffic_lights)

    # >>> START TICKER (only in synchronous mode)
    ticker = None
    if cfg.carla.sync:
        hz = 1.0 / float(cfg.carla.fixed_delta_seconds)
        ticker = WorldTicker(world, target_hz=hz)
        ticker.start()

    # Traffic Manager
    tm = session.get_traffic_manager(cfg.carla.traffic_manager.port)
    traffic_spawner = TrafficSpawner(session.client, world, tm, seed=cfg.carla.seed)
    traffic_spawner.configure_tm(
        synchronous_mode=cfg.carla.sync,
        hybrid_physics_radius=cfg.carla.traffic_manager.hybrid_physics_radius,
        global_speed_percentage_difference=cfg.carla.traffic_manager.global_speed_percentage_difference,
    )

    # Build route using GlobalRoutePlanner
    spawn_points = list(world.get_map().get_spawn_points())
    planner = RoutePlanner(world.get_map(), sampling_resolution=cfg.route.sampling_resolution)

    if mode == "test":
        # Test route: start near center, then visit far points to cover more map.
        start_idx = _pick_center_spawn_index(spawn_points)
        start_tr = spawn_points[start_idx]
        far1_idx = _pick_farthest_index(spawn_points, start_tr.location, {start_idx})
        far1_tr = spawn_points[far1_idx]
        far2_idx = _pick_farthest_index(spawn_points, far1_tr.location, {start_idx, far1_idx})
        far2_tr = spawn_points[far2_idx]
        far3_idx = _pick_farthest_index(spawn_points, far2_tr.location, {start_idx, far1_idx, far2_idx})
        far3_tr = spawn_points[far3_idx]

        print(f"[Runner] Route start spawn point: {start_idx}")
        print(f"[Runner] Route far1  spawn point: {far1_idx}")
        print(f"[Runner] Route far2  spawn point: {far2_idx}")
        print(f"[Runner] Route far3  spawn point: {far3_idx}")

        route = _concat_routes(
            [
                planner.build_route(start_tr.location, far1_tr.location),
                planner.build_route(far1_tr.location, far2_tr.location),
                planner.build_route(far2_tr.location, far3_tr.location),
                planner.build_route(far3_tr.location, start_tr.location),
            ]
        )
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

    # Spawn traffic vehicles (autopilot)
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

    # Spawn pedestrians
    ped_spawner = PedestrianSpawner(session.client, world, seed=cfg.carla.seed)
    walker_ids, controller_ids = ped_spawner.spawn_walkers(
        n=walkers_n,
        running_percentage=cfg.pedestrians.running_percentage,
        crossing_percentage=cfg.pedestrians.crossing_percentage,
        max_speed_walking=cfg.pedestrians.max_speed_walking,
        max_speed_running=cfg.pedestrians.max_speed_running,
    )
    print(f"[Runner] Spawned walkers: {len(walker_ids)} (controllers: {len(controller_ids)})")

    # Start lap monitor
    monitor = LapMonitor(
        world=world,
        route=route,
        start_transform=start_tr,
        waypoint_reached_threshold=cfg.route.waypoint_reached_threshold,
        auto_reset_on_finish=cfg.route.auto_reset_on_finish,
        reset_cooldown_seconds=cfg.route.reset_cooldown_seconds,
        preferred_role_name="hero",
    )
    monitor.start()

    # Dataset loggers
    map_name = world.get_map().name.split("/")[-1]
    output_dir = _resolve_output_dir(cfg.experiment.output_dir)
    dataset_suffix = ""
    if mode == "test":
        dataset_suffix = cfg.experiment.test_dataset_suffix or "_test"

    run_id = _compute_next_run_id(output_dir, cfg.experiment.user_id, dataset_suffix)
    context = DatasetContext(
        user_id=cfg.experiment.user_id,
        run_id=run_id,
        weather_label=cfg.experiment.weather_label,
        map_name=map_name,
    )
    error_logger = ErrorDatasetLogger(output_dir=output_dir, context=context, suffix=dataset_suffix)
    distraction_logger = DistractionDatasetLogger(output_dir=output_dir, context=context, suffix=dataset_suffix)

    # Error monitor
    error_monitor = ErrorMonitor(world=world, logger=error_logger, config=cfg.errors, preferred_role_name="hero")
    error_monitor.start()

    # Distraction windows (two independent windows, coordinated)
    monitor_rects = _get_monitor_rects()
    left_rect, center_rect, right_rect = _pick_monitor_layout(monitor_rects)
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
        ),
    ]
    for w in distraction_windows:
        w.start()

    proc = None
    mc_env = None
    if center_rect is not None:
        x, y, w, h = center_rect
        mc_env = os.environ.copy()
        mc_env["SIM_WINDOW_X"] = str(x)
        mc_env["SIM_WINDOW_Y"] = str(y)
        mc_env["SIM_WINDOW_W"] = str(w)
        mc_env["SIM_WINDOW_H"] = str(h)
        mc_env["SDL_VIDEO_WINDOW_POS"] = f"{x},{y}"
    try:
        if not args.no_launch_manual:
            mc_path = _resolve_manual_control_path(cfg.manual_control.path)
            proc = _launch_manual_control(
                mc_path,
                cfg.carla.host,
                cfg.carla.port,
                cfg.manual_control.extra_args,
                env=mc_env,
            )

            while True:
                rc = proc.poll()
                if rc is not None:
                    print(f"[Runner] manual_control exited with code: {rc}")
                    break
                time.sleep(0.5)
        else:
            print("[Runner] --no-launch-manual set. Scenario running; you can start manual_control separately.")
            while True:
                time.sleep(1.0)

    except KeyboardInterrupt:
        print("[Runner] KeyboardInterrupt received, shutting down.")
    finally:
        monitor.stop()
        error_monitor.stop()
        for w in distraction_windows:
            w.stop()

        if ticker is not None:
            ticker.stop()

        # Restore async mode so the server UI remains responsive after exit.
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

        _safe_destroy_ids(world, vehicle_ids + controller_ids + walker_ids)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
