from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from typing import List

import carla

from src.config import load_config
from src.carla_session import CarlaSession
from src.utils import seed_everything, pick_spawn_point_indices, dist
from src.traffic import TrafficSpawner
from src.pedestrians import PedestrianSpawner
from src.route import RoutePlanner, draw_route
from src.monitor import LapMonitor
from src.ticker import WorldTicker  # <<< ADD


def _resolve_manual_control_path(path: str) -> str:
    p = os.path.abspath(path)
    if os.path.isfile(p):
        return p
    raise FileNotFoundError(f"manual_control path not found: {p}")


def _launch_manual_control(script_path: str, host: str, port: int, extra_args: List[str]) -> subprocess.Popen:
    cmd = [sys.executable, script_path, f"--host={host}", f"--port={port}"]
    cmd.extend(extra_args)
    print(f"[Runner] Launching manual control: {' '.join(cmd)}")
    return subprocess.Popen(cmd)

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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to scenario YAML config.")
    ap.add_argument("--no-launch-manual", action="store_true", help="Do not launch manual_control_steeringwheel.py.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed_everything(cfg.carla.seed)

    session = CarlaSession(cfg.carla.host, cfg.carla.port, cfg.carla.timeout)

    world = session.load_preferred_map(cfg.carla.map_preference)
    original_world_settings = world.get_settings()

    # Configure sync + weather.
    session.configure_sync(world, cfg.carla.sync, cfg.carla.fixed_delta_seconds)
    session.set_weather_preset(world, cfg.weather.preset)

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

    planner = RoutePlanner(world.get_map(), sampling_resolution=cfg.route.sampling_resolution)
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
    vehicle_ids = traffic_spawner.spawn_vehicles(
        n=cfg.traffic.vehicles,
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
        n=cfg.pedestrians.walkers,
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

    proc = None
    try:
        if not args.no_launch_manual:
            mc_path = _resolve_manual_control_path(cfg.manual_control.path)
            proc = _launch_manual_control(mc_path, cfg.carla.host, cfg.carla.port, cfg.manual_control.extra_args)

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

        actors = world.get_actors(vehicle_ids + walker_ids + controller_ids)
        for a in actors:
            try:
                a.destroy()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
