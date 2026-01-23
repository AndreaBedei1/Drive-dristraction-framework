from __future__ import annotations

from typing import List, Tuple
import random

import carla

from .spawning import choose_vehicle_blueprints, set_random_color
from .utils import dist


class TrafficSpawner:
    def __init__(self, client: carla.Client, world: carla.World, tm: carla.TrafficManager, seed: int) -> None:
        self._client = client
        self._world = world
        self._tm = tm
        self._rng = random.Random(seed)

    def configure_tm(
        self,
        synchronous_mode: bool,
        hybrid_physics_radius: float,
        global_speed_percentage_difference: float,
    ) -> None:
        self._tm.set_synchronous_mode(synchronous_mode)
        self._tm.set_hybrid_physics_mode(True)
        self._tm.set_hybrid_physics_radius(float(hybrid_physics_radius))
        self._tm.set_random_device_seed(self._rng.randint(0, 10**9))
        self._tm.global_percentage_speed_difference(float(global_speed_percentage_difference))

    def spawn_vehicles(
        self,
        n: int,
        tm_port: int,
        safe_radius_from_start: float,
        route_start: carla.Transform,
        min_distance_to_leading_vehicle: float,
        ignore_lights_percentage: int,
        ignore_signs_percentage: int,
        random_left_lanechange_percentage: int,
        random_right_lanechange_percentage: int,
    ) -> List[int]:
        blueprints = self._world.get_blueprint_library()
        vehicle_bps = choose_vehicle_blueprints(blueprints)
        spawn_points = list(self._world.get_map().get_spawn_points())

        start_loc = route_start.location
        filtered: List[carla.Transform] = []
        for sp in spawn_points:
            if dist(sp.location, start_loc) >= safe_radius_from_start:
                filtered.append(sp)

        self._rng.shuffle(filtered)

        # Skip spawn points that are too close to existing or newly selected vehicles.
        existing_locs: List[carla.Location] = []
        for v in self._world.get_actors().filter("vehicle.*"):
            try:
                existing_locs.append(v.get_location())
            except Exception:
                pass

        min_spawn_dist = float(min_distance_to_leading_vehicle)
        selected: List[carla.Transform] = []
        for sp in filtered:
            too_close = False
            for loc in existing_locs:
                if dist(sp.location, loc) < min_spawn_dist:
                    too_close = True
                    break
            if too_close:
                continue
            for sel in selected:
                if dist(sp.location, sel.location) < min_spawn_dist:
                    too_close = True
                    break
            if too_close:
                continue
            selected.append(sp)
            if len(selected) >= max(n, 0):
                break

        batch = []
        for sp in selected:
            bp = self._rng.choice(vehicle_bps)
            set_random_color(bp, self._rng)
            bp.set_attribute("role_name", "autopilot")

            cmd = carla.command.SpawnActor(bp, sp).then(
                carla.command.SetAutopilot(carla.command.FutureActor, True, tm_port)
            )
            batch.append(cmd)

        results = self._client.apply_batch_sync(batch, True)

        actor_ids: List[int] = []
        for r in results:
            if r.error:
                continue
            actor_ids.append(r.actor_id)

        actors = self._world.get_actors(actor_ids)
        for a in actors:
            if not isinstance(a, carla.Vehicle):
                continue
            v: carla.Vehicle = a
            try:
                self._tm.distance_to_leading_vehicle(v, float(min_distance_to_leading_vehicle))
            except Exception:
                pass

            for fn_name, val in [
                ("ignore_lights_percentage", ignore_lights_percentage),
                ("ignore_signs_percentage", ignore_signs_percentage),
                ("random_left_lanechange_percentage", random_left_lanechange_percentage),
                ("random_right_lanechange_percentage", random_right_lanechange_percentage),
            ]:
                try:
                    fn = getattr(self._tm, fn_name)
                    fn(v, int(val))
                except Exception:
                    pass

        return actor_ids
