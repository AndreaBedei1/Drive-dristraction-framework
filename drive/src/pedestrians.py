from __future__ import annotations

from typing import List, Tuple
import random

import carla


class PedestrianSpawner:
    def __init__(self, client: carla.Client, world: carla.World, seed: int) -> None:
        self._client = client
        self._world = world
        self._rng = random.Random(seed)

    def spawn_walkers(
        self,
        n: int,
        running_percentage: int,
        crossing_percentage: int,
        max_speed_walking: float,
        max_speed_running: float,
    ) -> Tuple[List[int], List[int]]:
        blueprint_library = self._world.get_blueprint_library()
        walker_bps = blueprint_library.filter("walker.pedestrian.*")
        controller_bp = blueprint_library.find("controller.ai.walker")

        # This increases the probability of pedestrians crossing roads (if supported)
        try:
            self._world.set_pedestrians_cross_factor(float(crossing_percentage) / 100.0)
        except Exception:
            pass

        spawn_transforms: List[carla.Transform] = []
        for _ in range(n):
            loc = self._world.get_random_location_from_navigation()
            if loc is None:
                continue
            spawn_transforms.append(carla.Transform(loc))

        walker_batch = []
        for tr in spawn_transforms:
            bp = self._rng.choice(walker_bps)
            # Ensure pedestrians are not invincible so collisions affect them.
            if bp.has_attribute("is_invincible"):
                bp.set_attribute("is_invincible", "false")
            walker_batch.append(carla.command.SpawnActor(bp, tr))

        walker_results = self._client.apply_batch_sync(walker_batch, True)
        walker_ids: List[int] = []
        for r in walker_results:
            if r.error:
                continue
            walker_ids.append(r.actor_id)

        controller_batch = []
        for wid in walker_ids:
            controller_batch.append(carla.command.SpawnActor(controller_bp, carla.Transform(), wid))

        controller_results = self._client.apply_batch_sync(controller_batch, True)
        controller_ids: List[int] = []
        for r in controller_results:
            if r.error:
                continue
            controller_ids.append(r.actor_id)

        controllers = self._world.get_actors(controller_ids)

        for c in controllers:
            try:
                c.start()
                dest = self._world.get_random_location_from_navigation()
                if dest is not None:
                    c.go_to_location(dest)

                is_running = self._rng.randint(0, 99) < int(running_percentage)
                speed = float(max_speed_running if is_running else max_speed_walking)
                c.set_max_speed(speed)
            except Exception:
                pass

        return walker_ids, controller_ids

