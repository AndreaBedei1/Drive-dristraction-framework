from __future__ import annotations

from typing import Iterable, List, Optional
import random

import carla


def destroy_actors(world: carla.World, actor_ids: Iterable[int]) -> None:
    actors = world.get_actors(list(actor_ids))
    for a in actors:
        try:
            a.destroy()
        except Exception:
            pass


def choose_vehicle_blueprints(blueprints: carla.BlueprintLibrary) -> List[carla.ActorBlueprint]:
    bps = blueprints.filter("vehicle.*")
    out: List[carla.ActorBlueprint] = []
    for bp in bps:
        # Avoid bikes and a few special vehicles
        wheels = bp.get_attribute("number_of_wheels") if bp.has_attribute("number_of_wheels") else None
        if wheels is not None and int(wheels.as_int()) < 4:
            continue
        out.append(bp)
    return out


def set_random_color(bp: carla.ActorBlueprint, rng: random.Random) -> None:
    if bp.has_attribute("color"):
        colors = bp.get_attribute("color").recommended_values
        if colors:
            bp.set_attribute("color", rng.choice(colors))
