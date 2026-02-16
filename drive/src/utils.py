from __future__ import annotations

"""Shared utility helpers."""

from typing import List, Optional, Tuple
import math
import random

import carla


def seed_everything(seed: int) -> None:
    """Seed Python's RNG."""
    random.seed(seed)


def dist(a: carla.Location, b: carla.Location) -> float:
    """Return Euclidean distance between two locations."""
    dx = a.x - b.x
    dy = a.y - b.y
    dz = a.z - b.z
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def find_hero_vehicle(world: carla.World, preferred_role: str = "hero") -> Optional[carla.Vehicle]:
    """Return the newest alive vehicle with the given role name."""
    vehicles = world.get_actors().filter("vehicle.*")
    heroes: List[carla.Vehicle] = []
    for v in vehicles:
        try:
            role = v.attributes.get("role_name", "")
        except Exception:
            role = ""
        if role != preferred_role:
            continue
        try:
            if hasattr(v, "is_alive") and not v.is_alive:
                continue
        except Exception:
            pass
        heroes.append(v)
    if not heroes:
        return None
    try:
        heroes.sort(key=lambda a: int(a.id), reverse=True)
    except Exception:
        pass
    return heroes[0]


def pick_spawn_point_indices(
    spawn_points: List[carla.Transform],
    seed: int,
    start_spec: object,
    end_spec: object,
) -> Tuple[int, int]:
    """Pick start/end spawn point indices based on specs."""
    rng = random.Random(seed)

    n = len(spawn_points)
    if n < 2:
        raise RuntimeError("Not enough spawn points to build a route.")

    def _as_index(spec: object) -> Optional[int]:
        if isinstance(spec, int):
            if spec < 0 or spec >= n:
                raise ValueError(f"Spawn point index out of range: {spec}")
            return spec
        return None

    start_idx = _as_index(start_spec)
    end_idx = _as_index(end_spec)

    if start_idx is None and str(start_spec) == "auto" and str(end_spec) not in ("auto", "auto_far"):
        start_idx = rng.randrange(n)

    if start_idx is None and str(start_spec) == "auto" and str(end_spec) in ("auto", "auto_far"):
        start_idx = rng.randrange(n)

    if end_idx is None and str(end_spec) == "auto":
        end_idx = rng.randrange(n)
        if end_idx == start_idx:
            end_idx = (end_idx + 1) % n

    if end_idx is None and str(end_spec) == "auto_far":
        s_loc = spawn_points[start_idx].location
        best_i = None
        best_d = -1.0
        for i, t in enumerate(spawn_points):
            if i == start_idx:
                continue
            d = dist(s_loc, t.location)
            if d > best_d:
                best_d = d
                best_i = i
        end_idx = int(best_i)

    if end_idx is None and str(end_spec) == "auto":
        end_idx = rng.randrange(n)

    if start_idx == end_idx:
        end_idx = (end_idx + 1) % n

    return int(start_idx), int(end_idx)
