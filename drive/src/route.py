from __future__ import annotations

from typing import List
import carla

from src.agents.navigation.global_route_planner import GlobalRoutePlanner


class RoutePlanner:
    def __init__(self, world_map: carla.Map, sampling_resolution: float) -> None:
        # This GRP builds the graph in __init__, no setup() exists.
        self._grp = GlobalRoutePlanner(world_map, float(sampling_resolution))

    def build_route(self, start: carla.Location, end: carla.Location) -> List[carla.Location]:
        route = self._grp.trace_route(start, end)  # list[(Waypoint, RoadOption)]
        return [wp.transform.location for (wp, _opt) in route]





def draw_route(
    world: carla.World,
    route: List[carla.Location],
    step: int,
    life_time: float,
) -> None:
    if not route:
        return

    dbg = world.debug
    for i in range(0, len(route), max(step, 1)):
        loc = route[i]
        dbg.draw_string(
            loc + carla.Location(z=0.8),
            f"{i}",
            life_time=float(life_time),
            draw_shadow=False,
        )

    for i in range(1, len(route)):
        a = route[i - 1] + carla.Location(z=0.3)
        b = route[i] + carla.Location(z=0.3)
        dbg.draw_line(a, b, thickness=0.08, life_time=float(life_time))
