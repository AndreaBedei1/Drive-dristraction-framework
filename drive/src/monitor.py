from __future__ import annotations

"""Lap monitoring utilities for looping routes."""

from typing import List, Optional
import threading
import time

import carla

from .utils import dist, find_hero_vehicle


class LapMonitor(threading.Thread):
    """Track progress along a route and optionally reset on completion."""

    def __init__(
        self,
        world: carla.World,
        route: List[carla.Location],
        start_transform: carla.Transform,
        waypoint_reached_threshold: float,
        auto_reset_on_finish: bool,
        reset_cooldown_seconds: float,
        preferred_role_name: str = "hero",
        tick_source: Optional[object] = None,
    ) -> None:
        """Create a lap monitor for the hero vehicle."""
        super().__init__(daemon=True)

        self._world = world
        self._route = route
        self._start_transform = start_transform
        self._thr = float(waypoint_reached_threshold)
        self._auto_reset = bool(auto_reset_on_finish)
        self._cooldown = float(reset_cooldown_seconds)
        self._role = preferred_role_name
        self._tick_source = tick_source

        self._stop_event = threading.Event()

        self._lap = 0
        self._idx = 0
        self._hero: Optional[carla.Vehicle] = None
        self._hero_initialized = False

    def stop(self) -> None:
        """Stop the monitor loop."""
        self._stop_event.set()

    def run(self) -> None:
        """Main loop that tracks route progress."""
        if not self._route:
            return

        while not self._stop_event.is_set():
            try:
                if self._tick_source is None:
                    self._world.wait_for_tick()
                else:
                    snap = self._tick_source.wait_for_tick(timeout=1.0)
                    if snap is None:
                        continue
            except Exception:
                time.sleep(0.1)
                continue

            if self._hero is None:
                self._hero = find_hero_vehicle(self._world, preferred_role=self._role)
                if self._hero is None:
                    continue

            if not self._hero_initialized:
                try:
                    self._hero.set_transform(self._start_transform)
                    self._hero.set_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                    self._hero.set_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                except Exception:
                    pass

                self._idx = 0
                self._hero_initialized = True
                print("[LapMonitor] Hero initialized at route start.")

            try:
                hero_loc = self._hero.get_location()
            except Exception:
                self._hero = None
                self._hero_initialized = False
                continue

            if self._idx < len(self._route) and dist(hero_loc, self._route[self._idx]) <= self._thr:
                self._idx += 1

            if self._idx >= len(self._route):
                self._lap += 1
                print(f"[LapMonitor] Lap completed: {self._lap}")

                if self._auto_reset:
                    try:
                        self._hero.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))
                    except Exception:
                        pass

                    time.sleep(self._cooldown)

                    try:
                        self._hero.set_transform(self._start_transform)
                        self._hero.set_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                        self._hero.set_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
                    except Exception:
                        pass

                self._idx = 0
