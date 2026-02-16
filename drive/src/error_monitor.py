from __future__ import annotations

"""Error detection and logging for the driving scenario."""

import threading
import time
import weakref
from typing import Dict, Optional, List, Tuple

import carla

from src.agents.tools.misc import get_trafficlight_trigger_location, is_within_distance
from src.datasets import ErrorDatasetLogger
from src.utils import find_hero_vehicle


class ErrorMonitor(threading.Thread):
    """Monitor the hero vehicle for safety violations and log them."""

    def __init__(
        self,
        world: carla.World,
        logger: ErrorDatasetLogger,
        config,
        preferred_role_name: str = "hero",
        tick_source: Optional[object] = None,
    ) -> None:
        """Create a monitor bound to a CARLA world and logger."""
        super().__init__(daemon=True)
        self._world = world
        self._logger = logger
        self._cfg = config
        self._role = preferred_role_name
        self._tick_source = tick_source
        self._stop_event = threading.Event()

        self._hero: Optional[carla.Vehicle] = None
        self._hero_id: Optional[int] = None
        self._last_hero_refresh_wall_time = 0.0
        self._hero_refresh_interval_seconds = 1.0
        self._collision_sensor: Optional[carla.Actor] = None
        self._lane_sensor: Optional[carla.Actor] = None

        self._last_speed_mps: Optional[float] = None
        self._last_time: Optional[float] = None
        self._last_sim_time: Optional[float] = None
        self._last_snapshot: Optional[carla.WorldSnapshot] = None

        self._last_harsh_brake_time = -1e9
        self._last_red_light_time = -1e9
        self._last_red_light_wall_time = -1e9
        self._last_red_light_id: Optional[int] = None
        self._last_red_light_along: Dict[int, float] = {}
        self._tracked_red_light_id: Optional[int] = None
        self._tracked_red_light_stop_wps: List[carla.Waypoint] = []
        self._last_solid_line_time = -1e9
        self._last_collision_time = -1e9
        self._last_static_collision_time: Dict[int, float] = {}
        self._last_stop_violation_time = -1e9
        self._last_stop_violation_loc: Optional[carla.Location] = None
        self._last_stop_violation_id: Optional[str] = None
        self._last_tl_debug_time = 0.0
        self._red_light_check_period = 1.0 / max(0.1, float(getattr(self._cfg, "red_light_check_hz", 15.0)))
        self._stop_sign_check_period = 1.0 / max(0.1, float(getattr(self._cfg, "stop_sign_check_hz", 15.0)))
        self._next_red_light_check_time = -1e9
        self._next_stop_sign_check_time = -1e9

        self._lights_list = self._world.get_actors().filter("*traffic_light*")
        self._lights_map: Dict[int, carla.Waypoint] = {}
        self._lights_stop_wps: Dict[int, List[carla.Waypoint]] = {}
        self._stop_signs: List[Tuple[str, carla.Waypoint, Optional[int], Optional[int]]] = []
        self._stop_zone_state: Dict[str, Dict[str, bool]] = {}
        self._last_stop_refresh_time = 0.0
        try:
            self._world_map = self._world.get_map()
        except Exception:
            self._world_map = None

    def stop(self) -> None:
        """Stop the thread and detach sensors."""
        self._stop_event.set()
        self._destroy_sensors()

    def _destroy_sensors(self) -> None:
        """Safely stop and destroy attached sensors."""
        for s in (self._collision_sensor, self._lane_sensor):
            if s is not None:
                try:
                    s.stop()
                except Exception:
                    pass
                try:
                    s.destroy()
                except Exception:
                    pass
        self._collision_sensor = None
        self._lane_sensor = None

    def _ensure_hero(self) -> Optional[carla.Vehicle]:
        """Ensure a hero vehicle reference and attach sensors if needed."""
        now = time.monotonic()

        if self._hero is not None:
            try:
                if not self._hero.is_alive:
                    self._hero = None
                    self._hero_id = None
            except Exception:
                self._hero = None
                self._hero_id = None

        should_refresh = (
            self._hero is None
            or now - self._last_hero_refresh_wall_time >= self._hero_refresh_interval_seconds
        )
        if should_refresh:
            self._last_hero_refresh_wall_time = now
            candidate = find_hero_vehicle(self._world, preferred_role=self._role)
            if candidate is not None:
                candidate_id = None
                try:
                    candidate_id = int(candidate.id)
                except Exception:
                    pass
                if self._hero is None or candidate_id != self._hero_id:
                    self._hero = candidate
                    self._hero_id = candidate_id
                    self._attach_sensors(self._hero)

        return self._hero

    def _attach_sensors(self, hero: carla.Vehicle) -> None:
        """Attach collision and lane invasion sensors to the hero."""
        self._destroy_sensors()
        bp_lib = self._world.get_blueprint_library()

        collision_bp = bp_lib.find("sensor.other.collision")
        self._collision_sensor = self._world.spawn_actor(collision_bp, carla.Transform(), attach_to=hero)
        weak_self = weakref.ref(self)
        self._collision_sensor.listen(lambda event: ErrorMonitor._on_collision(weak_self, event))

        lane_bp = bp_lib.find("sensor.other.lane_invasion")
        self._lane_sensor = self._world.spawn_actor(lane_bp, carla.Transform(), attach_to=hero)
        self._lane_sensor.listen(lambda event: ErrorMonitor._on_lane_invasion(weak_self, event))

    @staticmethod
    def _on_collision(weak_self: "weakref.ReferenceType[ErrorMonitor]", event) -> None:
        """Handle collision events and log them with cooldown."""
        self = weak_self()
        if self is None or self._hero is None:
            return
        now = self._safe_sim_time()

        other = event.other_actor
        other_id: Optional[int] = None
        other_type = ""
        if other is not None:
            try:
                other_type = other.type_id
            except Exception:
                pass
            try:
                other_id = int(other.id)
            except Exception:
                other_id = None

        other_type_norm = str(other_type).strip().lower()
        is_static_actor = other_type_norm.startswith("static.")

        speed_kmh = 0.0
        try:
            v = self._hero.get_velocity()
            speed_kmh = 3.6 * (v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5
        except Exception:
            speed_kmh = 0.0

        impulse_norm = 0.0
        try:
            impulse = event.normal_impulse
            impulse_norm = (impulse.x * impulse.x + impulse.y * impulse.y + impulse.z * impulse.z) ** 0.5
        except Exception:
            impulse_norm = 0.0

        if is_static_actor:
            if (
                speed_kmh < float(getattr(self._cfg, "collision_static_min_speed_kmh", 2.5))
                and impulse_norm < float(getattr(self._cfg, "collision_static_min_impulse", 120.0))
            ):
                return
            if other_id is not None:
                relog_seconds = float(getattr(self._cfg, "collision_static_relog_seconds", 12.0))
                last_t = self._last_static_collision_time.get(other_id, -1e9)
                if now - last_t < relog_seconds:
                    return
                self._last_static_collision_time[other_id] = now

        if now - self._last_collision_time < self._cfg.collision_cooldown_seconds:
            return
        self._last_collision_time = now

        if (
            other_type_norm.startswith("walker.pedestrian")
            or other_type_norm.startswith("controller.ai.walker")
            or other_type_norm.startswith("walker.")
            or "pedestrian" in other_type_norm
        ):
            error_type = "Vehicle-pedestrian collision"
        elif other_type_norm.startswith("vehicle."):
            error_type = "Vehicle collision"
        else:
            error_type = "Collision"

        self._logger.log(
            self._world,
            self._hero,
            error_type,
            details=other_type or "unknown_actor",
        )

    @staticmethod
    def _on_lane_invasion(weak_self: "weakref.ReferenceType[ErrorMonitor]", event) -> None:
        """Handle lane invasion events for solid line crossings."""
        self = weak_self()
        if self is None or self._hero is None:
            return
        now = self._safe_sim_time()
        if now - self._last_solid_line_time < self._cfg.solid_line_cooldown_seconds:
            return

        markings = getattr(event, "crossed_lane_markings", [])
        for m in markings:
            m_type = ""
            try:
                m_type = str(m.type)
            except Exception:
                pass
            if "Solid" in m_type:
                self._last_solid_line_time = now
                self._logger.log(self._world, self._hero, "Solid line crossing", details=m_type)
                break

    def _safe_sim_time(self) -> float:
        """Return simulation time, falling back to monotonic time."""
        if self._last_sim_time is not None:
            return self._last_sim_time
        try:
            return float(self._world.get_snapshot().timestamp.elapsed_seconds)
        except Exception:
            return time.monotonic()

    def _get_map(self) -> Optional[carla.Map]:
        """Return cached map object if available."""
        if self._world_map is None:
            try:
                self._world_map = self._world.get_map()
            except Exception:
                return None
        return self._world_map

    def _landmark_location(self, landmark) -> Optional[carla.Location]:
        """Extract a location from a map landmark object."""
        try:
            tr = landmark.transform
            return tr.location
        except Exception:
            pass
        try:
            return landmark.get_transform().location
        except Exception:
            pass
        try:
            return landmark.location
        except Exception:
            pass
        try:
            return landmark.get_location()
        except Exception:
            return None

    def _is_stop_landmark(self, landmark) -> bool:
        """Heuristically determine if a landmark represents a stop sign."""
        tokens = []
        for attr in ("type", "name", "text", "signal_type", "id"):
            try:
                val = getattr(landmark, attr)
            except Exception:
                continue
            if callable(val):
                try:
                    val = val()
                except Exception:
                    continue
            tokens.append(str(val))
        hay = " ".join(tokens).lower()
        return "stop" in hay

    def _refresh_stop_signs(self, now: float) -> None:
        """Refresh cached stop sign waypoints from the map."""
        if now - self._last_stop_refresh_time < 5.0:
            return
        self._last_stop_refresh_time = now
        self._stop_signs = []

        try:
            world_map = self._get_map()
            if world_map is not None and hasattr(world_map, "get_all_landmarks"):
                for lm in world_map.get_all_landmarks():
                    if not self._is_stop_landmark(lm):
                        continue
                    loc = self._landmark_location(lm)
                    if loc is None:
                        continue
                    road_id = None
                    lane_id = None
                    wp = None
                    try:
                        wp = world_map.get_waypoint(
                            loc, project_to_road=True, lane_type=carla.LaneType.Driving
                        )
                        road_id = int(wp.road_id)
                        lane_id = int(wp.lane_id)
                    except Exception:
                        pass
                    lm_id = getattr(lm, "id", None)
                    if lm_id is None:
                        lm_id = getattr(lm, "name", None) or f"lm_{len(self._stop_signs)}"
                    if wp is not None:
                        self._stop_signs.append((str(lm_id), wp, road_id, lane_id))
        except Exception:
            self._stop_signs = []

        if not self._stop_signs:
            try:
                world_map = self._get_map()
                if world_map is None:
                    return
                actors = self._world.get_actors().filter("*stop*")
                for a in actors:
                    if "stop" not in a.type_id:
                        continue
                    road_id = None
                    lane_id = None
                    wp = None
                    try:
                        wp = world_map.get_waypoint(
                            a.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
                        )
                        road_id = int(wp.road_id)
                        lane_id = int(wp.lane_id)
                    except Exception:
                        pass
                    if wp is not None:
                        self._stop_signs.append((str(a.id), wp, road_id, lane_id))
            except Exception:
                self._stop_signs = []

        if self._stop_signs:
            self._dedupe_stop_signs()

        if self._cfg.debug_stop_visualization:
            self._draw_stop_debug()

    def _dedupe_stop_signs(self) -> None:
        """Remove nearby duplicate stop sign entries."""
        if len(self._stop_signs) < 2:
            return
        dedupe_dist = float(self._cfg.stop_sign_dedupe_distance_m)
        unique: List[Tuple[str, carla.Waypoint, Optional[int], Optional[int]]] = []
        for sign_id, sign_wp, road_id, lane_id in self._stop_signs:
            if sign_wp is None:
                continue
            try:
                loc = sign_wp.transform.location
            except Exception:
                unique.append((sign_id, sign_wp, road_id, lane_id))
                continue
            duplicate = False
            for _u_id, u_wp, u_road_id, u_lane_id in unique:
                if u_wp is None:
                    continue
                if road_id is not None and u_road_id is not None and road_id != u_road_id:
                    continue
                if lane_id is not None and u_lane_id is not None and lane_id != u_lane_id:
                    continue
                try:
                    if loc.distance(u_wp.transform.location) <= dedupe_dist:
                        duplicate = True
                        break
                except Exception:
                    continue
            if not duplicate:
                unique.append((sign_id, sign_wp, road_id, lane_id))
        self._stop_signs = unique

    def _draw_stop_debug(self) -> None:
        """Draw stop sign debug geometry in the world."""
        try:
            dbg = self._world.debug
        except Exception:
            return
        life = float(self._cfg.debug_stop_life_time)
        zone_half_width = float(self._cfg.stop_sign_zone_half_width_m)
        zone_length = float(self._cfg.stop_sign_zone_length_m)
        for _sign_id, sign_wp, _road_id, _lane_id in self._stop_signs:
            try:
                loc = sign_wp.transform.location + carla.Location(z=0.8)
                fwd = sign_wp.transform.get_forward_vector()
                right = sign_wp.transform.get_right_vector()
                back = carla.Location(x=-fwd.x * zone_length, y=-fwd.y * zone_length, z=-fwd.z * zone_length)
                side = carla.Location(
                    x=right.x * zone_half_width, y=right.y * zone_half_width, z=right.z * zone_half_width
                )
                p0 = loc - side
                p1 = loc + side
                p2 = loc + back + side
                p3 = loc + back - side
                dbg.draw_point(loc, size=0.18, color=carla.Color(255, 0, 0), life_time=life)
                dbg.draw_line(p0, p1, thickness=0.08, color=carla.Color(255, 0, 0), life_time=life)
                dbg.draw_line(p1, p2, thickness=0.08, color=carla.Color(0, 255, 0), life_time=life)
                dbg.draw_line(p2, p3, thickness=0.08, color=carla.Color(0, 255, 0), life_time=life)
                dbg.draw_line(p3, p0, thickness=0.08, color=carla.Color(0, 255, 0), life_time=life)
            except Exception:
                pass

    def _shift_stop_waypoints_forward(self, stop_wps: List[carla.Waypoint]) -> List[carla.Waypoint]:
        """Shift stop waypoints slightly forward along the lane."""
        if not stop_wps:
            return stop_wps
        adjusted: List[carla.Waypoint] = []
        for wp in stop_wps:
            if wp is None:
                continue
            try:
                nxt = wp.next(1.0)
                adjusted.append(nxt[0] if nxt else wp)
            except Exception:
                adjusted.append(wp)
        return adjusted

    def _get_stop_waypoints(self, traffic_light: carla.TrafficLight) -> List[carla.Waypoint]:
        """Return cached stop line waypoints for a traffic light."""
        cached = self._lights_stop_wps.get(traffic_light.id)
        if cached is not None:
            return cached

        stop_wps: List[carla.Waypoint] = []
        try:
            if hasattr(traffic_light, "get_stop_waypoints"):
                stop_wps = list(traffic_light.get_stop_waypoints())
        except Exception:
            stop_wps = []
        if not stop_wps:
            try:
                trigger_loc = get_trafficlight_trigger_location(traffic_light)
                world_map = self._get_map()
                if world_map is not None:
                    wp = world_map.get_waypoint(
                        trigger_loc, project_to_road=True, lane_type=carla.LaneType.Driving
                    )
                    stop_wps = [wp]
            except Exception:
                stop_wps = []

        self._lights_stop_wps[traffic_light.id] = stop_wps
        return stop_wps

    def _set_tracked_red_light(self, traffic_light: carla.TrafficLight, stop_wps: List[carla.Waypoint]) -> None:
        """Track a red light after the trigger volume is left."""
        if not stop_wps:
            return
        self._tracked_red_light_id = traffic_light.id
        self._tracked_red_light_stop_wps = stop_wps

    def _clear_tracked_red_light(self) -> None:
        """Clear the currently tracked red light."""
        if self._tracked_red_light_id is not None:
            self._last_red_light_along.pop(self._tracked_red_light_id, None)
        self._tracked_red_light_id = None
        self._tracked_red_light_stop_wps = []

    def _log_red_light_violation(self, hero: carla.Vehicle, traffic_light_id: int, sim_time: float) -> None:
        """Log a red light violation and update cooldown timers."""
        self._last_red_light_id = traffic_light_id
        self._last_red_light_time = sim_time
        self._last_red_light_wall_time = time.monotonic()
        self._logger.log(self._world, hero, "Red light violation", details=f"traffic_light_id={traffic_light_id}")

    def _along_from_stop(self, stop_wp: carla.Waypoint, location: carla.Location) -> float:
        """Return the signed distance along the lane from the stop waypoint."""
        vec = location - stop_wp.transform.location
        fwd = stop_wp.transform.get_forward_vector()
        return vec.x * fwd.x + vec.y * fwd.y + vec.z * fwd.z

    def _lateral_distance(self, wp: carla.Waypoint, location: carla.Location) -> Optional[float]:
        """Return the lateral distance from a waypoint to a location."""
        try:
            right = wp.transform.get_right_vector()
            diff = location - wp.transform.location
            return abs(diff.x * right.x + diff.y * right.y + diff.z * right.z)
        except Exception:
            return None

    def _passed_stop_line(self, hero: carla.Vehicle, stop_wps: List[carla.Waypoint]) -> bool:
        """Check if the hero has passed the stop line on its current lane."""
        if not stop_wps:
            return False
        ego_loc = hero.get_location()
        world_map = self._get_map()
        if world_map is None:
            return False
        try:
            ego_wp = world_map.get_waypoint(
                ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            ego_road_id = int(ego_wp.road_id)
            ego_lane_id = int(ego_wp.lane_id)
        except Exception:
            return False

        min_pass = float(self._cfg.red_light_pass_distance_m) + float(self._cfg.red_light_pass_buffer_m)
        max_after = float(self._cfg.red_light_distance_m) + min_pass

        for stop_wp in stop_wps:
            if stop_wp is None:
                continue
            if stop_wp.road_id != ego_road_id or stop_wp.lane_id != ego_lane_id:
                continue
            ego_fwd = hero.get_transform().get_forward_vector()
            dir_dot = ego_fwd.x * stop_wp.transform.get_forward_vector().x + ego_fwd.y * stop_wp.transform.get_forward_vector().y + ego_fwd.z * stop_wp.transform.get_forward_vector().z
            if dir_dot < 0.5:
                continue
            along = self._along_from_stop(stop_wp, ego_loc)
            if along <= min_pass:
                continue
            if along > max_after:
                continue
            return True
        return False

    def _check_harsh_brake(self, hero: carla.Vehicle, sim_time: float, dt: float) -> None:
        """Detect and log harsh braking events."""
        if dt <= 0:
            return
        try:
            control = hero.get_control()
            if float(control.brake) < self._cfg.harsh_brake_min_brake:
                return
        except Exception:
            return
        v = hero.get_velocity()
        speed_mps = (v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5
        if self._last_speed_mps is None:
            self._last_speed_mps = speed_mps
            return
        decel = (self._last_speed_mps - speed_mps) / dt
        self._last_speed_mps = speed_mps

        speed_kmh = speed_mps * 3.6
        if speed_kmh < self._cfg.harsh_brake_min_speed_kmh:
            return
        if decel < self._cfg.harsh_brake_threshold_mps2:
            return
        if sim_time - self._last_harsh_brake_time < self._cfg.harsh_brake_cooldown_seconds:
            return

        self._last_harsh_brake_time = sim_time
        self._logger.log(self._world, hero, "Harsh braking", details=f"decel_mps2={decel:.3f}")

    def _check_red_light(self, hero: carla.Vehicle, speed_kmh: float, sim_time: float) -> None:
        """Detect and log red light violations."""
        if speed_kmh < self._cfg.red_light_min_speed_kmh:
            return
        min_interval = float(getattr(self._cfg, "red_light_min_interval_seconds", 0.0))
        if min_interval > 0.0 and time.monotonic() - self._last_red_light_wall_time < min_interval:
            return
        if sim_time - self._last_red_light_time < self._cfg.red_light_cooldown_seconds:
            return

        pass_threshold = float(self._cfg.red_light_pass_distance_m) + float(self._cfg.red_light_pass_buffer_m)
        track_distance = float(getattr(self._cfg, "red_light_track_distance_m", self._cfg.red_light_distance_m))

        try:
            tl = hero.get_traffic_light()
            tl_state = hero.get_traffic_light_state()
        except Exception:
            tl = None
            tl_state = None

        used_direct_tl = False
        if tl is not None and tl_state == carla.TrafficLightState.Red:
            stop_wps = self._get_stop_waypoints(tl)
            if stop_wps:
                used_direct_tl = True
                self._set_tracked_red_light(tl, stop_wps)
                crossed = False
                hero_loc = hero.get_location()
                for stop_wp in stop_wps:
                    try:
                        along = self._along_from_stop(stop_wp, hero_loc)
                    except Exception:
                        continue
                    prev = self._last_red_light_along.get(tl.id)
                    self._last_red_light_along[tl.id] = along
                    if prev is None:
                        continue
                    if prev <= pass_threshold and along > pass_threshold:
                        crossed = True
                        break
                if crossed:
                    if self._last_red_light_id != tl.id or sim_time - self._last_red_light_time >= self._cfg.red_light_cooldown_seconds:
                        self._log_red_light_violation(hero, tl.id, sim_time)
                    self._clear_tracked_red_light()
            if used_direct_tl:
                return

        ego_loc = hero.get_location()
        world_map = self._get_map()
        if world_map is None:
            return
        try:
            ego_wp = world_map.get_waypoint(
                ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
            )
        except Exception:
            return
        ego_fwd = hero.get_transform().get_forward_vector()
        max_distance = self._cfg.red_light_distance_m

        used_candidate = False
        for traffic_light in self._lights_list:
            if traffic_light.state != carla.TrafficLightState.Red:
                continue

            stop_wps = self._get_stop_waypoints(traffic_light)
            if not stop_wps:
                continue

            if traffic_light.id in self._lights_map:
                trigger_wp = self._lights_map[traffic_light.id]
            else:
                trigger_loc = get_trafficlight_trigger_location(traffic_light)
                trigger_wp = world_map.get_waypoint(
                    trigger_loc, project_to_road=True, lane_type=carla.LaneType.Driving
                )
                self._lights_map[traffic_light.id] = trigger_wp

            matched_lane = False
            for stop_wp in stop_wps:
                if stop_wp is None:
                    continue
                if stop_wp.road_id != ego_wp.road_id or stop_wp.lane_id != ego_wp.lane_id:
                    continue
                if stop_wp.transform.location.distance(ego_loc) > max_distance:
                    continue
                vec = stop_wp.transform.location - ego_loc
                dot = vec.x * ego_fwd.x + vec.y * ego_fwd.y + vec.z * ego_fwd.z
                if dot < 0:
                    continue
                dir_dot = ego_fwd.x * stop_wp.transform.get_forward_vector().x + ego_fwd.y * stop_wp.transform.get_forward_vector().y + ego_fwd.z * stop_wp.transform.get_forward_vector().z
                if dir_dot < 0.5:
                    continue
                if not is_within_distance(stop_wp.transform, hero.get_transform(), max_distance, [0, 90]):
                    continue
                matched_lane = True
                break

            if not matched_lane:
                continue
            used_candidate = True
            self._set_tracked_red_light(traffic_light, stop_wps)
            crossed = False
            for stop_wp in stop_wps:
                try:
                    along = self._along_from_stop(stop_wp, ego_loc)
                except Exception:
                    continue
                prev = self._last_red_light_along.get(traffic_light.id)
                self._last_red_light_along[traffic_light.id] = along
                if prev is None:
                    continue
                if prev <= pass_threshold and along > pass_threshold:
                    crossed = True
                    break
            if not crossed:
                continue

            if self._last_red_light_id == traffic_light.id and sim_time - self._last_red_light_time < self._cfg.red_light_cooldown_seconds:
                continue

            self._log_red_light_violation(hero, traffic_light.id, sim_time)
            self._clear_tracked_red_light()
            break

        if used_candidate:
            return

        if self._tracked_red_light_id is None or not self._tracked_red_light_stop_wps:
            return
        try:
            tracked_light = self._world.get_actor(self._tracked_red_light_id)
        except Exception:
            tracked_light = None
        if tracked_light is None:
            self._clear_tracked_red_light()
            return
        try:
            if tracked_light.state != carla.TrafficLightState.Red:
                self._clear_tracked_red_light()
                return
        except Exception:
            return

        crossed = False
        max_along = None
        hero_loc = hero.get_location()
        for stop_wp in self._tracked_red_light_stop_wps:
            try:
                along = self._along_from_stop(stop_wp, hero_loc)
            except Exception:
                continue
            if max_along is None or along > max_along:
                max_along = along
            prev = self._last_red_light_along.get(self._tracked_red_light_id)
            self._last_red_light_along[self._tracked_red_light_id] = along
            if prev is None:
                continue
            if prev <= pass_threshold and along > pass_threshold:
                crossed = True
                break

        if crossed:
            if (
                self._last_red_light_id != self._tracked_red_light_id
                or sim_time - self._last_red_light_time >= self._cfg.red_light_cooldown_seconds
            ):
                self._log_red_light_violation(hero, self._tracked_red_light_id, sim_time)
            self._clear_tracked_red_light()
            return

        if max_along is not None and max_along > pass_threshold + track_distance:
            self._clear_tracked_red_light()

    def _check_stop_sign(self, hero: carla.Vehicle, speed_kmh: float, sim_time: float) -> None:
        """Detect and log stop sign violations."""
        self._refresh_stop_signs(sim_time)
        if not self._stop_signs:
            return

        ego_loc = hero.get_location()
        min_speed = float(self._cfg.stop_sign_min_speed_kmh)
        zone_half_width = float(self._cfg.stop_sign_zone_half_width_m)
        zone_length = float(self._cfg.stop_sign_zone_length_m)
        world_map = self._get_map()
        try:
            if world_map is None:
                raise RuntimeError("map unavailable")
            ego_wp = world_map.get_waypoint(
                ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            ego_road_id = int(ego_wp.road_id)
            ego_lane_id = int(ego_wp.lane_id)
        except Exception:
            ego_wp = None
            ego_road_id = None
            ego_lane_id = None

        ego_fwd = hero.get_transform().get_forward_vector()
        for sign_id, sign_wp, sign_road_id, _sign_lane_id in self._stop_signs:
            if sign_wp is None:
                continue
            if ego_road_id is not None and sign_road_id is not None and ego_road_id != sign_road_id:
                continue
            if ego_lane_id is not None and _sign_lane_id is not None and ego_lane_id != _sign_lane_id:
                continue

            dir_dot = ego_fwd.x * sign_wp.transform.get_forward_vector().x + ego_fwd.y * sign_wp.transform.get_forward_vector().y + ego_fwd.z * sign_wp.transform.get_forward_vector().z
            if dir_dot < 0.5:
                continue
            lateral = self._lateral_distance(sign_wp, ego_loc)
            along = self._along_from_stop(sign_wp, ego_loc)
            in_zone = (-zone_length <= along <= 0.0) and (lateral is not None and lateral <= zone_half_width)

            state = self._stop_zone_state.get(sign_id)
            prev_in_zone = bool(state.get("in_zone")) if state is not None else False
            stopped = bool(state.get("stopped")) if state is not None else False

            if in_zone:
                if state is None:
                    state = {"in_zone": True, "stopped": False}
                else:
                    state["in_zone"] = True
                if speed_kmh <= min_speed:
                    state["stopped"] = True
                self._stop_zone_state[sign_id] = state
                continue

            if prev_in_zone:
                if not stopped:
                    if self._should_log_stop_violation(sim_time, ego_loc, sign_id):
                        self._logger.log(self._world, hero, "Stop sign violation", details=f"stop_sign_id={sign_id}")
                self._stop_zone_state.pop(sign_id, None)
            elif state is not None:
                self._stop_zone_state.pop(sign_id, None)

    def _should_log_stop_violation(self, sim_time: float, ego_loc: carla.Location, sign_id: str) -> bool:
        """Return True if a stop violation should be logged."""
        dedupe_time = float(self._cfg.stop_sign_dedupe_time_s)
        if sim_time - self._last_stop_violation_time < dedupe_time:
            return False
        self._last_stop_violation_time = sim_time
        self._last_stop_violation_loc = ego_loc
        self._last_stop_violation_id = sign_id
        return True

    def run(self) -> None:
        """Main thread loop: read ticks, update sensors, and check errors."""
        while not self._stop_event.is_set():
            try:
                if self._tick_source is None:
                    snapshot = self._world.wait_for_tick()
                else:
                    snapshot = self._tick_source.wait_for_tick(timeout=1.0)
                    if snapshot is None:
                        continue
            except Exception:
                time.sleep(0.05)
                continue

            self._last_snapshot = snapshot
            hero = self._ensure_hero()
            if hero is None:
                continue

            sim_time = float(snapshot.timestamp.elapsed_seconds)
            self._last_sim_time = sim_time
            if self._last_time is None:
                self._last_time = sim_time
                continue
            dt = sim_time - self._last_time
            self._last_time = sim_time

            try:
                v = hero.get_velocity()
                speed_kmh = 3.6 * (v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5
            except Exception:
                continue

            try:
                self._draw_traffic_light_stop_line(hero, sim_time)
                self._check_harsh_brake(hero, sim_time, dt)
                if sim_time >= self._next_red_light_check_time:
                    self._next_red_light_check_time = sim_time + self._red_light_check_period
                    self._check_red_light(hero, speed_kmh, sim_time)
                if sim_time >= self._next_stop_sign_check_time:
                    self._next_stop_sign_check_time = sim_time + self._stop_sign_check_period
                    self._check_stop_sign(hero, speed_kmh, sim_time)
            except Exception:
                pass

    def _draw_traffic_light_stop_line(self, hero: carla.Vehicle, now: float) -> None:
        """Draw a debug stop line for the relevant traffic light."""
        if not self._cfg.debug_stop_visualization:
            return
        refresh = max(0.5, min(2.0, float(self._cfg.debug_stop_life_time) * 0.5))
        if now - self._last_tl_debug_time < refresh:
            return
        self._last_tl_debug_time = now
        try:
            dbg = self._world.debug
        except Exception:
            return

        tl = None
        try:
            tl = hero.get_traffic_light()
        except Exception:
            tl = None

        hero_loc = None
        hero_wp = None
        hero_fwd = None
        if tl is None:
            try:
                hero_loc = hero.get_location()
                world_map = self._get_map()
                if world_map is None:
                    raise RuntimeError("map unavailable")
                hero_wp = world_map.get_waypoint(
                    hero_loc, project_to_road=True, lane_type=carla.LaneType.Driving
                )
                hero_fwd = hero.get_transform().get_forward_vector()
            except Exception:
                hero_loc = None
                hero_wp = None
                hero_fwd = None
            max_distance = float(self._cfg.red_light_distance_m)
            if hero_loc is not None and hero_wp is not None and hero_fwd is not None:
                for candidate in self._lights_list:
                    try:
                        stop_wps = self._get_stop_waypoints(candidate)
                    except Exception:
                        continue
                    for stop_wp in stop_wps:
                        if stop_wp is None:
                            continue
                        if stop_wp.road_id != hero_wp.road_id or stop_wp.lane_id != hero_wp.lane_id:
                            continue
                        if stop_wp.transform.location.distance(hero_loc) > max_distance:
                            continue
                        vec = stop_wp.transform.location - hero_loc
                        dot = vec.x * hero_fwd.x + vec.y * hero_fwd.y + vec.z * hero_fwd.z
                        if dot <= 0:
                            continue
                        tl = candidate
                        break
                    if tl is not None:
                        break

        if tl is None:
            return

        try:
            stop_wps = self._get_stop_waypoints(tl)
        except Exception:
            stop_wps = []
        if not stop_wps:
            return

        try:
            state = tl.state
        except Exception:
            state = None
        if state == carla.TrafficLightState.Red:
            color = carla.Color(255, 0, 0)
        elif state == carla.TrafficLightState.Yellow:
            color = carla.Color(255, 255, 0)
        elif state == carla.TrafficLightState.Green:
            color = carla.Color(0, 255, 0)
        else:
            color = carla.Color(255, 255, 255)

        life = float(self._cfg.debug_stop_life_time)
        pass_threshold = float(self._cfg.red_light_pass_distance_m) + float(self._cfg.red_light_pass_buffer_m)
        for wp in stop_wps:
            if wp is None:
                continue
            try:
                fwd = wp.transform.get_forward_vector()
                offset = carla.Location(x=fwd.x * pass_threshold, y=fwd.y * pass_threshold, z=fwd.z * pass_threshold)
                loc = wp.transform.location + offset + carla.Location(z=0.4)
                right = wp.transform.get_right_vector()
                half = float(getattr(wp, "lane_width", 3.5)) * 0.5
                p0 = loc - carla.Location(x=right.x * half, y=right.y * half, z=0.0)
                p1 = loc + carla.Location(x=right.x * half, y=right.y * half, z=0.0)
                dbg.draw_line(p0, p1, thickness=0.12, color=color, life_time=life)
            except Exception:
                continue
