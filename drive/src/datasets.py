from __future__ import annotations

import csv
import os
import threading
from dataclasses import dataclass
import datetime
from typing import Any, Dict, Optional, List

import carla


@dataclass(frozen=True)
class DatasetContext:
    user_id: str
    run_id: int
    weather_label: str
    map_name: str


class _CsvWriter:
    def __init__(self, path: str, fieldnames: List[str]) -> None:
        self._path = path
        self._fieldnames = fieldnames
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()

    def append(self, row: Dict[str, Any]) -> None:
        with self._lock, open(self._path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow(row)


def _map_short_name(world: carla.World) -> str:
    try:
        return world.get_map().name.split("/")[-1]
    except Exception:
        return "unknown"


def _speed_kmh(vehicle: carla.Actor) -> float:
    try:
        v = vehicle.get_velocity()
        return 3.6 * (v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5
    except Exception:
        return 0.0


def _location_info(world: carla.World, location: carla.Location) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "x": float(location.x),
        "y": float(location.y),
        "z": float(location.z),
        "road_id": "",
        "lane_id": "",
    }
    try:
        wp = world.get_map().get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Any)
        data["road_id"] = int(wp.road_id)
        data["lane_id"] = int(wp.lane_id)
    except Exception:
        pass
    return data


def _snapshot_info(world: carla.World) -> Dict[str, Any]:
    try:
        snap = world.get_snapshot()
        ts = snap.timestamp
        return {
            "frame": int(ts.frame),
            "sim_time_seconds": float(ts.elapsed_seconds),
        }
    except Exception:
        return {
            "frame": "",
            "sim_time_seconds": "",
        }


def _wall_time_iso() -> str:
    return datetime.datetime.utcnow().isoformat()


class ErrorDatasetLogger:
    def __init__(self, output_dir: str, context: DatasetContext, suffix: str = "") -> None:
        path = os.path.join(output_dir, f"Dataset Errors{suffix}.csv")
        self._context = context
        self._writer = _CsvWriter(
            path,
            [
                "user_id",
                "run_id",
                "weather",
                "map_name",
                "error_type",
                "speed_kmh",
                "timestamp",
                "frame",
                "sim_time_seconds",
                "x",
                "y",
                "z",
                "road_id",
                "lane_id",
                "details",
            ],
        )

    def log(
        self,
        world: carla.World,
        vehicle: carla.Actor,
        error_type: str,
        details: str = "",
    ) -> None:
        try:
            location = vehicle.get_location()
        except Exception:
            return

        row: Dict[str, Any] = {
            "user_id": self._context.user_id,
            "run_id": self._context.run_id,
            "weather": self._context.weather_label,
            "map_name": self._context.map_name,
            "error_type": error_type,
            "speed_kmh": round(_speed_kmh(vehicle), 3),
            "timestamp": _wall_time_iso(),
            "details": details,
        }

        row.update(_snapshot_info(world))
        row.update(_location_info(world, location))
        self._writer.append(row)


class DistractionDatasetLogger:
    def __init__(self, output_dir: str, context: DatasetContext, suffix: str = "") -> None:
        path = os.path.join(output_dir, f"Dataset Distractions{suffix}.csv")
        self._context = context
        self._writer = _CsvWriter(
            path,
            [
                "user_id",
                "run_id",
                "weather",
                "map_name",
                "start_x",
                "start_y",
                "start_z",
                "end_x",
                "end_y",
                "end_z",
                "arousal_start",
                "arousal_end",
                "model_pred_start",
                "model_prob_start",
                "model_pred_end",
                "model_prob_end",
                "timestamp_start",
                "timestamp_end",
                "frame_start",
                "frame_end",
                "sim_time_start",
                "sim_time_end",
                "details",
            ],
        )
        self._active: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start(self, window_id: str, world: carla.World, vehicle: carla.Actor) -> None:
        with self._lock:
            if window_id in self._active:
                return
            try:
                location = vehicle.get_location()
            except Exception:
                return
            snap = _snapshot_info(world)
            self._active[window_id] = {
                "start_location": location,
                "frame_start": snap.get("frame", ""),
                "sim_time_start": snap.get("sim_time_seconds", ""),
                "timestamp_start": _wall_time_iso(),
            }

    def finish(self, window_id: str, world: carla.World, vehicle: carla.Actor) -> None:
        with self._lock:
            start_info = self._active.pop(window_id, None)
        if start_info is None:
            return
        try:
            end_location = vehicle.get_location()
        except Exception:
            return
        snap = _snapshot_info(world)

        start_loc: carla.Location = start_info["start_location"]
        row: Dict[str, Any] = {
            "user_id": self._context.user_id,
            "run_id": self._context.run_id,
            "weather": self._context.weather_label,
            "map_name": self._context.map_name,
            "start_x": float(start_loc.x),
            "start_y": float(start_loc.y),
            "start_z": float(start_loc.z),
            "end_x": float(end_location.x),
            "end_y": float(end_location.y),
            "end_z": float(end_location.z),
            "arousal_start": "",
            "arousal_end": "",
            "model_pred_start": "",
            "model_prob_start": "",
            "model_pred_end": "",
            "model_prob_end": "",
            "timestamp_start": start_info.get("timestamp_start", ""),
            "timestamp_end": _wall_time_iso(),
            "frame_start": start_info.get("frame_start", ""),
            "frame_end": snap.get("frame", ""),
            "sim_time_start": start_info.get("sim_time_start", ""),
            "sim_time_end": snap.get("sim_time_seconds", ""),
            "details": window_id,
        }
        self._writer.append(row)
