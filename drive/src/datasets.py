from __future__ import annotations

"""CSV dataset logging utilities for errors and distractions."""

import csv
import os
import threading
from dataclasses import dataclass
import datetime
from typing import Any, Dict, Optional, List, Protocol, Tuple

import carla

from src.arousal_provider import ArousalProvider, ArousalSnapshot

@dataclass(frozen=True)
class DatasetContext:
    """Shared metadata stored with each dataset row."""
    user_id: str
    run_id: int
    weather_label: str
    map_name: str


class ModelInferenceProvider(Protocol):
    """Protocol for model inference providers used by dataset loggers."""

    def get_window_summary(self) -> Tuple[str, float]:
        """Return (label, probability) for the latest inference window."""
        ...


class _CsvWriter:
    """Thread-safe CSV appender with header creation."""

    def __init__(self, path: str, fieldnames: List[str]) -> None:
        """Create a writer for the target CSV path."""
        self._path = path
        self._fieldnames = fieldnames
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()

    def append(self, row: Dict[str, Any]) -> None:
        """Append a row to the CSV file."""
        with self._lock, open(self._path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames)
            writer.writerow(row)


def _map_short_name(world: carla.World) -> str:
    """Return a short map name from the CARLA world."""
    try:
        return world.get_map().name.split("/")[-1]
    except Exception:
        return "unknown"


def _speed_kmh(vehicle: carla.Actor) -> float:
    """Compute speed in km/h for a CARLA actor."""
    try:
        v = vehicle.get_velocity()
        return 3.6 * (v.x * v.x + v.y * v.y + v.z * v.z) ** 0.5
    except Exception:
        return 0.0


def _location_info(world: carla.World, location: carla.Location) -> Dict[str, Any]:
    """Return location fields and road/lane ids for a location."""
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
    """Return frame and sim time information from the world snapshot."""
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
    """Return current UTC timestamp in ISO format."""
    return datetime.datetime.utcnow().isoformat()


class ErrorDatasetLogger:
    """Logger for error events into Dataset Errors CSV."""

    def __init__(
        self,
        output_dir: str,
        context: DatasetContext,
        suffix: str = "",
        model_provider: Optional[ModelInferenceProvider] = None,
    ) -> None:
        """Create a logger with a destination folder and context."""
        path = os.path.join(output_dir, f"Dataset Errors{suffix}.csv")
        self._context = context
        self._model_provider = model_provider
        self._writer = _CsvWriter(
            path,
            [
                "user_id",
                "run_id",
                "weather",
                "map_name",
                "error_type",
                "model_pred_start",
                "model_prob_start",
                "model_pred_end",
                "model_prob_end",
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

    def _model_snapshot(self) -> Tuple[str, float]:
        """Return the latest aggregated model label and probability."""
        if self._model_provider is None:
            return "None", 1.0
        try:
            label, prob = self._model_provider.get_window_summary()
            return str(label), float(prob)
        except Exception:
            return "None", 1.0

    def log(
        self,
        world: carla.World,
        vehicle: carla.Actor,
        error_type: str,
        details: str = "",
    ) -> None:
        """Append a single error row to the dataset."""
        try:
            location = vehicle.get_location()
        except Exception:
            return

        pred_label, pred_prob = self._model_snapshot()

        row: Dict[str, Any] = {
            "user_id": self._context.user_id,
            "run_id": self._context.run_id,
            "weather": self._context.weather_label,
            "map_name": self._context.map_name,
            "error_type": error_type,
            "model_pred_start": pred_label,
            "model_prob_start": round(pred_prob, 3),
            "model_pred_end": pred_label,
            "model_prob_end": round(pred_prob, 3),
            "speed_kmh": round(_speed_kmh(vehicle), 3),
            "timestamp": _wall_time_iso(),
            "details": details,
        }

        row.update(_snapshot_info(world))
        row.update(_location_info(world, location))
        self._writer.append(row)


class DistractionDatasetLogger:
    """Logger for distraction windows into Dataset Distractions CSV."""

    def __init__(
        self,
        output_dir: str,
        context: DatasetContext,
        suffix: str = "",
        model_provider: Optional[ModelInferenceProvider] = None,
        arousal_provider: Optional[ArousalProvider] = None,
    ) -> None:
        """Create a logger with a destination folder and context."""
        path = os.path.join(output_dir, f"Dataset Distractions{suffix}.csv")
        self._context = context
        self._model_provider = model_provider
        self._arousal_provider = arousal_provider
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

    def _model_snapshot(self) -> Tuple[str, float]:
        """Return the latest aggregated model label and probability."""
        if self._model_provider is None:
            return "None", 1.0
        try:
            label, prob = self._model_provider.get_window_summary()
            return str(label), float(prob)
        except Exception:
            return "None", 1.0

    def _arousal_snapshot(self) -> ArousalSnapshot:
        """Return the latest arousal snapshot."""
        if self._arousal_provider is None:
            return ArousalSnapshot(None, None, None, None)
        try:
            return self._arousal_provider.get_snapshot()
        except Exception:
            return ArousalSnapshot(None, None, None, None)

    @staticmethod
    def _format_arousal(value: Optional[float]) -> Any:
        if value is None:
            return ""
        try:
            return round(float(value), 3)
        except Exception:
            return ""

    def start(self, window_id: str, world: carla.World, vehicle: carla.Actor) -> None:
        """Record the start of a distraction window."""
        with self._lock:
            if window_id in self._active:
                return
            try:
                location = vehicle.get_location()
            except Exception:
                return
            snap = _snapshot_info(world)
            pred_label, pred_prob = self._model_snapshot()
            arousal = self._arousal_snapshot()
            self._active[window_id] = {
                "start_location": location,
                "frame_start": snap.get("frame", ""),
                "sim_time_start": snap.get("sim_time_seconds", ""),
                "timestamp_start": _wall_time_iso(),
                "model_pred_start": pred_label,
                "model_prob_start": round(pred_prob, 3),
                "arousal_start": self._format_arousal(arousal.value),
            }

    def finish(self, window_id: str, world: carla.World, vehicle: carla.Actor) -> None:
        """Record the end of a distraction window."""
        with self._lock:
            start_info = self._active.pop(window_id, None)
        if start_info is None:
            return
        try:
            end_location = vehicle.get_location()
        except Exception:
            return
        snap = _snapshot_info(world)
        end_pred_label, end_pred_prob = self._model_snapshot()
        end_arousal = self._arousal_snapshot()

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
            "arousal_start": start_info.get("arousal_start", ""),
            "arousal_end": self._format_arousal(end_arousal.value),
            "model_pred_start": start_info.get("model_pred_start", ""),
            "model_prob_start": start_info.get("model_prob_start", ""),
            "model_pred_end": end_pred_label,
            "model_prob_end": round(end_pred_prob, 3),
            "timestamp_start": start_info.get("timestamp_start", ""),
            "timestamp_end": _wall_time_iso(),
            "frame_start": start_info.get("frame_start", ""),
            "frame_end": snap.get("frame", ""),
            "sim_time_start": start_info.get("sim_time_start", ""),
            "sim_time_end": snap.get("sim_time_seconds", ""),
            "details": window_id,
        }
        self._writer.append(row)
