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
from src.emotion_provider import EmotionProvider, EmotionSnapshot

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
        self._fieldnames = self._merge_fieldnames([], list(fieldnames))
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path)
        if not write_header:
            try:
                write_header = os.path.getsize(path) == 0
            except Exception:
                write_header = False
        if write_header:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames)
                writer.writeheader()
            return

        existing_fieldnames = self._read_existing_fieldnames(path)
        if not existing_fieldnames:
            return
        merged_fieldnames = self._merge_fieldnames(existing_fieldnames, self._fieldnames)
        self._fieldnames = merged_fieldnames
        if merged_fieldnames != existing_fieldnames:
            try:
                self._rewrite_with_fieldnames(merged_fieldnames)
            except Exception:
                self._fieldnames = existing_fieldnames

    def append(self, row: Dict[str, Any]) -> None:
        """Append a row to the CSV file."""
        with self._lock, open(self._path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
            writer.writerow(row)

    @staticmethod
    def _read_existing_fieldnames(path: str) -> List[str]:
        """Read the first non-empty CSV row as fieldnames."""
        try:
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    cleaned = _CsvWriter._merge_fieldnames([], [str(col) for col in row])
                    if cleaned:
                        return cleaned
        except Exception:
            return []
        return []

    @staticmethod
    def _merge_fieldnames(existing: List[str], required: List[str]) -> List[str]:
        """Keep existing columns and append missing required columns."""
        merged: List[str] = []
        seen = set()
        for raw in list(existing) + list(required):
            name = str(raw).strip()
            if not name or name in seen:
                continue
            merged.append(name)
            seen.add(name)
        return merged

    @staticmethod
    def _normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize row keys by trimming whitespace and preferring non-empty values."""
        normalized: Dict[str, Any] = {}
        for raw_key, value in row.items():
            key = str(raw_key).strip()
            if not key:
                continue
            current = normalized.get(key, "")
            if current == "" and value not in ("", None):
                normalized[key] = value
            elif key not in normalized:
                normalized[key] = value
        return normalized

    def _rewrite_with_fieldnames(self, fieldnames: List[str]) -> None:
        """Rewrite an existing CSV file using an expanded header."""
        tmp_path = f"{self._path}.tmp"
        try:
            with open(self._path, "r", newline="", encoding="utf-8") as src, open(
                tmp_path, "w", newline="", encoding="utf-8"
            ) as dst:
                reader = csv.DictReader(src)
                writer = csv.DictWriter(dst, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in reader:
                    writer.writerow(self._normalize_row(row))
            os.replace(tmp_path, self._path)
        except Exception:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            raise


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


def _steer_angle_deg(vehicle: carla.Actor) -> float:
    """Estimate current steering angle in degrees from vehicle control."""
    try:
        steer_norm = float(vehicle.get_control().steer)
    except Exception:
        return 0.0

    max_steer_deg = 0.0
    try:
        physics = vehicle.get_physics_control()
        wheel_angles: List[float] = []
        for wheel in getattr(physics, "wheels", []):
            try:
                angle = abs(float(getattr(wheel, "max_steer_angle", 0.0)))
                if angle > 0.0:
                    wheel_angles.append(angle)
            except Exception:
                continue
        if wheel_angles:
            max_steer_deg = max(wheel_angles)
    except Exception:
        max_steer_deg = 0.0

    return steer_norm * max_steer_deg


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


def _vehicle_control_info(vehicle: carla.Actor) -> Dict[str, Any]:
    """Return useful low-level control state for the vehicle."""
    data: Dict[str, Any] = {
        "throttle": "",
        "brake": "",
        "hand_brake": "",
        "reverse": "",
        "gear": "",
    }
    try:
        control = vehicle.get_control()
    except Exception:
        return data

    try:
        data["throttle"] = round(float(control.throttle), 3)
    except Exception:
        pass
    try:
        data["brake"] = round(float(control.brake), 3)
    except Exception:
        pass
    try:
        data["hand_brake"] = int(bool(control.hand_brake))
    except Exception:
        pass
    try:
        data["reverse"] = int(bool(control.reverse))
    except Exception:
        pass
    try:
        data["gear"] = int(control.gear)
    except Exception:
        pass
    return data


def _wall_time_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.datetime.utcnow().isoformat()


def _format_emotion_label(value: Optional[str]) -> str:
    if value is None:
        return "None"
    try:
        text = str(value).strip()
    except Exception:
        return "None"
    return text or "None"


def _format_emotion_prob(value: Optional[float]) -> Any:
    if value is None:
        return ""
    try:
        return round(float(value), 3)
    except Exception:
        return ""


class BaselineDrivingTimeLogger:
    """Logger for baseline driving time and cumulative user totals."""

    def __init__(
        self,
        output_dir: str,
        context: DatasetContext,
        suffix: str = "",
    ) -> None:
        """Create a logger for baseline driving time rows."""
        self._context = context
        self._path = os.path.join(output_dir, f"Dataset Driving Time{suffix}.csv")
        self._writer = _CsvWriter(
            self._path,
            [
                "user_id",
                "run_id",
                "weather",
                "map_name",
                "run_duration_seconds",
                "run_duration_minutes",
                "total_duration_seconds",
                "total_duration_minutes",
                "timestamp",
                "hr_baseline",
                "arousal_baseline",
            ],
        )
        self._lock = threading.Lock()

    def _existing_total_for_user_seconds(self, user_id: str) -> float:
        """Return cumulative driving time already stored for the user."""
        total = 0.0
        if not os.path.exists(self._path):
            return total
        try:
            with open(self._path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if str(row.get("user_id", "")).strip() != user_id:
                        continue
                    try:
                        total += float(row.get("run_duration_seconds", 0.0))
                    except Exception:
                        continue
        except Exception:
            return 0.0
        return total

    @staticmethod
    def _format_arousal(value: Optional[float]) -> Any:
        if value is None:
            return ""
        try:
            arousal = float(value)
        except Exception:
            return ""
        if arousal < 0.0 or arousal > 1.0:
            return ""
        return round(arousal, 3)

    @staticmethod
    def _format_hr(value: Optional[int]) -> Any:
        if value is None:
            return ""
        try:
            hr = int(value)
        except Exception:
            return ""
        if hr < 35 or hr > 220:
            return ""
        return hr

    def _resolve_baseline_metrics(
        self,
        pre_drive_snapshot: Optional[ArousalSnapshot] = None,
    ) -> Tuple[Any, Any]:
        """Resolve baseline HR/arousal from a required pre-drive snapshot."""
        if pre_drive_snapshot is None:
            raise RuntimeError("Missing pre-drive arousal snapshot for baseline run.")

        hr_value = self._format_hr(pre_drive_snapshot.hr_bpm)
        arousal_value = self._format_arousal(pre_drive_snapshot.value)

        if hr_value == "":
            raise RuntimeError("Invalid baseline heart-rate sample from arousal sensor.")
        if arousal_value == "":
            raise RuntimeError("Invalid baseline arousal sample from arousal sensor.")

        return hr_value, arousal_value

    def log_run_duration(
        self,
        run_duration_seconds: float,
        pre_drive_snapshot: Optional[ArousalSnapshot] = None,
    ) -> float:
        """Append run duration and return updated cumulative user total."""
        run_seconds = max(0.0, float(run_duration_seconds))
        with self._lock:
            total_seconds = self._existing_total_for_user_seconds(self._context.user_id) + run_seconds
            hr_baseline, arousal_baseline = self._resolve_baseline_metrics(
                pre_drive_snapshot=pre_drive_snapshot
            )
            row: Dict[str, Any] = {
                "user_id": self._context.user_id,
                "run_id": self._context.run_id,
                "weather": self._context.weather_label,
                "map_name": self._context.map_name,
                "run_duration_seconds": round(run_seconds, 3),
                "run_duration_minutes": round(run_seconds / 60.0, 3),
                "total_duration_seconds": round(total_seconds, 3),
                "total_duration_minutes": round(total_seconds / 60.0, 3),
                "timestamp": _wall_time_iso(),
                "hr_baseline": hr_baseline,
                "arousal_baseline": arousal_baseline,
            }
            self._writer.append(row)
            return total_seconds


class TimelineDatasetLogger:
    """Second-by-second timeline logger for the whole simulation."""

    def __init__(
        self,
        output_dir: str,
        context: DatasetContext,
        suffix: str = "",
        model_provider: Optional[ModelInferenceProvider] = None,
        arousal_provider: Optional[ArousalProvider] = None,
        emotion_provider: Optional[EmotionProvider] = None,
    ) -> None:
        self._context = context
        self._model_provider = model_provider
        self._arousal_provider = arousal_provider
        self._emotion_provider = emotion_provider
        path = os.path.join(output_dir, f"Dataset Timeline{suffix}.csv")
        self._writer = _CsvWriter(
            path,
            [
                "user_id",
                "run_id",
                "weather",
                "map_name",
                "second_index",
                "sim_second_bucket",
                "second_complete",
                "timestamp",
                "frame",
                "sim_time_seconds",
                "x",
                "y",
                "z",
                "road_id",
                "lane_id",
                "speed_kmh",
                "steer_angle_deg",
                "throttle",
                "brake",
                "hand_brake",
                "reverse",
                "gear",
                "model_pred",
                "model_prob",
                "emotion_label",
                "emotion_prob",
                "emotion_timestamp",
                "arousal",
                "arousal_method",
                "arousal_quality",
                "arousal_timestamp_ms",
                "hr_bpm",
                "distraction_active",
                "active_distraction_count",
                "active_distraction_ids",
                "distraction_start_count",
                "distraction_start_ids",
                "distraction_finish_count",
                "distraction_finish_ids",
                "last_distraction_started_id",
                "last_distraction_finished_id",
                "last_distraction_duration_seconds",
                "seconds_since_last_distraction_end",
                "total_distraction_starts",
                "total_distraction_finishes",
                "error_occurred",
                "error_count",
                "error_types",
                "error_details",
                "last_error_type",
                "total_errors",
                "details",
            ],
        )
        self._lock = threading.Lock()
        self._first_bucket: Optional[int] = None
        self._current_bucket: Optional[int] = None
        self._current_state: Optional[Dict[str, Any]] = None
        self._error_events_by_bucket: Dict[int, List[Tuple[str, str]]] = {}
        self._error_history: List[Tuple[int, str, str]] = []
        self._distraction_starts_by_bucket: Dict[int, List[str]] = {}
        self._distraction_finishes_by_bucket: Dict[int, List[Tuple[str, Any]]] = {}
        self._distraction_start_history: List[Tuple[float, str]] = []
        self._distraction_finish_history: List[Tuple[float, str, Any]] = []
        self._active_distractions: Dict[str, float] = {}

    def _model_snapshot(self) -> Tuple[str, float]:
        if self._model_provider is None:
            return "None", 1.0
        try:
            label, prob = self._model_provider.get_window_summary()
            return str(label), float(prob)
        except Exception:
            return "None", 1.0

    def _arousal_snapshot(self) -> ArousalSnapshot:
        if self._arousal_provider is None:
            return ArousalSnapshot(None, None, None, None, None)
        try:
            return self._arousal_provider.get_snapshot()
        except Exception:
            return ArousalSnapshot(None, None, None, None, None)

    def _emotion_snapshot(self) -> EmotionSnapshot:
        if self._emotion_provider is None:
            return EmotionSnapshot(None, None, None)
        try:
            return self._emotion_provider.get_snapshot()
        except Exception:
            return EmotionSnapshot(None, None, None)

    @staticmethod
    def _format_arousal(value: Optional[float]) -> Any:
        if value is None:
            return ""
        try:
            return round(float(value), 3)
        except Exception:
            return ""

    @staticmethod
    def _format_hr(value: Optional[int]) -> Any:
        if value is None:
            return ""
        try:
            return int(value)
        except Exception:
            return ""

    @staticmethod
    def _format_timestamp(value: Optional[float]) -> Any:
        if value is None:
            return ""
        try:
            return round(float(value), 3)
        except Exception:
            return ""

    @staticmethod
    def _coerce_sim_time(value: Any) -> Optional[float]:
        try:
            sim_time = float(value)
        except Exception:
            return None
        if sim_time < 0.0:
            return None
        return sim_time

    @staticmethod
    def _bucket_for_sim_time(value: Any) -> Optional[int]:
        sim_time = TimelineDatasetLogger._coerce_sim_time(value)
        if sim_time is None:
            return None
        return int(sim_time)

    @staticmethod
    def _join_values(values: List[str]) -> str:
        cleaned = [str(v).strip() for v in values if str(v).strip()]
        return " | ".join(cleaned)

    def _capture_state(self, world: carla.World, vehicle: carla.Actor) -> Optional[Tuple[int, Dict[str, Any]]]:
        try:
            location = vehicle.get_location()
        except Exception:
            return None

        snap = _snapshot_info(world)
        sim_time = self._coerce_sim_time(snap.get("sim_time_seconds"))
        if sim_time is None:
            return None
        bucket = int(sim_time)
        model_pred, model_prob = self._model_snapshot()
        emotion = self._emotion_snapshot()
        arousal = self._arousal_snapshot()

        row: Dict[str, Any] = {
            "user_id": self._context.user_id,
            "run_id": self._context.run_id,
            "weather": self._context.weather_label,
            "map_name": self._context.map_name,
            "timestamp": _wall_time_iso(),
            "speed_kmh": round(_speed_kmh(vehicle), 3),
            "steer_angle_deg": round(_steer_angle_deg(vehicle), 3),
            "model_pred": model_pred,
            "model_prob": round(model_prob, 3),
            "emotion_label": _format_emotion_label(emotion.label),
            "emotion_prob": _format_emotion_prob(emotion.prob),
            "emotion_timestamp": self._format_timestamp(emotion.timestamp),
            "arousal": self._format_arousal(arousal.value),
            "arousal_method": str(arousal.method).strip() if arousal.method else "",
            "arousal_quality": str(arousal.quality).strip() if arousal.quality else "",
            "arousal_timestamp_ms": arousal.timestamp_ms if arousal.timestamp_ms is not None else "",
            "hr_bpm": self._format_hr(arousal.hr_bpm),
        }
        row.update(snap)
        row.update(_location_info(world, location))
        row.update(_vehicle_control_info(vehicle))
        return bucket, row

    def update(self, world: carla.World, vehicle: carla.Actor) -> None:
        captured = self._capture_state(world, vehicle)
        if captured is None:
            return
        bucket, state = captured

        with self._lock:
            if self._first_bucket is None:
                self._first_bucket = bucket
            if self._current_bucket is None:
                self._current_bucket = bucket
                self._current_state = state
                return
            if bucket != self._current_bucket:
                self._flush_locked(second_complete=True)
                self._current_bucket = bucket
            self._current_state = state

    def record_error(self, error_type: str, details: str = "", sim_time_seconds: Any = None) -> None:
        bucket = self._bucket_for_sim_time(sim_time_seconds)
        if bucket is None:
            return
        with self._lock:
            self._error_history.append((bucket, error_type, details))
            self._error_events_by_bucket.setdefault(bucket, []).append((error_type, details))

    def record_distraction_start(self, window_id: str, sim_time_seconds: Any = None) -> None:
        sim_time = self._coerce_sim_time(sim_time_seconds)
        if sim_time is None:
            return
        bucket = int(sim_time)
        with self._lock:
            self._active_distractions[window_id] = sim_time
            self._distraction_start_history.append((sim_time, window_id))
            self._distraction_starts_by_bucket.setdefault(bucket, []).append(window_id)

    def record_distraction_finish(self, window_id: str, sim_time_seconds: Any = None) -> None:
        sim_time = self._coerce_sim_time(sim_time_seconds)
        if sim_time is None:
            return
        bucket = int(sim_time)
        with self._lock:
            start_time = self._active_distractions.pop(window_id, None)
            duration: Any = ""
            if start_time is not None:
                duration = round(max(0.0, sim_time - start_time), 3)
            self._distraction_finish_history.append((sim_time, window_id, duration))
            self._distraction_finishes_by_bucket.setdefault(bucket, []).append((window_id, duration))

    def flush_pending(self) -> None:
        with self._lock:
            self._flush_locked(second_complete=False)

    def _flush_locked(self, second_complete: bool) -> None:
        if self._current_bucket is None or self._current_state is None:
            return

        bucket = self._current_bucket
        row = dict(self._current_state)
        bucket_end = float(bucket + 1)

        bucket_errors = self._error_events_by_bucket.pop(bucket, [])
        bucket_starts = self._distraction_starts_by_bucket.pop(bucket, [])
        bucket_finishes = self._distraction_finishes_by_bucket.pop(bucket, [])

        error_types = [item[0] for item in bucket_errors]
        error_details = [item[1] for item in bucket_errors if str(item[1]).strip()]
        finish_ids = [item[0] for item in bucket_finishes]
        finish_durations = [item[1] for item in bucket_finishes if item[1] not in ("", None)]

        latest_finish = None
        for finish_time, finish_id, duration in self._distraction_finish_history:
            if finish_time < bucket_end:
                latest_finish = (finish_time, finish_id, duration)
            else:
                break

        active_ids = sorted(
            window_id
            for window_id, start_time in self._active_distractions.items()
            if start_time < bucket_end
        )

        row.update(
            {
                "second_index": (bucket - self._first_bucket) if self._first_bucket is not None else 0,
                "sim_second_bucket": bucket,
                "second_complete": int(bool(second_complete)),
                "distraction_active": int(bool(active_ids)),
                "active_distraction_count": len(active_ids),
                "active_distraction_ids": self._join_values(active_ids),
                "distraction_start_count": len(bucket_starts),
                "distraction_start_ids": self._join_values(bucket_starts),
                "distraction_finish_count": len(bucket_finishes),
                "distraction_finish_ids": self._join_values(finish_ids),
                "last_distraction_started_id": bucket_starts[-1] if bucket_starts else "",
                "last_distraction_finished_id": finish_ids[-1] if finish_ids else "",
                "last_distraction_duration_seconds": finish_durations[-1] if finish_durations else "",
                "seconds_since_last_distraction_end": ""
                if latest_finish is None
                else round(max(0.0, bucket_end - float(latest_finish[0])), 3),
                "total_distraction_starts": sum(1 for start_time, _ in self._distraction_start_history if start_time < bucket_end),
                "total_distraction_finishes": sum(1 for finish_time, _, _ in self._distraction_finish_history if finish_time < bucket_end),
                "error_occurred": int(bool(bucket_errors)),
                "error_count": len(bucket_errors),
                "error_types": self._join_values(error_types),
                "error_details": self._join_values(error_details),
                "last_error_type": error_types[-1] if error_types else "",
                "total_errors": sum(1 for error_bucket, _, _ in self._error_history if error_bucket <= bucket),
            }
        )

        summary_parts: List[str] = []
        if bucket_starts:
            summary_parts.append(f"distraction_start={self._join_values(bucket_starts)}")
        if bucket_finishes:
            summary_parts.append(f"distraction_finish={self._join_values(finish_ids)}")
        if bucket_errors:
            summary_parts.append(f"errors={self._join_values(error_types)}")
        row["details"] = "; ".join(summary_parts)

        self._writer.append(row)
        self._current_state = None


class ErrorDatasetLogger:
    """Logger for error events into Dataset Errors CSV."""

    def __init__(
        self,
        output_dir: str,
        context: DatasetContext,
        suffix: str = "",
        model_provider: Optional[ModelInferenceProvider] = None,
        emotion_provider: Optional[EmotionProvider] = None,
        timeline_logger: Optional[TimelineDatasetLogger] = None,
    ) -> None:
        """Create a logger with a destination folder and context."""
        path = os.path.join(output_dir, f"Dataset Errors{suffix}.csv")
        self._context = context
        self._model_provider = model_provider
        self._emotion_provider = emotion_provider
        self._timeline_logger = timeline_logger
        self._writer = _CsvWriter(
            path,
            [
                "user_id",
                "run_id",
                "weather",
                "map_name",
                "error_type",
                "model_pred",
                "model_prob",
                "emotion_label",
                "emotion_prob",
                "speed_kmh",
                "steer_angle_deg",
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

    def _emotion_snapshot(self) -> EmotionSnapshot:
        """Return the latest emotion snapshot."""
        if self._emotion_provider is None:
            return EmotionSnapshot(None, None, None)
        try:
            return self._emotion_provider.get_snapshot()
        except Exception:
            return EmotionSnapshot(None, None, None)

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
        emotion = self._emotion_snapshot()

        row: Dict[str, Any] = {
            "user_id": self._context.user_id,
            "run_id": self._context.run_id,
            "weather": self._context.weather_label,
            "map_name": self._context.map_name,
            "error_type": error_type,
            "model_pred": pred_label,
            "model_prob": round(pred_prob, 3),
            "emotion_label": _format_emotion_label(emotion.label),
            "emotion_prob": _format_emotion_prob(emotion.prob),
            "speed_kmh": round(_speed_kmh(vehicle), 3),
            "steer_angle_deg": round(_steer_angle_deg(vehicle), 3),
            "timestamp": _wall_time_iso(),
            "details": details,
        }

        snap = _snapshot_info(world)
        row.update(snap)
        row.update(_location_info(world, location))
        self._writer.append(row)
        if self._timeline_logger is not None:
            self._timeline_logger.record_error(
                error_type=error_type,
                details=details,
                sim_time_seconds=snap.get("sim_time_seconds"),
            )


class DistractionDatasetLogger:
    """Logger for distraction windows into Dataset Distractions CSV."""

    def __init__(
        self,
        output_dir: str,
        context: DatasetContext,
        suffix: str = "",
        model_provider: Optional[ModelInferenceProvider] = None,
        arousal_provider: Optional[ArousalProvider] = None,
        emotion_provider: Optional[EmotionProvider] = None,
        timeline_logger: Optional[TimelineDatasetLogger] = None,
    ) -> None:
        """Create a logger with a destination folder and context."""
        path = os.path.join(output_dir, f"Dataset Distractions{suffix}.csv")
        self._context = context
        self._model_provider = model_provider
        self._arousal_provider = arousal_provider
        self._emotion_provider = emotion_provider
        self._timeline_logger = timeline_logger
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
                "speed_kmh_start",
                "speed_kmh_end",
                "steer_angle_deg_start",
                "steer_angle_deg_end",
                "arousal_start",
                "arousal_end",
                "hr_bpm_start",
                "hr_bpm_end",
                "model_pred_start",
                "model_prob_start",
                "model_pred_end",
                "model_prob_end",
                "emotion_label_start",
                "emotion_prob_start",
                "emotion_label_end",
                "emotion_prob_end",
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
            return ArousalSnapshot(None, None, None, None, None)
        try:
            return self._arousal_provider.get_snapshot()
        except Exception:
            return ArousalSnapshot(None, None, None, None, None)

    def _emotion_snapshot(self) -> EmotionSnapshot:
        """Return the latest emotion snapshot."""
        if self._emotion_provider is None:
            return EmotionSnapshot(None, None, None)
        try:
            return self._emotion_provider.get_snapshot()
        except Exception:
            return EmotionSnapshot(None, None, None)

    @staticmethod
    def _format_arousal(value: Optional[float]) -> Any:
        if value is None:
            return ""
        try:
            return round(float(value), 3)
        except Exception:
            return ""

    @staticmethod
    def _format_hr(value: Optional[int]) -> Any:
        if value is None:
            return ""
        try:
            return int(value)
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
            emotion = self._emotion_snapshot()
            self._active[window_id] = {
                "start_location": location,
                "frame_start": snap.get("frame", ""),
                "sim_time_start": snap.get("sim_time_seconds", ""),
                "timestamp_start": _wall_time_iso(),
                "model_pred_start": pred_label,
                "model_prob_start": round(pred_prob, 3),
                "speed_kmh_start": round(_speed_kmh(vehicle), 3),
                "steer_angle_deg_start": round(_steer_angle_deg(vehicle), 3),
                "arousal_start": self._format_arousal(arousal.value),
                "hr_bpm_start": self._format_hr(arousal.hr_bpm),
                "emotion_label_start": _format_emotion_label(emotion.label),
                "emotion_prob_start": _format_emotion_prob(emotion.prob),
            }
            if self._timeline_logger is not None:
                self._timeline_logger.record_distraction_start(
                    window_id=window_id,
                    sim_time_seconds=snap.get("sim_time_seconds"),
                )

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
        end_emotion = self._emotion_snapshot()

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
            "speed_kmh_start": start_info.get("speed_kmh_start", ""),
            "speed_kmh_end": round(_speed_kmh(vehicle), 3),
            "steer_angle_deg_start": start_info.get("steer_angle_deg_start", ""),
            "steer_angle_deg_end": round(_steer_angle_deg(vehicle), 3),
            "arousal_start": start_info.get("arousal_start", ""),
            "arousal_end": self._format_arousal(end_arousal.value),
            "hr_bpm_start": start_info.get("hr_bpm_start", ""),
            "hr_bpm_end": self._format_hr(end_arousal.hr_bpm),
            "model_pred_start": start_info.get("model_pred_start", ""),
            "model_prob_start": start_info.get("model_prob_start", ""),
            "model_pred_end": end_pred_label,
            "model_prob_end": round(end_pred_prob, 3),
            "emotion_label_start": start_info.get("emotion_label_start", ""),
            "emotion_prob_start": start_info.get("emotion_prob_start", ""),
            "emotion_label_end": _format_emotion_label(end_emotion.label),
            "emotion_prob_end": _format_emotion_prob(end_emotion.prob),
            "timestamp_start": start_info.get("timestamp_start", ""),
            "timestamp_end": _wall_time_iso(),
            "frame_start": start_info.get("frame_start", ""),
            "frame_end": snap.get("frame", ""),
            "sim_time_start": start_info.get("sim_time_start", ""),
            "sim_time_end": snap.get("sim_time_seconds", ""),
            "details": window_id,
        }
        self._writer.append(row)
        if self._timeline_logger is not None:
            self._timeline_logger.record_distraction_finish(
                window_id=window_id,
                sim_time_seconds=snap.get("sim_time_seconds"),
            )
