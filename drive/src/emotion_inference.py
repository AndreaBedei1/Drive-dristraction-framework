from __future__ import annotations

"""Asynchronous webcam emotion inference using DeepFace."""

import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from src.emotion_provider import EmotionProvider, EmotionSnapshot


DEFAULT_TARGET_HZ = 1.0
DEFAULT_DETECTOR_BACKEND = os.environ.get("EMOTION_DETECTOR_BACKEND", "opencv")
DEFAULT_ENFORCE_DETECTION = os.environ.get("EMOTION_ENFORCE_DETECTION", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
NONE_LABEL = "None"


class FrameProvider(Protocol):
    """Protocol for classes that can return the latest RGB frame."""

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Return the latest RGB frame."""
        ...


def _find_local_deepface_path() -> Optional[str]:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidate = os.path.join(repo_root, "emotion", "deepface")
    if os.path.isdir(os.path.join(candidate, "deepface")):
        return candidate
    return None


def _import_deepface():
    try:
        from deepface import DeepFace  # type: ignore

        return DeepFace
    except Exception:
        local_path = _find_local_deepface_path()
        if local_path and local_path not in sys.path:
            sys.path.insert(0, local_path)
        from deepface import DeepFace  # type: ignore

        return DeepFace


def _log_runtime_env(detector_backend: str) -> None:
    print(f"[Emotion] detector_backend={detector_backend}")
    try:
        import tensorflow as tf  # type: ignore

        gpus = tf.config.list_physical_devices("GPU")
        print(f"[Emotion] tensorflow={tf.__version__} gpus={len(gpus)} {gpus}")
    except Exception as exc:
        print(f"[Emotion] tensorflow_unavailable: {exc}")

    try:
        import torch  # type: ignore

        print(f"[Emotion] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[Emotion] torch_cuda_device={torch.cuda.get_device_name(0)}")
    except Exception as exc:
        print(f"[Emotion] torch_unavailable: {exc}")


def _normalize_results(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, list):
        if raw and isinstance(raw[0], list):
            raw = raw[0]
        return [item for item in raw if isinstance(item, dict)]
    return []


def _face_score(item: Dict[str, Any]) -> Tuple[float, float]:
    conf = item.get("face_confidence")
    try:
        conf_val = float(conf)
    except Exception:
        conf_val = 0.0

    region = item.get("region") or item.get("facial_area") or {}
    try:
        w = float(region.get("w", 0.0))
        h = float(region.get("h", 0.0))
    except Exception:
        w, h = 0.0, 0.0
    area = max(0.0, w) * max(0.0, h)
    return conf_val, area


def _select_best_result(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not results:
        return None
    return max(results, key=_face_score)


def _extract_emotion(result: Optional[Dict[str, Any]]) -> Tuple[str, Optional[float]]:
    if not result:
        return NONE_LABEL, None

    dominant = result.get("dominant_emotion")
    emotions = result.get("emotion")

    if not dominant and isinstance(emotions, dict) and emotions:
        try:
            dominant = max(emotions.items(), key=lambda kv: kv[1])[0]
        except Exception:
            dominant = None

    prob: Optional[float] = None
    if dominant and isinstance(emotions, dict):
        try:
            score = emotions.get(dominant)
            if score is not None:
                score_val = float(score)
                prob = score_val / 100.0 if score_val > 1.0 else score_val
        except Exception:
            prob = None

    label = str(dominant).strip() if dominant else NONE_LABEL
    if not label:
        label = NONE_LABEL
    return label, prob


class EmotionInferenceService(threading.Thread, EmotionProvider):
    """Run DeepFace emotion inference asynchronously on webcam frames."""

    def __init__(
        self,
        frame_provider: FrameProvider,
        target_hz: float = DEFAULT_TARGET_HZ,
        detector_backend: Optional[str] = None,
        enforce_detection: Optional[bool] = None,
    ) -> None:
        super().__init__(daemon=True)
        self._frame_provider = frame_provider
        self._target_hz = float(target_hz)
        self._detector_backend = detector_backend or DEFAULT_DETECTOR_BACKEND
        self._enforce_detection = DEFAULT_ENFORCE_DETECTION if enforce_detection is None else bool(enforce_detection)

        self._lock = threading.Lock()
        self._snapshot = EmotionSnapshot(None, None, None)
        self._stop_event = threading.Event()
        self._last_error: Optional[str] = None

    def stop(self) -> None:
        """Signal the worker thread to stop."""
        self._stop_event.set()

    def last_error(self) -> Optional[str]:
        """Return the last error message, if any."""
        return self._last_error

    def get_snapshot(self) -> EmotionSnapshot:
        """Return the latest emotion snapshot."""
        with self._lock:
            return self._snapshot

    def run(self) -> None:
        """Thread loop: read frames and update emotion snapshot."""
        try:
            DeepFace = _import_deepface()
        except Exception as exc:
            self._last_error = f"deepface_import_failed: {exc}"
            return
        _log_runtime_env(self._detector_backend)

        infer_period = 1.0 / max(0.1, self._target_hz)
        last_infer_t = 0.0

        while not self._stop_event.is_set():
            now = time.monotonic()
            elapsed_since_last = now - last_infer_t
            if elapsed_since_last < infer_period:
                sleep_s = min(0.05, infer_period - elapsed_since_last)
                if sleep_s > 0:
                    time.sleep(sleep_s)
                continue

            frame = None
            try:
                frame = self._frame_provider.get_latest_frame()
            except Exception:
                frame = None

            if frame is None:
                time.sleep(0.05)
                continue

            last_infer_t = now

            try:
                frame_bgr = np.ascontiguousarray(frame[:, :, ::-1])
            except Exception:
                continue

            try:
                raw = DeepFace.analyze(
                    img_path=frame_bgr,
                    actions=("emotion",),
                    enforce_detection=self._enforce_detection,
                    detector_backend=self._detector_backend,
                    align=True,
                    silent=True,
                )
                results = _normalize_results(raw)
                best = _select_best_result(results)
                label, prob = _extract_emotion(best)
                with self._lock:
                    self._snapshot = EmotionSnapshot(label=label, prob=prob, timestamp=now)
                self._last_error = None
            except Exception as exc:
                self._last_error = str(exc)
