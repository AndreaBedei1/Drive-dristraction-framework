from __future__ import annotations

"""Asynchronous webcam inference service for driver distraction labels."""

import os
import time
import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import joblib
import clip
from PIL import Image
import pyrealsense2 as rs


CLASS_NAMES: List[str] = [
    "nothing",      # 0
    "drink",        # 1
    "eat",          # 2
    "pick_floor",   # 3
    "reach_back",   # 4
    "sing",         # 5
    "phone_call",   # 6
    "yawn",         # 7 (mapped to None)
]

NONE_LABEL = "None"
RAW_THRESHOLD = 0.80
TARGET_INFER_HZ = 2.0
WINDOW_SIZE = 3

USE_ROI = False
ROI_REL = (0.0, 0.0, 1.0, 1.0)


@dataclass(frozen=True)
class InferenceResult:
    label: str
    prob: float
    timestamp: float


def _default_classifier_path() -> str:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(
        repo_root,
        "distraction",
        "model_ckpt",
        "frontcam_vitl14-per1_hypc_429_1000_ft.pkl",
    )


def _maybe_crop_roi(frame: np.ndarray) -> np.ndarray:
    if not USE_ROI:
        return frame

    h, w = frame.shape[:2]
    x0r, y0r, x1r, y1r = ROI_REL
    x0 = int(max(0, min(w - 1, x0r * w)))
    y0 = int(max(0, min(h - 1, y0r * h)))
    x1 = int(max(1, min(w, x1r * w)))
    y1 = int(max(1, min(h, y1r * h)))

    if x1 <= x0 or y1 <= y0:
        return frame

    return frame[y0:y1, x0:x1]


@torch.no_grad()
def _infer_probs(model, classifier, frame_rgb: np.ndarray, preprocess, device: str) -> np.ndarray:
    img = Image.fromarray(frame_rgb).convert("RGB")
    x = preprocess(img).unsqueeze(0).to(device)

    img_feat = model.encode_image(x)
    feat_np = img_feat.detach().cpu().numpy()
    probs = classifier.predict_proba(feat_np)[0].astype(float)
    return probs


def _postprocess_probs(probs: np.ndarray, threshold: float) -> Tuple[str, float]:
    if len(probs) > 7:
        probs = probs.copy()
        probs[0] = float(probs[0]) + float(probs[7])
        probs[7] = 0.0

    pred_id = int(np.argmax(probs))
    pred_score = float(probs[pred_id])

    if pred_score < threshold:
        return NONE_LABEL, 1.0

    if pred_id in (0, 7):
        return NONE_LABEL, pred_score

    label = CLASS_NAMES[pred_id] if pred_id < len(CLASS_NAMES) else f"class_{pred_id}"
    return label, pred_score


class ModelInferenceService(threading.Thread):
    """Run CLIP+classifier inference asynchronously and store a rolling window."""

    def __init__(
        self,
        classifier_path: Optional[str] = None,
        device: Optional[str] = None,
        target_hz: float = TARGET_INFER_HZ,
        window_size: int = WINDOW_SIZE,
        threshold: float = RAW_THRESHOLD,
    ) -> None:
        super().__init__(daemon=True)
        self._classifier_path = classifier_path or _default_classifier_path()
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._target_hz = float(target_hz)
        self._window_size = int(window_size)
        self._threshold = float(threshold)

        self._history: Deque[InferenceResult] = deque(maxlen=self._window_size)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._last_error: Optional[str] = None

    def stop(self) -> None:
        """Signal the worker thread to stop."""
        self._stop_event.set()

    def last_error(self) -> Optional[str]:
        """Return the last error message, if any."""
        return self._last_error

    def get_window_summary(self) -> Tuple[str, float]:
        """Return majority label and mean probability over the rolling window."""
        with self._lock:
            if not self._history:
                return NONE_LABEL, 1.0

            buckets: Dict[str, List[InferenceResult]] = {}
            for item in self._history:
                buckets.setdefault(item.label, []).append(item)

            best_label = None
            best_count = -1
            best_avg = -1.0
            best_ts = -1.0
            for label, items in buckets.items():
                count = len(items)
                avg = sum(x.prob for x in items) / count
                ts = max(x.timestamp for x in items)
                if (
                    count > best_count
                    or (count == best_count and avg > best_avg)
                    or (count == best_count and avg == best_avg and ts > best_ts)
                ):
                    best_label = label
                    best_count = count
                    best_avg = avg
                    best_ts = ts

            if best_label is None:
                return NONE_LABEL, 1.0
            return best_label, float(best_avg)

    def run(self) -> None:
        """Thread loop: read webcam frames and push inference results."""
        pipeline = None
        try:
            classifier = joblib.load(self._classifier_path)
            model, preprocess = clip.load("ViT-L/14", device=self._device)
            model.eval()

            width, height, cam_fps = 640, 480, 15
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, cam_fps)
            pipeline.start(config)
            time.sleep(0.5)

            dummy = np.zeros((height, width, 3), dtype=np.uint8)
            _ = _infer_probs(model, classifier, dummy, preprocess, self._device)

            infer_period = 1.0 / max(0.1, self._target_hz)
            last_infer_t = 0.0

            while not self._stop_event.is_set():
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                now = time.monotonic()
                if now - last_infer_t < infer_period:
                    continue
                last_infer_t = now

                frame_rgb = np.asanyarray(color_frame.get_data())
                frame_rgb = _maybe_crop_roi(frame_rgb)

                probs = _infer_probs(model, classifier, frame_rgb, preprocess, self._device)
                label, prob = _postprocess_probs(probs, self._threshold)

                with self._lock:
                    self._history.append(InferenceResult(label=label, prob=float(prob), timestamp=now))

        except Exception as exc:
            self._last_error = str(exc)
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass
