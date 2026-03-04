from __future__ import annotations

"""Event-driven synchronized inference on a shared camera frame."""

import datetime
import multiprocessing as mp
import queue
import threading
import time
from concurrent.futures import Future, ProcessPoolExecutor, TimeoutError
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

import numpy as np

try:
    import pyrealsense2 as rs
except Exception:  # pragma: no cover - handled at runtime
    rs = None

from src.arousal_provider import ArousalProvider, ArousalSnapshot
from src.emotion_provider import EmotionSnapshot


DEFAULT_CAPTURE_HZ = 15.0
DEFAULT_SAMPLE_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class SynchronizedInferenceSample:
    """Inference results tied to a single request timestamp and frame."""

    request_timestamp: float
    request_timestamp_iso: str
    frame_timestamp: Optional[float]
    model_label: str
    model_prob: Optional[float]
    model_timestamp: Optional[float]
    emotion_label: Optional[str]
    emotion_prob: Optional[float]
    emotion_timestamp: Optional[float]
    arousal_snapshot: ArousalSnapshot


class SynchronizedInferenceProvider(Protocol):
    """Protocol for event-driven synchronized inference."""

    def capture_sample(self, timeout: Optional[float] = None) -> SynchronizedInferenceSample:
        """Return a synchronized sample bundle for the current event."""
        ...


class FrameInferenceWorker(Protocol):
    """Common interface shared by thread and process inference workers."""

    def start(self) -> None:
        ...

    def stop(self) -> None:
        ...

    def join(self, timeout: Optional[float] = None) -> None:
        ...

    def is_alive(self) -> bool:
        ...

    def submit_frame(self, frame: np.ndarray, request_timestamp: float) -> Future:
        ...


class FrameCaptureService(threading.Thread):
    """Continuously capture frames and expose the latest RGB image."""

    def __init__(self, target_hz: float = DEFAULT_CAPTURE_HZ) -> None:
        super().__init__(daemon=True)
        self._target_hz = max(1.0, float(target_hz))
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._last_frame: Optional[np.ndarray] = None
        self._last_timestamp: Optional[float] = None
        self._last_error: Optional[str] = None

    def stop(self) -> None:
        self._stop_event.set()

    def last_error(self) -> Optional[str]:
        return self._last_error

    def wait_for_frame(self, timeout: Optional[float] = None) -> bool:
        return self._ready_event.wait(timeout=timeout)

    def get_latest_frame(self) -> Optional[np.ndarray]:
        frame, _timestamp = self.get_latest_frame_with_timestamp()
        return frame

    def get_latest_frame_with_timestamp(self) -> Tuple[Optional[np.ndarray], Optional[float]]:
        with self._frame_lock:
            if self._last_frame is None:
                return None, None
            return self._last_frame.copy(), self._last_timestamp

    def run(self) -> None:
        if rs is None:
            self._last_error = "pyrealsense2_not_available"
            return

        pipeline = None
        try:
            width, height, cam_fps = 640, 480, 15
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, cam_fps)
            pipeline.start(config)
            while not self._stop_event.is_set():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=int(max(100, 1000.0 / self._target_hz)))
                except Exception:
                    continue
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue
                frame_rgb = np.asanyarray(color_frame.get_data())
                stamp = time.monotonic()
                with self._frame_lock:
                    self._last_frame = frame_rgb.copy()
                    self._last_timestamp = stamp
                self._ready_event.set()
        except Exception as exc:
            self._last_error = str(exc)
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass


@dataclass
class _FrameInferenceRequest:
    frame: np.ndarray
    request_timestamp: float
    future: Future


class _BaseFrameWorker(threading.Thread):
    """Base class for request/response workers operating on RGB frames."""

    def __init__(self, queue_size: int = 16) -> None:
        super().__init__(daemon=True)
        self._queue: "queue.Queue[Optional[_FrameInferenceRequest]]" = queue.Queue(maxsize=max(1, int(queue_size)))
        self._stop_event = threading.Event()
        self._ready_event = threading.Event()
        self._last_error: Optional[str] = None

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except Exception:
            pass

    def last_error(self) -> Optional[str]:
        return self._last_error

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        return self._ready_event.wait(timeout=timeout)

    def submit_frame(self, frame: np.ndarray, request_timestamp: float) -> Future:
        future: Future = Future()
        if self._stop_event.is_set():
            future.set_result(self._default_result(request_timestamp))
            return future
        try:
            req = _FrameInferenceRequest(frame=frame.copy(), request_timestamp=float(request_timestamp), future=future)
            self._queue.put(req, timeout=1.0)
        except Exception as exc:
            self._last_error = str(exc)
            future.set_result(self._default_result(request_timestamp))
        return future

    def _default_result(self, request_timestamp: float):
        raise NotImplementedError


class DistractionInferenceWorker(_BaseFrameWorker):
    """Run the distraction classifier on demand in its own thread."""

    def __init__(
        self,
        classifier_path: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.80,
        queue_size: int = 16,
    ) -> None:
        super().__init__(queue_size=queue_size)
        self._classifier_path = classifier_path
        self._device = device
        self._threshold = float(threshold)

    def _default_result(self, request_timestamp: float):
        from src.model_inference import InferenceResult, NONE_LABEL

        return InferenceResult(label=NONE_LABEL, prob=1.0, timestamp=float(request_timestamp))

    def run(self) -> None:
        model = None
        classifier = None
        preprocess = None
        device = None
        try:
            import clip
            import joblib
            import torch
            from src.model_inference import (
                _default_classifier_path,
                _infer_probs,
                _maybe_crop_roi,
                _postprocess_probs,
                InferenceResult,
                NONE_LABEL,
            )

            classifier_path = self._classifier_path or _default_classifier_path()
            device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
            classifier = joblib.load(classifier_path)
            model, preprocess = clip.load("ViT-L/14", device=device)
            model.eval()
            dummy = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = _infer_probs(model, classifier, dummy, preprocess, device)
            self._ready_event.set()

            while not self._stop_event.is_set():
                try:
                    req = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if req is None:
                    continue
                try:
                    frame_rgb = _maybe_crop_roi(req.frame)
                    probs = _infer_probs(model, classifier, frame_rgb, preprocess, device)
                    label, prob = _postprocess_probs(probs, self._threshold)
                    req.future.set_result(
                        InferenceResult(label=label, prob=float(prob), timestamp=float(req.request_timestamp))
                    )
                except Exception as exc:
                    self._last_error = str(exc)
                    req.future.set_result(
                        InferenceResult(
                            label=NONE_LABEL,
                            prob=1.0,
                            timestamp=float(req.request_timestamp),
                        )
                    )
        except Exception as exc:
            self._last_error = str(exc)
            self._ready_event.set()
            while not self._stop_event.is_set():
                try:
                    req = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if req is None:
                    continue
                req.future.set_result(self._default_result(req.request_timestamp))


class EmotionInferenceWorker(_BaseFrameWorker):
    """Run the emotion model on demand in its own thread."""

    def __init__(
        self,
        detector_backend: Optional[str] = None,
        enforce_detection: Optional[bool] = None,
        queue_size: int = 16,
    ) -> None:
        super().__init__(queue_size=queue_size)
        self._detector_backend = detector_backend
        self._enforce_detection = enforce_detection

    def _default_result(self, request_timestamp: float):
        return EmotionSnapshot(label=None, prob=None, timestamp=float(request_timestamp))

    def run(self) -> None:
        try:
            from src.emotion_inference import (
                _extract_emotion,
                _import_deepface,
                _normalize_results,
                _select_best_result,
                DEFAULT_DETECTOR_BACKEND,
                DEFAULT_ENFORCE_DETECTION,
            )

            DeepFace = _import_deepface()
            detector_backend = self._detector_backend or DEFAULT_DETECTOR_BACKEND
            enforce_detection = (
                DEFAULT_ENFORCE_DETECTION if self._enforce_detection is None else bool(self._enforce_detection)
            )
            self._ready_event.set()

            while not self._stop_event.is_set():
                try:
                    req = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if req is None:
                    continue
                try:
                    frame_bgr = np.ascontiguousarray(req.frame[:, :, ::-1])
                    raw = DeepFace.analyze(
                        img_path=frame_bgr,
                        actions=("emotion",),
                        enforce_detection=enforce_detection,
                        detector_backend=detector_backend,
                        align=True,
                        silent=True,
                    )
                    results = _normalize_results(raw)
                    best = _select_best_result(results)
                    label, prob = _extract_emotion(best)
                    req.future.set_result(
                        EmotionSnapshot(
                            label=label,
                            prob=prob,
                            timestamp=float(req.request_timestamp),
                        )
                    )
                except Exception as exc:
                    self._last_error = str(exc)
                    req.future.set_result(
                        EmotionSnapshot(label=None, prob=None, timestamp=float(req.request_timestamp))
                    )
        except Exception as exc:
            self._last_error = str(exc)
            self._ready_event.set()
            while not self._stop_event.is_set():
                try:
                    req = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                if req is None:
                    continue
                req.future.set_result(self._default_result(req.request_timestamp))


_DISTRACTION_PROCESS_STATE = None
_EMOTION_PROCESS_STATE = None


def _load_distraction_process_state(classifier_path: Optional[str], device: Optional[str]):
    """Load and cache the distraction model inside a worker process."""
    global _DISTRACTION_PROCESS_STATE
    import torch
    from src.model_inference import _default_classifier_path, _infer_probs

    resolved_classifier_path = classifier_path or _default_classifier_path()
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if _DISTRACTION_PROCESS_STATE is not None:
        cached_path, cached_device, model, classifier, preprocess = _DISTRACTION_PROCESS_STATE
        if cached_path == resolved_classifier_path and cached_device == resolved_device:
            return model, classifier, preprocess, cached_device

    import clip
    import joblib
    classifier = joblib.load(resolved_classifier_path)
    model, preprocess = clip.load("ViT-L/14", device=resolved_device)
    model.eval()
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    _ = _infer_probs(model, classifier, dummy, preprocess, resolved_device)
    _DISTRACTION_PROCESS_STATE = (
        resolved_classifier_path,
        resolved_device,
        model,
        classifier,
        preprocess,
    )
    return model, classifier, preprocess, resolved_device


def _warmup_distraction_process(classifier_path: Optional[str], device: Optional[str]) -> bool:
    """Eagerly load CLIP+classifier in the subprocess."""
    _load_distraction_process_state(classifier_path, device)
    return True


def _run_distraction_process_inference(
    frame: np.ndarray,
    request_timestamp: float,
    classifier_path: Optional[str],
    device: Optional[str],
    threshold: float,
):
    """Run distraction inference in a subprocess."""
    from src.model_inference import InferenceResult, NONE_LABEL, _infer_probs, _maybe_crop_roi, _postprocess_probs

    try:
        model, classifier, preprocess, resolved_device = _load_distraction_process_state(classifier_path, device)
        frame_rgb = _maybe_crop_roi(frame)
        probs = _infer_probs(model, classifier, frame_rgb, preprocess, resolved_device)
        label, prob = _postprocess_probs(probs, float(threshold))
        return InferenceResult(label=label, prob=float(prob), timestamp=float(request_timestamp))
    except Exception:
        return InferenceResult(label=NONE_LABEL, prob=1.0, timestamp=float(request_timestamp))


def _load_emotion_process_state(detector_backend: Optional[str], enforce_detection: Optional[bool]):
    """Load and cache DeepFace inside a worker process."""
    global _EMOTION_PROCESS_STATE
    from src.emotion_inference import DEFAULT_DETECTOR_BACKEND, DEFAULT_ENFORCE_DETECTION, _import_deepface

    resolved_backend = detector_backend or DEFAULT_DETECTOR_BACKEND
    resolved_enforce = DEFAULT_ENFORCE_DETECTION if enforce_detection is None else bool(enforce_detection)
    if _EMOTION_PROCESS_STATE is not None:
        cached_backend, cached_enforce, DeepFace = _EMOTION_PROCESS_STATE
        if cached_backend == resolved_backend and cached_enforce == resolved_enforce:
            return DeepFace, resolved_backend, resolved_enforce

    DeepFace = _import_deepface()
    try:
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        DeepFace.analyze(
            img_path=np.ascontiguousarray(dummy[:, :, ::-1]),
            actions=("emotion",),
            enforce_detection=resolved_enforce,
            detector_backend=resolved_backend,
            align=True,
            silent=True,
        )
    except Exception:
        pass
    _EMOTION_PROCESS_STATE = (resolved_backend, resolved_enforce, DeepFace)
    return DeepFace, resolved_backend, resolved_enforce


def _warmup_emotion_process(detector_backend: Optional[str], enforce_detection: Optional[bool]) -> bool:
    """Eagerly load DeepFace in the subprocess."""
    _load_emotion_process_state(detector_backend, enforce_detection)
    return True


def _run_emotion_process_inference(
    frame: np.ndarray,
    request_timestamp: float,
    detector_backend: Optional[str],
    enforce_detection: Optional[bool],
):
    """Run emotion inference in a subprocess."""
    from src.emotion_inference import _extract_emotion, _normalize_results, _select_best_result

    try:
        DeepFace, resolved_backend, resolved_enforce = _load_emotion_process_state(
            detector_backend, enforce_detection
        )
        frame_bgr = np.ascontiguousarray(frame[:, :, ::-1])
        raw = DeepFace.analyze(
            img_path=frame_bgr,
            actions=("emotion",),
            enforce_detection=resolved_enforce,
            detector_backend=resolved_backend,
            align=True,
            silent=True,
        )
        results = _normalize_results(raw)
        best = _select_best_result(results)
        label, prob = _extract_emotion(best)
        return EmotionSnapshot(label=label, prob=prob, timestamp=float(request_timestamp))
    except Exception:
        return EmotionSnapshot(label=None, prob=None, timestamp=float(request_timestamp))


class DistractionInferenceProcessWorker:
    """Run distraction inference in a dedicated subprocess."""

    def __init__(
        self,
        classifier_path: Optional[str] = None,
        device: Optional[str] = None,
        threshold: float = 0.80,
    ) -> None:
        self._classifier_path = classifier_path
        self._device = device
        self._threshold = float(threshold)
        self._executor: Optional[ProcessPoolExecutor] = None
        self._ready_future: Optional[Future] = None
        self._last_error: Optional[str] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._executor is not None:
                return
            self._executor = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
            self._ready_future = self._executor.submit(
                _warmup_distraction_process,
                self._classifier_path,
                self._device,
            )

    def stop(self) -> None:
        with self._lock:
            executor = self._executor
            self._executor = None
            self._ready_future = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def join(self, timeout: Optional[float] = None) -> None:
        _ = timeout

    def is_alive(self) -> bool:
        with self._lock:
            return self._executor is not None

    def last_error(self) -> Optional[str]:
        return self._last_error

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        future = self._ready_future
        if future is None:
            return False
        try:
            future.result(timeout=timeout)
            return True
        except TimeoutError:
            return False
        except Exception as exc:
            self._last_error = str(exc)
            return True

    def submit_frame(self, frame: np.ndarray, request_timestamp: float) -> Future:
        from src.model_inference import InferenceResult, NONE_LABEL

        with self._lock:
            executor = self._executor
        if executor is None:
            future: Future = Future()
            future.set_result(InferenceResult(label=NONE_LABEL, prob=1.0, timestamp=float(request_timestamp)))
            return future
        try:
            return executor.submit(
                _run_distraction_process_inference,
                frame.copy(),
                float(request_timestamp),
                self._classifier_path,
                self._device,
                self._threshold,
            )
        except Exception as exc:
            self._last_error = str(exc)
            future = Future()
            future.set_result(InferenceResult(label=NONE_LABEL, prob=1.0, timestamp=float(request_timestamp)))
            return future


class EmotionInferenceProcessWorker:
    """Run emotion inference in a dedicated subprocess."""

    def __init__(
        self,
        detector_backend: Optional[str] = None,
        enforce_detection: Optional[bool] = None,
    ) -> None:
        self._detector_backend = detector_backend
        self._enforce_detection = enforce_detection
        self._executor: Optional[ProcessPoolExecutor] = None
        self._ready_future: Optional[Future] = None
        self._last_error: Optional[str] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._executor is not None:
                return
            self._executor = ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"))
            self._ready_future = self._executor.submit(
                _warmup_emotion_process,
                self._detector_backend,
                self._enforce_detection,
            )

    def stop(self) -> None:
        with self._lock:
            executor = self._executor
            self._executor = None
            self._ready_future = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)

    def join(self, timeout: Optional[float] = None) -> None:
        _ = timeout

    def is_alive(self) -> bool:
        with self._lock:
            return self._executor is not None

    def last_error(self) -> Optional[str]:
        return self._last_error

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        future = self._ready_future
        if future is None:
            return False
        try:
            future.result(timeout=timeout)
            return True
        except TimeoutError:
            return False
        except Exception as exc:
            self._last_error = str(exc)
            return True

    def submit_frame(self, frame: np.ndarray, request_timestamp: float) -> Future:
        with self._lock:
            executor = self._executor
        if executor is None:
            future: Future = Future()
            future.set_result(EmotionSnapshot(label=None, prob=None, timestamp=float(request_timestamp)))
            return future
        try:
            return executor.submit(
                _run_emotion_process_inference,
                frame.copy(),
                float(request_timestamp),
                self._detector_backend,
                self._enforce_detection,
            )
        except Exception as exc:
            self._last_error = str(exc)
            future = Future()
            future.set_result(EmotionSnapshot(label=None, prob=None, timestamp=float(request_timestamp)))
            return future


class SynchronizedInferenceCoordinator(SynchronizedInferenceProvider):
    """Capture one frame and dispatch all inferences against that same image."""

    def __init__(
        self,
        frame_capture: Optional[FrameCaptureService] = None,
        distraction_worker: Optional[FrameInferenceWorker] = None,
        emotion_worker: Optional[FrameInferenceWorker] = None,
        arousal_provider: Optional[ArousalProvider] = None,
        sample_timeout_seconds: float = DEFAULT_SAMPLE_TIMEOUT_SECONDS,
    ) -> None:
        self._frame_capture = frame_capture
        self._distraction_worker = distraction_worker
        self._emotion_worker = emotion_worker
        self._arousal_provider = arousal_provider
        self._sample_timeout_seconds = max(0.1, float(sample_timeout_seconds))
        self._lock = threading.Lock()
        self._last_sample: Optional[SynchronizedInferenceSample] = None

    def start(self) -> None:
        for worker in (self._frame_capture, self._distraction_worker, self._emotion_worker):
            if worker is None:
                continue
            try:
                if not worker.is_alive():
                    worker.start()
            except RuntimeError:
                pass

    def stop(self) -> None:
        for worker in (self._frame_capture, self._distraction_worker, self._emotion_worker):
            if worker is None:
                continue
            try:
                worker.stop()
            except Exception:
                pass

    def join(self, timeout: Optional[float] = None) -> None:
        for worker in (self._frame_capture, self._distraction_worker, self._emotion_worker):
            if worker is None:
                continue
            try:
                worker.join(timeout=timeout)
            except Exception:
                pass

    def get_latest_frame(self) -> Optional[np.ndarray]:
        if self._frame_capture is None:
            return None
        return self._frame_capture.get_latest_frame()

    def get_window_summary(self) -> Tuple[str, float]:
        with self._lock:
            if self._last_sample is None:
                return "None", 1.0
            prob = self._last_sample.model_prob
            return self._last_sample.model_label, 1.0 if prob is None else float(prob)

    def get_window_summary_with_timestamp(self) -> Tuple[str, float, Optional[float]]:
        with self._lock:
            if self._last_sample is None:
                return "None", 1.0, None
            prob = self._last_sample.model_prob
            return (
                self._last_sample.model_label,
                1.0 if prob is None else float(prob),
                self._last_sample.model_timestamp,
            )

    def get_latest_emotion_snapshot(self) -> EmotionSnapshot:
        with self._lock:
            if self._last_sample is None:
                return EmotionSnapshot(None, None, None)
            return EmotionSnapshot(
                label=self._last_sample.emotion_label,
                prob=self._last_sample.emotion_prob,
                timestamp=self._last_sample.emotion_timestamp,
            )

    def capture_sample(self, timeout: Optional[float] = None) -> SynchronizedInferenceSample:
        request_timestamp = time.monotonic()
        request_timestamp_iso = datetime.datetime.utcnow().isoformat()
        sample_timeout = self._sample_timeout_seconds if timeout is None else max(0.1, float(timeout))
        deadline = time.monotonic() + sample_timeout

        frame = None
        frame_timestamp = None
        if self._frame_capture is not None:
            frame, frame_timestamp = self._frame_capture.get_latest_frame_with_timestamp()
            if frame is None:
                self._frame_capture.wait_for_frame(timeout=min(1.0, sample_timeout))
                frame, frame_timestamp = self._frame_capture.get_latest_frame_with_timestamp()

        arousal_snapshot = ArousalSnapshot(None, None, None, None, None)
        if self._arousal_provider is not None:
            try:
                arousal_snapshot = self._arousal_provider.get_snapshot()
            except Exception:
                arousal_snapshot = ArousalSnapshot(None, None, None, None, None)

        model_future: Optional[Future] = None
        emotion_future: Optional[Future] = None
        if frame is not None:
            if self._distraction_worker is not None:
                model_future = self._distraction_worker.submit_frame(frame, request_timestamp)
            if self._emotion_worker is not None:
                emotion_future = self._emotion_worker.submit_frame(frame, request_timestamp)

        model_label = "None"
        model_prob: Optional[float] = 1.0
        model_timestamp: Optional[float] = None
        if model_future is not None:
            timeout_left = max(0.0, deadline - time.monotonic())
            try:
                result = model_future.result(timeout=timeout_left)
                model_label = str(getattr(result, "label", "None"))
                model_prob = float(getattr(result, "prob", 1.0))
                model_timestamp = float(getattr(result, "timestamp", request_timestamp))
            except TimeoutError:
                model_label = "None"
                model_prob = 1.0
                model_timestamp = request_timestamp
            except Exception:
                model_label = "None"
                model_prob = 1.0
                model_timestamp = request_timestamp

        emotion_label: Optional[str] = None
        emotion_prob: Optional[float] = None
        emotion_timestamp: Optional[float] = None
        if emotion_future is not None:
            timeout_left = max(0.0, deadline - time.monotonic())
            try:
                result = emotion_future.result(timeout=timeout_left)
                emotion_label = getattr(result, "label", None)
                emotion_prob = getattr(result, "prob", None)
                ts = getattr(result, "timestamp", request_timestamp)
                emotion_timestamp = float(ts) if ts is not None else None
            except TimeoutError:
                emotion_timestamp = request_timestamp
            except Exception:
                emotion_timestamp = request_timestamp

        sample = SynchronizedInferenceSample(
            request_timestamp=request_timestamp,
            request_timestamp_iso=request_timestamp_iso,
            frame_timestamp=frame_timestamp,
            model_label=model_label,
            model_prob=model_prob,
            model_timestamp=model_timestamp,
            emotion_label=emotion_label,
            emotion_prob=emotion_prob,
            emotion_timestamp=emotion_timestamp,
            arousal_snapshot=arousal_snapshot,
        )
        with self._lock:
            self._last_sample = sample
        return sample


class SynchronizedEmotionPreviewProvider:
    """Adapter exposing the latest synchronized emotion result as a provider."""

    def __init__(self, coordinator: SynchronizedInferenceCoordinator) -> None:
        self._coordinator = coordinator

    def get_snapshot(self) -> EmotionSnapshot:
        return self._coordinator.get_latest_emotion_snapshot()
