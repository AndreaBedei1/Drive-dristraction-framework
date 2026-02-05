from __future__ import annotations

"""OpenCV camera preview with classification and arousal overlays."""

import threading
import time
from typing import Optional, Protocol, Tuple

try:
    import cv2
except Exception:  # pragma: no cover - handled at runtime
    cv2 = None

from src.arousal_provider import ArousalProvider


class ModelInferenceProvider(Protocol):
    """Protocol for model inference providers."""

    def get_window_summary(self) -> Tuple[str, float]:
        """Return (label, probability) for the latest inference window."""
        ...

    def get_latest_frame(self):
        """Return the latest RGB frame."""
        ...


def preview_available() -> bool:
    return cv2 is not None


class CameraPreviewWindow(threading.Thread):
    """Preview window showing the camera feed and overlays."""

    def __init__(
        self,
        model_provider: Optional[ModelInferenceProvider],
        arousal_provider: Optional[ArousalProvider],
        title: str = "Driver Camera",
    ) -> None:
        super().__init__(daemon=True)
        self._model_provider = model_provider
        self._arousal_provider = arousal_provider
        self._title = title
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Stop the preview loop."""
        self._stop_event.set()
        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    def _arousal_text(self) -> str:
        if self._arousal_provider is None:
            return "Arousal: --"
        try:
            snap = self._arousal_provider.get_snapshot()
        except Exception:
            return "Arousal: --"
        if snap.value is None:
            return "Arousal: --"
        if snap.method:
            return f"Arousal: {snap.value:.2f} ({snap.method})"
        return f"Arousal: {snap.value:.2f}"

    def _model_text(self) -> Tuple[str, str]:
        if self._model_provider is None:
            return "Class: None", "Prob: --"
        try:
            label, prob = self._model_provider.get_window_summary()
        except Exception:
            return "Class: None", "Prob: --"
        return f"Class: {label}", f"Prob: {prob:.2f}"

    def run(self) -> None:
        """Start the OpenCV preview loop."""
        if cv2 is None:
            return

        while not self._stop_event.is_set():
            frame = None
            if self._model_provider is not None:
                try:
                    frame = self._model_provider.get_latest_frame()
                except Exception:
                    frame = None

            if frame is None:
                time.sleep(0.05)
                continue

            try:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            except Exception:
                time.sleep(0.05)
                continue

            class_text, prob_text = self._model_text()
            arousal_text = self._arousal_text()

            cv2.putText(frame_bgr, class_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame_bgr, prob_text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame_bgr, arousal_text, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

            cv2.imshow(self._title, frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                self.stop()
                break
