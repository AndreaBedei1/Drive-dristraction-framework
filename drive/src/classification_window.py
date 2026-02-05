from __future__ import annotations

"""Tkinter window showing model classification and arousal."""

import threading
from typing import Optional, Protocol, Tuple

try:
    import tkinter as tk
except Exception:  # pragma: no cover - handled at runtime
    tk = None

from src.arousal_provider import ArousalProvider


class ModelInferenceProvider(Protocol):
    """Protocol for model inference providers."""

    def get_window_summary(self) -> Tuple[str, float]:
        """Return (label, probability) for the latest inference window."""
        ...


class ClassificationWindow(threading.Thread):
    """Small window displaying current class, probability, and arousal."""

    def __init__(
        self,
        model_provider: Optional[ModelInferenceProvider],
        arousal_provider: Optional[ArousalProvider],
        title: str = "Driver Classification",
        update_hz: float = 2.0,
        size: Tuple[int, int] = (320, 160),
        monitor_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        super().__init__(daemon=True)
        self._model_provider = model_provider
        self._arousal_provider = arousal_provider
        self._title = title
        self._update_ms = int(max(0.2, 1.0 / max(0.1, update_hz)) * 1000)
        self._size = size
        self._monitor_rect = monitor_rect

        self._root = None
        self._label = None
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Stop the window loop."""
        self._stop_event.set()
        if self._root is not None:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass

    def run(self) -> None:
        """Start the Tkinter event loop."""
        if tk is None:
            return

        self._root = tk.Tk()
        self._root.title(self._title)
        self._root.configure(bg="#1e1e1e")
        self._root.protocol("WM_DELETE_WINDOW", self.stop)

        w, h = self._size
        if self._monitor_rect is not None:
            x, y, _mw, _mh = self._monitor_rect
            self._root.geometry(f"{w}x{h}+{x + 20}+{y + 20}")
        else:
            self._root.geometry(f"{w}x{h}")

        self._label = tk.Label(
            self._root,
            text="",
            font=("Arial", 14, "bold"),
            justify="left",
            fg="white",
            bg="#1e1e1e",
        )
        self._label.pack(expand=True, padx=12, pady=12, anchor="w")

        self._update_text()
        self._root.mainloop()

    def _model_snapshot(self) -> Tuple[str, Optional[float]]:
        if self._model_provider is None:
            return "None", None
        try:
            label, prob = self._model_provider.get_window_summary()
            return str(label), float(prob)
        except Exception:
            return "None", None

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

    def _update_text(self) -> None:
        if self._stop_event.is_set() or self._root is None or self._label is None:
            return

        label, prob = self._model_snapshot()
        prob_text = f"{prob:.2f}" if prob is not None else "--"
        arousal_text = self._arousal_text()

        self._label.configure(
            text=f"Class: {label}\nProb: {prob_text}\n{arousal_text}"
        )

        self._root.after(self._update_ms, self._update_text)
