from __future__ import annotations

import random
import threading
import time
from typing import Callable, Optional, Tuple

try:
    import tkinter as tk
except Exception:  # pragma: no cover - optional UI dependency
    tk = None


class DistractionCoordinator:
    def __init__(self, min_gap_seconds: float) -> None:
        self._lock = threading.Lock()
        self._active_id: Optional[str] = None
        self._next_allowed_time = 0.0
        self._min_gap = float(min_gap_seconds)

    def request_start(self, window_id: str) -> bool:
        with self._lock:
            now = time.monotonic()
            if self._active_id is None and now >= self._next_allowed_time:
                self._active_id = window_id
                return True
            return False

    def finish(self, window_id: str) -> None:
        with self._lock:
            if self._active_id == window_id:
                self._active_id = None
                self._next_allowed_time = time.monotonic() + self._min_gap


def focus_simulation_window(window_title: str) -> None:
    try:
        import ctypes

        hwnd = ctypes.windll.user32.FindWindowW(None, window_title)
        if hwnd:
            ctypes.windll.user32.SetForegroundWindow(hwnd)
    except Exception:
        pass


class DistractionWindow(threading.Thread):
    def __init__(
        self,
        window_id: str,
        title: str,
        coordinator: DistractionCoordinator,
        min_interval_seconds: float,
        max_interval_seconds: float,
        flash_duration_seconds: float,
        flash_start_interval_seconds: float,
        flash_min_interval_seconds: float,
        beep_frequency_hz: int,
        beep_duration_ms: int,
        on_start: Callable[[str], None],
        on_finish: Callable[[str], None],
        focus_callback: Callable[[], None],
        size: Tuple[int, int] = (360, 260),
        monitor_rect: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        super().__init__(daemon=True)
        self._id = window_id
        self._title = title
        self._coord = coordinator
        self._min_interval = float(min_interval_seconds)
        self._max_interval = float(max_interval_seconds)
        self._flash_duration = float(flash_duration_seconds)
        self._flash_start_interval = float(flash_start_interval_seconds)
        self._flash_min_interval = float(flash_min_interval_seconds)
        self._beep_frequency = int(beep_frequency_hz)
        self._beep_duration = int(beep_duration_ms)
        self._on_start = on_start
        self._on_finish = on_finish
        self._focus_callback = focus_callback
        self._size = size
        self._monitor_rect = monitor_rect

        self._root = None
        self._label = None
        self._stop_event = threading.Event()
        self._beep_stop_event = threading.Event()
        self._alert_start_time = 0.0
        self._flash_interval = self._flash_start_interval

    def stop(self) -> None:
        self._stop_event.set()
        self._beep_stop_event.set()
        if self._root is not None:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass

    def run(self) -> None:
        if tk is None:
            return

        self._root = tk.Tk()
        self._root.title(self._title)
        if self._monitor_rect is not None:
            x, y, w, h = self._monitor_rect
            self._root.geometry(f"{w}x{h}+{x}+{y}")
            self._root.overrideredirect(True)
        else:
            self._root.geometry(f"{self._size[0]}x{self._size[1]}")
        self._root.protocol("WM_DELETE_WINDOW", self.stop)

        self._label = tk.Label(self._root, text="", font=("Arial", 18, "bold"))
        self._label.pack(expand=True)

        self._set_idle()
        self._schedule_next()
        self._root.mainloop()

    def _set_idle(self) -> None:
        if self._root is None or self._label is None:
            return
        self._root.configure(bg="#1e1e1e")
        self._label.configure(text="")

    def _schedule_next(self) -> None:
        if self._stop_event.is_set() or self._root is None:
            return
        delay = random.uniform(self._min_interval, self._max_interval)
        self._root.after(int(delay * 1000), self._try_start_alert)

    def _try_start_alert(self) -> None:
        if self._stop_event.is_set() or self._root is None:
            return
        if not self._coord.request_start(self._id):
            self._root.after(500, self._try_start_alert)
            return
        self._start_flashing()

    def _start_flashing(self) -> None:
        if self._root is None or self._label is None:
            return
        self._alert_start_time = time.monotonic()
        self._flash_interval = self._flash_start_interval
        self._label.configure(text="")
        self._flash_step()

    def _flash_step(self) -> None:
        if self._stop_event.is_set() or self._root is None:
            return
        elapsed = time.monotonic() - self._alert_start_time
        if elapsed >= self._flash_duration:
            self._expire()
            return

        color = f"#{random.randint(0, 255):02x}{random.randint(0, 255):02x}{random.randint(0, 255):02x}"
        self._root.configure(bg=color)

        self._flash_interval = max(self._flash_min_interval, self._flash_interval * 0.85)
        self._root.after(int(self._flash_interval * 1000), self._flash_step)

    def _expire(self) -> None:
        if self._root is None or self._label is None:
            return
        self._root.configure(bg="#b00020")
        self._label.configure(text="PRESS\nTIME EXPIRED")
        self._on_start(self._id)
        self._start_beep()
        self._root.bind("<Button-1>", self._on_click)

    def _on_click(self, _event) -> None:
        if self._root is None:
            return
        self._root.unbind("<Button-1>")
        self._beep_stop_event.set()
        self._on_finish(self._id)
        self._coord.finish(self._id)
        self._set_idle()
        self._focus_callback()
        self._schedule_next()

    def _start_beep(self) -> None:
        self._beep_stop_event.clear()

        def _beep_loop() -> None:
            while not self._beep_stop_event.is_set():
                try:
                    import winsound

                    winsound.Beep(self._beep_frequency, self._beep_duration)
                except Exception:
                    time.sleep(self._beep_duration / 1000.0)

        threading.Thread(target=_beep_loop, daemon=True).start()
