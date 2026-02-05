from __future__ import annotations

"""Distraction window UI and coordination logic."""

import random
import string
import threading
import time
from typing import Callable, Optional, Tuple
import sys

try:
    import ctypes
except Exception:
    ctypes = None

try:
    import tkinter as tk
except Exception:
    tk = None


class DistractionCoordinator:
    """Ensure only one distraction window is active at a time."""

    def __init__(self, min_gap_seconds: float) -> None:
        """Create a coordinator with a minimum gap between alerts."""
        self._lock = threading.Lock()
        self._active_id: Optional[str] = None
        self._next_allowed_time = 0.0
        self._min_gap = float(min_gap_seconds)

    def request_start(self, window_id: str) -> bool:
        """Request permission to start an alert for a window."""
        with self._lock:
            now = time.monotonic()
            if self._active_id is None and now >= self._next_allowed_time:
                self._active_id = window_id
                return True
            return False

    def finish(self, window_id: str) -> None:
        """Mark a window as finished and set the next allowed time."""
        with self._lock:
            if self._active_id == window_id:
                self._active_id = None
                self._next_allowed_time = time.monotonic() + self._min_gap


def focus_simulation_window(window_title: str) -> None:
    """Bring the simulation window to the foreground if possible."""
    try:
        import ctypes

        hwnd = ctypes.windll.user32.FindWindowW(None, window_title)
        if hwnd:
            ctypes.windll.user32.SetForegroundWindow(hwnd)
    except Exception:
        pass


class DistractionWindow(threading.Thread):
    """Window that shows visual and audio distraction prompts."""

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
        """Create a distraction window with timing and audio settings."""
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
        self._awaiting_ack = False
        self._expected_letter = None
        self._expected_vk = None

    def stop(self) -> None:
        """Stop the window and any active audio."""
        self._stop_event.set()
        self._beep_stop_event.set()
        if self._root is not None:
            try:
                self._root.after(0, self._root.destroy)
            except Exception:
                pass

    def run(self) -> None:
        """Start the Tkinter event loop and schedule alerts."""
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
        self._start_key_listener()
        self._schedule_next()
        self._root.mainloop()

    def _set_idle(self) -> None:
        """Reset the window to the idle state."""
        if self._root is None or self._label is None:
            return
        self._root.configure(bg="#1e1e1e")
        self._expected_letter = None
        self._expected_vk = None
        self._label.configure(text="", font=("Arial", 18, "bold"))

    def _schedule_next(self) -> None:
        """Schedule the next alert window."""
        if self._stop_event.is_set() or self._root is None:
            return
        delay = random.uniform(self._min_interval, self._max_interval)
        self._root.after(int(delay * 1000), self._try_start_alert)

    def _try_start_alert(self) -> None:
        """Attempt to start an alert if coordination allows it."""
        if self._stop_event.is_set() or self._root is None:
            return
        if not self._coord.request_start(self._id):
            self._root.after(500, self._try_start_alert)
            return
        self._start_alert()

    def _start_alert(self) -> None:
        """Start the alert immediately and wait for acknowledgment."""
        if self._root is None or self._label is None:
            return
        self._root.configure(bg="#b00020")
        self._expected_letter = random.choice(string.ascii_uppercase)
        self._expected_vk = ord(self._expected_letter)
        self._label.configure(text=self._expected_letter, font=("Arial", 96, "bold"))
        self._awaiting_ack = True
        self._on_start(self._id)
        self._start_beep()
        try:
            self._root.focus_force()
        except Exception:
            pass
        self._root.bind("<Key>", self._on_key)

    def _on_key(self, event) -> None:
        """Handle a keyboard acknowledgment."""
        if not self._awaiting_ack:
            return
        if not self._expected_letter:
            return
        try:
            pressed = event.char.upper() if event.char else ""
        except Exception:
            pressed = ""
        if pressed == self._expected_letter:
            self._acknowledge()

    def _acknowledge(self) -> None:
        """Clear the alarm state and schedule the next alert."""
        if self._root is None:
            return
        if not self._awaiting_ack:
            return
        self._awaiting_ack = False
        self._root.unbind("<Key>")
        self._beep_stop_event.set()
        self._on_finish(self._id)
        self._coord.finish(self._id)
        self._set_idle()
        self._focus_callback()
        self._schedule_next()

    def _start_key_listener(self) -> None:
        """Listen for the expected key globally on Windows."""
        if ctypes is None or sys.platform != "win32":
            return

        def _loop() -> None:
            last_down = False
            last_vk = None
            while not self._stop_event.is_set():
                vk = self._expected_vk if self._awaiting_ack else None
                if vk != last_vk:
                    last_down = False
                    last_vk = vk
                if vk is None:
                    time.sleep(0.05)
                    continue
                try:
                    down = bool(ctypes.windll.user32.GetAsyncKeyState(int(vk)) & 0x8000)
                except Exception:
                    time.sleep(0.05)
                    continue
                if down and not last_down and self._awaiting_ack:
                    try:
                        if self._root is not None:
                            self._root.after(0, self._acknowledge)
                    except Exception:
                        pass
                last_down = down
                time.sleep(0.03)

        threading.Thread(target=_loop, daemon=True).start()

    def _start_beep(self) -> None:
        """Start the repeating beep until the alert is acknowledged."""
        self._beep_stop_event.clear()

        def _beep_loop() -> None:
            while not self._beep_stop_event.is_set():
                try:
                    import winsound

                    winsound.Beep(self._beep_frequency, self._beep_duration)
                except Exception:
                    time.sleep(self._beep_duration / 1000.0)

        threading.Thread(target=_beep_loop, daemon=True).start()
