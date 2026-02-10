from __future__ import annotations

"""Arousal snapshot types and provider interfaces."""

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class ArousalSnapshot:
    """Latest arousal sample (normalized 0..1) plus metadata."""

    value: Optional[float]
    method: Optional[str]
    timestamp_ms: Optional[int]
    quality: Optional[str]
    hr_bpm: Optional[int]


class ArousalProvider(Protocol):
    """Protocol for arousal data sources."""

    def get_snapshot(self) -> ArousalSnapshot:
        """Return the latest arousal snapshot."""
        ...


class StaticArousalProvider:
    """Return a fixed arousal snapshot (placeholder)."""

    def __init__(
        self,
        value: Optional[float],
        method: Optional[str] = None,
        quality: Optional[str] = None,
        hr_bpm: Optional[int] = None,
    ) -> None:
        self._snapshot = ArousalSnapshot(
            value=value,
            method=method,
            timestamp_ms=None,
            quality=quality,
            hr_bpm=hr_bpm,
        )

    def get_snapshot(self) -> ArousalSnapshot:
        return self._snapshot

    def stop(self) -> None:
        """No-op stop for compatibility."""
        return None
