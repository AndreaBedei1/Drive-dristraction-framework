from __future__ import annotations

"""Emotion snapshot types and provider protocol."""

from dataclasses import dataclass
from typing import Optional, Protocol


@dataclass(frozen=True)
class EmotionSnapshot:
    """Latest emotion prediction plus metadata."""

    label: Optional[str]
    prob: Optional[float]
    timestamp: Optional[float]


class EmotionProvider(Protocol):
    """Protocol for emotion data sources."""

    def get_snapshot(self) -> EmotionSnapshot:
        """Return the latest emotion snapshot."""
        ...
