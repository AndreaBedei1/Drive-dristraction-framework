from __future__ import annotations

"""MQTT arousal subscriber and snapshot types."""

import json
import threading
from dataclasses import dataclass
from typing import Optional, Protocol, Tuple

try:
    import paho.mqtt.client as mqtt
except Exception:  # pragma: no cover - handled at runtime
    mqtt = None


DEFAULT_AROUSAL_TOPIC = "sensor/driver/arousal"


@dataclass(frozen=True)
class ArousalSnapshot:
    """Latest arousal sample (normalized 0..1) plus metadata."""

    value: Optional[float]
    method: Optional[str]
    timestamp_ms: Optional[int]
    quality: Optional[str]


class ArousalProvider(Protocol):
    """Protocol for arousal data sources."""

    def get_snapshot(self) -> ArousalSnapshot:
        """Return the latest arousal snapshot."""
        ...


def _parse_mqtt_url(url: str) -> Tuple[str, int, str]:
    """Parse mqtt://host:port/base into host, port, base_topic."""
    raw = url.strip()
    if raw.startswith("mqtt://"):
        raw = raw[7:]

    if "/" in raw:
        host_port, base_topic = raw.split("/", 1)
    else:
        host_port, base_topic = raw, ""

    if ":" in host_port:
        host, port_s = host_port.rsplit(":", 1)
        try:
            port = int(port_s)
        except Exception:
            port = 1883
    else:
        host, port = host_port, 1883

    return host.strip(), int(port), base_topic.strip()


def _build_arousal_topic(base_topic: str) -> str:
    if base_topic:
        trimmed = base_topic.strip("/")
        if trimmed.endswith(DEFAULT_AROUSAL_TOPIC):
            return trimmed
        return f"{trimmed}/sensor/driver/arousal".strip("/")
    return DEFAULT_AROUSAL_TOPIC


class MqttArousalClient:
    """Subscribe to arousal updates published via MQTT."""

    def __init__(self, mqtt_url: str, topic: Optional[str] = None) -> None:
        self._mqtt_url = mqtt_url.strip()
        self._topic_override = topic.strip() if topic else ""
        self._topic = ""
        self._client = None
        self._lock = threading.Lock()
        self._snapshot = ArousalSnapshot(None, None, None, None)
        self._last_error: Optional[str] = None

    def last_error(self) -> Optional[str]:
        """Return the last startup error, if any."""
        return self._last_error

    def start(self) -> bool:
        """Connect to MQTT and start the background loop."""
        if not self._mqtt_url:
            self._last_error = "empty_mqtt_url"
            return False
        if mqtt is None:
            self._last_error = "paho-mqtt not available"
            return False

        host, port, base_topic = _parse_mqtt_url(self._mqtt_url)
        self._topic = self._topic_override or _build_arousal_topic(base_topic)

        self._client = mqtt.Client()
        self._client.on_connect = self._on_connect
        self._client.on_message = self._on_message
        try:
            self._client.connect(host, port, 60)
        except Exception as exc:
            self._last_error = f"connect_failed: {exc}"
            self._client = None
            return False

        try:
            self._client.loop_start()
        except Exception as exc:
            self._last_error = f"loop_failed: {exc}"
            try:
                self._client.disconnect()
            except Exception:
                pass
            self._client = None
            return False

        self._last_error = None
        return True

    def stop(self) -> None:
        """Stop the MQTT loop and disconnect."""
        if self._client is None:
            return
        try:
            self._client.loop_stop()
        except Exception:
            pass
        try:
            self._client.disconnect()
        except Exception:
            pass
        self._client = None

    def get_snapshot(self) -> ArousalSnapshot:
        """Return the latest arousal snapshot."""
        with self._lock:
            return self._snapshot

    def _on_connect(self, client, _userdata, _flags, rc) -> None:
        if rc != 0:
            self._last_error = f"connect_rc_{rc}"
            return
        try:
            client.subscribe(self._topic)
        except Exception as exc:
            self._last_error = f"subscribe_failed: {exc}"

    def _on_message(self, _client, _userdata, msg) -> None:
        payload = None
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except Exception:
            return

        value_raw = payload.get("arousal")
        method = payload.get("method")
        ts_raw = payload.get("timestamp")
        quality = payload.get("quality")

        value = None
        if value_raw is not None:
            try:
                value = float(value_raw)
            except Exception:
                value = None

        ts = None
        if ts_raw is not None:
            try:
                ts = int(ts_raw)
            except Exception:
                ts = None

        with self._lock:
            self._snapshot = ArousalSnapshot(
                value=value,
                method=str(method) if method is not None else None,
                timestamp_ms=ts,
                quality=str(quality) if quality is not None else None,
            )
