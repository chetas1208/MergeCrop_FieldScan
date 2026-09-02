"""DJI Cloud API MQTT telemetry client.

NEEDS_VERIFICATION_FROM_DJI_DOCS: the topic templates and payload field
names below are a placeholder best-understanding of DJI's Cloud API
Thing Model (recurring `osd`/`state` topics per device), NOT a confirmed
contract. Before treating a real Pilot 2 / RC Pro Enterprise session as
verified, check these against DJI's published Cloud API reference and
update this module. Until then this integration is tagged
UNVERIFIED_REQUIRES_HARDWARE in docs/DJI_M3M_FIELD_TEST.md.

This client never fabricates telemetry: any field it cannot confidently
parse from the payload is left None on the resulting DroneTelemetry.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable

from cropmerge.telemetry.schemas import DroneTelemetry

log = logging.getLogger("cropmerge.telemetry.dji_cloud")

# NEEDS_VERIFICATION_FROM_DJI_DOCS
DJI_OSD_TOPIC_TEMPLATE = "thing/product/{gateway_sn}/osd"
DJI_STATE_TOPIC_TEMPLATE = "thing/product/{gateway_sn}/state"


def mqtt_available() -> bool:
    try:
        import paho.mqtt.client  # noqa: F401

        return True
    except ImportError:
        return False


def _parse_osd_payload(raw: bytes) -> DroneTelemetry | None:
    """Best-effort parse of an `osd` message into DroneTelemetry.

    NEEDS_VERIFICATION_FROM_DJI_DOCS: field names below (`latitude`,
    `longitude`, `height`/`elevation`, `attitude_head`, `gimbal_pitch`)
    are placeholders. Unknown/missing fields are left None rather than
    guessed — defensive parsing per the no-hallucination rule.
    """
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, dict):
        return None
    now_ms = int(time.time() * 1000)
    return DroneTelemetry(
        timestamp_ms=int(payload.get("timestamp", now_ms)),
        latitude=data.get("latitude"),
        longitude=data.get("longitude"),
        altitude_m=data.get("height", data.get("elevation")),
        heading_deg=data.get("attitude_head"),
        gimbal_pitch_deg=data.get("gimbal_pitch"),
        source="dji-cloud-api",
    )


class DjiCloudMqttClient:
    """Subscribes to a single gateway's OSD/state topics and forwards
    normalized telemetry to `on_telemetry`. Never raises out of its
    background thread; connection state is exposed via `is_connected()`
    so callers (health checks, UI) report what was actually observed.
    """

    def __init__(
        self,
        broker_url: str,
        gateway_sn: str,
        on_telemetry: Callable[[DroneTelemetry], None],
    ) -> None:
        self._broker_url = broker_url
        self._gateway_sn = gateway_sn
        self._on_telemetry = on_telemetry
        self._client = None
        self._connected = threading.Event()
        self._stopped = threading.Event()

    def is_connected(self) -> bool:
        return self._connected.is_set() and not self._stopped.is_set()

    def start(self) -> None:
        if not mqtt_available():
            log.warning("paho-mqtt not installed; DJI telemetry ingest disabled")
            return
        import paho.mqtt.client as mqtt
        from urllib.parse import urlparse

        parsed = urlparse(self._broker_url)
        client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)

        def on_connect(c, userdata, flags, reason_code, properties=None):
            if reason_code == 0:
                self._connected.set()
                topics = [
                    DJI_OSD_TOPIC_TEMPLATE.format(gateway_sn=self._gateway_sn),
                    DJI_STATE_TOPIC_TEMPLATE.format(gateway_sn=self._gateway_sn),
                ]
                for topic in topics:
                    c.subscribe(topic)
                log.info("DJI MQTT connected, subscribed to %s", topics)
            else:
                log.warning("DJI MQTT connect failed: reason_code=%s", reason_code)

        def on_disconnect(c, userdata, flags, reason_code, properties=None):
            self._connected.clear()

        def on_message(c, userdata, message):
            telemetry = _parse_osd_payload(message.payload)
            if telemetry is not None:
                try:
                    self._on_telemetry(telemetry)
                except Exception:
                    log.exception("on_telemetry callback failed")

        client.on_connect = on_connect
        client.on_disconnect = on_disconnect
        client.on_message = on_message
        self._client = client
        try:
            client.connect_async(parsed.hostname or "localhost", parsed.port or 1883, keepalive=30)
            client.loop_start()
        except Exception:
            log.exception("Failed to start DJI MQTT client")

    def stop(self) -> None:
        self._stopped.set()
        if self._client is not None:
            try:
                self._client.loop_stop()
                self._client.disconnect()
            except Exception:
                pass
