from cropmerge.telemetry.association import nearest_telemetry
from cropmerge.telemetry.schemas import TelemetryAvailability


class _FakeStore:
    def __init__(self, row=None):
        self._row = row

    def nearest_telemetry_row(self, session_id, frame_ts_ms):
        return self._row


def _row(timestamp_ms):
    return {
        "timestamp_ms": timestamp_ms,
        "latitude": 40.0,
        "longitude": -88.0,
        "altitude_m": 41.0,
        "heading_deg": 90.0,
        "gimbal_pitch_deg": -5.0,
    }


def test_no_telemetry_at_all_is_unavailable():
    association = nearest_telemetry(_FakeStore(row=None), "s1", 1000)
    assert association.availability == TelemetryAvailability.UNAVAILABLE
    assert association.telemetry is None


def test_within_exact_window():
    association = nearest_telemetry(_FakeStore(row=_row(1100)), "s1", 1000, exact_ms=250, tolerance_ms=2000)
    assert association.availability == TelemetryAvailability.EXACT
    assert association.telemetry is not None
    assert association.telemetry.latitude == 40.0


def test_within_tolerance_but_not_exact_is_nearby():
    association = nearest_telemetry(_FakeStore(row=_row(1500)), "s1", 1000, exact_ms=250, tolerance_ms=2000)
    assert association.availability == TelemetryAvailability.NEARBY


def test_outside_tolerance_is_unavailable_and_never_fabricated():
    association = nearest_telemetry(_FakeStore(row=_row(9000)), "s1", 1000, exact_ms=250, tolerance_ms=2000)
    assert association.availability == TelemetryAvailability.UNAVAILABLE
    assert association.telemetry is None
