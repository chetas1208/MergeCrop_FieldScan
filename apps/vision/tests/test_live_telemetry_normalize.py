from cropmerge.telemetry.normalize import to_frame_pose
from cropmerge.telemetry.schemas import DroneTelemetry


def test_to_frame_pose_full():
    telemetry = DroneTelemetry(
        timestamp_ms=2500,
        latitude=40.1,
        longitude=-88.2,
        altitude_m=41.2,
        heading_deg=87.3,
        gimbal_pitch_deg=-10.0,
        source="dji-cloud-api",
    )
    pose = to_frame_pose(telemetry)
    assert pose.timestamp_sec == 2.5
    assert pose.latitude == 40.1
    assert pose.longitude == -88.2
    assert pose.altitude == 41.2
    assert pose.heading == 87.3
    assert pose.gimbal_pitch == -10.0


def test_to_frame_pose_missing_fields_stay_none():
    telemetry = DroneTelemetry(timestamp_ms=1000)
    pose = to_frame_pose(telemetry)
    assert pose.latitude is None
    assert pose.longitude is None
    assert pose.altitude is None
    assert pose.heading is None
    assert pose.gimbal_pitch is None
