from cropmerge.config import load_config
from cropmerge.live.rtsp_worker import LiveRtspWorker
from cropmerge.live.schemas import LiveSessionStatus


class _FakeStore:
    def __init__(self):
        self.status_calls = []

    def update_status(self, session_id, status, ended=False):
        self.status_calls.append((session_id, status, ended))


class _FakeHub:
    def __init__(self):
        self.published = []

    def publish_threadsafe(self, session_id, event):
        self.published.append((session_id, event))


def _worker() -> tuple[LiveRtspWorker, _FakeStore, _FakeHub]:
    store = _FakeStore()
    hub = _FakeHub()
    worker = LiveRtspWorker(
        session_id="s1",
        rtsp_url="rtsp://unused/test",
        sample_fps=2.0,
        cfg=load_config(),
        hub=hub,
        store=store,
    )
    return worker, store, hub


def test_set_status_publishes_state_change_event():
    worker, store, hub = _worker()
    worker._set_status(LiveSessionStatus.LIVE)
    assert store.status_calls == [("s1", LiveSessionStatus.LIVE, False)]
    assert len(hub.published) == 1
    session_id, event = hub.published[0]
    assert session_id == "s1"
    assert event == {"type": "state_change", "sessionId": "s1", "status": "live"}


def test_set_status_no_op_when_already_in_that_state():
    # LiveRtspWorker starts in CONNECTING (see __init__), so re-asserting
    # the same status it's already in must not re-publish or re-write.
    worker, store, hub = _worker()
    worker._set_status(LiveSessionStatus.CONNECTING)
    assert store.status_calls == []
    assert hub.published == []


def test_set_status_is_idempotent_for_repeated_status():
    worker, store, hub = _worker()
    worker._set_status(LiveSessionStatus.LIVE)
    worker._set_status(LiveSessionStatus.LIVE)
    assert len(store.status_calls) == 1
    assert len(hub.published) == 1


def test_set_status_transition_sequence():
    worker, store, hub = _worker()
    # Worker already starts in CONNECTING; only transitions away from it count.
    worker._set_status(LiveSessionStatus.LIVE)
    worker._set_status(LiveSessionStatus.STREAM_LOST)
    worker._set_status(LiveSessionStatus.RECONNECTING)
    worker._set_status(LiveSessionStatus.LIVE)
    statuses = [call[1] for call in store.status_calls]
    assert statuses == [
        LiveSessionStatus.LIVE,
        LiveSessionStatus.STREAM_LOST,
        LiveSessionStatus.RECONNECTING,
        LiveSessionStatus.LIVE,
    ]
