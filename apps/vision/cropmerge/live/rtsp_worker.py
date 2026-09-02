from __future__ import annotations

import logging
import threading
import time

import cv2
import numpy as np

from cropmerge.features.dinov3 import create_embedder
from cropmerge.live.hub import LiveEventHub
from cropmerge.live.rolling_zones import RollingZoneAggregator
from cropmerge.live.schemas import FrameAnalyzedEvent, LiveInspectionAreaEvent, LiveSessionStatus, StateChangeEvent
from cropmerge.live.session_store import LiveSessionStore
from cropmerge.pipeline.frame_analysis import analyze_single_frame
from cropmerge.segmentation import create_segmenter
from cropmerge.telemetry.association import nearest_telemetry

log = logging.getLogger("cropmerge.live.rtsp_worker")

RECONNECT_BACKOFF_SEC = 2.0
STREAM_LOST_AFTER_SEC = 5.0


class LiveRtspWorker:
    """Consumes MediaMTX's RTSP output for one session at a configurable
    sample rate, reusing the exact recorded-mode per-frame analysis
    (analyze_single_frame) and temporal consensus (RollingZoneAggregator).

    Backpressure by design: the grabber thread keeps only the single newest
    captured frame; the analyzer thread always works on whatever is
    currently latest and never accumulates a queue.
    """

    def __init__(
        self,
        session_id: str,
        rtsp_url: str,
        sample_fps: float,
        cfg: dict,
        hub: LiveEventHub,
        store: LiveSessionStore,
        segmentation_backend: str = "heuristic",
        dino_backend: str = "heuristic",
        grid_rows: int = 8,
        grid_cols: int = 8,
    ) -> None:
        self.session_id = session_id
        self.rtsp_url = rtsp_url
        self.sample_fps = max(0.1, min(sample_fps, 5.0))
        self.cfg = cfg
        self.hub = hub
        self.store = store
        self.segmentation_backend = segmentation_backend
        self.dino_backend = dino_backend
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols

        self._latest_frame: np.ndarray | None = None
        self._latest_ts: float = 0.0
        self._last_processed_ts: float = 0.0
        self._frame_lock = threading.Lock()
        self._stop = threading.Event()
        self._grabber_thread: threading.Thread | None = None
        self._analyzer_thread: threading.Thread | None = None
        self._aggregator: RollingZoneAggregator | None = None
        self._status = LiveSessionStatus.CONNECTING

    def _set_status(self, status: LiveSessionStatus) -> None:
        if status == self._status:
            return
        self._status = status
        self.store.update_status(self.session_id, status)
        self.hub.publish_threadsafe(
            self.session_id, StateChangeEvent(session_id=self.session_id, status=status).to_camel_dict()
        )

    def start(self) -> None:
        self._set_status(LiveSessionStatus.CONNECTING)
        self._grabber_thread = threading.Thread(target=self._grab_loop, daemon=True, name=f"live-grab-{self.session_id}")
        self._analyzer_thread = threading.Thread(target=self._analyze_loop, daemon=True, name=f"live-analyze-{self.session_id}")
        self._grabber_thread.start()
        self._analyzer_thread.start()

    def stop(self) -> None:
        self._stop.set()
        for thread in (self._grabber_thread, self._analyzer_thread):
            if thread is not None:
                thread.join(timeout=5.0)
        self._set_status(LiveSessionStatus.ENDED)

    def _grab_loop(self) -> None:
        last_success = time.monotonic()
        cap: cv2.VideoCapture | None = None
        while not self._stop.is_set():
            if cap is None:
                cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    cap.release()
                    cap = None
                    time.sleep(RECONNECT_BACKOFF_SEC)
                    if time.monotonic() - last_success > STREAM_LOST_AFTER_SEC:
                        self._set_status(LiveSessionStatus.RECONNECTING if self._status != LiveSessionStatus.CONNECTING else LiveSessionStatus.CONNECTING)
                    continue
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                cap = None
                if time.monotonic() - last_success > STREAM_LOST_AFTER_SEC:
                    self._set_status(LiveSessionStatus.STREAM_LOST)
                time.sleep(RECONNECT_BACKOFF_SEC)
                continue
            last_success = time.monotonic()
            if self._status in (LiveSessionStatus.CONNECTING, LiveSessionStatus.STREAM_LOST, LiveSessionStatus.RECONNECTING):
                self._set_status(LiveSessionStatus.LIVE)
            with self._frame_lock:
                self._latest_frame = frame
                self._latest_ts = time.time()
        if cap is not None:
            cap.release()

    def _analyze_loop(self) -> None:
        interval = 1.0 / self.sample_fps
        segmenter = create_segmenter(self.segmentation_backend, self.cfg, allow_fallback=True)
        embedder = create_embedder(self.dino_backend, allow_fallback=True)
        frame_index = 0
        while not self._stop.is_set():
            time.sleep(interval)
            with self._frame_lock:
                frame = self._latest_frame
                frame_ts = self._latest_ts
            if frame is None or frame_ts == self._last_processed_ts:
                continue
            self._last_processed_ts = frame_ts
            try:
                self._analyze_frame(segmenter, embedder, frame, frame_ts, frame_index)
            except Exception:
                log.exception("Live frame analysis failed for session=%s", self.session_id)
            frame_index += 1

    def _analyze_frame(self, segmenter, embedder, bgr: np.ndarray, frame_ts: float, frame_index: int) -> None:
        h, w = bgr.shape[:2]
        seg = segmenter.segment_image(bgr, frame_index, frame_ts)
        field = seg.field_mask if seg.field_mask is not None else np.zeros((h, w), dtype=bool)
        label = seg.label_map if seg.label_map is not None else np.full((h, w), "UNKNOWN", dtype=object)

        if self._aggregator is None:
            self._aggregator = RollingZoneAggregator(self.cfg, (h, w))

        cells, _heat, _sres = analyze_single_frame(
            bgr, field, label, seg.crop_mask, embedder, self.cfg, self.grid_rows, self.grid_cols
        )
        self._aggregator.add_frame(cells, frame_ts, quality_weight=1.0)
        zones = self._aggregator.current_zones()

        frame_ts_ms = int(frame_ts * 1000)
        live_zones = []
        for zone in zones:
            association = nearest_telemetry(self.store, self.session_id, frame_ts_ms)
            live_zones.append(
                LiveInspectionAreaEvent(
                    **zone.model_dump(),
                    live_frame_timestamp_ms=frame_ts_ms,
                    telemetry_association=association,
                )
            )
        event = FrameAnalyzedEvent(session_id=self.session_id, frame_timestamp_ms=frame_ts_ms, zones=live_zones)
        self.hub.publish_threadsafe(self.session_id, event.to_camel_dict())
