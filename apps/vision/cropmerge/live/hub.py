from __future__ import annotations

import asyncio
import logging
import threading

log = logging.getLogger("cropmerge.live.hub")


class LiveEventHub:
    """Per-session WebSocket event fan-out. `publish_threadsafe` is safe to
    call from the RTSP worker's background threads (not the asyncio loop);
    it hands the event to the loop via call_soon_threadsafe.
    """

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queues: dict[str, list[asyncio.Queue]] = {}
        self._lock = threading.Lock()

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, session_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        with self._lock:
            self._queues.setdefault(session_id, []).append(queue)
        return queue

    def unsubscribe(self, session_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            queues = self._queues.get(session_id, [])
            if queue in queues:
                queues.remove(queue)
            if not queues:
                self._queues.pop(session_id, None)

    def _publish(self, session_id: str, event: dict) -> None:
        with self._lock:
            queues = list(self._queues.get(session_id, []))
        for queue in queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                log.warning("Dropping live event for session=%s: subscriber queue full", session_id)

    def publish_threadsafe(self, session_id: str, event: dict) -> None:
        if self._loop is None:
            log.warning("LiveEventHub loop not bound; dropping event")
            return
        self._loop.call_soon_threadsafe(self._publish, session_id, event)
