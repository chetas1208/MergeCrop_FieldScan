"""Logical-object -> blob manifest with reference counting.

A "logical object" is whatever the application cares about (e.g. an
upload id, a run's source-frame id, a derived overlay id). Many logical
objects can point at the same content-addressed blob (e.g. two uploads of
byte-identical footage, or a re-run that produces an identical derived
artifact). This module tracks that mapping and makes sure a blob is only
ever deleted once nothing references it any more.

Reference counts are never stored as a mutable counter that could drift —
they are always computed as `COUNT(*)` over the objects table, so there is
no bookkeeping to get out of sync with reality. SQLite gives us
transactional (crash-safe) commits of the mapping table itself; the
manifest only ever asks the blob store to physically delete a blob *after*
committing the mapping change that dropped its last reference, so a crash
between those two steps can, at worst, leave an orphaned-but-harmless blob
on disk (recoverable by a future GC pass) — never a dangling reference to a
blob that no longer exists.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from cropmerge.storage.blob_store import BlobRef, BlobStore

_SCHEMA = """
CREATE TABLE IF NOT EXISTS objects (
    logical_id     TEXT PRIMARY KEY,
    sha256         TEXT NOT NULL,
    size_bytes     INTEGER NOT NULL,
    storage_class  TEXT,
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_objects_sha256 ON objects(sha256);
"""


@dataclass(frozen=True)
class ManifestEntry:
    logical_id: str
    sha256: str
    size_bytes: int
    storage_class: str | None
    created_at: str


class Manifest:
    """Crash-safe SQLite-backed logical-object -> blob mapping."""

    def __init__(self, db_path: str | Path, blob_store: BlobStore | None = None) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.blob_store = blob_store
        self._conn = sqlite3.connect(str(self.db_path), isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        # WAL + synchronous=FULL: durable commits, and readers never see a
        # torn/partial write — the manifest is metadata for scientific
        # evidence, so we favor durability over the small extra fsync cost.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._conn:
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Manifest:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writes -----------------------------------------------------------

    def add(
        self,
        logical_id: str,
        blob_ref: BlobRef,
        storage_class: str | None = None,
    ) -> ManifestEntry:
        """Record (or replace) the mapping from `logical_id` to `blob_ref`.

        If `logical_id` already mapped to a different blob, the old
        mapping is atomically replaced and, if that was the old blob's
        last reference, the old blob is deleted (only if a `blob_store`
        was provided).
        """
        created_at = datetime.now(timezone.utc).isoformat()
        old_sha256: str | None = None
        with self._conn:
            row = self._conn.execute(
                "SELECT sha256 FROM objects WHERE logical_id = ?", (logical_id,)
            ).fetchone()
            if row is not None:
                old_sha256 = row["sha256"]
            self._conn.execute(
                """
                INSERT INTO objects (logical_id, sha256, size_bytes, storage_class, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(logical_id) DO UPDATE SET
                    sha256=excluded.sha256,
                    size_bytes=excluded.size_bytes,
                    storage_class=excluded.storage_class,
                    created_at=excluded.created_at
                """,
                (logical_id, blob_ref.sha256, blob_ref.size_bytes, storage_class, created_at),
            )

        if old_sha256 is not None and old_sha256 != blob_ref.sha256:
            self._maybe_delete_blob(old_sha256)

        return ManifestEntry(
            logical_id=logical_id,
            sha256=blob_ref.sha256,
            size_bytes=blob_ref.size_bytes,
            storage_class=storage_class,
            created_at=created_at,
        )

    def remove(self, logical_id: str) -> bool:
        """Remove a logical object's mapping.

        Returns True if a mapping existed and was removed. If that was the
        referenced blob's last remaining reference, the blob itself is
        deleted (only if a `blob_store` was provided to this Manifest) —
        never before the mapping deletion is durably committed.
        """
        with self._conn:
            row = self._conn.execute(
                "SELECT sha256 FROM objects WHERE logical_id = ?", (logical_id,)
            ).fetchone()
            if row is None:
                return False
            sha256 = row["sha256"]
            self._conn.execute("DELETE FROM objects WHERE logical_id = ?", (logical_id,))

        self._maybe_delete_blob(sha256)
        return True

    def _maybe_delete_blob(self, sha256: str) -> None:
        if self.ref_count(sha256) == 0 and self.blob_store is not None:
            self.blob_store.delete(sha256)

    # -- reads --------------------------------------------------------------

    def get(self, logical_id: str) -> ManifestEntry | None:
        row = self._conn.execute(
            "SELECT * FROM objects WHERE logical_id = ?", (logical_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_entry(row)

    def ref_count(self, sha256: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM objects WHERE sha256 = ?", (sha256,)
        ).fetchone()
        return int(row["n"])

    def list_by_blob(self, sha256: str) -> list[ManifestEntry]:
        rows = self._conn.execute(
            "SELECT * FROM objects WHERE sha256 = ? ORDER BY created_at", (sha256,)
        ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def list_all(self) -> list[ManifestEntry]:
        rows = self._conn.execute("SELECT * FROM objects ORDER BY created_at").fetchall()
        return [_row_to_entry(r) for r in rows]


def _row_to_entry(row: sqlite3.Row) -> ManifestEntry:
    return ManifestEntry(
        logical_id=row["logical_id"],
        sha256=row["sha256"],
        size_bytes=row["size_bytes"],
        storage_class=row["storage_class"],
        created_at=row["created_at"],
    )
