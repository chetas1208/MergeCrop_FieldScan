"""Content-addressed blob storage.

Blobs are stored under:

    <root>/<sha256[:2]>/<sha256[2:4]>/<sha256>

Writes are atomic: content is streamed into a temp file inside the *same*
root directory (so it is guaranteed to be on the same filesystem as the
final destination), hashed as it is written, verified, and only then moved
into place with `os.replace()`. `os.replace()` is atomic on POSIX and
Windows for paths on the same filesystem, so a reader can never observe a
partially-written blob at its final path.

Deduplication is automatic: if a blob with the computed hash already
exists, the temp file is discarded instead of overwriting the existing
(byte-identical, by definition of content addressing) blob.

This module does not decide *when* to delete a blob — see manifest.py for
reference counting. `BlobStore.delete()` is a low-level, unconditional
removal that the manifest calls once it determines a blob has zero
remaining references.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from cropmerge.storage.hashing import DEFAULT_CHUNK_SIZE, is_valid_sha256_hex


class InvalidHashError(ValueError):
    """Raised when a hash string is not a well-formed SHA-256 hex digest.

    Guards against path traversal: nothing derived from an unvalidated
    string is ever joined into a filesystem path.
    """


@dataclass(frozen=True)
class BlobRef:
    """Reference to a stored blob: its content hash and size in bytes."""

    sha256: str
    size_bytes: int


class BlobStore:
    """Content-addressed blob storage rooted at `root`."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        # Temp files live under the same root so os.replace() is guaranteed
        # to be a same-filesystem (atomic) rename, not a cross-device copy.
        self._tmp_dir = self.root / ".tmp"
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, sha256: str) -> Path:
        """Return the on-disk path for a given content hash (without checking existence)."""
        if not is_valid_sha256_hex(sha256):
            raise InvalidHashError(f"not a valid sha256 hex digest: {sha256!r}")
        return self.root / sha256[:2] / sha256[2:4] / sha256

    def exists(self, sha256: str) -> bool:
        return self.path_for(sha256).is_file()

    def get(self, sha256: str) -> Path:
        """Return the path to a stored blob. Raises FileNotFoundError if absent."""
        path = self.path_for(sha256)
        if not path.is_file():
            raise FileNotFoundError(f"blob not found: {sha256}")
        return path

    def put(self, path_or_stream: str | Path | BinaryIO, chunk_size: int = DEFAULT_CHUNK_SIZE) -> BlobRef:
        """Store content and return its BlobRef.

        Accepts either a path to a file on disk or a binary file-like
        object. Content is streamed in `chunk_size` pieces so peak memory
        stays bounded regardless of input size. If a blob with the
        resulting hash already exists, the newly written temp file is
        discarded (no-op dedup) rather than replacing the existing blob.
        """
        if isinstance(path_or_stream, (str, Path)):
            with Path(path_or_stream).open("rb") as f:
                return self._put_stream(f, chunk_size)
        return self._put_stream(path_or_stream, chunk_size)

    def _put_stream(self, stream: BinaryIO, chunk_size: int) -> BlobRef:
        fd, tmp_name = tempfile.mkstemp(dir=self._tmp_dir, prefix="blob-")
        tmp_path = Path(tmp_name)
        try:
            digest = hashlib.sha256()
            size = 0
            with os.fdopen(fd, "wb") as tmp_file:
                while True:
                    chunk = stream.read(chunk_size)
                    if not chunk:
                        break
                    digest.update(chunk)
                    size += len(chunk)
                    tmp_file.write(chunk)
                tmp_file.flush()
                os.fsync(tmp_file.fileno())

            sha256 = digest.hexdigest()
            final_path = self.path_for(sha256)

            if final_path.is_file():
                # Dedup: identical content already stored. Discard temp copy.
                tmp_path.unlink(missing_ok=True)
                return BlobRef(sha256=sha256, size_bytes=size)

            final_path.parent.mkdir(parents=True, exist_ok=True)
            # Atomic on POSIX/Windows for same-filesystem paths: a concurrent
            # reader either sees the old state (nothing) or the fully
            # written file, never a partial one.
            os.replace(tmp_path, final_path)
            # Make blobs read-only-ish by convention (best-effort; not
            # security-critical, just guards against accidental in-place edits).
            try:
                os.chmod(final_path, 0o444)
            except OSError:
                pass
            return BlobRef(sha256=sha256, size_bytes=size)
        finally:
            # If we crashed/raised before os.replace(), clean up the temp
            # file so no stray partial data lingers. missing_ok handles the
            # already-moved / already-deduped case.
            tmp_path.unlink(missing_ok=True)

    def delete(self, sha256: str) -> bool:
        """Unconditionally remove a blob. Returns True if a file was removed.

        Callers (the manifest) are responsible for only calling this once
        no logical reference to the blob remains.
        """
        path = self.path_for(sha256)
        if not path.is_file():
            return False
        # Blobs are written read-only; allow removal regardless.
        try:
            os.chmod(path, 0o644)
        except OSError:
            pass
        path.unlink()
        # Best-effort cleanup of now-empty shard directories.
        for parent in (path.parent, path.parent.parent):
            try:
                if parent != self.root and not any(parent.iterdir()):
                    parent.rmdir()
            except OSError:
                pass
        return True

    def copy_out(self, sha256: str, destination: str | Path) -> Path:
        """Convenience: copy a stored blob to `destination` (for callers that need a mutable copy)."""
        src = self.get(sha256)
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        return dest
