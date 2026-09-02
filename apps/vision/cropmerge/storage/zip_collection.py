"""Safe ZIP image-collection ingestion.

Real requirement, not a theoretical one: today's low-altitude passes collect
many individual photos rather than one continuous video, and stitching them
into a single orthomosaic just to run analysis is explicitly out of scope
(a separate photogrammetry path, per the product decision). A ZIP of loose
images needs first-class support: treat it as a COLLECTION of independent
observations, not a single file to analyze.

SECURITY, non-negotiable (an uploaded archive is untrusted input): Python's
own zipfile documentation and OWASP both call out path traversal
("Zip Slip") and decompression bombs as real upload-borne threats. This
module NEVER extracts to arbitrary paths and NEVER calls extractall() on an
untrusted archive. Every member is inspected BEFORE any bytes are read:

  - reject absolute paths, ".." traversal, Windows drive-letter paths,
    symlinks, and directory entries masquerading as files
  - reject nested archives (a .zip/.rar/.7z inside the uploaded .zip) and any
    non-image file
  - enforce configurable limits on total archive size, total uncompressed
    size, member count, per-member size, and per-member compression ratio
    (the classic "10 MB declares 500 GB uncompressed" zip-bomb pattern)

Accepted members are streamed directly into the content-addressed store
(cropmerge/storage/blob_store.py) -- never extracted to a real directory
tree that mirrors the archive's (attacker-controlled) internal paths.
"""

from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath

log = logging.getLogger("cropmerge.storage.zip_collection")

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
# Real TIFF handling would need to distinguish ordinary RGB TIFF from
# scientific/multispectral TIFF (see cropmerge/video/metadata.py's existing
# GPS/EXIF extraction) before it's safe to accept generically -- not
# implemented here, so TIFF is deliberately rejected rather than silently
# mishandled. Nested archives and every other extension are also rejected.
_NESTED_ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"}


@dataclass(frozen=True)
class ZipLimits:
    """Engineering limits, not scientifically derived -- tune from real
    field-collection sizes. Defaults sized generously for hi-res drone
    photos (a single 4000x2250 JPEG is typically a few MB) while still
    bounding worst-case memory/disk/CPU from a malicious or corrupt archive."""

    max_zip_bytes: int = 2 * 1024**3  # 2 GiB compressed archive
    max_uncompressed_bytes: int = 4 * 1024**3  # 4 GiB total across all accepted members
    max_member_count: int = 2000
    max_member_bytes: int = 200 * 1024**2  # 200 MB per image
    max_compression_ratio: float = 100.0  # uncompressed/compressed per member


@dataclass(frozen=True)
class RejectedMember:
    name: str
    reason: str


@dataclass(frozen=True)
class AcceptedMember:
    archive_path: str  # original path INSIDE the archive -- metadata only, never used as a physical path
    relative_group: str | None  # top-level directory name, if the archive used one (e.g. "variation-a/")
    compressed_size: int
    uncompressed_size: int


@dataclass
class ZipInspectionResult:
    """Mutable by design -- built up incrementally as inspect_zip() walks
    the archive's members (accepted/rejected lists are appended to, and
    total_uncompressed_bytes is only known once every member has been
    seen)."""

    accepted: list[AcceptedMember] = field(default_factory=list)
    rejected: list[RejectedMember] = field(default_factory=list)
    total_uncompressed_bytes: int = 0
    archive_rejected: str | None = None  # set when the WHOLE archive is rejected outright


def _is_safe_member_path(name: str) -> bool:
    """True only for a plain, relative, traversal-free path. Rejects
    absolute paths, ".." components, Windows drive letters, and backslash-
    disguised traversal (a name can contain literal backslashes on
    non-Windows zip creators; normalize before judging)."""
    if not name:
        return False
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        return False
    if len(normalized) >= 2 and normalized[1] == ":":  # e.g. "C:/..."
        return False
    parts = PurePosixPath(normalized).parts
    if any(part == ".." for part in parts):
        return False
    return True


def _relative_group(name: str) -> str | None:
    """Top-level directory of a member's path, if any -- e.g.
    "variation-a/img001.jpg" -> "variation-a". None for a flat archive
    member with no directory component. This is preserved as METADATA only
    (see CollectionManifest) -- never used to construct a physical path."""
    normalized = name.replace("\\", "/").strip("/")
    parts = normalized.split("/")
    return parts[0] if len(parts) > 1 else None


def inspect_zip(
    zip_bytes: bytes | io.BytesIO,
    limits: ZipLimits | None = None,
) -> ZipInspectionResult:
    """Inspect every member of an uploaded ZIP archive WITHOUT extracting
    any file content, and classify each as accepted or rejected. Never
    raises on a malformed/malicious archive -- a bad archive is reported as
    `archive_rejected`, not an exception the caller has to handle specially.
    """
    limits = limits or ZipLimits()
    buf = zip_bytes if isinstance(zip_bytes, io.BytesIO) else io.BytesIO(zip_bytes)
    archive_size = buf.getbuffer().nbytes if hasattr(buf, "getbuffer") else len(zip_bytes)  # type: ignore[arg-type]

    if archive_size > limits.max_zip_bytes:
        return ZipInspectionResult(archive_rejected=f"archive exceeds {limits.max_zip_bytes} bytes")

    try:
        zf = zipfile.ZipFile(buf)
        infos = zf.infolist()
    except zipfile.BadZipFile as exc:
        return ZipInspectionResult(archive_rejected=f"not a valid zip archive: {exc}")

    if len(infos) > limits.max_member_count:
        return ZipInspectionResult(archive_rejected=f"archive has more than {limits.max_member_count} members")

    result = ZipInspectionResult()
    total_uncompressed = 0

    for info in infos:
        name = info.filename
        is_dir = name.endswith("/") or (info.external_attr >> 16) & 0o170000 == 0o040000

        if is_dir:
            continue  # directory entries carry no content; nothing to accept or reject

        if not _is_safe_member_path(name):
            result.rejected.append(RejectedMember(name, "unsafe path (absolute, traversal, or drive letter)"))
            continue

        # zipfile represents a Unix symlink via the upper 16 bits of
        # external_attr encoding S_IFLNK (0o120000) -- reject rather than
        # ever resolve/follow it.
        unix_mode = info.external_attr >> 16
        if unix_mode and (unix_mode & 0o170000) == 0o120000:
            result.rejected.append(RejectedMember(name, "symlink entries are not allowed"))
            continue

        suffix = PurePosixPath(name.replace("\\", "/")).suffix.lower()
        if suffix in _NESTED_ARCHIVE_EXTENSIONS:
            result.rejected.append(RejectedMember(name, "nested archives are not allowed"))
            continue
        if suffix not in ALLOWED_IMAGE_EXTENSIONS:
            result.rejected.append(RejectedMember(name, f"unsupported file type: {suffix or '(none)'}"))
            continue

        if info.file_size > limits.max_member_bytes:
            result.rejected.append(RejectedMember(name, f"member exceeds {limits.max_member_bytes} bytes"))
            continue

        compressed = max(info.compress_size, 1)
        ratio = info.file_size / compressed
        if ratio > limits.max_compression_ratio:
            result.rejected.append(
                RejectedMember(name, f"compression ratio {ratio:.0f}x exceeds {limits.max_compression_ratio}x limit")
            )
            continue

        total_uncompressed += info.file_size
        if total_uncompressed > limits.max_uncompressed_bytes:
            return ZipInspectionResult(
                archive_rejected=f"total uncompressed size exceeds {limits.max_uncompressed_bytes} bytes"
            )

        result.accepted.append(
            AcceptedMember(
                archive_path=name,
                relative_group=_relative_group(name),
                compressed_size=info.compress_size,
                uncompressed_size=info.file_size,
            )
        )

    result.total_uncompressed_bytes = total_uncompressed
    return result


def extract_member_bytes(zip_bytes: bytes | io.BytesIO, member: AcceptedMember, limits: ZipLimits) -> bytes:
    """Read one already-inspected, already-accepted member's content into
    memory (bounded by limits.max_member_bytes, already enforced during
    inspect_zip()) and return the raw bytes. The caller is expected to
    immediately hash/store these bytes via the content-addressed blob store
    (see cropmerge/storage/blob_store.py) -- this function never writes to
    any filesystem path derived from the archive's own member name.
    """
    buf = zip_bytes if isinstance(zip_bytes, io.BytesIO) else io.BytesIO(zip_bytes)
    with zipfile.ZipFile(buf) as zf:
        with zf.open(member.archive_path) as f:
            data = f.read(limits.max_member_bytes + 1)
    if len(data) > limits.max_member_bytes:
        raise ValueError(
            f"member {member.archive_path!r} read more than declared size "
            f"({limits.max_member_bytes} bytes) -- possible zip-bomb/corrupt entry"
        )
    return data
