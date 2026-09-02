"""Reversible JPEG -> JPEG XL archival.

Uses pillow-jxl-plugin (pure Rust libjxl bindings shipped as a wheel, no
system cjxl/djxl binary or apt package required — verified working in this
environment without sudo). This is the REVERSIBLE_SOURCE storage class from
cropmerge/storage/policies.py: JPEG XL's lossless-JPEG-reconstruction mode
can losslessly re-encode an existing JPEG and later reconstruct the exact
original JPEG bitstream byte-for-byte (verified: `docs/STORAGE_COMPRESSION_RESEARCH.md`).

Hard rule (matches the storage campaign's "lossy source deletion rule"):
an archive is only ever considered valid after round-tripping it and
confirming SHA256(reconstructed) == SHA256(original). If that check fails
for any reason, the JXL file is deleted and the caller is told to keep the
original — never delete or replace a source on an unverified archive.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("cropmerge.storage.jxl_archive")


def jxl_available() -> bool:
    try:
        import pillow_jxl  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass(frozen=True)
class JxlArchiveResult:
    verified: bool
    original_sha256: str
    original_bytes: int
    jxl_bytes: int | None
    savings_ratio: float  # 1 - jxl_bytes/original_bytes; 0.0 when not verified
    reason: str | None = None  # set when verified is False


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def archive_jpeg_reversible(jpeg_path: Path, jxl_path: Path) -> JxlArchiveResult:
    """Losslessly re-encode `jpeg_path` as JPEG XL at `jxl_path`, then
    immediately verify the round-trip. On any failure (missing dependency,
    encode error, or a reconstructed-bytes mismatch) the partial/incorrect
    `jxl_path` is removed and verified=False is returned — the caller must
    treat that as "keep the original JPEG, do not archive," never as a
    partial success.

    Because SHA256(reconstructed) == SHA256(original) is a full-bitstream
    comparison, it inherently covers EXIF/GPS/orientation/timestamp — those
    bytes are part of the JPEG bitstream being compared, so a passing check
    already proves metadata survived exactly (a dedicated GPS-field test in
    tests/test_jxl_archive.py additionally confirms this explicitly rather
    than only asserting it here).
    """
    jpeg_path = Path(jpeg_path)
    jxl_path = Path(jxl_path)
    original_bytes = jpeg_path.read_bytes()
    original_sha256 = _sha256_bytes(original_bytes)

    if not jxl_available():
        return JxlArchiveResult(
            verified=False,
            original_sha256=original_sha256,
            original_bytes=len(original_bytes),
            jxl_bytes=None,
            savings_ratio=0.0,
            reason="pillow_jxl not available",
        )

    try:
        from PIL import Image
        from pillow_jxl import Decoder

        with Image.open(jpeg_path) as im:
            if im.format != "JPEG":
                return JxlArchiveResult(
                    verified=False,
                    original_sha256=original_sha256,
                    original_bytes=len(original_bytes),
                    jxl_bytes=None,
                    savings_ratio=0.0,
                    reason=f"not a JPEG (format={im.format!r})",
                )
            jxl_path.parent.mkdir(parents=True, exist_ok=True)
            # lossless_jpeg=True explicitly selects the JPEG-bitstream
            # reconstruction encode path (requires im.format == "JPEG" and
            # im.filename set, i.e. Image.open(path) not a BytesIO) --
            # verified in a manual round-trip test before this module was
            # written. Explicit rather than relying on the library's
            # "None means auto-detect and warn" default.
            im.save(jxl_path, format="JXL", lossless_jpeg=True)

        jxl_bytes_blob = jxl_path.read_bytes()
        decoder = Decoder(num_threads=-1)
        is_jpeg_reconstruction, _info, reconstructed, _icc, _boxes = decoder(jxl_bytes_blob)

        if not is_jpeg_reconstruction or reconstructed is None:
            jxl_path.unlink(missing_ok=True)
            return JxlArchiveResult(
                verified=False,
                original_sha256=original_sha256,
                original_bytes=len(original_bytes),
                jxl_bytes=None,
                savings_ratio=0.0,
                reason="JXL archive did not encode as a reconstructible JPEG",
            )

        if _sha256_bytes(reconstructed) != original_sha256:
            jxl_path.unlink(missing_ok=True)
            return JxlArchiveResult(
                verified=False,
                original_sha256=original_sha256,
                original_bytes=len(original_bytes),
                jxl_bytes=None,
                savings_ratio=0.0,
                reason="reconstructed JPEG hash mismatch",
            )

        savings = 1.0 - (len(jxl_bytes_blob) / len(original_bytes)) if original_bytes else 0.0
        return JxlArchiveResult(
            verified=True,
            original_sha256=original_sha256,
            original_bytes=len(original_bytes),
            jxl_bytes=len(jxl_bytes_blob),
            savings_ratio=savings,
        )
    except Exception as exc:
        log.exception("JXL reversible archival failed for %s (original kept)", jpeg_path)
        jxl_path.unlink(missing_ok=True)
        return JxlArchiveResult(
            verified=False,
            original_sha256=original_sha256,
            original_bytes=len(original_bytes),
            jxl_bytes=None,
            savings_ratio=0.0,
            reason=str(exc),
        )


def reconstruct_jpeg_bytes(jxl_path: Path) -> bytes:
    """Materialize the exact original JPEG bytes from a verified reversible
    archive. Raises if the archive isn't a JPEG-reconstruction JXL (callers
    should only ever call this on archives that archive_jpeg_reversible()
    already verified)."""
    from pillow_jxl import Decoder

    decoder = Decoder(num_threads=-1)
    is_jpeg_reconstruction, _info, reconstructed, _icc, _boxes = decoder(Path(jxl_path).read_bytes())
    if not is_jpeg_reconstruction or reconstructed is None:
        raise ValueError(f"{jxl_path} is not a JPEG-reconstruction JXL archive")
    return reconstructed
