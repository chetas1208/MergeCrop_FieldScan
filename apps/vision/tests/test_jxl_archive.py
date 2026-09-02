from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import ExifTags, Image

from cropmerge.storage.jxl_archive import (
    archive_jpeg_reversible,
    jxl_available,
    reconstruct_jpeg_bytes,
)
from cropmerge.video.metadata import extract_metadata

pytestmark = pytest.mark.skipif(not jxl_available(), reason="pillow-jxl-plugin not installed")


def _write_jpeg(path: Path, *, with_gps: bool = False, size: tuple[int, int] = (64, 64)) -> None:
    rng = np.random.default_rng(0)
    arr = (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")
    if with_gps:
        exif = Image.Exif()
        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
        gps_ifd[1] = "N"
        gps_ifd[2] = (40.0, 7.0, 24.0)
        gps_ifd[3] = "W"
        gps_ifd[4] = (88.0, 34.0, 44.0)
        gps_ifd[5] = 0
        gps_ifd[6] = 41.2
        exif[ExifTags.IFD.GPSInfo] = gps_ifd
        img.save(path, format="JPEG", quality=88, exif=exif.tobytes())
    else:
        img.save(path, format="JPEG", quality=88)


def test_archive_and_reconstruct_is_byte_exact(tmp_path: Path):
    jpeg_path = tmp_path / "photo.jpg"
    jxl_path = tmp_path / "photo.jxl"
    _write_jpeg(jpeg_path)
    original_bytes = jpeg_path.read_bytes()

    result = archive_jpeg_reversible(jpeg_path, jxl_path)

    assert result.verified is True
    assert result.reason is None
    assert result.original_sha256 == hashlib.sha256(original_bytes).hexdigest()
    assert jxl_path.is_file()
    assert result.jxl_bytes == jxl_path.stat().st_size

    reconstructed = reconstruct_jpeg_bytes(jxl_path)
    assert reconstructed == original_bytes


def test_archive_preserves_gps_exif_exactly(tmp_path: Path):
    """The full-bitstream hash match already proves this, but verify the
    actually-decoded GPS fields explicitly too, per the campaign's
    'verify explicitly during tests' requirement."""
    jpeg_path = tmp_path / "photo.jpg"
    jxl_path = tmp_path / "photo.jxl"
    _write_jpeg(jpeg_path, with_gps=True)

    result = archive_jpeg_reversible(jpeg_path, jxl_path)
    assert result.verified is True

    reconstructed = reconstruct_jpeg_bytes(jxl_path)
    recon_path = tmp_path / "reconstructed.jpg"
    recon_path.write_bytes(reconstructed)

    original_meta = extract_metadata(jpeg_path)
    recon_meta = extract_metadata(recon_path)
    assert recon_meta.gps == original_meta.gps
    assert original_meta.gps is not None  # sanity: the fixture actually has GPS


def test_archive_rejects_non_jpeg_input(tmp_path: Path):
    png_path = tmp_path / "photo.png"
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8), "RGB").save(png_path, format="PNG")
    jxl_path = tmp_path / "photo.jxl"

    result = archive_jpeg_reversible(png_path, jxl_path)

    assert result.verified is False
    assert "not a JPEG" in (result.reason or "")
    assert not jxl_path.exists()


def test_archive_leaves_no_partial_file_on_failure(tmp_path: Path, monkeypatch):
    """Simulate a genuine reconstruction mismatch (corrupted/incompatible
    decode) by faking the Decoder's return value, and confirm the archive is
    rejected and cleaned up rather than left as a silently-wrong artifact."""
    jpeg_path = tmp_path / "photo.jpg"
    jxl_path = tmp_path / "photo.jxl"
    _write_jpeg(jpeg_path)

    import pillow_jxl

    class _MismatchDecoder:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, blob):
            corrupted = jpeg_path.read_bytes() + b"\x00"  # one extra byte -> hash mismatch
            return True, None, corrupted, None, []

    monkeypatch.setattr(pillow_jxl, "Decoder", _MismatchDecoder)

    from cropmerge.storage.jxl_archive import archive_jpeg_reversible as archive_fn

    result = archive_fn(jpeg_path, jxl_path)

    assert result.verified is False
    assert result.reason == "reconstructed JPEG hash mismatch"
    assert not jxl_path.exists()
    assert jpeg_path.exists()  # original always untouched
