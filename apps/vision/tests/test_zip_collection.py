from __future__ import annotations

import io
import zipfile

import pytest

from cropmerge.storage.zip_collection import (
    ZipLimits,
    extract_member_bytes,
    inspect_zip,
)


def _make_zip(entries: dict[str, bytes], *, zip64: bool = False) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, allowZip64=zip64) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _fake_jpeg_bytes(n: int = 200) -> bytes:
    return b"\xff\xd8\xff\xe0" + bytes(n)  # real-ish JPEG magic bytes + padding


def test_accepts_ordinary_flat_images():
    zip_bytes = _make_zip({"a.jpg": _fake_jpeg_bytes(), "b.png": _fake_jpeg_bytes()})

    result = inspect_zip(zip_bytes)

    assert result.archive_rejected is None
    assert {m.archive_path for m in result.accepted} == {"a.jpg", "b.png"}
    assert all(m.relative_group is None for m in result.accepted)
    assert result.rejected == []


def test_preserves_group_from_top_level_folders():
    zip_bytes = _make_zip(
        {
            "variation-a/img001.jpg": _fake_jpeg_bytes(),
            "variation-b/img001.jpg": _fake_jpeg_bytes(),
        }
    )

    result = inspect_zip(zip_bytes)

    groups = {m.archive_path: m.relative_group for m in result.accepted}
    assert groups == {"variation-a/img001.jpg": "variation-a", "variation-b/img001.jpg": "variation-b"}


def test_directory_entries_are_skipped_not_rejected():
    zip_bytes = _make_zip({"variation-a/": b"", "variation-a/img001.jpg": _fake_jpeg_bytes()})

    result = inspect_zip(zip_bytes)

    assert len(result.accepted) == 1
    assert result.rejected == []


def test_rejects_path_traversal():
    zip_bytes = _make_zip({"../../etc/passwd": b"malicious"})

    result = inspect_zip(zip_bytes)

    assert result.accepted == []
    assert len(result.rejected) == 1
    assert "unsafe path" in result.rejected[0].reason


def test_rejects_absolute_path():
    zip_bytes = _make_zip({"/etc/passwd": b"malicious"})

    result = inspect_zip(zip_bytes)

    assert result.accepted == []
    assert "unsafe path" in result.rejected[0].reason


def test_rejects_windows_drive_letter_path():
    zip_bytes = _make_zip({"C:/Windows/System32/evil.jpg": _fake_jpeg_bytes()})

    result = inspect_zip(zip_bytes)

    assert result.accepted == []
    assert "unsafe path" in result.rejected[0].reason


def test_rejects_nested_archive():
    zip_bytes = _make_zip({"inner.zip": b"PK\x03\x04fake"})

    result = inspect_zip(zip_bytes)

    assert result.accepted == []
    assert "nested archive" in result.rejected[0].reason


def test_rejects_unsupported_file_type():
    zip_bytes = _make_zip({"notes.txt": b"hello"})

    result = inspect_zip(zip_bytes)

    assert result.accepted == []
    assert "unsupported file type" in result.rejected[0].reason


def test_rejects_tiff_deliberately_not_silently_mishandled():
    """TIFF could be ordinary RGB or scientific/multispectral -- this module
    doesn't yet distinguish them, so it rejects rather than guessing."""
    zip_bytes = _make_zip({"scan.tiff": b"fake tiff bytes"})

    result = inspect_zip(zip_bytes)

    assert result.accepted == []
    assert "unsupported file type" in result.rejected[0].reason


def test_rejects_member_exceeding_max_size():
    zip_bytes = _make_zip({"huge.jpg": _fake_jpeg_bytes(1000)})
    limits = ZipLimits(max_member_bytes=500)

    result = inspect_zip(zip_bytes, limits)

    assert result.accepted == []
    assert "exceeds" in result.rejected[0].reason


def test_rejects_zip_bomb_style_compression_ratio():
    """A member declaring far more uncompressed bytes than its compressed
    size suggests -- the classic zip-bomb signature -- must be rejected
    based on the DECLARED ratio, without ever inflating the data."""
    # Highly compressible content produces a real high ratio without a
    # hand-crafted malicious CRC.
    zip_bytes = _make_zip({"suspicious.jpg": b"\x00" * 100_000})
    limits = ZipLimits(max_compression_ratio=10.0)

    result = inspect_zip(zip_bytes, limits)

    assert result.accepted == []
    assert "compression ratio" in result.rejected[0].reason


def test_rejects_archive_exceeding_total_uncompressed_budget():
    zip_bytes = _make_zip({"a.jpg": _fake_jpeg_bytes(1000), "b.jpg": _fake_jpeg_bytes(1000)})
    limits = ZipLimits(max_uncompressed_bytes=1500, max_compression_ratio=1000.0)

    result = inspect_zip(zip_bytes, limits)

    assert result.archive_rejected is not None
    assert "uncompressed size" in result.archive_rejected


def test_rejects_archive_exceeding_member_count():
    zip_bytes = _make_zip({f"img{i}.jpg": _fake_jpeg_bytes(10) for i in range(20)})
    limits = ZipLimits(max_member_count=10)

    result = inspect_zip(zip_bytes, limits)

    assert result.archive_rejected is not None
    assert "members" in result.archive_rejected


def test_rejects_archive_exceeding_compressed_byte_limit():
    zip_bytes = _make_zip({"a.jpg": _fake_jpeg_bytes(1000)})
    limits = ZipLimits(max_zip_bytes=10)

    result = inspect_zip(zip_bytes, limits)

    assert result.archive_rejected is not None
    assert "archive exceeds" in result.archive_rejected


def test_malformed_zip_reports_archive_rejected_not_an_exception():
    result = inspect_zip(b"this is not a zip file at all")

    assert result.archive_rejected is not None
    assert "not a valid zip" in result.archive_rejected


def test_mixed_valid_and_invalid_members_partitions_correctly():
    zip_bytes = _make_zip(
        {
            "good1.jpg": _fake_jpeg_bytes(),
            "../escape.jpg": _fake_jpeg_bytes(),
            "good2.png": _fake_jpeg_bytes(),
            "readme.txt": b"not an image",
        }
    )

    result = inspect_zip(zip_bytes)

    assert {m.archive_path for m in result.accepted} == {"good1.jpg", "good2.png"}
    assert len(result.rejected) == 2


def test_extract_member_bytes_returns_correct_content():
    content = _fake_jpeg_bytes(500)
    zip_bytes = _make_zip({"a.jpg": content})
    result = inspect_zip(zip_bytes)
    assert len(result.accepted) == 1

    extracted = extract_member_bytes(zip_bytes, result.accepted[0], ZipLimits())

    assert extracted == content


def test_inspect_zip_accepts_a_path_not_just_bytes(tmp_path):
    """Large uploads should be streamed to a temp file rather than held
    fully in memory (see api/main.py's collection upload endpoint) --
    inspect_zip()/extract_member_bytes() must work directly against a Path,
    not require the caller to read it into memory first."""
    zip_bytes = _make_zip({"a.jpg": _fake_jpeg_bytes(300)})
    zip_path = tmp_path / "upload.zip"
    zip_path.write_bytes(zip_bytes)

    result = inspect_zip(zip_path)

    assert result.archive_rejected is None
    assert len(result.accepted) == 1

    extracted = extract_member_bytes(zip_path, result.accepted[0], ZipLimits())
    assert extracted == _fake_jpeg_bytes(300)


def test_extract_member_bytes_enforces_size_bound_even_if_declared_size_lied():
    """Defense in depth: even if a member's declared uncompressed_size in
    the central directory understates reality (a known zip-bomb evasion),
    extraction itself must not read unbounded data."""
    content = b"x" * 1000
    zip_bytes = _make_zip({"a.jpg": content})
    result = inspect_zip(zip_bytes)
    tiny_limits = ZipLimits(max_member_bytes=100)

    with pytest.raises(ValueError, match="possible zip-bomb"):
        extract_member_bytes(zip_bytes, result.accepted[0], tiny_limits)
