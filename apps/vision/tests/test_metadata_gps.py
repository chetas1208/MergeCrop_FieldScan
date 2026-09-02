import numpy as np
import pytest
from PIL import ExifTags, Image

from cropmerge.video.metadata import _parse_iso6709, extract_metadata


def test_parse_iso6709_with_altitude():
    gps = _parse_iso6709("+40.1234-088.5678+041.200/")
    assert gps == {"latitude": 40.1234, "longitude": -88.5678, "altitude": 41.2}


def test_parse_iso6709_without_altitude():
    gps = _parse_iso6709("+40.1234-088.5678/")
    assert gps == {"latitude": 40.1234, "longitude": -88.5678, "altitude": None}


def test_parse_iso6709_rejects_garbage():
    assert _parse_iso6709(None) is None
    assert _parse_iso6709("not a coordinate") is None


def test_extract_metadata_reads_jpeg_exif_gps(tmp_path):
    path = tmp_path / "photo.jpg"
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    exif = Image.Exif()
    gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
    gps_ifd[1] = "N"  # GPSLatitudeRef
    gps_ifd[2] = (40.0, 7.0, 24.0)  # GPSLatitude (DMS)
    gps_ifd[3] = "W"  # GPSLongitudeRef
    gps_ifd[4] = (88.0, 34.0, 44.0)  # GPSLongitude (DMS)
    gps_ifd[5] = 0  # GPSAltitudeRef (0 = above sea level)
    gps_ifd[6] = 41.2  # GPSAltitude
    exif[ExifTags.IFD.GPSInfo] = gps_ifd
    img.save(path, exif=exif.tobytes())

    meta = extract_metadata(path)

    assert meta.gps is not None
    assert meta.gps["latitude"] == pytest.approx(40.1233333, rel=1e-4)
    assert meta.gps["longitude"] == pytest.approx(-88.5788888, rel=1e-4)
    assert meta.gps["altitude"] == 41.2


def test_extract_metadata_jpeg_without_gps_is_none(tmp_path):
    path = tmp_path / "photo.jpg"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(path)
    meta = extract_metadata(path)
    assert meta.gps is None
