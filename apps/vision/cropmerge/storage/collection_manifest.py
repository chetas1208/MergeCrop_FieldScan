"""ZIP image-collection ingestion into the content-addressed store.

Wires cropmerge/storage/zip_collection.py's safe inspection together with
the existing CAS blob store + manifest (cropmerge/storage/blob_store.py,
manifest.py) -- an accepted archive member never gets written to a
directory tree mirroring the archive's own (attacker-controlled) internal
paths; it is read into memory (bounded by ZipLimits.max_member_bytes),
hashed, and handed straight to the blob store.

Deduplication: two members with byte-identical content (whether the same
photo appears twice in one ZIP, or matches something already in the CAS
store from a prior upload) resolve to ONE physical blob, with each image
still getting its own logical manifest entry and its own CollectionImage
record (status="duplicate", duplicate_of=<the image id it matches>) -- the
collection statistics layer is responsible for not double-counting these,
not this module.
"""

from __future__ import annotations

import hashlib
import io
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from cropmerge.storage.manifest import Manifest
from cropmerge.storage.policies import StorageClass
from cropmerge.storage.zip_collection import (
    RejectedMember,
    ZipLimits,
    extract_member_bytes,
    inspect_zip,
)

log = logging.getLogger("cropmerge.storage.collection_manifest")


@dataclass(frozen=True)
class CollectionImage:
    id: str
    archive_path: str  # metadata only -- never used as a physical path
    relative_group: str | None
    source_sha256: str
    size_bytes: int
    status: str  # "valid" | "duplicate" | "failed"
    duplicate_of: str | None = None  # another image's id, when status == "duplicate"
    error: str | None = None  # set when status == "failed"


@dataclass
class CollectionIngestResult:
    collection_id: str
    images: list[CollectionImage] = field(default_factory=list)
    rejected: list[RejectedMember] = field(default_factory=list)
    archive_rejected: str | None = None

    @property
    def valid_images(self) -> list[CollectionImage]:
        return [img for img in self.images if img.status == "valid"]

    @property
    def duplicate_images(self) -> list[CollectionImage]:
        return [img for img in self.images if img.status == "duplicate"]


def ingest_zip_collection(
    zip_bytes: bytes | io.BytesIO | Path,
    manifest: Manifest,
    *,
    collection_id: str | None = None,
    limits: ZipLimits | None = None,
) -> CollectionIngestResult:
    """Safely ingest an uploaded ZIP into CAS-registered collection images.

    Never raises on a malicious/malformed archive or an individual bad
    member -- both are reported in the result (archive_rejected /
    per-member rejected list / a "failed" CollectionImage), matching this
    codebase's established never-block-the-real-result pattern.
    """
    limits = limits or ZipLimits()
    collection_id = collection_id or uuid.uuid4().hex[:12]

    inspection = inspect_zip(zip_bytes, limits)
    if inspection.archive_rejected is not None:
        return CollectionIngestResult(collection_id=collection_id, archive_rejected=inspection.archive_rejected)

    result = CollectionIngestResult(collection_id=collection_id, rejected=inspection.rejected)
    blob_store = manifest.blob_store
    if blob_store is None:
        raise ValueError("ingest_zip_collection requires a Manifest constructed with a BlobStore")

    seen_sha256_to_image_id: dict[str, str] = {}

    for member in inspection.accepted:
        image_id = uuid.uuid4().hex[:12]
        try:
            data = extract_member_bytes(zip_bytes, member, limits)
            sha256 = hashlib.sha256(data).hexdigest()
            already_seen = sha256 in seen_sha256_to_image_id or blob_store.exists(sha256)
            blob_ref = blob_store.put(io.BytesIO(data))
            manifest.add(
                f"collection:{collection_id}:image:{image_id}",
                blob_ref,
                storage_class=StorageClass.SCIENTIFIC_SOURCE.value,
            )
            if already_seen:
                result.images.append(
                    CollectionImage(
                        id=image_id,
                        archive_path=member.archive_path,
                        relative_group=member.relative_group,
                        source_sha256=sha256,
                        size_bytes=member.uncompressed_size,
                        status="duplicate",
                        duplicate_of=seen_sha256_to_image_id.get(sha256),
                    )
                )
            else:
                seen_sha256_to_image_id[sha256] = image_id
                result.images.append(
                    CollectionImage(
                        id=image_id,
                        archive_path=member.archive_path,
                        relative_group=member.relative_group,
                        source_sha256=sha256,
                        size_bytes=member.uncompressed_size,
                        status="valid",
                    )
                )
        except Exception as exc:
            log.exception("Failed to ingest collection member %s", member.archive_path)
            result.images.append(
                CollectionImage(
                    id=image_id,
                    archive_path=member.archive_path,
                    relative_group=member.relative_group,
                    source_sha256="",
                    size_bytes=member.uncompressed_size,
                    status="failed",
                    error=str(exc),
                )
            )

    return result
