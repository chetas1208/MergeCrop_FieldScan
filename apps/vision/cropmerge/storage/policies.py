"""Storage-class policy definitions.

This module defines *what class* of data an object is, and documents the
compression policy that class implies. It deliberately does NOT implement
any compression, transcoding, or re-encoding — that is a later phase, and
must be gated by measured analysis-drift (see
docs/STORAGE_COMPRESSION_RESEARCH.md), not implemented speculatively here.

The load-bearing distinction is SCIENTIFIC_SOURCE vs everything else:
scientific sources are the raw evidence a field-triage finding rests on,
and must never be silently, lossily re-encoded.
"""

from __future__ import annotations

from enum import Enum


class StorageClass(Enum):
    """Compression/retention policy class for a stored object."""

    SCIENTIFIC_SOURCE = "scientific_source"
    """Raw, as-captured evidence (original uploaded video/imagery frames
    that a field-triage finding is based on). MUST be stored byte-exact,
    forever (subject to explicit retention policy, not automatic
    compaction). No lossy re-encoding, no re-compression, no downsampling,
    ever — not even "for storage savings". Only bit-identical copies
    (e.g. content-addressed dedup) or fully reversible transforms
    (see REVERSIBLE_SOURCE) are permitted to touch these bytes.
    """

    REVERSIBLE_SOURCE = "reversible_source"
    """A source that has been repackaged through a *provably lossless,
    reversible* transform — e.g. JPEG XL's lossless JPEG recompression
    (`cjxl`/`djxl`), which can reconstruct the exact original JPEG
    bitstream byte-for-byte. Storage savings are allowed here because the
    original can always be regenerated exactly; the transform must be one
    with a verified round-trip (hash-checked) before the original bytes
    are ever released. Never used for a lossy or approximate transform.
    """

    VALIDATED_LOSSY_SOURCE = "validated_lossy_source"
    """A source that has been lossily re-encoded (e.g. re-compressed JPEG
    at a lower quality, or a resized copy) *and* that specific
    encoding/parameter choice has been validated to not meaningfully
    degrade downstream CV task performance — measured drift on the actual
    analysis pipeline (segmentation IoU, anomaly-detection precision/
    recall, etc.), not just PSNR/SSIM (per Ehrlich et al. 2021, lossy
    compression artifacts can degrade DL task accuracy in ways perceptual
    metrics miss). Promoting an object into this class without that
    measurement is a policy violation, not a storage optimization.
    """

    DERIVED_ANALYSIS = "derived_analysis"
    """Output of the analysis pipeline that is not itself evidence but is
    still meaningful, structured data derived from evidence (e.g.
    segmentation masks, anomaly score grids, GeoJSON overlays, metrics
    JSON). Can be compressed aggressively with *lossless* general-purpose
    codecs (Zstandard, Blosc2 for numeric arrays/masks, lossless
    tiled COG for georeferenced rasters) since it is fully reproducible
    from the scientific source plus the pipeline version, and losing
    precision here would corrupt the finding itself.
    """

    DERIVED_PREVIEW = "derived_preview"
    """Human-facing preview/thumbnail rendering (e.g. a small JPEG for a
    dashboard card, an overlay preview for a report). Lossy compression is
    expected and fine here — the preview is explicitly not the evidence,
    and is always regenerable from a SCIENTIFIC_SOURCE or DERIVED_ANALYSIS
    object plus rendering code.
    """

    TEMPORARY = "temporary"
    """Ephemeral working data (in-progress chunked-upload staging,
    intermediate pipeline scratch files). Not retained; subject to TTL-based
    cleanup (see retention.py). Never the sole copy of anything that
    matters — if it is, it should be reclassified before it can be swept.
    """
