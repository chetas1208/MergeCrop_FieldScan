# Storage & Compression Research Notes

Date: 2026-09-02
Purpose: ground FieldScan's storage-subsystem design in the actual
compression/CV literature and tooling, so that any future compression
phase is evidence-preserving by construction, not by policy alone. This
document is research and rationale only — no compression is implemented
in this phase (see `docs/STORAGE_BASELINE.md` for the current, uncompressed
baseline, and `apps/vision/cropmerge/storage/policies.py` for the
storage-class taxonomy this research motivates).

## 1. Lossy compression can degrade downstream CV accuracy — Ehrlich et al., ICCV Workshops 2021

**Citation:** Max Ehrlich, Larry Davis, Ser-Nam Lim, Abhinav Shrivastava,
"Analyzing and Mitigating JPEG Compression Defects in Deep Learning,"
*Proceedings of the IEEE/CVF International Conference on Computer Vision
(ICCV) Workshops*, October 2021, pp. 2357–2367. Presented at the ICCV 2021
MELEX workshop. Also available as arXiv:2011.08932. (Verified via the CVF
Open Access repository and arXiv listing.)

**What it actually shows:** The paper is a unified empirical study of how
JPEG compression affects a range of common computer-vision tasks and
datasets. Its central finding is that there is a **significant penalty on
standard performance metrics at high compression levels** — i.e.
degradation is not hypothetical or edge-case, it shows up across tasks.
The authors note that this had previously been under-studied because
benchmark datasets are often distributed losslessly compressed or at high
JPEG quality, which masks the effect in academic settings even though
production/consumer pipelines routinely apply much more aggressive
compression. The paper also proposes and evaluates mitigation methods,
including a no-label-required artifact-correction approach.

**Why it matters for FieldScan:** This is direct evidence that "does the
image still look fine to a human" (or even PSNR/SSIM, which correlate with
human perception) is not a valid proxy for "will the CV pipeline still
perform correctly on this image." A compression choice that looks
perceptually lossless can still measurably hurt a downstream model. For a
field-triage product whose entire value proposition is the correctness of
CV-derived findings, this means: **any decision to lossily re-encode a
source image must be validated against the actual downstream task's
performance (segmentation IoU, anomaly precision/recall, etc. — the same
metrics FieldScan already computes per run), not against file size or
generic image-quality metrics alone.**

## 2. Perceptually-optimized codecs are not necessarily optimal for machine consumption — remote-sensing "coding for machines" literature

**Citation (representative, verified via IEEE Xplore):** "Remote Sensing
Image Coding for Machines on Semantic Segmentation via Contrastive
Learning," *IEEE Transactions on Geoscience and Remote Sensing*, 2024
(IEEE Xplore document 10716527).

**The general point (broader than this one paper):** Classical and even
modern learned image codecs (JPEG, JPEG2000, and most rate-distortion-
optimized neural codecs) are tuned to minimize *perceptual* distortion —
i.e. to look good, or measure well on PSNR/SSIM/LPIPS, to a human viewer or
a human-vision-based metric. The "coding/compression for machines"
research area (sometimes called ICM, or under the MPEG "Video Coding for
Machines" standardization umbrella) starts from the observation that this
is the wrong objective when the actual consumer of the image is a
downstream model, not a human: a codec can preserve exactly the visual
information a human eye weights heavily while discarding
texture/frequency information a segmentation or detection model relies on,
or vice versa. The TGRS 2024 paper above addresses this specifically for
remote-sensing semantic segmentation, proposing a contrastive-learning-
based coding scheme intended to preserve the feature discriminability that
a segmentation model needs, rather than optimizing for a perceptual
metric. *(This is a fast-moving, actively-published sub-field as of this
writing; treat the general point — perceptual optimality != task
optimality — as the load-bearing claim, and treat any specific
architecture/quantitative-improvement numbers from a given paper as
needing re-verification if they are ever used to justify a specific
production choice.)*

**Why it matters for FieldScan:** FieldScan's imagery is drone-captured
field/crop RGB, closer in kind to remote sensing imagery than to
consumer photography. The same caution applies: a codec setting chosen
because it "looks fine" or scores well on a generic quality metric is not
thereby proven safe for the segmentation/anomaly pipeline that actually
drives FieldScan's findings.

## 3. JPEG XL reversible JPEG recompression (ISO/IEC 18181)

**What it is:** JPEG XL is an ISO/IEC 18181 image format. Its reference
implementation, `libjxl`, ships `cjxl` (encode) and `djxl` (decode) CLI
tools. `libjxl` supports **losslessly transcoding an existing JPEG file
into a `.jxl` container** — not re-encoding the pixels, but repacking the
existing JPEG DCT coefficient stream more efficiently — and `djxl` can
**reconstruct the original JPEG file bit-for-bit** from that `.jxl` file
(verified: `djxl` writes the exact original JPEG bytes back out when the
output filename has a `.jpg`/`.jpeg` extension; round-trip integrity is
commonly checked with `cmp original.jpg reconstructed.jpg`). Typical
reported size savings from this lossless JPEG recompression mode are
around 20%, with no pixel or bitstream information lost — it is a true
"free lunch" storage reduction on already-JPEG-encoded material.

**Why it matters for FieldScan:** This is the one compression technique
in this document that could apply to a `SCIENTIFIC_SOURCE` JPEG *without*
violating the no-lossy-touch rule — because it is provably reversible to
the original bitstream, not merely "visually similar." This is exactly why
`policies.py` defines a distinct `REVERSIBLE_SOURCE` class rather than
lumping every space-saving transform in with lossy re-encoding: the two
have fundamentally different risk profiles. Note this only applies to
already-JPEG source material; FieldScan's video sources and any non-JPEG
image sources are not addressed by this technique. **Not implemented in
this phase** — promoting this from "researched" to "used" requires
building the actual transcode-and-verify step (hash-check the
reconstructed JPEG against the original before ever deleting the
original), which belongs in a later phase.

## 4. General-purpose lossless compression for derived/non-image artifacts

- **Zstandard (Zstd):** A general-purpose lossless compressor (Meta/RFC
  8878) offering a strong speed/ratio tradeoff across a wide range of
  compression levels, with a fast decompressor. Well suited to
  compressing JSON results/metrics files, logs, and as the backing codec
  for tiled raster formats (see COG below). Appropriate for
  `DERIVED_ANALYSIS`-class objects.
- **Blosc2:** A compression library purpose-built for binary/numeric
  array data (NumPy arrays, masks), combining a fast blocking/shuffle
  pre-filter with a pluggable backend codec (commonly Zstd or LZ4). Because
  it operates on the array's element structure rather than treating the
  buffer as opaque bytes, it typically achieves both better ratios and
  higher throughput than applying a generic compressor directly to a raw
  array dump — relevant to FieldScan's segmentation masks and anomaly
  score grids, which are exactly this kind of numeric array data.
- **GDAL / rio-cogeo — lossless tiled Cloud-Optimized GeoTIFF (COG):**
  GDAL's GeoTIFF driver (and the `rio-cogeo` convenience layer over it)
  can produce **tiled, overview-pyramided GeoTIFFs using lossless
  compression** (`ZSTD`, `DEFLATE`, or `LZW`, optionally with horizontal
  differencing/predictor filters). This gives efficient partial/streamed
  access to large georeferenced rasters (e.g. field-scale orthomosaics or
  per-pixel score maps) while remaining fully lossless — no pixel value is
  approximated. Relevant wherever FieldScan produces or will produce
  georeferenced raster output, as opposed to plain preview imagery.

None of these four are implemented in this phase either; they are named
here so the `DERIVED_ANALYSIS` storage class's docstring ("can compress
aggressively with lossless codecs") points at concrete, real tooling
rather than a vague aspiration.

## 5. Implications for FieldScan's storage-class design

Putting the above together, the storage subsystem must distinguish (at
minimum) between:

- **`SCIENTIFIC_SOURCE`** — the original, as-captured evidence. Per
  Ehrlich et al., even "high quality" lossy compression is not proven safe
  for a CV pipeline's accuracy, so these bytes are never lossily touched.
  The only permitted storage-saving move is bit-identical deduplication
  (this phase's `blob_store.py`).
- **`REVERSIBLE_SOURCE`** — a source repackaged through a transform with a
  *proven* bit-exact round trip (e.g. JPEG XL's lossless JPEG
  recompression). Storage savings are allowed because nothing is actually
  lost; the original can always be regenerated and hash-verified.
- **`VALIDATED_LOSSY_SOURCE`** — a source that has been lossily
  re-encoded, where that specific encoding choice has been validated by
  measuring FieldScan's own analysis-pipeline outputs (segmentation IoU,
  anomaly precision/recall — not PSNR/SSIM, per Ehrlich et al., and not a
  human eyeball check) before being trusted as a replacement. A file is
  never placed in this class based on file-size reduction alone.
- **`DERIVED_ANALYSIS`** — pipeline outputs (masks, score grids, GeoJSON)
  that are reproducible from a source plus the pipeline version. Safe to
  compress aggressively with lossless general-purpose tooling (Zstd,
  Blosc2 for arrays, lossless COG for georeferenced rasters), because
  losing precision here would silently corrupt the finding rather than
  just cost storage.
- **`DERIVED_PREVIEW`** — human-facing thumbnails/previews. Lossy
  compression is fine and expected; these are explicitly not evidence.
- **`TEMPORARY`** — ephemeral working data, subject to TTL cleanup, never
  the sole copy of anything durable.

This taxonomy is implemented (as an enum plus per-class policy docstrings
only, no transcoding logic) in
`apps/vision/cropmerge/storage/policies.py`. Any future phase that adds
actual compression/transcoding should be judged against whether it
correctly assigns objects to these classes and respects each class's
stated policy — in particular, whether it can prove (via measured
analysis-drift, not file size) that a `VALIDATED_LOSSY_SOURCE` promotion
is safe.

## Sources

- [Ehrlich et al., "Analyzing and Mitigating JPEG Compression Defects in Deep Learning" — CVF Open Access](https://openaccess.thecvf.com/content/ICCV2021W/MELEX/html/Ehrlich_Analyzing_and_Mitigating_JPEG_Compression_Defects_in_Deep_Learning_ICCVW_2021_paper.html)
- [Ehrlich et al. — arXiv:2011.08932](https://arxiv.org/abs/2011.08932)
- ["Remote Sensing Image Coding for Machines on Semantic Segmentation via Contrastive Learning" — IEEE Xplore, TGRS 2024](https://ieeexplore.ieee.org/document/10716527/)
- [djxl(1) manual page — lossless JPEG reconstruction behavior](https://manpages.debian.org/unstable/libjxl-tools/djxl.1.en.html)
- ["Save 22% of your storage with this one easy trick" — practical notes on JPEG XL lossless JPEG transcoding](https://mina86.com/2025/use-jpeg-xl-already/)
