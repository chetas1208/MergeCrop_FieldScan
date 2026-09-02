# FarmTech-Native Research Basis

This document grounds the "cheap agricultural geometry/formulas first, foundation
models only as an uncertainty-gated fallback" architecture direction in the
underlying agronomy/remote-sensing literature. It exists so every formula that
ships in `cropmerge/features/` traces back to a named source, and so the
no-hallucination product boundary (never diagnose disease/nutrient/yield; never
fake a multispectral index from RGB) stays visible next to the math that could
otherwise be mistaken for more than it is.

Where a citation detail below (exact page range, exact reported metric) could
not be independently re-verified in this session, it is marked **[needs
verification]** rather than stated as fact. No number in this document was
invented — anything not directly re-derivable from the codebase's own math is
flagged.

---

## 1. Woebbecke et al. 1995 — Excess Green (ExG)

**Citation:** Woebbecke, D.M., Meyer, G.E., Von Bargen, K., Mortensen, D.A.
(1995). "Color Indices for Weed Identification Under Various Soil, Residue,
and Lighting Conditions." *Transactions of the ASAE*, 38(1), 259–269.

**What it establishes:** A family of chromatic (lighting-normalized) color
indices built from normalized RGB channels (r = R/(R+G+B), g = G/(R+G+B), b =
B/(R+G+B)) for distinguishing green plant material from soil/residue
backgrounds under variable outdoor illumination. Excess Green is defined as:

```
ExG = 2g - r - b
```

The paper's contribution — and the reason ExG is used here instead of raw
green-channel thresholding — is that normalizing by total intensity before
differencing makes the index far more stable across the shifting outdoor
lighting a drone encounters over a flight (varying sun angle, cloud cover,
shadow) than an unnormalized greenness measure would be.

**Product boundary:** ExG is a *vegetation-presence-vs-background* proxy, not
a plant-health measure. It answers "is this pixel more consistent with green
plant material than with soil/residue," nothing about vigor, stress, or
disease. FieldScan's implementation in
`apps/vision/cropmerge/features/rgb_indices.py::excess_green()` already
carries this caveat in its docstring ("ExG) visible proxy — NOT NDVI") and
must keep it.

---

## 2. Gitelson et al. 2002 — Visible Atmospherically Resistant Index (VARI)

**Citation:** Gitelson, A.A., Kaufman, Y.J., Stark, R., Rundquist, D. (2002).
"Novel algorithms for remote estimation of vegetation fraction." *Remote
Sensing of Environment*, 80(1), 76–87.

**What it establishes:** VARI is designed to estimate the *fraction of a
pixel/scene covered by green vegetation* using only the visible spectrum
(no NIR band required), while being relatively insensitive to atmospheric
effects — hence "atmospherically resistant." It is defined as:

```
VARI = (G - R) / (G + R - B)
```

**Critical distinction FieldScan must preserve:** Gitelson et al. built VARI
to estimate *vegetation fraction* (how much of the scene is plant vs.
non-plant), explicitly as a visible-band alternative to NDVI-family indices
that need a NIR band. It is **not** validated as, and must never be presented
as, a vegetation-health, chlorophyll, or NDVI-equivalent index. FieldScan's
`vari()` implementation already documents this
("NOT NDVI, NOT crop health — supporting evidence only, same as
excess_green()") — this document exists partly to make that boundary
traceable to the source paper rather than an unsourced assertion.

---

## 3. Ullah, Islam & Bais 2024 — Lightweight U-Net for Crop-Row Segmentation

**Citation:** Ullah, M.N., Islam, M.T., Bais, A. (2024). "Real-time crop row
detection for autonomous navigation using semantic segmentation with a
lightweight neural network," or closely related title under the same author
group — *Computers and Electronics in Agriculture* (2024). **[needs
verification: exact title/issue]** — the paper's core contributions below are
stated as understood from the FarmTech-native design discussion that
motivated this task, not re-derived from a re-read of the PDF in this
session, and should be spot-checked against the published version before
being cited externally.

**What it establishes (as relied upon here):**
- A lightweight (~14M parameter) U-Net-style encoder-decoder trained to
  segment crop rows directly (row/inter-row binary or narrow-class mask)
  from RGB UAV/ground imagery, sized to be far cheaper than a
  general-purpose foundation segmentation model (e.g. SAM2) at inference
  time.
- A post-processing geometry pipeline on top of the row mask: **Guo-Hall
  thinning** to skeletonize the predicted row mask down to 1-pixel-wide
  centerlines, followed by a **Probabilistic Hough Transform** over the
  skeleton to recover dominant straight-line row candidates (angle,
  endpoints, support length) for row-geometry claims (row spacing, row
  orientation, row continuity).
- A reported mean Intersection-over-Union (mIoU) of approximately **0.8444**
  for the row-segmentation task. **[needs verification: exact figure/split]**

**How FieldScan uses this:** `apps/vision/cropmerge/features/row_geometry.py`
implements the *geometry half* of this pipeline (morphological cleanup →
thinning → Probabilistic Hough Transform → angle/position clustering)
operating on whatever crop mask is available (today: the existing
segmentation/vegetation mask; in a future phase, a trained lightweight
row-segmentation model's output, once one is trained — **out of scope for
this task**, see `docs/FARMTECH_METHOD_TRACEABILITY.md` limitations column).
Training the ~14M-param model itself is explicitly **not** part of this task
(no labeled row-segmentation data or annotation pipeline available this
session) — see the "Explicitly deferred" section at the bottom of this file.

---

## 4. Wang et al. 2023 (Agronomy) — Maize Stand-Counting / Spacing from UAV Imagery

**Citation:** Wang et al. (2023). Maize stand-counting and plant-spacing
estimation from UAV RGB imagery. *Agronomy* (MDPI), 2023. **[needs
verification: exact title, volume/issue]** — relied upon here for its
methodological framing, not a verbatim quote.

**What it establishes (as relied upon here):**
- A "gap" in a crop row is only a meaningful, reportable claim when there is
  a **resolvable plant detected before the empty span AND a resolvable
  plant detected after it** — i.e., the emptiness must be bounded by
  positively identified plants, not simply inferred from a stretch of bare
  or low-vegetation pixels with no plant-level detections on either side.
  Bare pixels alone (no plant-level evidence at all) support a *coverage /
  soil-exposure* claim, not a *gap-in-the-row* claim.
  This paper's framing is the reasoning root of the observability router's
  distinction between `ROW_RESOLVABLE` (row-continuity/gap claims allowed)
  and `CANOPY_ONLY` (only coverage/fragmentation/soil-exposure evidence
  valid) in `apps/vision/cropmerge/features/observability.py`.
- Plant-level claims (individual plant detection, inter-plant spacing) are
  explicitly conditioned on having sufficient image resolution and the crop
  being at an appropriate growth stage for individual plants to be visually
  separable. The paper's own UAV data collection used approximately **V3
  growth stage** maize at approximately **0.37 cm/pixel ground sample
  distance (GSD)** as the regime in which individual-plant detection was
  validated. **[needs verification: exact GSD/growth-stage figures]**
  This is the reasoning root of the `PLANT_RESOLVABLE` tier in the
  observability router being the *only* tier at which
  `spacing_metrics.py` functions are permitted to run.

**FieldScan boundary implication:** FieldScan does not currently run a plant
detector (no such detector exists in this codebase, and building one is
explicitly out of scope for this task). `spacing_metrics.py` therefore
operates purely on caller-supplied plant-center positions / interplant
distances and documents, rather than enforces at import time, that its
functions require `PLANT_RESOLVABLE` observability — enforcement is a wiring
task for the phase that also wires a real plant detector into the pipeline.

---

## 5. Kachman & Smith — Classical Planting-Uniformity Metrics

**Citation:** Kachman, S.D., Smith, J.A. (1995). "Alternative measures of
accuracy in plant spacing for planters using single seed metering."
*Transactions of the ASAE*, 38(2), 379–387. (Commonly cited as the source of
the Miss Index / Multiple Index / Quality-of-Feed Index formulation used in
row-crop planter evaluation literature.) **[needs verification: exact
journal/page range — this is the standard citation used across planter
uniformity literature, but was not independently re-confirmed against the
original PDF in this session.]**

**What it establishes:** Given a nominal (target) seed/plant spacing and a
sequence of measured interplant spacings, three summary indices classify
planter (or, by extension, observed stand) uniformity:

```
Miss Index      = 100 * count(spacing > 1.5 * nominal) / N
Multiple Index  = 100 * count(spacing <= 0.5 * nominal) / N
Quality-of-Feed Index = 100 - Miss Index - Multiple Index
```

where `N` is the number of measured interplant spacings and `nominal` is the
known/intended plant spacing (e.g., from the planter's seed-drop rate).

**Precondition FieldScan must enforce (documented, not code-gated in this
task):** These indices are only meaningful when (a) the nominal spacing is
actually known, and (b) individual plants are reliably, individually
resolvable in the imagery (see Wang et al. 2023 above) — i.e., under
`PLANT_RESOLVABLE` observability. Applying them to a spacing array derived
from noisy or unresolved detections silently produces numbers that look
precise but are not backed by resolvable plants. `spacing_metrics.py`
documents this precondition on every public function; it does not have
access to the observability router's runtime state, so the *caller* is
responsible for checking `classify_observability(...) == "PLANT_RESOLVABLE"`
before invoking these functions on real detections.

**Fallback for unknown nominal spacing:** When nominal spacing is not known
(e.g., no planter metadata, or organic/irregular stands), `spacing_metrics.py`
also implements a **relative spacing irregularity** fallback that compares
each interplant distance to `median(interplant_distances)` instead of a
known nominal — this is a FieldScan-specific fallback, not part of the
Kachman & Smith formulation, and is documented as such in the module and in
the traceability table.

---

## 6. DJI Mavic 3 Multispectral (M3M) Sensor Bands — Documented, Not Implemented

**What it is:** The DJI Mavic 3 Multispectral (M3M) drone carries a
multispectral camera producing four narrow-band channels in addition to its
RGB camera: **Green (~560 nm), Red (~650 nm), Red Edge (~730 nm), and Near
Infrared/NIR (~860 nm)**, per DJI's published sensor specification.
**[needs verification: exact center wavelengths — figures above are DJI's
commonly published nominal values and should be checked against the current
official M3M datasheet before being used in any customer-facing document.]**

**Why it's documented here and not implemented:** These four bands are what
would be required to compute true vegetation-health-oriented indices such as
NDVI (`(NIR-Red)/(NIR+Red)`), NDRE (`(NIR-RedEdge)/(NIR+RedEdge)`), or SAVI
(a soil-adjusted NDVI variant). FieldScan 2.0's current imagery pipeline is
**RGB-only** (standard drone RGB video/photo, not M3M multispectral capture).

**Hard product-boundary rule (already in force, restated here for
traceability):** FieldScan must **never** compute or imply NDVI, NDRE, or
SAVI from RGB-only imagery. ExG and VARI are the only vegetation-evidence
indices in use today, and both are explicitly visible-band vegetation
*fraction/presence* proxies, not spectral health indices. A documented
**future branch** — gated on the input actually being confirmed M3M
multispectral capture with the Green/Red/RedEdge/NIR bands present — could
implement real NDVI/NDRE/SAVI against that data. That branch is **not
implemented in this task** and no code in this change computes any
NDVI/NDRE/SAVI-shaped formula against RGB channels.

---

## Explicitly deferred (not part of this task, and why)

- **Training the ~14M-param lightweight row-segmentation U-Net** (Ullah,
  Islam & Bais 2024): requires labeled crop-row segmentation data and a
  training/annotation pipeline that does not exist in this repository and
  cannot be produced autonomously in this session. `row_geometry.py` is
  written to operate on *any* binary crop mask (today: existing
  segmentation/vegetation masks) so it can be pointed at that model's output
  once trained, without changes to the geometry math itself.
- **Pseudo-label / human-correction pipeline** for bootstrapping the above
  training data: same reason — needs a labeling workflow and human review
  loop, out of scope for autonomous code work in this session.
- **Wiring `row_geometry.py` / `spacing_metrics.py` / `observability.py`
  into `cropmerge/pipeline/processor.py`**: deferred to keep this change's
  blast radius to new, independently testable modules. The live per-frame
  pipeline is a shared, deployed path; integrating new gating logic into it
  is a separate, riskier change that deserves its own review and rollout
  rather than landing bundled with new formula modules.
- **Uncertainty-gated SAM2/DINOv2 escalation** (the "foundation models only
  as a fallback when cheap geometry is uncertain" half of the FarmTech-native
  architecture): depends on the above integration existing first, so that
  there is a live confidence/observability signal to escalate on. Not
  implemented here.
