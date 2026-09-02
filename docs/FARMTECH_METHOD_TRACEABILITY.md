# FarmTech Method Traceability

Every formula introduced by the FarmTech-native phase 1 change, traced from
paper/source through to code path and runtime role. Cross-reference
`docs/FARMTECH_RESEARCH_BASIS.md` for the full citation detail and
verification-confidence notes.

| Method | Paper / source | Formula | Code path | Runtime role | Limitations |
|---|---|---|---|---|---|
| Excess Green (ExG) | Woebbecke et al. 1995 | `ExG = 2g - r - b` (chromatic-normalized RGB) | `apps/vision/cropmerge/features/rgb_indices.py::excess_green()` (pre-existing) | Vegetation-vs-background pixel evidence; input to occupancy map in `cropmerge/anomaly/structural.py` | Chromatic proxy only; not health/NDVI; sensitive to strong color casts (e.g. heavy shadow, blue-tinted twilight) |
| Vegetation mask threshold | Derived from ExG | `ExG > threshold` (default 0.05) | `apps/vision/cropmerge/features/rgb_indices.py::vegetation_mask()` (pre-existing) | Binarizes ExG into a crop-candidate mask, consumed by `row_geometry.py` as its input mask | Threshold is a tuned constant (`configs/default.yaml: features.vegetation_exg_threshold`), not learned; may over/under-include under unusual lighting |
| Visible Atmospherically Resistant Index (VARI) | Gitelson et al. 2002 | `VARI = (G-R)/(G+R-B)` | `apps/vision/cropmerge/features/rgb_indices.py::vari()` (pre-existing) | Vegetation-fraction evidence, secondary/cross-check signal alongside ExG | Visible-band fraction estimate only; explicitly NOT NDVI, NOT a health proxy; denominator can be numerically unstable near zero (guarded with `1e-6` floor in code) |
| Guo-Hall thinning (skeletonization) | Guo & Hall 1989, used as row-geometry post-processing per Ullah, Islam & Bais 2024 | Topological thinning to 1px skeleton (algorithm, not closed-form formula) | `apps/vision/cropmerge/features/row_geometry.py::thin_mask()` via `cv2.ximgproc.thinning(..., THINNING_GUOHALL)` | Skeletonizes cleaned crop mask before Hough line detection | **Unavailable in this project's venv** -- `cv2.ximgproc` requires opencv-contrib, not installed (project depends on `opencv-python-headless`, confirmed via `pip show` in this session: only headless build present, no ximgproc module). `guo_hall_available()` reports this at runtime; see next row for the fallback actually exercised in this environment |
| Zhang-Suen thinning (fallback) | Zhang & Suen 1984 | Two-subiteration pixel-removal topological thinning (algorithm) | `apps/vision/cropmerge/features/row_geometry.py::_zhang_suen_thinning()`, invoked by `thin_mask()` when `cv2.ximgproc` import fails | Fallback skeletonization actually used in this environment; `RowGeometryResult.thinning_method == "zhang_suen_fallback"` flags this explicitly | Pure-Python/numpy vectorized loop -- slower than a native Guo-Hall call on large masks; not numerically identical to Guo-Hall (different algorithm), though qualitatively comparable (both produce ~1px topology-preserving skeletons) |
| Probabilistic Hough Transform (row-line candidates) | Standard CV technique (Matas, Galambos & Kittler 1998 for the probabilistic variant); applied to row geometry per Ullah, Islam & Bais 2024's post-processing pipeline | `cv2.HoughLinesP` over skeleton | `apps/vision/cropmerge/features/row_geometry.py::hough_row_lines()` | Produces candidate row-line segments (x1,y1,x2,y2) from the skeleton | Threshold/min-length/max-gap are tuned constants (`configs/farmtech_v1.yaml: row_geometry.*`); sensitive to skeleton quality and mask cleanliness; can miss short/broken rows below `hough_min_line_length_px` |
| Robust circular-median angle clustering | FieldScan-specific (this task); not from a cited paper | Discrete circular median: point minimizing summed circular distance to all others, mod 180 deg | `apps/vision/cropmerge/features/row_geometry.py::robust_median_angle_deg()`, `cluster_row_lines()` | Determines dominant row orientation and groups Hough segments into per-row clusters (`RowLineCluster`), feeding `num_consistent_rows` / `row_line_support_px` | O(n^2) in segment count (fine for typical per-frame segment counts, not designed for very dense line sets); one round of angle-outlier trimming, not fully iterative RANSAC |
| Structure-tensor orientation/coherence | Standard structure-tensor formulation (as already used in `cropmerge/anomaly/structural.py::_structure_tensor_orientation()`); independent re-implementation here | `theta = 0.5*atan2(2*Jxy, Jxx-Jyy)`; `coherence = sqrt((Jxx-Jyy)^2+4*Jxy^2)/(Jxx+Jyy+eps)` | `apps/vision/cropmerge/features/row_geometry.py::structure_tensor_orientation()` | Second, independent orientation/coherence estimate, cross-checked against the Hough-derived angle (`RowGeometryResult.angle_agreement_deg`); also the direct input to the observability router's `row_coherence` | Global (whole-mask) aggregate statistic -- does not localize *where* rows are, only how strongly *an* orientation dominates; degrades on masks with multiple non-parallel field sections |
| Miss Index | Kachman & Smith (planter-uniformity literature) | `100 * count(spacing > 1.5*nominal) / N` | `apps/vision/cropmerge/features/spacing_metrics.py::miss_index()` | Planting-uniformity evidence, only valid under `PLANT_RESOLVABLE` observability (see contract in module docstring) | Requires known nominal spacing and individually resolvable plants; not enforced at runtime by this module (documented caller contract only, no plant detector exists yet) |
| Multiple Index | Kachman & Smith | `100 * count(spacing <= 0.5*nominal) / N` | `apps/vision/cropmerge/features/spacing_metrics.py::multiple_index()` | Same as above | Same as above |
| Quality-of-Feed Index (QFI) | Kachman & Smith | `100 - Miss Index - Multiple Index` | `apps/vision/cropmerge/features/spacing_metrics.py::quality_of_feed_index()` | Same as above | Same as above; can be negative in pathological synthetic inputs (Miss+Multiple > 100 is impossible for a true partition of N, but not algebraically prevented if misused outside its precondition) |
| Relative spacing irregularity | FieldScan-specific fallback (this task); not from Kachman & Smith | Same 1.5x/0.5x multipliers applied to `median(spacings)` instead of a known nominal, plus coefficient of variation `std/median` | `apps/vision/cropmerge/features/spacing_metrics.py::relative_spacing_irregularity()` | Fallback planting-uniformity evidence when nominal spacing is unknown; still requires `PLANT_RESOLVABLE` observability | Self-referential (median of the same sample) -- cannot detect a stand that is *uniformly* wrong (e.g. globally too sparse but internally regular); still needs individually resolvable plants |
| Observability classification | FieldScan-specific (this task), grounded in Wang et al. 2023's resolvability framing | Threshold gate: `ObservabilityThresholds` vs. `(row_coherence, row_line_support_px, num_consistent_rows, plant_detection_count, plant_detection_confidence)` | `apps/vision/cropmerge/features/observability.py::classify_observability()` | Gates which downstream claims are valid: `PLANT_RESOLVABLE` -> spacing_metrics allowed; `ROW_RESOLVABLE` -> row-continuity/gap claims allowed; `CANOPY_ONLY` -> only coverage/fragmentation/soil-exposure evidence valid | Pure classification function; not wired into `cropmerge/pipeline/processor.py` yet, so nothing in the live pipeline currently calls it or enforces the gate -- see "Explicitly deferred" in `docs/FARMTECH_RESEARCH_BASIS.md` |

## Config surface

`apps/vision/configs/farmtech_v1.yaml` holds the row-geometry and
observability parameters as named YAML values (mirroring
`RowGeometryConfig` / `ObservabilityThresholds` field-for-field), matching
the format `cropmerge/config.py::load_config()` already expects. Not loaded
by default -- `load_config()` still defaults to `configs/default.yaml`
unless a path is passed explicitly; loading `farmtech_v1.yaml` by default is
part of the (deferred) pipeline-integration phase.

## Explicitly deferred (see docs/FARMTECH_RESEARCH_BASIS.md for the "why")

- Training the ~14M-param lightweight row-segmentation U-Net (Ullah, Islam &
  Bais 2024).
- The pseudo-label / human-correction pipeline to bootstrap that training
  data.
- Wiring `row_geometry.py`, `spacing_metrics.py`, and `observability.py`
  into `cropmerge/pipeline/processor.py`.
- Uncertainty-gated SAM2/DINOv2 escalation (foundation models as a fallback
  when cheap geometry is uncertain).
- A real plant detector (spacing_metrics.py takes plant positions/distances
  as input; it does not detect plants).
