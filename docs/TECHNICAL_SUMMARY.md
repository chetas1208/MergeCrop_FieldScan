# Technical Summary — CropMerge Field Triage

## 1. Problem understanding

CropMerge needs a **field triage** layer: turn ordinary RGB drone flights into stable, explainable **where to look** signals for farmers and agronomists — without pretending RGB is multispectral science.

## 2. Architecture

Nuxt monorepo product shell + Python vision engine. Typed JSON contract (`schemaVersion: 1.0`). See ARCHITECTURE.md.

## 3. Model choices

| Role | Choice | Why |
|------|--------|-----|
| What / where | SAM 3.1 (adapter) | Text/video promptable segmentation + tracking path |
| What looks different | DINOv3 dense features | Within-field similarity, not health class labels |
| Explainable cues | ExG, Lab, texture | Farmer-readable reasons (ExG ≠ NDVI) |
| Is it real | Temporal consensus | Suppress one-frame spikes |
| Offline demo | Heuristic backends | Honest fallback when weights/GPU missing |

## 4. Data

- Synthetic procedural field video (reproducible CI)
- Hooks for Agriculture-Vision (RGB-only eval)
- USDA CDL as future geospatial layer
- Real Midwest Mini-2 flights: drop in `samples/` (do not commit copyrighted media)

## 5. What works

- End-to-end local path: video → JSON + MP4 + heatmap + UI
- Quality weighting, grid anomaly, zone IDs, priority language
- No-field suppression
- Explicit fallback marking

## 6. What struggles

- Heuristic segmentation ≠ SAM quality on real canopy
- Without real DINO weights, embedding channel is weak
- Planar homography fails on strong altitude/perspective change
- Shadows can inflate visual anomaly

## 7. RGB limitations

Cannot manufacture NDVI. Cannot diagnose disease/N/water/yield from RGB alone. Anomaly score = relative visual deviation.

## 8. Video stability

Registration confidence logged; temporal linking by centroid proximity + score gate; min persistent frames configurable.

## 9. Compute

CPU-capable with heuristic backends. SAM/DINO want GPU + multi-GB checkpoints.

## 10. DJI Mini 2–style footage

4K RGB is enough for segmentation + variation POC. Expect motion blur, exposure swings, missing telemetry — pipeline tolerates missing GPS.

## 11. Geospatial roadmap

Pose hooks → SfM/ODM → orthomosaic → CDL join. Not in V1 critical path.

## 12. Best next experiment

**Fly 45–90s of Illinois corn or soybean at ~30–50m AGL (Mini 2), include a known bare/thin patch and a headland road, overcast if possible. Run heuristic then SAM3+DINOv3 when weights available. Compare zone persistence vs ground notes — no health labels required.**
