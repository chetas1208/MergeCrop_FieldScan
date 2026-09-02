# Limitations (read before demo)

## Scientific

- **RGB only.** No NDVI/NDRE from a Mini 2 RGB stream.
- **Visual anomaly ≠ health.** Score is relative within-field difference.
- **Many confounds:** shadow, exposure, weeds, tracks, maturity, soil, different crop, actual stress — system recommends **review**, not cause.
- **Not calibrated probabilities.** No “84% unhealthy.”

## Technical V1

- SAM 3.1 / DINOv3 require install + checkpoints; default path is **explicit heuristic fallback**.
- Temporal linking is centroid/grid based, not full multi-object tracker quality of production SAM video.
- Registration assumes roughly planar scene; altitude jumps degrade alignment.
- No orthomosaic / georeferenced map unless GPS+SfM added later.
- Synthetic demo video is for pipeline proof, not accuracy claims.

## Product language

Never show disease/nutrient/irrigation/yield conclusions from this stack alone.

## Failure modes tested in spirit

| Case | Expected behavior |
|------|-------------------|
| No field / urban | `field.detected=false`, no zones |
| Heavy blur | Low quality weight / unusable flags |
| Model missing | Clear error or marked fallback — no silent fake SAM |
| Segmentation empty | Report + limitations, no crash |
