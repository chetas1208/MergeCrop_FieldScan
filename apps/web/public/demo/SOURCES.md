# Demo analysis assets

Pre-computed CropMerge Field Triage outputs for **Demo mode** (served from this repo via Vercel static files).

| Case | Input | Run ID | Zones | Generated |
| --- | --- | --- | --- | --- |
| `drone-field-video` | [real_field_drone.mp4](../samples/real_field_drone.mp4) | `d4b7789904cc` | 12 | Aug 2026 |
| `soybean-field-image` | [real_soybean_field.jpg](../samples/real_soybean_field.jpg) | `c9cf2ebba109` | 4 | Aug 2026 |

Pipeline: sam2 segmentation + dinov2 features on local CUDA. Results include `job.json`, heatmap, montage, annotated video, and per-frame overlays.

Input media attribution: see [samples/SOURCES.md](../samples/SOURCES.md).
