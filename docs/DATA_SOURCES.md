# Data sources

| Name | Source | License | Type | Resolution | Context | Why used | Domain gap |
|------|--------|---------|------|------------|---------|----------|------------|
| synthetic_field | `scripts/generate_demo_video.py` | Generated | Synthetic RGB aerial-like | 640×360 | Stylized corn/soy + soil patch + road | CI + offline demo | Not real Mini-2 |
| Agriculture-Vision | [SHI-Labs](https://github.com/SHI-Labs/Agriculture-Vision) | Upstream terms | Aerial farm RGB (+NIR) | Varies | U.S. fields | RGB pipeline sanity | Still vs video; **do not use NIR for RGB-only claims** |
| USDA CDL | [NASS](https://www.nass.usda.gov/Research_and_Science/Cropland/Viewer/index.php) | U.S. gov (verify) | Crop land cover | ~30m | IL corn/soy | Future geo join | Not drone RGB |
| CropMerge pilot | Customer/pilot flights | Restricted | DJI RGB video | up to 4K | Midwest farms | Target domain | Not in repo by default |

**Policy:** Do not commit large copyrighted videos. Place licensed pilot files under `data/` or `apps/vision/samples/` locally.
