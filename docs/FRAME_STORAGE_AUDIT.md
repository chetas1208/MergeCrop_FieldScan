# Frame Storage Audit (Phase 16, storage integration campaign)

Real audit, 2026-09-02 — what `cropmerge/pipeline/processor.py` actually writes to disk
per 1 FPS observation, and what of that is actually consumed by the product.

## What gets written per observation (before this audit)

For every sampled frame in `run_dir`:

| File | Gate | Consumed by? |
|---|---|---|
| `frames/frame_NNNN.jpg` | `output.write_frames` (default `true`) | Yes — `FrameReview.vue` + `/api/analyses/[id]/frames.get.ts` (`rawUrl`) |
| `overlays/overlay_NNNN.jpg` | none (always written) | Yes — same route (`overlayUrl`) |
| `overlays/continuity_NNNN.jpg` | none (always written) | **No — grepped `apps/web` for every reference to `continuity_`/`continuity/`; zero hits.** Not linked by `frames.get.ts`, not bundled by any export, not read by any other backend code path. |

Plus, once per run (not per observation): `annotated_video.mp4`, `heatmap.png`,
`segmentation_montage.jpg`, `results.json`, `metrics.json`.

## Finding

`continuity_NNNN.jpg` is a second full-resolution rendered JPEG, computed via a full
`annotate_frame(..., overlay_mode="continuity")` call (real CPU cost — drawing overlays,
not free) and written to disk (real storage cost), for **every single observation**, with
**zero product consumers**. This is exactly the class of waste the campaign's "1 FPS
storage rule" and "store evidence, not debug junk" sections describe — it survived
because nothing ever needed to delete it to notice.

## Fix

`configs/default.yaml`: added `output.write_continuity_overlays` (default `false`).
`processor.py`: the continuity render + write now only runs when that flag is explicitly
enabled — useful for manual debugging of the continuity-evidence scoring, off by default
in production. `overlay_NNNN.jpg` (the one actually consumed) is untouched and still
writes unconditionally, matching its real usage.

Verified (`tests/test_continuity_overlay_gating.py`): the flag correctly gates the file's
presence/absence, and toggling it produces byte-identical `inspection_zones`/`field`
analysis output either way — this was always a pure rendering side-effect, never fed back
into scoring, so removing it by default carries zero analysis-correctness risk.

## What this audit did NOT change (explicitly deferred, not silently dropped)

- `frame_NNNN.jpg` and `overlay_NNNN.jpg` stay as one-per-observation full-resolution
  JPEGs — genuinely consumed by `FrameReview.vue`'s per-frame review UI. The campaign's
  broader "sparse evidence storage" idea (store per-observation *metrics*, not images,
  except for actual Inspection Area evidence/highest-priority findings/user selection)
  would mean redesigning that frontend feature to seek into the source video on demand
  instead of pre-rendering every frame — a real, larger, frontend-affecting change, not
  attempted in this audit. Flagged for a dedicated future round.
- No change to `annotated_video.mp4`/`heatmap.png` generation or gating (already
  config-gated, already once-per-run not once-per-observation).
- The real disk-savings number from this specific fix has not been measured against a
  long real flight (would need a real multi-minute video re-run with before/after byte
  counts) — the fix's *correctness* is verified, its production-scale *impact* is not yet
  quantified. Expected to be non-trivial: previously 3 full JPEGs/observation, now 2.
