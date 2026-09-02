# FieldScan 2.0 — Storage Baseline Audit

Date of audit: 2026-09-02
Scope: `/home/923873155/CrropMerge Assignment/FieldScan 2.0` (repo root). All
numbers below are from real `du`/`find`/`md5sum` runs against the working
tree at audit time, not estimates.

## 1. Top-level disk usage

| Path | Size |
|---|---|
| `apps/` (includes `.venv`, models, third_party) | 14 GB |
| `outputs/` (repo-root vision run outputs) | 496 MB (515,311,393 bytes exact) |
| `node_modules/` | 236 MB |
| `data/` (uploads + sqlite DBs) | 80 MB |
| `.git/` | 33 MB |

`apps/` at 14 GB is dominated by the Python virtualenvs
(`apps/vision/.venv`, `apps/vision/.venv-py313-broken-cuda`) and any local
model checkpoints under `apps/vision/models` — not application data, so it
is out of scope for this storage-subsystem audit (venvs/checkpoints are a
packaging/deployment concern, not upload/output data).

The two directories in scope for a storage subsystem are `outputs/`
(vision pipeline run artifacts) and `data/uploads/` (source uploads), plus
the SQLite job/run databases under `data/db/`.

## 2. Uploads (`data/uploads/`)

- Total: **79,157,094 bytes (75.49 MB)** across **94 files**
- Average file size: **842,097 bytes (~822 KB)**
- Structure: flat legacy files directly under `data/uploads/`, plus a
  `completed/` subdirectory (40 files, 60 MB) and a `jobs/` subdirectory
  (per-upload job metadata JSON), plus a `.incomplete/` staging area for
  in-progress chunked uploads.
- File types: `.mp4` (28 files) and `.jpg` (24 files) as source media,
  plus `.json` sidecar/job files.

### Duplicate content in uploads

Byte-for-byte duplicate detection via MD5 (`find data/uploads -type f -exec
md5sum {} \;`, grouped by hash) found **7 duplicate-content groups**
spanning both the legacy flat files and `completed/`:

| Hash group | File count | Type |
|---|---|---|
| `0276da43...` | 19 | `.mp4` |
| `cc0c46a2...` | 13 | `.jpg` |
| `fe97fbdd...` | 4 | `.jpg` |
| `589e7166...` | 4 | `.mp4` |
| `991490e1...` | 3 | `.jpg` |
| `8bb7e51e...` | 3 | `.mp4` |
| `18d13c99...` | 2 | `.jpg` |

**Wasted bytes from duplication: 63,680,603 bytes (60.73 MB) — 80.4% of all
upload storage is redundant copies of content that already exists
elsewhere in the same directory tree.** This is the single strongest
finding in this audit and is the direct motivating case for
content-addressed storage: the same handful of demo/test source videos and
images have been uploaded repeatedly (e.g. once as a flat legacy file, once
again into `completed/` under a new upload id), and every one of those was
stored as a fully independent full-size copy.

### Orphaned/temporary files

`data/uploads/.incomplete/` contains **40 subdirectories**, each holding a
lone `upload.json` staging file (324 KB total) with no corresponding
completed upload. These are leftovers from interrupted or abandoned
chunked-upload sessions — orphaned temporary state that nothing currently
cleans up. Individually small, but unbounded: this directory will grow
without limit as long as any client starts-and-abandons uploads, since
there is no TTL sweep today. This is exactly the gap `retention.py`'s
`find_orphaned_temp_files()` is built to detect (detection only — no
automatic deletion is implemented in this phase).

No `.tmp`, `.temp`, `~`-suffixed, `.part`, or `.DS_Store` files were found
anywhere else in the repository (checked via `find` across the whole tree,
excluding `node_modules`, `.venv*`, and `.git`).

## 3. Outputs (`outputs/` at repo root)

- Total: **515,311,393 bytes (491.44 MB)** across **1,806 files**
- Average file size: **285,333 bytes (~279 KB)**
- **57 run directories** (56 real pipeline runs, keyed by an md5-looking
  hex run id, plus one `demo/` directory containing 4 further sub-runs)
- Average size per run directory (excluding `demo/`): **~8.78 MB**
- File types by count: `.jpg` (1,575 — dominant, both `frames/` and
  `overlays/` subdirectories), `.json` (114 — `results.json` +
  `metrics.json` per run), `.png` (56 — one `heatmap.png` per run), `.mp4`
  (56 — one `annotated_video.mp4` per run), `.zip` (2), `.geojson` (2)

### Per-run structure (consistent across all runs inspected)

```
outputs/<run_id>/
  metrics.json
  results.json
  segmentation_montage.jpg
  heatmap.png
  annotated_video.mp4
  frames/            (~20 frame_XXXX.jpg per run)
  overlays/           (~40 overlay_XXXX.jpg + continuity_XXXX.jpg per run)
```

### Duplicate content in outputs

Byte-for-byte duplicate detection via MD5 across all 1,806 files in
`outputs/` found **319 duplicate-content groups**, representing **1,047
redundant file copies**.

**Wasted bytes from duplication: 339,865,228 bytes (324.12 MB) — 66.0% of
all output storage is byte-identical content stored more than once.** This
is consistent with the per-run structure above: the same small set of
source videos have been re-run through the pipeline dozens of times
(demo/test iterations), and every run wrote its own independent full-size
copies of extracted frames, overlays, heatmaps, and the re-encoded
annotated video — even where the pipeline produced pixel-identical output
each time.

Two empty directories were found (`outputs/cdd4d03a9773/overlays` and
`outputs/cdd4d03a9773/frames`), consistent with a run that failed after
directory creation but before writing any artifacts — a second, smaller
category of orphaned state alongside the `.incomplete` upload staging
directories.

## 4. `apps/vision/outputs/` and `apps/vision/samples/`

- `apps/vision/outputs/`: 9,184,088 bytes (8.76 MB) across 118 files — a
  separate, smaller output tree used by the vision app's own dev/test runs
  (distinct from the repo-root `outputs/` used by the deployed API).
- `apps/vision/samples/`: 63,620,288 bytes (60.67 MB) across 12 files
  (average 5.3 MB/file) — fixture media used by tests/scripts, not runtime
  data. Not audited for duplication since these are intentionally-committed
  fixtures, not accumulating runtime state.

## 5. SQLite databases (`data/db/`)

| File | Size |
|---|---|
| `vision.sqlite` | 2,801,664 bytes (2.67 MB) |
| `vision-jobs.sqlite` | 651,264 bytes (636 KB) |
| `vision-jobs.sqlite-shm` | 32,768 bytes |
| `vision-jobs.sqlite-wal` | 0 bytes (clean checkpoint at audit time) |
| `cropmerge.sqlite` | 151,552 bytes (148 KB) |

Three separate SQLite databases exist for what appears to be overlapping
concerns (vision runs, vision jobs, and a general "cropmerge" DB) — noted
here as an observation for a future consolidation phase, not something
this storage-subsystem task touches.

## 6. Summary of findings that motivate this phase's design

1. **Uploads are ~80% duplicate bytes**, outputs are **~66% duplicate
   bytes** — a content-addressed blob store with hash-based dedup (this
   task's `blob_store.py`) would eliminate the overwhelming majority of
   this waste immediately, with zero risk to scientific evidence, since
   deduplication only ever collapses byte-identical copies.
2. **Orphaned temp state exists today and is unbounded** (`.incomplete/`
   upload staging directories, empty run directories from failed runs).
   Nothing currently cleans these up. `retention.py`'s
   `find_orphaned_temp_files()` gives a pure, testable way to *identify*
   this class of file; wiring an actual deletion sweep is deliberately
   deferred to a later phase per this task's scope.
3. **No compression is applied anywhere today** — uploads and outputs are
   stored as whatever bytes the client sent / the pipeline wrote, at full
   resolution. This is the safe starting point; see
   `docs/STORAGE_COMPRESSION_RESEARCH.md` for why any move away from that
   must be evidence-preserving and validated, not just "smaller."
4. Average artifact sizes (upload ~822 KB, output file ~279 KB, ~8.78 MB
   per full run) give a concrete baseline against which any future
   compression or storage-tiering phase should measure real improvement.
