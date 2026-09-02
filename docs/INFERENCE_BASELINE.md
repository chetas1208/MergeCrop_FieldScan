# Inference Performance Baseline (Phase 0)

Real measurement, 2026-09-02, on this host's actual hardware — not estimated, not
copied from a different GPU's published numbers.

## Environment

- 2x NVIDIA RTX 3090 (24GB each), driver 535.288.01, CUDA 12.2
- `apps/vision/.venv`: Python 3.12, `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`
- SAM2: `sam2.1_hiera_small` checkpoint, real (not fallback) backend confirmed
  (`usedFallback: false`)
- DINOv2: `dinov2_vitb14` via torch.hub cache, real (not fallback) backend confirmed
- Measurement isolated to `CUDA_VISIBLE_DEVICES=0` — the same physical GPU the live
  production server uses for CV (per `deploy/local/vision.env`; GPU1 is reserved for LLM
  enrichment) — so this is a realistic single-GPU number, and GPU1 stayed untouched
  (338 MiB, unchanged) throughout.
- Standalone process, isolated from the live `api.main:app` server (PID confirmed
  separately running throughout, undisturbed) — no shared state, no risk to it.
- Input: `apps/web/public/samples/real_soybean_field.jpg` (single image, i.e. one
  observation — a deliberately small/fast probe, not a full multi-minute video, to keep
  this measurement short next to a live service rather than running a long benchmark
  unsupervised).

## Method

`FieldTriageProcessor.process()` called twice in the same Python process against the
same image, to decompose cold model construction from steady-state per-observation
compute — the existing `report.analysis.stageLatencySec` is a single number per stage
per run and conflates these two very different costs.

## Results

| Stage | Run 1 (cold) | Run 2 (steady-state, same process) | Delta |
|---|---:|---:|---:|
| segmentation (SAM2) | 6.53s | 3.39s | **-3.13s** |
| feature_anomaly (DINOv2 + structural/FarmTech) | 8.13s | 8.25s | +0.12s (noise) |
| video_decode | 0.05s | 0.03s | ~0 |
| quality_analysis | 0.04s | 0.03s | ~0 |
| registration | 0.21s | 0.16s | ~0 |
| rendering | 0.46s | 0.49s | ~0 |
| **total** | **15.73s** | **12.68s** | **-3.05s** |

GPU0 memory: 593 MiB idle baseline → 6,141 MiB during/after both runs (well under the
10 GiB single-job budget). GPU1 (LLM): untouched at 338 MiB throughout. After the
standalone process exited, GPU0 returned to the exact 593 MiB idle baseline — confirmed
by `nvidia-smi` before and after — no leaked VRAM.

## Findings (evidence, not guesses)

1. **SAM2 image-predictor construction costs ~3s**, and is a real, separable cost from
   its actual per-image inference — this matches the campaign's "T_model_load" term in
   its `T_total` breakdown exactly. Under the zero-idle-GPU architecture (a fresh worker
   per job, no persistent model residency), every single job pays this cost once,
   unavoidably, unless a future architecture change keeps a warm worker pool — which the
   zero-idle-GPU requirement explicitly rules out. So the only way to reduce this
   further, within the current architecture, is making the construction itself faster
   (e.g. `weights_only=True`/mmap-friendly checkpoint loading — not yet checked) or
   accepting it as a fixed per-job floor.
2. **feature_anomaly (DINOv2 tile embedding + structural/FarmTech CPU work) does NOT
   show a cold-vs-steady difference** (8.13s → 8.25s, within noise). This means its cost
   is genuinely dominated by real per-observation compute, not model construction —
   contradicting my own working hypothesis (and the campaign author's stated hypothesis)
   that DINO batching would be the obvious highest-leverage fix.
3. **DINOv2 tile embedding is ALREADY batched** — checked
   `cropmerge/features/dinov2.py::embed_tiles()` directly: it collects all valid grid
   tiles into a list, then runs `torch.stack(...)` in batches of 16 through one
   `self._model(batch)` call per batch, under `torch.inference_mode()`. This is NOT the
   naive per-tile-Python-call pattern the campaign's own text speculated about ("If tile-
   by-tile: FIX THIS") — that specific fix is already in place. The real bottleneck is
   somewhere else within that ~8s: candidates not yet isolated are (a) the actual ViT-B/14
   forward pass cost at FP32 with no autocast (no `torch.autocast` wrapping found in
   `embed_tiles()` — BF16/FP16 autocast, which the campaign specifically calls out as
   Meta's own recommended SAM2/DINO pattern, is NOT currently used anywhere in this
   pipeline), (b) the CPU-side structural/FarmTech scoring folded into the same
   `analyze_single_frame()` call and thus the same timed stage, not yet separated out.
4. For a **single image** (not even a multi-frame video), total analysis time is
   ~13-16 seconds, of which segmentation+feature_anomaly together are ~90%+. Confirmed
   by reading `processor.py` directly (not assumed): `create_segmenter()` is called
   exactly once per `.process()` job (line 123), then `segmenter.track_video(...)`
   handles every sampled frame internally — so the ~3s SAM2 construction cost is a
   **one-time-per-job** cost, already amortized across however many observations a real
   video has. `feature_anomaly`, by contrast, runs its per-observation loop
   (`analyze_single_frame()`, which calls `embed_tiles()`) once per sampled frame. This
   means **a real multi-minute flight's total runtime is DOMINATED by feature_anomaly
   time repeated per observation** (steady-state ~8s/observation measured here), not by
   one-time cold-start cost — which reframes where torch.compile/autocast/batching effort
   should go first: the per-observation DINO+structural stage, not model construction.

## NOT yet done (explicitly deferred, not silently skipped)

This was a short, safe, single-image Phase 0 probe — deliberately narrow given this
machine also runs a live, real vision backend (PID confirmed active throughout,
undisturbed) that a longer/heavier benchmarking session could risk contending with for
GPU/VRAM. The following remain genuinely unstarted:

- **NVTX ranges / `torch.profiler` instrumentation** inside `embed_tiles()` and
  `score_structural_cells()` to split the ~8s feature_anomaly stage into actual GPU
  forward-pass time vs. CPU structural/FarmTech compute vs. Python/copy overhead —
  needed before picking a specific optimization (autocast vs. torch.compile vs.
  reducing grid resolution vs. something in the structural code entirely).
- **A multi-frame video baseline** (not just one image) to confirm the "dominated by
  per-observation cost, not cold-start" reasoning above against real, measured
  repeated-observation behavior (the reasoning above is derived from reading
  `processor.py`'s call structure plus this single-image timing, not yet directly
  measured on a real multi-observation video).
- **The full candidate tournament** (eager+AMP, `torch.compile` in its various modes,
  Meta's `compile_image_encoder=True` SAM2 path, Torch-TensorRT, ONNX Runtime) — each of
  these needs real repeated-run measurement (the campaign itself asks for ≥3 full runs
  and ≥10 timed microbenchmark runs per candidate), which means sustained GPU time over
  many minutes per candidate. Not run tonight, specifically because of the shared-GPU
  risk with the live server noted above — this deserves either an explicit low-traffic
  window or the user's conscious go-ahead given the duration and (for Torch-TensorRT) new
  dependencies in the shared production venv, matching the campaign's own "do NOT destroy
  the functioning production environment" and "test in an ISOLATED environment" cautions.
- `apps/vision/scripts/benchmark_inference.py` (the parameterized, repeatable CLI tool
  the campaign specifies) has not been built yet — this baseline was captured with a
  one-off, non-repository script, deleted after use.
