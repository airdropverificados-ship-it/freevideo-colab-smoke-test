# FreeVideo — Colab free-tier smoke test

**Status: not run yet.** This notebook answers one question: can FreeVideo
(MiniMax H3) generate a 5 s clip on a free Colab T4? No run has been done
yet, so there are no results here — only the test itself. Part of the
[free-video-orchestrator](https://github.com/airdropverificados-ship-it/free-video-orchestrator)
project ($0-budget AI video generation).

**Notebook:** `freevideo-colab-test.ipynb` (25 cells, nbformat-validated).
**Builder:** `build_notebook.py` — re-run it to regenerate the notebook after edits.

## How to run
1. Upload `freevideo-colab-test.ipynb` to Google Colab (free tier, GPU runtime = T4).
2. Run top to bottom in one session. Do not let the runtime go idle.
3. Total time: ~30–90 min setup/downloads + 15–60+ min for the 5 s test clip.

## What it does
1. Env check (GPU/VRAM/**driver 580+**/RAM/disk — needs ≥ 50 GB free).
2. Creates a 16 GiB swap file (Colab free has only ~12.67 GiB RAM vs the 16 GiB minimum).
3. Clones FreeVideo pinned to commit `22599748` (main @ 2026-11-20).
4. Inspects `setup.sh` for flags, then runs it non-interactively (`yes |`, 90 min cap).
   Downloads the runtime + ComfyUI + model packs (~35 GB).
5. Verifies install, prints `./freevideo plan` dry-run for this machine.
6. Stops ComfyUI to free RAM, then generates **one 5 s clip, Light preset,
   768×432, single-pass** → `/content/freevideo-test/test-5s.mp4`
   (+ `test-5s.request.json` with plan/timings/memory peaks).
7. Verifies the output (ffprobe + inline playback). Optional `./test.sh` cell
   for diagnostics if generation fails.

## Model downloads (expected)
- Video model, prepared FP8 `OpenVDN/vdn-minimax-h3-edge`: ~22.9 GB
- Text encoder + video/audio decoder + sampling caches: ~12 GB
- **Total weights ~35 GB** (the full upstream VDN-H3 repo is 82 GB — FreeVideo
  does NOT download that). Plus ComfyUI + runtime env (a few GB) + 16 GB swap.

## Key risks
- **RAM:** 12.67 GiB < 16 GiB minimum. Mitigated by swap + FreeVideo's own disk
  streaming (planner puts excess transformer blocks on disk; slower but designed).
- **T4 = Turing (SM75):** not in FreeVideo's documented GEMM matrix (int8 on
  Ada/Blackwell/Ampere; FP8 scales listed for SM120/SM89/SM90). FP8 datacenter
  path should apply, but unconfirmed — the test answers this.
- **Driver 580+** required by the Linux guide; checked in cell 1.
- If the minimum working set doesn't fit, FreeVideo rejects cleanly and reports
  the shortfall — that is itself a useful result.

## Report back
Whether `test-5s.mp4` was produced, wall time, the `.request.json`
(timings/memory peaks), and the step-7 plan output (blocks in VRAM vs RAM vs disk).
Download the mp4 + request.json + generate.log before the runtime recycles.
