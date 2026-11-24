#!/usr/bin/env python3
"""Build freevideo-colab-test.ipynb — FreeVideo smoke test on Colab free tier.

Sources (fetched 2026-11-20):
- https://github.com/FlashML-org/FreeVideo (README, commit 22599748)
- https://github.com/FlashML-org/FreeVideo/blob/main/docs/Linux.md
- https://github.com/FlashML-org/FreeVideo/blob/main/docs/execution-planning.md
- https://huggingface.co/OpenVDN/vdn-minimax-h3
- https://comfyui-wiki.com/en/news/2026-10-04-freevideo-h3
"""
import json
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(OUT_DIR, "freevideo-colab-test.ipynb")
PINNED_COMMIT = "225997487d52a873882183245c594e3b0249a3d6"  # main @ 2026-11-20


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}


cells = []

cells.append(md("""# FreeVideo on Google Colab (free tier) — smoke test

**What this is:** a one-shot test of [FreeVideo](https://github.com/FlashML-org/FreeVideo)
(FlashML-org, Apache-2.0) — local AI video generation with the MiniMax H3 model
(OpenVDN 8-step VDN-H3 checkpoint, FP8). Goal: **one short test clip** on a free
Colab T4, to see whether this route can generate video without paid credits.

**What the run does, in order:**
1. Checks GPU / RAM / disk (and the driver version — FreeVideo wants NVIDIA driver **580+**).
2. Adds a 16 GiB swap file (Colab free has only ~12.67 GiB RAM; FreeVideo asks for 16 GiB).
3. Clones FreeVideo (pinned commit) and runs the official `./setup.sh` (installs the
   runtime, ComfyUI and the model packs, ~35 GB download).
4. Prints the execution **plan** for this machine without loading weights.
5. Generates **one 5-second clip, Light preset, 768×432**, and saves it as
   `/content/freevideo-test/test-5s.mp4` (+ a `.request.json` with timings/memory).

**Expected time:** ~30–90 min for setup/downloads, then **15–60+ min** for the 5 s
clip on a T4 (reference: an RTX 4060 Ti 16 GB does 10 s at 1344×768 two-pass in
~558 s; the T4 is much slower and will stream weights from disk). Run the cells
top to bottom in one session; do not let the runtime go idle.

**Known risks (see the last cell for how to read the result):**
- Colab free RAM (12.67 GiB) is below the 16 GiB minimum → swap + disk streaming.
- T4 is Turing (SM75): it is **not** in FreeVideo's documented GEMM matrix
  (int8: Ada/Blackwell/Ampere; FP8 scales listed for SM120/SM89/SM90). The FP8
  datacenter path should apply, but this is unconfirmed on Turing — the test
  itself is what answers it.
- Driver 580+ is required by the Linux guide; Colab's driver is checked in cell 1.
- If the minimum working set does not fit, FreeVideo rejects the request cleanly
  and reports the shortfall instead of crashing — that is itself a useful result.
"""))

cells.append(md("""## 1 — Environment check

GPU model + VRAM + **driver version**, RAM, free disk. FreeVideo needs an NVIDIA
GPU, driver 580+, ~35 GB of downloads and room for a 16 GiB swap file, so we want
**≥ 50 GB free** before starting."""))

cells.append(code("""%%bash
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo
echo "=== RAM ==="
free -h
echo
echo "=== Disk ==="
df -h / /content 2>/dev/null
echo
echo "=== CPU / OS ==="
nproc
python3 --version
grep -E '^(NAME|VERSION)=' /etc/os-release | head -2
"""))

cells.append(md("""## 2 — Swap file (RAM mitigation)

Colab free gives ~12.67 GiB RAM; FreeVideo's minimum is 16 GiB. The planner reads
OS/cgroup memory (swap is likely **not** counted in its budget — its capacity grid
was measured without swap), so swap does not make the planner plan for more RAM.
What it does is keep the OS from OOM-killing host-side processes (the text
encoder runs as a separate process; activations spill to host buffers) while
FreeVideo's own **disk streaming** handles the transformer blocks that do not fit
in RAM. Slower, but it is the designed fallback for low host memory."""))

cells.append(code("""%%bash
set -e
swapon --show || true
CURRENT_SWAP_GB=$(free -g | awk '/Swap:/ {print $2}')
echo "Current swap: ${CURRENT_SWAP_GB} GiB"
if [ "$CURRENT_SWAP_GB" -lt 8 ]; then
  echo "Creating 16 GiB swap file (Colab runs as root, no sudo needed)..."
  fallocate -l 16G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=16384 status=progress
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  echo "Swap enabled."
else
  echo "Swap already >= 8 GiB, keeping it."
fi
free -h
echo "--- disk after swap file ---"
df -h / | tail -1
"""))

cells.append(md("""## 3 — Clone FreeVideo (pinned commit)

Pinned to the `main` commit current when this notebook was written (2026-11-20),
so the test is reproducible even if the repo moves on."""))

cells.append(code("""%%bash
set -e
cd /content
if [ ! -d FreeVideo ]; then
  git clone https://github.com/FlashML-org/FreeVideo.git
fi
cd FreeVideo
git fetch --tags origin
git checkout PINNED_COMMIT
git log --oneline -1
ls
""".replace("PINNED_COMMIT", PINNED_COMMIT)))

cells.append(md("""## 4 — Inspect `setup.sh` before running it

`setup.sh` checks GPU/memory/disk, shows a plan and **asks for confirmation**
(interactive). We look for non-interactive flags first; if there are none, the
next cell answers its prompts with `yes`. Nothing is executed here."""))

cells.append(code("""%%bash
cd /content/FreeVideo
echo "--- long flags mentioned in setup.sh (if any) ---"
grep -oE '\\-\\-[a-z][a-z0-9-]*' setup.sh | sort -u | head -40 || true
echo
echo "--- first 40 lines of setup.sh ---"
head -40 setup.sh
echo
echo "--- disk free before install ---"
df -h / | tail -1
FREE_GB=$(df -BG / --output=avail | tail -1 | tr -dc '0-9')
echo "Free disk: ${FREE_GB} GB (want >= 45 GB to proceed safely)"
"""))

cells.append(md("""## 5 — Run the official setup (the long step)

Installs FreeVideo's runtime, ComfyUI and the model packs, then starts ComfyUI
in the background. Downloads (from the FreeVideo docs + ComfyUI wiki):

| Pack | Approx. size |
| --- | --- |
| Video model — prepared FP8 `OpenVDN/vdn-minimax-h3-edge` | ~22.9 GB |
| Text encoder + video & audio decoder + sampling caches | ~12 GB |
| **Total weights** | **~35 GB** |
| ComfyUI + Python runtime env | a few GB |

Note: the *full* upstream VDN-H3 repo is ~82 GB — FreeVideo does **not** download
that; it downloads only the prepared edge release above. The optional audio
reference cache is skipped (only needed when a reference includes audio).

The cell aborts if free disk is under 45 GB, and caps the run at 90 min
(`timeout 5400`). Output is logged to `/content/freevideo-setup.log`.
At the end setup tries to open a browser — that fails harmlessly in headless
Colab; ComfyUI keeps running in the background and we stop it before generating
to free RAM."""))

cells.append(code("""%%bash
cd /content/FreeVideo
export DEBIAN_FRONTEND=noninteractive
FREE_GB=$(df -BG / --output=avail | tail -1 | tr -dc '0-9')
if [ "$FREE_GB" -lt 45 ]; then echo "ABORT: only ${FREE_GB} GB free, need >= 45 GB."; exit 1; fi
echo "Starting setup (downloads runtime + ~35 GB of model packs). Can take 30-90 min."
echo "Log: /content/freevideo-setup.log"
yes | timeout 5400 ./setup.sh 2>&1 | tee /content/freevideo-setup.log
echo "setup.sh exit (via timeout): ${PIPESTATUS[1]}  (124 = hit the 90 min cap)"
"""))

cells.append(md("""## 6 — Verify the install

Prints the CLI help (saved to files — the generation cell below reads the
`generate --help` output to pick supported flags) and the on-disk size."""))

cells.append(code("""%%bash
cd /content/FreeVideo
./freevideo status || true
echo "--- ./freevideo --help ---"
./freevideo --help 2>&1 | tee /content/freevideo-help.txt
echo "--- ./freevideo generate --help ---"
./freevideo generate --help 2>&1 | tee /content/freevideo-generate-help.txt
echo "--- disk usage ---"
du -sh /content/FreeVideo 2>/dev/null
df -h / | tail -1
"""))

cells.append(md("""## 7 — Dry-run the execution plan (no weights loaded)

`./freevideo plan` prints how the Adaptive Execution Planner will place the 50
transformer blocks across VRAM / pinned host memory / disk for this machine and
this request geometry. Read it before generating: if it says the minimum working
set does not fit, or that almost everything streams from disk, generation will
be very slow — that is still worth knowing."""))

cells.append(code(r"""import re, subprocess

smi = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader'],
                     capture_output=True, text=True).stdout
vram_gib = int(re.search(r'(\d+)', smi).group(1)) // 1024
meminfo = open('/proc/meminfo').read()
ram_gib = int(re.search(r'MemTotal:\s+(\d+)', meminfo).group(1)) // 1024 // 1024
print(f"Measured: {vram_gib} GiB VRAM, {ram_gib} GiB RAM")

# Plan with 1 GiB less VRAM than total (headroom) and the measured RAM.
cmd = ['/content/FreeVideo/freevideo', 'plan',
       '--vram-gib', str(max(vram_gib - 1, 8)),
       '--ram-gib', str(ram_gib),
       '--width', '768', '--height', '432', '--seconds', '5']
print('Running:', ' '.join(cmd))
r = subprocess.run(cmd, cwd='/content/FreeVideo', capture_output=True, text=True)
print(r.stdout[-6000:])
print(r.stderr[-2000:])
print('Exit code:', r.returncode)
"""))

cells.append(md("""## 8 — Free RAM: stop ComfyUI before generating

Setup leaves ComfyUI running in the background. The CLI generation does not need
it, so stop it to give the text encoder and the transformer as much host RAM as
possible."""))

cells.append(code("""%%bash
cd /content/FreeVideo
./freevideo stop || true
sleep 5
free -h
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
"""))

cells.append(md("""## 9 — Test generation: ONE short clip

Simple prompt, **Light preset, 768×432, 5 seconds, single-pass** (no two-pass
sampling) — the cheapest request that still exercises the full pipeline
(text encoder → transformer → VAE decode → MP4 with synchronized audio).

The cell reads `./freevideo generate --help` (saved in step 6) and only adds
flags the installed version actually supports, so it also works if flag names
differ. Documented base form: `./freevideo generate --prompt-file prompt.txt
--out video.mp4`.

Output: `/content/freevideo-test/test-5s.mp4` + `/content/freevideo-test/test-5s.request.json`
(plan, stage timings, memory peaks). Log: `/content/freevideo-test/generate.log`.
First run also compiles kernels (several minutes) — that time is one-off.
If the engine hits OOM it retries up to 2× in a fresh process with a lighter
placement on its own."""))

cells.append(code(r"""import os, re, subprocess, time

os.makedirs('/content/freevideo-test', exist_ok=True)
prompt = ("A cheerful red apple character with a smiling face standing in a sunny fruit market, "
          "waving at the camera, bright daylight, vivid colors, smooth gentle motion.")
open('/content/freevideo-test/prompt.txt', 'w').write(prompt + "\n")
print("Prompt:", prompt)

help_path = '/content/freevideo-generate-help.txt'
help_text = open(help_path).read() if os.path.exists(help_path) else ''
def has(flag):
    return flag in help_text

cmd = ['/content/FreeVideo/freevideo', 'generate',
       '--prompt-file', '/content/freevideo-test/prompt.txt',
       '--out', '/content/freevideo-test/test-5s.mp4']

# Preset: prefer the lightest one the CLI offers.
m = re.search(r'--preset\s+\{([^}]+)\}', help_text)
if m:
    choices = [c.strip() for c in m.group(1).split(',')]
    light = next((c for c in choices if c.lower() == 'light'), choices[0])
    cmd += ['--preset', light]
elif has('--preset'):
    cmd += ['--preset', 'light']
elif has('--quality'):
    cmd += ['--quality', 'light']

if has('--width'):
    cmd += ['--width', '768']
if has('--height'):
    cmd += ['--height', '432']
if has('--seconds'):
    cmd += ['--seconds', '5']
elif has('--duration'):
    cmd += ['--duration', '5']
elif has('--frames'):
    cmd += ['--frames', '121']
if has('--seed'):
    cmd += ['--seed', '1']
if has('--no-two-pass'):
    cmd += ['--no-two-pass']

print("Command:", ' '.join(cmd))
print("Starting generation — expect 15-60+ min on a T4 with disk streaming.")
print("Log: /content/freevideo-test/generate.log")

t0 = time.time()
with open('/content/freevideo-test/generate.log', 'w') as log:
    p = subprocess.Popen(cmd, cwd='/content/FreeVideo',
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    for line in p.stdout:
        print(line, end='', flush=True)
        log.write(line)
        log.flush()
    rc = p.wait()
print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Exit code: {rc}")
"""))

cells.append(md("""## 10 — Verify the output

Checks the file exists, its size/duration/resolution, and prints the
`.request.json` (hardware measured, plan chosen, stage timings, memory peaks) —
that file is the evidence to report back."""))

cells.append(code("""%%bash
ls -lh /content/freevideo-test/
echo "--- ffprobe ---"
ffprobe -v error -show_entries format=duration,size \
  -show_entries stream=codec_name,width,height,r_frame_rate \
  -of default=noprint_wrappers=1 /content/freevideo-test/test-5s.mp4 2>&1 \
  || file /content/freevideo-test/test-5s.mp4
echo "--- request.json (plan, timings, memory peaks) ---"
cat /content/freevideo-test/test-5s.request.json 2>/dev/null | head -120 \
  || echo "No .request.json found"
"""))

cells.append(code("""from IPython.display import Video
Video('/content/freevideo-test/test-5s.mp4', width=640)
"""))

cells.append(md("""## 11 (optional) — Project smoke test, only if generation failed

`./test.sh` is the repo's own test script. Run it if step 9 failed, and keep its
log — it narrows down whether the failure is the install, the GPU path, or the
request itself."""))

cells.append(code("""%%bash
# OPTIONAL: only run if the generation in step 9 failed.
cd /content/FreeVideo
./test.sh 2>&1 | tee /content/freevideo-test/test-sh.log
"""))

cells.append(md("""## How to read the result (report back)

**Success** = `test-5s.mp4` exists, ~5 s, 768×432 (or the preset's resolution),
plays with video + audio. Report: generation wall time, the `.request.json`
timings/memory peaks, and the plan output from step 7 (how many of the 50
transformer blocks stayed in VRAM vs host RAM vs disk).

**Clean rejection** = FreeVideo says the minimum working set does not fit and
reports the shortfall. Report the shortfall numbers — that tells us exactly how
far off Colab free is.

**Failure modes to note:** driver < 580 refusing setup; setup.sh stalling on a
prompt (`tail` the setup log); kernel/attention-backend probe failures on Turing
(SM75); out-of-disk (check `df -h`); OOM despite retries.

**Saving the clip:** everything lives in the Colab runtime (`/content`) and is
wiped when the runtime is recycled. Download `test-5s.mp4` (and
`test-5s.request.json` + `generate.log`) from the Colab file browser, or copy
them to Google Drive from a new cell (`from google.colab import drive;
drive.mount('/content/drive')`, then copy).

**License note:** FreeVideo's code is Apache-2.0, but the MiniMax H3 model
weights carry the MiniMax H3 Community License, whose applicable territory
excludes the EU, UK, Republic of Korea and the USA. This test only downloads
and runs the weights locally for evaluation.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

os.makedirs(OUT_DIR, exist_ok=True)
with open(OUT, "w") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f"Wrote {OUT} ({len(cells)} cells)")
