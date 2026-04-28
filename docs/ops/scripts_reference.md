# Scripts Reference

Single index for everything under `scripts/`. Each script's own header
docstring is authoritative — this file is the lookup table.

Quick map:

| Area | Scripts |
|---|---|
| [Demo sweep (Orin TRT)](#demo-sweep-orin-trt) | `run_demos.sh` |
| [Training](#training) | `train_deim.sh`, `train_yolov13.sh` |
| [Dataset conversion + merge](#dataset-conversion--merge) | `convert_bstld.py`, `convert_lisa.py`, `convert_s2tld.py`, `merge_datasets.py`, `yolo_to_coco.py` |
| [Manual annotation](#manual-annotation) | `annotate_bstld.py`, `annotate_s2tld.py` |
| [Model export](#model-export) | `strip_yolo26_head.py` |
| [Flicker / tracker validation](#flicker--tracker-validation) | `measure_flicker.py`, `validate_flicker_reduction.py` |
| [Network / ops](#network--ops) | `setup_reverse_tunnel.sh` |

---

## Demo sweep (Orin TRT)

`run_demos.sh` runs every `(run × engine × demo)` triple through `tl_demo`
and writes annotated `.mp4` outputs. Detector-only and tracker modes
coexist under the same `OUT_DIR` because tracker mode auto-suffixes the
run dir with `_tracker`. Sequential by design — each call holds the
GPU/TRT context.

```bash
./scripts/run_demos.sh                    # detector-only → demo/<run>/<engine>/<demo>.mp4
TRACK=1 ./scripts/run_demos.sh            # tracker     → demo/<run>_tracker/<engine>/<demo>.mp4
```

### Env overrides (always available)

| Env var | Default | Effect |
|---|---|---|
| `TL_DEMO` | `inference/cpp/build/tl_demo` | Path to the `tl_demo` binary |
| `RUNS_DIR` | `runs` | Root containing `<run>/*.engine` |
| `DEMOS_DIR` | `demo` | Root containing `demo*.mp4` clips |
| `OUT_DIR` | `$DEMOS_DIR` | Output root |
| `CONF` | `0.25` | Detector confidence threshold |
| `TRACK` | `0` | `1` enables `--track` and the `_tracker` suffix |
| `SKIP_EXIST` | `1` | `1` skips outputs already on disk (resume-friendly) |
| `OVERWRITE` | `0` | `1` deletes stale output (and any companion `.tracks.jsonl`) before re-running; forces `SKIP_EXIST=0` |

### Tracker tuning (`TRACK=1` only)

Unset env vars fall through to the `tl_demo` built-in defaults.

| Env var | Effect |
|---|---|
| `ALPHA` | Tracker EMA alpha |
| `MIN_HITS` | Min hits before a track is confirmed |
| `HIGH_THRESH` | First-pass IoU/score threshold |
| `MATCH_THRESH` | Match cost cutoff |
| `TRACK_BUFFER` | Frames a lost track survives |
| `SAVE_TRACK_JSON` | `1` writes `<demo>.tracks.jsonl` next to the mp4 |

Resolution is inferred from engine filename: `*1536* → 1536`, `*1280* →
1280`, otherwise `640`.

### Common combinations

```bash
# Resume — only fills in missing outputs.
./scripts/run_demos.sh

# Force regenerate everything (e.g. after retraining).
OVERWRITE=1 ./scripts/run_demos.sh

# Tracker sweep, dump parity JSONL alongside each mp4, fresh output dir.
TRACK=1 SAVE_TRACK_JSON=1 \
    RUNS_DIR=runs/yolo26m-r1 OUT_DIR=demo_v2 \
    ./scripts/run_demos.sh

# Higher confidence tracker sweep, regenerate.
TRACK=1 CONF=0.35 OVERWRITE=1 \
    ./scripts/run_demos.sh
```

### Output formats

`tl_demo --save` writes the annotated `.mp4`. With `TRACK=1
SAVE_TRACK_JSON=1`, a parity `<demo>.tracks.jsonl` is also written. The
JSONL schema (`{"frame": N, "tracks": [...]}`) matches `inference/demo.py
--track-json`, so Python and C++ outputs are diffable for parity.

For flicker analysis, see [`measure_flicker.py`](#flicker--tracker-validation)
below — it reads a different schema (`--json`, not `--track-json`), so
generate it with a one-off `inference/demo.py` invocation rather than
the sweep.

---

## Training

### `train_yolov13.sh`

YOLOv13 ships custom modules (DSC3k2, HyperACE) that need its own
Ultralytics fork — keep it in a separate venv:

```bash
git clone https://github.com/iMoonLab/yolov13.git
uv venv yolov13/.venv --python 3.11
source yolov13/.venv/bin/activate
uv pip install -r yolov13/requirements.txt
uv pip install -e yolov13/

# Verify
python -c "from ultralytics.nn.modules.block import DSC3k2; print('DSC3k2 OK')"
```

Then (extra args are forwarded to Ultralytics `yolo train`, which uses
`key=value` syntax — not `--flag`):

```bash
./scripts/train_yolov13.sh s
./scripts/train_yolov13.sh s imgsz=1280 epochs=100
```

Weights expected at `weights/yolov13{n,s,m,l}.pt`. **Python 3.11 only**
(the upstream `requirements.txt` pins cp311-only wheels).

### `train_deim.sh`

DEIM-D-FINE training. Needs the COCO-format dataset first
(`uv run python scripts/yolo_to_coco.py`). Single-GPU is the default;
use `NPROC=N` for multi-GPU. Extra args after the size are forwarded to
DEIM's `train.py`, **not** to `torchrun` — so don't pass torchrun flags
like `--nproc_per_node` here.

```bash
# Single GPU (default)
./scripts/train_deim.sh s

# Multi-GPU
NPROC=4 ./scripts/train_deim.sh s

# Fine-tune from COCO checkpoint (-t is a train.py flag)
./scripts/train_deim.sh m -t weights/deim_dfine_m_coco.pth
```

DEIM uses its own standalone venv on the training server. The main
project venv (`uv` workspace at the repo root) is YOLO26-only — do not
add DEIM torch/torchvision pins to project `pyproject.toml`.

---

## Dataset conversion + merge

Run once per dataset, then merge.

| Script | Reads | Writes |
|---|---|---|
| `convert_bstld.py` | `data/raw/BSTLD/{train,test}` (YAML + Pascal VOC) | `data/raw/BSTLD/yolo_labels/*.txt` |
| `convert_lisa.py` | `data/raw/LISA/Annotations/.../frameAnnotationsBOX.csv` | `data/raw/LISA/yolo_labels/*.txt` |
| `convert_s2tld.py` | `data/raw/S2TLD/{,normal_1,normal_2}/Annotations-fix/*.xml` | `data/raw/S2TLD/yolo_labels/*.txt` |

Then merge:

```bash
uv run python scripts/merge_datasets.py                         # default: 80/20 split, seed=42
uv run python scripts/merge_datasets.py --val-ratio 0.15 --seed 7
```

Output: `data/merged/{images,labels}/{train,val}/` with dataset-prefixed
filenames so collisions are impossible.

For DEIM (COCO-format) on top of the merged dataset:

```bash
uv run python scripts/yolo_to_coco.py
uv run python scripts/yolo_to_coco.py --splits train val   # explicit
```

Writes `data/merged/annotations/instances_{train,val}.json`. Categories
are 0-indexed to match `traffic_light.yaml` (so set DEIM's
`remap_mscoco_category: False`, `num_classes: 7`).

---

## Manual annotation

Tk GUIs for re-annotating BSTLD test set + S2TLD with directional labels.

```bash
python scripts/annotate_bstld.py
python scripts/annotate_s2tld.py
```

Both: arrow keys to navigate, click-drag to draw, auto-save 500ms after
last edit. `Ctrl+q` to quit. Reads/writes the dataset's
`Annotations-fix/` (or `annotations_fix/`) directory.

---

## Model export

### `strip_yolo26_head.py`

YOLO26's in-graph NMS-free head can't be parsed by TRT 8.5.2 (JetPack
5.1). This script rewrites the exported ONNX so the final output is the
head `Concat` (`[1, 4+nc, N]` of decoded `xyxy || sigmoid class scores`).
The C++ pipeline already decodes that directly.

```bash
uv run python scripts/strip_yolo26_head.py best.onnx best_stripped.onnx
uv run python scripts/strip_yolo26_head.py best.onnx best_stripped.onnx --num-classes 7
```

Then on the Orin: `trtexec --onnx=best_stripped.onnx --saveEngine=...
--fp16`.

---

## Flicker / tracker validation

### `measure_flicker.py`

Computes per-track raw-vs-smoothed class-flip counts. **Reads
`inference/demo.py --json` output**, where each record is
`{"frame": N, "detections": [...]}` and tracked detections carry
`tracking_id` + `raw_class_id`. The sweep scripts' `--track-json`
output uses a different schema (`tracks[]`) and is **not** compatible —
feeding `.tracks.jsonl` here would silently report 0 detections.

```bash
# Generate the right schema with a one-off demo run
python -m inference.demo --source demo/demo.mp4 --model weights/best.engine \
    --track --no-show --json > /tmp/tracked.jsonl

# Then analyze
uv run python scripts/measure_flicker.py /tmp/tracked.jsonl
uv run python scripts/measure_flicker.py /tmp/tracked.jsonl --dump-per-track
```

Reads from stdin if no path given (`-`).

### `validate_flicker_reduction.py`

Synthetic stress test for `TrackSmoother`. Drives 300 frames of a
deterministically noisy detection stream and reports raw-vs-smoothed flip
reduction against the ≥50% gate.

```bash
uv run python scripts/validate_flicker_reduction.py
uv run python scripts/validate_flicker_reduction.py --seed 1 --flip-rate 0.4
```

Defaults: `--frames 300 --flip-rate 0.3 --seed 0`. Useful when the real
demo footage is too clean to exercise the reduction path.

---

## Network / ops

Workaround for the unstable campus network — see
[`tailscale_runbook.md`](tailscale_runbook.md) for the underlying root
cause and triage.

### `setup_reverse_tunnel.sh`

Installs an `autossh` reverse tunnel (systemd-managed) from this host to
a public VPS, so SSH still works when Tailscale's control plane is being
throttled. This is the production workaround.

```bash
sudo VPS_HOST=vps.example.com VPS_USER=ubuntu VPS_PORT=22 \
     REMOTE_PORT=2222 LOCAL_SSH_PORT=22 \
     bash scripts/setup_reverse_tunnel.sh

sudo bash scripts/setup_reverse_tunnel.sh --status
sudo bash scripts/setup_reverse_tunnel.sh --disable
```

After setup: `ssh -p 2222 jun-user@vps.example.com`. VPS sshd needs
`GatewayPorts yes`.

---

## When in doubt

Each script's top-of-file docstring is the authoritative source of truth
— this index just points you to the right one. If something here drifts
from a script's actual behavior, the script wins.
