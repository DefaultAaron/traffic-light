# Traffic Light Recognition

Traffic-light detection and state classification for autonomous driving on NVIDIA Jetson AGX Orin.

- **R1 (current)**: 7-class traffic-light detector — `red`, `yellow`, `green`, `redLeft`, `greenLeft`, `redRight`, `greenRight`
- **R2 (planned)**: joint detector expanded to **10–14 classes** — 9–12 traffic-light classes (adds `forwardRed` / `forwardGreen` + PM-pending extensions) and 1–2 barrier classes (`barrier` MVP, optional `armOn` / `armOff`). Authoritative scope: [`docs/reports/phase_2_round_1_report.md`](docs/reports/phase_2_round_1_report.md) §"R2 范围扩展".

## Project Structure

```
traffic-light/
├── main.py                     # train / val / export CLI (YOLO26, YOLO11)
├── configs/                    # per-model Ultralytics training configs
├── data/                       # datasets (gitignored)
│   ├── traffic_light.yaml      # Ultralytics dataset config (7-class)
│   ├── raw/                    # originals (S2TLD / BSTLD / LISA)
│   └── merged/                 # unified YOLO format (+ COCO JSON for DEIM)
├── scripts/                    # dataset conversion + training wrappers
├── inference/                  # Python + C++ TRT pipelines
├── DEIM/                       # DEIM-D-FINE repo (Apache-2.0 alternative track)
├── yolov13/                    # YOLOv13 fork (separate venv; optional)
├── weights/                    # pretrained / downloaded checkpoints
├── docs/                       # documentation — see docs/README.md
└── runs/                       # training outputs (gitignored)
```

## Environment Setup

Main venv (`uv`, covers YOLO26 / YOLO11, ONNX export, demo tooling):

```bash
uv sync                         # base
uv sync --extra deim            # + DEIM training deps (torch already pinned)
uv sync --extra onnx            # + ONNX Runtime for dev
uv sync --extra orin            # + pycuda (aarch64, Orin only)
```

YOLOv13 needs its **own venv** because its fork ships custom modules (`DSC3k2`, `HyperACE`) that stock Ultralytics can't unpickle:

```bash
git clone https://github.com/iMoonLab/yolov13.git         # verify URL against the paper repo
# Pin 3.11 — requirements.txt includes cp311-only wheels (onnxruntime 1.15.1, optional flash-attn).
# Python 3.12 fails resolution.
uv venv yolov13/.venv --python 3.11
source yolov13/.venv/bin/activate

# Strip the flash-attn local-wheel line (upstream ships a pre-compiled cp311 wheel reference);
# flash-attn is an optional speedup, training works without it.
grep -v -i 'flash.attn\|flash_attn' yolov13/requirements.txt > /tmp/y13-reqs.txt
uv pip install -r /tmp/y13-reqs.txt
uv pip install -e yolov13/

# verify (both must succeed)
python -c "from ultralytics.nn.modules.block import DSC3k2; print('DSC3k2 OK')"
command -v yolo                                            # expect yolov13/.venv/bin/yolo
```

`uv venv` doesn't provision `pip` — always use `uv pip ...` inside the venv. If `uv pip install -e yolov13/` errors with *"does not appear to be a Python project"*, the clone is incomplete (upstream has a root-level `pyproject.toml`) — re-clone: `rm -rf yolov13 && git clone https://github.com/iMoonLab/yolov13.git` (preserve the venv first: `mv yolov13/.venv /tmp/y13venv` then move it back after re-clone).

## Data Preparation

R1 datasets (LISA / BSTLD / S2TLD) and their conversion scripts have been retired post-R2 self-collection — R2/R3 build on self-collected multimodal data per [`docs/data/r2_data_collection_sop.md`](docs/data/r2_data_collection_sop.md). The R1 conversion + merge recipe (`scripts/convert_*.py`, `scripts/merge_datasets.py`) is preserved in git history if needed for historical reproducibility.

## Training — Main Track (YOLO26)

```bash
# Train single model
uv run python main.py train yolo26s                               # default config
uv run python main.py train yolo26s --imgsz 1280 --epochs 100     # override
uv run python main.py train --resume runs/yolo26n/weights/last.pt # resume (CLI overrides ignored)

# Train multiple variants sequentially
uv run python main.py train-all --models yolo26n yolo26s --imgsz 1280

# Validate
uv run python main.py val runs/detect/yolo26s/weights/best.pt --split test --imgsz 1280

# Export (format: engine | coreml | onnx; add --half for FP16)
uv run python main.py export runs/detect/yolo26s/weights/best.pt --format onnx --imgsz 1280
```

Ultralytics restores epochs, batch, device, and hyperparameters from `args.yaml` when `--resume` is set.

## Training — Alternative Track 1: YOLOv13

Lightweight hedge against YOLO26; same AGPL license and similar deployment path. Requires the separate venv above.

```bash
./scripts/training/train_yolov13.sh s                                      # 640, 100 epochs
./scripts/training/train_yolov13.sh s --imgsz 1280 --epochs 100            # override
```

Outputs: `runs/detect/yolov13s/`. Weights expected at `weights/yolov13s.pt`.

## Training — Alternative Track 2: DEIM-D-FINE

Apache-2.0 detector as commercial-license hedge and precision ceiling probe. See [`research/surveys/alt_detector_architectures.md`](research/surveys/alt_detector_architectures.md) for rationale and size selection (S + M; N skipped).

**Download pretrained COCO checkpoints** to `weights/`:

| Size | File | Link |
|---|---|---|
| S | `weights/deim_dfine_s_coco.pth` | [Google Drive](https://drive.google.com/file/d/1tB8gVJNrfb6dhFvoHJECKOF5VpkthhfC/view) |
| M | `weights/deim_dfine_m_coco.pth` | [Google Drive](https://drive.google.com/file/d/18Lj2a6UN6k_n_UzqnJyiaiLGpDzQQit8/view) |

HGNetv2 backbones auto-download on first run.

**Train** (fine-tune from COCO — strongly recommended on 55k images):

```bash
NPROC=1 ./scripts/training/train_deim.sh s -t weights/deim_dfine_s_coco.pth
NPROC=1 ./scripts/training/train_deim.sh m -t weights/deim_dfine_m_coco.pth
```

Outputs: `runs/deim_dfine_{s,m}_r2/`. Best weight is `best_stg2.pth` (post-augmentation refinement phase). Resume via `-r runs/deim_dfine_s_r2/last.pth`.

Configs are tuned for **single RTX 4090 24GB**: batch 8 (S) / 4 (M), LRs scaled accordingly. See inline comments in `DEIM/configs/deim_dfine/deim_hgnetv2_{s,m}_traffic_light.yml` for 1280 override.

## Deployment

Target: **NVIDIA Jetson AGX Orin 64GB**. Two equivalent backends under `inference/`:

- **Python** — `inference/demo.py`, quick dev / JSON telemetry
- **C++** — `inference/cpp/`, production on Orin (Orin-measured: ~25 ms/frame at 1280 FP16)

End-to-end Orin workflow (source sync, environment, CMake upgrade, ONNX head stripping, `trtexec`, run): **[`docs/integration/trt_deployment.md`](docs/integration/trt_deployment.md)**.

ROS 2 message contract for the planning module: **[`docs/integration/ros2_contract.md`](docs/integration/ros2_contract.md)**.

### Tracker (flicker mitigation)

Both backends accept `--track` to enable ByteTrack + per-track EMA class voting. Python and C++ share JSON fixtures for parity (`tests/fixtures/tracker/`). Design and tuning knobs: **[`docs/integration/tracker.md`](docs/integration/tracker.md)**.

```bash
./inference/cpp/build/tl_demo --source video.mp4 --model best.engine \
    --track --alpha 0.3 --min-hits 3 --track-json runs/out.jsonl
```

### Demo sweep across engines

`scripts/run_demos.sh` runs `tl_demo` sequentially across every `runs/<run>/*.engine` × `demo/demo*.mp4` pair, writing to `demo/<run>/<engine_stem>/<demo_name>.mp4`. Resume-friendly (`SKIP_EXIST=1`) and single-engine-at-a-time by design (TRT contexts are not safe to share).

```bash
./scripts/run_demos.sh                              # defaults
CONF=0.3 TRACK=1 ./scripts/run_demos.sh             # override
nohup ./scripts/run_demos.sh \
    > logs/run_demos.log 2>&1 &                     # long-running sweep
```

## Documentation

- [`docs/planning/development_plan.md`](docs/planning/development_plan.md) — candidate models, dataset licensing, milestones
- [`research/surveys/alt_detector_architectures.md`](research/surveys/alt_detector_architectures.md) — YOLOv13 + DEIM-D-FINE rationale and setup
- [`research/surveys/detection_enhancements.md`](research/surveys/detection_enhancements.md) — model + system level enhancement methods beyond the three core plans (training aug, hard-neg mining, KD, SAHI, map-prior, HDR camera, multi-camera, planner-prior)
- [`research/contributions/field_gaps_and_contributions.md`](research/contributions/field_gaps_and_contributions.md) — external research view: TL field-wide failure modes + public-dataset coverage gaps + contribution candidates from R2 self-collected data (CN-gantry benchmark, TL × barrier joint, hard-condition slice, engineering refs)
- [`docs/data/r2_data_collection_sop.md`](docs/data/r2_data_collection_sop.md) — **R2 multimodal data collection + annotation SOP** — dual 8MP cameras (normal + wide) + LiDAR; sync/calibration, site coverage, 10–14 class taxonomy, hard-case slices, LiDAR-aided distance GT + vibration diagnostics, site-based splits, release prep
- [`docs/planning/temporal_optimization_plan.md`](docs/planning/temporal_optimization_plan.md) — temporal optimization track (TSM detector-level + post-detector smoothers); R2/R3 optional, parallel to main detector selection
- [`docs/planning/cross_detection_reasoning_plan.md`](docs/planning/cross_detection_reasoning_plan.md) — same-frame multi-light co-occurrence reasoning (Bayesian post-processing → CRF → Relation Network); R3 candidate, gated on R2 self-collected data validation
- [`docs/reports/phase_2_round_1_report.md`](docs/reports/phase_2_round_1_report.md) — R1 7-class results, Orin deployment, alt-track launch, R2 scope lock (living doc)
- [`docs/reports/phase_2_round_1_results.md`](docs/reports/phase_2_round_1_results.md) — raw R1 eval tables
- [`docs/_archive/phase_1_report.md`](docs/_archive/phase_1_report.md) — Phase 1 3-class baseline (historical, archived 2026-05-12; fully superseded by R1 7-class)
- [`docs/README.md`](docs/README.md) — full documentation index
