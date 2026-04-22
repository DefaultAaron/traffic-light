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
uv venv yolov13/.venv --python 3.12
source yolov13/.venv/bin/activate && pip install -e yolov13/
```

## Data Preparation

Raw datasets are placed under `data/raw/{S2TLD,BSTLD,LISA}/` (see `scripts/convert_*.py`). Then:

```bash
# Per-dataset YOLO label conversion (run each once)
uv run python scripts/convert_s2tld.py
uv run python scripts/convert_bstld.py
uv run python scripts/convert_lisa.py

# Merge + stratified 80/20 split → data/merged/{images,labels}/{train,val}
uv run python scripts/merge_datasets.py

# COCO JSON for DEIM (images NOT duplicated; points at data/merged/images/)
uv run python scripts/yolo_to_coco.py
```

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
./scripts/train_yolov13.sh s                                      # 640, 100 epochs
./scripts/train_yolov13.sh s --imgsz 1280 --epochs 100            # override
```

Outputs: `runs/detect/yolov13s/`. Weights expected at `weights/yolov13s.pt`.

## Training — Alternative Track 2: DEIM-D-FINE

Apache-2.0 detector as commercial-license hedge and precision ceiling probe. See [`docs/proposals/yolo26_alternatives_survey.md`](docs/proposals/yolo26_alternatives_survey.md) for rationale and size selection (S + M; N skipped).

**Download pretrained COCO checkpoints** to `weights/`:

| Size | File | Link |
|---|---|---|
| S | `weights/deim_dfine_s_coco.pth` | [Google Drive](https://drive.google.com/file/d/1tB8gVJNrfb6dhFvoHJECKOF5VpkthhfC/view) |
| M | `weights/deim_dfine_m_coco.pth` | [Google Drive](https://drive.google.com/file/d/18Lj2a6UN6k_n_UzqnJyiaiLGpDzQQit8/view) |

HGNetv2 backbones auto-download on first run.

**Train** (fine-tune from COCO — strongly recommended on 55k images):

```bash
NPROC=1 ./scripts/train_deim.sh s -t weights/deim_dfine_s_coco.pth
NPROC=1 ./scripts/train_deim.sh m -t weights/deim_dfine_m_coco.pth
```

Outputs: `runs/deim_dfine_{s,m}_r2/`. Best weight is `best_stg2.pth` (post-augmentation refinement phase). Resume via `-r runs/deim_dfine_s_r2/last.pth`.

Configs are tuned for **single RTX 4090 24GB**: batch 8 (S) / 4 (M), LRs scaled accordingly. See inline comments in `DEIM/configs/deim_dfine/deim_hgnetv2_{s,m}_traffic_light.yml` for 1280 override.

## Deployment

Target: **NVIDIA Jetson AGX Orin 64GB**. Two equivalent backends under `inference/`:

- **Python** — `inference/demo.py`, quick dev / JSON telemetry
- **C++** — `inference/cpp/`, production on Orin

End-to-end Orin workflow (source sync, environment, CMake upgrade, ONNX head stripping, `trtexec`, run): **[`docs/integration/trt_pipeline_guide.md`](docs/integration/trt_pipeline_guide.md)**.

ROS 2 message contract for the planning module: **[`docs/integration/ros2_integration_guide.md`](docs/integration/ros2_integration_guide.md)**.

## Documentation

- [`docs/planning/development_plan.md`](docs/planning/development_plan.md) — candidate models, dataset licensing, milestones
- [`docs/proposals/yolo26_alternatives_survey.md`](docs/proposals/yolo26_alternatives_survey.md) — YOLOv13 + DEIM-D-FINE rationale and setup
- [`docs/proposals/temporal_encoder_feasibility.md`](docs/proposals/temporal_encoder_feasibility.md) — LSTM/GRU/tracker analysis (deferred post-5/15)
- [`docs/reports/phase_1_report.md`](docs/reports/phase_1_report.md) — 3-class baseline results
- [`docs/reports/phase_2_round_1_report.md`](docs/reports/phase_2_round_1_report.md) — 7-class R1 + R2 scope lock
- [`docs/README.md`](docs/README.md) — full documentation index
