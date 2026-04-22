# Traffic Light Recognition

Traffic-light detection and state classification for autonomous driving on NVIDIA Jetson AGX Orin.

- **R1 (current)**: 7-class traffic-light detector — `red`, `yellow`, `green`, `redLeft`, `greenLeft`, `redRight`, `greenRight`
- **R2 (planned)**: joint detector expanded to **10–14 classes** — 9–12 traffic-light classes (adds `forwardRed` / `forwardGreen` + PM-pending extensions) and 1–2 barrier classes (`barrier` MVP, optional `armOn` / `armOff`). Authoritative scope: [`docs/reports/phase_2_round_1_report.md`](docs/reports/phase_2_round_1_report.md) §"R2 范围扩展".

## Project Structure

```
traffic-light/
├── main.py                     # train / val / export CLI
├── configs/                    # per-model training configs
├── data/                       # datasets (gitignored)
│   ├── traffic_light.yaml      # Ultralytics dataset config
│   ├── raw/                    # originals
│   └── merged/                 # unified YOLO format
├── scripts/                    # dataset conversion + utilities
├── inference/                  # Python + C++ TRT pipelines
├── docs/                       # documentation — see docs/README.md
└── runs/                       # training outputs (gitignored)
```

## Usage

```bash
# Train
python main.py train yolo26s                          # default config
python main.py train yolo26n --epochs 50 --device 0   # override
python main.py train --resume runs/yolo26n/weights/last.pt  # resume

# Train every variant sequentially
python main.py train-all --models yolo26n yolo26s     # subset

# Validate
python main.py val runs/yolo26s/weights/best.pt --split test

# Export (format: engine | coreml | onnx; add --half for FP16)
python main.py export runs/yolo26s/weights/best.pt --format onnx --imgsz 1280
```

Ultralytics restores epochs, batch, device, and hyperparameters from `args.yaml` when `--resume` is set, so CLI overrides are ignored on resume.

Candidate model comparison, dataset licensing, and milestones: [`docs/planning/development_plan.md`](docs/planning/development_plan.md).

## Alternative Detector Track: DEIM-D-FINE

A parallel track evaluates DEIM-D-FINE (Apache-2.0) as a hedge against Ultralytics AGPL licensing and as a precision ceiling probe. See [`docs/proposals/yolo26_alternatives_survey.md`](docs/proposals/yolo26_alternatives_survey.md) for the rationale.

**Environment setup (recommended: same uv venv)**

```bash
uv sync --extra deim               # adds DEIM-specific deps to the existing venv
```

This reuses torch/torchvision already pinned by ultralytics. If resolver conflicts emerge (unlikely), fall back to a separate conda env per DEIM's README:

```bash
conda create -n deim python=3.11.9 && conda activate deim
pip install -r DEIM/requirements.txt
```

**Data conversion (one-time, shared with Ultralytics)**

```bash
uv run python scripts/yolo_to_coco.py
# → data/merged/annotations/instances_{train,val}.json
# Images are NOT copied — DEIM reads from data/merged/images/{train,val}/
```

**Training**

```bash
# S / M variants (N is skipped — see proposal §size selection)
NPROC=1 ./scripts/train_deim.sh s -t weights/deim_dfine_s_coco.pth
NPROC=1 ./scripts/train_deim.sh m -t weights/deim_dfine_m_coco.pth
```

Checkpoints and logs land in `runs/deim_dfine_{s,m}_r2/`. Use `-t` to fine-tune from COCO weights (strongly recommended on our 55k-image set).

## Deployment

Target: **NVIDIA Jetson AGX Orin 64GB**. Two equivalent backends under `inference/`:

- **Python** — `inference/demo.py`, quick dev / JSON telemetry
- **C++** — `inference/cpp/`, production on Orin

End-to-end Orin workflow (source sync, environment, CMake upgrade, ONNX head stripping, `trtexec`, run): **[`docs/integration/trt_pipeline_guide.md`](docs/integration/trt_pipeline_guide.md)**.

ROS 2 message contract for the planning module: **[`docs/integration/ros2_integration_guide.md`](docs/integration/ros2_integration_guide.md)**.

All documentation: **[`docs/README.md`](docs/README.md)**.
