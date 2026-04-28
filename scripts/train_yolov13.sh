#!/usr/bin/env bash
# Train YOLOv13 on the traffic-light dataset.
#
# YOLOv13 ships custom modules (DSC3k2, HyperACE) not present in stock
# Ultralytics, so its checkpoint can't be loaded via `main.py`. Instead,
# we run it through YOLOv13's own fork of Ultralytics in a separate venv.
#
# One-time setup (run from traffic-light/ project root):
#   git clone https://github.com/iMoonLab/yolov13.git      # upstream ships pyproject.toml at root
#   uv venv yolov13/.venv --python 3.11                    # pinned — reqs include cp311-only wheels (onnxruntime 1.15.1, optional flash-attn)
#   source yolov13/.venv/bin/activate
#   uv pip install -r yolov13/requirements.txt             # uv venv has no `pip` — always use `uv pip`
#   uv pip install -e yolov13/                             # editable install
#
# Verify after install:
#   python -c "from ultralytics.nn.modules.block import DSC3k2; print('DSC3k2 OK')"
#   command -v yolo        # expect <repo>/yolov13/.venv/bin/yolo
#
# Usage:
#   scripts/train_yolov13.sh s                             # YOLOv13-s defaults
#   scripts/train_yolov13.sh s --imgsz 1280 --epochs 100   # override
#
# Weights expected at weights/yolov13{n,s,m,l}.pt (project root).

set -e

if [[ -z "${1:-}" ]]; then
    echo "usage: scripts/train_yolov13.sh <n|s|m|l> [extra args]" >&2
    exit 1
fi
SIZE="${1//$'\r'/}"       # strip CR in case of CRLF
shift

case "$SIZE" in
    n|s|m|l) ;;
    *) echo "size must be one of: n s m l" >&2; exit 1 ;;
esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

if [[ ! -d "$ROOT/yolov13" ]]; then
    echo "ERROR: $ROOT/yolov13 not found. Clone the YOLOv13 fork first." >&2
    echo "  git clone https://github.com/iMoonLab/yolov13.git" >&2
    exit 1
fi

if [[ ! -f "$ROOT/yolov13/.venv/bin/activate" ]]; then
    echo "ERROR: yolov13/.venv not found. One-time setup:" >&2
    echo "  uv venv $ROOT/yolov13/.venv --python 3.12" >&2
    echo "  source $ROOT/yolov13/.venv/bin/activate && pip install -e $ROOT/yolov13/" >&2
    exit 1
fi

source "$ROOT/yolov13/.venv/bin/activate"

WEIGHTS="$ROOT/weights/yolov13${SIZE}.pt"
DATA="$ROOT/data/traffic_light.yaml"
SEED="${SEED:-0}"        # override with: SEED=42 scripts/train_yolov13.sh s
RUN_NAME="yolov13${SIZE}"
PROJECT_DIR="$ROOT/runs/detect"

# Predict the dir Ultralytics will pick (it auto-increments on conflict:
# yolov13s, yolov13s2, yolov13s3, …) so we can pre-write SEED.txt BEFORE
# training. Markers written at training END vanish on interrupted runs.
PREDICTED_NAME="$RUN_NAME"
if [[ -d "$PROJECT_DIR/$RUN_NAME" ]]; then
    n=2
    while [[ -d "$PROJECT_DIR/${RUN_NAME}${n}" ]]; do
        n=$((n+1))
    done
    PREDICTED_NAME="${RUN_NAME}${n}"
fi
RUN_DIR="$PROJECT_DIR/$PREDICTED_NAME"
mkdir -p "$RUN_DIR"
echo "$SEED" > "$RUN_DIR/SEED.txt"
echo "pre-created $RUN_DIR with SEED.txt (seed=$SEED)"

# exist_ok=True tells Ultralytics not to auto-increment again — write into the
# dir we just prepared. `exec` replaces the shell, so yolo's exit code IS the
# script's exit code (no risk of trailing commands masking a failed training).
exec yolo train \
    model="$WEIGHTS" \
    data="$DATA" \
    epochs=100 \
    imgsz=640 \
    batch=-1 \
    flipud=0.0 \
    fliplr=0.0 \
    patience=20 \
    seed="$SEED" \
    project="$PROJECT_DIR" \
    name="$PREDICTED_NAME" \
    exist_ok=True \
    "$@"
