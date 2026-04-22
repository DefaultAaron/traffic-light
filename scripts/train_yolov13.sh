#!/usr/bin/env bash
# Train YOLOv13 on the traffic-light dataset.
#
# YOLOv13 ships custom modules (DSC3k2, HyperACE) not present in stock
# Ultralytics, so its checkpoint can't be loaded via `main.py`. Instead,
# we run it through YOLOv13's own fork of Ultralytics in a separate venv.
#
# One-time setup (on the training machine):
#   git clone https://github.com/iMoonLab/yolov13.git      # verify URL
#   uv venv yolov13/.venv --python 3.12
#   source yolov13/.venv/bin/activate
#   pip install -e yolov13/                                # or: pip install -r yolov13/requirements.txt
#   pip install -r yolov13/requirements.txt               # if -e path doesn't pull all deps
#
# Usage:
#   scripts/train_yolov13.sh s                             # YOLOv13-s defaults
#   scripts/train_yolov13.sh s --imgsz 1280 --epochs 100   # override
#
# Weights expected at weights/yolov13{n,s,m,l}.pt (project root).

set -e

SIZE="${1:?usage: scripts/train_yolov13.sh {n|s|m|l} [extra args]}"
SIZE="${SIZE//$'\r'/}"   # strip CR in case the script was checked out with CRLF
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

# YOLOv13 uses stock Ultralytics CLI conventions via its fork.
exec yolo train \
    model="$WEIGHTS" \
    data="$DATA" \
    epochs=100 \
    imgsz=640 \
    batch=-1 \
    flipud=0.0 \
    fliplr=0.0 \
    patience=20 \
    project="$ROOT/runs/detect" \
    name="yolov13${SIZE}" \
    "$@"
