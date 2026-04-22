#!/usr/bin/env bash
# Train DEIM-D-FINE on the traffic-light dataset.
#
# Usage:
#   scripts/train_deim.sh <size> [extra torchrun args...]
#   scripts/train_deim.sh s --nproc_per_node=1                     # single GPU
#   scripts/train_deim.sh m -t weights/deim_dfine_m_coco.pth       # fine-tune from COCO checkpoint
#
# Prerequisite: run `uv run python scripts/yolo_to_coco.py` once so
# data/merged/annotations/instances_{train,val}.json exist.

set -e

SIZE="${1:?usage: scripts/train_deim.sh {n|s|m} [extra args]}"
shift

case "$SIZE" in
    n|s|m) ;;
    *) echo "size must be one of: n s m" >&2; exit 1 ;;
esac

CFG="configs/deim_dfine/deim_hgnetv2_${SIZE}_traffic_light.yml"
cd "$(dirname "$0")/../DEIM"

NPROC="${NPROC:-1}"
PORT="${PORT:-7777}"

exec torchrun \
    --master_port="$PORT" \
    --nproc_per_node="$NPROC" \
    train.py -c "$CFG" --use-amp --seed=0 "$@"
