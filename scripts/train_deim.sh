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

if [[ -z "${1:-}" ]]; then
    echo "usage: scripts/train_deim.sh <n|s|m> [extra args]" >&2
    exit 1
fi
SIZE="${1//$'\r'/}"       # strip CR in case of CRLF
shift

case "$SIZE" in
    n|s|m) ;;
    *) echo "size must be one of: n s m" >&2; exit 1 ;;
esac

CFG="configs/deim_dfine/deim_hgnetv2_${SIZE}_traffic_light.yml"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT/DEIM"

NPROC="${NPROC:-1}"
PORT="${PORT:-7777}"
SEED="${SEED:-0}"        # override with: SEED=42 scripts/train_deim.sh s

# Lift any seed override from passthrough args into SEED so SEED.txt agrees
# with the actual training seed. DEIM accepts both `--seed=42` and `--seed 42`.
prev=""
for arg in "$@"; do
    case "$prev" in --seed) SEED="$arg" ;; esac
    case "$arg" in --seed=*) SEED="${arg#--seed=}" ;; esac
    prev="$arg"
done

# DEIM's output_dir is fixed per config (no auto-increment), so we can
# pre-create the dir + write SEED.txt BEFORE training. This survives
# interrupted runs — a marker written only at the end vanishes on crash.
DEIM_OUTPUT_REL=$(awk '/^output_dir:/ { print $2 }' "$CFG" | tr -d '"')
DEIM_OUTPUT_ABS=""
if [[ -n "$DEIM_OUTPUT_REL" ]]; then
    # Resolve relative to DEIM/ (the cwd) since output_dir is config-relative.
    DEIM_OUTPUT_ABS="$(cd "$PROJECT_ROOT/DEIM" && mkdir -p "$DEIM_OUTPUT_REL" && cd "$DEIM_OUTPUT_REL" && pwd)"
    echo "$SEED" > "$DEIM_OUTPUT_ABS/SEED.txt"
    echo "pre-created $DEIM_OUTPUT_ABS with SEED.txt (seed=$SEED)"
fi

# `exec` replaces the shell, so torchrun's exit code IS the script's exit code
# (no risk of trailing commands masking a failed training).
exec torchrun \
    --master_port="$PORT" \
    --nproc_per_node="$NPROC" \
    train.py -c "$CFG" --use-amp --seed="$SEED" "$@"
