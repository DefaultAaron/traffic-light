#!/usr/bin/env bash
# Re-eval DEIM-D-FINE S/M/L on the deployment-eligible best checkpoint.
# Replaces the methodologically-inconsistent eval/latest.pth (last epoch)
# with eval on best_stg2.pth (S, M) and best_stg1.pth (L).
#
# Usage (on training rig with DEIM/.venv + GPU):
#   bash scripts/eval/_remote_deim_eval.sh
#
# Outputs land under logs/deim_eval_<size>/:
#   - test_only.log    DEIM stdout/stderr
#   - eval.pth         pickled COCOeval.eval dict (precision/recall arrays)
#   - per_class.json   parsed table (machine)
#   - per_class.txt    parsed table (paste-ready markdown)
#
# Rsync the logs/deim_eval_*/ dirs back to the dev box afterwards.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO/DEIM"

DEIM_VENV="${DEIM_VENV:-.venv}"
if [ ! -d "$DEIM_VENV" ]; then
    echo "ERROR: DEIM venv at $REPO/DEIM/$DEIM_VENV not found" >&2
    exit 1
fi
. "$DEIM_VENV/bin/activate"

ANN="$REPO/data/merged/annotations/instances_val.json"
if [ ! -f "$ANN" ]; then
    echo "[info] $ANN missing — running scripts/dataset/yolo_to_coco.py --splits val"
    ( cd "$REPO" && python scripts/dataset/yolo_to_coco.py --splits val )
fi

run_eval() {
    local SIZE="$1" CKPT="$2"
    local CFG="configs/deim_dfine/deim_hgnetv2_${SIZE}_traffic_light.yml"
    local CKPT_ABS="$REPO/runs/detect/deim_dfine_${SIZE}-r1/${CKPT}"
    local OUT="$REPO/logs/deim_eval_${SIZE}"

    if [ ! -f "$CKPT_ABS" ]; then
        echo "ERROR: checkpoint not found: $CKPT_ABS" >&2
        return 1
    fi
    mkdir -p "$OUT"

    echo
    echo "==== DEIM-${SIZE^^} eval on $(basename "$CKPT_ABS") -> $OUT ===="
    python train.py \
        -c "$CFG" \
        --test-only \
        -r "$CKPT_ABS" \
        --output-dir "$OUT" \
        2>&1 | tee "$OUT/test_only.log"

    if [ ! -f "$OUT/eval.pth" ]; then
        echo "ERROR: eval.pth not produced at $OUT/eval.pth" >&2
        return 1
    fi

    python "$REPO/scripts/eval/_parse_deim_per_class.py" \
        --eval-pth "$OUT/eval.pth" \
        --ann-json "$ANN" \
        --data-yaml "$REPO/data/traffic_light.yaml" \
        --out-json "$OUT/per_class.json" \
        --out-txt  "$OUT/per_class.txt"
}

run_eval s best_stg2.pth
run_eval m best_stg2.pth
# DEIM-L: stage-2 (no-aug, ep 90-101) never beat stage-1 ep-72 global best;
# best_stg2.pth was never written. best_stg1.pth IS the deployment-eligible
# checkpoint. See DEIM/engine/solver/det_solver.py:130-149.
run_eval l best_stg1.pth

echo
echo "DONE. Rsync logs/deim_eval_{s,m,l}/ back to dev box."
