#!/usr/bin/env bash
# Export a trained DEIM-D-FINE checkpoint to ONNX, then optionally to a
# TensorRT engine. Mirrors the train_deim.sh contract for size selection.
#
# Pipeline:
#   <ckpt.pth> -- _export_deim_onnx.py --> <ckpt.onnx>    (CPU, in DEIM venv)
#   <ckpt.onnx> --      trtexec       --> <ckpt.engine>   (Orin only)
#
# The shared inference/trt_pipeline.{py,cpp} auto-detects the DEIM
# arch from tensor names (orig_target_sizes input + labels/boxes/scores
# outputs) — no flag needed downstream.
#
# Usage:
#   scripts/export_deim.sh <size> <ckpt.pth> [--build-engine]
#   scripts/export_deim.sh s runs/detect/deim_dfine_s-r1/best_stg2.pth
#   scripts/export_deim.sh s runs/detect/deim_dfine_s-r1/best_stg2.pth --build-engine
#
# Outputs (next to the input checkpoint):
#   <ckpt>.onnx        full deploy graph (model + postprocessor.deploy())
#   <ckpt>.onnx.imgsz  sidecar with the spatial size baked in (one integer)
#   <ckpt>.engine      FP16 TRT engine, only when --build-engine is passed
#                      and trtexec is on PATH (Orin)
#
# Why no IMGSZ override:
#   The DEIM traffic_light configs declare eval_spatial_size in the YAML
#   chain. Changing the export size without retraining (which requires
#   resize ops, base_size, batch, LR all updated in tandem — see comments
#   in DEIM/configs/deim_dfine/deim_hgnetv2_s_traffic_light.yml lines 13-18)
#   silently produces a model with bad accuracy. We read the size from
#   the resolved config and use that as the single source of truth.
#
# Env overrides:
#   PYTHON           Python interpreter for the DEIM venv (default: python)
#   FP16             1 to ask trtexec for --fp16 (default: 1).
#   WORKSPACE_MB     trtexec --memPoolSize=workspace:N (default: 4096).
#   TRTEXEC          trtexec binary path (default: trtexec on PATH).

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 2 ]]; then
    cat <<'EOF' >&2
usage: scripts/export_deim.sh <n|s|m> <ckpt.pth> [--build-engine]

env: PYTHON, FP16, WORKSPACE_MB, TRTEXEC
EOF
    exit 1
fi

SIZE="${1//$'\r'/}"
CKPT="$2"
BUILD_ENGINE=0
[[ "${3:-}" == "--build-engine" ]] && BUILD_ENGINE=1

case "$SIZE" in
    n|s|m) ;;
    *) echo "size must be one of: n s m" >&2; exit 1 ;;
esac

if [[ ! -f "$CKPT" ]]; then
    echo "checkpoint not found: $CKPT" >&2
    exit 1
fi

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="configs/deim_dfine/deim_hgnetv2_${SIZE}_traffic_light.yml"
PYBIN="${PYTHON:-python}"

# DEIM's export_onnx.py writes the .onnx next to the .pth (replaces extension)
# and reads the config relative to DEIM/. Resolve to an absolute path on the
# host so it survives the `cd DEIM`.
CKPT_ABS="$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")"
ONNX_ABS="${CKPT_ABS%.pth}.onnx"
SIDECAR="${ONNX_ABS}.imgsz"
ENGINE_ABS="${CKPT_ABS%.pth}.engine"

echo "=== ONNX export ==="
echo "  size:       $SIZE"
echo "  config:     DEIM/$CFG"
echo "  checkpoint: $CKPT_ABS"
echo "  onnx out:   $ONNX_ABS"
echo "  imgsz:      <read from config eval_spatial_size>"

# scripts/_export_deim_onnx.py reads eval_spatial_size from the resolved
# YAML and bakes that into the ONNX (no override). The upstream
# DEIM/tools/deployment/export_onnx.py hardcodes 640 unconditionally, so we
# never call it directly.
"$PYBIN" "$PROJECT_ROOT/scripts/_export_deim_onnx.py" \
    --check --simplify \
    -c "$CFG" \
    -r "$CKPT_ABS"

if [[ ! -f "$ONNX_ABS" ]]; then
    echo "_export_deim_onnx.py finished but $ONNX_ABS missing — check stderr above" >&2
    exit 1
fi
if [[ ! -f "$SIDECAR" ]]; then
    echo "_export_deim_onnx.py finished but $SIDECAR missing — check stderr above" >&2
    exit 1
fi
IMGSZ=$(tr -d '[:space:]' < "$SIDECAR")
if ! [[ "$IMGSZ" =~ ^[0-9]+$ ]] || [[ "$IMGSZ" -lt 64 ]]; then
    echo "sidecar $SIDECAR contains invalid imgsz '$IMGSZ'" >&2
    exit 1
fi
echo "ONNX written: $ONNX_ABS  (imgsz=$IMGSZ from sidecar)"

if [[ "$BUILD_ENGINE" -ne 1 ]]; then
    echo
    echo "skip engine build (pass --build-engine to also produce $ENGINE_ABS)"
    exit 0
fi

TRTEXEC_BIN="${TRTEXEC:-trtexec}"
if ! command -v "$TRTEXEC_BIN" >/dev/null 2>&1; then
    echo "trtexec not found on PATH — engine build only works on Orin." >&2
    echo "Run on the deployment host:  scripts/export_deim.sh $SIZE $CKPT --build-engine" >&2
    exit 1
fi

FP16="${FP16:-1}"
WORKSPACE_MB="${WORKSPACE_MB:-4096}"
FP16_FLAG=""
[[ "$FP16" == "1" ]] && FP16_FLAG="--fp16"

# DEIM's exporter declares a dynamic batch axis ("N"). For deployment we
# bake batch=1 by setting min/opt/max all to 1, which lets TRT specialize
# the kernels and avoids the dynamic-shape penalty. Optimizing further
# (multi-batch) would require shapes like 1x..,opt=4x..,max=8x...
echo
echo "=== Engine build ==="
echo "  onnx:       $ONNX_ABS"
echo "  engine:     $ENGINE_ABS"
echo "  fp16:       $FP16  workspace_mb: $WORKSPACE_MB"

"$TRTEXEC_BIN" \
    --onnx="$ONNX_ABS" \
    --saveEngine="$ENGINE_ABS" \
    $FP16_FLAG \
    --memPoolSize=workspace:"$WORKSPACE_MB" \
    --minShapes=images:1x3x"$IMGSZ"x"$IMGSZ",orig_target_sizes:1x2 \
    --optShapes=images:1x3x"$IMGSZ"x"$IMGSZ",orig_target_sizes:1x2 \
    --maxShapes=images:1x3x"$IMGSZ"x"$IMGSZ",orig_target_sizes:1x2

echo "Engine written: $ENGINE_ABS"
