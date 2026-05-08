#!/usr/bin/env bash
# Export an Ultralytics YOLO checkpoint to ONNX, strip its NMS-free head,
# then optionally build a TensorRT engine via trtexec. Mirrors the
# scripts/export_deim.sh contract for size, precision, _fp32 suffix, and
# SKIP_EXPORT semantics.
#
# Pipeline (with --build-engine):
#   <ckpt.pt>  -- yolo export format=onnx --> <ckpt>.onnx                (full head)
#   <ckpt>.onnx -- strip_yolo26_head.py  --> <ckpt>_stripped.onnx        (raw [1,4+nc,N])
#   <ckpt>_stripped.onnx -- trtexec      --> <ckpt>.engine               (TRT FP16/FP32)
#                                            <ckpt>.engine.meta.json    (atomic sidecar)
#
# Why the strip exists: TRT 8.5.2 (JetPack 5.1.x) cannot parse YOLO26's
# in-graph NMS-free head due to a shape-propagation bug around
# `ReduceMax -> TopK` (the assertion `axis <= nbDims` fails — TRT sees an
# inconsistent rank). The C++ pipeline (inference/cpp/src/trt_pipeline.cpp)
# already decodes the raw [1, 4+nc, N] anchor output with confidence
# thresholding, so the TopK is unnecessary at inference. See
# scripts/strip_yolo26_head.py for the full rationale.
#
# Why we do NOT use `yolo export format=engine`:
#   yolo's unified pipeline calls TRT internally without the strip step,
#   producing engines that either fail to build (TRT 8.5.2 parser bug) or
#   produce graphs the C++ pipeline can't decode. The split-and-strip
#   flow above is the only correct path on JetPack 5.1.
#
# Target detector: YOLO26 (the path-segment regex below enforces this; set
# ALLOW_NON_YOLO26=1 to bypass at caller's risk).
#
# Usage:
#   scripts/export_yolo.sh <size> <ckpt.pt> [--build-engine]
#   scripts/export_yolo.sh s runs/yolo26_s-r1/weights/best.pt
#   scripts/export_yolo.sh s runs/yolo26_s-r1/weights/best.pt --build-engine
#
# Outputs (next to the input checkpoint):
#   <ckpt>.onnx                 deploy graph (full head, for Python ORT parity)
#   <ckpt>_stripped.onnx        head-stripped ONNX (TRT 8.5 input)
#   <ckpt>.engine               FP16 TRT engine (default)
#   <ckpt>_fp32.engine          FP32 TRT engine (FP16=0); coexists with FP16
#   <engine>.meta.json          atomic precision metadata sidecar
#
# Env overrides:
#   YOLO_BIN         yolo CLI path (default: `yolo` on PATH)
#   PYTHON           Python interpreter (default: auto-detect python ->
#                    python3). Used for onnx.checker, the strip script,
#                    and the JSON sidecar writer.
#   TRTEXEC          trtexec binary path (default: `trtexec` on PATH).
#                    Required when --build-engine is passed.
#   FP16             1 for FP16 (default), 0 for FP32. FP32 lands at
#                    <ckpt>_fp32.engine (sidecar precision="fp32").
#   SKIP_EXPORT      1 to reuse existing .onnx + _stripped.onnx (default 0).
#   WORKSPACE_GB     trtexec workspace size in GB (default: 4). Converted
#                    to MB for trtexec --memPoolSize. Note: DEIM uses
#                    WORKSPACE_MB; YOLO uses GB to keep human-friendly
#                    units at this script's surface.
#   IMGSZ            override input size (default: read from .pt baked
#                    imgsz). Most callers should leave unset.
#   NUM_CLASSES      override class count (default: read from .pt's
#                    model.names). Useful when .pt was trained with a
#                    custom class list and Ultralytics' loader is finicky.
#   ALLOW_NON_YOLO26 1 bypasses the YOLO26 path-segment regex gate.
#   ALLOW_LARGE_WORKSPACE 1 bypasses the 32GB cap on WORKSPACE_GB.

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 2 ]]; then
    cat <<'EOF' >&2
usage: scripts/export_yolo.sh <n|s|m|l|x> <ckpt.pt> [--build-engine]

env: YOLO_BIN, PYTHON, TRTEXEC, FP16, SKIP_EXPORT, WORKSPACE_GB, IMGSZ,
     NUM_CLASSES, ALLOW_NON_YOLO26, ALLOW_LARGE_WORKSPACE
EOF
    exit 1
fi

SIZE="${1//$'\r'/}"
CKPT="$2"
BUILD_ENGINE=0
[[ "${3:-}" == "--build-engine" ]] && BUILD_ENGINE=1

case "$SIZE" in
    n|s|m|l|x) ;;
    *) echo "size must be one of: n s m l x" >&2; exit 1 ;;
esac

if [[ ! -f "$CKPT" ]]; then
    echo "checkpoint not found: $CKPT" >&2
    exit 1
fi

# Detector-family soft gate: YOLO26 only. Path SEGMENT must start with
# "yolo26" — substring `*yolo26*` would let `runs/old_yolo26_failed/yolo13_v2/best.pt`
# pass even though the actual variant is yolo13.
ALLOW_NON_YOLO26="${ALLOW_NON_YOLO26:-0}"
if [[ "$ALLOW_NON_YOLO26" != "1" ]]; then
    if ! [[ "/$CKPT" =~ /yolo26[^/]* ]]; then
        echo "ERROR: checkpoint path '$CKPT' has no path segment starting with 'yolo26'." >&2
        echo "       This script is validated for YOLO26 only in R2." >&2
        echo "       Set ALLOW_NON_YOLO26=1 to bypass (at caller's risk)." >&2
        exit 1
    fi
fi

CKPT_ABS="$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")"
ONNX_ABS="${CKPT_ABS%.pt}.onnx"
STRIPPED_ABS="${CKPT_ABS%.pt}_stripped.onnx"

FP16="${FP16:-1}"
case "$FP16" in
    0|1) ;;
    *) echo "FP16 must be 0 or 1, got '$FP16'" >&2; exit 1 ;;
esac
if [[ "$FP16" == "1" ]]; then
    ENGINE_ABS="${CKPT_ABS%.pt}.engine"
    PRECISION_LABEL="fp16"
else
    ENGINE_ABS="${CKPT_ABS%.pt}_fp32.engine"
    PRECISION_LABEL="fp32"
fi

YOLO_CMD="${YOLO_BIN:-yolo}"
if ! command -v "$YOLO_CMD" >/dev/null 2>&1; then
    echo "yolo CLI not found at '$YOLO_CMD'." >&2
    echo "Install ultralytics or set YOLO_BIN=<path-to-yolo>" >&2
    exit 1
fi

# Auto-detect Python: prefer `python` for parity with export_deim.sh,
# else fall back to `python3`. Preflight `import onnx` to fail fast.
PYBIN="${PYTHON:-}"
if [[ -z "$PYBIN" ]]; then
    if command -v python >/dev/null 2>&1; then
        PYBIN=python
    elif command -v python3 >/dev/null 2>&1; then
        PYBIN=python3
    else
        echo "no python interpreter found on PATH; set PYTHON=<path>" >&2
        exit 1
    fi
fi
echo "  python:     $PYBIN ($("$PYBIN" --version 2>&1))"
if ! "$PYBIN" -c "import onnx" 2>/dev/null; then
    echo "ERROR: $PYBIN does not have the 'onnx' package installed." >&2
    echo "       Install via 'pip install onnx', or set PYTHON=<other-python>" >&2
    exit 1
fi
if ! "$PYBIN" -c "import onnx_graphsurgeon" 2>/dev/null; then
    echo "ERROR: $PYBIN does not have 'onnx_graphsurgeon' installed (required by strip_yolo26_head.py)." >&2
    echo "       Install via 'pip install onnx-graphsurgeon', or set PYTHON=<other-python>" >&2
    exit 1
fi

SKIP_EXPORT="${SKIP_EXPORT:-0}"
WORKSPACE_GB="${WORKSPACE_GB:-4}"
if ! [[ "$WORKSPACE_GB" =~ ^[0-9]+$ ]] || [[ "$WORKSPACE_GB" -lt 1 ]]; then
    echo "WORKSPACE_GB must be a positive integer (GB), got '$WORKSPACE_GB'" >&2
    echo "Note: DEIM uses WORKSPACE_MB; YOLO uses WORKSPACE_GB. Did you copy from DEIM?" >&2
    exit 1
fi
ALLOW_LARGE_WORKSPACE="${ALLOW_LARGE_WORKSPACE:-0}"
if [[ "$WORKSPACE_GB" -gt 32 && "$ALLOW_LARGE_WORKSPACE" != "1" ]]; then
    echo "WORKSPACE_GB=$WORKSPACE_GB exceeds 32GB cap (likely a unit mistake)." >&2
    echo "Set ALLOW_LARGE_WORKSPACE=1 to bypass." >&2
    exit 1
fi
WORKSPACE_MB=$((WORKSPACE_GB * 1024))

IMGSZ_FLAG=()
[[ -n "${IMGSZ:-}" ]] && IMGSZ_FLAG=("imgsz=$IMGSZ")

# Resolve the strip script next to this shell script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRIP_SCRIPT="$SCRIPT_DIR/strip_yolo26_head.py"
if [[ ! -f "$STRIP_SCRIPT" ]]; then
    echo "strip script not found: $STRIP_SCRIPT" >&2
    exit 1
fi

# === ONNX export (always — both build modes need a fresh ONNX) ===
echo "=== ONNX export ==="
echo "  size:       $SIZE"
echo "  checkpoint: $CKPT_ABS"
echo "  onnx out:   $ONNX_ABS"

# ONNX backup/restore guards. yolo export overwrites $ONNX_ABS
# unconditionally; if validation fails after the new write, restore the
# prior known-good ONNX from the backup.
ONNX_BACKUP=""
ONNX_VALIDATED=0
restore_onnx_on_fail() {
    local rc=$?
    if (( rc != 0 )); then
        if [[ -n "$ONNX_BACKUP" && -f "$ONNX_BACKUP" ]]; then
            mv "$ONNX_BACKUP" "$ONNX_ABS" 2>/dev/null || true
            echo "Cleanup: restored prior $ONNX_ABS from backup" >&2
        elif [[ -z "$ONNX_BACKUP" && -f "$ONNX_ABS" && "$ONNX_VALIDATED" -ne 1 ]]; then
            rm -f "$ONNX_ABS"
            echo "Cleanup: removed unvalidated freshly-written $ONNX_ABS" >&2
        fi
    fi
}

if [[ "$SKIP_EXPORT" == "1" && -s "$ONNX_ABS" ]]; then
    if [[ "$CKPT_ABS" -nt "$ONNX_ABS" ]]; then
        echo "WARNING: $CKPT_ABS is newer than $ONNX_ABS — SKIP_EXPORT may use a stale ONNX." >&2
    fi
    if ! "$PYBIN" -c "import sys, onnx; onnx.checker.check_model(onnx.load(sys.argv[1]))" "$ONNX_ABS" 2>/dev/null; then
        echo "SKIP_EXPORT=1 but $ONNX_ABS failed onnx.checker (corrupt/truncated)." >&2
        echo "Delete the .onnx and re-run without SKIP_EXPORT=1 to regenerate." >&2
        exit 1
    fi
    echo "SKIP_EXPORT=1 and existing ONNX validated — reusing $ONNX_ABS"
    ONNX_VALIDATED=1
else
    if [[ "$SKIP_EXPORT" == "1" ]]; then
        echo "SKIP_EXPORT=1 requested but $ONNX_ABS missing/empty — running full export"
    fi
    if [[ -f "$ONNX_ABS" ]]; then
        ONNX_BACKUP="${ONNX_ABS}.bak.$$"
        mv "$ONNX_ABS" "$ONNX_BACKUP"
    fi
    trap restore_onnx_on_fail EXIT
    "$YOLO_CMD" export \
        model="$CKPT_ABS" \
        format=onnx \
        simplify=True \
        ${IMGSZ_FLAG[@]+"${IMGSZ_FLAG[@]}"}
    if ! "$PYBIN" -c "import sys, onnx; onnx.checker.check_model(onnx.load(sys.argv[1]))" "$ONNX_ABS" 2>/dev/null; then
        echo "freshly-exported $ONNX_ABS failed onnx.checker validation." >&2
        exit 1
    fi
    ONNX_VALIDATED=1
    backup_to_remove="$ONNX_BACKUP"
    ONNX_BACKUP=""
    if [[ -n "$backup_to_remove" && -f "$backup_to_remove" ]]; then
        rm -f "$backup_to_remove"
    fi
    trap - EXIT
fi

if [[ ! -s "$ONNX_ABS" ]]; then
    echo "$ONNX_ABS missing or empty after export step" >&2
    exit 1
fi
echo "ONNX written + validated: $ONNX_ABS"

# === Strip YOLO26 head ===
# Required step (see header docstring): TRT 8.5 cannot parse YOLO26's full
# head, and the C++ pipeline expects raw [1, 4+nc, N] output. Done in both
# the no-build-engine path (so a useful stripped artifact is on disk for
# future engine builds elsewhere) and the engine-build path.
echo
echo "=== Strip YOLO26 head ==="

# Read num_classes + imgsz from the .pt (one Python call). NUM_CLASSES env
# override wins if set. Imgsz inferred from baked-in arg, falls back to
# 1280 (R2 default) if the .pt args are unusual.
read DETECTED_NC DETECTED_IMGSZ <<<"$("$PYBIN" - "$CKPT_ABS" <<'PYEOF'
import sys
from ultralytics import YOLO
m = YOLO(sys.argv[1])
nc = len(m.names)
imgsz = None
try:
    imgsz = m.overrides.get("imgsz") or (
        m.model.args.get("imgsz") if hasattr(m.model, "args") else None
    )
except Exception:
    pass
if isinstance(imgsz, (list, tuple)):
    imgsz = imgsz[0]
if not imgsz:
    imgsz = 1280
print(int(nc), int(imgsz))
PYEOF
)"

NUM_CLASSES="${NUM_CLASSES:-$DETECTED_NC}"
IMGSZ_VAL="${IMGSZ:-$DETECTED_IMGSZ}"
if ! [[ "$NUM_CLASSES" =~ ^[0-9]+$ ]] || ! [[ "$IMGSZ_VAL" =~ ^[0-9]+$ ]]; then
    echo "failed to read num_classes / imgsz from $CKPT_ABS" >&2
    echo "  detected: nc='$DETECTED_NC' imgsz='$DETECTED_IMGSZ'" >&2
    echo "  resolved: nc='$NUM_CLASSES' imgsz='$IMGSZ_VAL'" >&2
    exit 1
fi
echo "  num_classes: $NUM_CLASSES"
echo "  imgsz:       $IMGSZ_VAL"
echo "  stripped:    $STRIPPED_ABS"

# Stripped backup/restore: same disarm-before-rm pattern as ONNX_BACKUP.
STRIPPED_BACKUP=""
STRIPPED_VALIDATED=0
restore_stripped_on_fail() {
    local rc=$?
    if (( rc != 0 )); then
        if [[ -n "$STRIPPED_BACKUP" && -f "$STRIPPED_BACKUP" ]]; then
            mv "$STRIPPED_BACKUP" "$STRIPPED_ABS" 2>/dev/null || true
            echo "Cleanup: restored prior $STRIPPED_ABS from backup" >&2
        elif [[ -z "$STRIPPED_BACKUP" && -f "$STRIPPED_ABS" && "$STRIPPED_VALIDATED" -ne 1 ]]; then
            rm -f "$STRIPPED_ABS"
            echo "Cleanup: removed unvalidated freshly-written $STRIPPED_ABS" >&2
        fi
    fi
}

if [[ "$SKIP_EXPORT" == "1" && -s "$STRIPPED_ABS" ]]; then
    if [[ "$ONNX_ABS" -nt "$STRIPPED_ABS" ]]; then
        echo "WARNING: $ONNX_ABS is newer than $STRIPPED_ABS — SKIP_EXPORT may use a stale stripped ONNX." >&2
    fi
    if ! "$PYBIN" -c "import sys, onnx; onnx.checker.check_model(onnx.load(sys.argv[1]))" "$STRIPPED_ABS" 2>/dev/null; then
        echo "SKIP_EXPORT=1 but $STRIPPED_ABS failed onnx.checker (corrupt/truncated)." >&2
        exit 1
    fi
    echo "SKIP_EXPORT=1 and existing stripped ONNX validated — reusing $STRIPPED_ABS"
    STRIPPED_VALIDATED=1
else
    if [[ -f "$STRIPPED_ABS" ]]; then
        STRIPPED_BACKUP="${STRIPPED_ABS}.bak.$$"
        mv "$STRIPPED_ABS" "$STRIPPED_BACKUP"
    fi
    trap restore_stripped_on_fail EXIT
    "$PYBIN" "$STRIP_SCRIPT" "$ONNX_ABS" "$STRIPPED_ABS" --num-classes "$NUM_CLASSES"
    if ! "$PYBIN" -c "import sys, onnx; onnx.checker.check_model(onnx.load(sys.argv[1]))" "$STRIPPED_ABS" 2>/dev/null; then
        echo "stripped ONNX $STRIPPED_ABS failed onnx.checker validation." >&2
        exit 1
    fi
    STRIPPED_VALIDATED=1
    backup_to_remove="$STRIPPED_BACKUP"
    STRIPPED_BACKUP=""
    if [[ -n "$backup_to_remove" && -f "$backup_to_remove" ]]; then
        rm -f "$backup_to_remove"
    fi
    trap - EXIT
fi
echo "Stripped ONNX validated: $STRIPPED_ABS"

if [[ "$BUILD_ENGINE" -ne 1 ]]; then
    echo
    echo "skip engine build (pass --build-engine to also produce $ENGINE_ABS)"
    exit 0
fi

# === Engine build via trtexec ===
echo
echo "=== Engine build (trtexec) ==="
echo "  stripped:   $STRIPPED_ABS"
echo "  engine:     $ENGINE_ABS"
echo "  fp16:       $FP16   workspace_gb: $WORKSPACE_GB ($WORKSPACE_MB MB)"

if [[ -f "$ENGINE_ABS" ]]; then
    echo "ERROR: $ENGINE_ABS already exists — delete before re-building." >&2
    exit 1
fi

TRTEXEC_BIN="${TRTEXEC:-trtexec}"
if ! command -v "$TRTEXEC_BIN" >/dev/null 2>&1; then
    echo "trtexec not found at '$TRTEXEC_BIN'." >&2
    echo "Set TRTEXEC=<path> or run on a host with trtexec on PATH (Orin)." >&2
    exit 1
fi

SIDECAR="${ENGINE_ABS}.meta.json"
SIDECAR_TMP="${SIDECAR}.tmp"
cleanup_engine_partial() {
    local rc=$?
    if (( rc != 0 )); then
        if [[ -e "$SIDECAR_TMP" ]]; then
            rm -f "$SIDECAR_TMP"
        fi
        if [[ -f "$ENGINE_ABS" ]]; then
            echo "Cleanup: removing $ENGINE_ABS + any partial sidecar (run failed; treat as untrusted)" >&2
            rm -f "$ENGINE_ABS" "$SIDECAR"
        fi
    fi
}
trap cleanup_engine_partial EXIT

TRTEXEC_ARGS=(
    --onnx="$STRIPPED_ABS"
    --saveEngine="$ENGINE_ABS"
    --memPoolSize=workspace:"$WORKSPACE_MB"
    --minShapes=images:1x3x"$IMGSZ_VAL"x"$IMGSZ_VAL"
    --optShapes=images:1x3x"$IMGSZ_VAL"x"$IMGSZ_VAL"
    --maxShapes=images:1x3x"$IMGSZ_VAL"x"$IMGSZ_VAL"
)
[[ "$FP16" == "1" ]] && TRTEXEC_ARGS+=(--fp16)

"$TRTEXEC_BIN" "${TRTEXEC_ARGS[@]}"

if [[ ! -s "$ENGINE_ABS" ]]; then
    echo "engine missing or empty at $ENGINE_ABS after trtexec" >&2
    exit 1
fi
echo "Engine written: $ENGINE_ABS"

# Build the trtexec_cmdline replay string (for sidecar provenance) using the
# same array we just invoked, via shlex.join so paths/quotes are escaped.
TRTEXEC_CMDLINE=$(TRTEXEC_BIN_VAL="$TRTEXEC_BIN" \
    "$PYBIN" - "${TRTEXEC_ARGS[@]}" <<'PYEOF'
import os, shlex, sys
print(shlex.join([os.environ["TRTEXEC_BIN_VAL"], *sys.argv[1:]]))
PYEOF
)

EXPORTER_CMDLINE="$YOLO_CMD export model=$CKPT_ABS format=onnx simplify=True"
[[ -n "${IMGSZ:-}" ]] && EXPORTER_CMDLINE="$EXPORTER_CMDLINE imgsz=$IMGSZ"
EXPORTER_CMDLINE="$EXPORTER_CMDLINE && $PYBIN $STRIP_SCRIPT $ONNX_ABS $STRIPPED_ABS --num-classes $NUM_CLASSES"

# === Atomic sidecar (locked plan Engineering Task 6) ===
size_of() {
    stat -c%s "$1" 2>/dev/null || stat -f%z "$1"
}

ATTEMPT=0
MAX_ATTEMPTS=3
SIZE1=""
while (( ATTEMPT < MAX_ATTEMPTS )); do
    SIZE1=$(size_of "$ENGINE_ABS")
    sleep 0.5
    SIZE2=$(size_of "$ENGINE_ABS")
    if [[ -n "$SIZE1" && "$SIZE1" == "$SIZE2" && "$SIZE1" -gt 0 ]]; then
        break
    fi
    ATTEMPT=$((ATTEMPT + 1))
    echo "engine size unstable (attempt $ATTEMPT: '$SIZE1' -> '$SIZE2'), waiting..." >&2
    sleep 1
done
if (( ATTEMPT >= MAX_ATTEMPTS )); then
    echo "engine file size did not stabilize after $MAX_ATTEMPTS attempts" >&2
    exit 1
fi
if [[ -z "$SIZE1" || ! "$SIZE1" =~ ^[0-9]+$ ]]; then
    echo "engine_size_bytes is empty or non-numeric: '$SIZE1'" >&2
    exit 1
fi
ENGINE_BYTES="$SIZE1"

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}
ENGINE_SHA256=$(sha256_of "$ENGINE_ABS")
SOURCE_PT_SHA256=$(sha256_of "$CKPT_ABS")
SOURCE_ONNX_SHA256=$(sha256_of "$ONNX_ABS")
SOURCE_ONNX_STRIPPED_SHA256=$(sha256_of "$STRIPPED_ABS")
if [[ -z "$SOURCE_ONNX_SHA256" || -z "$SOURCE_ONNX_STRIPPED_SHA256" ]]; then
    echo "internal: source onnx sha256 unexpectedly empty after validation gate" >&2
    exit 1
fi

TRT_VERSION=$("$PYBIN" -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null || echo "unknown")
[[ -z "$TRT_VERSION" ]] && TRT_VERSION="unknown"
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | head -1 | awk '{print $2}' || true)
[[ -z "$CUDA_VERSION" ]] && CUDA_VERSION="unknown"
JETPACK_VERSION=$(grep -oE 'R[0-9]+ \(release\), REVISION: [0-9.]+' /etc/nv_tegra_release 2>/dev/null | head -1 || true)
[[ -z "$JETPACK_VERSION" ]] && JETPACK_VERSION="unknown"
BUILD_HOST=$(hostname)
BUILD_TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

EXPORT_PRECISION="$PRECISION_LABEL" \
EXPORT_EXPORTER="ultralytics yolo + strip_yolo26_head + trtexec" \
EXPORT_EXPORTER_CMDLINE="$EXPORTER_CMDLINE" \
EXPORT_TRTEXEC_CMDLINE="$TRTEXEC_CMDLINE" \
EXPORT_TRT_VERSION="$TRT_VERSION" \
EXPORT_CUDA_VERSION="$CUDA_VERSION" \
EXPORT_JETPACK_VERSION="$JETPACK_VERSION" \
EXPORT_BUILD_HOST="$BUILD_HOST" \
EXPORT_BUILD_TIMESTAMP="$BUILD_TIMESTAMP" \
EXPORT_SOURCE_PT="$CKPT_ABS" \
EXPORT_SOURCE_PT_SHA256="$SOURCE_PT_SHA256" \
EXPORT_SOURCE_ONNX_SHA256="$SOURCE_ONNX_SHA256" \
EXPORT_SOURCE_ONNX_STRIPPED_SHA256="$SOURCE_ONNX_STRIPPED_SHA256" \
EXPORT_NUM_CLASSES="$NUM_CLASSES" \
EXPORT_IMGSZ="$IMGSZ_VAL" \
EXPORT_ENGINE_SHA256="$ENGINE_SHA256" \
EXPORT_ENGINE_SIZE_BYTES="$ENGINE_BYTES" \
EXPORT_WORKSPACE_GB="$WORKSPACE_GB" \
EXPORT_ALLOW_LARGE_WORKSPACE="$ALLOW_LARGE_WORKSPACE" \
EXPORT_SIDECAR_PATH="$SIDECAR" \
EXPORT_SIDECAR_TMP="$SIDECAR_TMP" \
"$PYBIN" - <<'PYEOF'
import json, os
fields = {
    "precision": os.environ["EXPORT_PRECISION"],
    "exporter": os.environ["EXPORT_EXPORTER"],
    "exporter_cmdline": os.environ["EXPORT_EXPORTER_CMDLINE"],
    "trtexec_cmdline": os.environ["EXPORT_TRTEXEC_CMDLINE"],
    "trt_version": os.environ["EXPORT_TRT_VERSION"],
    "cuda_version": os.environ["EXPORT_CUDA_VERSION"],
    "jetpack_version": os.environ["EXPORT_JETPACK_VERSION"],
    "build_host": os.environ["EXPORT_BUILD_HOST"],
    "build_timestamp": os.environ["EXPORT_BUILD_TIMESTAMP"],
    "source_pt": os.environ["EXPORT_SOURCE_PT"],
    "source_pt_sha256": os.environ["EXPORT_SOURCE_PT_SHA256"],
    "source_onnx_sha256": os.environ["EXPORT_SOURCE_ONNX_SHA256"],
    "source_onnx_stripped_sha256": os.environ["EXPORT_SOURCE_ONNX_STRIPPED_SHA256"],
    "num_classes": int(os.environ["EXPORT_NUM_CLASSES"]),
    "imgsz": int(os.environ["EXPORT_IMGSZ"]),
    "engine_sha256": os.environ["EXPORT_ENGINE_SHA256"],
    "engine_size_bytes": int(os.environ["EXPORT_ENGINE_SIZE_BYTES"]),
    "workspace_gb": int(os.environ["EXPORT_WORKSPACE_GB"]),
    "allow_large_workspace": os.environ["EXPORT_ALLOW_LARGE_WORKSPACE"] == "1",
}
tmp = os.environ["EXPORT_SIDECAR_TMP"]
final = os.environ["EXPORT_SIDECAR_PATH"]
with open(tmp, "w") as f:
    json.dump(fields, f, indent=2)
    f.write("\n")
    f.flush()
    os.fsync(f.fileno())
os.replace(tmp, final)
PYEOF

if [[ -e "$SIDECAR_TMP" ]]; then
    echo "sidecar tmp file remains at $SIDECAR_TMP — atomic move failed" >&2
    exit 1
fi
if [[ ! -s "$SIDECAR" ]]; then
    echo "sidecar not written at $SIDECAR" >&2
    exit 1
fi

trap - EXIT

echo "Sidecar written: $SIDECAR"
echo "  precision:                   $PRECISION_LABEL"
echo "  num_classes:                 $NUM_CLASSES"
echo "  imgsz:                       $IMGSZ_VAL"
echo "  engine_sha256:               $ENGINE_SHA256"
echo "  source_onnx_sha256:          $SOURCE_ONNX_SHA256"
echo "  source_onnx_stripped_sha256: $SOURCE_ONNX_STRIPPED_SHA256"
