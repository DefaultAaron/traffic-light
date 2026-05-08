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
#   <ckpt>.onnx          full deploy graph (model + postprocessor.deploy())
#   <ckpt>.onnx.imgsz    sidecar with the spatial size baked in (one integer)
#   <ckpt>.engine        FP16 TRT engine (default), only when --build-engine
#                        passed and trtexec available on the host (Orin)
#   <ckpt>_fp32.engine   FP32 TRT engine when FP16=0; suffix lets it coexist
#                        with the production FP16 engine for parity tests.
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
#   FP16             1 to ask trtexec for --fp16 (default: 1). When 0, the
#                    output is named <ckpt>_fp32.engine.
#   SKIP_EXPORT      1 to skip the .pth → .onnx step when .onnx + .imgsz
#                    already exist (default: 0). Useful when rebuilding only
#                    the engine for a precision comparison — the .onnx is a
#                    deterministic intermediate of the .pth, no need to
#                    re-export.
#                    NOTE: trust boundary — the script does NOT verify the
#                    existing .onnx matches the current .pth. Caller is
#                    responsible for keeping them in sync (e.g. delete the
#                    .onnx after retraining, or run without SKIP_EXPORT once
#                    to refresh). A best-effort mtime warning fires when
#                    .pth is newer than .onnx but the build still proceeds.
#   WORKSPACE_MB     trtexec --memPoolSize=workspace:N (default: 4096).
#   TRTEXEC          trtexec binary path (default: trtexec on PATH,
#                    fallback /usr/src/tensorrt/bin/trtexec on JetPack).

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 2 ]]; then
    cat <<'EOF' >&2
usage: scripts/export_deim.sh <n|s|m|l> <ckpt.pth> [--build-engine]

env: PYTHON, FP16, SKIP_EXPORT, WORKSPACE_MB, TRTEXEC
EOF
    exit 1
fi

SIZE="${1//$'\r'/}"
CKPT="$2"
BUILD_ENGINE=0
[[ "${3:-}" == "--build-engine" ]] && BUILD_ENGINE=1

case "$SIZE" in
    n|s|m|l) ;;
    *) echo "size must be one of: n s m l" >&2; exit 1 ;;
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

# Engine output name: FP16 (default, production) lands at <ckpt>.engine; FP32
# (parity / accuracy comparison only) gets a _fp32 suffix so it coexists with
# the production engine.
FP16="${FP16:-1}"
case "$FP16" in
    0|1) ;;
    *) echo "FP16 must be 0 or 1, got '$FP16'" >&2; exit 1 ;;
esac
if [[ "$FP16" == "1" ]]; then
    ENGINE_ABS="${CKPT_ABS%.pth}.engine"
else
    ENGINE_ABS="${CKPT_ABS%.pth}_fp32.engine"
fi

SKIP_EXPORT="${SKIP_EXPORT:-0}"

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
# `-s` (exists AND non-empty) instead of `-f`: a 0-byte ONNX or empty
# sidecar from an interrupted prior run / partial scp would otherwise pass
# `-f` and silently poison the engine build. Treat empty == missing.
# Plus a full onnx.checker validation — `-s` doesn't catch truncated /
# corrupt protobuf payloads (ONNX has no fixed magic byte to grep for),
# and a malformed reused ONNX would only surface as a confusing trtexec
# error far from its cause, or worse, hand a broken graph to ORT consumers.
if [[ "$SKIP_EXPORT" == "1" && -s "$ONNX_ABS" && -s "$SIDECAR" ]]; then
    if [[ "$CKPT_ABS" -nt "$ONNX_ABS" ]]; then
        echo "WARNING: $CKPT_ABS is newer than $ONNX_ABS — SKIP_EXPORT may use a stale ONNX." >&2
        echo "         Re-run without SKIP_EXPORT=1 to refresh." >&2
    fi
    if ! "$PYBIN" -c "import sys, onnx; onnx.checker.check_model(onnx.load(sys.argv[1]))" "$ONNX_ABS" 2>/dev/null; then
        echo "SKIP_EXPORT=1 but $ONNX_ABS failed onnx.checker validation (corrupt or truncated)." >&2
        echo "Delete the .onnx + .imgsz and re-run without SKIP_EXPORT=1 to regenerate." >&2
        exit 1
    fi
    # C3 iter-4 fix: also verify the .imgsz sidecar matches the ONNX's
    # actual `images` input spatial dimension. A stale .imgsz alongside a
    # valid .onnx would build a TRT engine with shape specialization that
    # doesn't match what Python ORT consumes — silent parity failure.
    SIDECAR_IMGSZ_RAW=$(tr -d '[:space:]' < "$SIDECAR")
    if ! "$PYBIN" - "$ONNX_ABS" "$SIDECAR_IMGSZ_RAW" <<'PYEOF' 2>/dev/null
import sys, onnx
onnx_path, imgsz_str = sys.argv[1], sys.argv[2]
imgsz = int(imgsz_str)
m = onnx.load(onnx_path)
img_input = next((i for i in m.graph.input if i.name == "images"), None)
if img_input is None:
    print(f"ONNX has no 'images' input", file=sys.stderr)
    sys.exit(1)
dims = img_input.type.tensor_type.shape.dim
# Expected layout NCHW with N dynamic; H == W == imgsz
if len(dims) != 4:
    print(f"images input has {len(dims)} dims, expected 4 (NCHW)", file=sys.stderr)
    sys.exit(1)
h, w = dims[2].dim_value, dims[3].dim_value
if h != imgsz or w != imgsz:
    print(f".imgsz says {imgsz} but ONNX images input is {h}x{w}", file=sys.stderr)
    sys.exit(1)
PYEOF
    then
        echo "SKIP_EXPORT=1 but $SIDECAR ($SIDECAR_IMGSZ_RAW) does not match $ONNX_ABS shape." >&2
        echo "Delete the .onnx + .imgsz and re-run without SKIP_EXPORT=1 to regenerate consistent pair." >&2
        exit 1
    fi
    echo "SKIP_EXPORT=1 and existing ONNX validated (shape $SIDECAR_IMGSZ_RAW matches) — reusing $ONNX_ABS"
else
    if [[ "$SKIP_EXPORT" == "1" ]]; then
        echo "SKIP_EXPORT=1 requested but $ONNX_ABS or $SIDECAR is missing or empty — running full export"
    fi
    # Stop-gate fix: `_export_deim_onnx.py` overwrites both $ONNX_ABS and
    # $SIDECAR unconditionally. If post-simplify validation or the
    # .imgsz↔ONNX cross-check fails, the prior known-good pair would be
    # lost. Backup both, install a trap that restores on failure, drop
    # the backups + clear the trap on success. Mirrors YOLO's
    # restore_onnx_on_fail pattern but covers BOTH artifacts (DEIM's
    # imgsz sidecar is paired with the .onnx).
    DEIM_ONNX_BACKUP=""
    DEIM_SIDECAR_BACKUP=""
    DEIM_ONNX_VALIDATED=0
    if [[ -f "$ONNX_ABS" ]]; then
        DEIM_ONNX_BACKUP="${ONNX_ABS}.bak.$$"
        mv "$ONNX_ABS" "$DEIM_ONNX_BACKUP"
    fi
    if [[ -f "$SIDECAR" ]]; then
        DEIM_SIDECAR_BACKUP="${SIDECAR}.bak.$$"
        mv "$SIDECAR" "$DEIM_SIDECAR_BACKUP"
    fi
    restore_deim_onnx_pair_on_fail() {
        local rc=$?
        if (( rc != 0 )); then
            # Restore prior pair (overwrites any partial fresh write).
            if [[ -n "$DEIM_ONNX_BACKUP" && -f "$DEIM_ONNX_BACKUP" ]]; then
                mv "$DEIM_ONNX_BACKUP" "$ONNX_ABS" 2>/dev/null || true
                echo "Cleanup: restored prior $ONNX_ABS from backup" >&2
            elif [[ -z "$DEIM_ONNX_BACKUP" && -f "$ONNX_ABS" && "$DEIM_ONNX_VALIDATED" -ne 1 ]]; then
                rm -f "$ONNX_ABS"
                echo "Cleanup: removed unvalidated freshly-written $ONNX_ABS" >&2
            fi
            if [[ -n "$DEIM_SIDECAR_BACKUP" && -f "$DEIM_SIDECAR_BACKUP" ]]; then
                mv "$DEIM_SIDECAR_BACKUP" "$SIDECAR" 2>/dev/null || true
                echo "Cleanup: restored prior $SIDECAR from backup" >&2
            elif [[ -z "$DEIM_SIDECAR_BACKUP" && -f "$SIDECAR" && "$DEIM_ONNX_VALIDATED" -ne 1 ]]; then
                rm -f "$SIDECAR"
                echo "Cleanup: removed unvalidated freshly-written $SIDECAR" >&2
            fi
        fi
    }
    trap restore_deim_onnx_pair_on_fail EXIT

    "$PYBIN" "$PROJECT_ROOT/scripts/_export_deim_onnx.py" \
        --check --simplify \
        -c "$CFG" \
        -r "$CKPT_ABS"

    # `_export_deim_onnx.py` runs onnx.checker BEFORE onnxsim.simplify
    # and saves the simplified graph afterward. The post-simplify final
    # artifact is therefore unchecked at the script level. Re-validate
    # here so a malformed simplified graph cannot silently feed trtexec.
    if [[ ! -s "$ONNX_ABS" ]]; then
        echo "$ONNX_ABS missing or empty after fresh export" >&2
        exit 1
    fi
    if ! "$PYBIN" -c "import sys, onnx; onnx.checker.check_model(onnx.load(sys.argv[1]))" "$ONNX_ABS" 2>/dev/null; then
        echo "freshly-exported (and simplified) $ONNX_ABS failed onnx.checker validation." >&2
        echo "onnxsim.simplify may have produced a malformed graph; re-run without --simplify or report upstream." >&2
        exit 1
    fi
    # Mirror the SKIP path: cross-check .imgsz sidecar matches the
    # post-simplify ONNX images-input spatial dim.
    if [[ ! -s "$SIDECAR" ]]; then
        echo "$SIDECAR missing or empty after fresh export" >&2
        exit 1
    fi
    SIDECAR_IMGSZ_RAW=$(tr -d '[:space:]' < "$SIDECAR")
    if ! "$PYBIN" - "$ONNX_ABS" "$SIDECAR_IMGSZ_RAW" <<'PYEOF' 2>/dev/null
import sys, onnx
onnx_path, imgsz_str = sys.argv[1], sys.argv[2]
imgsz = int(imgsz_str)
m = onnx.load(onnx_path)
img_input = next((i for i in m.graph.input if i.name == "images"), None)
if img_input is None:
    print(f"ONNX has no 'images' input", file=sys.stderr)
    sys.exit(1)
dims = img_input.type.tensor_type.shape.dim
if len(dims) != 4:
    print(f"images input has {len(dims)} dims, expected 4 (NCHW)", file=sys.stderr)
    sys.exit(1)
h, w = dims[2].dim_value, dims[3].dim_value
if h != imgsz or w != imgsz:
    print(f".imgsz says {imgsz} but ONNX images input is {h}x{w}", file=sys.stderr)
    sys.exit(1)
PYEOF
    then
        echo "freshly-exported $ONNX_ABS shape does not match $SIDECAR ($SIDECAR_IMGSZ_RAW)." >&2
        echo "Likely an onnxsim simplification altered the input dim or _export_deim_onnx.py wrote a stale .imgsz." >&2
        exit 1
    fi

    # All validations passed: mark validated, drop backups, clear trap.
    # Mirror YOLO's disarm-before-rm pattern so a failed rm under set -e
    # cannot trigger restore over the validated pair.
    DEIM_ONNX_VALIDATED=1
    onnx_backup_to_remove="$DEIM_ONNX_BACKUP"
    sidecar_backup_to_remove="$DEIM_SIDECAR_BACKUP"
    DEIM_ONNX_BACKUP=""
    DEIM_SIDECAR_BACKUP=""
    if [[ -n "$onnx_backup_to_remove" && -f "$onnx_backup_to_remove" ]]; then
        rm -f "$onnx_backup_to_remove"
    fi
    if [[ -n "$sidecar_backup_to_remove" && -f "$sidecar_backup_to_remove" ]]; then
        rm -f "$sidecar_backup_to_remove"
    fi
    trap - EXIT
fi

if [[ ! -f "$ONNX_ABS" ]]; then
    echo "$ONNX_ABS missing after export step (SKIP_EXPORT=$SKIP_EXPORT) — check stderr above" >&2
    exit 1
fi
if [[ ! -f "$SIDECAR" ]]; then
    echo "$SIDECAR missing after export step (SKIP_EXPORT=$SKIP_EXPORT) — check stderr above" >&2
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

# Honor an explicit TRTEXEC override; only fall back to the JetPack default
# when the user did NOT set TRTEXEC. A typo'd or stale TRTEXEC must NOT
# silently substitute the wrong binary into a production engine build —
# engine provenance is correctness-critical for Python↔C++ TRT parity.
if [[ -n "${TRTEXEC:-}" ]]; then
    TRTEXEC_BIN="$TRTEXEC"
    if ! command -v "$TRTEXEC_BIN" >/dev/null 2>&1; then
        echo "TRTEXEC=$TRTEXEC was explicitly set but is not executable or not found." >&2
        echo "Refusing to fall back — fix TRTEXEC or unset it to use auto-detection." >&2
        exit 1
    fi
else
    TRTEXEC_BIN=trtexec
    if ! command -v "$TRTEXEC_BIN" >/dev/null 2>&1; then
        if [[ -x /usr/src/tensorrt/bin/trtexec ]]; then
            TRTEXEC_BIN=/usr/src/tensorrt/bin/trtexec
            echo "trtexec not on PATH; using JetPack default $TRTEXEC_BIN"
        else
            echo "trtexec not found on PATH and not at /usr/src/tensorrt/bin/trtexec." >&2
            echo "Set TRTEXEC=<path> or run on the deployment host (Orin)." >&2
            exit 1
        fi
    fi
fi

WORKSPACE_MB="${WORKSPACE_MB:-4096}"
# Defense against the WORKSPACE_GB→WORKSPACE_MB unit-of-confusion footgun
# (YOLO uses GB; DEIM uses MB to match trtexec's native arg). A caller
# copying YOLO's `WORKSPACE_GB=4` mental model into DEIM gets a 4 MB
# workspace, which silently changes builder tactics or fails late.
if ! [[ "$WORKSPACE_MB" =~ ^[0-9]+$ ]] || [[ "$WORKSPACE_MB" -lt 1 ]]; then
    echo "WORKSPACE_MB must be a positive integer (MB), got '$WORKSPACE_MB'" >&2
    exit 1
fi
ALLOW_SMALL_WORKSPACE="${ALLOW_SMALL_WORKSPACE:-0}"
if [[ "$WORKSPACE_MB" -lt 256 && "$ALLOW_SMALL_WORKSPACE" != "1" ]]; then
    echo "WORKSPACE_MB=$WORKSPACE_MB is below 256 MB (likely a GB→MB unit mistake)." >&2
    echo "Note: YOLO uses WORKSPACE_GB; DEIM uses WORKSPACE_MB. Did you copy from YOLO?" >&2
    echo "Set ALLOW_SMALL_WORKSPACE=1 to bypass." >&2
    exit 1
fi
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

# Atomic sidecar contract (locked plan Engineering Task 6 — parity with
# scripts/export_yolo.sh): treat engine-without-sidecar (or with corrupt
# sidecar) as "untrusted". On any failure or interrupt between trtexec
# starting and sidecar landing, clean up artifacts.
SIDECAR="${ENGINE_ABS}.meta.json"
SIDECAR_TMP="${SIDECAR}.tmp"
cleanup_partial_deim() {
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
# Refuse to clobber an existing engine of the same precision.
if [[ -f "$ENGINE_ABS" ]]; then
    echo "ERROR: $ENGINE_ABS already exists." >&2
    echo "       Delete it explicitly before re-building." >&2
    exit 1
fi
trap cleanup_partial_deim EXIT

# C3 iter-4 fix: build trtexec args as a single bash array, used for both
# the actual invocation AND the recorded cmdline. No string drift between
# what runs and what the sidecar claims — if a path contains spaces or
# unusual chars, the array invocation is correct and the recorded form
# uses Python shlex.join for a faithful, replayable representation.
TRTEXEC_ARGS=(
    --onnx="$ONNX_ABS"
    --saveEngine="$ENGINE_ABS"
    --memPoolSize=workspace:"$WORKSPACE_MB"
    --minShapes=images:1x3x"$IMGSZ"x"$IMGSZ",orig_target_sizes:1x2
    --optShapes=images:1x3x"$IMGSZ"x"$IMGSZ",orig_target_sizes:1x2
    --maxShapes=images:1x3x"$IMGSZ"x"$IMGSZ",orig_target_sizes:1x2
)
[[ "$FP16" == "1" ]] && TRTEXEC_ARGS+=(--fp16)

"$TRTEXEC_BIN" "${TRTEXEC_ARGS[@]}"

if [[ ! -s "$ENGINE_ABS" ]]; then
    echo "engine missing or empty at $ENGINE_ABS after trtexec" >&2
    exit 1
fi

# Record the full invocation faithfully — shlex.join emits a string the
# user can paste directly into a shell to reproduce the build.
TRTEXEC_CMDLINE=$(TRTEXEC_BIN_VAL="$TRTEXEC_BIN" \
    "$PYBIN" - "${TRTEXEC_ARGS[@]}" <<'PYEOF'
import os, shlex, sys
print(shlex.join([os.environ["TRTEXEC_BIN_VAL"], *sys.argv[1:]]))
PYEOF
)

echo "Engine written: $ENGINE_ABS"

# === Atomic sidecar generation (size-stability + JSON via Python) ===
size_of() { stat -c%s "$1" 2>/dev/null || stat -f%z "$1"; }
sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
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

ENGINE_SHA256=$(sha256_of "$ENGINE_ABS")
SOURCE_PTH_SHA256=$(sha256_of "$CKPT_ABS")
SOURCE_ONNX_SHA256=$(sha256_of "$ONNX_ABS")

PRECISION_LABEL="fp32"
[[ "$FP16" == "1" ]] && PRECISION_LABEL="fp16"

TRT_VERSION=$("$PYBIN" -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null || echo "unknown")
[[ -z "$TRT_VERSION" ]] && TRT_VERSION="unknown"
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | head -1 | awk '{print $2}' || true)
[[ -z "$CUDA_VERSION" ]] && CUDA_VERSION="unknown"
JETPACK_VERSION=$(grep -oE 'R[0-9]+ \(release\), REVISION: [0-9.]+' /etc/nv_tegra_release 2>/dev/null | head -1 || true)
[[ -z "$JETPACK_VERSION" ]] && JETPACK_VERSION="unknown"
BUILD_HOST=$(hostname)
BUILD_TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# JSON sidecar via Python json.dumps fed through env vars (no escaping
# bugs in heredoc interpolation). Atomic: write to ${SIDECAR}.tmp then
# os.replace.
EXPORT_PRECISION="$PRECISION_LABEL" \
EXPORT_EXPORTER="trtexec" \
EXPORT_TRTEXEC_CMDLINE="$TRTEXEC_CMDLINE" \
EXPORT_TRT_VERSION="$TRT_VERSION" \
EXPORT_CUDA_VERSION="$CUDA_VERSION" \
EXPORT_JETPACK_VERSION="$JETPACK_VERSION" \
EXPORT_BUILD_HOST="$BUILD_HOST" \
EXPORT_BUILD_TIMESTAMP="$BUILD_TIMESTAMP" \
EXPORT_SOURCE_PTH="$CKPT_ABS" \
EXPORT_SOURCE_PTH_SHA256="$SOURCE_PTH_SHA256" \
EXPORT_SOURCE_ONNX_SHA256="$SOURCE_ONNX_SHA256" \
EXPORT_ENGINE_SHA256="$ENGINE_SHA256" \
EXPORT_ENGINE_SIZE_BYTES="$ENGINE_BYTES" \
EXPORT_WORKSPACE_MB="$WORKSPACE_MB" \
EXPORT_IMGSZ="$IMGSZ" \
EXPORT_SIDECAR_PATH="$SIDECAR" \
EXPORT_SIDECAR_TMP="$SIDECAR_TMP" \
"$PYBIN" - <<'PYEOF'
import json, os
fields = {
    "precision": os.environ["EXPORT_PRECISION"],
    "exporter": os.environ["EXPORT_EXPORTER"],
    # exporter_cmdline kept null for schema parity with export_yolo.sh
    # sidecar (which records the yolo wrapper invocation, not raw trtexec).
    "exporter_cmdline": None,
    "trtexec_cmdline": os.environ["EXPORT_TRTEXEC_CMDLINE"],
    "trt_version": os.environ["EXPORT_TRT_VERSION"],
    "cuda_version": os.environ["EXPORT_CUDA_VERSION"],
    "jetpack_version": os.environ["EXPORT_JETPACK_VERSION"],
    "build_host": os.environ["EXPORT_BUILD_HOST"],
    "build_timestamp": os.environ["EXPORT_BUILD_TIMESTAMP"],
    "source_pth": os.environ["EXPORT_SOURCE_PTH"],
    "source_pth_sha256": os.environ["EXPORT_SOURCE_PTH_SHA256"],
    "source_onnx_sha256": os.environ["EXPORT_SOURCE_ONNX_SHA256"],
    "engine_sha256": os.environ["EXPORT_ENGINE_SHA256"],
    "engine_size_bytes": int(os.environ["EXPORT_ENGINE_SIZE_BYTES"]),
    "workspace_mb": int(os.environ["EXPORT_WORKSPACE_MB"]),
    "imgsz": int(os.environ["EXPORT_IMGSZ"]),
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

# Sidecar landed; clear the trap.
trap - EXIT

echo "Sidecar written: $SIDECAR"
echo "  precision:     $PRECISION_LABEL"
echo "  engine_sha256: $ENGINE_SHA256"
