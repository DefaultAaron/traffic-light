#!/usr/bin/env bash
# Export an Ultralytics YOLO checkpoint to ONNX, then optionally to a
# TensorRT engine. Mirrors the scripts/export_deim.sh contract for size,
# precision, _fp32 suffix, and SKIP_EXPORT semantics.
#
# Pipeline (with --build-engine):
#   <ckpt.pt> -- yolo export format=engine --> <ckpt.onnx> + <ckpt.engine>
#                                              (Orin only)
#
# Why no separate ONNX-step like DEIM:
#   Ultralytics' `yolo export format=engine` produces BOTH the .onnx and the
#   .engine in one CLI call. SKIP_EXPORT therefore controls a different
#   contract here than for DEIM: when --build-engine is NOT passed,
#   SKIP_EXPORT=1 is honored against an existing .onnx; when --build-engine
#   IS passed, SKIP_EXPORT=1 is logged-and-ignored (yolo's unified pipeline
#   re-runs both stages — short-circuiting it is more fragile than paying
#   the small re-export cost).
#
# Target detector: YOLO26 (the Ultralytics CLI accepts other YOLO families
# transparently — YOLOv13 etc. would Just Work — but this script is only
# validated for YOLO26 in R2; using it for other families is at caller's
# risk).
#
# Usage:
#   scripts/export_yolo.sh <size> <ckpt.pt> [--build-engine]
#   scripts/export_yolo.sh s runs/yolo26_s-r1/weights/best.pt
#   scripts/export_yolo.sh s runs/yolo26_s-r1/weights/best.pt --build-engine
#
# Outputs (next to the input checkpoint):
#   <ckpt>.onnx               deploy graph
#   <ckpt>.engine             FP16 TRT engine (default)
#   <ckpt>_fp32.engine        FP32 TRT engine (FP16=0); coexists with FP16
#   <engine>.meta.json        atomic precision metadata sidecar (locked plan
#                             Engineering Task 6)
#
# Env overrides:
#   YOLO_BIN         yolo CLI path (default: `yolo` on PATH)
#   PYTHON           Python interpreter (default: auto-detect python ->
#                    python3). Used for onnx.checker and the JSON sidecar
#                    writer.
#   FP16             1 to ask for FP16 (default: 1). When 0, output named
#                    <ckpt>_fp32.engine and sidecar precision="fp32".
#   SKIP_EXPORT      1 to reuse existing .onnx (only meaningful WITHOUT
#                    --build-engine; default: 0). Logged-and-ignored when
#                    --build-engine is set.
#   WORKSPACE_GB     trtexec workspace size in GB passed to yolo export
#                    (default: 4). DEIM uses WORKSPACE_MB; YOLO uses GB to
#                    match the underlying Ultralytics arg's native unit.
#   IMGSZ            override input size (default: read from model). Most
#                    callers should leave unset — Ultralytics uses the
#                    training imgsz baked into the .pt.

set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 2 ]]; then
    cat <<'EOF' >&2
usage: scripts/export_yolo.sh <n|s|m|l|x> <ckpt.pt> [--build-engine]

env: YOLO_BIN, PYTHON, FP16, SKIP_EXPORT, WORKSPACE_GB, IMGSZ
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

# Detector-family soft gate: this script is validated for YOLO26 only.
# C3 iter-2 fix: require a path SEGMENT (not arbitrary substring) to start
# with "yolo26". The substring check `*yolo26*` would let
# `runs/old_yolo26_failed/yolo13_v2/best.pt` pass even though the actual
# variant is yolo13. The regex `/yolo26[^/]*` anchors to a path-separator
# boundary so it requires `/yolo26...` (slash directly before yolo26 +
# arbitrary chars within the same segment).
ALLOW_NON_YOLO26="${ALLOW_NON_YOLO26:-0}"
if [[ "$ALLOW_NON_YOLO26" != "1" ]]; then
    # Prepend / so a path with no leading slash still matches a segment
    # boundary at position 0 of the basename.
    if ! [[ "/$CKPT" =~ /yolo26[^/]* ]]; then
        echo "ERROR: checkpoint path '$CKPT' has no path segment starting with 'yolo26'." >&2
        echo "       This script is validated for YOLO26 only in R2." >&2
        echo "       Set ALLOW_NON_YOLO26=1 to bypass (at caller's risk)." >&2
        exit 1
    fi
fi

CKPT_ABS="$(cd "$(dirname "$CKPT")" && pwd)/$(basename "$CKPT")"
ONNX_ABS="${CKPT_ABS%.pt}.onnx"

# Engine output name: FP16 (default, production) lands at <ckpt>.engine; FP32
# (parity / accuracy comparison) gets a _fp32 suffix so it coexists with the
# production FP16 engine.
FP16="${FP16:-1}"
case "$FP16" in
    0|1) ;;
    *) echo "FP16 must be 0 or 1, got '$FP16'" >&2; exit 1 ;;
esac
if [[ "$FP16" == "1" ]]; then
    HALF_FLAG="True"
    ENGINE_ABS="${CKPT_ABS%.pt}.engine"
    PRECISION_LABEL="fp16"
else
    HALF_FLAG="False"
    ENGINE_ABS="${CKPT_ABS%.pt}_fp32.engine"
    PRECISION_LABEL="fp32"
fi

# Ultralytics always writes the engine to <ckpt>.engine regardless of `half`.
# When FP16=0 we rename to the _fp32 suffix after the build. CRITICAL: this
# means an existing FP16 engine at <ckpt>.engine would be silently clobbered
# by the FP32 build — the H1 bug from the B2 review. The clobber guard below
# checks BOTH paths to prevent this.
ENGINE_RAW="${CKPT_ABS%.pt}.engine"

YOLO_CMD="${YOLO_BIN:-yolo}"
if ! command -v "$YOLO_CMD" >/dev/null 2>&1; then
    echo "yolo CLI not found at '$YOLO_CMD'." >&2
    echo "Install ultralytics or set YOLO_BIN=<path-to-yolo>" >&2
    exit 1
fi

# Auto-detect Python: respect PYTHON override, else prefer `python` for
# parity with export_deim.sh, else fall back to `python3` (default on
# JetPack 5.1 stock user env).
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
# C3 iter-2 fix: log selected interpreter and preflight `import onnx`. The
# user may have multiple Python envs (system python3 with onnx vs venv
# python without). A successful expensive engine export followed by a
# validation failure due to missing onnx package is a bad failure mode —
# fail fast instead.
echo "  python:     $PYBIN ($("$PYBIN" --version 2>&1))"
if ! "$PYBIN" -c "import onnx" 2>/dev/null; then
    echo "ERROR: $PYBIN does not have the 'onnx' package installed." >&2
    echo "       This script needs onnx for ONNX validation." >&2
    echo "       Install via 'pip install onnx', or set PYTHON=<other-python>" >&2
    exit 1
fi

SKIP_EXPORT="${SKIP_EXPORT:-0}"
WORKSPACE_GB="${WORKSPACE_GB:-4}"
# Defense against the WORKSPACE_MB→WORKSPACE_GB unit-of-confusion footgun
# (DEIM uses MB, YOLO uses GB to match Ultralytics' native arg). Reject
# non-numeric and obviously-wrong-unit values; cap at 32GB unless caller
# explicitly opts in via ALLOW_LARGE_WORKSPACE=1.
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

if [[ "$BUILD_ENGINE" -eq 1 && "$SKIP_EXPORT" == "1" ]]; then
    echo "NOTE: SKIP_EXPORT=1 is ignored when --build-engine is set" >&2
    echo "      (yolo's unified pipeline re-runs both .onnx and .engine stages)" >&2
fi

IMGSZ_FLAG=()
[[ -n "${IMGSZ:-}" ]] && IMGSZ_FLAG=("imgsz=$IMGSZ")

echo "=== ONNX export ==="
echo "  size:       $SIZE"
echo "  checkpoint: $CKPT_ABS"
echo "  onnx out:   $ONNX_ABS"

# C3 iter-2 fix: ONNX backup/restore. yolo export overwrites $ONNX_ABS
# unconditionally, so a prior known-good ONNX would be lost if validation
# fails after the new write. Move the existing .onnx to a backup, restore
# on failure via the trap, delete the backup on success.
# C3 iter-6 fix: ONNX_VALIDATED is script-global (used by both
# restore_onnx_on_fail in the format=onnx branch AND cleanup_partial in
# the engine branch). Set to 1 immediately after each onnx.checker pass.
# Trap branches that would delete a fresh ONNX must skip when
# ONNX_VALIDATED=1 — losing a validated graph destroys the Python ORT
# parity artifact.
ONNX_BACKUP=""
ONNX_VALIDATED=0
restore_onnx_on_fail() {
    local rc=$?
    if (( rc != 0 )); then
        if [[ -n "$ONNX_BACKUP" && -f "$ONNX_BACKUP" ]]; then
            # Restore prior known-good ONNX, overwriting any partial write
            # yolo may have left at $ONNX_ABS.
            mv "$ONNX_BACKUP" "$ONNX_ABS" 2>/dev/null || true
            echo "Cleanup: restored prior $ONNX_ABS from backup" >&2
        elif [[ -z "$ONNX_BACKUP" && -f "$ONNX_ABS" && "$ONNX_VALIDATED" -ne 1 ]]; then
            # No backup, fresh ONNX is unvalidated → delete (lying-to-
            # consumers prevention). Mirrors the elif branch in
            # cleanup_partial. If ONNX_VALIDATED=1 (rm of backup failed
            # post-validation), the validated fresh ONNX is preserved.
            rm -f "$ONNX_ABS"
            echo "Cleanup: removed unvalidated freshly-written $ONNX_ABS" >&2
        fi
    fi
}

# When NOT building an engine, honor SKIP_EXPORT against the existing .onnx.
if [[ "$BUILD_ENGINE" -ne 1 ]]; then
    if [[ "$SKIP_EXPORT" == "1" && -s "$ONNX_ABS" ]]; then
        if [[ "$CKPT_ABS" -nt "$ONNX_ABS" ]]; then
            echo "WARNING: $CKPT_ABS is newer than $ONNX_ABS — SKIP_EXPORT may use a stale ONNX." >&2
            echo "         Re-run without SKIP_EXPORT=1 to refresh." >&2
        fi
        if ! "$PYBIN" -c "import sys, onnx; onnx.checker.check_model(onnx.load(sys.argv[1]))" "$ONNX_ABS" 2>/dev/null; then
            echo "SKIP_EXPORT=1 but $ONNX_ABS failed onnx.checker validation (corrupt or truncated)." >&2
            echo "Delete the .onnx and re-run without SKIP_EXPORT=1 to regenerate." >&2
            exit 1
        fi
        echo "SKIP_EXPORT=1 and existing ONNX validated — reusing $ONNX_ABS"
    else
        if [[ "$SKIP_EXPORT" == "1" ]]; then
            echo "SKIP_EXPORT=1 requested but $ONNX_ABS is missing or empty — running full export"
        fi
        # Backup any existing ONNX so a failed export can roll back
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
        # Validate freshly-exported ONNX with onnx.checker. A truncated
        # or malformed graph passes -s but breaks Python ORT parity.
        if ! "$PYBIN" -c "import sys, onnx; onnx.checker.check_model(onnx.load(sys.argv[1]))" "$ONNX_ABS" 2>/dev/null; then
            echo "freshly-exported $ONNX_ABS failed onnx.checker validation." >&2
            echo "yolo export may have produced a truncated or malformed graph; check stderr above." >&2
            exit 1
        fi
        # C3 iter-6 fix: mark ONNX validated BEFORE attempting backup rm.
        # If the rm fails under `set -e` and the trap fires, the elif
        # branch in restore_onnx_on_fail must see ONNX_VALIDATED=1 and
        # preserve the freshly-validated ONNX (the Python ORT parity
        # artifact) instead of deleting it. Mirrors the engine branch.
        ONNX_VALIDATED=1
        # C3 iter-5 fix: mirror the engine-path disarm-before-rm pattern
        # in this branch too. If `rm` fails under `set -e`, the trap must
        # not see a populated ONNX_BACKUP — otherwise it would restore
        # the stale pre-export graph over the freshly-validated one.
        backup_to_remove="$ONNX_BACKUP"
        ONNX_BACKUP=""
        if [[ -n "$backup_to_remove" && -f "$backup_to_remove" ]]; then
            rm -f "$backup_to_remove"
        fi
        trap - EXIT
    fi
    if [[ ! -s "$ONNX_ABS" ]]; then
        echo "$ONNX_ABS missing or empty after export step — check stderr above" >&2
        exit 1
    fi
    echo "ONNX written: $ONNX_ABS"
    echo
    echo "skip engine build (pass --build-engine to also produce $ENGINE_ABS)"
    exit 0
fi

# --- engine build path ---
echo
echo "=== Engine build ==="
echo "  onnx:       $ONNX_ABS  (re-emitted by yolo export)"
echo "  engine:     $ENGINE_ABS"
echo "  fp16:       $FP16   workspace_gb: $WORKSPACE_GB"

# Refuse to clobber an existing engine of the SAME precision unless caller
# removes it first. Avoids accidentally invalidating a known-good engine.
if [[ -f "$ENGINE_ABS" ]]; then
    echo "ERROR: $ENGINE_ABS already exists." >&2
    echo "       Delete it explicitly before re-building." >&2
    exit 1
fi

# B2 review H1 fix: yolo always writes to ENGINE_RAW (= <ckpt>.engine)
# regardless of `half=`. When building FP32, ENGINE_RAW != ENGINE_ABS and
# yolo's output would silently overwrite a pre-existing FP16 engine at
# ENGINE_RAW before our `mv` ever fires. Guard it.
if [[ "$ENGINE_RAW" != "$ENGINE_ABS" && -f "$ENGINE_RAW" ]]; then
    echo "ERROR: $ENGINE_RAW exists (likely a prior FP16 build of the same .pt)." >&2
    echo "       Yolo writes to that path unconditionally and would overwrite it." >&2
    echo "       Move/rename the existing engine first, or build into a clean dir." >&2
    exit 1
fi

# B2/C3 review fix: trap to remove engine + any partial sidecar artifacts on
# interrupt. Locked plan treats engine-without-sidecar (or with corrupt
# sidecar) as "untrusted"; an interrupted run must NOT leave such artifacts.
# Sidecar is staged at ${SIDECAR}.tmp and atomically moved into place ONLY
# when fully written. ONNX is backed up before yolo overwrites it so a
# failed validation can roll back to the prior known-good graph.
SIDECAR="${ENGINE_ABS}.meta.json"
SIDECAR_TMP="${SIDECAR}.tmp"
ONNX_BACKUP=""
# ONNX_VALIDATED is declared script-global earlier (used by both
# restore_onnx_on_fail and cleanup_partial). Re-zero here defensively in
# case any prior code path set it (the format=onnx branch can't reach
# here — it exits before — but explicit beats implicit).
ONNX_VALIDATED=0
if [[ -f "$ONNX_ABS" ]]; then
    ONNX_BACKUP="${ONNX_ABS}.bak.$$"
    mv "$ONNX_ABS" "$ONNX_BACKUP"
fi
cleanup_partial() {
    local rc=$?
    if (( rc != 0 )); then
        if [[ -e "$SIDECAR_TMP" ]]; then
            rm -f "$SIDECAR_TMP"
        fi
        # FP32 build sequence: yolo writes engine to ENGINE_RAW (= <ckpt>.engine),
        # then we mv to ENGINE_ABS (= <ckpt>_fp32.engine). If the script is
        # interrupted between yolo's write and our mv, an untrusted engine sits
        # at ENGINE_RAW masquerading as an FP16 engine. Delete it.
        # For FP16 builds ENGINE_RAW == ENGINE_ABS so this is a no-op (the
        # ENGINE_ABS branch below covers that path).
        if [[ "$ENGINE_RAW" != "$ENGINE_ABS" && -f "$ENGINE_RAW" ]]; then
            echo "Cleanup: removing intermediate $ENGINE_RAW (interrupted before FP32 rename; would masquerade as FP16)" >&2
            rm -f "$ENGINE_RAW"
        fi
        if [[ -f "$ENGINE_ABS" ]]; then
            echo "Cleanup: removing $ENGINE_ABS + any partial sidecar (run failed; treat as untrusted)" >&2
            rm -f "$ENGINE_ABS" "$SIDECAR"
        fi
        # ONNX rollback. Three cases:
        # (a) backup exists → restore (overwrites partial fresh write).
        # (b) no backup, fresh ONNX exists, ONNX_VALIDATED=0 → delete the
        #     unvalidated fresh ONNX (lying-to-consumers prevention).
        # (c) no backup, fresh ONNX exists, ONNX_VALIDATED=1 → KEEP the
        #     ONNX. It passed onnx.checker; it's the Python ORT parity
        #     artifact and downstream eval needs it. The engine + sidecar
        #     can be rebuilt later without losing the validated graph.
        if [[ -n "$ONNX_BACKUP" && -f "$ONNX_BACKUP" ]]; then
            mv "$ONNX_BACKUP" "$ONNX_ABS" 2>/dev/null || true
            echo "Cleanup: restored prior $ONNX_ABS from backup" >&2
        elif [[ -z "$ONNX_BACKUP" && -f "$ONNX_ABS" && "$ONNX_VALIDATED" -ne 1 ]]; then
            rm -f "$ONNX_ABS"
            echo "Cleanup: removed unvalidated freshly-written $ONNX_ABS" >&2
        fi
    fi
}
trap cleanup_partial EXIT

# yolo export format=engine handles both .onnx and .engine in one call.
"$YOLO_CMD" export \
    model="$CKPT_ABS" \
    format=engine \
    half="$HALF_FLAG" \
    device=0 \
    workspace="$WORKSPACE_GB" \
    simplify=True \
    ${IMGSZ_FLAG[@]+"${IMGSZ_FLAG[@]}"}

if [[ ! -f "$ENGINE_RAW" ]]; then
    echo "expected $ENGINE_RAW after yolo export but it's missing" >&2
    exit 1
fi

# C3 review: format=engine should also produce a co-located .onnx for
# Python ORT parity testing. Fail if missing or invalid — empty
# source_onnx_sha256 in the sidecar would silently break the parity story.
if [[ ! -s "$ONNX_ABS" ]]; then
    echo "ERROR: $ONNX_ABS missing after format=engine. Ultralytics is expected to" >&2
    echo "       co-emit ONNX alongside the engine; without it the Python ORT parity" >&2
    echo "       artifact is absent and source_onnx_sha256 would be empty." >&2
    exit 1
fi
if ! "$PYBIN" -c "import sys, onnx; onnx.checker.check_model(onnx.load(sys.argv[1]))" "$ONNX_ABS" 2>/dev/null; then
    echo "ERROR: $ONNX_ABS exists but fails onnx.checker validation." >&2
    echo "       Engine may have built from a malformed intermediate graph." >&2
    exit 1
fi

# ONNX has now passed onnx.checker. Mark it validated so cleanup_partial
# preserves it on any subsequent failure (engine rename, sha256 hashing,
# version probes, sidecar write, SIGINT) — losing a validated ONNX would
# destroy the Python ORT parity artifact even though the engine pipeline
# can simply be re-run.
ONNX_VALIDATED=1
# C3 iter-4 fix: disarm the restore branch BEFORE attempting `rm`. If the
# rm fails (permission, read-only fs) under `set -e`, the script exits
# and the trap fires; with ONNX_BACKUP already cleared the trap goes
# straight to the elif (which sees ONNX_VALIDATED=1 and preserves). A
# leftover backup file is harmless trash; restoring it over the validated
# ONNX would silently corrupt the parity artifact.
backup_to_remove="$ONNX_BACKUP"
ONNX_BACKUP=""
if [[ -n "$backup_to_remove" && -f "$backup_to_remove" ]]; then
    rm -f "$backup_to_remove"
fi

# Rename FP32 to the _fp32 suffix (only when FP16=0; FP16 case keeps the
# raw path which already equals ENGINE_ABS).
if [[ "$ENGINE_RAW" != "$ENGINE_ABS" ]]; then
    mv "$ENGINE_RAW" "$ENGINE_ABS"
fi

if [[ ! -s "$ENGINE_ABS" ]]; then
    echo "engine missing or empty at $ENGINE_ABS after build step" >&2
    exit 1
fi

echo "Engine written: $ENGINE_ABS"

# === Atomic sidecar (locked plan Engineering Task 6) ===
# Size-stability protocol: confirm the engine file size is stable across two
# reads before computing SHA256. Handles the rare case where yolo export
# returns before the underlying trtexec child fully flushes the engine to
# disk. Three retries with 0.5s/1s sleeps; no `sync(1)` call needed —
# normal Linux flush behavior is sufficient.
size_of() {
    # Linux first; fall back to BSD/macOS stat. Either way: bytes-only.
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
# Defense-in-depth (B2 review M1): refuse to interpolate empty/non-numeric
# size into JSON even though the loop above guarantees SIZE1 is positive.
if [[ -z "$SIZE1" || ! "$SIZE1" =~ ^[0-9]+$ ]]; then
    echo "engine_size_bytes is empty or non-numeric: '$SIZE1'" >&2
    exit 1
fi
ENGINE_BYTES="$SIZE1"

# Linux has sha256sum; macOS has shasum -a 256. Try both.
sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d' ' -f1
    else
        shasum -a 256 "$1" | cut -d' ' -f1
    fi
}
ENGINE_SHA256=$(sha256_of "$ENGINE_ABS")
SOURCE_PT_SHA256=$(sha256_of "$CKPT_ABS")
# ONNX presence + validity already enforced above (see the format=engine
# post-build check); SOURCE_ONNX_SHA256 is required, not optional.
SOURCE_ONNX_SHA256=$(sha256_of "$ONNX_ABS")
if [[ -z "$SOURCE_ONNX_SHA256" ]]; then
    echo "internal: source_onnx_sha256 unexpectedly empty after validation gate" >&2
    exit 1
fi

# Best-effort tool versions; "unknown" when probe fails (script must NOT
# fail because nvcc isn't on PATH or tensorrt python module is in another
# venv). The sidecar is metadata for traceability, not a build dependency.
TRT_VERSION=$("$PYBIN" -c "import tensorrt; print(tensorrt.__version__)" 2>/dev/null || echo "unknown")
[[ -z "$TRT_VERSION" ]] && TRT_VERSION="unknown"
CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oE 'release [0-9.]+' | head -1 | awk '{print $2}' || true)
[[ -z "$CUDA_VERSION" ]] && CUDA_VERSION="unknown"
JETPACK_VERSION=$(grep -oE 'R[0-9]+ \(release\), REVISION: [0-9.]+' /etc/nv_tegra_release 2>/dev/null | head -1 || true)
[[ -z "$JETPACK_VERSION" ]] && JETPACK_VERSION="unknown"
BUILD_HOST=$(hostname)
BUILD_TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

EXPORTER_CMDLINE="$YOLO_CMD export model=$CKPT_ABS format=engine half=$HALF_FLAG device=0 workspace=$WORKSPACE_GB simplify=True"
[[ -n "${IMGSZ:-}" ]] && EXPORTER_CMDLINE="$EXPORTER_CMDLINE imgsz=$IMGSZ"

# B2 review H2 fix: write JSON via Python json.dumps so any quote/backslash
# in paths, hostnames, or version strings is escaped correctly. Values are
# passed via env vars to avoid a second escaping layer in Python source.
EXPORT_PRECISION="$PRECISION_LABEL" \
EXPORT_EXPORTER="ultralytics yolo" \
EXPORT_EXPORTER_CMDLINE="$EXPORTER_CMDLINE" \
EXPORT_TRT_VERSION="$TRT_VERSION" \
EXPORT_CUDA_VERSION="$CUDA_VERSION" \
EXPORT_JETPACK_VERSION="$JETPACK_VERSION" \
EXPORT_BUILD_HOST="$BUILD_HOST" \
EXPORT_BUILD_TIMESTAMP="$BUILD_TIMESTAMP" \
EXPORT_SOURCE_PT="$CKPT_ABS" \
EXPORT_SOURCE_PT_SHA256="$SOURCE_PT_SHA256" \
EXPORT_SOURCE_ONNX_SHA256="$SOURCE_ONNX_SHA256" \
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
    # trtexec_cmdline kept null for schema parity with future export_deim.sh
    # sidecar (which will record the raw trtexec invocation directly).
    "trtexec_cmdline": None,
    "trt_version": os.environ["EXPORT_TRT_VERSION"],
    "cuda_version": os.environ["EXPORT_CUDA_VERSION"],
    "jetpack_version": os.environ["EXPORT_JETPACK_VERSION"],
    "build_host": os.environ["EXPORT_BUILD_HOST"],
    "build_timestamp": os.environ["EXPORT_BUILD_TIMESTAMP"],
    "source_pt": os.environ["EXPORT_SOURCE_PT"],
    "source_pt_sha256": os.environ["EXPORT_SOURCE_PT_SHA256"],
    "source_onnx_sha256": os.environ["EXPORT_SOURCE_ONNX_SHA256"],
    "engine_sha256": os.environ["EXPORT_ENGINE_SHA256"],
    "engine_size_bytes": int(os.environ["EXPORT_ENGINE_SIZE_BYTES"]),
    "workspace_gb": int(os.environ["EXPORT_WORKSPACE_GB"]),
    # C3 iter-2 fix: provenance for the ALLOW_LARGE_WORKSPACE escape so
    # later artifact audits can tell a 64GB build was an intentional
    # override vs an accidental unit confusion.
    "allow_large_workspace": os.environ["EXPORT_ALLOW_LARGE_WORKSPACE"] == "1",
}
# Atomic sidecar write: dump+fsync into ${SIDECAR}.tmp, then os.replace into
# place. If interrupted, the trap deletes the .tmp (and the engine).
# os.replace is POSIX-atomic on a single filesystem — the final path either
# is the new content or doesn't exist; never partial.
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

# Sidecar landed: clear the trap so a (highly unlikely) post-write failure
# doesn't delete a good engine + good sidecar.
trap - EXIT

echo "Sidecar written: $SIDECAR"
echo "  precision:     $PRECISION_LABEL"
echo "  engine_sha256: $ENGINE_SHA256"
