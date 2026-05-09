#!/usr/bin/env bash
# Diagnose DEIM-S + DEIM-M residual det count after commit 8a3ece8 (per-query
# letterbox dedup). Runs the new tl_demo binary against FP16 and FP32 engines
# of both detectors on a single source clip at conf=0.25, captures stdout,
# and samples 3 overlay frames per run for visual hand-counting if stdout is
# silent.
#
# Output (under repo root): logs/deim_diag/
#   summary.txt            — env, build state, condensed first-30-frame log,
#                            per-run exit codes
#   <model>_fp{16,32}.log  — full tl_demo stdout per run
#   <model>_fp{16,32}_t{1,3,5}s.png — sampled overlay frames per run
#   bundle.tgz             — single tar to ship back; only created if ALL 4
#                            runs exited 0
#
# Run from repo root on Orin:
#   chmod +x scripts/_diag_deim_dets.sh
#   ./scripts/_diag_deim_dets.sh

set -euo pipefail
cd "$(dirname "$0")/.."

OUT=logs/deim_diag
mkdir -p "$OUT"

# Models we diagnose. Add another row here to extend coverage.
MODELS=( s m )

# Build the closed allowlist of artifact names the script writes. Cleanup
# (below) and bundling (end of script) iterate this list; nothing outside it
# is ever touched.
artifacts=( summary.txt bundle.tgz )
for m in "${MODELS[@]}"; do
    for prec in fp16 fp32; do
        tag="${m}_${prec}"
        artifacts+=( "${tag}.log" "out_${tag}.mp4" )
        for t in 1 3 5; do
            artifacts+=( "${tag}_t${t}s.png" )
        done
    done
done

# Clear stale artifacts BEFORE we write anything new. Without this, a prior
# successful run's out_*.mp4 + .png + bundle.tgz remain on disk; if the
# current run fails partway, the user can ship a bundle that mixes stale
# frames + fresh summary.txt and misread the result.
for a in "${artifacts[@]}"; do
    rm -f "$OUT/$a"
done

# Source clip — pick the first match in canonical demo layout.
SRC=$(ls demo/demo1.mp4 2>/dev/null || ls demo/*.mp4 2>/dev/null | head -1 || true)
if [[ -z "$SRC" ]]; then
    echo "FATAL: no source video found under demo/*.mp4" >&2
    exit 2
fi

# Auto-discover engines so we work under either canonical layout
# (runs/<run>/*.engine per run_demos.sh, or runs/detect/<run>/*.engine
# per export_deim.sh's example checkpoint path). Hard-fail on missing or
# ambiguous matches.
find_engine() {
    local model=$1 pattern=$2
    local hits
    hits=$(find runs -type f -name "$pattern" -path "*deim_dfine_${model}*" 2>/dev/null | sort)
    if [[ -z "$hits" ]]; then
        echo "FATAL: no DEIM-${model^^} engine matching '$pattern' under runs/" >&2
        exit 3
    fi
    if [[ $(wc -l <<<"$hits") -gt 1 ]]; then
        echo "FATAL: multiple DEIM-${model^^} engines match '$pattern':" >&2
        echo "$hits" >&2
        echo "       narrow with explicit ENG_<MODEL>_<PREC> env override" >&2
        exit 3
    fi
    echo "$hits"
}

# Per-model engine paths. Override examples:
#   ENG_S_FP16=runs/.../best_stg2.engine ENG_M_FP32=... ./scripts/...
declare -A ENG
for m in "${MODELS[@]}"; do
    M=${m^^}
    var_fp16="ENG_${M}_FP16"
    var_fp32="ENG_${M}_FP32"
    ENG["${m}_fp16"]=${!var_fp16:-$(find_engine "$m" 'best_stg2.engine')}
    ENG["${m}_fp32"]=${!var_fp32:-$(find_engine "$m" 'best_stg2_fp32.engine')}
done

BIN=./inference/cpp/build/tl_demo
if [[ ! -x "$BIN" ]]; then
    echo "FATAL: binary not found or not executable: $BIN" >&2
    exit 4
fi

{
  echo "=== environment ==="
  date
  uname -a
  echo
  echo "=== inputs ==="
  echo "src   : $SRC"
  for m in "${MODELS[@]}"; do
      for prec in fp16 fp32; do
          tag="${m}_${prec}"
          path=${ENG[$tag]}
          echo "${tag} : $path  ($(stat -c %y "$path" 2>/dev/null))"
      done
  done
  echo "bin   : $BIN  ($(stat -c %y "$BIN" 2>/dev/null))"
  echo
  echo "=== source has phase 0 sort + phase 2 NMS? expected count >= 3 ==="
  # Sentinel: stable_sort + sorted_idx + kIoUThresh — phase-0 sort is
  # the load-bearing piece that landed alongside per-class IoU NMS.
  # The earlier `seen_lbox` sentinel went stale once the C++ dedup
  # buffer was consolidated into `survivors` (B2 iter-1 finding I-3).
  grep -cE 'std::stable_sort|sorted_idx|kIoUThresh' inference/cpp/src/trt_pipeline.cpp
  echo
  echo "=== demo.cpp per-frame log format ==="
  grep -nE "Frame|dets|detect=" inference/cpp/src/demo.cpp | head -20
  echo
} > "$OUT/summary.txt"

# Run tl_demo and capture its REAL exit code (tee always succeeds, so we use
# PIPESTATUS[0]). Returns the exit code; the caller tallies per-run statuses.
run_one() {
    local tag=$1 eng=$2
    local rc
    echo "[$tag] running tl_demo..."
    set +e
    "$BIN" --source "$SRC" --model "$eng" --conf 0.25 --imgsz 640 --no-show \
        --save "$OUT/out_${tag}.mp4" 2>&1 | tee "$OUT/${tag}.log"
    rc=${PIPESTATUS[0]}
    set -e
    echo "[$tag] tl_demo exit=$rc"
    return "$rc"
}

declare -A RC
for m in "${MODELS[@]}"; do
    for prec in fp16 fp32; do
        tag="${m}_${prec}"
        set +e
        run_one "$tag" "${ENG[$tag]}"
        RC[$tag]=$?
        set -e
    done
done

# Sample 3 frames per run. Skip if the .mp4 is missing (run failed) or
# ffmpeg is absent — neither is fatal for the diagnostic itself; the .log
# still has the answer.
sample_frames() {
    local tag=$1
    local mp4="$OUT/out_${tag}.mp4"
    if [[ ! -s "$mp4" ]]; then
        echo "[$tag] skipping frame sampling — overlay .mp4 missing or empty"
        return
    fi
    if ! command -v ffmpeg >/dev/null 2>&1; then
        echo "[$tag] skipping frame sampling — ffmpeg not on PATH"
        return
    fi
    for t in 1 3 5; do
        ffmpeg -y -ss 0:0:${t} -i "$mp4" -frames:v 1 -q:v 2 \
            "$OUT/${tag}_t${t}s.png" 2>/dev/null || true
    done
}
for m in "${MODELS[@]}"; do
    for prec in fp16 fp32; do
        sample_frames "${m}_${prec}"
    done
done

# Append condensed summary, including per-run exit codes.
{
  echo
  echo "=== run exit codes ==="
  for m in "${MODELS[@]}"; do
      for prec in fp16 fp32; do
          tag="${m}_${prec}"
          printf "  %-10s %d\n" "$tag" "${RC[$tag]}"
      done
  done
  echo
  echo "=== startup banners ==="
  for m in "${MODELS[@]}"; do
      for prec in fp16 fp32; do
          tag="${m}_${prec}"
          [[ -s "$OUT/${tag}.log" ]] && {
              echo "--- $tag ---"
              grep -hE "^Model:|imgsz=|conf=" "$OUT/${tag}.log" 2>/dev/null | head -5 || true
          }
      done
  done
  echo
  for m in "${MODELS[@]}"; do
      for prec in fp16 fp32; do
          tag="${m}_${prec}"
          echo "=== first 30 frame log lines, $tag ==="
          [[ -s "$OUT/${tag}.log" ]] && grep -E "Frame [0-9]" "$OUT/${tag}.log" 2>/dev/null | head -30 || true
          echo
      done
  done
  echo "=== det-count keyword scan ==="
  for m in "${MODELS[@]}"; do
      for prec in fp16 fp32; do
          tag="${m}_${prec}"
          [[ -s "$OUT/${tag}.log" ]] && grep -niE "det(s|ections)?[ =:]" "$OUT/${tag}.log" 2>/dev/null | head -10 || true
      done
  done
} >> "$OUT/summary.txt"

ls -la "$OUT/"

# Refuse to bundle if any run crashed.
fails=()
for m in "${MODELS[@]}"; do
    for prec in fp16 fp32; do
        tag="${m}_${prec}"
        [[ "${RC[$tag]}" -ne 0 ]] && fails+=( "${tag}=${RC[$tag]}" )
    done
done
if (( ${#fails[@]} > 0 )); then
    echo
    echo "FATAL: at least one tl_demo run exited non-zero: ${fails[*]}" >&2
    echo "       inspect $OUT/*.log; not bundling" >&2
    exit 1
fi

# Bundle only artifacts that exist + are non-empty. tar of a missing path
# would error and false-report otherwise.
bundle_files=( summary.txt )
for m in "${MODELS[@]}"; do
    for prec in fp16 fp32; do
        tag="${m}_${prec}"
        [[ -s "$OUT/${tag}.log" ]] && bundle_files+=( "${tag}.log" )
        for t in 1 3 5; do
            f="${tag}_t${t}s.png"
            [[ -s "$OUT/$f" ]] && bundle_files+=( "$f" )
        done
    done
done
tar czf "$OUT/bundle.tgz" -C "$OUT" "${bundle_files[@]}"
echo
echo "Done. Bundle: $OUT/bundle.tgz"
