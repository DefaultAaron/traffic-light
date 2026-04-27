#!/usr/bin/env bash
# Sweep every (run, engine, demo) triple through the C++ TRT pipeline demo.
#
# Layout expected on the Orin:
#   runs/<run>/*.engine                       e.g. runs/yolo26s-r1/best_1280.engine
#   demo/demo*.mp4                            input clips
#
# Produced layout:
#   demo/<run>/<engine_stem>/<demo_name>.mp4
#
# Sequential by design: each tl_demo invocation blocks the next one so a single
# GPU/TRT context is in flight at any time (TRT engines are not safe to share
# concurrently and we want clean latency numbers per run).
#
# Usage:
#   ./scripts/run_demos_all_engines.sh
#   TL_DEMO=/path/to/tl_demo ./scripts/run_demos_all_engines.sh
#   CONF=0.3 TRACK=1 ./scripts/run_demos_all_engines.sh
#
# Env overrides:
#   TL_DEMO      path to tl_demo binary       (default: inference/cpp/build/tl_demo)
#   RUNS_DIR     root of engines              (default: runs)
#   DEMOS_DIR    root of input .mp4 clips     (default: demo)
#   OUT_DIR      root for output clips        (default: $DEMOS_DIR)
#   CONF         detector confidence          (default: 0.25)
#   TRACK        1 to enable --track          (default: 0)
#   SKIP_EXIST   1 to skip outputs already    (default: 1)
#                present (resume friendly)
#   OVERWRITE    1 to delete + regenerate     (default: 0)
#                stale outputs (forces SKIP_EXIST=0)

set -euo pipefail

TL_DEMO=${TL_DEMO:-inference/cpp/build/tl_demo}
RUNS_DIR=${RUNS_DIR:-runs}
DEMOS_DIR=${DEMOS_DIR:-demo}
OUT_DIR=${OUT_DIR:-$DEMOS_DIR}
CONF=${CONF:-0.25}
TRACK=${TRACK:-0}
SKIP_EXIST=${SKIP_EXIST:-1}
OVERWRITE=${OVERWRITE:-0}
# OVERWRITE forces a fresh run; the two flags would otherwise contradict.
[[ "$OVERWRITE" == "1" ]] && SKIP_EXIST=0

if [[ ! -x "$TL_DEMO" ]]; then
    echo "tl_demo binary not found or not executable: $TL_DEMO" >&2
    echo "Build it first:" >&2
    echo "  cmake -S inference/cpp -B inference/cpp/build && cmake --build inference/cpp/build -j" >&2
    exit 1
fi

# Infer inference resolution from engine filename. The server carries both
# `best_1280.engine` and `best-1280.engine` conventions — match either.
imgsz_for() {
    local name=$1
    case "$name" in
        *1536*) echo 1536 ;;
        *1280*) echo 1280 ;;
        *)      echo 640  ;;
    esac
}

shopt -s nullglob

# Collect + sort demos once so every engine sees the same order.
demos=( "$DEMOS_DIR"/demo*.mp4 )
if (( ${#demos[@]} == 0 )); then
    echo "no demo videos found under $DEMOS_DIR/demo*.mp4" >&2
    exit 1
fi
IFS=$'\n' demos=( $(printf '%s\n' "${demos[@]}" | sort -V) )
unset IFS

runs=( "$RUNS_DIR"/*/ )
if (( ${#runs[@]} == 0 )); then
    echo "no runs found under $RUNS_DIR/*/" >&2
    exit 1
fi

total=0
done_count=0
skipped=0
failed=0

start_ts=$(date +%s)

for run_path in "${runs[@]}"; do
    run_name=$(basename "$run_path")
    engines=( "$run_path"*.engine )
    if (( ${#engines[@]} == 0 )); then
        echo "[$run_name] no .engine files — skipping"
        continue
    fi
    IFS=$'\n' engines=( $(printf '%s\n' "${engines[@]}" | sort) )
    unset IFS

    for eng in "${engines[@]}"; do
        eng_name=$(basename "$eng" .engine)
        imgsz=$(imgsz_for "$eng_name")
        out_subdir="$OUT_DIR/$run_name/$eng_name"
        mkdir -p "$out_subdir"

        echo
        echo "=== $run_name / $eng_name  (imgsz=$imgsz, conf=$CONF, track=$TRACK) ==="

        for demo in "${demos[@]}"; do
            demo_name=$(basename "$demo")
            out="$out_subdir/$demo_name"
            total=$((total + 1))

            if [[ "$SKIP_EXIST" == "1" && -s "$out" ]]; then
                echo "  [skip] $out (already exists)"
                skipped=$((skipped + 1))
                continue
            fi

            # If we're regenerating, drop the stale file first so a write
            # failure can't leave the old result behind to be silently kept.
            if [[ "$OVERWRITE" == "1" && -e "$out" ]]; then
                echo "  [overwrite] removing stale $out"
                rm -f "$out"
            fi

            cmd=( "$TL_DEMO"
                  --source "$demo"
                  --model  "$eng"
                  --conf   "$CONF"
                  --imgsz  "$imgsz"
                  --no-show
                  --save   "$out" )
            if [[ "$TRACK" == "1" ]]; then
                cmd+=( --track )
            fi

            echo "  -> $demo_name"
            if "${cmd[@]}"; then
                done_count=$((done_count + 1))
            else
                rc=$?
                echo "  [fail] rc=$rc for $demo_name -> $out" >&2
                failed=$((failed + 1))
                # Remove a partial/empty output so a re-run picks it up cleanly.
                [[ -f "$out" && ! -s "$out" ]] && rm -f "$out"
            fi
        done
    done
done

elapsed=$(( $(date +%s) - start_ts ))
echo
echo "Sweep done: $done_count ran, $skipped skipped, $failed failed, $total total (${elapsed}s)"
(( failed == 0 ))
