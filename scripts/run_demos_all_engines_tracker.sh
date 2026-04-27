#!/usr/bin/env bash
# Same sweep as run_demos_all_engines.sh but always with --track enabled,
# and outputs land in a sibling tree suffixed with _tracker so detector-only
# and tracker runs sit side by side without collisions.
#
# Layout expected on the Orin:
#   runs/<run>/*.engine                       e.g. runs/yolo26m-r1/best_1280.engine
#   demo/demo*.mp4                            input clips
#
# Produced layout:
#   demo/<run>_tracker/<engine_stem>/<demo_name>.mp4
#   demo/<run>_tracker/<engine_stem>/<demo_name>.tracks.jsonl   (if SAVE_TRACK_JSON=1)
#
# Usage:
#   ./scripts/run_demos_all_engines_tracker.sh
#   CONF=0.3 ./scripts/run_demos_all_engines_tracker.sh
#   SAVE_TRACK_JSON=1 ./scripts/run_demos_all_engines_tracker.sh
#
# Env overrides (in addition to those in run_demos_all_engines.sh):
#   TL_DEMO          path to tl_demo binary       (default: inference/cpp/build/tl_demo)
#   RUNS_DIR         root of engines              (default: runs)
#   DEMOS_DIR        root of input .mp4 clips     (default: demo)
#   OUT_DIR          root for output clips        (default: $DEMOS_DIR)
#   CONF             detector confidence          (default: 0.25)
#   ALPHA            tracker EMA alpha            (default: tl_demo built-in)
#   MIN_HITS         min hits before confirmed    (default: tl_demo built-in)
#   HIGH_THRESH      first-pass IoU/score thresh  (default: tl_demo built-in)
#   MATCH_THRESH     match cost cutoff            (default: tl_demo built-in)
#   TRACK_BUFFER     frames a lost track survives (default: tl_demo built-in)
#   SAVE_TRACK_JSON  1 to also dump tracks.jsonl  (default: 0)
#   SKIP_EXIST       1 to skip outputs already    (default: 1)
#                    present (resume friendly)
#   OVERWRITE        1 to delete + regenerate     (default: 0)
#                    stale .mp4 (and matching
#                    .tracks.jsonl); forces
#                    SKIP_EXIST=0

set -euo pipefail

TL_DEMO=${TL_DEMO:-inference/cpp/build/tl_demo}
RUNS_DIR=${RUNS_DIR:-runs}
DEMOS_DIR=${DEMOS_DIR:-demo}
OUT_DIR=${OUT_DIR:-$DEMOS_DIR}
CONF=${CONF:-0.25}
SKIP_EXIST=${SKIP_EXIST:-1}
SAVE_TRACK_JSON=${SAVE_TRACK_JSON:-0}
OVERWRITE=${OVERWRITE:-0}
[[ "$OVERWRITE" == "1" ]] && SKIP_EXIST=0

if [[ ! -x "$TL_DEMO" ]]; then
    echo "tl_demo binary not found or not executable: $TL_DEMO" >&2
    echo "Build it first:" >&2
    echo "  cmake -S inference/cpp -B inference/cpp/build && cmake --build inference/cpp/build -j" >&2
    exit 1
fi

imgsz_for() {
    local name=$1
    case "$name" in
        *1536*) echo 1536 ;;
        *1280*) echo 1280 ;;
        *)      echo 640  ;;
    esac
}

shopt -s nullglob

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
    # Skip already-tracker-suffixed dirs in case OUT_DIR == RUNS_DIR.
    [[ "$run_name" == *_tracker ]] && continue

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
        out_subdir="$OUT_DIR/${run_name}_tracker/$eng_name"
        mkdir -p "$out_subdir"

        echo
        echo "=== ${run_name}_tracker / $eng_name  (imgsz=$imgsz, conf=$CONF, track=1) ==="

        for demo in "${demos[@]}"; do
            demo_name=$(basename "$demo")
            out="$out_subdir/$demo_name"
            total=$((total + 1))

            if [[ "$SKIP_EXIST" == "1" && -s "$out" ]]; then
                echo "  [skip] $out (already exists)"
                skipped=$((skipped + 1))
                continue
            fi

            # If regenerating, drop the stale .mp4 + companion .tracks.jsonl
            # first so a write failure can't leave half-stale state behind.
            if [[ "$OVERWRITE" == "1" ]]; then
                track_json_path="${out%.mp4}.tracks.jsonl"
                if [[ -e "$out" || -e "$track_json_path" ]]; then
                    echo "  [overwrite] removing stale $out (and any .tracks.jsonl)"
                    rm -f "$out" "$track_json_path"
                fi
            fi

            cmd=( "$TL_DEMO"
                  --source "$demo"
                  --model  "$eng"
                  --conf   "$CONF"
                  --imgsz  "$imgsz"
                  --no-show
                  --save   "$out"
                  --track )

            [[ -n "${ALPHA:-}"        ]] && cmd+=( --alpha        "$ALPHA"        )
            [[ -n "${MIN_HITS:-}"     ]] && cmd+=( --min-hits     "$MIN_HITS"     )
            [[ -n "${HIGH_THRESH:-}"  ]] && cmd+=( --high-thresh  "$HIGH_THRESH"  )
            [[ -n "${MATCH_THRESH:-}" ]] && cmd+=( --match-thresh "$MATCH_THRESH" )
            [[ -n "${TRACK_BUFFER:-}" ]] && cmd+=( --track-buffer "$TRACK_BUFFER" )

            if [[ "$SAVE_TRACK_JSON" == "1" ]]; then
                cmd+=( --track-json "${out%.mp4}.tracks.jsonl" )
            fi

            echo "  -> $demo_name"
            if "${cmd[@]}"; then
                done_count=$((done_count + 1))
            else
                rc=$?
                echo "  [fail] rc=$rc for $demo_name -> $out" >&2
                failed=$((failed + 1))
                [[ -f "$out" && ! -s "$out" ]] && rm -f "$out"
            fi
        done
    done
done

elapsed=$(( $(date +%s) - start_ts ))
echo
echo "Tracker sweep done: $done_count ran, $skipped skipped, $failed failed, $total total (${elapsed}s)"
(( failed == 0 ))
