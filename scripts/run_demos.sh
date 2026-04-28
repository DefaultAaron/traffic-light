#!/usr/bin/env bash
# Sweep every (run, engine, demo) triple through the C++ TRT pipeline demo.
#
# Layout expected:
#   runs/<run>/*.engine                       e.g. runs/yolo26s-r1/best_1280.engine
#   demo/demo*.mp4                            input clips
#
# Produced layout:
#   demo/<run>/<engine_stem>/<demo_name>.mp4                    (TRACK=0, default)
#   demo/<run>_tracker/<engine_stem>/<demo_name>.mp4            (TRACK=1)
#   demo/<run>_tracker/<engine_stem>/<demo_name>.tracks.jsonl   (TRACK=1 + SAVE_TRACK_JSON=1)
#
# The `_tracker` suffix on TRACK=1 lets detector-only and tracker sweeps coexist
# under the same OUT_DIR without overwriting each other.
#
# Sequential by design: each tl_demo invocation blocks the next so a single
# GPU/TRT context is in flight at a time (TRT engines are not safe to share
# concurrently and we want clean per-run latency numbers).
#
# Usage:
#   ./scripts/run_demos.sh                           # detector-only sweep
#   TRACK=1 ./scripts/run_demos.sh                   # tracker sweep
#   CONF=0.3 ./scripts/run_demos.sh
#   TRACK=1 SAVE_TRACK_JSON=1 ./scripts/run_demos.sh
#
# Env overrides:
#   TL_DEMO          path to tl_demo binary       (default: inference/cpp/build/tl_demo)
#   RUNS_DIR         root of engines              (default: runs)
#   DEMOS_DIR        root of input .mp4 clips     (default: demo)
#   OUT_DIR          root for output clips        (default: $DEMOS_DIR)
#   CONF             detector confidence          (default: 0.25)
#   TRACK            1 to enable --track          (default: 0)
#   SKIP_EXIST       1 to skip outputs already    (default: 1)
#                    present (resume friendly)
#   OVERWRITE        1 to delete + regenerate     (default: 0)
#                    stale outputs (forces SKIP_EXIST=0)
#
# Tracker tuning (TRACK=1 only; falls through to tl_demo built-in if unset):
#   ALPHA            tracker EMA alpha
#   MIN_HITS         min hits before confirmed
#   HIGH_THRESH      first-pass IoU/score thresh
#   MATCH_THRESH     match cost cutoff
#   TRACK_BUFFER     frames a lost track survives
#   SAVE_TRACK_JSON  1 to also dump <demo>.tracks.jsonl  (default: 0)

set -euo pipefail

TL_DEMO=${TL_DEMO:-inference/cpp/build/tl_demo}
RUNS_DIR=${RUNS_DIR:-runs}
DEMOS_DIR=${DEMOS_DIR:-demo}
OUT_DIR=${OUT_DIR:-$DEMOS_DIR}
CONF=${CONF:-0.25}
TRACK=${TRACK:-0}
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

# In-flight target paths: the trap below removes these on SIGINT/SIGTERM so a
# signal-killed tl_demo can't leave a truncated .mp4 (which the default
# SKIP_EXIST=1 would later mistake for a finished run). Set right before we
# launch tl_demo, cleared right after it returns.
CURRENT_OUT=""
CURRENT_TRACK_JSON=""

on_signal() {
    local sig=$1
    if [[ -n "$CURRENT_OUT" ]]; then
        echo "  [signal:$sig] removing in-flight $CURRENT_OUT" >&2
        rm -f "$CURRENT_OUT"
        if [[ -n "$CURRENT_TRACK_JSON" ]]; then
            rm -f "$CURRENT_TRACK_JSON"
        fi
    fi
    # Re-raise the signal so the parent shell sees the standard exit status
    # (130 for INT, 143 for TERM) instead of an arbitrary numeric `exit`.
    trap - INT TERM
    kill -s "$sig" $$
}
trap 'on_signal INT'  INT
trap 'on_signal TERM' TERM

# Skip predicate: when TRACK=1 && SAVE_TRACK_JSON=1, both the .mp4 AND the
# companion .tracks.jsonl must be present and non-empty. Skipping on .mp4
# alone would silently drop the JSON when an earlier sweep ran without
# SAVE_TRACK_JSON=1 — the rerun must converge to the advertised state.
should_skip() {
    local mp4=$1
    local trk=$2
    [[ "$SKIP_EXIST" != "1" ]] && return 1
    [[ -s "$mp4" ]] || return 1
    if [[ "$TRACK" == "1" && "$SAVE_TRACK_JSON" == "1" ]]; then
        [[ -s "$trk" ]] || return 1
    fi
    return 0
}

# Engine resolution: filename-based — `*1536* → 1536`, `*1280* → 1280`, else 640.
imgsz_for() {
    local name=$1
    case "$name" in
        *1536*) echo 1536 ;;
        *1280*) echo 1280 ;;
        *)      echo 640  ;;
    esac
}

# Suffix the run output dir when tracking, so detector-only and tracker
# results sit side by side under the same OUT_DIR without colliding.
run_suffix=""
mode_label="detector-only"
if [[ "$TRACK" == "1" ]]; then
    run_suffix="_tracker"
    mode_label="tracker"
fi

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
        out_subdir="$OUT_DIR/${run_name}${run_suffix}/$eng_name"
        mkdir -p "$out_subdir"

        echo
        echo "=== ${run_name}${run_suffix} / $eng_name  (imgsz=$imgsz, conf=$CONF, $mode_label) ==="

        for demo in "${demos[@]}"; do
            demo_name=$(basename "$demo")
            out="$out_subdir/$demo_name"
            track_json_path="${out%.mp4}.tracks.jsonl"
            total=$((total + 1))

            if should_skip "$out" "$track_json_path"; then
                echo "  [skip] $out (already exists)"
                skipped=$((skipped + 1))
                continue
            fi

            # If regenerating, drop the stale .mp4 + companion .tracks.jsonl
            # first so a write failure can't leave half-stale state behind.
            if [[ "$OVERWRITE" == "1" && ( -e "$out" || -e "$track_json_path" ) ]]; then
                echo "  [overwrite] removing stale $out (and any .tracks.jsonl)"
                rm -f "$out" "$track_json_path"
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
                [[ -n "${ALPHA:-}"        ]] && cmd+=( --alpha        "$ALPHA"        )
                [[ -n "${MIN_HITS:-}"     ]] && cmd+=( --min-hits     "$MIN_HITS"     )
                [[ -n "${HIGH_THRESH:-}"  ]] && cmd+=( --high-thresh  "$HIGH_THRESH"  )
                [[ -n "${MATCH_THRESH:-}" ]] && cmd+=( --match-thresh "$MATCH_THRESH" )
                [[ -n "${TRACK_BUFFER:-}" ]] && cmd+=( --track-buffer "$TRACK_BUFFER" )
                if [[ "$SAVE_TRACK_JSON" == "1" ]]; then
                    cmd+=( --track-json "$track_json_path" )
                fi
            fi

            echo "  -> $demo_name"
            # Mark the in-flight target so the INT/TERM trap can clean up if
            # the user Ctrl+C's mid-write. Only the .tracks.jsonl path is
            # tracked when we actually requested one (TRACK=1 && SAVE_TRACK_JSON=1).
            CURRENT_OUT="$out"
            if [[ "$TRACK" == "1" && "$SAVE_TRACK_JSON" == "1" ]]; then
                CURRENT_TRACK_JSON="$track_json_path"
            else
                CURRENT_TRACK_JSON=""
            fi
            if "${cmd[@]}"; then
                done_count=$((done_count + 1))
            else
                rc=$?
                echo "  [fail] rc=$rc for $demo_name -> $out" >&2
                failed=$((failed + 1))
                # Remove a partial/empty output so a re-run picks it up cleanly.
                [[ -f "$out" && ! -s "$out" ]] && rm -f "$out"
            fi
            CURRENT_OUT=""
            CURRENT_TRACK_JSON=""
        done
    done
done

elapsed=$(( $(date +%s) - start_ts ))
echo
echo "Sweep ($mode_label) done: $done_count ran, $skipped skipped, $failed failed, $total total (${elapsed}s)"
(( failed == 0 ))
