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
#   ENGINE_FILTER    substring; if set, only      (default: empty)
#                    sweep engines whose stem
#                    contains it (e.g. `_fp32`)
#   ENGINE_EXCLUDE   substring; if set, skip      (default: empty)
#                    engines whose stem contains
#                    it (e.g. `_fp32` for prod-only)
#
# Note: ENGINE_FILTER / ENGINE_EXCLUDE are unanchored substring matches
# against the engine basename (no path component). Today's only convention
# is `_fp32` as a trailing suffix (`<ckpt>_fp32.engine`); if future engines
# add other tags (e.g. `_int8`, `_mixed`), keep the tag as a trailing suffix
# before `.engine` to avoid accidental matches against unrelated stems.
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
ENGINE_FILTER=${ENGINE_FILTER:-}
ENGINE_EXCLUDE=${ENGINE_EXCLUDE:-}
[[ "$OVERWRITE" == "1" ]] && SKIP_EXIST=0

if [[ -n "$ENGINE_FILTER" && -n "$ENGINE_EXCLUDE" && "$ENGINE_FILTER" == "$ENGINE_EXCLUDE" ]]; then
    echo "ENGINE_FILTER and ENGINE_EXCLUDE are both set to '$ENGINE_FILTER' — every engine would be skipped" >&2
    exit 1
fi

# tl_demo binary check is deferred until after engine discovery — see
# below. A truly empty workspace (no engines AND no tl_demo build) must
# resolve to "nothing to sweep → exit 0", not "missing binary → exit 1",
# so the binary requirement is only enforced when we actually have work.

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

# Engine resolution: prefer the engine sidecar's `imgsz` when present;
# else fall back to the legacy filename heuristic (`*1536* → 1536`,
# `*1280* → 1280`, else 640). Sidecar lives at `<engine>.meta.json`
# next to the engine and is emitted by both export_yolo.sh and
# export_deim.sh.
#
# C3 iter-2 C2 fix: when a sidecar EXISTS but doesn't yield a valid
# positive imgsz (corrupt JSON, missing field, non-integer), hard-fail
# with a clear error rather than silently falling through to the
# filename heuristic. Per the locked-plan engine sidecar contract, an
# engine without a matching trusted sidecar is untrusted; demoing it at
# the wrong shape is exactly the failure mode the contract exists to
# prevent. The filename-heuristic fallback path is reserved for legacy
# engines built before the sidecar contract landed (no sidecar at all).
imgsz_for() {
    local engine_path=$1
    local name=$2
    local sidecar="${engine_path}.meta.json"
    if [[ -s "$sidecar" ]]; then
        local sidecar_imgsz
        sidecar_imgsz=$(python3 -c "
import json, sys
try:
    with open(sys.argv[1]) as f:
        d = json.load(f)
    v = d.get('imgsz')
    if v is not None:
        print(int(v))
except Exception:
    pass
" "$sidecar" 2>/dev/null)
        if [[ -n "$sidecar_imgsz" && "$sidecar_imgsz" =~ ^[0-9]+$ && "$sidecar_imgsz" -gt 0 ]]; then
            echo "$sidecar_imgsz"
            return 0
        fi
        echo "ERROR: $sidecar exists but does not yield a valid positive imgsz." >&2
        echo "       The engine is untrusted per the sidecar contract; re-export to repair." >&2
        echo "       (Filename-heuristic fallback only applies when the sidecar is ABSENT.)" >&2
        return 1
    fi
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

# Two-pass design: the binary check + demos requirement must NOT fire
# until we know at least one (run × engine) actually passes the filters.
# Otherwise a CI-clean workspace, or a typo'd ENGINE_FILTER that rejects
# everything, gets misreported as "missing tl_demo" / "no demo videos"
# instead of "nothing to sweep" / "filter rejected all engines".
#
# Pass 1 (silent): discover runs, count engines that survive the filters
# into selected_engines. No side effects.
# Pass 2 (verbose): the original sweep loop, with announcements + exec.

runs=( "$RUNS_DIR"/*/ )
selected_engines=0
for run_path in "${runs[@]}"; do
    run_name=$(basename "$run_path")
    [[ "$run_name" == *_tracker ]] && continue
    engines=( "$run_path"*.engine )
    (( ${#engines[@]} == 0 )) && continue
    for eng in "${engines[@]}"; do
        eng_name=$(basename "$eng" .engine)
        [[ -n "$ENGINE_FILTER"  && "$eng_name" != *"$ENGINE_FILTER"*  ]] && continue
        [[ -n "$ENGINE_EXCLUDE" && "$eng_name" == *"$ENGINE_EXCLUDE"* ]] && continue
        selected_engines=$((selected_engines + 1))
    done
done

# Branch on the selection count BEFORE requiring demos / tl_demo. Empty
# selection with a filter set is loud-fail (likely typo); without filter
# it's "nothing to sweep" → exit 0.
if (( selected_engines == 0 )); then
    if [[ -n "$ENGINE_FILTER" || -n "$ENGINE_EXCLUDE" ]]; then
        echo "ENGINE_FILTER='$ENGINE_FILTER' / ENGINE_EXCLUDE='$ENGINE_EXCLUDE' selected zero engines under $RUNS_DIR — refusing to exit successfully" >&2
        exit 1
    fi
    echo "no engines under $RUNS_DIR — nothing to sweep"
    exit 0
fi

demos=( "$DEMOS_DIR"/demo*.mp4 )
if (( ${#demos[@]} == 0 )); then
    echo "no demo videos found under $DEMOS_DIR/demo*.mp4" >&2
    exit 1
fi
IFS=$'\n' demos=( $(printf '%s\n' "${demos[@]}" | sort -V) )
unset IFS

# Engines selected AND demos present — only now is tl_demo a hard prereq.
# Failing here points the user at the right action (build the binary)
# rather than masking it as "no work to do".
if [[ ! -x "$TL_DEMO" ]]; then
    echo "tl_demo binary not found or not executable: $TL_DEMO" >&2
    echo "Build it first:" >&2
    echo "  cmake -S inference/cpp -B inference/cpp/build && cmake --build inference/cpp/build -j" >&2
    exit 1
fi

total=0
done_count=0
skipped=0
failed=0

start_ts=$(date +%s)

# Pass 2: actual sweep. Filter checks are repeated here because pass 1 was
# silent — pass 2 logs each skip so the operator can see why a particular
# engine was excluded. The selected_engines counter from pass 1 is the
# authoritative selection size; this pass does not increment it.
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
        if [[ -n "$ENGINE_FILTER" && "$eng_name" != *"$ENGINE_FILTER"* ]]; then
            echo "[$run_name/$eng_name] does not match ENGINE_FILTER=$ENGINE_FILTER — skipping"
            continue
        fi
        if [[ -n "$ENGINE_EXCLUDE" && "$eng_name" == *"$ENGINE_EXCLUDE"* ]]; then
            echo "[$run_name/$eng_name] matches ENGINE_EXCLUDE=$ENGINE_EXCLUDE — skipping"
            continue
        fi
        imgsz=$(imgsz_for "$eng" "$eng_name")
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
echo "Sweep ($mode_label) done: $done_count ran, $skipped skipped, $failed failed, $total total"
echo "  selected $selected_engines engine(s) after ENGINE_FILTER/ENGINE_EXCLUDE; elapsed ${elapsed}s"

# Note: the zero-selection branch is handled in pass 1 above (before
# demos / tl_demo are required), so we don't need a final-tally guard
# here. Reaching this point means selected_engines >= 1.

(( failed == 0 ))
