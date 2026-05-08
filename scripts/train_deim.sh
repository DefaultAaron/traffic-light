#!/usr/bin/env bash
# Train DEIM-D-FINE on the traffic-light dataset.
#
# Usage:
#   scripts/train_deim.sh <size> [extra torchrun args...]
#   scripts/train_deim.sh s --nproc_per_node=1                     # single GPU
#   scripts/train_deim.sh m -t weights/deim_dfine_m_coco.pth       # fine-tune from COCO checkpoint
#
# Prerequisite: run `uv run python scripts/yolo_to_coco.py` once so
# data/merged/annotations/instances_{train,val}.json exist.

set -e

if [[ -z "${1:-}" ]]; then
    echo "usage: scripts/train_deim.sh <n|s|m|l> [extra args]" >&2
    exit 1
fi
SIZE="${1//$'\r'/}"       # strip CR in case of CRLF
shift

case "$SIZE" in
    n|s|m|l) ;;
    *) echo "size must be one of: n s m l" >&2; exit 1 ;;
esac

CFG="configs/deim_dfine/deim_hgnetv2_${SIZE}_traffic_light.yml"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT/DEIM"

NPROC="${NPROC:-1}"
PORT="${PORT:-7777}"
SEED="${SEED:-0}"        # override with: SEED=42 scripts/train_deim.sh s

# Detect resume mode + lift any seed override from passthrough args.
#
# Resume rule (project policy): resuming an existing run MUST NOT overwrite
# SEED.txt — the original run dir owns the correct seed metadata. AND the
# runtime --seed passed to DEIM MUST match SEED.txt, otherwise DEIM's
# `setup_distributed(..., seed=args.seed)` (DEIM/train.py:31) re-seeds the
# RNG to a value the metadata doesn't claim. The fix on resume: read
# SEED.txt and source the seed FROM there, ignoring the env $SEED and any
# CLI `--seed=...`. On fresh runs the existing direction (env $SEED → SEED.txt)
# is preserved.
#
# DEIM resume flag forms (argparse `-r` / `--resume` accepts space, `=`, and
# attached-short variants): `-r <ckpt>`, `-r=<ckpt>`, `-r<ckpt>`,
# `--resume <ckpt>`, `--resume=<ckpt>`.
# Walk passthrough args ONCE to extract: resume flag, seed override, output-dir
# override. Each parse is a separate guard so they compose cleanly.
#
# DEIM/ is gitignored (vendored upstream), so any local edit to
# DEIM/train.py — including `allow_abbrev=False` on the ArgumentParser —
# does NOT ship with this wrapper. A fresh clone gets the upstream
# default `allow_abbrev=True`, which prefix-matches `--o`/`--r`/`--se`
# etc. to one of `--output-dir`/`--resume`/`--seed`. The wrapper-side
# rejection below is therefore the PRIMARY (not "defense-in-depth")
# defense against the prefix-match desync, and must be exhaustive.
#
# Recommended local hardening (not required, not shipped): set
# `allow_abbrev=False` on DEIM/train.py's ArgumentParser. The wrapper
# behaves identically with or without that local edit.
IS_RESUME=0
OUTPUT_DIR_OVERRIDE=""
prev=""
for arg in "$@"; do
    # Reject empty values for the three flags whose value the wrapper
    # depends on — `--seed=` writes empty SEED.txt, `--output-dir=` makes
    # the wrapper fall back to config default while DEIM accepts an empty
    # string as the actual run dir, and `--resume=` puts the wrapper in
    # resume mode while DEIM treats `args.resume==""` as falsey and starts
    # fresh — both ways desync wrapper and trainer.
    case "$arg" in
        --seed=)         echo "ERROR: '--seed=' has no value. Use --seed=<n> or --seed <n>." >&2; exit 1 ;;
        --output-dir=)   echo "ERROR: '--output-dir=' has no value. Use --output-dir=<dir> or --output-dir <dir>." >&2; exit 1 ;;
        --resume=)       echo "ERROR: '--resume=' has no value. Use --resume=<ckpt> or --resume <ckpt>." >&2; exit 1 ;;
        -r=)             echo "ERROR: '-r=' has no value. Use -r=<ckpt> or -r <ckpt>." >&2; exit 1 ;;
    esac
    # Reject space-form `--output-dir <next>` where <next> is empty or starts
    # with `-` (caller forgot the value).
    case "$prev" in
        --seed|--output-dir|--resume|-r)
            case "$arg" in
                ""|-*) echo "ERROR: '$prev' missing value (got '$arg')." >&2; exit 1 ;;
            esac ;;
    esac
    case "$prev" in
        --seed)        SEED="$arg" ;;
        --output-dir)  OUTPUT_DIR_OVERRIDE="$arg" ;;
    esac
    case "$arg" in
        --seed=*)        SEED="${arg#--seed=}" ;;
        --output-dir=*)  OUTPUT_DIR_OVERRIDE="${arg#--output-dir=}" ;;
    esac
    case "$arg" in
        -r|--resume)   IS_RESUME=1 ;;
        --resume=*)    IS_RESUME=1 ;;
        -r=*)          IS_RESUME=1 ;;
        -r?*)          IS_RESUME=1 ;;        # attached short form: -rfoo
    esac
    # Reject every argparse-unique abbreviation prefix that resolves to
    # `--output-dir`/`--resume`/`--seed` under DEIM's default
    # allow_abbrev=True. Enumerated against DEIM/train.py:64-81 long
    # options: only `--output-dir` starts with `--o`, only `--resume`
    # starts with `--r` (--print-rank/--local-rank don't), and only
    # `--seed` matches `--se` (--summary-dir is ambiguous at `--s`
    # alone, so `--s` is rejected by argparse already; `--se` resolves
    # uniquely to --seed).
    case "$arg" in
        --output-dir|--output-dir=*) ;;
        --output|--output=*|--output-|--output-=*|--outp|--outp=*|--outpu|--outpu=*|--output-d|--output-d=*|--output-di|--output-di=*|--ou|--ou=*|--out|--out=*|--o|--o=*)
            echo "ERROR: abbreviated '$arg' is wrapper-blind. Use --output-dir or --output-dir=<dir> exactly." >&2
            exit 1 ;;
    esac
    case "$arg" in
        --resume|--resume=*) ;;
        --r|--r=*|--re|--re=*|--res|--res=*|--resu|--resu=*|--resum|--resum=*)
            echo "ERROR: abbreviated '$arg' is wrapper-blind. Use --resume or --resume=<ckpt> exactly." >&2
            exit 1 ;;
    esac
    case "$arg" in
        --seed|--seed=*) ;;
        --see|--see=*|--se|--se=*)
            echo "ERROR: abbreviated '$arg' is wrapper-blind. Use --seed or --seed=<n> exactly." >&2
            exit 1 ;;
    esac
    prev="$arg"
done

# Trailing-flag-with-no-value detection. The in-loop `case "$prev" in ... -*)`
# guard fires only when there IS a next arg; if any of `--seed`, `--output-dir`,
# `--resume`, or `-r` is THE LAST arg, prev still equals that flag after the
# loop and the value-less form would otherwise reach side-effect code (SEED.txt
# write, dir creation, resume mode entered with no checkpoint) before argparse
# rejects the run. Reject up front so wrapper-side state matches DEIM state.
case "$prev" in
    --seed)
        echo "ERROR: trailing '--seed' has no value. Use --seed=<n> or --seed <n>." >&2
        exit 1 ;;
    --output-dir)
        echo "ERROR: trailing '--output-dir' has no value. Use --output-dir=<dir> or --output-dir <dir>." >&2
        exit 1 ;;
    --resume|-r)
        echo "ERROR: trailing '$prev' has no value. Use $prev=<ckpt> or $prev <ckpt>." >&2
        exit 1 ;;
esac

# Validate the (possibly user-overridden) seed BEFORE writing SEED.txt or
# passing it to DEIM. A `--seed=` with empty value or a `--seed` with no
# following value would otherwise produce an invalid SEED.txt that survives
# a subsequent argparse failure.
#
# On resume the env $SEED is intentionally ignored — the resume branch
# (~30 lines below) sources $SEED from the existing run's SEED.txt and
# strips any user `--seed` from the passthrough args. Validating env $SEED
# here would reject an otherwise-valid resume invocation like
# `SEED=abc scripts/train_deim.sh s --resume <ckpt>` whose SEED.txt is fine.
# The resume branch performs its OWN integer check on the SEED.txt value;
# fresh-run validation stays here.
if [[ "$IS_RESUME" != "1" ]]; then
    if ! [[ "$SEED" =~ ^-?[0-9]+$ ]]; then
        echo "ERROR: SEED='$SEED' is not a valid integer (env SEED or --seed argument)." >&2
        exit 1
    fi
fi

# DEIM's `train.py` accepts `--output-dir <dir>` to override the config's
# output_dir. If the user passes it, SEED.txt MUST land in the override
# dir, not the config default — otherwise SEED metadata and run artifacts
# desynchronize, and on resume the seed would be sourced from the wrong
# SEED.txt entirely.
if [[ -n "$OUTPUT_DIR_OVERRIDE" ]]; then
    DEIM_OUTPUT_REL="$OUTPUT_DIR_OVERRIDE"
else
    DEIM_OUTPUT_REL=$(awk '/^output_dir:/ { print $2 }' "$CFG" | tr -d '"')
fi

# DEIM's output_dir is fixed per config (no auto-increment), so we can
# pre-create the dir + write SEED.txt BEFORE training. This survives
# interrupted runs — a marker written only at the end vanishes on crash.
# On resume we still create the dir if needed but DO NOT touch SEED.txt;
# we ALSO source SEED from the existing SEED.txt so the runtime seed
# matches the recorded one.
DEIM_OUTPUT_ABS=""
if [[ -n "$DEIM_OUTPUT_REL" ]]; then
    # Resolve relative to DEIM/ (the cwd) since output_dir is config-relative
    # by DEIM convention. Absolute paths fall through unchanged.
    DEIM_OUTPUT_ABS="$(cd "$PROJECT_ROOT/DEIM" && mkdir -p "$DEIM_OUTPUT_REL" && cd "$DEIM_OUTPUT_REL" && pwd)"
    if [[ "$IS_RESUME" == "1" ]]; then
        if [[ -s "$DEIM_OUTPUT_ABS/SEED.txt" ]]; then
            EXISTING_SEED=$(tr -d '[:space:]' < "$DEIM_OUTPUT_ABS/SEED.txt")
            if [[ "$EXISTING_SEED" =~ ^-?[0-9]+$ ]]; then
                if [[ -n "${SEED_OVERRIDE_ON_RESUME:-}" || "$SEED" != "$EXISTING_SEED" ]]; then
                    echo "resume mode: sourcing seed=$EXISTING_SEED from $DEIM_OUTPUT_ABS/SEED.txt (was env SEED=$SEED)"
                fi
                SEED="$EXISTING_SEED"
            else
                echo "ERROR: $DEIM_OUTPUT_ABS/SEED.txt is non-numeric ('$EXISTING_SEED'); cannot resume safely." >&2
                echo "       Fix or delete the SEED.txt and re-run without --resume to start fresh." >&2
                exit 1
            fi
        else
            echo "WARNING: --resume requested but $DEIM_OUTPUT_ABS/SEED.txt missing or empty." >&2
            echo "         Proceeding with env SEED=$SEED but reproducibility metadata is incomplete." >&2
        fi
    else
        # Refuse fresh launch into a dir that already holds SEED.txt — the
        # user either forgot to pass `-r` (in which case they would clobber
        # the original run's metadata + collide with its artifacts), OR a
        # parallel run is already in flight (file race). Override with
        # FORCE_FRESH=1 if the dir is intentionally being reused.
        #
        # The pre-check + atomic noclobber write together close the
        # concurrent-race window: even if two invocations both pass the
        # pre-check before either writes, only one can win the
        # `set -C; > SEED.txt` create-exclusive (POSIX O_EXCL semantics);
        # the loser fails and bails. FORCE_FRESH=1 explicitly removes the
        # existing SEED.txt before the noclobber write so the operation
        # still succeeds.
        if [[ -e "$DEIM_OUTPUT_ABS/SEED.txt" ]]; then
            if [[ "${FORCE_FRESH:-0}" == "1" ]]; then
                rm -f "$DEIM_OUTPUT_ABS/SEED.txt"
            else
                echo "ERROR: $DEIM_OUTPUT_ABS/SEED.txt already exists." >&2
                echo "       This means either (a) a prior run lives here — pass -r/--resume to continue it," >&2
                echo "       or (b) a parallel run is in flight on the same output dir (race)." >&2
                echo "       To deliberately overwrite: FORCE_FRESH=1 scripts/train_deim.sh ..." >&2
                exit 1
            fi
        fi
        if ! ( set -C; echo "$SEED" > "$DEIM_OUTPUT_ABS/SEED.txt" ) 2>/dev/null; then
            echo "ERROR: failed to create $DEIM_OUTPUT_ABS/SEED.txt atomically." >&2
            echo "       Most likely a parallel fresh launch beat this one to the file." >&2
            echo "       Pass -r/--resume to join the in-flight run, or FORCE_FRESH=1 to override." >&2
            exit 1
        fi
        echo "pre-created $DEIM_OUTPUT_ABS with SEED.txt (seed=$SEED)"
    fi
fi

# Strip user-supplied --seed=* / --seed <n> from "$@" on resume so the
# trailing `--seed="$SEED"` (sourced from SEED.txt) is the only one DEIM
# sees. Without this, a user who passes both `-r` and `--seed=42` would
# see DEIM use 42 (last-wins on argparse) while SEED.txt says something
# else — silent metadata/runtime divergence.
if [[ "$IS_RESUME" == "1" ]]; then
    new_args=()
    skip_next=0
    for arg in "$@"; do
        if (( skip_next )); then skip_next=0; continue; fi
        case "$arg" in
            --seed) skip_next=1 ;;       # next arg is the seed value
            --seed=*) ;;                  # drop equals form
            *) new_args+=("$arg") ;;
        esac
    done
    set -- "${new_args[@]}"
fi

# `exec` replaces the shell, so torchrun's exit code IS the script's exit code
# (no risk of trailing commands masking a failed training).
exec torchrun \
    --master_port="$PORT" \
    --nproc_per_node="$NPROC" \
    train.py -c "$CFG" --use-amp --seed="$SEED" "$@"
