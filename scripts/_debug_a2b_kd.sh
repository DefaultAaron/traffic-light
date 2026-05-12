#!/usr/bin/env bash
# scripts/_debug_a2b_kd.sh
#
# Surfaces the REAL worker stderr from the A2b KD launcher, which is
# normally hidden by:
#   (a) torchrun's "FAILED / <NO_OTHER_FAILURES>" summary that swallows
#       the worker python traceback, and
#   (b) the runner's subprocess.run(capture_output=True) buffering.
#
# What this does:
#   1. cd into DEIM, activate DEIM's standalone venv.
#   2. Print env probe (VIRTUAL_ENV / python / torchrun / torch info).
#   3. Probe the launcher/config state before training.
#   4. Run the launcher under torchrun by default, matching the real A2b path.
#      Set A2B_DEBUG_MODE=single only when you specifically need a native Python
#      traceback before DEIM initializes distributed mode.
#   5. Tee everything to logs/a2b_debug_<UTC-timestamp>.log .
#
# Output dir is ../runs/rehearsal_kd_A2b_deim_s_seed0_DEBUG so this never
# collides with the real A2b output dir. Auto-cleans it on each run.
#
# Usage:
#   scripts/_debug_a2b_kd.sh                         # torchrun mode
#   A2B_DEBUG_MODE=both scripts/_debug_a2b_kd.sh      # single + torchrun
#   TEACHER_CKPT=path/to.pth scripts/_debug_a2b_kd.sh # override
#
# Run from project root.

set -uo pipefail   # NO -e: we WANT to capture failure output, not abort.
                   # pipefail propagates the brace-group's exit code through the
                   # tee pipe so the launcher's non-zero exit isn't masked.

# Exit code contract (for CI / callers / the next reviewer):
#   0   launcher ran to completion (exit 0)
#   2   teacher ckpt not found (pre-flight)
#   3   DEIM/.venv missing (pre-flight)
#   N   launcher's own non-zero exit code (worker crashed)
# A non-zero exit ALWAYS means something failed — the script is never "silent on failure".

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT" || { echo "ERROR: cannot cd to $PROJECT_ROOT" >&2; exit 1; }

mkdir -p logs
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="logs/a2b_debug_${TS}.log"

TEACHER_CKPT="${TEACHER_CKPT:-runs/detect/deim_dfine_m-r1/best_stg2.pth}"
TEACHER_CFG="configs/deim_dfine/deim_hgnetv2_m_traffic_light.yml"
STUDENT_CFG="configs/deim_dfine/deim_hgnetv2_s_traffic_light.yml"
DEBUG_OUT="../runs/rehearsal_kd_A2b_deim_s_seed0_DEBUG"
A2B_DEBUG_MODE="${A2B_DEBUG_MODE:-torchrun}"  # torchrun | single | both
A2B_EPOCHS="${A2B_EPOCHS:-1}"
A2B_PORT="${A2B_PORT:-7778}"

# Absolute teacher path (DEIM CWD differs from project root).
if [ ! -f "$TEACHER_CKPT" ]; then
    echo "ERROR: teacher ckpt not found: $TEACHER_CKPT" | tee "$LOG"
    echo "       Override with TEACHER_CKPT=path/to.pth" | tee -a "$LOG"
    exit 2
fi
TEACHER_CKPT_ABS="$(cd "$(dirname "$TEACHER_CKPT")" && pwd)/$(basename "$TEACHER_CKPT")"

# Clean prior debug output dir so SEED.txt noclobber doesn't block us.
rm -rf "$PROJECT_ROOT/runs/rehearsal_kd_A2b_deim_s_seed0_DEBUG"

{
    echo "=== A2b KD debug run @ $TS ==="
    echo "PROJECT_ROOT=$PROJECT_ROOT"
    echo "LOG=$LOG"
    echo "TEACHER_CKPT_ABS=$TEACHER_CKPT_ABS"
    echo "DEBUG_OUT=$DEBUG_OUT"
    echo "A2B_DEBUG_MODE=$A2B_DEBUG_MODE"
    echo "A2B_EPOCHS=$A2B_EPOCHS"
    echo "A2B_PORT=$A2B_PORT"
    echo

    cd "$PROJECT_ROOT/DEIM" || { echo "ERROR: cannot cd to $PROJECT_ROOT/DEIM" >&2; exit 4; }
    echo "--- cd DEIM, pwd=$(pwd) ---"

    if [ ! -d .venv ]; then
        echo "ERROR: DEIM/.venv missing — run DEIM venv bootstrap first" >&2
        exit 3
    fi
    # shellcheck disable=SC1091
    . .venv/bin/activate

    echo "--- env probe ---"
    echo "VIRTUAL_ENV=${VIRTUAL_ENV:-<unset>}"
    echo "which python: $(command -v python || echo MISSING)"
    echo "which torchrun: $(command -v torchrun || echo MISSING)"
    python - <<'PY' 2>&1 || true
import sys
print(f"python: {sys.executable}")
print(f"version: {sys.version.split()[0]}")
try:
    import torch
    print(f"torch: {torch.__version__}  cuda_built={torch.version.cuda}  cuda_avail={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device0: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"torch import FAILED: {type(e).__name__}: {e}")
try:
    import torchvision
    print(f"torchvision: {torchvision.__version__}")
except Exception as e:
    print(f"torchvision import FAILED: {type(e).__name__}: {e}")
PY
    echo

    echo "--- launcher import probe (no run) ---"
    PYTHONPATH=.. python -c "
import sys
sys.path.insert(0, '.')
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'deim_kd_launch',
        '../components/knowledge_distillation/integration/deim_kd_launch.py',
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print('launcher import: OK')
except Exception as e:
    import traceback
    print(f'launcher import FAILED: {type(e).__name__}: {e}')
    traceback.print_exc()
    " 2>&1 || true
    echo

    echo "--- launcher/config probe (no train) ---"
    PYTHONPATH=.. python - <<PY 2>&1 || true
import sys
from pathlib import Path
sys.path.insert(0, ".")
from engine.core import YAMLConfig

teacher_cfg = "$TEACHER_CFG"
student_cfg = "$STUDENT_CFG"
launcher = Path("../components/knowledge_distillation/integration/deim_kd_launch.py")
text = launcher.read_text()
override_pattern = 'cfg.yaml_cfg["HGNetv2"]["pretrained"] = False'
print(f"launcher path: {launcher.resolve()}")
print(f"launcher has HGNetv2 pretrained override: {override_pattern in text}")
for label, cfg_path in (("teacher", teacher_cfg), ("student", student_cfg)):
    cfg = YAMLConfig(cfg_path)
    before = cfg.yaml_cfg.get("HGNetv2", {}).get("pretrained")
    merged_before = cfg.global_cfg.get("HGNetv2", {}).get("pretrained")
    if label == "teacher" and "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    after = cfg.yaml_cfg.get("HGNetv2", {}).get("pretrained")
    merged_after = cfg.global_cfg.get("HGNetv2", {}).get("pretrained")
    print(
        f"{label} cfg={cfg_path} HGNetv2.pretrained "
        f"yaml_before={before!r} global_before={merged_before!r} "
        f"yaml_after={after!r} global_after={merged_after!r}"
    )
PY
    echo

    run_single() {
        echo "--- launcher run (single-process, no torchrun; diagnostic-only) ---"
        PYTHONUNBUFFERED=1 PYTHONPATH=.. python ../components/knowledge_distillation/integration/deim_kd_launch.py \
            --teacher-cfg "$TEACHER_CFG" \
            --teacher-ckpt "$TEACHER_CKPT_ABS" \
            --kd-lambda 1.0 --ld-lambda 1.0 --kd-temperature 2.0 --kd-reg-max 32 \
            -c "$STUDENT_CFG" \
            --use-amp -u "epoches=$A2B_EPOCHS" \
            --output-dir "$DEBUG_OUT" \
            --seed=0
    }

    run_torchrun() {
        echo "--- launcher run (torchrun nproc=1; real A2b path) ---"
        PYTHONUNBUFFERED=1 TORCH_DISTRIBUTED_DEBUG=DETAIL PYTHONPATH=.. \
        torchrun --master_port="$A2B_PORT" --nproc_per_node=1 \
            ../components/knowledge_distillation/integration/deim_kd_launch.py \
            --teacher-cfg "$TEACHER_CFG" \
            --teacher-ckpt "$TEACHER_CKPT_ABS" \
            --kd-lambda 1.0 --ld-lambda 1.0 --kd-temperature 2.0 --kd-reg-max 32 \
            -c "$STUDENT_CFG" \
            --use-amp -u "epoches=$A2B_EPOCHS" \
            --output-dir "$DEBUG_OUT" \
            --seed=0
    }

    case "$A2B_DEBUG_MODE" in
      single)
        run_single
        RC=$?
        ;;
      torchrun)
        run_torchrun
        RC=$?
        ;;
      both)
        run_single
        SINGLE_RC=$?
        echo
        echo "=== single-process diagnostic exit code: $SINGLE_RC ==="
        echo "=== resetting debug output dir before torchrun ==="
        rm -rf "$PROJECT_ROOT/runs/rehearsal_kd_A2b_deim_s_seed0_DEBUG"
        run_torchrun
        RC=$?
        ;;
      *)
        echo "ERROR: A2B_DEBUG_MODE must be one of: torchrun, single, both (got '$A2B_DEBUG_MODE')" >&2
        RC=5
        ;;
    esac
    echo
    echo "=== launcher exit code: $RC ==="
    # CRITICAL: propagate the launcher's RC as the subshell's exit code so
    # `pipefail` carries it through `tee` and out as the script's exit code.
    # Without this, the trailing `echo` returns 0 and silently masks failure.
    exit "$RC"
} 2>&1 | tee "$PROJECT_ROOT/$LOG"
PIPE_RC=${PIPESTATUS[0]}

echo
echo "Full debug log → $PROJECT_ROOT/$LOG"
if [ "$PIPE_RC" -ne 0 ]; then
    echo "RESULT: FAILED (exit=$PIPE_RC). Inspect the log above — look for the"
    echo "        last 'Traceback', 'Error', 'ModuleNotFoundError', or"
    echo "        'FAILED' before the exit-code marker."
else
    echo "RESULT: OK (launcher completed; debug output dir was $DEBUG_OUT)."
fi
exit "$PIPE_RC"
