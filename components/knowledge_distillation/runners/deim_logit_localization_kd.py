"""KD cell A2b — DEIM-D-FINE-S student ← DEIM-D-FINE-M teacher.

Method stack: cls-logit KL + LD on FDR (per v1.3 §5.5, joint signal — A2b - A1
isolates the JOINT contribution; LD-only isolation is out of scope).

Why DEIM-specific: DEIM retains FDR (`reg_max=32` → 33-bin distribution per
side per query), so external LD applies structurally; YOLO26 A2a is cls-only.

Run (rehearsal on R1):
    # Dry-run (print dispatch + register pending JSON):
    uv run python -m components.knowledge_distillation.runners.deim_logit_localization_kd \\
        --teacher-cfg configs/deim_dfine/deim_hgnetv2_m_traffic_light.yml \\
        --teacher-ckpt runs/detect/deim_dfine_m-r1/best_stg2.pth \\
        --student-cfg configs/deim_dfine/deim_hgnetv2_s_traffic_light.yml \\
        --rehearsal-on-r1 --epochs 1 --dry-run

    # Execute (training server, DEIM venv):
    cd <project_root>
    uv run python -m components.knowledge_distillation.runners.deim_logit_localization_kd \\
        --teacher-cfg configs/deim_dfine/deim_hgnetv2_m_traffic_light.yml \\
        --teacher-ckpt runs/detect/deim_dfine_m-r1/best_stg2.pth \\
        --student-cfg configs/deim_dfine/deim_hgnetv2_s_traffic_light.yml \\
        --rehearsal-on-r1 --epochs 1 --execute

Output: `runs/rehearsal_kd_A2b_R1.json` (rehearsal_kind=r1_data).
Spec: docs/planning/additional_components_plan.md §七 row A2b + §5.5.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "1.0"
DEFAULT_REHEARSAL_OUTPUT = "runs/rehearsal_kd_A2b_R1.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_existing(output: Path) -> dict:
    if not output.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "rehearsal_kind": "r1_data",
            "rehearsal_name": "kd_A2b_R1",
            "cell": "A2b",
            "method_stack": "LD_on_FDR + cls_logit_KL",
            "family": "deim",
            "results": {},
        }
    return json.loads(output.read_text())


def _save_entry(output: Path, key: str, entry: dict) -> None:
    record = _load_existing(output)
    record["results"][key] = entry
    record["last_updated_utc"] = _now_iso()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")


def _to_deim_relative(cfg_path: str) -> str:
    """Normalize a config path to DEIM-relative.

    Accepts either `DEIM/configs/...` (project-root-relative, matches user's
    repo navigation) or `configs/...` (already DEIM-relative). Strips a leading
    `DEIM/` so the launcher (which runs with CWD=DEIM/) finds the file.

    Absolute paths are rejected — DEIM resolves config-relative includes from
    its own CWD, so absolute paths break recursive `__include__` resolution
    inside DEIM YAMLs.
    """
    if Path(cfg_path).is_absolute():
        raise ValueError(
            f"cfg path must be DEIM/-relative (e.g. 'configs/...') or "
            f"project-root-relative (e.g. 'DEIM/configs/...'); got absolute {cfg_path!r}"
        )
    norm = cfg_path.lstrip("./")
    if norm.startswith("DEIM/"):
        norm = norm[len("DEIM/"):]
    return norm


def build_dispatch(
    *,
    teacher_cfg: str,
    teacher_ckpt: str,
    student_cfg: str,
    epochs: int,
    seed: int,
    kd_lambda: float,
    ld_lambda: float,
    kd_temperature: float,
    kd_reg_max: int,
    nproc: int,
    port: int,
) -> tuple[list[str], str]:
    """Construct the torchrun argv + a posix-shell-quoted preview.

    Layout (must keep CWD=DEIM at exec time so DEIM imports resolve):
        cd DEIM && PYTHONPATH=.. torchrun --master_port=<port> --nproc_per_node=<nproc> \\
            ../components/knowledge_distillation/integration/deim_kd_launch.py \\
            --teacher-cfg <DEIM-rel cfg> --teacher-ckpt <abs_ckpt> \\
            --kd-lambda --ld-lambda --kd-temperature --kd-reg-max \\
            -c <DEIM-rel student cfg> --use-amp -u epoches=<epochs> --seed=<seed>
    """
    teacher_ckpt_abs = str(Path(teacher_ckpt).resolve())
    teacher_cfg_rel = _to_deim_relative(teacher_cfg)
    student_cfg_rel = _to_deim_relative(student_cfg)
    # The launcher script lives outside DEIM/. From DEIM/ cwd, use `../...`.
    launcher_rel = "../components/knowledge_distillation/integration/deim_kd_launch.py"
    # Dedicated rehearsal output dir avoids collision with R1/R2 production runs.
    # Path is DEIM-CWD-relative ("../runs/..." resolves to project_root/runs/...).
    output_dir_rel = f"../runs/rehearsal_kd_A2b_deim_s_seed{seed}"
    argv = [
        "bash", "-c",
        " ".join([
            "cd DEIM &&",
            "PYTHONPATH=..",
            "torchrun", f"--master_port={port}", f"--nproc_per_node={nproc}",
            launcher_rel,
            "--teacher-cfg", shlex.quote(teacher_cfg_rel),
            "--teacher-ckpt", shlex.quote(teacher_ckpt_abs),
            "--kd-lambda", str(kd_lambda),
            "--ld-lambda", str(ld_lambda),
            "--kd-temperature", str(kd_temperature),
            "--kd-reg-max", str(kd_reg_max),
            "-c", shlex.quote(student_cfg_rel),
            "--use-amp",
            "-u", f"epoches={epochs}",
            "--output-dir", shlex.quote(output_dir_rel),
            f"--seed={seed}",
        ]),
    ]
    preview = argv[-1]
    return argv, preview


def main() -> int:
    ap = argparse.ArgumentParser(description="KD A2b DEIM-D-FINE-S ← DEIM-D-FINE-M (LD on FDR + cls-logit KL)")
    ap.add_argument("--teacher-cfg", required=True,
                    help="DEIM YAML for teacher (e.g. configs/deim_dfine/deim_hgnetv2_m_traffic_light.yml)")
    ap.add_argument("--teacher-ckpt", required=True,
                    help="Path to teacher .pth (best_stg2.pth typically)")
    ap.add_argument("--student-cfg", required=True,
                    help="DEIM YAML for student (e.g. .../deim_hgnetv2_s_traffic_light.yml)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kd-lambda", type=float, default=1.0,
                    help="cls-logit KL weight (default 1.0; MUST be > 0 for A2b)")
    ap.add_argument("--ld-lambda", type=float, default=1.0,
                    help="LD-on-FDR KL weight (default 1.0; MUST be > 0 for A2b)")
    ap.add_argument("--kd-temperature", type=float, default=2.0)
    ap.add_argument("--kd-reg-max", type=int, default=32)
    ap.add_argument("--nproc", type=int, default=1, help="torchrun --nproc_per_node")
    ap.add_argument("--port", type=int, default=7777, help="torchrun --master_port")
    ap.add_argument("--rehearsal-on-r1", action="store_true")
    ap.add_argument("--output", default=None,
                    help=f"default {DEFAULT_REHEARSAL_OUTPUT} when --rehearsal-on-r1; required otherwise")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dry-run", action="store_true")
    grp.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    if args.kd_lambda <= 0 or args.ld_lambda <= 0:
        ap.error(
            f"--kd-lambda AND --ld-lambda must both be > 0 for A2b KD rehearsal "
            f"(got kd_lambda={args.kd_lambda}, ld_lambda={args.ld_lambda}). "
            "Zero / negative weight = silent no-op on that head — the rehearsal "
            "would complete without applying that KD signal. The A2b cell is "
            "joint LD-on-FDR + cls-logit KL by spec (§七 v2); single-head KD "
            "needs its own cell. Use deim_baseline_golsd_off runner for the "
            "no-external-KD ablation (A0)."
        )

    if args.output is None:
        if not args.rehearsal_on_r1:
            ap.error("--output is required unless --rehearsal-on-r1 is set")
        args.output = DEFAULT_REHEARSAL_OUTPUT
    output = Path(args.output)
    if args.rehearsal_on_r1 and not output.name.startswith("rehearsal_"):
        ap.error(f"--rehearsal-on-r1 requires output filename to start with 'rehearsal_'; got {output.name}")

    if args.execute:
        if not Path(args.teacher_ckpt).exists():
            ap.error(f"teacher checkpoint not found: {args.teacher_ckpt}")
        if not Path("DEIM").is_dir():
            ap.error("DEIM/ vendored repo not found at project root (required for --execute)")
        # Cfg paths accept either DEIM-relative (resolved from inside DEIM/) or
        # project-root-relative (`DEIM/...`); existence check tries both forms.
        for label, cfg in (("teacher", args.teacher_cfg), ("student", args.student_cfg)):
            candidates = [Path(cfg)]
            if not cfg.startswith("DEIM/"):
                candidates.append(Path("DEIM") / cfg)
            if not any(c.exists() for c in candidates):
                tried = ", ".join(str(c) for c in candidates)
                ap.error(f"{label} cfg not found in: {tried}")

    argv, preview = build_dispatch(
        teacher_cfg=args.teacher_cfg,
        teacher_ckpt=args.teacher_ckpt,
        student_cfg=args.student_cfg,
        epochs=args.epochs,
        seed=args.seed,
        kd_lambda=args.kd_lambda,
        ld_lambda=args.ld_lambda,
        kd_temperature=args.kd_temperature,
        kd_reg_max=args.kd_reg_max,
        nproc=args.nproc,
        port=args.port,
    )

    key = f"deim_dfine_s_seed{args.seed}"

    if args.dry_run:
        entry = {
            "status": "pending",
            "cell": "A2b",
            "teacher_cfg": args.teacher_cfg,
            "teacher_ckpt": args.teacher_ckpt,
            "student_cfg": args.student_cfg,
            "epochs": args.epochs,
            "seed": args.seed,
            "kd_lambda": args.kd_lambda,
            "ld_lambda": args.ld_lambda,
            "kd_temperature": args.kd_temperature,
            "kd_reg_max": args.kd_reg_max,
            "nproc": args.nproc,
            "port": args.port,
            "train_command": preview,
            "wall_clock_seconds": None,
            "exit_code": None,
            "run_started_utc": None,
            "run_finished_utc": None,
            "registered_utc": _now_iso(),
        }
        _save_entry(output, key, entry)
        print(f"[dry-run] A2b: registered pending entry → {output}")
        print(f"[dry-run] command:\n  {preview}")
        print("[dry-run] re-run with --execute on the training server (DEIM venv) to record wall-clock.")
        return 0

    started = _now_iso()
    t0 = time.monotonic()
    proc = subprocess.run(argv, check=False)
    t1 = time.monotonic()
    finished = _now_iso()
    wall = t1 - t0

    entry = {
        "status": "completed" if proc.returncode == 0 else "failed",
        "cell": "A2b",
        "teacher_cfg": args.teacher_cfg,
        "teacher_ckpt": args.teacher_ckpt,
        "student_cfg": args.student_cfg,
        "epochs": args.epochs,
        "seed": args.seed,
        "kd_lambda": args.kd_lambda,
        "ld_lambda": args.ld_lambda,
        "kd_temperature": args.kd_temperature,
        "kd_reg_max": args.kd_reg_max,
        "nproc": args.nproc,
        "port": args.port,
        "train_command": preview,
        "wall_clock_seconds": wall,
        "exit_code": proc.returncode,
        "run_started_utc": started,
        "run_finished_utc": finished,
    }
    _save_entry(output, key, entry)
    print(f"A2b {key}: {entry['status']} in {wall:.2f}s (exit={proc.returncode}) → {output}")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
