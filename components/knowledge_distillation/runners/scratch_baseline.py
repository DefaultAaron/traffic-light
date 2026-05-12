"""KD cell A1 — scratch baseline + wall-clock anchor.

Role: control + wall-clock anchor for KD-acceptance §六 gate #1 / #4.
`T_scratch_A1` is the reference; A2+ runs must finish under 2.0 × T_scratch_A1.
A1 does NOT pass §六 #1 against itself — it IS the reference.

Pre-R2 rehearsal usage (`docs/_archive/pre_r2_kickoff_checklist.md (2026-05-12 归档)` §2.5 row "KD A1 wall-clock"):
    Measures 1-epoch wall-clock on R1 data for both deployees (YOLO26-s and
    DEIM-D-FINE-S) so the §六 #1 budget cap is calibrated before R2 in-round.

Run (rehearsal on R1):
    # YOLO26-s, dry-run (print command + pending JSON, no training):
    uv run python -m components.knowledge_distillation.runners.scratch_baseline \\
        --family yolo --size s --rehearsal-on-r1 --epochs 1 --dry-run

    # DEIM-D-FINE-S, dry-run:
    uv run python -m components.knowledge_distillation.runners.scratch_baseline \\
        --family deim --size s --rehearsal-on-r1 --epochs 1 --dry-run

    # Execute (on GPU server with appropriate venv):
    uv run python -m components.knowledge_distillation.runners.scratch_baseline \\
        --family yolo --size s --rehearsal-on-r1 --epochs 1 --execute

Output: `runs/rehearsal_kd_A1_walltime_estimate.json` (rehearsal_kind = r1_data).
Both families merge into the same JSON across separate invocations.

Resume: `--resume` not implemented for the rehearsal — re-run the family.
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
DEFAULT_REHEARSAL_OUTPUT = "runs/rehearsal_kd_A1_walltime_estimate.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_command(family: str, size: str, epochs: int, imgsz: int | None, seed: int) -> list[str]:
    """Construct the training command for the chosen family.

    Returns argv list; subprocess.run uses shell=False.
    """
    if family == "yolo":
        # Dispatch via main.py CLI — keeps SEED.txt + args.yaml + --epochs override path canonical.
        # main.py: `model` is a POSITIONAL arg (main.py:333 `add_argument("model", nargs="?", ...)`),
        # not `--model`. The positional goes BEFORE any --flag so argparse binds it correctly.
        cmd = [
            "uv", "run", "python", "main.py", "train",
            f"yolo26{size}",
            "--epochs", str(epochs),
            "--seed", str(seed),
        ]
        if imgsz is not None:
            cmd += ["--imgsz", str(imgsz)]
        return cmd
    if family == "deim":
        # scripts/train_deim.sh + DEIM `-u epoches=N` config override (DEIM uses
        # "epoches" sic in its YAML schema — DEIM/configs/base/optimizer.yml:9).
        # --output-dir override is REQUIRED for the rehearsal: DEIM configs
        # default output_dir to a stable name (e.g. `../runs/deim_dfine_s_r2`),
        # and train_deim.sh refuses fresh launch when SEED.txt already exists
        # in that dir (collision with prior runs). Route rehearsal artifacts to
        # a dedicated `runs/rehearsal_kd_A1_deim_<size>_seed<seed>/` path.
        # NOTE: path is DEIM-CWD-relative (`../runs/...` resolves to
        # project_root/runs/...).
        cmd = [
            "bash", "scripts/train_deim.sh", size,
            "-u", f"epoches={epochs}",
            "--seed", str(seed),
            "--output-dir", f"../runs/rehearsal_kd_A1_deim_{size}_seed{seed}",
        ]
        return cmd
    raise ValueError(f"unsupported family: {family!r}")


def load_existing(output: Path) -> dict:
    if not output.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "rehearsal_kind": "r1_data",
            "rehearsal_name": "kd_A1_walltime_estimate",
            "results": {},
        }
    return json.loads(output.read_text())


def merge_and_save(output: Path, family_key_str: str, entry: dict) -> None:
    record = load_existing(output)
    record["results"][family_key_str] = entry
    record["last_updated_utc"] = _now_iso()
    record["merged_finalized"] = (
        "yolo26s" in record["results"]
        and "deim_dfine_s" in record["results"]
        and record["results"]["yolo26s"].get("status") == "completed"
        and record["results"]["deim_dfine_s"].get("status") == "completed"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")


def family_key(family: str, size: str) -> str:
    if family == "yolo":
        return f"yolo26{size}"
    if family == "deim":
        return f"deim_dfine_{size}"
    raise ValueError(family)


def main() -> int:
    ap = argparse.ArgumentParser(description="KD A1 scratch baseline + wall-clock anchor")
    ap.add_argument("--family", required=True, choices=["yolo", "deim"])
    ap.add_argument("--size", default="s", choices=["n", "s", "m", "l"])
    ap.add_argument("--epochs", type=int, default=1,
                    help="rehearsal default 1; in-round value comes from config")
    ap.add_argument("--imgsz", type=int, default=None, help="optional override; YOLO only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rehearsal-on-r1", action="store_true",
                    help="pre-R2 rehearsal flag; forces rehearsal_kind=r1_data + rehearsal_ output prefix")
    ap.add_argument("--output", default=None,
                    help=f"default {DEFAULT_REHEARSAL_OUTPUT} when --rehearsal-on-r1; required otherwise")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dry-run", action="store_true",
                     help="print command + write pending JSON; do not execute training")
    grp.add_argument("--execute", action="store_true",
                     help="run training subprocess and record wall-clock")
    args = ap.parse_args()

    if args.output is None:
        if not args.rehearsal_on_r1:
            ap.error("--output is required unless --rehearsal-on-r1 is set")
        args.output = DEFAULT_REHEARSAL_OUTPUT
    output = Path(args.output)

    if args.epochs < 1:
        ap.error(
            f"--epochs must be >= 1 for A1 wall-clock anchor (got {args.epochs}). "
            "Zero epochs measures no training time — the anchor would be useless "
            "and downstream §六 #1 gate (T_A2+ < 2.0 × T_scratch_A1) would divide "
            "by ~0."
        )

    if args.rehearsal_on_r1 and not output.name.startswith("rehearsal_"):
        ap.error(f"--rehearsal-on-r1 requires output filename to start with 'rehearsal_'; got {output.name}")

    cmd = build_command(args.family, args.size, args.epochs, args.imgsz, args.seed)
    fkey = family_key(args.family, args.size)
    cmd_str = shlex.join(cmd)

    if args.dry_run:
        entry = {
            "status": "pending",
            "family": args.family,
            "size": args.size,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "seed": args.seed,
            "train_command": cmd_str,
            "wall_clock_seconds": None,
            "exit_code": None,
            "run_started_utc": None,
            "run_finished_utc": None,
            "registered_utc": _now_iso(),
        }
        merge_and_save(output, fkey, entry)
        print(f"[dry-run] {fkey}: registered pending entry → {output}")
        print(f"[dry-run] command:\n  {cmd_str}")
        print("[dry-run] re-run with --execute on the appropriate training-rig env to record wall-clock.")
        return 0

    started = _now_iso()
    t0 = time.monotonic()
    proc = subprocess.run(cmd, check=False)
    t1 = time.monotonic()
    finished = _now_iso()
    wall = t1 - t0

    entry = {
        "status": "completed" if proc.returncode == 0 else "failed",
        "family": args.family,
        "size": args.size,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "seed": args.seed,
        "train_command": cmd_str,
        "wall_clock_seconds": wall,
        "exit_code": proc.returncode,
        "run_started_utc": started,
        "run_finished_utc": finished,
    }
    merge_and_save(output, fkey, entry)
    print(f"{fkey}: {entry['status']} in {wall:.2f}s (exit={proc.returncode}) → {output}")
    return 0 if proc.returncode == 0 else proc.returncode


if __name__ == "__main__":
    sys.exit(main())
