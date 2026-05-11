"""KD cell A2a — YOLO26-s student ← YOLO26-m teacher, cls-logit KL.

YOLO26 has `reg_max=1` (no DFL distribution head), so localization-distillation
(LD) does not apply on the YOLO path; A2a is cls-logit KL only. DEIM students go
through A2b instead.

Integration: a Ultralytics `DetectionTrainer` subclass (`KDDetectionTrainer`)
that monkey-patches the wrapped student model's `.loss(batch, preds)` method
in `_setup_train` to add a cls-logit KL term against a frozen teacher.

Run (rehearsal on R1):
    # Dry-run (print command + register pending JSON):
    uv run python -m components.knowledge_distillation.runners.yolo_logit_kd \\
        --teacher-ckpt runs/detect/yolo26m-r1/weights/best.pt \\
        --student-init scratch --rehearsal-on-r1 --epochs 1 --dry-run

    # Execute (run KD training in-process):
    uv run python -m components.knowledge_distillation.runners.yolo_logit_kd \\
        --teacher-ckpt runs/detect/yolo26m-r1/weights/best.pt \\
        --student-init scratch --rehearsal-on-r1 --epochs 1 --execute

Output:
    `runs/rehearsal_kd_A2a_R1.json` — rehearsal_kind=r1_data, status pending or completed.

Spec: `docs/planning/additional_components_plan.md` §七 row A2a.
"""
from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "1.0"
DEFAULT_REHEARSAL_OUTPUT = "runs/rehearsal_kd_A2a_R1.json"
DEFAULT_DATA_YAML = "data/traffic_light.yaml"

# Student init → model architecture or ckpt for cold-start.
# "scratch" MUST resolve to an Ultralytics arch YAML (e.g. yolo26s.yaml resolvable
# under the ultralytics package cfg dir), NOT a project-local training-config YAML
# (configs/yolo26s.yaml is {model, data, epochs, ...} — no backbone/head, would
# crash on YOLO(...) instantiation).
STUDENT_INIT_TO_MODEL = {
    "scratch": "yolo26s.yaml",
    "coco": "yolo26s.pt",
    "r2_baseline": "runs/detect/yolo26s/weights/best.pt",
    "r1_rehearsal": "runs/detect/yolo26s-r1/weights/best.pt",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load_existing(output: Path) -> dict:
    if not output.exists():
        return {
            "schema_version": SCHEMA_VERSION,
            "rehearsal_kind": "r1_data",
            "rehearsal_name": "kd_A2a_R1",
            "cell": "A2a",
            "method_stack": "cls_logit_KL",
            "family": "yolo",
            "results": {},
        }
    return json.loads(output.read_text())


def _save_entry(output: Path, key: str, entry: dict) -> None:
    record = _load_existing(output)
    record["results"][key] = entry
    record["last_updated_utc"] = _now_iso()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n")


def freeze_bn_in_train_mode(module) -> None:
    """Set module to .train() (so train-mode output branches fire) but lock BN.

    YOLO26 `Detect.forward` returns the dict `{'one2many', 'one2one'}` when
    `self.training == True`; in eval mode it instead runs `_inference()` and
    returns a single concatenated tensor that has no logit access path.
    We need the train-mode dict for KD, so the teacher runs in `.train()`,
    but BN running stats MUST NOT update over the rehearsal epoch.

    Lazy-imports torch.nn so this module is import-safe pre-uv-env.
    """
    import torch.nn as nn
    module.train()
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()


def extract_one2many_scores(preds_dict):
    """Return cls-logit tensor (B, num_anchors, nc) from YOLO26 train-mode preds.

    Train-mode `Detect.forward` yields `{'one2many': {'boxes', 'scores', 'feats'},
    'one2one': {...}}` when end2end=True. We distill the `one2many` head (the
    loss-bearing branch); `'scores'` is shape `(B, nc, num_anchors)`. Returns
    `None` on any structure mismatch so the caller falls back to base-loss-only.
    """
    if not isinstance(preds_dict, dict):
        return None
    branch = preds_dict.get("one2many")
    if not isinstance(branch, dict):
        # Non-end2end YOLO returns the forward_head dict directly.
        branch = preds_dict
    scores = branch.get("scores")
    if scores is None or scores.dim() != 3:
        return None
    # (B, nc, A) → (B, A, nc) for cls_logit_kl
    return scores.permute(0, 2, 1).contiguous()


def build_kd_trainer_class(teacher_ckpt: str, kd_lambda: float, kd_temperature: float):
    """Build a fresh KDDetectionTrainer subclass closed over KD hyperparams.

    Imports Ultralytics + torch lazily so dry-run / unit-test paths don't pay
    the import cost.
    """
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model

    from components.knowledge_distillation.losses import cls_logit_kl

    class KDDetectionTrainer(DetectionTrainer):
        _teacher_ckpt = teacher_ckpt
        _kd_lambda = kd_lambda
        _kd_temperature = kd_temperature

        def _setup_train(self):
            super()._setup_train()
            # SEED.txt at training START (CLAUDE.md "Reproducibility plumbing" —
            # survives interrupted runs). Parent's _setup_train already created
            # save_dir; mirror main.py's _register_seed_marker contract here.
            try:
                save_dir = Path(self.save_dir)
                save_dir.mkdir(parents=True, exist_ok=True)
                seed_path = save_dir / "SEED.txt"
                if not seed_path.exists():
                    seed_path.write_text(f"{int(self.args.seed)}\n")
            except (OSError, ValueError, AttributeError):
                # Don't crash training on a metadata-write failure; SEED.txt
                # absence is a known minor reproducibility hole, not a blocker.
                pass

            teacher_yolo = YOLO(self._teacher_ckpt)
            teacher_module = teacher_yolo.model.to(self.device)
            for p in teacher_module.parameters():
                p.requires_grad_(False)
            freeze_bn_in_train_mode(teacher_module)
            teacher_module._kd_role = "teacher"

            student = unwrap_model(self.model)
            original_loss = student.loss
            kd_lambda_ = float(self._kd_lambda)
            kd_T = float(self._kd_temperature)

            def kd_loss(batch, preds=None):
                if preds is None:
                    preds = student.forward(batch["img"])
                base_loss, loss_items = original_loss(batch, preds)

                # FAIL-LOUD: the A2a rehearsal exists to validate KD; silent
                # degrade to base-loss-only defeats the point. Any of these
                # conditions means the wiring is broken — abort, don't pretend.
                s_scores = extract_one2many_scores(preds)
                if s_scores is None:
                    raise RuntimeError(
                        f"A2a kd_loss: cannot extract one2many.scores from student preds "
                        f"(preds type={type(preds).__name__}). YOLO26 must run with "
                        "end2end=True (default). Check scratch arch YAML."
                    )

                # Teacher MUST stay in BN-frozen train mode each batch — the
                # parent trainer's training loop calls .train() on student each
                # epoch, but the teacher is independent and we keep it locked.
                freeze_bn_in_train_mode(teacher_module)
                with torch.no_grad():
                    teacher_preds = teacher_module(batch["img"])
                t_scores = extract_one2many_scores(teacher_preds)
                if t_scores is None:
                    raise RuntimeError(
                        f"A2a kd_loss: cannot extract one2many.scores from teacher preds "
                        f"(type={type(teacher_preds).__name__}). Teacher ckpt {teacher_ckpt!r} "
                        "may be from a non-end2end build; verify with `head._end2end` probe."
                    )
                if t_scores.shape != s_scores.shape:
                    raise RuntimeError(
                        f"A2a kd_loss: teacher/student score shape mismatch "
                        f"(student={tuple(s_scores.shape)} vs teacher={tuple(t_scores.shape)}). "
                        "Teacher and student must share nc + anchor grid; verify both YAMLs "
                        "have identical nc and stride."
                    )

                # fp32 promotion for stable KL under autocast.
                kd = cls_logit_kl(
                    s_scores.float(), t_scores.float(), temperature=kd_T
                )
                total = base_loss + kd_lambda_ * kd
                return total, loss_items

            student.loss = kd_loss

    return KDDetectionTrainer


def execute_in_process(args) -> dict:
    """Run the KD training in-process and return the wall-clock entry."""
    from ultralytics import YOLO

    started = _now_iso()
    t0 = time.monotonic()
    error_msg = None
    rc = 0
    try:
        student_model_arg = STUDENT_INIT_TO_MODEL[args.student_init]
        model = YOLO(student_model_arg)
        trainer_cls = build_kd_trainer_class(
            teacher_ckpt=args.teacher_ckpt,
            kd_lambda=args.kd_lambda,
            kd_temperature=args.kd_temperature,
        )
        train_kwargs = dict(
            data=args.data,
            epochs=args.epochs,
            seed=args.seed,
            project="runs/detect",
            name=f"rehearsal_kd_A2a_yolo26s_R1_seed{args.seed}",
        )
        if args.imgsz is not None:
            train_kwargs["imgsz"] = args.imgsz
        if args.device is not None:
            train_kwargs["device"] = args.device
        if args.batch is not None:
            train_kwargs["batch"] = args.batch
        model.train(trainer=trainer_cls, **train_kwargs)
    except Exception as exc:
        rc = 1
        error_msg = f"{type(exc).__name__}: {exc}"

    t1 = time.monotonic()
    return {
        "status": "completed" if rc == 0 else "failed",
        "wall_clock_seconds": t1 - t0,
        "exit_code": rc,
        "error": error_msg,
        "run_started_utc": started,
        "run_finished_utc": _now_iso(),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="KD A2a YOLO26-s ← YOLO26-m cls-logit KL")
    ap.add_argument("--teacher-ckpt", required=True,
                    help="Path to YOLO26-m teacher checkpoint (.pt)")
    ap.add_argument("--student-init", default="scratch",
                    choices=list(STUDENT_INIT_TO_MODEL.keys()))
    ap.add_argument("--data", default=DEFAULT_DATA_YAML,
                    help=f"Ultralytics data YAML (default: {DEFAULT_DATA_YAML})")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kd-lambda", type=float, default=1.0,
                    help="cls-logit KL loss weight (default 1.0; MUST be > 0 — "
                         "zero-weight KD = silent no-op, defeats A2a rehearsal's "
                         "purpose of validating KD wiring. Use scratch_baseline "
                         "runner for the kd_lambda=0 ablation.)")
    ap.add_argument("--kd-temperature", type=float, default=2.0,
                    help="softmax temperature for KL (default 2.0)")
    ap.add_argument("--rehearsal-on-r1", action="store_true",
                    help="pre-R2 rehearsal flag; forces rehearsal_ prefix on output")
    ap.add_argument("--output", default=None,
                    help=f"default {DEFAULT_REHEARSAL_OUTPUT} when --rehearsal-on-r1; required otherwise")
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--dry-run", action="store_true")
    grp.add_argument("--execute", action="store_true")
    args = ap.parse_args()

    if args.kd_lambda <= 0:
        ap.error(
            f"--kd-lambda must be > 0 for A2a KD rehearsal (got {args.kd_lambda}). "
            "Zero / negative weight = silent no-op = the rehearsal would complete "
            "without ever applying KD signal, defeating its purpose. Use "
            "scratch_baseline runner for the no-KD ablation cell (A1)."
        )

    if args.output is None:
        if not args.rehearsal_on_r1:
            ap.error("--output is required unless --rehearsal-on-r1 is set")
        args.output = DEFAULT_REHEARSAL_OUTPUT
    output = Path(args.output)
    if args.rehearsal_on_r1 and not output.name.startswith("rehearsal_"):
        ap.error(f"--rehearsal-on-r1 requires output filename to start with 'rehearsal_'; got {output.name}")

    if not Path(args.teacher_ckpt).exists():
        # Dry-run still permits non-existent ckpt — useful for command preview.
        if args.execute:
            ap.error(f"teacher checkpoint not found: {args.teacher_ckpt}")

    cmd_preview = shlex.join(sys.argv)

    if args.dry_run:
        entry = {
            "status": "pending",
            "cell": "A2a",
            "student_init": args.student_init,
            "teacher_ckpt": args.teacher_ckpt,
            "data": args.data,
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "device": args.device,
            "seed": args.seed,
            "kd_lambda": args.kd_lambda,
            "kd_temperature": args.kd_temperature,
            "invocation": cmd_preview,
            "wall_clock_seconds": None,
            "registered_utc": _now_iso(),
        }
        _save_entry(output, f"yolo26s_seed{args.seed}", entry)
        print(f"[dry-run] A2a: registered pending entry → {output}")
        print(f"[dry-run] re-run with --execute to actually run the KD training.")
        return 0

    result = execute_in_process(args)
    entry = {
        "status": result["status"],
        "cell": "A2a",
        "student_init": args.student_init,
        "teacher_ckpt": args.teacher_ckpt,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "seed": args.seed,
        "kd_lambda": args.kd_lambda,
        "kd_temperature": args.kd_temperature,
        "invocation": cmd_preview,
        "wall_clock_seconds": result["wall_clock_seconds"],
        "exit_code": result["exit_code"],
        "error": result["error"],
        "run_started_utc": result["run_started_utc"],
        "run_finished_utc": result["run_finished_utc"],
    }
    _save_entry(output, f"yolo26s_seed{args.seed}", entry)
    print(f"A2a yolo26s_seed{args.seed}: {entry['status']} in {result['wall_clock_seconds']:.2f}s → {output}")
    return result["exit_code"]


if __name__ == "__main__":
    sys.exit(main())
