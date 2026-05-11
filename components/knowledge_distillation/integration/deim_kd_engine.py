"""KD-augmented `train_one_epoch` for DEIM (cell A2b).

Wraps `DEIM.engine.solver.det_engine.train_one_epoch` with a closure that:

1. Forwards the frozen teacher model on the SAME `samples` batch (no_grad).
2. Computes cls-logit KL (`cls_logit_kl` over `outputs['pred_logits']`).
3. Computes FDR LD KL (`fdr_localization_kl` over `outputs['pred_corners']`).
4. Adds the weighted KD terms to `loss_dict` BEFORE `loss = sum(loss_dict.values())`.
5. Defers to the original epoch loop otherwise.

KD is applied across ALL queries (no Hungarian-matched mask). This is the
simplest rehearsal variant — production A2b can later switch to
matched-positives if signal is weak.

Activation: call `install(teacher, kd_lambda, ld_lambda, kd_temperature, reg_max)`
ONCE before `DEIM/train.py:main()` runs. The launcher
(`deim_kd_launch.py`) handles that.

Hard contract: this module imports `DEIM.engine.solver.det_engine` at
`install()` time only. `py_compile` and standalone imports succeed without
the DEIM venv; runtime use requires DEIM importable.
"""
from __future__ import annotations

import math
import sys
from typing import Iterable

import torch
import torch.amp
import torch.nn as nn

from components.knowledge_distillation.losses import cls_logit_kl, fdr_localization_kl


def freeze_bn_in_train_mode(module: torch.nn.Module) -> None:
    """Set module to .train() while keeping BN running stats frozen.

    Rationale: DEIM's `DFINETransformer.forward` (dfine_decoder.py:754-758)
    gates its output dict on `self.training`. In eval mode the output is
    `{pred_logits, pred_boxes}` — `pred_corners` is STRIPPED, so LD on FDR
    has nothing to distill. Teacher must therefore run in train mode to
    emit `pred_corners`. But teacher's BN running stats MUST stay locked,
    so we walk modules and re-eval() every BatchNorm variant.

    LayerNorm/GroupNorm have no running stats — train vs eval is identical
    for them, so no special handling needed.
    """
    module.train()
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.eval()


def _kd_terms(
    student_out: dict,
    teacher_out: dict,
    kd_lambda: float,
    ld_lambda: float,
    kd_temperature: float,
    reg_max: int,
) -> dict[str, torch.Tensor]:
    """Compute KD terms; returns dict suitable for merge into loss_dict.

    FAIL-LOUD: the A2b rehearsal exists to validate KD on the LD-on-FDR +
    cls-logit KL stack. Silent skip on missing/mismatched keys means the
    rehearsal trains as standard DEIM, defeating its purpose. Both heads
    are required; any absence is a wiring bug — abort with diagnostic.
    """
    terms: dict[str, torch.Tensor] = {}

    s_logits = student_out.get("pred_logits")
    t_logits = teacher_out.get("pred_logits")
    if s_logits is None or t_logits is None:
        raise RuntimeError(
            f"A2b _kd_terms: 'pred_logits' missing from "
            f"{'student' if s_logits is None else 'teacher'} output. "
            "DEIM decoder must run in train mode to emit pred_logits — verify "
            "freeze_bn_in_train_mode(teacher) was called and teacher cfg matches."
        )
    if s_logits.shape != t_logits.shape:
        raise RuntimeError(
            f"A2b _kd_terms: pred_logits shape mismatch "
            f"(student={tuple(s_logits.shape)} vs teacher={tuple(t_logits.shape)}). "
            "Teacher/student must share num_queries + num_classes; check cfg parity."
        )
    terms["kd_cls"] = kd_lambda * cls_logit_kl(
        s_logits, t_logits, temperature=kd_temperature
    )

    s_corners = student_out.get("pred_corners")
    t_corners = teacher_out.get("pred_corners")
    if s_corners is None or t_corners is None:
        raise RuntimeError(
            f"A2b _kd_terms: 'pred_corners' missing from "
            f"{'student' if s_corners is None else 'teacher'} output. "
            "DEIM decoder train-mode branch must emit pred_corners (FDR logits); "
            "verify teacher is in train mode (decoder.training=True) and "
            "decoder.num_denoising=0 (to avoid denoising-target crash)."
        )
    if s_corners.shape != t_corners.shape:
        raise RuntimeError(
            f"A2b _kd_terms: pred_corners shape mismatch "
            f"(student={tuple(s_corners.shape)} vs teacher={tuple(t_corners.shape)}). "
            "Teacher/student must share num_queries + reg_max; check cfg parity."
        )
    terms["kd_ld"] = ld_lambda * fdr_localization_kl(
        s_corners,
        t_corners,
        reg_max=reg_max,
        temperature=kd_temperature,
    )
    return terms


def build_kd_train_one_epoch(
    teacher: torch.nn.Module,
    kd_lambda: float,
    ld_lambda: float,
    kd_temperature: float,
    reg_max: int,
):
    """Return a drop-in replacement for DEIM's `train_one_epoch` with KD injection.

    Body mirrors `DEIM/engine/solver/det_engine.train_one_epoch` so the patched
    function preserves DEIM's AMP, EMA, warmup, max-norm clipping, and metric
    logging behavior. The only inserted lines are the teacher forward and the
    KD-term merge into `loss_dict`.
    """
    # Lazy import — DEIM modules only resolvable inside DEIM venv at runtime.
    from torch.utils.tensorboard import SummaryWriter
    from torch.cuda.amp.grad_scaler import GradScaler

    from DEIM.engine.optim import ModelEMA, Warmup
    from DEIM.engine.misc import MetricLogger, SmoothedValue, dist_utils

    def kd_train_one_epoch(
        self_lr_scheduler,
        lr_scheduler,
        model: torch.nn.Module,
        criterion: torch.nn.Module,
        data_loader: Iterable,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        max_norm: float = 0,
        **kwargs,
    ):
        model.train()
        criterion.train()
        # Teacher in BN-frozen train mode each epoch — DEIM's decoder only emits
        # `pred_corners` (the FDR distribution needed for LD KD) when
        # `self.training=True`. Eval mode strips that key. We MUST keep teacher
        # in train mode but lock BN running stats. See freeze_bn_in_train_mode.
        freeze_bn_in_train_mode(teacher)

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
        header = "Epoch: [{}]".format(epoch)

        print_freq = kwargs.get("print_freq", 10)
        writer: SummaryWriter = kwargs.get("writer", None)
        ema: ModelEMA = kwargs.get("ema", None)
        scaler: GradScaler = kwargs.get("scaler", None)
        lr_warmup_scheduler: Warmup = kwargs.get("lr_warmup_scheduler", None)
        cur_iters = epoch * len(data_loader)

        for i, (samples, targets) in enumerate(
            metric_logger.log_every(data_loader, print_freq, header)
        ):
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            global_step = epoch * len(data_loader) + i
            metas = dict(
                epoch=epoch, step=i, global_step=global_step, epoch_step=len(data_loader)
            )

            if scaler is not None:
                with torch.autocast(device_type=str(device), cache_enabled=True):
                    outputs = model(samples, targets=targets)
                    # Teacher forward inside autocast so dtype matches student outputs.
                    with torch.no_grad():
                        teacher_outputs = teacher(samples)

                if torch.isnan(outputs["pred_boxes"]).any() or torch.isinf(outputs["pred_boxes"]).any():
                    print(outputs["pred_boxes"])
                    state = model.state_dict()
                    new_state = {}
                    for key, value in model.state_dict().items():
                        new_key = key.replace("module.", "")
                        state[new_key] = value
                    new_state["model"] = state
                    dist_utils.save_on_master(new_state, "./NaN.pth")

                with torch.autocast(device_type=str(device), enabled=False):
                    loss_dict = criterion(outputs, targets, **metas)
                    # Promote student/teacher outputs to fp32 for stable KL.
                    kd = _kd_terms(
                        {k: v.float() if torch.is_tensor(v) else v for k, v in outputs.items()},
                        {k: v.float() if torch.is_tensor(v) else v for k, v in teacher_outputs.items()},
                        kd_lambda=kd_lambda,
                        ld_lambda=ld_lambda,
                        kd_temperature=kd_temperature,
                        reg_max=reg_max,
                    )
                    loss_dict.update(kd)

                loss = sum(loss_dict.values())
                scaler.scale(loss).backward()

                if max_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            else:
                outputs = model(samples, targets=targets)
                with torch.no_grad():
                    teacher_outputs = teacher(samples)
                loss_dict = criterion(outputs, targets, **metas)
                kd = _kd_terms(
                    outputs,
                    teacher_outputs,
                    kd_lambda=kd_lambda,
                    ld_lambda=ld_lambda,
                    kd_temperature=kd_temperature,
                    reg_max=reg_max,
                )
                loss_dict.update(kd)

                loss = sum(loss_dict.values())
                optimizer.zero_grad()
                loss.backward()

                if max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

                optimizer.step()

            if ema is not None:
                ema.update(model)

            if self_lr_scheduler:
                optimizer = lr_scheduler.step(cur_iters + i, optimizer)
            else:
                if lr_warmup_scheduler is not None:
                    lr_warmup_scheduler.step()

            loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
            loss_value = sum(loss_dict_reduced.values())

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

            metric_logger.update(loss=loss_value, **loss_dict_reduced)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])

            if writer and dist_utils.is_main_process() and global_step % 10 == 0:
                writer.add_scalar("Loss/total", loss_value.item(), global_step)
                for j, pg in enumerate(optimizer.param_groups):
                    writer.add_scalar(f"Lr/pg_{j}", pg["lr"], global_step)
                for k, v in loss_dict_reduced.items():
                    writer.add_scalar(f"Loss/{k}", v.item(), global_step)

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return kd_train_one_epoch


def install(
    teacher: torch.nn.Module,
    kd_lambda: float = 1.0,
    ld_lambda: float = 1.0,
    kd_temperature: float = 2.0,
    reg_max: int = 32,
) -> list[str]:
    """Monkey-patch DEIM's `train_one_epoch` with the KD-augmented version.

    Patches BOTH the `det_engine` module (definition site) AND `det_solver`
    (which does `from .det_engine import train_one_epoch` at import time —
    a local binding that is NOT updated by patching det_engine alone).

    Also patches BOTH module-name forms — DEIM-prefixed (`DEIM.engine.solver.*`)
    and DEIM-relative (`engine.solver.*`) — because the launcher and DEIM's own
    train.py may import via different path forms depending on sys.path.

    Idempotent: subsequent calls re-patch with the latest closure. Returns the
    list of module names actually patched, for caller-side logging.
    """
    import importlib

    for p in teacher.parameters():
        p.requires_grad_(False)
    # NOT teacher.eval() — DEIM's DFINETransformer.forward (dfine_decoder.py:754)
    # only emits `pred_corners` when self.training=True. The kd_train_one_epoch
    # closure calls freeze_bn_in_train_mode(teacher) each batch.
    freeze_bn_in_train_mode(teacher)
    # Defensive: disable teacher's denoising arm. DEIM's decoder gates denoising
    # on `self.training and self.num_denoising > 0` (dfine_decoder.py:710); the
    # KD path calls `teacher(samples)` with no targets → denoising path crashes.
    # Setting num_denoising=0 makes the branch a no-op, leaving pred_logits +
    # pred_corners (the keys KD consumes) intact. Idempotent across calls.
    if hasattr(teacher, "decoder") and hasattr(teacher.decoder, "num_denoising"):
        teacher.decoder.num_denoising = 0

    closure = build_kd_train_one_epoch(
        teacher=teacher,
        kd_lambda=kd_lambda,
        ld_lambda=ld_lambda,
        kd_temperature=kd_temperature,
        reg_max=reg_max,
    )

    patched: list[str] = []
    for modname in (
        "DEIM.engine.solver.det_engine",
        "DEIM.engine.solver.det_solver",
        "engine.solver.det_engine",
        "engine.solver.det_solver",
    ):
        try:
            mod = importlib.import_module(modname)
        except ImportError:
            continue
        if hasattr(mod, "train_one_epoch"):
            mod.train_one_epoch = closure
            patched.append(modname)

    if not patched:
        raise RuntimeError(
            "deim_kd_engine.install: could not import any DEIM solver module to patch. "
            "Ensure CWD=DEIM/ or PYTHONPATH includes project root so DEIM imports resolve."
        )
    return patched
