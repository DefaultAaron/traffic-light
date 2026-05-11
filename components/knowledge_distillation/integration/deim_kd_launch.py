"""DEIM KD launcher (cell A2b) — torchrun entrypoint.

Activates the KD train_one_epoch patch then defers to DEIM's normal training
driver. Designed to be invoked exactly like DEIM/train.py but with extra
`--teacher-cfg` / `--teacher-ckpt` / `--kd-*` flags consumed BEFORE delegation.

Invocation (training server, DEIM venv, SINGLE-NODE only):
    cd <project_root>/DEIM   # DEIM's train.py expects CWD = DEIM/
    PYTHONPATH=.. torchrun --nproc_per_node=1 \\
        ../components/knowledge_distillation/integration/deim_kd_launch.py \\
        --teacher-cfg configs/deim_dfine/deim_hgnetv2_m_traffic_light.yml \\
        --teacher-ckpt ../runs/detect/deim_dfine_m-r1/best_stg2.pth \\
        --kd-lambda 1.0 --ld-lambda 1.0 --kd-temperature 2.0 --kd-reg-max 32 \\
        -c configs/deim_dfine/deim_hgnetv2_s_traffic_light.yml \\
        --use-amp -u epoches=1 --output-dir ../runs/rehearsal_kd_A2b_seed0 --seed=0

Required guarantees enforced by main():
- CWD = DEIM/ (DEIM's import path + config-resolution all assume CWD=DEIM/).
- Single-node only (WORLD_SIZE == LOCAL_WORLD_SIZE). DEIM's setup_distributed
  uses global rank for cuda:set_device — multi-node would mismatch the
  LOCAL_RANK-keyed teacher device. Blocked with a clear error.
- --output-dir is REQUIRED (no bare-config output_dir fallback; KD rehearsal
  artifacts need an explicit, isolated dir to avoid colliding with prior runs).
- Teacher rebuilt from --teacher-cfg with STRICT state_dict load — any
  missing/unexpected key raises immediately rather than silently producing
  a partial-init teacher and contaminating KD signal.

Notes:
- DEIM teacher arch (M) and student arch (S) differ by hidden_dim / num_layers,
  but BOTH emit `pred_logits` shape (B, num_queries, num_classes) and
  `pred_corners` shape (B, num_queries, 4*(reg_max+1)) where num_queries=300
  and reg_max=32 by default. KD is shape-aligned without further projection.
- If teacher num_queries differs from student, the per-batch shape gate in
  deim_kd_engine._kd_terms silently skips KD on that mismatch; the caller
  must pin num_queries equal across teacher/student configs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def parse_kd_args() -> tuple[argparse.Namespace, list[str]]:
    """Consume KD-only flags; return (kd_args, remaining_argv_for_DEIM)."""
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--teacher-cfg", required=True,
                    help="DEIM YAML config for teacher arch (rebuilds model skeleton)")
    ap.add_argument("--teacher-ckpt", required=True,
                    help="Path to teacher .pth checkpoint (loaded into rebuilt teacher)")
    ap.add_argument("--kd-lambda", type=float, default=1.0)
    ap.add_argument("--ld-lambda", type=float, default=1.0)
    ap.add_argument("--kd-temperature", type=float, default=2.0)
    ap.add_argument("--kd-reg-max", type=int, default=32,
                    help="Must match DEIM cfg reg_max; default 32 per DEIM base config")
    return ap.parse_known_args()


def _validate_required_flag(argv: list[str], flag: str) -> None:
    """Validate a required `--flag VALUE` or `--flag=VALUE` token in argv.

    Mirrors scripts/train_deim.sh:72-145 rejection rules:
    - `--flag=` (empty equals form) → error
    - `--flag` as last token (no following value) → error
    - `--flag` followed by another flag (value starts with `-`) → error
    - `--flag` followed by empty string → error
    Otherwise: returns silently.

    Without this validation, an invalid form would be passed through to DEIM's
    argparse (which rejects it later) BUT the launcher's SEED.txt pre-write
    would have already written into `Path("")` or otherwise mis-placed files.
    """
    prev = ""
    found = False
    for arg in argv:
        if arg == flag:
            found = True
        elif arg.startswith(f"{flag}="):
            value = arg.split("=", 1)[1]
            if not value:
                raise RuntimeError(
                    f"deim_kd_launch: '{flag}=' has no value. "
                    f"Use {flag}=<value> or {flag} <value>."
                )
            found = True
        if prev == flag:
            # Space-form value follows the flag.
            if not arg or arg.startswith("-"):
                raise RuntimeError(
                    f"deim_kd_launch: '{flag}' missing value (got {arg!r})."
                )
        prev = arg

    if not found:
        raise RuntimeError(
            f"deim_kd_launch: {flag} is required. "
            "Either invoke via the deim_logit_localization_kd runner "
            "(which always passes both --output-dir and --seed), or add "
            "the flag to your dispatch."
        )
    # Trailing-bare-flag: last arg equals the flag with no following value.
    if argv and argv[-1] == flag:
        raise RuntimeError(
            f"deim_kd_launch: trailing '{flag}' has no value. "
            f"Use {flag}=<value> or {flag} <value>."
        )


def _is_resume_arg(arg: str) -> bool:
    """Detect a DEIM resume argv token.

    DEIM's `-r/--resume` accepts: space form (`-r CKPT` / `--resume CKPT`),
    equals form (`-r=CKPT` / `--resume=CKPT`), AND attached short form
    (`-rCKPT`). Mirrors scripts/train_deim.sh:94-99.

    NOTE: this only detects the TOKEN form; `_validate_resume_arg` enforces
    that the value is non-empty (which DEIM argparse treats as falsey-skip).
    """
    if arg in ("-r", "--resume"):
        return True
    if arg.startswith("--resume="):
        return True
    if arg.startswith("-r") and arg != "-r":
        # Covers both `-r=CKPT` and the attached short form `-rCKPT`.
        return True
    return False


def _validate_resume_arg(argv: list[str]) -> None:
    """Reject malformed resume forms before _pre_write_seed_marker runs.

    Mirrors scripts/train_deim.sh:75-84 + :142-144 rejection rules. The hazard:
    `--resume=` (empty equals) is structurally a resume token but DEIM argparse
    parses `args.resume == ""` as falsey and starts a fresh run — causing
    SEED.txt skip on the launcher side and fresh metadata loss on DEIM's side.
    """
    if not argv:
        return
    # Equals-form empty
    for arg in argv:
        if arg == "--resume=":
            raise RuntimeError(
                "deim_kd_launch: '--resume=' has no value. "
                "Use --resume=<ckpt> or --resume <ckpt>."
            )
        if arg == "-r=":
            raise RuntimeError(
                "deim_kd_launch: '-r=' has no value. Use -r=<ckpt> or -r <ckpt>."
            )
    # Trailing bare flag (no following value)
    if argv[-1] in ("-r", "--resume"):
        raise RuntimeError(
            f"deim_kd_launch: trailing {argv[-1]!r} has no value. "
            f"Use {argv[-1]}=<ckpt> or {argv[-1]} <ckpt>."
        )
    # Space-form value validation: next arg after --resume/-r
    prev = ""
    for arg in argv:
        if prev in ("-r", "--resume"):
            if not arg or arg.startswith("-"):
                raise RuntimeError(
                    f"deim_kd_launch: {prev!r} missing value (got {arg!r})."
                )
        prev = arg


def _extract_output_dir(argv: list[str]) -> str | None:
    """Pull --output-dir VALUE out of argv (space and equals forms)."""
    prev = ""
    for arg in argv:
        if prev == "--output-dir":
            return arg
        if arg.startswith("--output-dir="):
            return arg.split("=", 1)[1]
        prev = arg
    return None


def _strip_seed_args(argv: list[str]) -> list[str]:
    """Remove `--seed N` / `--seed=N` tokens from argv (for resume seed override).

    Mirrors scripts/train_deim.sh:240-264 — on resume, the recorded SEED.txt is
    the canonical source; any user-supplied --seed must be stripped to avoid
    DEIM seeing two seed values and the metadata desyncing from the runtime
    seed.
    """
    out: list[str] = []
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == "--seed":
            skip_next = True
            continue
        if arg.startswith("--seed="):
            continue
        out.append(arg)
    return out


def _pre_write_seed_marker(deim_argv: list[str]) -> None:
    """Mirror scripts/train_deim.sh's SEED.txt pre-write contract.

    The launcher bypasses scripts/train_deim.sh, so the wrapper's reproducibility
    plumbing (SEED.txt written at run START, not after — survives interrupted
    runs) must be recreated here. We scan `deim_argv` for `--seed`/`--seed=N`
    and `--output-dir`/`--output-dir=DIR` (forms accepted by DEIM/train.py
    argparse), create the dir if missing, and write SEED.txt — UNLESS the run
    is a resume (per _is_resume_arg), in which case SEED.txt MUST NOT be touched
    (CLAUDE.md: "the resume branch ... skips the SEED marker by design").

    Fail-fast on existing SEED.txt for fresh launches (matches train_deim.sh
    fresh-launch refusal at line 219). Override with `FORCE_FRESH=1` env var
    (also mirrors train_deim.sh contract).

    No-op if either --seed or --output-dir is missing — we don't speculate paths.
    """
    import os as _os

    seed: str | None = None
    output_dir: str | None = None
    is_resume = False
    prev = ""
    for arg in deim_argv:
        if prev == "--seed":
            seed = arg
        elif prev == "--output-dir":
            output_dir = arg
        if arg.startswith("--seed="):
            seed = arg.split("=", 1)[1]
        elif arg.startswith("--output-dir="):
            output_dir = arg.split("=", 1)[1]
        if _is_resume_arg(arg):
            is_resume = True
        prev = arg

    if is_resume or seed is None or output_dir is None:
        return
    if not seed.lstrip("-").isdigit():
        return  # mirror train_deim.sh: refuse non-integer seeds silently here (DEIM argparse will reject downstream)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    seed_path = out / "SEED.txt"
    if seed_path.exists():
        if _os.environ.get("FORCE_FRESH") == "1":
            seed_path.unlink()
        else:
            raise RuntimeError(
                f"deim_kd_launch: {seed_path} already exists. "
                "This means either (a) a prior run owns this dir — pass -r/--resume to continue it, "
                "or (b) a parallel run is in flight. "
                "Override with FORCE_FRESH=1 to deliberately overwrite."
            )
    # Atomic noclobber create. set_C-equivalent via open(..., 'x') semantics.
    try:
        with open(seed_path, "x") as f:
            f.write(f"{seed}\n")
    except FileExistsError:
        raise RuntimeError(
            f"deim_kd_launch: race writing {seed_path} (another process created it). "
            "Pass -r/--resume or FORCE_FRESH=1."
        )


def build_teacher(teacher_cfg: str, teacher_ckpt: str, device):
    """Rebuild teacher from DEIM YAML cfg + STRICT state_dict load.

    Strict load is load-bearing for KD correctness: a partial-load teacher
    would silently leave layers at random init and contaminate the KD signal
    with a meaningless distillation target. We refuse to proceed on any
    missing or unexpected keys (after the standard `module.` DDP-prefix strip).
    """
    import torch
    from engine.core import YAMLConfig  # DEIM module — imports only when run in DEIM venv

    cfg = YAMLConfig(teacher_cfg)
    teacher = cfg.model.to(device)
    state = torch.load(teacher_ckpt, map_location=device)
    # DEIM saves under "model" or "ema" / "ema.module" depending on stage.
    if isinstance(state, dict):
        if "ema" in state and isinstance(state["ema"], dict) and "module" in state["ema"]:
            sd = state["ema"]["module"]
            src = "state['ema']['module']"
        elif "model" in state:
            sd = state["model"]
            src = "state['model']"
        else:
            sd = state
            src = "state (raw dict)"
    else:
        sd = state
        src = "state (raw)"
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = teacher.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"deim_kd_launch: teacher state_dict load failed strictness check.\n"
            f"  source: {teacher_ckpt} (sub-key {src})\n"
            f"  missing keys ({len(missing)} total, first 5): {list(missing)[:5]}\n"
            f"  unexpected keys ({len(unexpected)} total, first 5): {list(unexpected)[:5]}\n"
            "Teacher checkpoint must match teacher cfg architecture exactly — "
            "any mismatch corrupts the KD signal silently. Verify --teacher-cfg "
            "matches the cfg the teacher was trained with."
        )
    # Disable teacher's denoising branch BEFORE any forward call. DEIM's decoder
    # gates denoising on `self.training and self.num_denoising > 0`
    # (dfine_decoder.py:710); if active, train-mode forward calls
    # `get_contrastive_denoising_training_group(targets, ...)` with the
    # `targets` arg of model.forward. The KD path calls `teacher(samples)` with
    # `targets=None`, which would crash. We keep teacher in train mode (for the
    # `pred_corners` output branch) but turn its denoising arm off — KD only
    # consumes top-level `pred_logits` + `pred_corners`, so denoising aux is
    # unnecessary anyway.
    if hasattr(teacher, "decoder") and hasattr(teacher.decoder, "num_denoising"):
        teacher.decoder.num_denoising = 0
    # Freezing + train-mode-with-BN-frozen happens in install() (deim_kd_engine).
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def main() -> int:
    import os
    import runpy
    import torch

    # CWD must be DEIM/ (the dispatch wrapper sets this via `cd DEIM && ...`).
    # Ensure DEIM-relative imports resolve: insert "." so `engine.solver.*`
    # (used by DEIM/train.py) finds DEIM internals.
    if "." not in sys.path:
        sys.path.insert(0, ".")
    # Sanity-check we're in the right place — DEIM/train.py must exist here.
    if not Path("train.py").exists() or not Path("engine").is_dir():
        raise RuntimeError(
            f"deim_kd_launch: CWD must be DEIM/ (current: {os.getcwd()!r}). "
            "Use `cd DEIM && torchrun ../components/.../deim_kd_launch.py ...`."
        )

    # Strip KD-only argv before DEIM's argparse sees it.
    kd_args, remaining = parse_kd_args()
    sys.argv = [sys.argv[0]] + remaining

    # Multi-node guard: DEIM's setup_distributed (DEIM/engine/misc/dist_utils.py:47-49)
    # calls torch.cuda.set_device(GLOBAL_rank), which can mismatch our LOCAL_RANK
    # teacher device on multi-node. Until DEIM upstream is fixed, the KD launcher
    # only supports single-node. Block multi-node explicitly with a clear error.
    _world_size = int(os.environ.get("WORLD_SIZE", "1"))
    _local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    if _world_size > _local_world_size:
        raise RuntimeError(
            f"deim_kd_launch: multi-node not supported (WORLD_SIZE={_world_size} > "
            f"LOCAL_WORLD_SIZE={_local_world_size}). DEIM's setup_distributed uses "
            "global rank for cuda:set_device — teacher (LOCAL_RANK-keyed) would "
            "mismatch student (global-rank-keyed). Single-node torchrun only."
        )

    # Validate resume value forms first (rejects empty `--resume=` etc. that
    # are token-shaped but DEIM treats as falsey-fresh). After this passes,
    # _is_resume_arg correctly partitions argv into resume vs fresh launches.
    _validate_resume_arg(remaining)
    _is_resume_launch = any(_is_resume_arg(a) for a in remaining)

    # --output-dir is REQUIRED for BOTH fresh AND resume launches:
    #   fresh:  destination of new SEED.txt
    #   resume: source of existing SEED.txt (where the original seed was recorded)
    # Mirrors scripts/train_deim.sh's contract (output-dir flows through both branches).
    _validate_required_flag(remaining, "--output-dir")
    if not _is_resume_launch:
        # Fresh launches MUST also supply --seed (we write the marker).
        # Resume launches must NOT supply --seed (we source it from SEED.txt).
        _validate_required_flag(remaining, "--seed")
    else:
        # On resume: source seed from existing SEED.txt, strip any user --seed
        # from argv, append --seed=<sourced> so DEIM's RNG matches the marker.
        # Mirrors scripts/train_deim.sh:180-204 + :240-264.
        output_dir = _extract_output_dir(remaining)
        assert output_dir is not None  # _validate_required_flag passed above
        seed_path = Path(output_dir) / "SEED.txt"
        if not seed_path.is_file() or seed_path.stat().st_size == 0:
            raise RuntimeError(
                f"deim_kd_launch: resume requires {seed_path} to exist with a "
                "non-empty integer seed. Use FORCE_FRESH=1 to override with a "
                "fresh launch (will overwrite SEED.txt)."
            )
        recorded_seed = seed_path.read_text().strip()
        if not recorded_seed.lstrip("-").isdigit():
            raise RuntimeError(
                f"deim_kd_launch: {seed_path} contains non-integer "
                f"value {recorded_seed!r}; cannot resume safely."
            )
        # Strip user seed args + inject the recorded seed.
        remaining = _strip_seed_args(remaining)
        remaining.append(f"--seed={recorded_seed}")
        sys.argv = [sys.argv[0]] + remaining
        print(
            f"[deim_kd_launch] resume mode: sourcing seed={recorded_seed} "
            f"from {seed_path}; user --seed args stripped if any."
        )

    # SEED.txt parity with scripts/train_deim.sh: the launcher bypasses the
    # shell wrapper, so we recreate its SEED.txt pre-write contract here so
    # the rehearsal output dir gains reproducibility metadata. CLAUDE.md
    # §"Reproducibility plumbing" requires SEED.txt to be written at START,
    # not after — survives interrupted runs.
    _pre_write_seed_marker(remaining)

    # Per-rank device selection BEFORE building teacher. Under
    # `torchrun --nproc_per_node>1`, every rank would otherwise materialize the
    # teacher on cuda:0 (default `torch.device("cuda")` resolves to cuda:0),
    # blowing up rank-0 GPU memory and mismatching student device. torchrun sets
    # LOCAL_RANK; we honor it before DEIM's setup_distributed call.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    teacher = build_teacher(kd_args.teacher_cfg, kd_args.teacher_ckpt, device)

    # Patch DEIM's train_one_epoch BEFORE running train.py. install() patches
    # both det_engine AND det_solver (the latter binds train_one_epoch locally
    # at import time — patching only det_engine would miss det_solver's copy).
    from components.knowledge_distillation.integration.deim_kd_engine import install
    patched = install(
        teacher=teacher,
        kd_lambda=kd_args.kd_lambda,
        ld_lambda=kd_args.ld_lambda,
        kd_temperature=kd_args.kd_temperature,
        reg_max=kd_args.kd_reg_max,
    )
    print(
        f"[deim_kd_launch] KD patch installed: kd_lambda={kd_args.kd_lambda} "
        f"ld_lambda={kd_args.ld_lambda} T={kd_args.kd_temperature} "
        f"reg_max={kd_args.kd_reg_max} | patched modules: {patched}"
    )

    # Defer to DEIM's main() via runpy. DEIM/train.py's `__main__` guard runs
    # its own argparse (allow_abbrev=False, see DEIM/train.py:67) on the
    # mutated sys.argv, then calls main(args). Direct `import train; train.main()`
    # fails because train.main requires the parsed Namespace positional.
    runpy.run_path("train.py", run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
