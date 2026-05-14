"""YAML → typed-dataclass loader for the copy-paste / class-balance ablation.

Sister-file precedent: ``components/hmm_smoother/config.py`` (locked
iter-3, 2026-05-09). Loader exposes the YAML→dataclass conversion as
the locked a-stage public API entry point so b-stage doesn't invent a
hidden coercion contract inside the runner.

The loader returns a single ``CopyPasteBalanceYamlConfig`` bundling
every YAML knob (data + copy_paste + class_balance + decision tolerance).
Per-subsystem dataclasses (``CopyPasteConfig``, ``ClassBalanceWeights``
constructor args, etc.) are built FROM ``CopyPasteBalanceYamlConfig`` —
never directly from raw YAML — so the coercion contract has exactly
one load-bearing entry point.

Scaffold (a-stage): function signature only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from components.copy_paste_balance.modules.class_balance import (
    ClassBalanceApplyMode,
)
from components.copy_paste_balance.modules.copy_paste import CopyPasteConfig


@dataclass(frozen=True)
class CopyPasteBalanceYamlConfig:
    """Typed representation of ``configs/copy_paste_balance.yaml``.

    Every field corresponds 1:1 to a YAML key (no renames, no derivations).
    String enums and list-of-int are coerced into their typed counterparts
    here so downstream consumers never see raw YAML scalars.

    Cross-knob invariants enforced in ``__post_init__`` (pure value checks
    only — no I/O, no filesystem):
      * ``num_classes`` is set (the YAML default ``null`` is forbidden at
        this layer; the loader/runner is responsible for refusing to load
        a YAML with ``num_classes: null``).
      * ``class_names`` length equals ``num_classes``.
      * ``rare_class_threshold >= 0``.
      * ``safety_class_ids`` is a subset of ``range(num_classes)``.
      * ``copy_paste_config`` (delegated): see ``CopyPasteConfig.__post_init__``
        for the y-center mask, fliplr lock, source-id range checks.
      * ``class_balance_beta`` is one of {0.0, 0.99, 0.999, 0.9999} —
        plan §3.7 sensitivity-sweep set; YAML deviation is a contract
        violation (mirrors HMM's hard-fail on laplace_alpha).
      * ``map_regression_tolerance_pp`` is finite and >= 0.

    Filesystem-state checks: ``class_counts_path`` existence is NOT
    checked here (mirrors HMM's transition_matrix_path policy — frozen-
    dataclass ``__post_init__`` doing I/O makes construction non-
    deterministic). The runner enforces existence + readability before
    consuming the path; missing or unreadable paths produce diagnostic
    ``decision: "executor_error"`` rows.
    """

    num_classes: int
    class_names: tuple[str, ...]
    rare_class_threshold: int
    safety_class_ids: tuple[int, ...]

    # Copy-paste subsystem (delegated to CopyPasteConfig for granular checks)
    copy_paste_config: CopyPasteConfig

    # Class-balance subsystem (loose form here; the runner builds the
    # ClassBalanceWeights dataclass via from_counts() at training time)
    class_balance_beta: float
    class_balance_apply_mode: ClassBalanceApplyMode
    class_balance_max_weight_ratio: float
    class_counts_path: Path | None

    # Decision-rule tolerance (the only §3.7 threshold lifted to YAML; the
    # rest are pinned as ClassVars on DecisionInputs for the same reason
    # HMM's gate pinned its non-tolerance thresholds).
    map_regression_tolerance_pp: float

    _ALLOWED_BETAS: ClassVar[tuple[float, ...]] = (0.0, 0.99, 0.999, 0.9999)

    def __post_init__(self) -> None:
        # Pure value checks only — NO filesystem I/O (mirrors HMM iter-2
        # NEW-MINOR 5).
        if self.num_classes is None:
            raise ValueError(
                "num_classes must be set explicitly; got None "
                "(YAML default `null` is rejected — the loader / runner MUST "
                "provide a concrete int matching the detector's class count)"
            )
        if not isinstance(self.num_classes, int) or isinstance(self.num_classes, bool):
            raise ValueError(
                f"num_classes must be int; got "
                f"{type(self.num_classes).__name__}={self.num_classes!r}"
            )
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be > 0; got {self.num_classes}")
        # class_names must be a tuple of str, length == num_classes. YAML
        # serializes as list[str] so the loader MUST coerce to tuple before
        # construction; without this, frozen-dataclass mutation of a list
        # field would surface as a confusing TypeError far from the loader.
        if not isinstance(self.class_names, tuple):
            raise ValueError(
                f"class_names must be tuple; got {type(self.class_names).__name__} "
                f"(loader must coerce YAML list → tuple before construction)"
            )
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"class_names length ({len(self.class_names)}) must equal "
                f"num_classes ({self.num_classes})"
            )
        for i, name in enumerate(self.class_names):
            if not isinstance(name, str):
                raise ValueError(
                    f"class_names[{i}] must be str; got {type(name).__name__}={name!r}"
                )
            if not name:
                raise ValueError(f"class_names[{i}] must be non-empty")
        if not isinstance(self.rare_class_threshold, int) or isinstance(self.rare_class_threshold, bool):
            raise ValueError(
                f"rare_class_threshold must be int; got "
                f"{type(self.rare_class_threshold).__name__}={self.rare_class_threshold!r}"
            )
        if self.rare_class_threshold < 0:
            raise ValueError(
                f"rare_class_threshold must be >= 0; got {self.rare_class_threshold}"
            )
        # safety_class_ids — int tuple, every entry in [0, num_classes).
        # Bool exclusion mirrors transition.py's illegal-cell loop: YAML may
        # serialize a bare `true` as a class ID and silently alias to 1.
        if not isinstance(self.safety_class_ids, tuple):
            raise ValueError(
                f"safety_class_ids must be tuple; got "
                f"{type(self.safety_class_ids).__name__} "
                f"(loader must coerce YAML list → tuple before construction)"
            )
        for i, cid in enumerate(self.safety_class_ids):
            if not isinstance(cid, int) or isinstance(cid, bool):
                raise ValueError(
                    f"safety_class_ids[{i}] must be int; got "
                    f"{type(cid).__name__}={cid!r}"
                )
            if not (0 <= cid < self.num_classes):
                raise ValueError(
                    f"safety_class_ids[{i}]={cid} out of range "
                    f"[0, {self.num_classes})"
                )
        # B2 review S4 2026-05-09: reject duplicates at the input boundary
        # (the output schema's MetricsBlock.rare_safety_class_ids already
        # enforces uniqueItems; symmetrize here so a typo'd YAML doesn't
        # silently dedupe at the rare ∩ safety set computation).
        if len(set(self.safety_class_ids)) != len(self.safety_class_ids):
            raise ValueError(
                f"safety_class_ids must contain no duplicates; got "
                f"{self.safety_class_ids}"
            )
        # copy_paste_config: type check only — its own __post_init__ runs
        # at construction so per-field validation already happened.
        if not isinstance(self.copy_paste_config, CopyPasteConfig):
            raise ValueError(
                f"copy_paste_config must be CopyPasteConfig; got "
                f"{type(self.copy_paste_config).__name__}"
            )
        # Cross-config invariant: nested CopyPasteConfig.num_classes must
        # match the outer num_classes — otherwise the source-id range check
        # in CopyPasteConfig is computed against the wrong C and a stale
        # nested config could ship with mismatched bounds.
        if self.copy_paste_config.num_classes != self.num_classes:
            raise ValueError(
                f"copy_paste_config.num_classes ({self.copy_paste_config.num_classes}) "
                f"must equal outer num_classes ({self.num_classes})"
            )
        # class_balance_beta: hard-pin to plan §3.7 sweep set. Bool-exclusion
        # before the membership test (mirrors HMM laplace_alpha I2): `True
        # in (0.0, 0.99, 0.999, 0.9999)` is True (Python coerces bool→float
        # in equality), so `class_balance_beta=True` would silently alias to
        # 1.0 and corrupt the sensitivity-sweep artifact (β=1.0 is also
        # forbidden by Cui's formula since it divides by 1-β).
        if not isinstance(self.class_balance_beta, float) or isinstance(self.class_balance_beta, bool):
            raise ValueError(
                f"class_balance_beta must be float; got "
                f"{type(self.class_balance_beta).__name__}={self.class_balance_beta!r}"
            )
        if not math.isfinite(self.class_balance_beta):
            raise ValueError(
                f"class_balance_beta must be finite; got {self.class_balance_beta!r}"
            )
        if self.class_balance_beta not in self._ALLOWED_BETAS:
            raise ValueError(
                f"class_balance_beta must be one of {self._ALLOWED_BETAS}; "
                f"got {self.class_balance_beta}. Plan §3.7 mandates this "
                f"fixed sweep set; deviation is a contract violation."
            )
        if not isinstance(self.class_balance_apply_mode, ClassBalanceApplyMode):
            raise ValueError(
                f"class_balance_apply_mode must be ClassBalanceApplyMode enum; "
                f"got {type(self.class_balance_apply_mode).__name__} "
                f"(loader must coerce YAML string before construction)"
            )
        if not isinstance(self.class_balance_max_weight_ratio, float) or isinstance(self.class_balance_max_weight_ratio, bool):
            raise ValueError(
                f"class_balance_max_weight_ratio must be float; got "
                f"{type(self.class_balance_max_weight_ratio).__name__}="
                f"{self.class_balance_max_weight_ratio!r}"
            )
        if not math.isfinite(self.class_balance_max_weight_ratio):
            raise ValueError(
                f"class_balance_max_weight_ratio must be finite; got "
                f"{self.class_balance_max_weight_ratio!r}"
            )
        if self.class_balance_max_weight_ratio < 1.0:
            raise ValueError(
                f"class_balance_max_weight_ratio must be >= 1.0; got "
                f"{self.class_balance_max_weight_ratio}"
            )
        # class_counts_path: Path-vs-str type guard (mirrors HMM
        # transition_matrix_path C3 stop-hook iter-2 NEW-MINOR fix). The
        # loader is responsible for str→Path coercion; without this guard
        # a forgotten coercion ships a raw YAML str downstream and the
        # failure surfaces later as a confusing AttributeError.
        if self.class_counts_path is not None and not isinstance(self.class_counts_path, Path):
            raise ValueError(
                f"class_counts_path must be Path or None; got "
                f"{type(self.class_counts_path).__name__}={self.class_counts_path!r} "
                f"(loader must coerce YAML string → pathlib.Path before construction)"
            )
        # map_regression_tolerance_pp: float-with-bool-exclusion + finite
        # + non-negative (mirrors HMM C3 stop-hook NEW-MAJOR fix). NaN/inf
        # bypass `< 0.0` and would silently widen the deploy gate; finite
        # check before range check.
        if not isinstance(self.map_regression_tolerance_pp, float) or isinstance(self.map_regression_tolerance_pp, bool):
            raise ValueError(
                f"map_regression_tolerance_pp must be float; got "
                f"{type(self.map_regression_tolerance_pp).__name__}="
                f"{self.map_regression_tolerance_pp!r}"
            )
        if not math.isfinite(self.map_regression_tolerance_pp):
            raise ValueError(
                f"map_regression_tolerance_pp must be finite; got "
                f"{self.map_regression_tolerance_pp!r}"
            )
        if self.map_regression_tolerance_pp < 0.0:
            raise ValueError(
                f"map_regression_tolerance_pp must be >= 0; got "
                f"{self.map_regression_tolerance_pp}"
            )
        # C3 iter-7 NEW-MAJOR (tolerance ceiling) 2026-05-09: cap at the
        # plan-pinned drop threshold (0.5 pp; mirrors
        # DecisionInputs.DROP_TOTAL_MAP_REGRESSION_PP). Beyond 0.5, the
        # deploy guard would tolerate regressions that the drop trigger
        # also rejects, making the rule incoherent (deploy and drop
        # would both fire). Hardcoded here to avoid a circular import
        # with gates/decision_gate.py — the value is the source of truth
        # over there; if that ClassVar changes, this constant changes
        # in lock-step.
        if self.map_regression_tolerance_pp > 0.5:
            raise ValueError(
                f"map_regression_tolerance_pp must be <= 0.5 (the §3.7 "
                f"drop threshold for total mAP regression); got "
                f"{self.map_regression_tolerance_pp} (a tolerance > 0.5 makes "
                f"the deploy guard broader than the drop trigger, which is "
                f"incoherent — both cases would fire on the same input)"
            )


def load_copy_paste_balance_yaml(path: str | Path) -> CopyPasteBalanceYamlConfig:
    """Parse ``configs/copy_paste_balance.yaml`` into a typed config bundle.

    Coercion contract (b-stage MUST honor ALL of these):
      * ``class_names`` (YAML list[str]) → ``tuple(str(name) for name in raw)``.
      * ``safety_class_ids`` (YAML list[int]) → ``tuple(int(c) for c in raw)``.
      * ``paste_source_class_ids`` inside ``copy_paste:`` (YAML list[int]) →
        ``tuple(int(c) for c in raw)``.
      * ``class_balance.apply_mode`` (YAML string) →
        ``ClassBalanceApplyMode(raw)``. Unknown values raise ``ValueError``.
      * ``class_counts_path`` (YAML string or null) → ``Path`` or ``None``.
        Empty string is normalized to ``None``.
      * ``num_classes: null`` raises ``ValueError`` immediately — the
        loader / runner is responsible for setting an explicit value
        before consuming this YAML.
      * ``class_balance.beta`` must be one of {0.0, 0.99, 0.999, 0.9999};
        any other value raises ``ValueError`` rather than silently
        widening the §3.7 sensitivity-sweep set (mirrors HMM's
        laplace_alpha hard-pin).

    Args:
        path: filesystem path to a ``copy_paste_balance.yaml`` file.

    Returns:
        ``CopyPasteBalanceYamlConfig`` — fully validated, ready for
        downstream use by the ablation runner and trainer callbacks.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: schema / range / enum violations as described above.
    """
    import yaml as _yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    with path.open("r", encoding="utf-8") as fh:
        raw = _yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: top-level YAML must be a mapping; got {type(raw).__name__}"
        )

    def _path_or_none(val: object, field: str) -> Path | None:
        if val is None:
            return None
        if isinstance(val, str):
            return None if val == "" else Path(val)
        raise ValueError(
            f"{path}: {field} must be string or null; got "
            f"{type(val).__name__}={val!r}"
        )

    def _str_tuple(val: object, field: str) -> tuple[str, ...]:
        # C3 review MINOR-3 fix: reject non-string entries at the YAML
        # layer instead of stringifying. ``class_names: [0, 1, ...]`` would
        # silently coerce to ``["0", "1", ...]`` and break the per-class
        # label contract downstream. Mirrors HN's strict _str_tuple.
        if not isinstance(val, list):
            raise ValueError(
                f"{path}: {field} must be a YAML list; got "
                f"{type(val).__name__}"
            )
        out: list[str] = []
        for i, item in enumerate(val):
            if not isinstance(item, str):
                raise ValueError(
                    f"{path}: {field}[{i}] must be str; got "
                    f"{type(item).__name__}={item!r}"
                )
            out.append(item)
        return tuple(out)

    def _int_tuple(val: object, field: str) -> tuple[int, ...]:
        if not isinstance(val, list):
            raise ValueError(
                f"{path}: {field} must be a YAML list; got "
                f"{type(val).__name__}"
            )
        out: list[int] = []
        for i, item in enumerate(val):
            if not isinstance(item, int) or isinstance(item, bool):
                raise ValueError(
                    f"{path}: {field}[{i}] must be int; got "
                    f"{type(item).__name__}={item!r}"
                )
            out.append(item)
        return tuple(out)

    if raw.get("num_classes") is None:
        raise ValueError(
            f"{path}: num_classes must be set explicitly; got null"
        )

    cp_raw = raw.get("copy_paste", {})
    if not isinstance(cp_raw, dict):
        raise ValueError(
            f"{path}: copy_paste block must be a mapping; got "
            f"{type(cp_raw).__name__}"
        )
    cp = CopyPasteConfig(
        num_classes=raw["num_classes"],
        probability=float(cp_raw.get("probability", 0.0))
        if not isinstance(cp_raw.get("probability"), bool)
        else cp_raw["probability"],
        y_center_max_frac=(
            None if cp_raw.get("y_center_max_frac") is None
            else float(cp_raw["y_center_max_frac"])
        ),
        paste_source_class_ids=_int_tuple(
            cp_raw.get("paste_source_class_ids", []),
            "copy_paste.paste_source_class_ids",
        ),
        min_per_batch_K=cp_raw.get("min_per_batch_K", 0),
        required_fliplr=float(cp_raw.get("required_fliplr", 0.0))
        if not isinstance(cp_raw.get("required_fliplr"), bool)
        else cp_raw["required_fliplr"],
        required_mosaic_lock=cp_raw.get("required_mosaic_lock", False),
    )

    cb_raw = raw.get("class_balance", {})
    if not isinstance(cb_raw, dict):
        raise ValueError(
            f"{path}: class_balance block must be a mapping; got "
            f"{type(cb_raw).__name__}"
        )
    apply_mode_raw = cb_raw.get("apply_mode")
    if not isinstance(apply_mode_raw, str):
        raise ValueError(
            f"{path}: class_balance.apply_mode must be str; got "
            f"{type(apply_mode_raw).__name__}={apply_mode_raw!r}"
        )
    try:
        apply_mode = ClassBalanceApplyMode(apply_mode_raw)
    except ValueError as e:
        raise ValueError(
            f"{path}: class_balance.apply_mode={apply_mode_raw!r} is not a "
            f"valid ClassBalanceApplyMode "
            f"({[m.value for m in ClassBalanceApplyMode]})"
        ) from e

    return CopyPasteBalanceYamlConfig(
        num_classes=raw["num_classes"],
        class_names=_str_tuple(raw.get("class_names", []), "class_names"),
        rare_class_threshold=raw.get("rare_class_threshold", 0),
        safety_class_ids=_int_tuple(
            raw.get("safety_class_ids", []), "safety_class_ids"
        ),
        copy_paste_config=cp,
        class_balance_beta=float(cb_raw.get("beta", 0.0))
        if not isinstance(cb_raw.get("beta"), bool)
        else cb_raw["beta"],
        class_balance_apply_mode=apply_mode,
        class_balance_max_weight_ratio=float(
            cb_raw.get("max_weight_ratio", 1.0)
        )
        if not isinstance(cb_raw.get("max_weight_ratio"), bool)
        else cb_raw["max_weight_ratio"],
        class_counts_path=_path_or_none(
            cb_raw.get("class_counts_path"), "class_balance.class_counts_path"
        ),
        map_regression_tolerance_pp=float(
            raw.get("map_regression_tolerance_pp", 0.0)
        )
        if not isinstance(raw.get("map_regression_tolerance_pp"), bool)
        else raw["map_regression_tolerance_pp"],
    )
