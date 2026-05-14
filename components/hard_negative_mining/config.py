"""YAML → typed-dataclass loader for the §四 hard-negative-mining ablation.

Sister-file precedent (internal B2 audit trail; b-stage authors can
ignore): ``components/copy_paste_balance/config.py`` (locked iter-11,
2026-05-09). Loader exposes the YAML→dataclass conversion as the
locked a-stage public API entry point so b-stage doesn't invent a
hidden coercion contract inside the runner.

The loader returns a single ``HardNegativeMiningYamlConfig`` bundling
every YAML knob (mining sources + verification protocol + decision
tolerance). Per-subsystem dataclasses
(``HardNegativeMinerConfig``, ``VerificationConfig``, etc.) are built
FROM ``HardNegativeMiningYamlConfig`` — never directly from raw YAML —
so the coercion contract has exactly one load-bearing entry point.

Scaffold (a-stage): function signature only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from components.hard_negative_mining._internals import (
    PLAN_LOCKED_MIN_SAMPLE_FRACTION,
    PLAN_MAP_TOLERANCE_CEILING_PP,
    PLAN_MINING_SOURCES,
    PLAN_TPM_RATE_CEILING,
    is_hex_sha256,
)


@dataclass(frozen=True)
class HardNegativeMiningYamlConfig:
    """Typed representation of ``configs/hard_negative_mining.yaml``.

    Every field corresponds 1:1 to a YAML key. String enums and list-of-
    Path / list-of-str are coerced into their typed counterparts here so
    downstream consumers never see raw YAML scalars.

    Cross-knob invariants enforced in ``__post_init__`` (pure value
    checks only — no I/O, no filesystem):
      * ``schema_version`` is the literal string ``"1"`` (S9: forward-
        compat discriminator for b-stage / R3 migrations).
      * ``num_classes`` is a positive int (bool excluded; YAML default
        ``null`` is forbidden).
      * ``class_names`` length equals ``num_classes``.
      * **C3 iter-1 NEW-MAJOR 1 2026-05-10**: ``data_yaml_sha256`` is
        a 64-char lowercase hex string stamped into the YAML at fit
        time. It is the LOAD-BEARING anchor for cross-artifact
        class-set provenance: the runner Step 0 cross-checks this
        value against the ``data_yaml_sha256`` recorded in BOTH
        per-arm eval JSONs AND the artifact's top-level
        ``data_yaml_sha256``. Without this field, the runner's
        Step 0 contract was undocumented vs. unimplementable; the
        previous design relied solely on cross-eval-JSON equality
        (silently accepting a YAML pointed at a different data.yaml
        than the one the trainer evaluated against).
      * ``map_regression_tolerance_pp`` is finite, non-negative, and
        bounded above by ``PLAN_MAP_TOLERANCE_CEILING_PP`` (0.5; the
        §3.7 / §4.7 drop threshold for total mAP regression).
      * ``frozen_manifest_path`` (b-stage consumer enforces existence)
        is Path or None — Path-vs-str type guard mirrors §3.7 / HMM.
      * ``baseline_weights_path`` is Path or None.
      * ``mining_sources`` is a tuple of source labels (subset of
        plan-pinned set); empty is allowed when the b-stage config
        only drives the d-stage runner (mining already done).
      * ``min_sample_fraction`` is hard-pinned to
        ``PLAN_LOCKED_MIN_SAMPLE_FRACTION`` (plan §4.5).
      * **B2 review I4 2026-05-10**: when both
        ``baseline_weights_path`` and ``output_candidates_path`` are
        set, they must differ (writing the candidates JSON over the
        model checkpoint would clobber the baseline). Same check
        between ``output_candidates_path`` and ``frozen_manifest_path``.

    Filesystem-state checks: existence is NOT checked here (mirrors
    §3.7 config.py policy). The runner / mining pass enforces existence
    before consuming paths; missing or unreadable paths produce
    diagnostic ``decision: "executor_error"`` rows.
    """

    schema_version: str
    num_classes: int
    class_names: tuple[str, ...]
    # C3 iter-1 NEW-MAJOR 1 2026-05-10: load-bearing class-set
    # provenance anchor. 64-char lowercase hex; cross-checked at the
    # runner against the per-arm eval JSONs' data_yaml_sha256 fields
    # AND the output artifact's top-level data_yaml_sha256.
    data_yaml_sha256: str

    # Mining-side knobs (consumed by modules/miner.py)
    baseline_weights_path: Path | None
    mining_sources: tuple[str, ...]              # subset of plan-pinned set
    # B2 review I5 2026-05-10: rename ``candidates_output_path`` →
    # ``output_candidates_path`` to match
    # ``HardNegativeMinerConfig.output_candidates_path``. Loader maps
    # the legacy key for one-version transition; b-stage drops the
    # legacy key.
    output_candidates_path: Path | None

    # Verification-side knobs (consumed by modules/verifier.py)
    min_sample_fraction: float                   # plan §4.5: 0.10
    max_true_positive_missed_rate: float

    # Frozen-manifest path (consumed by runners/ablation.py)
    frozen_manifest_path: Path | None

    # Decision-rule tolerance (the only §4.7 threshold lifted to YAML;
    # the rest are pinned as ClassVars on DecisionInputs).
    map_regression_tolerance_pp: float

    # B2 final-review S3 2026-05-10: this literal MUST be kept in sync
    # with the schema's const at
    # _hard_negative_decision_schema.json#/properties/schema_version/const.
    # JSON Schema can't import Python constants, so the two layers are
    # documented anchors rather than referentially linked. R3 cutover to
    # "2" must edit BOTH sites in lock-step.
    _LOCKED_SCHEMA_VERSION: ClassVar[str] = "1"
    _ALLOWED_MINING_SOURCES: ClassVar[tuple[str, ...]] = PLAN_MINING_SOURCES
    _LOCKED_MIN_SAMPLE_FRACTION: ClassVar[float] = PLAN_LOCKED_MIN_SAMPLE_FRACTION
    _MAX_TPM_RATE_CEILING: ClassVar[float] = PLAN_TPM_RATE_CEILING
    _MAP_TOLERANCE_CEILING_PP: ClassVar[float] = PLAN_MAP_TOLERANCE_CEILING_PP

    def __post_init__(self) -> None:
        # schema_version: forward-compat discriminator. Hard-pinned at
        # the loader because YAML-format migrations are the b-stage
        # dropping-the-legacy-key concern; here we want a stable surface.
        if not isinstance(self.schema_version, str):
            raise ValueError(
                f"schema_version must be str; got "
                f"{type(self.schema_version).__name__}={self.schema_version!r}"
            )
        if self.schema_version != self._LOCKED_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must equal {self._LOCKED_SCHEMA_VERSION!r}; "
                f"got {self.schema_version!r} (b-stage migration drops the "
                f"old key; do not silently widen)"
            )
        # data_yaml_sha256: 64-char lowercase hex (the load-bearing
        # class-set provenance anchor; cross-checked at the runner).
        if not is_hex_sha256(self.data_yaml_sha256):
            raise ValueError(
                f"data_yaml_sha256 must be 64-char lowercase hex; got "
                f"{self.data_yaml_sha256!r} (this is the load-bearing "
                f"class-set provenance anchor; the runner Step 0 "
                f"cross-checks this against per-arm eval JSONs and the "
                f"output artifact's top-level data_yaml_sha256)"
            )
        # num_classes: positive int with bool exclusion.
        if self.num_classes is None:
            raise ValueError(
                "num_classes must be set explicitly; got None "
                "(YAML default `null` is rejected)"
            )
        if not isinstance(self.num_classes, int) or isinstance(self.num_classes, bool):
            raise ValueError(
                f"num_classes must be int; got "
                f"{type(self.num_classes).__name__}={self.num_classes!r}"
            )
        if self.num_classes <= 0:
            raise ValueError(
                f"num_classes must be > 0; got {self.num_classes}"
            )
        # class_names: tuple of non-empty str, length matches num_classes.
        if not isinstance(self.class_names, tuple):
            raise ValueError(
                f"class_names must be tuple; got "
                f"{type(self.class_names).__name__} "
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
                    f"class_names[{i}] must be str; got "
                    f"{type(name).__name__}={name!r}"
                )
            if not name:
                raise ValueError(f"class_names[{i}] must be non-empty")
        # baseline_weights_path: Path or None.
        if self.baseline_weights_path is not None and not isinstance(
            self.baseline_weights_path, Path
        ):
            raise ValueError(
                f"baseline_weights_path must be Path or None; got "
                f"{type(self.baseline_weights_path).__name__}="
                f"{self.baseline_weights_path!r} "
                f"(loader must coerce YAML string → pathlib.Path before construction)"
            )
        # mining_sources: tuple of plan-pinned strings; duplicates rejected.
        if not isinstance(self.mining_sources, tuple):
            raise ValueError(
                f"mining_sources must be tuple; got "
                f"{type(self.mining_sources).__name__}"
            )
        for i, src in enumerate(self.mining_sources):
            if not isinstance(src, str):
                raise ValueError(
                    f"mining_sources[{i}] must be str; got "
                    f"{type(src).__name__}={src!r}"
                )
            if src not in self._ALLOWED_MINING_SOURCES:
                raise ValueError(
                    f"mining_sources[{i}]={src!r} not in plan-pinned set "
                    f"{self._ALLOWED_MINING_SOURCES} (adding a new source "
                    f"requires re-running the §四 adversarial loop)"
                )
        if len(set(self.mining_sources)) != len(self.mining_sources):
            raise ValueError(
                f"mining_sources must contain no duplicates; got "
                f"{self.mining_sources}"
            )
        # output_candidates_path: Path or None.
        if self.output_candidates_path is not None and not isinstance(
            self.output_candidates_path, Path
        ):
            raise ValueError(
                f"output_candidates_path must be Path or None; got "
                f"{type(self.output_candidates_path).__name__}="
                f"{self.output_candidates_path!r}"
            )
        # frozen_manifest_path: Path or None.
        if self.frozen_manifest_path is not None and not isinstance(
            self.frozen_manifest_path, Path
        ):
            raise ValueError(
                f"frozen_manifest_path must be Path or None; got "
                f"{type(self.frozen_manifest_path).__name__}="
                f"{self.frozen_manifest_path!r}"
            )
        # B2 review I4 2026-05-10: pairwise distinctness for path fields
        # that are BOTH set (None excluded — None means "not configured").
        # Catastrophic-pair enumeration is cheap; a generic "all distinct"
        # would over-trigger on legit None / None matches.
        path_pairs = [
            ("baseline_weights_path", self.baseline_weights_path,
             "output_candidates_path", self.output_candidates_path,
             "writing the candidates JSON over the model checkpoint would clobber the baseline"),
            ("output_candidates_path", self.output_candidates_path,
             "frozen_manifest_path", self.frozen_manifest_path,
             "writing the candidates JSON over the frozen manifest would corrupt the FP-denominator"),
            ("baseline_weights_path", self.baseline_weights_path,
             "frozen_manifest_path", self.frozen_manifest_path,
             "the model checkpoint and the frozen manifest are distinct artifacts; equal paths indicates a misconfiguration"),
        ]
        for a_name, a_val, b_name, b_val, hazard in path_pairs:
            if a_val is not None and b_val is not None and a_val == b_val:
                raise ValueError(
                    f"{a_name} and {b_name} must differ when both are set; "
                    f"got {a_val} for both ({hazard})"
                )
        # min_sample_fraction: plan §4.5 hard-pin. Bool-exclusion before
        # equality check (True == 1.0 silently aliases otherwise).
        if not isinstance(self.min_sample_fraction, float) or isinstance(
            self.min_sample_fraction, bool
        ):
            raise ValueError(
                f"min_sample_fraction must be float; got "
                f"{type(self.min_sample_fraction).__name__}="
                f"{self.min_sample_fraction!r}"
            )
        if not math.isfinite(self.min_sample_fraction):
            raise ValueError(
                f"min_sample_fraction must be finite; got "
                f"{self.min_sample_fraction!r}"
            )
        if self.min_sample_fraction != self._LOCKED_MIN_SAMPLE_FRACTION:
            raise ValueError(
                f"min_sample_fraction must equal "
                f"{self._LOCKED_MIN_SAMPLE_FRACTION} (plan §4.5 hard-pin: "
                f"'每 200 帧抽检 ≥ 20' = 10%); got "
                f"{self.min_sample_fraction}"
            )
        # max_true_positive_missed_rate: bounded ceiling [0, PLAN_TPM_RATE_CEILING].
        if not isinstance(self.max_true_positive_missed_rate, float) or isinstance(
            self.max_true_positive_missed_rate, bool
        ):
            raise ValueError(
                f"max_true_positive_missed_rate must be float; got "
                f"{type(self.max_true_positive_missed_rate).__name__}="
                f"{self.max_true_positive_missed_rate!r}"
            )
        if not math.isfinite(self.max_true_positive_missed_rate):
            raise ValueError(
                f"max_true_positive_missed_rate must be finite; got "
                f"{self.max_true_positive_missed_rate!r}"
            )
        if not (0.0 <= self.max_true_positive_missed_rate <= self._MAX_TPM_RATE_CEILING):
            raise ValueError(
                f"max_true_positive_missed_rate must be in "
                f"[0, {self._MAX_TPM_RATE_CEILING}]; got "
                f"{self.max_true_positive_missed_rate}"
            )
        # map_regression_tolerance_pp: float, bool-exclusion, finite,
        # non-negative, ceiling at PLAN_MAP_TOLERANCE_CEILING_PP.
        if not isinstance(self.map_regression_tolerance_pp, float) or isinstance(
            self.map_regression_tolerance_pp, bool
        ):
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
        if self.map_regression_tolerance_pp > self._MAP_TOLERANCE_CEILING_PP:
            raise ValueError(
                f"map_regression_tolerance_pp must be <= "
                f"{self._MAP_TOLERANCE_CEILING_PP} (the §3.7/§4.7 drop "
                f"threshold for total mAP regression); got "
                f"{self.map_regression_tolerance_pp} (a tolerance > "
                f"{self._MAP_TOLERANCE_CEILING_PP} makes the deploy guard "
                f"broader than the drop catch-all, which is incoherent)"
            )


def load_hard_negative_mining_yaml(path: str | Path) -> HardNegativeMiningYamlConfig:
    """Parse ``configs/hard_negative_mining.yaml`` into a typed config bundle.

    Coercion contract (b-stage MUST honor ALL of these):
      * ``schema_version`` (YAML string) is read verbatim; loader rejects
        any value other than ``"1"`` at the dataclass boundary.
      * ``data_yaml_sha256`` (YAML string) is the SHA256 of the active
        ``data.yaml`` at fit time; the loader does NOT recompute the
        hash (it's authoritative as written into the YAML by the
        fit-time pipeline). Validation: 64-char lowercase hex.
      * ``class_names`` (YAML list[str]) → ``tuple(str(name) for name in raw)``.
      * ``mining_sources`` (YAML list[str]) → ``tuple(str(s) for s in raw)``.
      * ``baseline_weights_path`` / ``output_candidates_path`` /
        ``frozen_manifest_path`` (YAML string or null) → ``Path`` or
        ``None``. Empty string is normalized to ``None``.
      * ``num_classes: null`` raises ``ValueError`` immediately — the
        loader / runner is responsible for setting an explicit value
        before consuming this YAML.
      * ``min_sample_fraction`` must equal 0.10 — plan §4.5 hard-pin;
        any other value raises ``ValueError``.
      * **Legacy key transition (B2 review I5 2026-05-10)**: a YAML
        with the old ``candidates_output_path`` key (no
        ``output_candidates_path``) is read by the loader for one
        version with a ``DeprecationWarning`` issued; a YAML carrying
        BOTH keys raises ``ValueError`` (ambiguous). b-stage drops
        the legacy-key support after R3 cutover.

    Args:
        path: filesystem path to a ``hard_negative_mining.yaml`` file.

    Returns:
        ``HardNegativeMiningYamlConfig`` — fully validated, ready for
        downstream use by the mining / verification / d-stage runner.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: schema / range / enum violations as described above.
    """
    import warnings

    import yaml as _yaml

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    with path.open("r", encoding="utf-8") as fh:
        raw = _yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: top-level YAML must be a mapping; got "
            f"{type(raw).__name__}"
        )

    # Legacy-key transition per B2 review I5 2026-05-10.
    has_new = "output_candidates_path" in raw
    has_legacy = "candidates_output_path" in raw
    if has_new and has_legacy:
        raise ValueError(
            f"{path}: ambiguous — both 'output_candidates_path' and the "
            f"legacy 'candidates_output_path' are present. Drop the legacy "
            f"key; the loader supports the new spelling only post-R3 cutover."
        )
    if has_legacy and not has_new:
        warnings.warn(
            f"{path}: 'candidates_output_path' is deprecated; rename to "
            f"'output_candidates_path' before the R3 cutover.",
            DeprecationWarning,
            stacklevel=2,
        )
        raw["output_candidates_path"] = raw.pop("candidates_output_path")

    def _path_or_none(val: object, field: str) -> Path | None:
        if val is None:
            return None
        if isinstance(val, str):
            if val == "":
                return None
            return Path(val)
        raise ValueError(
            f"{path}: {field} must be string or null; got "
            f"{type(val).__name__}={val!r}"
        )

    def _str_tuple(val: object, field: str, *, allow_none: bool = False) -> tuple[str, ...]:
        # C3 review MAJOR-1 fix: ``mining_sources: null`` or an omitted key
        # is a load-bearing config choice — it must be explicit (empty
        # list ``[]`` for "no mining sources" vs omitted key entirely).
        # Allow ``None`` only for fields where the dataclass explicitly
        # supports it (no current HN field does — see allow_none=False
        # default).
        if val is None:
            if allow_none:
                return ()
            raise ValueError(
                f"{path}: {field} must be an explicit YAML list (use "
                f"``{field}: []`` for empty); got null/omitted. Silent "
                f"acceptance of an omitted key would let a misconfigured "
                f"YAML produce a structurally-valid downstream artifact "
                f"with zero mining sources."
            )
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

    # num_classes: explicit, not null.
    num_classes_raw = raw.get("num_classes", None)
    if num_classes_raw is None:
        raise ValueError(
            f"{path}: num_classes must be set explicitly; got null. The "
            f"loader does not infer it from class_names — the trainer must "
            f"stamp a concrete value."
        )

    # C3 review MINOR-2 fix: data_yaml_sha256 must be string at the YAML
    # layer — silently stringifying ``123`` → ``"123"`` masks a real bug
    # (an int in the source). Class-label provenance hash is load-bearing
    # for cross-artifact integrity; reject malformed types at the source.
    raw_data_sha = raw.get("data_yaml_sha256")
    if not isinstance(raw_data_sha, str):
        raise ValueError(
            f"{path}: data_yaml_sha256 must be a YAML string (64-char "
            f"lowercase hex); got "
            f"{type(raw_data_sha).__name__}={raw_data_sha!r}"
        )
    raw_schema_version = raw.get("schema_version")
    if not isinstance(raw_schema_version, str):
        raise ValueError(
            f"{path}: schema_version must be a YAML string; got "
            f"{type(raw_schema_version).__name__}={raw_schema_version!r}"
        )

    return HardNegativeMiningYamlConfig(
        schema_version=raw_schema_version,
        num_classes=num_classes_raw,
        class_names=_str_tuple(raw.get("class_names"), "class_names"),
        data_yaml_sha256=raw_data_sha,
        baseline_weights_path=_path_or_none(
            raw.get("baseline_weights_path"), "baseline_weights_path"
        ),
        mining_sources=_str_tuple(raw.get("mining_sources"), "mining_sources"),
        output_candidates_path=_path_or_none(
            raw.get("output_candidates_path"), "output_candidates_path"
        ),
        min_sample_fraction=float(raw.get("min_sample_fraction", 0.0))
        if not isinstance(raw.get("min_sample_fraction"), bool)
        else raw["min_sample_fraction"],  # propagate bool so __post_init__ catches it
        max_true_positive_missed_rate=float(
            raw.get("max_true_positive_missed_rate", 0.0)
        )
        if not isinstance(raw.get("max_true_positive_missed_rate"), bool)
        else raw["max_true_positive_missed_rate"],
        frozen_manifest_path=_path_or_none(
            raw.get("frozen_manifest_path"), "frozen_manifest_path"
        ),
        map_regression_tolerance_pp=float(
            raw.get("map_regression_tolerance_pp", 0.0)
        )
        if not isinstance(raw.get("map_regression_tolerance_pp"), bool)
        else raw["map_regression_tolerance_pp"],
    )
