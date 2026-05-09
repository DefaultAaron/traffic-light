"""3-arm ablation aggregator: no_aug vs cp_only vs cp_balanced (β sweep).

Drives the d-stage flow end-to-end:

  0a. Stub-not-populated guard (B2 review I4 2026-05-09): if the
      ``configs/data_R2_class_weights.yaml`` referenced by ``config.weights_yaml``
      ships with ``data_yaml_sha256: ""`` (the un-fitted stub state),
      the runner refuses to proceed — running the ablation against an
      un-populated weights file would silently produce an unweighted
      baseline for every β > 0 cell, corrupting the cp_balanced sweep.
      Note: ``ClassCountsTable.__post_init__`` already rejects empty
      ``data_yaml_sha256`` (length != 64) at construction, so the
      contract is now strictly stronger — the loader fails with a
      structured ValueError before the runner reaches Step 1.

  0b. Cross-artifact config/counts invariants (C3 iter-4 NEW-MAJOR
      2026-05-09): JSON Schema can't express the cross-file equality
      ``config.num_classes == counts.num_classes`` etc., and each
      file's own __post_init__ only sees its half. The runner is the
      ONLY layer that holds both — it MUST hard-fail unless ALL of:

        * ``counts.num_classes == config.num_classes``
        * ``counts.class_names == config.class_names``
        * ``counts.rare_class_threshold == config.rare_class_threshold``

      Without this, a stale class-counts file (e.g. fitted against a
      previous data.yaml with 10 classes) passes its own validation
      but produces a class-weight vector for the wrong class set,
      silently corrupting the ablation arm. b-stage hard-fails with a
      diagnostic naming the mismatched fields.

  1. Load per-arm eval JSONs from already-trained runs:
       * ``no_aug``       — baseline (copy-paste OFF, β=0)
       * ``cp_only``      — copy-paste ON, β=0
       * ``cp_balanced``  — copy-paste ON, β > 0; one eval per β in
                            {0.99, 0.999, 0.9999} (5 eval JSONs total)
     Each eval JSON has shape (C3 iter-7 NEW-MAJOR 2026-05-09:
     ``data_yaml_sha256`` is REQUIRED — without it, two evals can
     ship matching numeric class IDs while the active ``data.yaml``
     has remapped the names underneath, silently corrupting per-class
     AP comparisons. C3 iter-10 NEW-MAJOR 2026-05-09:
     ``map_no_regression`` is now per-arm; the prior global
     ``--eval-metrics-json`` contract is REPLACED — each eval JSON
     carries its own verdict computed by the upstream trainer eval
     pipeline against the same tolerance):
       ``{"per_class_AP": [{"class_id", "class_name", "ap_at_0_5",
                            "full_val_support"}, ...],
          "total_mAP_at_0_5": float,
          "rare_fp_count": int,                # on the §3.7+§4.7 manifest
          "data_yaml_sha256": str,             # 64-char lowercase hex; REQUIRED
          "eval_manifest_sha256": str,
          "fp_manifest_sha256": str,
          "map_no_regression": bool,           # THIS arm vs no_aug; baseline=True trivially
          "map_regression_tolerance_pp": float}``

  2. Verify cross-arm invariants (the runner is the only layer that can):
       * All five evals reference the SAME ``eval_manifest_sha256``.
       * All five evals reference the SAME ``fp_manifest_sha256``.
       * All five evals reference the SAME ``data_yaml_sha256``.
         (C3 iter-7 NEW-MAJOR 2026-05-09: stale class-label remap is
         the high-impact failure mode for a traffic-light detector —
         numeric class IDs can match while the semantic meaning has
         shifted underneath. The runner hard-fails on mismatch.)
       * That ``data_yaml_sha256`` MUST also equal both the active
         ``CopyPasteBalanceYamlConfig`` data_yaml hash AND the
         ``ClassCountsTable.data_yaml_sha256`` recorded in the
         weights file. Three-way equality across config / counts /
         eval is the only way to guarantee the rare population, the
         class-balance weights, and the per-class AP delta are all
         computed against the SAME ``data.yaml``.
       * All five evals' ``per_class_AP`` arrays contain the SAME class
         IDs in the SAME order.
       * All five evals reference the SAME ``map_regression_tolerance_pp``
         (the runner-side knob; per-arm tolerance drift is a contract
         violation).
       * The no_aug eval MUST have ``map_no_regression=True``
         (self-comparison; the runner hard-fails if False).
       * The runner reads ``data_yaml_sha256`` + ``weights_yaml_sha256``
         from the active config + weights file and records both into the
         output for downstream auditability.

  3. Compute the rare-class set FROM THE BASELINE (no_aug) per-class
     supports only — switching the rare set per arm would let an arm
     that depleted a rare class's support appear "deployed" without
     improving it.

  4. Build ``ArmMetrics`` for each non-baseline cell (cp_only +
     3 × cp_balanced) by paired-class diff against the baseline arm.

  5. Apply ``apply_decision_rule`` to each non-baseline cell.

  6. Select the headline cell from CLI flags
     (``--anchor-arm`` + ``--anchor-beta``); copy that cell's metrics
     and decision into the top-level ``headline_*`` fields. Validate
     against ``_copy_paste_decision_schema.json``. Atomic-rename
     ``.tmp`` → ``runs/_copy_paste_decision.json``.

Anchor selection contract:
  * ``--anchor-arm cp_balanced`` requires ``--anchor-beta {0.99|0.999|0.9999}``.
  * ``--anchor-arm cp_only`` ignores ``--anchor-beta`` (forces null).
  * ``--anchor-arm no_aug`` is REJECTED — the no_aug cell has no decision
    (it's the baseline reference); choosing it as the headline would
    serialize a ``decision: "executor_error"`` row at the top level,
    which is meaningless for a deployment-policy artifact.

Sensitivity-sweep contract (plan §3.7 + sister HMM gate alignment):
  * β ∈ {0.99, 0.999, 0.9999} is FIXED (plan-pinned). The YAML's
    ``class_balance.beta`` MUST equal ``--anchor-beta`` (when
    ``--anchor-arm cp_balanced``); ``load_copy_paste_balance_yaml``
    rejects any other value rather than silently widening the sweep.
  * **Anchor exactly-once invariant**: zero or multiple rows with
    ``deploy_anchor=true`` in the cp_balanced sweep MUST raise
    ``ValueError`` BEFORE the output JSON is written. JSON Schema
    cannot express "exactly one of array elements has property X = true";
    the runner is the only layer that can.

Schema-validation contract (mirrors HMM gate iter-2 boundary split):

  * **Per-cell input validation (typed-Python layer)**: the metrics
    feeding ``apply_decision_rule`` arrive as ``ArmMetrics`` and
    ``DecisionInputs`` dataclasses. Missing fields / wrong types fail
    at dataclass construction (``TypeError``); range-and-finiteness
    checks (NaN, FP delta out of range, etc.) fail at the executor's
    "numeric malformed input" gate and surface as
    ``decision: "executor_error"`` rows with diagnostic ``notes``.
    The JSON Schema file at
    ``components/copy_paste_balance/gates/_copy_paste_decision_schema.json``
    is NOT used here — it is for the OUTPUT artifact shape, not the
    per-cell input shape.
  * **Output-artifact validation (jsonschema layer)**: BEFORE atomic-rename
    to ``runs/_copy_paste_decision.json``, the runner validates the
    candidate JSON against the schema via ``jsonschema`` (Draft-7).
    Output validation failure is a HARD error (process exit non-zero),
    not an ``executor_error`` row.

mAP no-regression contract (C3 iter-10 NEW-MAJOR 2026-05-09 — replaces
the prior single-eval-metrics-json contract):
  ``map_no_regression`` is PER-ARM (different cp_balanced β cells can
  have different total_mAP deltas; one global verdict cannot stamp
  every arm). Each of the 5 eval JSONs carries its own verdict
  pre-computed by the trainer's upstream eval pipeline against the
  shared tolerance. The runner verifies cross-arm consistency:
  ``map_regression_tolerance_pp`` is identical across all 5 evals;
  the no_aug eval has ``map_no_regression=True``; each value is
  recorded into the corresponding ArmMetrics cell. b-stage drops
  ``--eval-metrics-json`` from the CLI — the per-arm verdicts arrive
  inside each eval JSON now.

Scope fence reminder: this runner MUST NOT import anything from
``inference.tracker`` or ``inference.cpp`` (parallel to HMM gate's
fence). The arms are post-hoc compared from already-trained eval JSONs;
the runner does NOT drive training.

Scaffold (a-stage): public CLI entry signature only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class AblationConfig:
    """Top-level knobs for one ablation run; derived from CLI + YAML.

    Cross-knob invariants enforced in ``__post_init__`` (B2 review C2
    2026-05-09 — sister scaffolds (HMM ``HmmYamlConfig``) enforce these
    at construction; this dataclass had a documented contract floating
    in module prose without a load-bearing ``__post_init__``).

    Hazards guarded:
      * ``anchor_arm`` outside {"cp_only", "cp_balanced"} — the runner
        prose REJECTS ``"no_aug"`` (the baseline cell carries no
        decision) but the dataclass would accept any string without this.
      * ``anchor_beta`` mismatched with ``anchor_arm`` (required iff
        cp_balanced; must be None for cp_only).
      * ``anchor_beta`` not in the plan-pinned sweep set
        {0.99, 0.999, 0.9999} (with bool-exclusion + finite check).
      * ``cp_balanced_eval_jsons`` / ``cp_balanced_betas`` length drift
        — both must be exactly 3, parallel-indexed.
      * ``cp_balanced_betas`` must equal {0.99, 0.999, 0.9999} as a set
        (covers the plan-pinned sweep set without forcing a particular
        order).
      * Path-vs-str type drift on every ``Path``-typed field (mirrors
        ``HmmYamlConfig.transition_matrix_path`` C3 stop-hook iter-2
        NEW-MINOR fix — without this, a CLI parser that forgets
        ``Path(raw)`` ships a raw str downstream).

    Filesystem-state checks (mirrors HMM iter-2 NEW-MINOR 5): existence
    / readability are NOT checked here. The runner enforces filesystem
    state when it consumes the path; missing paths surface as
    diagnostic ``decision: "executor_error"`` rows.
    """

    no_aug_eval_json: Path                      # baseline arm eval output (carries map_no_regression=True)
    cp_only_eval_json: Path                     # copy-paste arm eval output (carries its own map_no_regression)
    cp_balanced_eval_jsons: tuple[Path, ...]    # one per β; len == 3; each carries its own map_no_regression
    cp_balanced_betas: tuple[float, ...]        # parallel to cp_balanced_eval_jsons
    config_yaml: Path                           # configs/copy_paste_balance.yaml
    weights_yaml: Path                          # configs/data_R2_class_weights.yaml
    output_json: Path                           # runs/_copy_paste_decision.json
    anchor_arm: str                             # "cp_only" or "cp_balanced"
    anchor_beta: float | None                   # required iff anchor_arm == cp_balanced

    _ALLOWED_ANCHOR_ARMS: ClassVar[tuple[str, ...]] = ("cp_only", "cp_balanced")
    _ALLOWED_ANCHOR_BETAS: ClassVar[tuple[float, ...]] = (0.99, 0.999, 0.9999)
    _CP_BALANCED_SWEEP_LEN: ClassVar[int] = 3

    def __post_init__(self) -> None:
        # Path-vs-str guard on every Path field. Loader / CLI parser is
        # responsible for str→Path coercion; without this, a forgotten
        # Path() wrap surfaces later as a confusing AttributeError on a
        # str value rather than a structured ValueError at construction.
        path_fields = (
            ("no_aug_eval_json", self.no_aug_eval_json),
            ("cp_only_eval_json", self.cp_only_eval_json),
            ("config_yaml", self.config_yaml),
            ("weights_yaml", self.weights_yaml),
            ("output_json", self.output_json),
        )
        for name, value in path_fields:
            if not isinstance(value, Path):
                raise ValueError(
                    f"{name} must be Path; got "
                    f"{type(value).__name__}={value!r} "
                    f"(loader / CLI must coerce str → pathlib.Path before construction)"
                )
        # cp_balanced_eval_jsons: tuple of Path, length == 3.
        if not isinstance(self.cp_balanced_eval_jsons, tuple):
            raise ValueError(
                f"cp_balanced_eval_jsons must be tuple; got "
                f"{type(self.cp_balanced_eval_jsons).__name__}"
            )
        if len(self.cp_balanced_eval_jsons) != self._CP_BALANCED_SWEEP_LEN:
            raise ValueError(
                f"cp_balanced_eval_jsons must have exactly "
                f"{self._CP_BALANCED_SWEEP_LEN} entries (one per β in the "
                f"plan §3.7 sweep); got {len(self.cp_balanced_eval_jsons)}"
            )
        for i, p in enumerate(self.cp_balanced_eval_jsons):
            if not isinstance(p, Path):
                raise ValueError(
                    f"cp_balanced_eval_jsons[{i}] must be Path; got "
                    f"{type(p).__name__}={p!r}"
                )
        # cp_balanced_betas: tuple of float, length == 3, set equality with
        # the plan-pinned sweep. Bool-exclusion before the membership test
        # mirrors the laplace_alpha / class_balance_beta hard-pin pattern.
        if not isinstance(self.cp_balanced_betas, tuple):
            raise ValueError(
                f"cp_balanced_betas must be tuple; got "
                f"{type(self.cp_balanced_betas).__name__}"
            )
        if len(self.cp_balanced_betas) != self._CP_BALANCED_SWEEP_LEN:
            raise ValueError(
                f"cp_balanced_betas must have exactly "
                f"{self._CP_BALANCED_SWEEP_LEN} entries; got "
                f"{len(self.cp_balanced_betas)}"
            )
        for i, b in enumerate(self.cp_balanced_betas):
            if not isinstance(b, float) or isinstance(b, bool):
                raise ValueError(
                    f"cp_balanced_betas[{i}] must be float; got "
                    f"{type(b).__name__}={b!r}"
                )
            if not math.isfinite(b):
                raise ValueError(
                    f"cp_balanced_betas[{i}] must be finite; got {b!r}"
                )
        if set(self.cp_balanced_betas) != set(self._ALLOWED_ANCHOR_BETAS):
            raise ValueError(
                f"cp_balanced_betas must equal {set(self._ALLOWED_ANCHOR_BETAS)} "
                f"as a set (plan §3.7 sweep set); got "
                f"{self.cp_balanced_betas}"
            )
        # anchor_arm: string in {"cp_only", "cp_balanced"}. Module prose
        # explicitly REJECTS "no_aug".
        if not isinstance(self.anchor_arm, str):
            raise ValueError(
                f"anchor_arm must be str; got "
                f"{type(self.anchor_arm).__name__}={self.anchor_arm!r}"
            )
        if self.anchor_arm not in self._ALLOWED_ANCHOR_ARMS:
            raise ValueError(
                f"anchor_arm must be one of {self._ALLOWED_ANCHOR_ARMS}; "
                f"got {self.anchor_arm!r} "
                f"(no_aug is REJECTED — the baseline cell carries no decision)"
            )
        # anchor_beta: required iff anchor_arm == cp_balanced; must match
        # one of the sweep values when set.
        if self.anchor_arm == "cp_balanced":
            if self.anchor_beta is None:
                raise ValueError(
                    "anchor_beta is required when anchor_arm == 'cp_balanced'; "
                    "got None"
                )
            if not isinstance(self.anchor_beta, float) or isinstance(self.anchor_beta, bool):
                raise ValueError(
                    f"anchor_beta must be float; got "
                    f"{type(self.anchor_beta).__name__}={self.anchor_beta!r}"
                )
            if not math.isfinite(self.anchor_beta):
                raise ValueError(
                    f"anchor_beta must be finite; got {self.anchor_beta!r}"
                )
            if self.anchor_beta not in self._ALLOWED_ANCHOR_BETAS:
                raise ValueError(
                    f"anchor_beta must be one of {self._ALLOWED_ANCHOR_BETAS}; "
                    f"got {self.anchor_beta}"
                )
        else:
            # anchor_arm == "cp_only" — anchor_beta MUST be None (the
            # cp_only cell has no β; module prose forces null in headline_beta).
            if self.anchor_beta is not None:
                raise ValueError(
                    f"anchor_beta must be None when anchor_arm != 'cp_balanced'; "
                    f"got {self.anchor_beta!r}"
                )


def run_ablation(config: AblationConfig) -> None:
    """End-to-end ablation aggregator: read eval JSONs → metrics → decision.

    Side effects:
        * Reads all five eval JSONs.
        * Reads ``config.config_yaml`` + ``config.weights_yaml``; records
          both SHA256 hashes in the output.
        * mAP no-regression verdict is per-arm (read from each eval
          JSON's ``map_no_regression`` field; C3 iter-10 NEW-MAJOR
          2026-05-09 replaces the prior single eval-metrics-json contract).
        * Validates the cross-arm manifest invariants (all evals share
          the same eval + fp manifest hash; same class-ID set).
        * Applies the decision rule to each non-baseline cell.
        * Validates the candidate output against the JSON Schema.
        * Atomically renames ``output_json.tmp`` → ``output_json``.

    Args:
        config: ``AblationConfig`` resolved from CLI + YAML.

    Raises:
        ValueError: schema / config violations from the underlying modules.
        FileNotFoundError: any input path missing.
        NotImplementedError: a-stage scaffold.
    """
    raise NotImplementedError("b-stage")


def main() -> int:
    """CLI entry: parse args → build ``AblationConfig`` → ``run_ablation``.

    Argparse contract (b-stage; C3 iter-10 NEW-MAJOR 2026-05-09:
    --eval-metrics-json REMOVED — verdicts now arrive per-arm inside
    each eval JSON):
        --no-aug-eval FILE       baseline arm's eval JSON (must carry
                                  map_no_regression=True)
        --cp-only-eval FILE      copy-paste arm's eval JSON
        --cp-balanced-eval FILE  one per β; pass 3 times in β order
        --config FILE            configs/copy_paste_balance.yaml path
        --weights FILE           configs/data_R2_class_weights.yaml path
        --output FILE            runs/_copy_paste_decision.json path
        --anchor-arm {cp_only,cp_balanced}
                                  which arm populates headline_*; no_aug
                                  is REJECTED (no decision row exists).
        --anchor-beta FLOAT      required iff anchor-arm == cp_balanced;
                                  must match one of {0.99, 0.999, 0.9999}.

    Returns:
        process exit code: 0 on AGREED-CLEAN run, 1 on
        ``executor_error`` in the headline row, 2 on
        ``NotImplementedError`` from any a-stage stub (so scaffold-time
        smoke tests don't silently exit 0).

    Raises:
        NotImplementedError: a-stage scaffold.
    """
    raise NotImplementedError("b-stage")


def assert_artifact_invariants(artifact: dict) -> None:
    """Cross-row invariants that JSON Schema cannot express.

    C3 iter-2 NEW-MAJOR 3 2026-05-09: the schema's ``allOf+contains``
    pattern guarantees the cp_balanced sweep has exactly 3 rows over
    β ∈ {0.99, 0.999, 0.9999}, but cannot express:

      1. Exactly one sweep row sets ``deploy_anchor=true``.
      2. ``cp_balanced.deploy_anchor_beta`` matches the
         ``beta`` value of the unique anchor row.
      3. When ``headline_arm == "cp_balanced"``, ``headline_beta``,
         ``headline_metrics``, and ``headline_decision`` match the
         anchor row's corresponding fields.
      4. When ``headline_arm == "cp_only"``, ``headline_metrics``
         matches ``cp_only.metrics`` and ``headline_decision``
         matches ``cp_only.decision``.

    C3 iter-5 NEW-MAJOR 2026-05-09: extended to enforce per-block
    rare-set subset invariants:

      5. For EVERY MetricsBlock in the artifact (``headline_metrics``,
         ``no_aug.metrics``, ``cp_only.metrics``, every
         ``cp_balanced.sensitivity_sweep[*].metrics``):
           * ``set(rare_safety_class_ids) ⊆ set(rare_class_ids)``
           * ``set(zero_support_rare_classes) ⊆ set(rare_class_ids)``
         The dataclass ``ArmMetrics.__post_init__`` enforces these on
         in-memory construction, but JSON Schema can't express
         dynamic array-subset checks without ``$data`` (a non-Draft-7
         extension), so the runner is the load-bearing layer.

      6. C3 iter-10 NEW-MAJOR 2026-05-09: decision-vs-verdict
         consistency. Per plan §3.7, both deploy and defer require
         total mAP no-regression. A cell with
         ``decision in {"deploy", "defer"}`` AND
         ``metrics.map_no_regression == False`` is a malformed
         decision artifact. Enforced for every candidate cell
         (cp_only + 3 cp_balanced sweep rows + headline). The
         baseline cell has no decision so doesn't apply.

    This validator is implemented (not stubbed) because cross-row
    invariants are pure dict→raise logic — no I/O, no infrastructure
    — and a stub would defeat the whole point of the iter-2 finding
    (the runner would atomic-rename a malformed artifact). The
    ``run_ablation`` body remains a stub; b-stage MUST call this
    helper AFTER the schema validator passes and BEFORE the atomic
    rename.

    Args:
        artifact: the candidate output dict, post-aggregation,
            post-schema-validation.

    Raises:
        ValueError: any cross-row invariant fails.
    """
    # Per-block rare-set subset invariants (iter-5 NEW-MAJOR). Walk every
    # MetricsBlock in the artifact and verify rare_safety / zero_support
    # are subsets of rare_class_ids — schema can express uniqueness within
    # one array but not subset across arrays in the same object.
    metrics_blocks = [
        ("headline_metrics", artifact["headline_metrics"]),
        ("no_aug.metrics", artifact["no_aug"]["metrics"]),
        ("cp_only.metrics", artifact["cp_only"]["metrics"]),
    ]
    for i, row in enumerate(artifact["cp_balanced"]["sensitivity_sweep"]):
        metrics_blocks.append(
            (f"cp_balanced.sensitivity_sweep[{i}].metrics", row["metrics"])
        )
    for path, metrics in metrics_blocks:
        rare = set(metrics["rare_class_ids"])
        rare_safety = set(metrics["rare_safety_class_ids"])
        zero_support = set(metrics["zero_support_rare_classes"])
        if not rare_safety.issubset(rare):
            raise ValueError(
                f"{path}.rare_safety_class_ids must be a subset of "
                f"rare_class_ids; got rare_safety={sorted(rare_safety)}, "
                f"rare={sorted(rare)} (the rare-safety guard is reported "
                f"against classes outside the rare population — malformed "
                f"decision artifact)"
            )
        if not zero_support.issubset(rare):
            raise ValueError(
                f"{path}.zero_support_rare_classes must be a subset of "
                f"rare_class_ids; got zero_support={sorted(zero_support)}, "
                f"rare={sorted(rare)} (zero-support exclusions are reported "
                f"against classes outside the rare population — malformed "
                f"decision artifact)"
            )
    # C3 iter-7 NEW-MAJOR (cross-block rare-set equality) 2026-05-09:
    # the §3.7 decision rule is built on a SINGLE rare population
    # derived from the no_aug arm and propagated to every candidate
    # cell. A hand-built artifact with different rare sets per cell
    # would corrupt the deploy/defer/drop interpretation across
    # arms (each cell would be reporting against a different rare
    # population). DecisionInputs enforces this for the in-memory
    # baseline+candidate pair, but NOT across all 6 cells. The runner
    # is the only layer that holds all six. Take no_aug.metrics as
    # canonical and require every block's three rare-set fields to
    # match exactly.
    canonical_rare = tuple(artifact["no_aug"]["metrics"]["rare_class_ids"])
    canonical_rare_safety = tuple(
        artifact["no_aug"]["metrics"]["rare_safety_class_ids"]
    )
    canonical_zero_support = tuple(
        artifact["no_aug"]["metrics"]["zero_support_rare_classes"]
    )
    # C3 iter-8 NEW-MAJOR (cross-block manifest equality) 2026-05-09:
    # extend the canonical-vs-cell pattern to manifest hashes. The
    # schema's 64-hex per-cell pattern doesn't enforce equality across
    # cells — a hand-built artifact could pair AP deltas computed
    # against different val manifests, silently corrupting the
    # deploy/defer/drop comparison. Same hazard class as the rare-set
    # fields; same canonical-derived-from-no_aug fix.
    canonical_eval_manifest = artifact["no_aug"]["metrics"]["eval_manifest_sha256"]
    canonical_fp_manifest = artifact["no_aug"]["metrics"]["fp_manifest_sha256"]
    # C3 iter-9 NEW-MAJOR 2026-05-09: cross-block equality also covers the
    # new decision-provenance fields. data_yaml_sha256 must additionally
    # match the artifact's top-level data_yaml_sha256 (which is the
    # config-level hash; three-way equality across config / per-cell /
    # top-level closes the class-label drift gap).
    canonical_data_yaml = artifact["no_aug"]["metrics"]["data_yaml_sha256"]
    canonical_tolerance = artifact["no_aug"]["metrics"]["map_regression_tolerance_pp"]
    if canonical_data_yaml != artifact["data_yaml_sha256"]:
        raise ValueError(
            f"no_aug.metrics.data_yaml_sha256 ({canonical_data_yaml}) must "
            f"equal artifact.data_yaml_sha256 ({artifact['data_yaml_sha256']}) "
            f"— top-level config hash and per-cell eval hash must agree, else "
            f"the per-class AP delta is computed against a different class set "
            f"than the rare population was derived from"
        )
    for path, metrics in metrics_blocks:
        if path == "no_aug.metrics":
            continue
        if tuple(metrics["rare_class_ids"]) != canonical_rare:
            raise ValueError(
                f"{path}.rare_class_ids must match no_aug.metrics.rare_class_ids "
                f"(canonical={canonical_rare}); got "
                f"{tuple(metrics['rare_class_ids'])} (rare population is "
                f"baseline-derived and propagated to every cell; per-cell "
                f"drift corrupts the rule's deploy/defer/drop interpretation)"
            )
        if tuple(metrics["rare_safety_class_ids"]) != canonical_rare_safety:
            raise ValueError(
                f"{path}.rare_safety_class_ids must match "
                f"no_aug.metrics.rare_safety_class_ids "
                f"(canonical={canonical_rare_safety}); got "
                f"{tuple(metrics['rare_safety_class_ids'])}"
            )
        if tuple(metrics["zero_support_rare_classes"]) != canonical_zero_support:
            raise ValueError(
                f"{path}.zero_support_rare_classes must match "
                f"no_aug.metrics.zero_support_rare_classes "
                f"(canonical={canonical_zero_support}); got "
                f"{tuple(metrics['zero_support_rare_classes'])}"
            )
        if metrics["eval_manifest_sha256"] != canonical_eval_manifest:
            raise ValueError(
                f"{path}.eval_manifest_sha256 must match "
                f"no_aug.metrics.eval_manifest_sha256 "
                f"(canonical={canonical_eval_manifest}); got "
                f"{metrics['eval_manifest_sha256']} (every cell must consume "
                f"the SAME frozen val manifest — manifest drift across cells "
                f"corrupts the AP-delta comparison the rule operates on)"
            )
        if metrics["fp_manifest_sha256"] != canonical_fp_manifest:
            raise ValueError(
                f"{path}.fp_manifest_sha256 must match "
                f"no_aug.metrics.fp_manifest_sha256 "
                f"(canonical={canonical_fp_manifest}); got "
                f"{metrics['fp_manifest_sha256']}"
            )
        if metrics["data_yaml_sha256"] != canonical_data_yaml:
            raise ValueError(
                f"{path}.data_yaml_sha256 must match "
                f"no_aug.metrics.data_yaml_sha256 "
                f"(canonical={canonical_data_yaml}); got "
                f"{metrics['data_yaml_sha256']} (class-label drift across "
                f"cells silently corrupts per-class AP delta interpretation)"
            )
        if metrics["map_regression_tolerance_pp"] != canonical_tolerance:
            raise ValueError(
                f"{path}.map_regression_tolerance_pp must match "
                f"no_aug.metrics.map_regression_tolerance_pp "
                f"(canonical={canonical_tolerance}); got "
                f"{metrics['map_regression_tolerance_pp']} (tolerance is a "
                f"runner-side knob; the same value must stamp every cell)"
            )
    # C3 iter-10 NEW-MAJOR 2026-05-09: decision-vs-verdict consistency.
    # Per plan §3.7, both deploy and defer require total mAP no-regression.
    # A candidate cell with decision in {deploy, defer} AND
    # metrics.map_no_regression == False is a malformed artifact.
    decision_map_check_sites = [
        ("headline", artifact["headline_decision"], artifact["headline_metrics"]),
        ("cp_only", artifact["cp_only"]["decision"], artifact["cp_only"]["metrics"]),
    ]
    for i, row in enumerate(artifact["cp_balanced"]["sensitivity_sweep"]):
        decision_map_check_sites.append(
            (f"cp_balanced.sensitivity_sweep[{i}]", row["decision"], row["metrics"])
        )
    for site, decision, metrics in decision_map_check_sites:
        if decision in ("deploy", "defer") and metrics["map_no_regression"] is not True:
            raise ValueError(
                f"{site}.decision={decision!r} is incompatible with "
                f"metrics.map_no_regression={metrics['map_no_regression']!r} — "
                f"plan §3.7 requires total mAP no-regression for both deploy "
                f"and defer; a cell that regressed total mAP MUST land on "
                f"drop, not deploy or defer (malformed decision artifact)"
            )

    sweep = artifact["cp_balanced"]["sensitivity_sweep"]
    anchor_rows = [row for row in sweep if row.get("deploy_anchor")]
    if len(anchor_rows) != 1:
        raise ValueError(
            f"cp_balanced.sensitivity_sweep must have exactly one row with "
            f"deploy_anchor=true; got {len(anchor_rows)}"
        )
    anchor = anchor_rows[0]
    deploy_anchor_beta = artifact["cp_balanced"]["deploy_anchor_beta"]
    if anchor["beta"] != deploy_anchor_beta:
        raise ValueError(
            f"cp_balanced.deploy_anchor_beta ({deploy_anchor_beta}) must equal "
            f"the unique anchor sweep row's beta ({anchor['beta']})"
        )
    headline_arm = artifact["headline_arm"]
    if headline_arm == "cp_balanced":
        if artifact["headline_beta"] != anchor["beta"]:
            raise ValueError(
                f"headline_beta ({artifact['headline_beta']}) must equal "
                f"the anchor sweep row's beta ({anchor['beta']}) when "
                f"headline_arm == 'cp_balanced'"
            )
        if artifact["headline_metrics"] != anchor["metrics"]:
            raise ValueError(
                "headline_metrics must equal the anchor sweep row's metrics "
                "when headline_arm == 'cp_balanced' (the headline is a copy "
                "of the anchor cell, not an independent computation)"
            )
        if artifact["headline_decision"] != anchor["decision"]:
            raise ValueError(
                f"headline_decision ({artifact['headline_decision']}) must "
                f"equal the anchor sweep row's decision ({anchor['decision']}) "
                f"when headline_arm == 'cp_balanced'"
            )
    elif headline_arm == "cp_only":
        cp_only = artifact["cp_only"]
        if artifact["headline_metrics"] != cp_only["metrics"]:
            raise ValueError(
                "headline_metrics must equal cp_only.metrics when "
                "headline_arm == 'cp_only'"
            )
        if artifact["headline_decision"] != cp_only["decision"]:
            raise ValueError(
                f"headline_decision ({artifact['headline_decision']}) must "
                f"equal cp_only.decision ({cp_only['decision']}) when "
                f"headline_arm == 'cp_only'"
            )
        # headline_arm == "cp_only" requires headline_beta is None;
        # schema's if/then enforces this, but cross-check at the runner
        # boundary too.
        if artifact["headline_beta"] is not None:
            raise ValueError(
                "headline_beta must be null when headline_arm == 'cp_only'"
            )
    # headline_arm cannot be "no_aug" — the schema's enum already rejects
    # that, but the dict could conceivably arrive here pre-schema; assert
    # defensively.
    else:
        raise ValueError(
            f"headline_arm must be 'cp_only' or 'cp_balanced'; got {headline_arm!r}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
