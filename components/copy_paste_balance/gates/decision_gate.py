"""d-stage decision rule for the copy-paste / class-balance ablation.

Mirrors ``components/hmm_smoother/gates/decision_gate.py`` (locked iter-3,
2026-05-09): pre-committed thresholds frozen as ClassVars on
``DecisionInputs``, first-match cascade with explicit catch-all,
``executor_error`` reserved for malformed inputs only.

Plan §3.7 cases (verbatim — do NOT widen thresholds during impl, do NOT
silently re-order; threshold knobs may be moved into
``configs/copy_paste_balance.yaml`` only after a fresh adversarial loop):

  * deploy : (rare-class avg ΔAP ≥ +5pp OR any single rare class ΔAP ≥ +5pp)
              AND every rare safety class ΔAP ≥ −1pp
              AND rare-related FP rise ≤ 10%
              AND total mAP no-regression (Δ ≥ −map_regression_tolerance_pp,
                  default 0.2pp).

  * defer  : rare-class avg ΔAP < 2pp (no significant improvement)
              AND total mAP no-regression
              AND no drop trigger fires.
              Action: re-evaluate with more aggressive copy-paste in R3
              (plan §3.7 defer prose). Sweep-cell records a defer reason
              in ``notes``.

  * drop   : catch-all. Any legitimate input that does NOT match deploy
              or defer falls into drop. This explicitly includes the
              plan's literal drop triggers:
                - total mAP regression > 0.5pp
                - any rare safety class regression > 1pp
                - rare-related FP rise > 30%
              AND the middle-case (rare avg 2-5pp improvement that fails
              one of deploy's guards but doesn't qualify for defer's
              "<2pp" framing). The middle case is captured here BY
              DESIGN: the cascade order is deploy → defer → drop, and a
              middle-case row that doesn't meet deploy's full guard set
              and doesn't fit defer's "<2pp" prose is not a legitimate
              ship — it needs investigation. The runner records the
              triggering condition in ``notes`` so reviewers can
              distinguish "literal drop" from "middle-case drop".

Boundary semantics (mirrors HMM gate; plan §3.7 prose uses strict
greater-than for every drop trigger and ≥/≤ for deploy guards):
  * rare avg ΔAP = 5pp                     → deploy guard PASSES (matches "≥ +5pp")
  * rare avg ΔAP = 2pp                     → drop (defer requires "< 2pp", strict)
  * rare safety min ΔAP = −1pp             → deploy guard PASSES (matches "≥ −1pp")
  * rare safety min ΔAP = −1.0001pp        → deploy guard FAILS
  * rare safety min ΔAP = −1pp exactly     → drop trigger does NOT fire
                                              (drop requires "regress > 1pp", strict)
  * rare safety min ΔAP = −1.0001pp        → drop trigger FIRES
  * rare-related FP rise = 0.10            → deploy guard PASSES (matches "≤ 10%")
  * rare-related FP rise = 0.30            → drop trigger does NOT fire
                                              (drop requires "> 30%", strict)
  * rare-related FP rise = 0.3001          → drop trigger FIRES
  * total mAP delta = −0.2pp (default tol) → deploy guard PASSES (matches "≥ −0.2pp")
  * total mAP delta = −0.5pp               → drop trigger does NOT fire
                                              (drop requires "regress > 0.5pp", strict)
  * total mAP delta = −0.5001pp            → drop trigger FIRES

C3 iter-4 NEW-MAJOR (boundary-doc fix) 2026-05-09: earlier prose
contradicted itself by saying ``= 0.30 → drop PASSES`` while the same
trigger text said ``> 0.30 fires drop``. Plan §3.7 uses strict
greater-than throughout the drop conditions, and the deploy guards
use non-strict ≥ / ≤. Mid-cases that don't match deploy AND don't
trigger drop fall into drop via the cascade catch-all (e.g.
``total_map_delta_pp = −0.4pp`` is past the deploy tolerance but not
past the drop threshold — drop catches it as legitimate-but-not-
shippable, with notes recording the non-deploy reason).

Executor semantics: cases evaluated in order ``deploy → defer → drop``;
first match wins. ``drop`` is the unconditional catch-all for legitimate
input; ``executor_error`` is reserved for **malformed input only**:

  * NaN / inf in any metric
  * Class-set mismatch between baseline and candidate (the runner builds
    ``ArmMetrics`` from baseline-derived rare set, so this is a contract
    violation — likely a pairing bug)
  * ``eval_manifest_sha256`` differs between baseline and candidate
  * ``fp_manifest_sha256`` differs between baseline and candidate
  * ``baseline.is_baseline_reference is False`` (caller passed a
    non-baseline arm as the baseline)
  * Threshold ranges out of bounds (FP fraction outside [-1, 10],
    ΔAP_pp outside [-100, 100])

Schema for ``runs/_copy_paste_decision.json`` (plan §3.7 d output):

    {
      "schema_version": "1",
      "config_yaml_sha256": str,            # source-of-truth hash
      "data_yaml_sha256": str,              # rare-class derivation source
      "weights_yaml_sha256": str,           # configs/data_R2_class_weights.yaml hash
      "headline_arm": "cp_balanced" | "cp_only",   # no_aug REJECTED
      "headline_beta": float | null,        # null when arm == cp_only
      "headline_metrics": MetricsBlock,
      "headline_decision": "deploy" | "defer" | "drop" | "executor_error",
      "notes": str,
      "no_aug": {                           # baseline reference; no decision
        "metrics": MetricsBlock,
        "notes": str
      },
      "cp_only": {
        "metrics": MetricsBlock,
        "decision": "deploy" | "defer" | "drop" | "executor_error",
        "notes": str
      },
      "cp_balanced": {
        "deploy_anchor_beta": float,        # which β populates headline_*
        "sensitivity_sweep": [
          {
            "beta": float,                  # one of {0.99, 0.999, 0.9999}
            "deploy_anchor": bool,          # exactly one row is true
            "metrics": MetricsBlock,
            "decision": "deploy" | "defer" | "drop" | "executor_error",
            "notes": str
          },
          ...
        ]
      }
    }

The cp_balanced ``sensitivity_sweep`` block carries one row per
``beta ∈ {0.99, 0.999, 0.9999}``. β=0 is NOT in this sweep — it's the
cp_only arm. Anchor selection is human-driven (b-stage exposes
``--anchor-arm {cp_only,cp_balanced} --anchor-beta FLOAT``); auto-pick
is explicitly out of scope so b-stage doesn't silently install a
ship-decision policy.

Sister-file alignment: shape modeled on
``components/hmm_smoother/gates/_hmm_decision_schema.json`` and
``runs/_kd_decisions.json`` — ``schema_version`` + ``headline_*`` +
per-arm cells + cp_balanced sensitivity sweep.

Validation boundary (mirrors HMM gate iter-3 contract): the JSON Schema
file at ``components/copy_paste_balance/gates/_copy_paste_decision_schema.json``
is **OUTPUT-ONLY**. Per-cell input validation happens at dataclass
construction (``ArmMetrics``, ``DecisionInputs``); numeric malformed
values surface as ``executor_error`` rows with diagnostic notes. Output-
validation failure is a hard error (non-zero exit).

Scaffold (a-stage): API only; thresholds frozen as ClassVars.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from components.copy_paste_balance.gates.ablation_gate import ArmMetrics


class CopyPasteDecision(str, Enum):
    DEPLOY = "deploy"
    DEFER = "defer"
    DROP = "drop"
    EXECUTOR_ERROR = "executor_error"


class ArmId(str, Enum):
    """The three §3.8 c arms."""

    NO_AUG = "no_aug"
    CP_ONLY = "cp_only"
    CP_BALANCED = "cp_balanced"


@dataclass(frozen=True)
class DecisionInputs:
    """One arm's metrics + the upstream mAP-no-regression verdict.

    Plan §3.7 thresholds are pinned as ClassVars on this dataclass so
    they're co-located with the cases that consume them — moving any of
    them into the YAML requires re-running the §三 adversarial loop.

    The mAP-no-regression verdict is a SEPARATE input from the
    ``total_map_delta_pp`` field on ``ArmMetrics`` because the runner is
    responsible for reconciling the trained eval JSON against the
    frozen ``runs/_r2_val_manifest`` (hash check + tolerance lookup).
    The gate trusts the verdict; the runner builds it.
    """

    baseline: ArmMetrics                       # the no_aug arm
    candidate: ArmMetrics                      # cp_only or cp_balanced cell
    map_no_regression: bool                    # upstream verdict
    map_regression_tolerance_pp: float         # cross-check field; default 0.2

    # Plan §3.7 deploy guards ------------------------------------------------
    DEPLOY_RARE_AVG_DELTA_AP_PP: ClassVar[float] = 5.0
    DEPLOY_ANY_RARE_DELTA_AP_PP: ClassVar[float] = 5.0
    DEPLOY_RARE_SAFETY_MIN_DELTA_AP_PP: ClassVar[float] = -1.0
    DEPLOY_RARE_FP_RISE_MAX_FRAC: ClassVar[float] = 0.10

    # Plan §3.7 defer condition ----------------------------------------------
    DEFER_RARE_AVG_LT_PP: ClassVar[float] = 2.0

    # Plan §3.7 drop triggers ------------------------------------------------
    DROP_TOTAL_MAP_REGRESSION_PP: ClassVar[float] = 0.5
    DROP_RARE_SAFETY_REGRESSION_PP: ClassVar[float] = 1.0
    DROP_RARE_FP_RISE_FRAC: ClassVar[float] = 0.30

    def __post_init__(self) -> None:
        # C3 NEW-MAJOR 2 2026-05-09: same boundary discipline as
        # ArmMetrics. Without these, ``map_no_regression="false"`` (a
        # truthy string) and ``map_regression_tolerance_pp=NaN``
        # silently bypass the rule and corrupt the deploy gate.
        if not isinstance(self.baseline, ArmMetrics):
            raise ValueError(
                f"baseline must be ArmMetrics; got "
                f"{type(self.baseline).__name__}"
            )
        if not isinstance(self.candidate, ArmMetrics):
            raise ValueError(
                f"candidate must be ArmMetrics; got "
                f"{type(self.candidate).__name__}"
            )
        if not self.baseline.is_baseline_reference:
            raise ValueError(
                "baseline.is_baseline_reference must be True (the rule is "
                "defined as candidate vs the no_aug baseline arm only)"
            )
        if self.candidate.is_baseline_reference:
            raise ValueError(
                "candidate.is_baseline_reference must be False (the rule is "
                "undefined on a self-comparison; pass a cp_only or "
                "cp_balanced ArmMetrics as the candidate)"
            )
        # Manifest-hash equality between sides — the runner enforces same
        # eval / fp manifest across all 5 evals; an in-memory cell with
        # mismatched hashes is a construction bug, not a legitimate input.
        if self.baseline.eval_manifest_sha256 != self.candidate.eval_manifest_sha256:
            raise ValueError(
                f"baseline.eval_manifest_sha256 != candidate.eval_manifest_sha256 "
                f"({self.baseline.eval_manifest_sha256} vs "
                f"{self.candidate.eval_manifest_sha256})"
            )
        if self.baseline.fp_manifest_sha256 != self.candidate.fp_manifest_sha256:
            raise ValueError(
                f"baseline.fp_manifest_sha256 != candidate.fp_manifest_sha256"
            )
        # Rare-set equality: the runner derives the rare set from the
        # baseline arm only and propagates it to every candidate cell;
        # mismatch indicates a construction bug (the candidate was
        # built against a different rare set than the baseline).
        if self.baseline.rare_class_ids != self.candidate.rare_class_ids:
            raise ValueError(
                "baseline.rare_class_ids != candidate.rare_class_ids "
                "(rare set is plan-required to be derived from the baseline "
                "arm and propagated to every candidate)"
            )
        if self.baseline.rare_safety_class_ids != self.candidate.rare_safety_class_ids:
            raise ValueError(
                "baseline.rare_safety_class_ids != candidate.rare_safety_class_ids"
            )
        # C3 iter-2 NEW-MAJOR (zero_support equality) 2026-05-09: same
        # cross-side equality discipline as the rare-set fields.
        # Zero-support rare classes are excluded from the rare mean/min
        # computations; mismatched exclusions between baseline and
        # candidate would let the decision rule operate on different
        # effective rare populations while passing the rare_class_ids /
        # rare_safety_class_ids guards. Silent decision corruption.
        if self.baseline.zero_support_rare_classes != self.candidate.zero_support_rare_classes:
            raise ValueError(
                "baseline.zero_support_rare_classes != candidate.zero_support_rare_classes "
                "(zero-support exclusions are derived from baseline supports and "
                "must propagate identically to every candidate cell; mismatch "
                "would change the effective rare population the rule operates on)"
            )
        # C3 iter-9 NEW-MAJOR 2026-05-09: cross-side equality on the new
        # decision-provenance fields landed on ArmMetrics. tolerance and
        # data_yaml_sha256 are the SAME across both sides (runner-side
        # constants); map_no_regression is per-arm (baseline trivially
        # True; candidate is the upstream verdict for THIS arm vs baseline).
        if self.baseline.map_regression_tolerance_pp != self.candidate.map_regression_tolerance_pp:
            raise ValueError(
                f"baseline.map_regression_tolerance_pp != "
                f"candidate.map_regression_tolerance_pp "
                f"({self.baseline.map_regression_tolerance_pp} vs "
                f"{self.candidate.map_regression_tolerance_pp}) — tolerance is "
                f"a runner-side knob; the SAME value must propagate to every cell"
            )
        if self.baseline.data_yaml_sha256 != self.candidate.data_yaml_sha256:
            raise ValueError(
                f"baseline.data_yaml_sha256 != candidate.data_yaml_sha256 "
                f"({self.baseline.data_yaml_sha256} vs "
                f"{self.candidate.data_yaml_sha256}) — class-label drift across "
                f"arms silently corrupts per-class AP delta interpretation"
            )
        # DecisionInputs's own map_no_regression / map_regression_tolerance_pp
        # are now redundant with ArmMetrics's per-cell copies; cross-check
        # they match the candidate's view so the runner can't pass
        # inconsistent values.
        if self.map_no_regression != self.candidate.map_no_regression:
            raise ValueError(
                f"DecisionInputs.map_no_regression ({self.map_no_regression}) "
                f"must equal candidate.map_no_regression "
                f"({self.candidate.map_no_regression}) — the upstream verdict "
                f"is per-arm and must be the same value the runner stamped on "
                f"the ArmMetrics"
            )
        if self.map_regression_tolerance_pp != self.candidate.map_regression_tolerance_pp:
            raise ValueError(
                f"DecisionInputs.map_regression_tolerance_pp ({self.map_regression_tolerance_pp}) "
                f"must equal candidate.map_regression_tolerance_pp "
                f"({self.candidate.map_regression_tolerance_pp})"
            )
        # map_no_regression: must be plain bool, not a truthy string.
        if not isinstance(self.map_no_regression, bool):
            raise ValueError(
                f"map_no_regression must be bool; got "
                f"{type(self.map_no_regression).__name__}={self.map_no_regression!r}"
            )
        # map_regression_tolerance_pp: float-with-bool-exclusion + finite
        # + non-negative (mirrors HmmYamlConfig's parallel field).
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
        # C3 iter-7 NEW-MAJOR (tolerance ceiling) 2026-05-09: cap at
        # DROP_TOTAL_MAP_REGRESSION_PP. Beyond this, the deploy guard
        # tolerates regressions that the drop trigger also rejects,
        # making the rule incoherent. CopyPasteBalanceYamlConfig
        # mirrors this ceiling at its layer; both must change in
        # lock-step if the §3.7 drop threshold is ever amended.
        if self.map_regression_tolerance_pp > self.DROP_TOTAL_MAP_REGRESSION_PP:
            raise ValueError(
                f"map_regression_tolerance_pp must be <= "
                f"DROP_TOTAL_MAP_REGRESSION_PP ({self.DROP_TOTAL_MAP_REGRESSION_PP}); "
                f"got {self.map_regression_tolerance_pp} (a tolerance > the drop "
                f"threshold makes the deploy guard broader than the drop trigger, "
                f"which is incoherent — both cases would fire on the same input)"
            )


@dataclass(frozen=True)
class DecisionResult:
    """One arm's decision row in the output JSON.

    Mirrors ``components/hmm_smoother/gates/decision_gate.DecisionResult``:
    ``decision`` is the case (deploy / defer / drop / executor_error);
    ``notes`` carries the triggering-condition trace so a non-deploy row
    is auditable. ``escalation`` is omitted here (the §三 plan has no
    escalation ladder — deploy/defer/drop are terminal).
    """

    arm_id: ArmId
    beta: float | None                         # None for no_aug / cp_only
    decision: CopyPasteDecision
    notes: str

    _ALLOWED_BETAS_FOR_CP_BALANCED: ClassVar[tuple[float, ...]] = (0.99, 0.999, 0.9999)
    _ALLOWED_DECISION_ARM_IDS: ClassVar[tuple[ArmId, ...]] = (
        ArmId.CP_ONLY,
        ArmId.CP_BALANCED,
    )

    def __post_init__(self) -> None:
        # C3 NEW-MAJOR 2 2026-05-09: enum membership + arm/beta
        # consistency. Without these, an in-memory DecisionResult with
        # ``beta=0.5`` would serialize past the JSON Schema's enum check
        # only because the runner happens not to put it on the headline
        # row — a fragile coincidence.
        if not isinstance(self.arm_id, ArmId):
            raise ValueError(
                f"arm_id must be ArmId enum; got {type(self.arm_id).__name__}"
            )
        # C3 NEW-MAJOR 3 follow-on 2026-05-09: ArmId.NO_AUG is rejected
        # here. The baseline arm carries no decision (the rule is
        # undefined on a self-comparison; DecisionInputs.__post_init__
        # rejects ``candidate.is_baseline_reference=True``). Mirroring
        # the schema's headline_arm enum (no_aug REJECTED) at the
        # dataclass boundary.
        if self.arm_id not in self._ALLOWED_DECISION_ARM_IDS:
            raise ValueError(
                f"arm_id must be one of {[m.value for m in self._ALLOWED_DECISION_ARM_IDS]}; "
                f"got {self.arm_id.value!r} (no_aug is REJECTED — the "
                f"baseline arm carries no decision)"
            )
        if not isinstance(self.decision, CopyPasteDecision):
            raise ValueError(
                f"decision must be CopyPasteDecision enum; got "
                f"{type(self.decision).__name__}"
            )
        if not isinstance(self.notes, str):
            raise ValueError(
                f"notes must be str; got {type(self.notes).__name__}"
            )
        # C3 NEW-MAJOR 2 follow-on 2026-05-09: notes must be non-empty.
        # Every DecisionResult cell needs an audit trail; an empty notes
        # field on a non-deploy row is exactly the silent decision-corruption
        # vector the §3.7 cascade is designed to surface.
        if not self.notes:
            raise ValueError(
                "notes must be non-empty (every DecisionResult requires a "
                "trace string for audit; deploy rows record the gate condition, "
                "non-deploy rows record the triggering condition)"
            )
        # beta required iff arm_id == CP_BALANCED.
        if self.arm_id == ArmId.CP_BALANCED:
            if self.beta is None:
                raise ValueError(
                    "beta is required when arm_id == ArmId.CP_BALANCED; got None"
                )
            if not isinstance(self.beta, float) or isinstance(self.beta, bool):
                raise ValueError(
                    f"beta must be float; got "
                    f"{type(self.beta).__name__}={self.beta!r}"
                )
            if not math.isfinite(self.beta):
                raise ValueError(f"beta must be finite; got {self.beta!r}")
            if self.beta not in self._ALLOWED_BETAS_FOR_CP_BALANCED:
                raise ValueError(
                    f"beta must be one of {self._ALLOWED_BETAS_FOR_CP_BALANCED}; "
                    f"got {self.beta}"
                )
        else:
            if self.beta is not None:
                raise ValueError(
                    f"beta must be None when arm_id != ArmId.CP_BALANCED; "
                    f"got {self.beta!r}"
                )


def apply_decision_rule(
    inputs: DecisionInputs,
    *,
    beta: float | None,
) -> DecisionResult:
    """Evaluate the §3.7 4-case rule on a single arm vs the no_aug baseline.

    Executor semantics: first-match cascade over
    ``deploy → defer → drop``; ``drop`` is the unconditional catch-all
    for legitimate input. ``executor_error`` fires only on malformed
    input (see module docstring "Executor semantics" block).

    Middle-case framing (B2 review I5 2026-05-09): the §三 plan prose
    enumerates 3 cases (deploy / defer / drop), but a strict reading
    leaves a middle-case region (rare avg ΔAP ∈ [2pp, 5pp) without
    every deploy guard passing) unclassified. The cascade order
    deploy → defer → drop captures that region in drop BY DESIGN
    (defer's "<2pp" prose excludes it; deploy's full guard set
    excludes it). The runner records the triggering condition in
    ``DecisionResult.notes`` so reviewers can distinguish "literal
    drop" (mAP regress / safety regress / FP rise) from
    "middle-case drop" (improvement insufficient under the deploy
    guards). See module-level docstring for the full plan-prose
    mapping.

    Baseline-reference rejection (C3 NEW-MAJOR 2 2026-05-09):
    ``DecisionInputs.__post_init__`` rejects
    ``candidate.is_baseline_reference=True`` at construction; the rule
    is undefined on a self-comparison and the runner is responsible for
    never building such an input. The earlier "sentinel result" path
    described here is now unreachable — fail loudly at construction
    rather than coerce a meaningless decision.

    Args:
        inputs: precomputed ``ArmMetrics`` for both sides + map_no_regression
            verdict from the upstream eval pipeline + tolerance.
        beta: the β value of the sweep cell when ``inputs.candidate.arm_id ==
            "cp_balanced"``; ``None`` when ``arm_id == "cp_only"``. This is a
            keyword-only argument with no default — the runner MUST be
            explicit at every call site (DecisionResult.__post_init__ enforces
            the cross-knob arm/beta consistency). The rule body itself is
            β-agnostic; ``beta`` flows through to the returned DecisionResult
            so the runner can serialize per-β sweep rows without rebuilding
            the result.

    Returns:
        ``DecisionResult`` ready to be serialized into one cell of the
        output JSON. The runner aggregates per-arm + per-β results and
        selects the headline row from CLI flags.

    Raises:
        ValueError: hard programmer errors only (wrong dataclass type,
            unexpected ``candidate.arm_id`` value, etc.). Legitimate-but-
            malformed numeric inputs are returned as
            ``CopyPasteDecision.EXECUTOR_ERROR`` with diagnostic notes —
            NOT raised — so the runner can still serialize a complete
            output block.
    """
    if not isinstance(inputs, DecisionInputs):
        raise ValueError(
            f"inputs must be DecisionInputs; got {type(inputs).__name__}"
        )

    arm_id_str = inputs.candidate.arm_id
    if arm_id_str == "cp_only":
        arm_id = ArmId.CP_ONLY
    elif arm_id_str == "cp_balanced":
        arm_id = ArmId.CP_BALANCED
    else:
        # DecisionInputs already rejects baseline candidates, but defense
        # in depth — surface an unexpected arm_id as a hard programmer
        # error rather than coercing a meaningless decision.
        raise ValueError(
            f"candidate.arm_id must be 'cp_only' or 'cp_balanced'; got "
            f"{arm_id_str!r} (no_aug is REJECTED upstream by "
            f"DecisionInputs.__post_init__'s is_baseline_reference check)"
        )

    cand = inputs.candidate

    # Defensive NaN/inf re-check on the five candidate metric fields the
    # rule consumes PLUS inputs.map_regression_tolerance_pp (used by the
    # verdict ↔ delta consistency check below — a NaN tolerance makes
    # ``total_map >= -tolerance`` evaluate False, which can silently
    # match a stamped ``map_no_regression=False`` and slip past the
    # consistency check). ArmMetrics.__post_init__ +
    # DecisionInputs.__post_init__ already reject these at construction;
    # defense-in-depth against a future refactor that bypasses either
    # dataclass. On failure, return EXECUTOR_ERROR (not raise) so the
    # runner can still serialize a complete output block.
    for label, value in (
        ("candidate.rare_class_mean_delta_AP_pp",
         cand.rare_class_mean_delta_AP_pp),
        ("candidate.rare_class_max_delta_AP_pp",
         cand.rare_class_max_delta_AP_pp),
        ("candidate.rare_safety_min_delta_AP_pp",
         cand.rare_safety_min_delta_AP_pp),
        ("candidate.rare_related_fp_delta_frac",
         cand.rare_related_fp_delta_frac),
        ("candidate.total_map_delta_pp", cand.total_map_delta_pp),
        ("inputs.map_regression_tolerance_pp",
         inputs.map_regression_tolerance_pp),
    ):
        if not math.isfinite(value):
            return DecisionResult(
                arm_id=arm_id,
                beta=beta,
                decision=CopyPasteDecision.EXECUTOR_ERROR,
                notes=(
                    f"executor_error: defensive NaN/inf re-check failed on "
                    f"{label}={value!r} (ArmMetrics / DecisionInputs "
                    f"construction should have rejected this — a future "
                    f"refactor likely bypassed the dataclass boundary)"
                ),
            )

    rare_avg = cand.rare_class_mean_delta_AP_pp
    rare_max = cand.rare_class_max_delta_AP_pp
    rare_safety_min = cand.rare_safety_min_delta_AP_pp
    rare_fp = cand.rare_related_fp_delta_frac
    total_map = cand.total_map_delta_pp
    map_no_regr = inputs.map_no_regression
    tolerance = inputs.map_regression_tolerance_pp

    # Defensive verdict ↔ delta consistency re-check. Upstream is
    # supposed to compute ``map_no_regression`` from
    # ``(candidate_total_map - baseline_total_map)`` vs
    # ``map_regression_tolerance_pp``, but a buggy upstream pipeline (or
    # a hand-built ArmMetrics that paired a stale verdict with a fresh
    # delta) could stamp ``map_no_regression=True`` while the recorded
    # delta is below tolerance — and ship as DEPLOY without ever
    # triggering the existing literal-drop guards. Re-derive the verdict
    # from the recorded delta and tolerance; mismatch is malformed
    # input → EXECUTOR_ERROR. (Boundary semantics: ``total_map_pp ==
    # -tolerance_pp`` exactly satisfies the no-regression guard per the
    # docstring's pinned boundary table, matching the non-strict ≥ in
    # the deploy condition.)
    expected_no_regression = total_map >= -tolerance
    if map_no_regr != expected_no_regression:
        return DecisionResult(
            arm_id=arm_id,
            beta=beta,
            decision=CopyPasteDecision.EXECUTOR_ERROR,
            notes=(
                f"executor_error: map_no_regression verdict / "
                f"total_map_delta inconsistency. Upstream stamped "
                f"map_no_regression={map_no_regr}, but "
                f"total_map_delta_pp={total_map:.6f}pp vs "
                f"map_regression_tolerance_pp={tolerance} implies "
                f"expected_no_regression={expected_no_regression}. The "
                f"upstream eval pipeline must re-derive the verdict "
                f"from the SAME (delta, tolerance) inputs the rule "
                f"consumes — a desynchronized verdict can otherwise "
                f"ship a regressing arm as DEPLOY."
            ),
        )

    # Deploy guards (plan §3.7 deploy condition).
    deploy_rare_improvement = (
        rare_avg >= DecisionInputs.DEPLOY_RARE_AVG_DELTA_AP_PP
        or rare_max >= DecisionInputs.DEPLOY_ANY_RARE_DELTA_AP_PP
    )
    deploy_rare_safety = (
        rare_safety_min >= DecisionInputs.DEPLOY_RARE_SAFETY_MIN_DELTA_AP_PP
    )
    deploy_rare_fp = rare_fp <= DecisionInputs.DEPLOY_RARE_FP_RISE_MAX_FRAC

    if deploy_rare_improvement and deploy_rare_safety and deploy_rare_fp and map_no_regr:
        return DecisionResult(
            arm_id=arm_id,
            beta=beta,
            decision=CopyPasteDecision.DEPLOY,
            notes=(
                f"deploy: rare_avg={rare_avg:.3f}pp, rare_max={rare_max:.3f}pp "
                f"(≥{DecisionInputs.DEPLOY_RARE_AVG_DELTA_AP_PP}pp via avg-or-max); "
                f"rare_safety_min={rare_safety_min:.3f}pp "
                f"(≥{DecisionInputs.DEPLOY_RARE_SAFETY_MIN_DELTA_AP_PP}pp); "
                f"rare_fp={rare_fp:.4f} "
                f"(≤{DecisionInputs.DEPLOY_RARE_FP_RISE_MAX_FRAC}); "
                f"total_map={total_map:.3f}pp, "
                f"map_no_regression=True (tolerance={tolerance}pp)"
            ),
        )

    # Literal drop triggers (plan §3.7 drop prose) — strict greater-than on
    # magnitude per the boundary-semantics table in the module docstring.
    drop_map_regress = total_map < -DecisionInputs.DROP_TOTAL_MAP_REGRESSION_PP
    drop_safety_regress = (
        rare_safety_min < -DecisionInputs.DROP_RARE_SAFETY_REGRESSION_PP
    )
    drop_fp_rise = rare_fp > DecisionInputs.DROP_RARE_FP_RISE_FRAC
    any_literal_drop = drop_map_regress or drop_safety_regress or drop_fp_rise

    # Defer condition (plan §3.7 defer prose): rare improvement < 2pp AND
    # map_no_regression AND no literal drop trigger fires. The "no drop
    # trigger fires" guard is co-located in defer's condition rather than
    # relying on cascade order alone — a regressing-safety-class with
    # rare_avg < 2pp would otherwise serialize as defer instead of drop.
    defer_match = (
        rare_avg < DecisionInputs.DEFER_RARE_AVG_LT_PP
        and map_no_regr
        and not any_literal_drop
    )
    if defer_match:
        return DecisionResult(
            arm_id=arm_id,
            beta=beta,
            decision=CopyPasteDecision.DEFER,
            notes=(
                f"defer: rare_avg={rare_avg:.3f}pp "
                f"(<{DecisionInputs.DEFER_RARE_AVG_LT_PP}pp, no significant "
                f"rare improvement); map_no_regression=True; no literal drop "
                f"trigger fires (rare_safety_min={rare_safety_min:.3f}pp, "
                f"rare_fp={rare_fp:.4f}, total_map={total_map:.3f}pp). "
                f"R3 action: re-evaluate with more aggressive copy-paste."
            ),
        )

    # Drop catch-all. Distinguish "literal drop" from "middle-case drop"
    # in the notes so reviewers can audit the triggering condition.
    if any_literal_drop:
        trigger_parts: list[str] = []
        if drop_map_regress:
            trigger_parts.append(
                f"total_map={total_map:.3f}pp (regress > "
                f"{DecisionInputs.DROP_TOTAL_MAP_REGRESSION_PP}pp)"
            )
        if drop_safety_regress:
            trigger_parts.append(
                f"rare_safety_min={rare_safety_min:.3f}pp (regress > "
                f"{DecisionInputs.DROP_RARE_SAFETY_REGRESSION_PP}pp)"
            )
        if drop_fp_rise:
            trigger_parts.append(
                f"rare_fp={rare_fp:.4f} (rise > "
                f"{DecisionInputs.DROP_RARE_FP_RISE_FRAC})"
            )
        notes = "drop (literal trigger): " + "; ".join(trigger_parts)
    else:
        # Middle-case drop: improvement insufficient under deploy guards
        # AND defer's "<2pp" framing also fails (so rare_avg ≥ 2pp without
        # meeting full deploy guards) OR map_no_regression=False with no
        # literal drop trigger firing (total_map ∈ [-0.5, -tolerance)).
        deploy_fail_parts: list[str] = []
        if not deploy_rare_improvement:
            deploy_fail_parts.append(
                f"rare improvement insufficient: rare_avg={rare_avg:.3f}pp, "
                f"rare_max={rare_max:.3f}pp (neither ≥ "
                f"{DecisionInputs.DEPLOY_RARE_AVG_DELTA_AP_PP}pp)"
            )
        if not deploy_rare_safety:
            deploy_fail_parts.append(
                f"rare_safety_min={rare_safety_min:.3f}pp (< "
                f"{DecisionInputs.DEPLOY_RARE_SAFETY_MIN_DELTA_AP_PP}pp deploy floor "
                f"but ≥ -{DecisionInputs.DROP_RARE_SAFETY_REGRESSION_PP}pp drop "
                f"threshold)"
            )
        if not deploy_rare_fp:
            deploy_fail_parts.append(
                f"rare_fp={rare_fp:.4f} (> "
                f"{DecisionInputs.DEPLOY_RARE_FP_RISE_MAX_FRAC} deploy ceiling but ≤ "
                f"{DecisionInputs.DROP_RARE_FP_RISE_FRAC} drop threshold)"
            )
        if not map_no_regr:
            deploy_fail_parts.append(
                f"map_no_regression=False (total_map={total_map:.3f}pp vs "
                f"tolerance {tolerance}pp; below tolerance but above "
                f"-{DecisionInputs.DROP_TOTAL_MAP_REGRESSION_PP}pp drop threshold)"
            )
        notes = (
            "drop (middle-case): deploy guards failed: "
            + "; ".join(deploy_fail_parts)
            if deploy_fail_parts
            else "drop (middle-case): no specific deploy-guard failure isolated"
        )

    return DecisionResult(
        arm_id=arm_id,
        beta=beta,
        decision=CopyPasteDecision.DROP,
        notes=notes,
    )
