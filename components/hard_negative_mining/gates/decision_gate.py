"""d-stage decision rule for the §四 hard-negative-mining ablation.

Mirrors ``components/copy_paste_balance/gates/decision_gate.py`` (locked
iter-11, 2026-05-09 — internal B2 audit trail; b-stage authors can
ignore the sister-precedent annotation): pre-committed thresholds
frozen as ClassVars on ``DecisionInputs``, first-match cascade with
explicit catch-all, ``executor_error`` reserved for malformed inputs
only.

Plan §4.7 cases (verbatim — do NOT widen thresholds during impl, do
NOT silently re-order; threshold knobs may be moved into
``configs/hard_negative_mining.yaml`` only after a fresh adversarial
loop):

  * deploy : fp_drop_frac ≥ 0.50
              AND real_light_recall_delta_pp ≥ −0.5
              AND total mAP no-regression (Δ ≥ −map_regression_tolerance_pp,
                  default 0.2pp).

  * defer  : 0.20 ≤ fp_drop_frac < 0.50
              AND real_light_recall_delta_pp ≥ −0.5.
              **Plan-prose ambiguity resolution (B2 review S2 2026-05-10)**:
              plan §4.7 line 137 says "FP 仅降 20-50%" — Chinese-prose
              "20-50%" is genuinely ambiguous (could be inclusive both
              ends OR inclusive lower / exclusive upper). The impl
              resolves to ``[0.20, 0.50)`` (inclusive lower, EXCLUSIVE
              upper) so 0.50 cleanly belongs to deploy via the
              non-strict "≥ 0.50" guard — first-match cascade would
              mask the overlap, but explicit dataclass thresholds
              make the partition unambiguous. Future plan revisions
              that intend inclusive-both-ends MUST re-run the §四
              adversarial loop.
              **Codex stop-gate fix 2026-05-10**: defer is plan-legal
              even when total mAP regressed past the tolerance. Plan
              §4.7 defer gate is GATED SOLELY on fp_drop range +
              recall floor; mAP regression is NOT a defer guard.
              (Compare §3.7 sister where defer DOES require mAP
              no-regression — the §3.7 plan prose explicitly states
              that constraint; §四 plan prose does not. Do not
              cargo-cult the §3.7 constraint into §四.)
              Action: re-evaluate in R3 with broader / re-tuned mining
              (plan §4.7 defer prose).

  * drop   : catch-all. Plan §4.7 literal drop triggers:
                - real_light_recall_delta_pp < −0.5
                - fp_drop_frac < 0.20
              The cascade order deploy → defer → drop also sweeps
              into drop the (fp_drop ≥ 0.50, mAP regress, recall OK)
              corner that plan §4.7 leaves implicit (deploy fails on
              mAP guard; defer fails on fp_drop ≥ 0.50; literal drop
              triggers don't fire either — catch-all puts it in drop
              by construction). Note: the (fp_drop ∈ [0.20, 0.50),
              mAP regress, recall OK) case is DEFER per plan, not
              drop — defer is mAP-agnostic.

Boundary semantics (mirrors §3.7 gate; plan §4.7 prose uses strict
greater-than for every drop trigger and ≥ for deploy/defer guards):
  * fp_drop_frac = 0.50           → deploy guard PASSES (matches "≥ 0.50")
  * fp_drop_frac = 0.20           → defer guard PASSES (matches "≥ 0.20")
  * fp_drop_frac = 0.4999         → defer guard PASSES (< 0.50, ≥ 0.20)
  * fp_drop_frac = 0.20 exactly   → drop trigger does NOT fire
                                    (drop requires "< 0.20", strict)
  * fp_drop_frac = 0.1999         → drop trigger FIRES
  * real_light_recall_delta_pp = -0.5    → deploy/defer guard PASSES
                                            (matches "≥ −0.5pp")
  * real_light_recall_delta_pp = -0.5 exactly → drop trigger does NOT fire
                                            (drop requires "< −0.5pp", strict)
  * real_light_recall_delta_pp = -0.5001 → drop trigger FIRES
  * total_map_delta_pp = −0.2pp (default tol) → deploy mAP sub-guard
                                            PASSES (matches "≥ −0.2pp");
                                            defer is mAP-agnostic per
                                            plan §4.7 — actual outcome
                                            depends on (fp_drop, recall)
                                            jointly, NOT on the mAP
                                            value alone.
  * total_map_delta_pp = −0.5pp   → no literal drop on mAP regression
                                    in §4.7; outcome depends on
                                    (fp_drop, recall):
                                      fp_drop ∈ [0.20, 0.50), recall
                                      ≥ −0.5pp → DEFER (defer is
                                      mAP-agnostic per plan §4.7)
                                      fp_drop ≥ 0.50, recall ≥ −0.5pp
                                      → catch-all DROP (deploy fails
                                      on mAP; defer fails on fp range)

Executor semantics (B2 review C2 2026-05-10 clarification): cases
evaluated in order ``deploy → defer → drop``; first match wins.
``drop`` is the unconditional catch-all for legitimate input.

By the time ``apply_decision_rule`` runs, ``ArmMetrics.__post_init__``
has rejected NaN/inf and out-of-range numerics; ``DecisionInputs.__post_init__``
has rejected manifest-hash drift, baseline-reference inversion, and
tolerance ceiling violations. So the legitimate ``executor_error``
trigger surface inside the rule body itself is empty under the
current contract — every malformed-input case is rejected upstream
of the rule.

The b-stage implementation MUST nonetheless perform a defensive NaN/inf
re-check at the rule's entry on the four metric fields it consumes
(``fp_drop_frac``, ``real_light_recall_delta_pp``, ``total_map_delta_pp``,
``map_regression_tolerance_pp``) — defense-in-depth against a future
refactor that bypasses ``ArmMetrics``. If the re-check ever fires, it
returns ``HardNegativeDecision.EXECUTOR_ERROR`` with diagnostic notes
rather than raising. The runner uses that return path to serialize a
complete output block; ``ValueError`` from this function is reserved
for hard programmer errors (wrong dataclass type, etc.) — those DO
propagate.

Schema for ``runs/_hard_negative_decision.json`` (plan §4.7 d output):

    {
      "schema_version": "1",
      "config_yaml_sha256": str,
      "data_yaml_sha256": str,
      "fp_manifest_sha256": str,        # frozen FP eval manifest
      "eval_manifest_sha256": str,      # frozen R2 val manifest
      "headline_arm": "with_hn",        # only candidate; no_hn REJECTED
      "headline_metrics": MetricsBlock,
      "headline_decision": "deploy" | "defer" | "drop" | "executor_error",
      "notes": str,
      "no_hn":   { "metrics": BaselineMetricsBlock, "notes": str },
      "with_hn": { "metrics": CandidateMetricsBlock,
                   "decision": "deploy" | "defer" | "drop" | "executor_error",
                   "notes": str }
    }

Validation boundary (mirrors §3.7 gate iter-3 contract): the JSON
Schema file at
``components/hard_negative_mining/gates/_hard_negative_decision_schema.json``
is **OUTPUT-ONLY**. Per-cell input validation happens at dataclass
construction (``ArmMetrics``, ``DecisionInputs``); numeric malformed
values that survive into the rule (defensive re-check above) surface
as ``executor_error`` rows. Output-validation failure is a hard error
(non-zero exit).

Scaffold (a-stage): API only; thresholds frozen as ClassVars.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from components.hard_negative_mining._internals import (
    PLAN_MAP_TOLERANCE_CEILING_PP,
)
from components.hard_negative_mining.gates.ablation_gate import ArmMetrics


class HardNegativeDecision(str, Enum):
    DEPLOY = "deploy"
    DEFER = "defer"
    DROP = "drop"
    EXECUTOR_ERROR = "executor_error"


class ArmId(str, Enum):
    """The two §4.8 c arms."""

    NO_HN = "no_hn"
    WITH_HN = "with_hn"


@dataclass(frozen=True)
class DecisionInputs:
    """One arm's metrics + the upstream mAP-no-regression verdict.

    Plan §4.7 thresholds are pinned as ClassVars on this dataclass so
    they're co-located with the cases that consume them — moving any of
    them into the YAML requires re-running the §四 adversarial loop.

    Field redundancy (B2 review I12 2026-05-10): ``map_no_regression``
    and ``map_regression_tolerance_pp`` ARE redundant with the
    corresponding fields on ``candidate``. After the cross-side
    equality check below, the two values are guaranteed identical.
    The redundancy is INTENTIONAL — it mirrors the §3.7 sister
    contract where the runner builds ``DecisionInputs`` from the
    upstream eval pipeline, and the per-cell ``ArmMetrics`` are the
    on-disk artifact representation. Removing the duplication would
    force the runner to build a partial ``DecisionInputs`` from the
    candidate alone, breaking the symmetry across §3.7 / §四. b-stage
    MUST NOT silently drop the redundancy.
    """

    baseline: ArmMetrics                       # the no_hn arm
    candidate: ArmMetrics                      # the with_hn arm
    map_no_regression: bool                    # upstream verdict (== candidate's; see I12 above)
    map_regression_tolerance_pp: float         # cross-check field (== candidate's); default 0.2

    # Plan §4.7 deploy guards ------------------------------------------------
    DEPLOY_FP_DROP_MIN_FRAC: ClassVar[float] = 0.50
    DEPLOY_RECALL_MIN_DELTA_PP: ClassVar[float] = -0.5
    # (mAP no-regression handled via map_no_regression bool)

    # Plan §4.7 defer condition ----------------------------------------------
    DEFER_FP_DROP_MIN_FRAC: ClassVar[float] = 0.20

    # Plan §4.7 drop triggers (strict-greater-than against magnitude) -------
    # real_light_recall_delta_pp < -DROP_RECALL_REGRESSION_PP triggers drop;
    # fp_drop_frac < DEFER_FP_DROP_MIN_FRAC triggers drop (i.e. < 0.20).
    DROP_RECALL_REGRESSION_PP: ClassVar[float] = 0.5
    # mAP regression handled by the deploy gate's map_no_regression check;
    # there is no separate §4.7 literal drop trigger on mAP — the catch-all
    # sweeps a regressing-mAP candidate into drop via the cascade.

    _MAP_TOLERANCE_CEILING_PP: ClassVar[float] = PLAN_MAP_TOLERANCE_CEILING_PP

    def __post_init__(self) -> None:
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
                "defined as candidate vs the no_hn baseline arm only)"
            )
        if self.candidate.is_baseline_reference:
            raise ValueError(
                "candidate.is_baseline_reference must be False (the rule is "
                "undefined on a self-comparison; pass a with_hn ArmMetrics "
                "as the candidate)"
            )
        # B2 review I1 2026-05-10: type-check fields BEFORE running cross-side
        # equality. If self.map_no_regression is the string "True" and
        # self.candidate.map_no_regression is the bool True, the inequality
        # check would fire with a confusing "cross-side mismatch" message
        # rather than naming the actual bug (wrong type).
        if not isinstance(self.map_no_regression, bool):
            raise ValueError(
                f"map_no_regression must be bool; got "
                f"{type(self.map_no_regression).__name__}="
                f"{self.map_no_regression!r}"
            )
        # map_regression_tolerance_pp: float-with-bool-exclusion + finite +
        # non-negative + ceiling at PLAN_MAP_TOLERANCE_CEILING_PP. Type
        # check must precede cross-side equality for the same I1 reason.
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
                f"broader than the drop trigger, which is incoherent)"
            )
        # Manifest-hash equality between sides — the runner enforces same
        # eval / fp / data manifests across both arms; an in-memory cell
        # with mismatched hashes is a construction bug.
        if self.baseline.eval_manifest_sha256 != self.candidate.eval_manifest_sha256:
            raise ValueError(
                f"baseline.eval_manifest_sha256 != candidate.eval_manifest_sha256 "
                f"({self.baseline.eval_manifest_sha256} vs "
                f"{self.candidate.eval_manifest_sha256})"
            )
        if self.baseline.fp_manifest_sha256 != self.candidate.fp_manifest_sha256:
            raise ValueError(
                f"baseline.fp_manifest_sha256 != candidate.fp_manifest_sha256 "
                f"({self.baseline.fp_manifest_sha256} vs "
                f"{self.candidate.fp_manifest_sha256}) — §4.7 frozen manifest "
                f"is the FP-denominator anti-gaming safeguard; mismatch is a "
                f"contract violation"
            )
        if self.baseline.data_yaml_sha256 != self.candidate.data_yaml_sha256:
            raise ValueError(
                f"baseline.data_yaml_sha256 != candidate.data_yaml_sha256 "
                f"({self.baseline.data_yaml_sha256} vs "
                f"{self.candidate.data_yaml_sha256}) — class-label drift "
                f"across arms silently corrupts AP comparisons"
            )
        # Cross-side equality on tolerance (runner-side knob; identical
        # across arms) and verdict-vs-arm consistency.
        # B2 final-review S1 2026-05-10: the next two checks plus the
        # cross-side equality below are TRANSITIVELY equal — given:
        #   baseline.tolerance == candidate.tolerance
        #   self.tolerance     == candidate.tolerance
        # they imply self.tolerance == baseline.tolerance trivially.
        # The redundancy is INTENTIONAL (mirrors I12's redundancy
        # rationale): every consuming layer must independently witness
        # its precondition. Do NOT consolidate these to a single check.
        if self.baseline.map_regression_tolerance_pp != self.candidate.map_regression_tolerance_pp:
            raise ValueError(
                f"baseline.map_regression_tolerance_pp != "
                f"candidate.map_regression_tolerance_pp "
                f"({self.baseline.map_regression_tolerance_pp} vs "
                f"{self.candidate.map_regression_tolerance_pp})"
            )
        if self.map_no_regression != self.candidate.map_no_regression:
            raise ValueError(
                f"DecisionInputs.map_no_regression ({self.map_no_regression}) "
                f"must equal candidate.map_no_regression "
                f"({self.candidate.map_no_regression}) — the upstream verdict "
                f"is per-arm and must be the same value the runner stamped "
                f"on the ArmMetrics"
            )
        if self.map_regression_tolerance_pp != self.candidate.map_regression_tolerance_pp:
            raise ValueError(
                f"DecisionInputs.map_regression_tolerance_pp "
                f"({self.map_regression_tolerance_pp}) must equal "
                f"candidate.map_regression_tolerance_pp "
                f"({self.candidate.map_regression_tolerance_pp})"
            )


@dataclass(frozen=True)
class DecisionResult:
    """One arm's decision row in the output JSON.

    Mirrors ``components/copy_paste_balance/gates/decision_gate.DecisionResult``:
    ``decision`` is the case (deploy / defer / drop / executor_error);
    ``notes`` carries the triggering-condition trace so a non-deploy row
    is auditable.
    """

    arm_id: ArmId
    decision: HardNegativeDecision
    notes: str

    # Plan §4 only one candidate arm; baseline carries no decision.
    _ALLOWED_DECISION_ARM_IDS: ClassVar[tuple[ArmId, ...]] = (ArmId.WITH_HN,)

    def __post_init__(self) -> None:
        if not isinstance(self.arm_id, ArmId):
            raise ValueError(
                f"arm_id must be ArmId enum; got {type(self.arm_id).__name__}"
            )
        # ArmId.NO_HN is REJECTED here. The baseline arm carries no
        # decision (the rule is undefined on a self-comparison;
        # DecisionInputs.__post_init__ rejects
        # candidate.is_baseline_reference=True). Mirroring the schema's
        # headline_arm enum at the dataclass boundary.
        if self.arm_id not in self._ALLOWED_DECISION_ARM_IDS:
            raise ValueError(
                f"arm_id must be one of "
                f"{[m.value for m in self._ALLOWED_DECISION_ARM_IDS]}; "
                f"got {self.arm_id.value!r} (no_hn is REJECTED — the "
                f"baseline arm carries no decision)"
            )
        if not isinstance(self.decision, HardNegativeDecision):
            raise ValueError(
                f"decision must be HardNegativeDecision enum; got "
                f"{type(self.decision).__name__}"
            )
        if not isinstance(self.notes, str):
            raise ValueError(
                f"notes must be str; got {type(self.notes).__name__}"
            )
        if not self.notes:
            raise ValueError(
                "notes must be non-empty (every DecisionResult requires a "
                "trace string for audit; deploy rows record the gate "
                "condition, non-deploy rows record the triggering condition)"
            )


def apply_decision_rule(inputs: DecisionInputs) -> DecisionResult:
    """Evaluate the §4.7 3-case rule on the with_hn cell vs no_hn baseline.

    Executor semantics (see module docstring "Executor semantics" block
    for the C2 clarification): first-match cascade over
    ``deploy → defer → drop``; ``drop`` is the unconditional catch-all
    for legitimate input.

    By the time this function runs, ``ArmMetrics.__post_init__`` and
    ``DecisionInputs.__post_init__`` have rejected NaN/inf, range
    violations, hash drift, and tolerance-ceiling violations. The
    b-stage body MUST nonetheless perform a defensive NaN/inf re-check
    on the four metric fields it consumes; defense-in-depth against a
    future refactor that bypasses ``ArmMetrics``. If the re-check
    fires, return ``HardNegativeDecision.EXECUTOR_ERROR`` with
    diagnostic notes — the runner serializes a complete output block.
    ``ValueError`` is reserved for hard programmer errors (wrong
    dataclass type, etc.) and DOES propagate.

    Middle-case framing (C3 iter-1 NEW-MINOR fix 2026-05-10): the §四
    plan prose enumerates 3 cases (deploy / defer / drop) and is more
    cleanly partitioned than §3.7 — the fp_drop_frac thresholds at
    0.50 and 0.20 with strict / non-strict semantics fully cover the
    [−∞, ∞) range.

    The cascade catch-all sweeps into drop ONLY the
    ``(fp_drop_frac ≥ 0.50, recall_delta ≥ −0.5pp, mAP regress)``
    corner: deploy fails on its mAP-no-regression sub-guard, defer
    fails because fp_drop is OUTSIDE [0.20, 0.50), and the literal
    drop triggers (recall < −0.5pp, fp_drop < 0.20) don't fire either
    — cascade order forces drop. b-stage MUST NOT extend this
    catch-all to the
    ``(fp_drop_frac ∈ [0.20, 0.50), recall_delta ≥ −0.5pp, mAP regress)``
    case: per plan §4.7 line 137, that case is DEFER (defer is
    mAP-agnostic; cargo-culting §3.7's "defer requires mAP
    no-regression" constraint into §四 is the exact divergence the
    Codex stop-gate caught on commit `e802250`).

    The runner records the triggering condition in
    ``DecisionResult.notes`` so reviewers can distinguish "literal
    drop" (recall regress / FP gain insufficient) from "catch-all
    drop" (fp_drop ≥ 0.50 + mAP regression beyond tolerance — the
    only legitimate mAP-driven catch-all path).

    Baseline-reference rejection: ``DecisionInputs.__post_init__``
    rejects ``candidate.is_baseline_reference=True`` at construction;
    the rule is undefined on a self-comparison and the runner is
    responsible for never building such an input.

    Args:
        inputs: precomputed ``ArmMetrics`` for both sides + map_no_regression
            verdict + tolerance.

    Returns:
        ``DecisionResult`` ready to be serialized into the with_hn cell.
        The runner copies this into the headline if anchor_arm matches.

    Raises:
        NotImplementedError: a-stage scaffold.
        ValueError: hard programmer errors only (wrong dataclass type,
            etc.). Legitimate-but-malformed numeric inputs are returned
            as ``HardNegativeDecision.EXECUTOR_ERROR`` rather than
            raised — see the executor semantics block above.
    """
    raise NotImplementedError("b-stage")
