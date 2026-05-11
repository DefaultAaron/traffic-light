"""d-stage 4-case decision rule for the HMM ablation.

Mirrors the structure of ``scripts/_r2_decide_precision.py`` and
``components/knowledge_distillation/gates/`` decision
helpers: pre-committed thresholds, first-match cascade with explicit
catch-all, executor-error sink reserved for malformed inputs only.

Plan §2.2 决策规则 cases (formula-pinned per `temporal_optimization_plan.md`
§2.2 — do NOT widen thresholds during impl, do NOT silently re-order;
threshold knobs may be moved into ``configs/temporal_hmm.yaml`` only
after a fresh adversarial loop):

  flicker_improvement_pct =
      100 * (baseline_flicker_rate - candidate_flicker_rate)
            / baseline_flicker_rate
      # PRECONDITION: baseline_flicker_rate > 0 (else executor_error;
      # the main branch cannot evaluate at all).
  illegal_transition_improvement_pct =
      100 * (baseline_illegal_count - candidate_illegal_count)
            / baseline_illegal_count
      # PRECONDITION: baseline_illegal_count > 0 (else the OR-branch
      # below is SKIPPED — NOT executor_error; main flicker branch
      # still evaluates normally).
  Pre-conditions:
   - ``baseline_flicker_rate > 0`` else outcome="executor_error".
   - ``baseline_illegal_count > 0`` ONLY for OR-branch eligibility;
     when zero, OR-branch does not participate (see drop catch-all).

  * deploy : ``flicker_improvement_pct > 50.0`` (strict) AND
              ``eligible_illegal_transition_count == 0`` AND
              total mAP / safety-class AP no-regression
              (``map_regression_tolerance_pp`` from config, default 0.2 pp).
              Tracker-artifact exception: residual illegal transitions
              fully traceable to ByteTrack ID switch / short-track noise
              still pass under deploy + a notes field; the runner is
              responsible for surfacing the trace and setting the
              ``tracker_artifact_residual`` input flag.

  * defer  : ``20.0 <= flicker_improvement_pct <= 50.0`` (closed interval,
              both ends inclusive) OR (``baseline_illegal_count > 0`` AND
              ``illegal_transition_improvement_pct >= 50.0`` AND
              ``eligible_illegal_transition_count > 0``).
              When ``baseline_illegal_count == 0`` the OR-branch is
              SKIPPED (NOT executor_error); flicker main branch alone
              decides between defer (closed [20, 50]) and drop.
              Action: escalate to §2.3 AdaEMA per the §2.1 ladder.

  * drop   : catch-all. Any legitimate input (i.e. non-malformed) that
              does NOT match deploy or defer falls into drop. This
              includes the plan's literal drop case (flicker improvement
              < 20% AND illegals eliminated only by hard_zero mask) AND
              residual cases the plan §2.2 决策规则 does not enumerate
              explicitly (e.g. flicker improves > 50% but mAP regresses,
              or flicker 0%-20% with non-mask illegal elimination).
              Default escalation: §2.3 AdaEMA; direct jump to §2.4 GRU /
              §2.5 Transformer requires explicit failure-mode evidence
              in ``escalation_rationale``.

Boundary semantics (closed-interval convention):
  * flicker improvement = 20%  → defer (lower bound of [20, 50])
  * flicker improvement = 50%  → defer (upper bound of [20, 50])
  * illegal-rate improvement = 50%  → defer

Executor semantics: cases are evaluated in order ``deploy → defer → drop``;
first match wins. ``drop`` is the unconditional catch-all for legitimate
input; ``executor_error`` is reserved for **malformed input only**.

Boundary between the runner+schema layer and the executor:
``apply_decision_rule`` accepts pre-constructed dataclasses
(typed at the language level), so missing fields / wrong types fail
upstream at dataclass-construction or JSON Schema validation time. The
runner is responsible for catching those failures, serializing them as
``decision: "executor_error"`` rows in ``runs/_hmm_decisions.json``,
and recording the underlying error in ``notes``. The schema file at
``components/hmm_smoother/gates/_hmm_decision_schema.json`` is the
load-bearing contract for raw-input validation; the executor only
handles **numeric malformed values** that pass dataclass construction
but cannot be classified:

  * NaN / inf in any metric (any side)
  * ``baseline_flicker_rate == 0`` (zero-divide on flicker improvement pct;
    blocks the main flicker branch — whole decision invalid)
  * ``total_transitions == 0`` on either side (zero-divide on illegal rate)
  * ``baseline.total_transitions != candidate.total_transitions``
    (denominator drift; same eligible-tracks rule on the same EVAL
    JSONL must produce the same denominator regardless of smoother)
  * ``baseline.eligible_track_count != candidate.eligible_track_count``
    (track-count denominator drift; same EVAL JSONL → same eligible
    track membership regardless of smoother)
  * ``baseline.total_track_count != candidate.total_track_count``
    (total-track-count denominator drift; same hazard)

Note: ``baseline_illegal_count == 0`` is **NOT** an executor_error.
Per plan §2.2 决策规则, when ``baseline_illegal_count == 0`` the defer
OR-branch (``illegal_transition_improvement_pct >= 50.0``) does not
participate; the flicker main branch evaluates normally and the cascade
proceeds as deploy → defer (flicker-only) → drop.
  * Threshold ranges out of bounds (negative counts, > 1.0 fractions)

Cases are NOT mutually exclusive in the plan-prose sense (defer's flicker
condition and drop's catch-all overlap by design once you accept first-match
cascade). The earlier "exactly one matches" claim is replaced by:
"first-match cascade with explicit precedence; drop is a structural
catch-all, not a conditional case."

Schema for ``runs/_hmm_decisions.json`` (plan §2.2 决策规则 output, with
``runners/ablation.py`` sensitivity-sweep contract folded in):

    {
      "schema_version": "1",
      "config_yaml_sha256": str,                # source-of-truth hash
      "headline_config": {                       # the deploy_anchor row's α
        "laplace_alpha": float,
        "illegal_transition_policy": "hard_zero" | "downweight",
        "viterbi_window": int | null,
        "inference_mode": "forward_backward" | "viterbi",
        "observation_mode": "soft" | "hard"
      },
      "headline_metrics": {
        "baseline_flicker_rate": float,
        "candidate_flicker_rate": float,
        "flicker_rate_delta_pct": float,
        "baseline_illegal_count": int,
        "candidate_illegal_count": int,
        "baseline_total_transitions": int,           # split denominator;
        "candidate_total_transitions": int,          # executor asserts ==.
        "baseline_eligible_track_count": int,        # split denominator
        "candidate_eligible_track_count": int,       # for eligible + total
        "baseline_total_track_count": int,           # track-count fields;
        "candidate_total_track_count": int,          # executor asserts ==.
        "illegal_rate_delta_pct": float,
        "map_no_regression": bool,
        "map_regression_tolerance_pp": float,
        "tracker_artifact_residual": bool
      },
      "decision": "deploy" | "defer" | "drop" | "executor_error",
      "deploy_via_tracker_artifact_exception": bool,
      "escalation": "ada_ema" | "gru" | "transformer" | null,
      "escalation_rationale": str,
      "notes": str,
      "sensitivity_sweep": [
        {
          "laplace_alpha": float,
          "deploy_anchor": bool,                 # exactly one row is true
          "modes": {
            "forward_backward": {                # ModeCell — diagnostic
              "metrics": { ... headline_metrics shape ... },
              "decision": "deploy" | "defer" | "drop" | "executor_error",
              "deploy_via_tracker_artifact_exception": bool,
              "escalation": "ada_ema" | "gru" | "transformer" | null,
              "escalation_rationale": str,        # fields preserved so
              "notes": str                        # non-anchor cells stay
            },                                    # auditable.
            "viterbi": {                          # ModeCell mirror.
              "metrics": { ... headline_metrics shape ... },
              "decision": ...,
              "deploy_via_tracker_artifact_exception": bool,
              "escalation": ... | null,
              "escalation_rationale": str,
              "notes": str
            }
          }
        },
        ...
      ]
    }

The ``sensitivity_sweep`` block carries one row per
``laplace_alpha ∈ {0.01, 0.1, 1.0}`` (plan §2.2 α sweep).
Each row nests both inference modes (``forward_backward``, ``viterbi``).
Exactly one row sets ``deploy_anchor: true`` and that row's α + mode pair
populates the top-level ``headline_config`` / ``headline_metrics`` /
``decision``. Anchor selection is human-driven (b-stage exposes a
``--anchor-alpha FLOAT --anchor-mode {forward_backward,viterbi}`` CLI),
not auto-picked from the sweep.

Sister-file alignment: shape modeled on
``runs/_kd_decisions.json`` and ``runs/_r2_precision_decisions.json`` —
``schema_version`` + ``headline_*`` + ``sensitivity_sweep`` mirror the
precision plan's pattern (`scripts/_r2_decision_schema.json`). The
load-bearing JSON Schema file is co-located at
``components/hmm_smoother/gates/_hmm_decision_schema.json``.

Validation boundary: the JSON Schema is **OUTPUT-ONLY**.
The runner constructs typed ``GateMetrics`` / ``DecisionInputs`` per cell
(missing fields / wrong types fail at dataclass construction; numeric
malformed values fail at the executor's gate and are serialized as
``decision: "executor_error"`` rows with diagnostic ``notes``). Only the
final aggregated artifact is validated against
``_hmm_decision_schema.json`` — once, post-aggregation, pre-rename to
``runs/_hmm_decisions.json``. Output-validation failure is a hard error
(non-zero exit), not an ``executor_error`` row.

Scaffold (a-stage): API only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from components.hmm_smoother.gates.ablation_gate import GateMetrics


class HmmDecision(str, Enum):
    DEPLOY = "deploy"
    DEFER = "defer"
    DROP = "drop"
    EXECUTOR_ERROR = "executor_error"


class HmmEscalation(str, Enum):
    """Post-defer / post-drop escalation target.

    Naming note (B2 review S1 2026-05-09): renamed from ``AD_A_EMA`` →
    ``ADA_EMA`` so the string value (``"ada_ema"``) is grep-friendly and
    matches the §2.3 AdaEMA / "Adaptive EMA" prose.
    """

    ADA_EMA = "ada_ema"
    GRU = "gru"
    TRANSFORMER = "transformer"


@dataclass(frozen=True)
class DecisionInputs:
    """Both sides of the A/B + the safety-set mAP gate result.

    ``total_transitions`` is exposed on ``GateMetrics`` and is read off
    the candidate side here; it MUST be > 0 or the executor returns
    ``executor_error`` (would zero-divide the illegal-rate gate).
    """

    baseline: GateMetrics
    candidate: GateMetrics
    map_no_regression: bool
    tracker_artifact_residual: bool   # see plan §2.2 决策规则 deploy-exception
    tracker_artifact_evidence_ref: dict | None = None  # structured evidence
    # for the deploy exception clause. Required keys (all non-empty):
    #   trace_id: str
    #   track_id: str
    #   frame_start: int
    #   frame_end: int
    #   reviewer: str
    # Missing / None / dict with any empty field, or evidence whose
    # trace_id does not resolve to a real track or whose frame_range
    # is out of bounds, is NOT executor_error: the exception clause is
    # SKIPPED and the row falls through cascade. The runner emits
    # decision_reason ∈ {"tracker_artifact_evidence_missing",
    #                    "tracker_artifact_evidence_malformed",
    #                    "tracker_artifact_evidence_unverifiable"}
    # to make the cascade reason auditable.


@dataclass(frozen=True)
class DecisionResult:
    """One row of the ``runs/_hmm_decisions.json`` sweep.

    ``deploy_via_tracker_artifact_exception`` is a first-class boolean
    so the deploy-via-exception path is
    grep-able / tabulatable across runs. It MUST be ``False`` whenever
    ``decision != DEPLOY``; the executor enforces this.
    """

    decision: HmmDecision
    deploy_via_tracker_artifact_exception: bool
    decision_reason: str  # required reason code for every row, e.g.
    # "deploy", "deploy_via_tracker_artifact_exception",
    # "tracker_artifact_evidence_missing",
    # "tracker_artifact_evidence_malformed",
    # "tracker_artifact_evidence_unverifiable",
    # "defer_flicker_band", "defer_illegal_or", "defer_eligible_zero",
    # "drop_flicker_low", "drop_illegal_weak",
    # "executor_error_baseline_flicker_zero", etc.
    escalation: HmmEscalation | None
    escalation_rationale: str
    notes: str


def apply_decision_rule(inputs: DecisionInputs) -> DecisionResult:
    """Evaluate the §2.2 决策规则 4-case rule on a single A/B run.

    Executor semantics: first-match cascade over ``deploy → defer →
    drop``; drop is the unconditional catch-all for legitimate input.
    ``executor_error`` fires only on malformed input (see module
    docstring "Executor semantics" block).

    Truth table (b-stage MUST honor; b-stage entry criterion: add
    executable tests for every row R1a-R5 below; see
    ``docs/planning/temporal_optimization_plan.md`` §2.2 b-stage entry
    requirement for the canonical case list).

    Row | flicker_improvement_pct | other inputs                        | outcome
    ----+-------------------------+-------------------------------------+--------
    R1a | > 50.0                  | eligible == 0 AND mAP pass          | deploy
    R1b | > 50.0                  | eligible == 0 AND mAP fail          | cascade
    R1c | > 50.0                  | eligible > 0 AND tracker_residual=F | cascade
    R1d | > 50.0                  | eligible > 0 AND tracker_residual=T | deploy via
        |                         | AND evidence_ref dict 5字段全填     | exception
        |                         | AND evidence verifiable AND mAP pass|
    R1e | > 50.0                  | eligible > 0 AND tracker_residual=T | cascade
        |                         | AND evidence_ref missing/None/      | (not error)
        |                         | malformed/unverifiable              |
    R2  | ∈ [20.0, 50.0] closed   | baseline_illegal_count == 0         | defer
    R3  | < 20.0                  | baseline_illegal_count == 0         | drop
    R4  | < 20.0                  | baseline_illegal_count > 0 AND      | defer
        |                         | illegal_transition_improvement_pct  |
        |                         | >= 50.0 AND eligible > 0            |
    R5  | < 20.0                  | baseline_illegal_count > 0 AND      | drop
        |                         | illegal_transition_improvement_pct  |
        |                         | < 50.0                              |

    Note: ``baseline_illegal_count > 0 AND eligible_illegal_transition_count
    == 0`` (candidate eliminated all illegal transitions) does NOT need
    its own row; it routes through R1/R2/R3/R5 by flicker band:
        - flicker > 50 with mAP pass: R1a deploy
        - flicker > 50 with mAP fail: R1b cascade (defer/drop)
        - flicker ∈ [20, 50]: R2 defer
        - flicker < 20: R3/R5 drop

    "cascade" means R1 falls through deploy and the row is re-evaluated
    against R2/R3/R4/R5 in order; whichever matches first wins.

    Deploy gate (R1 explicit clauses): row R1 lands ``deploy`` IFF
    ALL of:
      (a) ``eligible_illegal_transition_count == 0`` OR
          (``tracker_artifact_residual == True`` AND
           ``tracker_artifact_evidence_ref`` is a dict with all five
           keys (``trace_id``, ``track_id``, ``frame_start``,
           ``frame_end``, ``reviewer``) non-empty AND ``trace_id``
           resolves to a real track AND ``frame_start <= frame_end``
           in track range);
      (b) ``map_no_regression == True`` within
          ``map_regression_tolerance_pp`` (default 0.2 pp), where the
          checked quantity is ``ap_delta_pp >= -0.2`` evaluated on
          BOTH total mAP AND every safety-class AP whose
          ``full_val_support >= 30`` (each must independently pass).
          ``ap_delta_pp`` is the delta from baseline in percentage
          points; absolute AP values are out of scope.

    Precedence (v2.0 lock): flicker-gate failure (``flicker_improvement_pct
    < 50.0``) dominates illegal-transition elimination. Even when a
    candidate eliminates all illegal transitions
    (``eligible_illegal_transition_count == 0``), it cannot be promoted
    to deploy if flicker_improvement_pct is below 50.0; the row stays
    in the cascade (routes through R2/R3/R5 by flicker band). Adding a
    "safety-override" rule that lets illegal-elimination upgrade the
    decision requires a fresh adversarial review pass — not allowed
    by default.

    Critical contracts:
      - ``baseline_illegal_count == 0`` does NOT raise
        ``executor_error``. The OR-branch is SKIPPED; flicker main
        branch alone decides via cascade.
      - ``tracker_artifact_evidence_ref`` missing / None / dict with
        any empty field / dict whose ``trace_id`` does not resolve /
        whose frame_range is out-of-bounds, when paired with
        ``tracker_artifact_residual == True``, does NOT raise
        ``executor_error``. The exception clause is SKIPPED; the row
        falls through cascade as if exception did not apply.
        Reason code one of: ``tracker_artifact_evidence_missing``,
        ``tracker_artifact_evidence_malformed``,
        ``tracker_artifact_evidence_unverifiable``.

    Args:
        inputs: precomputed metrics from both sides + map_no_regression
            verdict from the upstream eval pipeline + tracker-artifact
            residual flag from the runner.

    Returns:
        ``DecisionResult`` ready to be serialized into one
        ``sensitivity_sweep`` cell of the JSON schema in the module
        docstring. The runner aggregates per-row results into the final
        artifact and selects the headline row from a CLI flag.

    Raises:
        ValueError: malformed ``inputs`` (NaN, missing schema, etc.) is
            surfaced as a return value (``HmmDecision.EXECUTOR_ERROR``)
            with diagnostic notes — NOT raised — so the runner can
            still serialize a complete sweep block. ``ValueError`` is
            reserved for hard programmer errors (wrong dataclass type).
        NotImplementedError: a-stage scaffold.
    """
    raise NotImplementedError("b-stage")
