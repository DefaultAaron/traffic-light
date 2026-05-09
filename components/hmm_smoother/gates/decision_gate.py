"""d-stage 4-case decision rule for the HMM ablation.

Mirrors the structure of ``scripts/_r2_decide_precision.py`` (precision plan
locked iter-3) and ``components/knowledge_distillation/gates/`` decision
helpers: pre-committed thresholds, first-match cascade with explicit
catch-all, executor-error sink reserved for malformed inputs only.

Plan §2.2.1 d cases (verbatim — do NOT widen thresholds during impl, do
NOT silently re-order; threshold knobs may be moved into
``configs/temporal_hmm.yaml`` only after a fresh adversarial loop):

  * deploy : flicker rate −50% AND illegal-transition count → 0 AND
              total mAP / safety-class AP no-regression
              (``map_regression_tolerance_pp`` from config, default 0.2 pp).
              Tracker-artifact exception: residual illegal transitions
              fully traceable to ByteTrack ID switch / short-track noise
              still pass under deploy + a notes field; the runner is
              responsible for surfacing the trace and setting the
              ``tracker_artifact_residual`` input flag.

  * defer  : flicker rate improves 20–50% (closed interval, both ends
              inclusive) OR illegal-transition rate
              (count / total_transitions) improves ≥ 50% but is not 0.
              Action: escalate to §2.3 AdaEMA per the §2.1 ladder.

  * drop   : catch-all. Any legitimate input (i.e. non-malformed) that
              does NOT match deploy or defer falls into drop. This
              includes the plan's literal drop case (flicker improvement
              < 20% AND illegals eliminated only by hard_zero mask) AND
              residual cases the plan §2.2.1 d does not enumerate
              explicitly (e.g. flicker improves ≥ 50% but mAP regresses,
              or flicker 0%-20% with non-mask illegal elimination).
              Default escalation: §2.3 AdaEMA; direct jump to §2.4 GRU /
              §2.5 Transformer requires explicit failure-mode evidence
              in ``escalation_rationale``.

Boundary semantics (closed-interval convention, 2026-05-09 plan amendment
following B2 review C1):
  * flicker improvement = 20%  → defer (lower bound of [20, 50])
  * flicker improvement = 50%  → defer (upper bound of [20, 50])
  * illegal-rate improvement = 50%  → defer

Executor semantics: cases are evaluated in order ``deploy → defer → drop``;
first match wins. ``drop`` is the unconditional catch-all for legitimate
input; ``executor_error`` is reserved for **malformed input only**.

Boundary between the runner+schema layer and the executor (C3 NEW-MAJOR 3
2026-05-09): ``apply_decision_rule`` accepts pre-constructed dataclasses
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
  * ``total_transitions == 0`` on either side (zero-divide on illegal rate)
  * ``baseline.total_transitions != candidate.total_transitions``
    (denominator drift; same eligible-tracks rule on the same EVAL
    JSONL must produce the same denominator regardless of smoother)
  * ``baseline.eligible_track_count != candidate.eligible_track_count``
    (C3 iter-2 NEW-MAJOR 3: same hazard for the track-count
    denominator; same EVAL JSONL → same eligible track membership)
  * ``baseline.total_track_count != candidate.total_track_count``
    (C3 iter-2 NEW-MAJOR 3: same hazard for the total-track count)
  * Threshold ranges out of bounds (negative counts, > 1.0 fractions)

Cases are NOT mutually exclusive in the plan-prose sense (defer's flicker
condition and drop's catch-all overlap by design once you accept first-match
cascade). The earlier "exactly one matches" claim is replaced by:
"first-match cascade with explicit precedence; drop is a structural
catch-all, not a conditional case."

Schema for ``runs/_hmm_decisions.json`` (plan §2.2.1 d output, with
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
        "baseline_total_transitions": int,           # C3 iter-1 NEW-MAJOR 2:
        "candidate_total_transitions": int,          # split denominator;
                                                     # executor asserts ==.
        "baseline_eligible_track_count": int,        # C3 iter-2 NEW-MAJOR 3:
        "candidate_eligible_track_count": int,       # same hazard for the
        "baseline_total_track_count": int,           # eligible + total
        "candidate_total_track_count": int,          # track-count fields.
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
            "forward_backward": {                # ModeCell — C3 iter-2
              "metrics": { ... headline_metrics shape ... },
              "decision": "deploy" | "defer" | "drop" | "executor_error",
              "deploy_via_tracker_artifact_exception": bool,
              "escalation": "ada_ema" | "gru" | "transformer" | null,
              "escalation_rationale": str,        # NEW-MAJOR 2: diagnostic
              "notes": str                        # fields preserved so
            },                                    # non-anchor cells stay
            "viterbi": {                          # auditable.
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
``laplace_alpha ∈ {0.01, 0.1, 1.0}`` (plan §2.2 conflictor iter-2 sweep).
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
``components/hmm_smoother/gates/_hmm_decision_schema.json`` (C3
iter-1 NEW-MAJOR 1 2026-05-09).

Validation boundary (C3 iter-2 NEW-MAJOR 1, refined by iter-3 NEW-MAJOR 1
2026-05-09 — earlier prose said the schema validated raw inputs, which
contradicts the iter-2 boundary split): the JSON Schema is **OUTPUT-ONLY**.
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

    ``total_transitions`` is exposed on ``GateMetrics`` (B2 review I1) and
    is read off the candidate side here; it MUST be > 0 or the executor
    returns ``executor_error`` (would zero-divide the illegal-rate gate).
    """

    baseline: GateMetrics
    candidate: GateMetrics
    map_no_regression: bool
    tracker_artifact_residual: bool   # see plan §2.2.1 d deploy-exception


@dataclass(frozen=True)
class DecisionResult:
    """One row of the ``runs/_hmm_decisions.json`` sweep.

    ``deploy_via_tracker_artifact_exception`` (B2 review C3 2026-05-09)
    is a first-class boolean so the deploy-via-exception path is
    grep-able / tabulatable across runs. It MUST be ``False`` whenever
    ``decision != DEPLOY``; the executor enforces this.
    """

    decision: HmmDecision
    deploy_via_tracker_artifact_exception: bool
    escalation: HmmEscalation | None
    escalation_rationale: str
    notes: str


def apply_decision_rule(inputs: DecisionInputs) -> DecisionResult:
    """Evaluate the §2.2.1 d 4-case rule on a single A/B run.

    Executor semantics (B2 review C1 2026-05-09): first-match cascade
    over ``deploy → defer → drop``; drop is the unconditional catch-all
    for legitimate input. ``executor_error`` fires only on malformed
    input (see module docstring "Executor semantics" block).

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
