"""HMM smoother — post-detector temporal smoother (R2/R3 optional, ablation only).

Authoritative spec: ``docs/planning/temporal_optimization_plan.md`` §2.2 +
§2.2.1 (HMM lifecycle rollup; scaffold redirects from ``inference/temporal/``
→ ``components/hmm_smoother/``). Lifecycle convention:
``docs/planning/additional_components_plan.md`` §一 (a-scaffold → b-impl →
c-ablation → d-decision → e-report).

Scope fence (load-bearing — read before adding any file under this package):

  * This package is ABLATION-ONLY. It MUST NOT import from ``inference.tracker``
    or ``inference.cpp``, and the deployment pipeline (``inference/tracker/
    smoother.py`` ByteTrack + fixed-EMA) MUST stay byte-for-byte unchanged.
  * Inputs are JSONL replays from ``demo.cpp`` / ``inference.demo`` ``--json``
    output (one record per ``(frame, tracking_id)`` with ``raw_class_id``,
    ``raw_confidence``, and — when present — ``class_probs``). The HMM consumes
    track-level sequences offline and emits a smoothed class assignment per
    frame for A/B comparison against the EMA baseline.
  * If §2.2 决策规则 decides ``deploy``, the deployment-pipeline integration
    (originally targeted at ``inference/temporal/``) is opened as a SEPARATE
    round. Nothing in this package fast-paths into ``inference/`` — that path
    is gated on a successful ablation deploy decision.

Subpackage contracts (each is a-stage scaffold; b-stage fills bodies):

  ``modules``  -- TransitionMatrix (Laplace α, illegal-mask), ObservationModel
                  (detector softmax → emission likelihood), and inference
                  algorithms (forward-backward + Viterbi +
                  posterior-argmax adapter).
  ``data``     -- Offline transition-count estimation from JSONL replays
                  (Plan A baseline track sequences → C×C frequency matrix).
  ``gates``    -- c-stage acceptance gates (flicker rate, illegal-transition
                  count, mAP-no-regression) + d-stage 4-case decision rule
                  executor; emits ``runs/_hmm_decisions.json``.
  ``runners``  -- A/B ablation driver: replays JSONL through the EMA baseline
                  and through HmmSmoother, computes both metric streams, and
                  hands them to ``gates``.

Public API: re-exports below are the FULL public surface — the small set of
types and functions that ablation runners + tests will need at the top level.
Anything not re-exported here is package-private and must be imported via the
deep path (``components.hmm_smoother.<sub>``). This list is exhaustive; do NOT
add ad-hoc top-level imports without also adding them to ``__all__``.
"""

from components.hmm_smoother.config import (
    HmmYamlConfig,
    load_hmm_yaml,
)
from components.hmm_smoother.data.transition_counts import (
    GapPolicy,
    estimate_from_jsonl,
)
from components.hmm_smoother.gates.ablation_gate import (
    GateMetrics,
    TrackSequence,
    compute_metrics,
)
from components.hmm_smoother.gates.decision_gate import (
    DecisionInputs,
    DecisionResult,
    HmmDecision,
    HmmEscalation,
    apply_decision_rule,
)
from components.hmm_smoother.modules.inference import (
    HmmInferenceMode,
    forward_backward,
    posterior_argmax_sequence,
    viterbi,
)
from components.hmm_smoother.modules.observation import (
    ObservationMode,
    ObservationModel,
)
from components.hmm_smoother.modules.transition import (
    IllegalTransitionPolicy,
    TransitionConfig,
    TransitionMatrix,
)
from components.hmm_smoother.runners.ablation import (
    AblationConfig,
    run_ablation,
)

__all__ = [
    # modules.transition
    "TransitionMatrix",
    "TransitionConfig",
    "IllegalTransitionPolicy",
    # modules.observation
    "ObservationModel",
    "ObservationMode",
    # modules.inference
    "HmmInferenceMode",
    "forward_backward",
    "viterbi",
    "posterior_argmax_sequence",
    # data.transition_counts
    "estimate_from_jsonl",
    "GapPolicy",
    # config (YAML loader)
    "HmmYamlConfig",
    "load_hmm_yaml",
    # gates.ablation_gate
    "TrackSequence",
    "GateMetrics",
    "compute_metrics",
    # gates.decision_gate
    "HmmDecision",
    "HmmEscalation",
    "DecisionInputs",
    "DecisionResult",
    "apply_decision_rule",
    # runners.ablation
    "AblationConfig",
    "run_ablation",
]
