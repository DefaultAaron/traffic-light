"""Hard-negative mining — training-data-prep-time ablation component (R2,
ablation only).

Authoritative spec: ``docs/planning/additional_components_plan.md`` §四
(硬负样本挖掘). Lifecycle convention:
``docs/planning/additional_components_plan.md`` §一 (a-scaffold →
b-impl → c-ablation → d-decision → e-report).

Scope fence (load-bearing — read before adding any file under this
package):

  * This package is ABLATION-ONLY. The deployment pipeline
    (``inference/`` runtime) is not affected — hard-negative mining is
    a TRAINING-TIME data-prep concern. Default behavior is UNCHANGED:
    the b-stage workflow consumes mined / verified background frames
    via the data.yaml (Ultralytics ``bg/`` directory or DEIM empty-image
    rows). NO main.py training-time CLI flag is required at a-stage —
    the bg/ injection happens through the existing data manifest
    routing primitive (b-stage may add a ``--bg-dir`` shortcut if a
    workflow ergonomic gap surfaces, after a fresh adversarial loop).
  * The pipeline has THREE distinct concerns, all gated by the §4.7
    d-stage decision rule:
      - **Mining** (``modules/miner.py``): run R1 baseline on demo
        videos / R2 self-collected hard scenes; collect FP frames as
        candidates. Output: candidate manifest JSON.
      - **Verification** (``modules/verifier.py``): plan §4.5 risk
        mitigation — every mining batch requires ≥10% human review
        before any candidate enters the bg/ training directory. Output:
        verification report JSON.
      - **Frozen eval manifest** (``data/eval_manifest.py``): the
        post-verification, post-curation FIXED evaluation set
        (``runs/_hard_negative_eval_manifest.json``) — image SHA256 +
        label source + locked thresholds (conf ≥ 0.25, NMS IoU = 0.5).
        Plan §4.7 anti-gaming: this manifest is FROZEN at fit time and
        MUST NOT be modified for the duration of the ablation. Shared
        with §3.7 (copy-paste-balance rare-related FP denominator).

Subpackage contracts (each is a-stage scaffold; b-stage fills bodies):

  ``data``     -- ``FrozenEvalManifest`` typed wrapper around the §4.7
                  manifest with locked thresholds + image SHA256 list +
                  has_real_light flag (separates recall / FP populations).
  ``modules``  -- ``HardNegativeMinerConfig`` + mine_candidates() stub
                  for the candidate-collection pass; ``VerificationConfig``
                  + run_verification() stub for the §4.5 human-review
                  protocol with sample-fraction floor + TPM-rate ceiling.
  ``gates``    -- c-stage acceptance metrics (fp_drop_frac,
                  real_light_recall_delta_pp, total_map_delta_pp) +
                  d-stage 3-case decision rule executor (deploy / defer
                  / drop, with drop as catch-all); emits
                  ``runs/_hard_negative_decision.json``.
  ``runners``  -- 2-arm ablation aggregator: consumes per-arm eval JSONs
                  (no_hn / with_hn) + frozen manifest, applies the gate,
                  emits the decision artifact.

Public API: re-exports below are the FULL public surface — the small
set of types and functions ablation consumers (runners + tests) need at
the top level. Anything not re-exported is package-private and must be
imported via the deep path.
"""

from components.hard_negative_mining.config import (
    HardNegativeMiningYamlConfig,
    load_hard_negative_mining_yaml,
)
from components.hard_negative_mining.data.eval_manifest import (
    FrozenEvalManifest,
    FrozenEvalManifestEntry,
    load_frozen_eval_manifest,
)
from components.hard_negative_mining.gates.ablation_gate import (
    ArmMetrics,
    compute_arm_metrics,
)
from components.hard_negative_mining.gates.decision_gate import (
    ArmId,
    DecisionInputs,
    DecisionResult,
    HardNegativeDecision,
    apply_decision_rule,
)
from components.hard_negative_mining.modules.miner import (
    HardNegativeMinerConfig,
    MiningSourceVideo,
    mine_candidates,
)
from components.hard_negative_mining.modules.verifier import (
    VerificationConfig,
    VerificationSampleProtocol,
    VerificationVerdict,
    run_verification,
)
from components.hard_negative_mining.runners.ablation import (
    AblationConfig,
    assert_artifact_invariants,
    run_ablation,
)

__all__ = [
    # config (YAML loader)
    "HardNegativeMiningYamlConfig",
    "load_hard_negative_mining_yaml",
    # data.eval_manifest
    "FrozenEvalManifest",
    "FrozenEvalManifestEntry",
    "load_frozen_eval_manifest",
    # modules.miner
    "MiningSourceVideo",
    "HardNegativeMinerConfig",
    "mine_candidates",
    # modules.verifier
    "VerificationVerdict",
    "VerificationSampleProtocol",
    "VerificationConfig",
    "run_verification",
    # gates.ablation_gate
    "ArmMetrics",
    "compute_arm_metrics",
    # gates.decision_gate
    "ArmId",
    "HardNegativeDecision",
    "DecisionInputs",
    "DecisionResult",
    "apply_decision_rule",
    # runners.ablation
    "AblationConfig",
    "run_ablation",
    "assert_artifact_invariants",
]
