"""Copy-paste augmentation + class-balanced loss — training-time ablation
component (R2 training, ablation only).

Authoritative spec: ``docs/planning/additional_components_plan.md`` §三
(Copy-paste 增强 + 类平衡损失). Lifecycle convention:
``docs/planning/additional_components_plan.md`` §一 (a-scaffold → b-impl →
c-ablation → d-decision → e-report).

Scope fence (load-bearing — read before adding any file under this package):

  * This package is ABLATION-ONLY. The deployment pipeline
    (``inference/`` runtime) is not affected — copy-paste and
    class-balance are training-time concerns. Default behavior is
    UNCHANGED: only when ``main.py train --copy-paste`` or ``--cls-weight``
    is supplied does the active training run consume any knob from this
    package. The ablation arm comparison happens by post-hoc aggregating
    metrics from already-trained checkpoints — this package does NOT
    drive training itself.
  * Two distinct mechanisms live here, both gated by the §三 d-stage
    decision rule together (because the plan deliberately couples them
    — copy-paste increases TP on rare classes, class-balance reweights
    the loss; deploy/defer/drop is a single decision over the joint
    effect):
      - **Copy-paste**: bbox-level paste of rare-class instances onto
        any background image with a y-center mask constraint (plan §3.5
        risk: pasted lights "floating in the sky"). Ultralytics ships
        a built-in ``copy_paste=`` flag; DEIM needs a custom dataloader
        hook.
      - **Class-balanced loss**: per-class loss reweighting via the
        Cui et al. 2019 effective-number-of-samples formula. β sweep
        over {0.0, 0.99, 0.999, 0.9999} (β=0 is uniform / baseline).

Subpackage contracts (each is a-stage scaffold; b-stage fills bodies):

  ``modules``  -- CopyPasteAugment (paste op + y-center mask + interaction
                  locks for fliplr=0 and mosaic) + ClassBalanceWeights
                  (Cui 2019 effective-number-of-samples formula).
  ``data``     -- Per-class instance count fitter from a YOLO/DEIM data
                  manifest → ``configs/data_R2_class_weights.yaml``.
  ``gates``    -- c-stage acceptance gate (rare-class AP delta, rare safety
                  AP delta, rare-related FP delta, total mAP no-regression)
                  + d-stage 4-case decision rule executor; emits
                  ``runs/_copy_paste_decision.json``.
  ``runners``  -- Ablation aggregator: consumes per-arm eval JSONs from
                  three already-trained runs (no_aug / cp_only / cp_balanced)
                  + the cp_balanced β sensitivity sweep, applies the gate,
                  emits the decision artifact.

Public API: re-exports below are the FULL public surface — the small set
of types and functions ablation consumers (runners + tests) need at the
top level. Anything not re-exported is package-private and must be
imported via the deep path.
"""

from components.copy_paste_balance.config import (
    CopyPasteBalanceYamlConfig,
    load_copy_paste_balance_yaml,
)
from components.copy_paste_balance.data.class_weights import (
    ClassCountsTable,
    estimate_from_dataset,
)
from components.copy_paste_balance.gates.ablation_gate import (
    ArmMetrics,
    PerClassAP,
    compute_arm_metrics,
)
from components.copy_paste_balance.gates.decision_gate import (
    ArmId,
    CopyPasteDecision,
    DecisionInputs,
    DecisionResult,
    apply_decision_rule,
)
from components.copy_paste_balance.modules.class_balance import (
    ClassBalanceApplyMode,
    ClassBalanceWeights,
)
from components.copy_paste_balance.modules.copy_paste import (
    CopyPasteAugment,
    CopyPasteConfig,
)
from components.copy_paste_balance.runners.ablation import (
    AblationConfig,
    assert_artifact_invariants,
    run_ablation,
)

__all__ = [
    # config (YAML loader)
    "CopyPasteBalanceYamlConfig",
    "load_copy_paste_balance_yaml",
    # modules.copy_paste
    "CopyPasteConfig",
    "CopyPasteAugment",
    # modules.class_balance
    "ClassBalanceApplyMode",
    "ClassBalanceWeights",
    # data.class_weights
    "ClassCountsTable",
    "estimate_from_dataset",
    # gates.ablation_gate
    "PerClassAP",
    "ArmMetrics",
    "compute_arm_metrics",
    # gates.decision_gate
    "ArmId",
    "CopyPasteDecision",
    "DecisionInputs",
    "DecisionResult",
    "apply_decision_rule",
    # runners.ablation
    "AblationConfig",
    "run_ablation",
    "assert_artifact_invariants",
]
