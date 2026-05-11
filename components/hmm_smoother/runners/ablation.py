"""A/B ablation runner: Plan A (fixed EMA) vs Plan A + HMM.

Drives the full c-stage flow end-to-end:

  1. Fit transition matrix via ``data.transition_counts.estimate_from_jsonl``
     on a held-out FIT split of the replay (separate from the EVAL split,
     so the prior is not estimated and evaluated on the same frames).
  2. Build ``TransitionMatrix`` with the active ``TransitionConfig`` (one
     run per ``α ∈ {0.01, 0.1, 1.0}`` per the plan §2.2 sensitivity sweep).
  3. Replay the EVAL split twice:
       * baseline : argmax of EMA-smoothed ``class_probs`` (plan A,
                     mirroring ``inference.tracker.smoother.TrackSmoother``).
                     We re-implement this offline rather than importing
                     from ``inference.tracker`` to keep the scope fence
                     intact.
       * candidate: HMM (forward-backward OR Viterbi, swept) over the
                     same per-track raw class sequence. Forward-backward
                     posteriors are converted to int sequences via
                     ``modules.inference.posterior_argmax_sequence``
                     before feeding the gate.
     Each side emits one ``TrackSequence`` per track.
  4. Compute ``GateMetrics`` for each side via
     ``gates.ablation_gate.compute_metrics``.
  5. Apply ``gates.decision_gate.apply_decision_rule`` per (α, mode) pair;
     aggregate into the ``sensitivity_sweep`` block of the output JSON
     (see ``gates.decision_gate`` module docstring for the full schema).
  6. Select the headline row from CLI flags
     (``--anchor-alpha`` + ``--anchor-mode``); copy that row's metrics
     and decision into the top-level ``headline_*`` fields. Serialize
     atomically (``.tmp`` → ``rename``).

Sensitivity-sweep contract (plan §2.2):
  * α ∈ {0.01, 0.1, 1.0} are run as separate rows in the output JSON's
    ``sensitivity_sweep`` array — this set is FIXED. The YAML's
    ``laplace_alpha`` MUST be one of these values; ``load_hmm_yaml``
    raises ``ValueError`` on any other value rather than silently
    widening the sweep. If a future plan amendment changes the sweep
    set, this constraint AND ``_hmm_decision_schema.json`` AND the
    YAML comment all need to change in lock-step.
  * Inference-mode sweep (forward_backward, viterbi) is nested under each
    α in the ``modes`` sub-object.
  * Exactly one sweep row sets ``deploy_anchor: true``; that row's α +
    inference-mode pair populates the top-level ``headline_*`` fields.
  * Anchor selection is human-driven post-hoc (the runner reads
    ``--anchor-alpha`` / ``--anchor-mode``); auto-picking is explicitly
    out of scope so b-stage doesn't quietly install a decision policy.
  * ``--anchor-alpha`` must match one of the three sweep values exactly;
    a YAML-supplied default that's not in the set is rejected at load
    time (above), so the CLI value cannot drift from the sweep set.
  * **Anchor exactly-once invariant**: zero or multiple rows with
    ``deploy_anchor=true`` MUST raise ``ValueError`` BEFORE the output JSON is
    written; never warn, never auto-select, never skip output. The runner is
    the only layer that can enforce this — JSON Schema cannot express
    "exactly-one-of-array-elements has property X = true". A silent
    fallback would silently change which row populates the ``headline_*``
    fields, exactly the class of decision-gate drift this scaffold is
    designed to prevent.

Schema-validation contract:

The contract is split into TWO distinct validation layers — they
guard different shapes and CANNOT be conflated:

  * **Per-cell input validation (typed-Python layer).** The metrics
    feeding ``apply_decision_rule`` arrive as ``GateMetrics`` and
    ``DecisionInputs`` dataclasses. Missing fields / wrong types fail
    at dataclass construction (``TypeError``); range-and-finiteness
    checks (NaN, negative counts, > 1.0 fractions) fail at the
    executor's "numeric malformed input" gate and surface as
    ``decision: "executor_error"`` rows with diagnostic ``notes``.
    The JSON Schema file at
    ``components/hmm_smoother/gates/_hmm_decision_schema.json`` is
    NOT used here — it is for the OUTPUT artifact shape, not the
    per-cell input shape.
  * **Output-artifact validation (jsonschema layer).** The runner
    aggregates per-cell ``DecisionResult``s into the
    ``sensitivity_sweep`` array, picks the headline row from the
    ``--anchor-alpha`` / ``--anchor-mode`` flags, and serializes to a
    candidate JSON. BEFORE atomic-rename to
    ``runs/_hmm_decisions.json``, the runner validates the candidate
    against ``_hmm_decision_schema.json`` via ``jsonschema`` (Draft-7,
    matching ``scripts/_r2_decision_schema.json``'s precedent). Output
    validation failure is a HARD error (process exit non-zero), not
    an ``executor_error`` row — by this point the artifact is already
    malformed and there's nothing useful to write.

Why two layers: per-cell inputs and the output artifact have
DIFFERENT shapes (an input is a 5-field metrics record; the output is
a tree with ``headline_*`` + a 3-row sweep). A single schema cannot
guard both.

mAP no-regression contract:
  The runner reads the verdict from a frozen eval-metrics JSON (default
  ``runs/_r2_val_manifest``-derived) rather than a CLI boolean flag, so
  the artifact's ``map_no_regression`` field is derivable, not asserted.
  Schema for the eval JSON: ``{"map_no_regression": bool,
  "tolerance_pp": float}``; b-stage may extend the shape to surface the
  per-class AP details, but the boolean MUST come from a hash-pinned
  artifact path so a stale eval can't silently win the gate.

Scope fence reminder: this runner MUST NOT import anything from
``inference.tracker`` or ``inference.cpp``. The EMA baseline is reimplemented
inline (fixed α, no ByteTrack — IDs come from the JSONL replay).

Scaffold (a-stage): public CLI entry signature only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AblationConfig:
    """Top-level knobs for one ablation run; derived from CLI + YAML."""

    fit_jsonl: Path                 # JSONL replay used to fit transitions
    eval_jsonl: Path                # JSONL replay used to evaluate metrics
    config_yaml: Path               # configs/temporal_hmm.yaml
    eval_metrics_json: Path         # frozen mAP-no-regression artifact
    output_json: Path               # runs/_hmm_decisions.json
    anchor_alpha: float             # which sweep row is the headline
    anchor_mode: str                # "forward_backward" or "viterbi"


def run_ablation(config: AblationConfig) -> None:
    """End-to-end ablation: fit → replay both sides → metrics → decision.

    Side effects:
        * Reads ``config.fit_jsonl`` and ``config.eval_jsonl``.
        * Reads ``config.config_yaml``.
        * Reads ``config.eval_metrics_json`` (mAP no-regression verdict
          + tolerance), records its SHA256 in the output JSON.
        * **Conditionally** reads the YAML's ``transition_matrix_path``
          (when non-null) and validates existence + readability + shape
          + row-stochastic + illegal-cell-policy compatibility BEFORE
          replay (filesystem checks live here, not in
          ``HmmYamlConfig.__post_init__``; missing or unreadable ``.npy``
          produces a diagnostic ``decision: "executor_error"`` row with the
          path in ``notes``).
        * Writes ``config.output_json`` (atomic: ``.tmp`` → ``rename``).

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

    Argparse contract (b-stage):
        --fit FILE              JSONL replay for transition fitting
        --eval FILE             JSONL replay for evaluation
        --config FILE           configs/temporal_hmm.yaml path
        --eval-metrics-json FILE
                                 frozen JSON with
                                 ``{"map_no_regression": bool,
                                   "tolerance_pp": float}``;
                                 SHA256 captured in output
        --output FILE           runs/_hmm_decisions.json path
        --anchor-alpha FLOAT    which sweep row populates headline_*;
                                must match one of {0.01, 0.1, 1.0}
        --anchor-mode {forward_backward,viterbi}
                                which inference mode populates headline_*

    Returns:
        process exit code: 0 on AGREED-CLEAN run, 1 on
        ``executor_error`` in the headline row, 2 on
        ``NotImplementedError`` from any a-stage stub (so scaffold-time
        smoke tests don't silently exit 0). b-stage MUST replace the
        ``NotImplementedError`` catch with real error paths before
        wiring to CI.

    Raises:
        NotImplementedError: a-stage scaffold.
    """
    raise NotImplementedError("b-stage")


if __name__ == "__main__":
    raise SystemExit(main())
