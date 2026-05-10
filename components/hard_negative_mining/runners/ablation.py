"""2-arm ablation aggregator: no_hn vs with_hn.

Drives the d-stage flow end-to-end:

  0. Cross-artifact equality (the runner is the only layer that holds
     all of: config YAML + frozen FP manifest file + per-arm eval JSONs).
     Hard-fail unless ALL of:

       * ``frozen_manifest.manifest_sha256`` matches the on-disk file's
         actual SHA256 (anti-tamper at fit time vs eval time)
       * ``frozen_manifest.manifest_sha256`` matches the
         ``fp_manifest_sha256`` recorded in BOTH per-arm eval JSONs
       * ``config.data_yaml_sha256`` (loaded YAML field) matches the
         ``data_yaml_sha256`` recorded in both per-arm eval JSONs
       * The two eval JSONs reference the SAME ``eval_manifest_sha256``
         (frozen R2 val manifest)
       * The two eval JSONs reference the SAME
         ``map_regression_tolerance_pp`` (runner-side knob; per-arm
         tolerance drift is a contract violation)
       * The no_hn eval has ``map_no_regression=True`` (self-comparison;
         hard-fail otherwise)

     Without this, a stale class-counts file or a mismatched manifest
     silently produces a wrong-arm decision — exactly the §4.7 anti-
     gaming hazard.

  1. Load per-arm eval JSONs from already-trained runs:
       * ``no_hn``    — baseline (no hard-neg bg/ injection)
       * ``with_hn``  — candidate (hard-neg bg/ injection ON)
     Each eval JSON has shape:

       ``{"fp_count": int,                    # on the §4.7 manifest
          "real_light_recall": float,         # on the manifest's has_real_light=True subset
          "total_mAP_at_0_5": float,          # on the frozen R2 val set
          "data_yaml_sha256": str,            # 64-char lowercase hex; REQUIRED
          "eval_manifest_sha256": str,        # frozen R2 val manifest hash
          "fp_manifest_sha256": str,          # frozen §4.7 manifest hash
          "map_no_regression": bool,          # THIS arm vs no_hn; baseline=True trivially
          "map_regression_tolerance_pp": float}``

  2. Build ``ArmMetrics`` for the candidate cell by paired diff
     against the baseline arm.

  3. Apply ``apply_decision_rule`` to the with_hn cell.

  4. Anchor selection: ``--anchor-arm with_hn`` is the ONLY legal value.
     ``--anchor-arm no_hn`` is REJECTED at AblationConfig construction
     (the baseline cell carries no decision; choosing it as the headline
     would serialize an executor_error row at the top level — meaningless
     for a deployment-policy artifact).

  5. Validate against ``_hard_negative_decision_schema.json``;
     ``assert_artifact_invariants`` for cross-row checks the schema
     can't express; atomic-rename ``.tmp`` → final path.

Schema-validation contract (mirrors §3.7 boundary split):

  * **Per-cell input validation (typed-Python layer)**: ``ArmMetrics``
    and ``DecisionInputs`` validate at construction; numeric malformed
    values surface as ``executor_error`` rows.
  * **Output-artifact validation (jsonschema layer)**: BEFORE atomic-
    rename, validate via ``jsonschema`` (Draft-7). Output validation
    failure is a HARD error (process exit non-zero).

mAP no-regression contract: ``map_no_regression`` is per-arm (different
arms have different total_mAP deltas; one global verdict cannot stamp
both arms). Each eval JSON carries its own verdict pre-computed by the
trainer's upstream eval pipeline against the shared tolerance. The
runner verifies cross-arm tolerance consistency and that the no_hn
eval is self-True.

Scope fence reminder: this runner MUST NOT import anything from
``inference.tracker`` or ``inference.cpp`` (parallel to §3.7 fence).
The arms are post-hoc compared from already-trained eval JSONs; the
runner does NOT drive training and does NOT drive mining (mining lives
in ``modules/miner.py`` and runs as a separate b-stage step).

Scaffold (a-stage): public CLI entry signature only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class AblationConfig:
    """Top-level knobs for one ablation run; derived from CLI + YAML.

    Mirrors ``components/copy_paste_balance/runners/ablation.AblationConfig``
    structurally but with the 2-arm shape: only ``no_hn_eval_json`` +
    ``with_hn_eval_json`` instead of the 5-eval cp_balanced sweep.

    Cross-knob invariants enforced in ``__post_init__``:
      * ``anchor_arm`` must be ``"with_hn"`` (the only legal anchor).
        ``"no_hn"`` is REJECTED — baseline carries no decision.
      * Path-vs-str type drift on every Path-typed field (mirrors §3.7
        AblationConfig discipline).
      * **B2 review I4 2026-05-10**: every pair of distinct Path fields
        is enforced to refer to distinct paths. ``output_json`` equal to
        any input path would clobber the input mid-write; equal pairs of
        input paths would silently collapse two arms onto one eval
        source.
      * No filesystem-existence check (HMM iter-2 NEW-MINOR convention;
        the runner enforces existence at consumption time and surfaces
        missing paths as ``decision: "executor_error"`` rows).
    """

    no_hn_eval_json: Path                       # baseline arm eval output (carries map_no_regression=True)
    with_hn_eval_json: Path                     # candidate arm eval output (carries its own map_no_regression)
    config_yaml: Path                           # configs/hard_negative_mining.yaml
    frozen_manifest_json: Path                  # runs/_hard_negative_eval_manifest.json
    output_json: Path                           # runs/_hard_negative_decision.json
    anchor_arm: str                             # MUST be "with_hn"

    _ALLOWED_ANCHOR_ARMS: ClassVar[tuple[str, ...]] = ("with_hn",)

    def __post_init__(self) -> None:
        # Path-vs-str guard on every Path field. Loader / CLI parser is
        # responsible for str→Path coercion.
        path_fields = (
            ("no_hn_eval_json", self.no_hn_eval_json),
            ("with_hn_eval_json", self.with_hn_eval_json),
            ("config_yaml", self.config_yaml),
            ("frozen_manifest_json", self.frozen_manifest_json),
            ("output_json", self.output_json),
        )
        for name, value in path_fields:
            if not isinstance(value, Path):
                raise ValueError(
                    f"{name} must be Path; got "
                    f"{type(value).__name__}={value!r} "
                    f"(loader / CLI must coerce str → pathlib.Path before construction)"
                )
        # B2 review I4 2026-05-10: reject path collisions across the 5
        # Path-typed fields. ``output_json == any-input-path`` is the
        # catastrophic case; ``no_hn_eval_json == with_hn_eval_json`` is
        # the silent-arm-collapse case. A generic "all distinct" check
        # is cheaper than enumerating dangerous pairs.
        path_values = [value for _, value in path_fields]
        if len(set(path_values)) != len(path_values):
            duplicates = sorted({
                str(p) for p in path_values
                if path_values.count(p) > 1
            })
            raise ValueError(
                f"AblationConfig path fields must all be distinct; got "
                f"{len(path_values)} fields collapsing to "
                f"{len(set(path_values))} unique paths "
                f"(duplicates: {duplicates}). Common offenders: "
                f"output_json equal to any input path (would clobber the "
                f"input mid-write); no_hn_eval_json == with_hn_eval_json "
                f"(would silently collapse the two arms onto one eval "
                f"source)."
            )
        # anchor_arm: string in {"with_hn"} only. Module prose explicitly
        # REJECTS "no_hn" — mirrored at the dataclass boundary.
        if not isinstance(self.anchor_arm, str):
            raise ValueError(
                f"anchor_arm must be str; got "
                f"{type(self.anchor_arm).__name__}={self.anchor_arm!r}"
            )
        if self.anchor_arm not in self._ALLOWED_ANCHOR_ARMS:
            raise ValueError(
                f"anchor_arm must be one of {self._ALLOWED_ANCHOR_ARMS}; "
                f"got {self.anchor_arm!r} "
                f"(no_hn is REJECTED — the baseline cell carries no decision)"
            )


def run_ablation(config: AblationConfig) -> None:
    """End-to-end ablation aggregator: read eval JSONs → metrics → decision.

    Side effects:
        * Reads both eval JSONs.
        * Reads ``config.config_yaml`` + ``config.frozen_manifest_json``;
          records both SHA256 hashes in the output (config_yaml_sha256
          + fp_manifest_sha256).
        * Validates Step 0 cross-artifact equality:
          - frozen_manifest_json's actual SHA256 == its self-recorded hash
          - both eval JSONs' fp_manifest_sha256 == frozen_manifest's hash
          - both eval JSONs' eval_manifest_sha256 match each other
          - both eval JSONs' data_yaml_sha256 match each other
          - both eval JSONs' map_regression_tolerance_pp match each other
          - no_hn eval's map_no_regression == True
        * Builds ArmMetrics for both arms (baseline self-comparison +
          candidate diff).
        * Applies the decision rule to the with_hn cell.
        * Validates the candidate output against the JSON Schema.
        * Calls ``assert_artifact_invariants`` for cross-row checks.
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

    Argparse contract (b-stage):
        --no-hn-eval FILE       baseline arm's eval JSON (must carry
                                 map_no_regression=True)
        --with-hn-eval FILE     candidate arm's eval JSON
        --config FILE           configs/hard_negative_mining.yaml path
        --frozen-manifest FILE  runs/_hard_negative_eval_manifest.json path
        --output FILE           runs/_hard_negative_decision.json path
        --anchor-arm {with_hn}  REQUIRED; argparse enforces single-value
                                 choice (with_hn). The single-element
                                 enum is intentional defense (mirrors
                                 the schema's single-element
                                 headline_arm enum) — making the
                                 constraint explicit at the CLI is the
                                 documentation that no_hn is rejected.

    a-stage scaffold behavior: at this layer the function raises
    ``NotImplementedError`` (process exit code 1 — Python's default for
    uncaught exceptions). b-stage will replace the body with the
    aggregator implementation; b-stage's contract (NOT the a-stage
    behavior) is documented below for forward reference:

      Returns (b-stage):
        0 — AGREED-CLEAN run; output JSON written.
        1 — ``executor_error`` in the headline row (output JSON still
            written but contains diagnostic notes).
        2 — internal contract violation (e.g. schema validation failure
            after ``assert_artifact_invariants`` passed; should be
            unreachable in practice).

    Raises:
        NotImplementedError: a-stage scaffold (b-stage replaces the body).
    """
    raise NotImplementedError("b-stage")


def assert_artifact_invariants(artifact: dict) -> None:
    """Cross-row invariants that JSON Schema cannot express.

    Mirrors ``components/copy_paste_balance/runners/ablation.assert_artifact_invariants``
    but for the 2-arm shape (no β sweep, single candidate cell).

    Implemented (not stubbed) because cross-row invariants are pure
    dict→raise logic — no I/O, no infrastructure — and a stub would
    defeat the whole point of the cross-row gate. ``run_ablation``
    body remains a stub; b-stage MUST call this helper AFTER schema
    validation passes and BEFORE atomic-rename.

    **Precondition (B2 review I11 2026-05-10)**: the caller MUST have
    already passed ``artifact`` through ``jsonschema.validate(...,
    _hard_negative_decision_schema)`` BEFORE invoking this helper.
    Missing required keys would otherwise raise raw ``KeyError`` from
    the dict accesses below — this helper documents ``ValueError`` as
    the exception class but the precondition is what makes that
    contract honest. ``run_ablation`` enforces the precondition by
    construction; consumers calling this helper directly (e.g. from
    a test fixture) MUST schema-validate first.

    Invariants enforced:

      1. ``headline_arm == "with_hn"``: the schema's enum already
         single-values this, but check defensively at the dict layer in
         case the artifact arrives pre-schema.
      2. ``headline_metrics == with_hn.metrics``: the headline is a
         copy of the with_hn cell, not an independent computation.
      3. ``headline_decision == with_hn.decision``: same.
      4. **Cross-block equality** (canonical = no_hn.metrics):
         every metrics block in the artifact (headline_metrics,
         with_hn.metrics) must agree with no_hn.metrics on:
           * ``eval_manifest_sha256``
           * ``fp_manifest_sha256``
           * ``data_yaml_sha256``
           * ``map_regression_tolerance_pp``
         Per-cell drift on any of these silently corrupts the
         deploy/defer/drop interpretation across arms.
      5. **Top-level data_yaml_sha256 equality**:
         ``artifact["data_yaml_sha256"] == no_hn.metrics.data_yaml_sha256``.
         Three-way equality (top-level / no_hn / with_hn) closes the
         class-label drift gap.
      6. **Top-level fp_manifest_sha256 equality**:
         ``artifact["fp_manifest_sha256"] == no_hn.metrics.fp_manifest_sha256``.
         The §4.7 frozen manifest is the FP-denominator anti-gaming
         safeguard; the top-level hash MUST match the per-cell hash.
      7. **Top-level eval_manifest_sha256 equality**:
         ``artifact["eval_manifest_sha256"] == no_hn.metrics.eval_manifest_sha256``.
      8. **B2 review C5 2026-05-10 — headline-vs-with_hn map_no_regression
         equality**: ``headline_metrics.map_no_regression ==
         with_hn.metrics.map_no_regression``. Today this is redundant
         with invariant #2 (headline_metrics is dict-equal to
         with_hn.metrics, which implies every field including
         map_no_regression matches). The redundancy is INTENTIONAL —
         a future b-stage refactor that loosens #2 to "headline derived
         from with_hn but with optional override" silently re-opens
         the verdict-mismatch gap; this explicit invariant survives that
         refactor.
      9. **Decision-vs-verdict consistency**: per plan §4.7, both
         deploy and defer require total mAP no-regression. A cell with
         ``decision in {"deploy", "defer"}`` AND
         ``metrics.map_no_regression == False`` is malformed.

    Args:
        artifact: the candidate output dict, post-aggregation,
            post-schema-validation.

    Raises:
        ValueError: any cross-row invariant fails. (Missing required
            keys raise ``KeyError`` instead — see Precondition above.)
    """
    # Headline-arm defensive check: schema is single-value, but verify
    # at the dict layer in case this is called pre-schema.
    headline_arm = artifact["headline_arm"]
    if headline_arm != "with_hn":
        raise ValueError(
            f"headline_arm must be 'with_hn'; got {headline_arm!r} "
            f"(no_hn is REJECTED — baseline carries no decision)"
        )

    # Headline copy invariants.
    with_hn = artifact["with_hn"]
    if artifact["headline_metrics"] != with_hn["metrics"]:
        raise ValueError(
            "headline_metrics must equal with_hn.metrics (the headline is a "
            "copy of the with_hn cell, not an independent computation)"
        )
    if artifact["headline_decision"] != with_hn["decision"]:
        raise ValueError(
            f"headline_decision ({artifact['headline_decision']!r}) must "
            f"equal with_hn.decision ({with_hn['decision']!r})"
        )

    # B2 review C5 2026-05-10: explicit map_no_regression cross-cell
    # equality. Today redundant with the dict-eq above; survives a
    # future refactor that loosens the dict-eq to a partial copy.
    if artifact["headline_metrics"]["map_no_regression"] != with_hn["metrics"]["map_no_regression"]:
        raise ValueError(
            f"headline_metrics.map_no_regression "
            f"({artifact['headline_metrics']['map_no_regression']!r}) must "
            f"equal with_hn.metrics.map_no_regression "
            f"({with_hn['metrics']['map_no_regression']!r}) — verdict-mismatch "
            f"between headline and with_hn cell silently corrupts the "
            f"decision-vs-verdict consistency check"
        )

    # Cross-block equality: canonical = no_hn.metrics. Every block must
    # agree on the manifest hashes + tolerance + data_yaml hash. Per-cell
    # drift silently corrupts the rule's interpretation across arms.
    no_hn_metrics = artifact["no_hn"]["metrics"]
    canonical_eval_manifest = no_hn_metrics["eval_manifest_sha256"]
    canonical_fp_manifest = no_hn_metrics["fp_manifest_sha256"]
    canonical_data_yaml = no_hn_metrics["data_yaml_sha256"]
    canonical_tolerance = no_hn_metrics["map_regression_tolerance_pp"]

    metrics_blocks = [
        ("headline_metrics", artifact["headline_metrics"]),
        ("with_hn.metrics", with_hn["metrics"]),
    ]
    for path, metrics in metrics_blocks:
        if metrics["eval_manifest_sha256"] != canonical_eval_manifest:
            raise ValueError(
                f"{path}.eval_manifest_sha256 must match "
                f"no_hn.metrics.eval_manifest_sha256 "
                f"(canonical={canonical_eval_manifest}); got "
                f"{metrics['eval_manifest_sha256']} (every cell must consume "
                f"the SAME frozen R2 val manifest — drift across cells "
                f"corrupts the AP-delta comparison the rule operates on)"
            )
        if metrics["fp_manifest_sha256"] != canonical_fp_manifest:
            raise ValueError(
                f"{path}.fp_manifest_sha256 must match "
                f"no_hn.metrics.fp_manifest_sha256 "
                f"(canonical={canonical_fp_manifest}); got "
                f"{metrics['fp_manifest_sha256']} (the §4.7 frozen FP "
                f"manifest is the anti-gaming safeguard; per-cell drift is "
                f"a contract violation)"
            )
        if metrics["data_yaml_sha256"] != canonical_data_yaml:
            raise ValueError(
                f"{path}.data_yaml_sha256 must match "
                f"no_hn.metrics.data_yaml_sha256 "
                f"(canonical={canonical_data_yaml}); got "
                f"{metrics['data_yaml_sha256']} (class-label drift across "
                f"cells silently corrupts AP delta interpretation)"
            )
        if metrics["map_regression_tolerance_pp"] != canonical_tolerance:
            raise ValueError(
                f"{path}.map_regression_tolerance_pp must match "
                f"no_hn.metrics.map_regression_tolerance_pp "
                f"(canonical={canonical_tolerance}); got "
                f"{metrics['map_regression_tolerance_pp']} (tolerance is a "
                f"runner-side knob; the same value must stamp every cell)"
            )

    # Top-level / per-cell three-way equality on the manifest + data_yaml
    # hashes. Without these, a hand-built artifact could pair a top-level
    # hash with mismatched per-cell hashes and silently corrupt provenance.
    if artifact["data_yaml_sha256"] != canonical_data_yaml:
        raise ValueError(
            f"artifact.data_yaml_sha256 ({artifact['data_yaml_sha256']}) must "
            f"equal no_hn.metrics.data_yaml_sha256 ({canonical_data_yaml}) — "
            f"top-level config hash and per-cell eval hash must agree"
        )
    if artifact["fp_manifest_sha256"] != canonical_fp_manifest:
        raise ValueError(
            f"artifact.fp_manifest_sha256 ({artifact['fp_manifest_sha256']}) "
            f"must equal no_hn.metrics.fp_manifest_sha256 "
            f"({canonical_fp_manifest}) — the §4.7 frozen manifest hash is "
            f"the load-bearing anti-gaming check"
        )
    if artifact["eval_manifest_sha256"] != canonical_eval_manifest:
        raise ValueError(
            f"artifact.eval_manifest_sha256 "
            f"({artifact['eval_manifest_sha256']}) must equal "
            f"no_hn.metrics.eval_manifest_sha256 "
            f"({canonical_eval_manifest})"
        )

    # Decision-vs-verdict consistency: deploy/defer require map_no_regression.
    # drop is compatible with regression (the catch-all sweeps a regressing-
    # mAP candidate into drop).
    decision = with_hn["decision"]
    metrics = with_hn["metrics"]
    if decision in ("deploy", "defer") and metrics["map_no_regression"] is not True:
        raise ValueError(
            f"with_hn.decision={decision!r} is incompatible with "
            f"metrics.map_no_regression={metrics['map_no_regression']!r} — "
            f"plan §4.7 requires total mAP no-regression for both deploy "
            f"and defer; a cell that regressed total mAP MUST land on drop "
            f"(malformed decision artifact)"
        )
    # Same check on the headline (it must mirror with_hn but we re-check
    # in case headline copy was correct but the verdict is still inconsistent).
    headline_metrics = artifact["headline_metrics"]
    headline_decision = artifact["headline_decision"]
    if headline_decision in ("deploy", "defer") and headline_metrics["map_no_regression"] is not True:
        raise ValueError(
            f"headline_decision={headline_decision!r} is incompatible with "
            f"headline_metrics.map_no_regression="
            f"{headline_metrics['map_no_regression']!r}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
