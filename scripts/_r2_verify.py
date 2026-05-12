"""R2 Round-Close Verification — §2.1.1 a-stage interface framework.

================================================================================
SCOPE CLAMP — read this before adding a single line of logic to this module.
================================================================================

This module does ONE thing: it walks the round artifacts at R2 close, validates
schema/path/hash invariants, and emits `runs/_r2_verification.json` (the
canonical index that `phase_R2.md` cites).

This module MUST NOT:

  - Apply the precision decision rule (Cases A/B/C/D, sensitivity sweep,
    threshold checks). That's `_r2_decide_precision.py`'s monopoly.
  - Compute mAP deltas, bootstrap CIs, audit propagation flags, or any
    other metric.
  - Branch on threshold values (loc_regression_pp, safety_regression_pp,
    build_map_variance_pp, build_tdetect_variance_pct).
  - Reclassify decision_case based on the records it reads. It only
    confirms the writer wrote the case the schema accepts.

The single allowed runtime decision pattern is:
    "value as written != value re-computed by hash" → hard-fail

Anything else is a scope-clamp violation. B2 + C3 reviewers explicitly check
that no logic in this file branches on metric values (per docs/_archive/
pre_r2_kickoff_checklist.md §2.1.1 d-stage scoped review brief; 2026-05-12 归档).

================================================================================
What this module DOES do
================================================================================

1. Schema-validate every R2 §1.0 / §2.1.1 JSON artifact via
   `_r2_schema_utils.validate_and_enforce` with the canonical UNIQUENESS_KEYS.

2. Hash-verify hash-pinned artifacts (recompute SHA256 against persisted
   .sha256, hard-fail on mismatch).

3. Soak-SHA binding (§4.1): when `runs/_r2_orin_soak_records.json` contains
   any entry, hard-fail when `engine_sha256 != selected_artifact_sha256`
   (pure binary comparison, not threshold logic).

4. Path-existence checks: every path referenced inside any decision/audit/
   carry-forward record must resolve to an actual file.

5. Cross-artifact integrity: the audit_coverage_path/sha256 cited in each
   PrecisionDecisionsInput must match the canonical
   `runs/_r2_audit_coverage.json` and its `.sha256`.

6. Aggregate the above into `runs/_r2_verification.json` — a per-checkpoint
   pass/fail roll-up that `phase_R2.md` and the demo-coverage hook consume.

STATUS: a-stage scaffold; every function below raises NotImplementedError.
b-stage lands schema/path/hash validation logic; b-stage lifecycle order does
NOT depend on R2 data freeze, but the actual artifacts the verifier reads
DO — so the b-stage tests run on synthetic fixtures, and the live invocation
happens at R2 close.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping, Sequence

from _r2_schema_utils import (
    CI_FIELDS_PER_DECISION_OUTPUT_RECORD,
    UNIQUENESS_KEYS,
    SchemaUniquenessError,
    ValidationError,
    enforce_ci_ordering,
    validate_and_enforce,
)

SCRIPTS = Path(__file__).resolve().parent
COMPONENT_SCHEMA = SCRIPTS / "_r2_component_decision_schema.json"
AUDIT_COVERAGE_SCHEMA = SCRIPTS / "_r2_audit_coverage_schema.json"
CARRY_FORWARD_SCHEMA = SCRIPTS / "_r2_carry_forward_schema.json"
DECISION_SCHEMA = SCRIPTS / "_r2_decision_schema.json"


def _scope_clamp_assertion() -> None:
    """Reviewer hook: this assertion is here so every reader sees the scope
    clamp. If you find yourself wanting to disable it because you 'just need
    to compare two thresholds', stop and route the work to
    `_r2_decide_precision.py` instead.
    """
    # Intentionally a no-op. The DOCSTRING is the load-bearing part.
    pass


def verify_path_exists(path: Path) -> None:
    """Hard-fail if path does not exist. Pure filesystem check.

    Raises:
        FileNotFoundError: in b-stage on missing path.
        NotImplementedError: a-stage scaffold.
    """
    raise NotImplementedError("§2.1.1 a-stage scaffold")


def verify_sha256(path: Path, expected_sha256: str) -> None:
    """Recompute SHA256 of `path` and hard-fail on mismatch with `expected_sha256`.
    Pure binary comparison, NOT a threshold check.

    Raises:
        FileNotFoundError: missing path.
        ValueError: hash mismatch (b-stage).
        NotImplementedError: a-stage scaffold.
    """
    raise NotImplementedError("§2.1.1 a-stage scaffold")


def verify_schema_compliance(records, schema_path: Path, unique_keys=None) -> None:
    """Wraps `_r2_schema_utils.validate_and_enforce`. Centralizes the call
    pattern so future schema-vs-uniqueness order changes propagate
    consistently.

    Raises:
        ValidationError: schema failure.
        SchemaUniquenessError: uniqueness failure.
        NotImplementedError: a-stage scaffold.
    """
    raise NotImplementedError("§2.1.1 a-stage scaffold")


def verify_decisions_artifact(decisions_path: Path) -> None:
    """Verify `runs/_r2_precision_decisions.json` is well-formed at round close.

    THREE non-bypassable runtime gates (in this exact order — each closes a
    failure mode the others can't catch):

    1. validate_and_enforce(records, DECISION_SCHEMA,
       unique_keys=UNIQUENESS_KEYS["_r2_decision_schema.json"],
       definition="PrecisionDecisionsArray") — schema shape + sub-key
       uniqueness on `detector`.
    2. enforce_ci_ordering(records, CI_FIELDS_PER_DECISION_OUTPUT_RECORD) —
       ci_low <= point <= ci_high on every CI sub-object. Schema validation
       above does NOT cover this (JSON Schema draft-07 has no cross-property
       numeric ordering operator); without this gate, a writer/external-edit
       could persist `{point: -0.1, ci_low: 0.2, ci_high: 0.3}` and pass round
       close.
    3. Path-existence checks on `selected_artifact_path` for every record.

    Does NOT: re-evaluate the decision rule, recompute deltas, or check
    whether ship_precision is "the right call". Pure structural + ordering
    + filesystem-existence checks (per scope clamp at top of file).

    Why all three matter: artifacts can be produced by future b-stage code,
    hand-edited at debugging time, regenerated through alternative paths, or
    partially bypass the writer helpers. The verifier is the last gate before
    R2 close treats the artifact as canonical evidence. A gate that's only
    enforced at write-time is non-load-bearing for round closure.

    Raises:
        NotImplementedError: a-stage scaffold.
        ValidationError: schema-level failure (b-stage).
        SchemaUniquenessError: duplicate detector record (b-stage).
        ValueError: CI ordering violation (b-stage).
        FileNotFoundError: selected_artifact_path doesn't exist (b-stage).
    """
    raise NotImplementedError("§2.1.1 a-stage scaffold")


def verify_audit_coverage_consistency(
    audit_path: Path, audit_sha256_path: Path, decisions: Sequence[Mapping]
) -> None:
    """Cross-artifact integrity: every PrecisionDecisionOutput record's
    class_wise items must reference an audit-coverage record present in the
    canonical `runs/_r2_audit_coverage.json`. Hash-pin via .sha256 file.

    This is a CROSS-REFERENCE check, not a value computation: it asserts the
    decision JSON's audit fields are byte-equal to the audit coverage JSON
    for the named classes.

    Raises:
        NotImplementedError: a-stage scaffold.
    """
    raise NotImplementedError("§2.1.1 a-stage scaffold")


def verify_soak_sha_binding(soak_records_path: Path, decisions: Sequence[Mapping]) -> None:
    """§4.1 soak SHA binding — when `runs/_r2_orin_soak_records.json` exists
    and contains any per-detector record, the soak record's `engine_sha256`
    MUST equal the corresponding precision decision's
    `selected_artifact_sha256`. Pure binary comparison; hard-fail on mismatch
    blocks R2 close.

    Skipped silently if soak_records_path doesn't exist (no Case A FP32
    ship → no soak required). The decision to require soak is the locked
    plan's pre-deploy carry-forward — verifier only enforces the binding
    once a soak record exists.

    Raises:
        NotImplementedError: a-stage scaffold.
        ValueError: in b-stage on engine_sha256 != selected_artifact_sha256.
    """
    raise NotImplementedError("§2.1.1 a-stage scaffold")


def verify_carry_forward_artifact(carry_path: Path) -> None:
    """Schema-validate `runs/_r2_carry_forward.json` against
    `_r2_carry_forward_schema.json` AND enforce uniqueness on `item_id` via
    UNIQUENESS_KEYS. Path-exists check on `unblock_evidence_path` when the
    optional field is set.

    Raises:
        NotImplementedError: a-stage scaffold.
    """
    raise NotImplementedError("§2.1.1 a-stage scaffold")


def verify_component_decisions_artifact(component_path: Path) -> None:
    """Schema-validate `runs/_r2_component_decisions.json` against
    `_r2_component_decision_schema.json` AND enforce uniqueness on `component`.
    Path-exists check on every blocking_artifact path.

    Raises:
        NotImplementedError: a-stage scaffold.
    """
    raise NotImplementedError("§2.1.1 a-stage scaffold")


def emit_verification_index(
    output_path: Path,
    artifact_results: Mapping[str, Mapping],
) -> None:
    """Write `runs/_r2_verification.json` with the per-checkpoint verdict
    aggregated from each verify_* function above. The schema for this file
    is NOT one of the §1.0 schemas — it's an output-only roll-up. Future
    versions may pin it to its own schema; for now the description
    convention lives in the locked plan's `Verification — canonical index`
    section.

    Raises:
        NotImplementedError: a-stage scaffold.
    """
    raise NotImplementedError("§2.1.1 a-stage scaffold")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="R2 round-close verifier (§2.1.1; scope-clamped to schema/path/hash; "
        "MUST NOT branch on metric values)",
    )
    parser.add_argument("--runs-dir", required=True, type=Path, help="repo runs/ directory")
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="output path for runs/_r2_verification.json",
    )
    args = parser.parse_args(argv)

    _scope_clamp_assertion()

    raise NotImplementedError(
        "§2.1.1 a-stage scaffold. b-stage lands schema/path/hash validation "
        f"(args parsed OK: runs_dir={args.runs_dir} out={args.out}). "
        "Reviewer reminder: any future b-stage diff that branches on metric "
        "values is a scope-clamp violation — route that to _r2_decide_precision.py."
    )


if __name__ == "__main__":
    sys.exit(main())
