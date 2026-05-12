"""Self-test for the four R2 shared schemas (§1.0 c-stage + §2.1.1 a-stage).

Validates each schema against JSON Schema draft-07 meta-schema, then runs
positive + negative fixtures asserting the conditional rules pinned in
docs/planning/_archive/pre_r2_kickoff_checklist.md (2026-05-12 归档) v1.4 §1.0 and §2.1.1, plus the
cross-file $ref load-bearing path through _r2_schema_utils.

Run:
    uv run python scripts/_r2_schemas_test.py

Exit 0 = all fixtures behaved as expected. Exit 1 = at least one failure;
stdout reports which schema/fixture deviated.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from jsonschema import Draft7Validator
from jsonschema.exceptions import SchemaError, ValidationError
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT7

from _r2_schema_utils import (
    CI_FIELDS_PER_DECISION_OUTPUT_RECORD,
    SchemaUniquenessError,
    UNIQUENESS_KEYS,
    enforce_ci_ordering,
    enforce_unique_by,
    get_default_registry,
    load_validator,
    validate_and_enforce,
    validate_records,
)

SCRIPTS = Path(__file__).resolve().parent
COMPONENT_SCHEMA = SCRIPTS / "_r2_component_decision_schema.json"
AUDIT_SCHEMA = SCRIPTS / "_r2_audit_coverage_schema.json"
CARRY_SCHEMA = SCRIPTS / "_r2_carry_forward_schema.json"
DECISION_SCHEMA = SCRIPTS / "_r2_decision_schema.json"


def build_decision_validator(definition_name: str) -> Draft7Validator:
    """Construct a validator for one of the named definitions in the decision
    schema (e.g. PrecisionDecisionsArray, PrecisionDecisionsInput). Delegates
    to `_r2_schema_utils.load_validator` so the test driver exercises the
    SAME entry point a production caller (b-stage _r2_decide_precision.py /
    _r2_verify.py) uses — closes the load-bearing-vs-fixture-only gap.
    """
    return load_validator(DECISION_SCHEMA, definition=definition_name)

# UNIQUENESS_KEYS in _r2_schema_utils is the authoritative declaration; we
# pin it here so callers can locate the schema-keyed mapping during review.


def load_schema(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def expect_valid(validator: Draft7Validator, instance, label: str, errors: list[str]) -> None:
    errs = list(validator.iter_errors(instance))
    if errs:
        errors.append(f"[{label}] expected VALID but got {len(errs)} error(s):")
        for e in errs:
            errors.append(f"  - path={list(e.absolute_path)} msg={e.message}")


def expect_invalid(
    validator: Draft7Validator,
    instance,
    label: str,
    errors: list[str],
    must_mention: str | None = None,
    must_keyword: str | None = None,
) -> None:
    """Assert that `instance` is rejected. Optionally pin to a specific failure.

    must_mention: legacy substring check on error messages. 'enum' aliases the
    actual jsonschema wording 'is not one of'.
    must_keyword: stricter assertion that at least one error's `validator`
    attribute matches (e.g. 'pattern', 'enum', 'minLength', 'required',
    'minItems', 'maxItems', 'const', 'not'). Closes the C3-noted gap where
    a fixture passes by being rejected for the WRONG reason.
    """
    errs = list(validator.iter_errors(instance))
    if not errs:
        errors.append(f"[{label}] expected INVALID but schema accepted it")
        return
    if must_mention is not None:
        joined = " | ".join(e.message for e in errs)
        needle = must_mention.lower()
        if needle == "enum":
            needle = "is not one of"
        if needle not in joined.lower():
            errors.append(f"[{label}] error messages did not mention '{must_mention}': {joined}")
    if must_keyword is not None:
        keywords = {e.validator for e in errs}
        if must_keyword not in keywords:
            errors.append(
                f"[{label}] expected validator-keyword '{must_keyword}' in errors, got {sorted(keywords)}"
            )


def assert_unique_by(records: list[dict], key, label: str, errors: list[str]) -> None:
    """Test-driver wrapper around `enforce_unique_by` that converts raised
    exceptions into the fixture-driver's `errors` accumulator pattern.

    The actual uniqueness check lives in `_r2_schema_utils.enforce_unique_by`
    so writers/verifiers (`_r2_decide_precision.py`, `_r2_verify.py`, etc.)
    can call the same function at runtime, not just inside this test driver.
    """
    try:
        enforce_unique_by(records, key)
    except SchemaUniquenessError as e:
        errors.append(f"[{label}] {e}")


def self_check(path: Path, errors: list[str]) -> Draft7Validator | None:
    schema = load_schema(path)
    try:
        Draft7Validator.check_schema(schema)
    except SchemaError as e:
        errors.append(f"[{path.name}] FAILED draft-07 self-check: {e.message}")
        return None
    return Draft7Validator(schema)


def test_component_decision(errors: list[str]) -> None:
    v = self_check(COMPONENT_SCHEMA, errors)
    if v is None:
        return

    # POSITIVE — deploy with empty fields, deploy with non-empty fields, and
    # map-prior defer with branch=replay_only.
    expect_valid(
        v,
        [
            {
                "component": "copy_paste_class_balanced_loss",
                "outcome": "deploy",
                "reason": "",
                "blocking_artifacts": [
                    "runs/_r2_copy_paste_ablation.json",
                    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
                ],
                "next_round_action": "",
            },
            {
                "component": "hard_negative_mining",
                "outcome": "deploy",
                "reason": "+0.4 pp aggregate, no per-class regression",
                "blocking_artifacts": ["runs/_hard_negative_eval_manifest.json"],
                "next_round_action": "carried into R3 ablation table for SAHI-on comparison",
            },
            {
                "component": "map_prior_gating",
                "outcome": "defer",
                "reason": "GPS topic deadline 5/12 missed",
                "blocking_artifacts": ["runs/_map_prior_negative_controls.json"],
                "next_round_action": "R3 live-branch verification post-GPS approval",
                "branch": "replay_only",
            },
        ],
        "component:positive_deploy+deploy_filled+defer",
        errors,
    )

    # NEGATIVE 1 — defer outcome with empty reason → conditional minLength=1 fails.
    expect_invalid(
        v,
        [
            {
                "component": "hard_negative_mining",
                "outcome": "defer",
                "reason": "",
                "blocking_artifacts": [],
                "next_round_action": "R3 re-mining",
            }
        ],
        "component:negative_empty_reason_on_defer",
        errors,
        must_keyword="minLength",
    )

    # NEGATIVE 1b — drop outcome with empty next_round_action (other half of dual-minLength rule).
    expect_invalid(
        v,
        [
            {
                "component": "hard_negative_mining",
                "outcome": "drop",
                "reason": "abandoned for R3",
                "blocking_artifacts": [],
                "next_round_action": "",
            }
        ],
        "component:negative_empty_next_action_on_drop",
        errors,
        must_keyword="minLength",
    )

    # NEGATIVE 2 — outcome enum violation.
    expect_invalid(
        v,
        [
            {
                "component": "copy_paste_class_balanced_loss",
                "outcome": "ship",
                "reason": "",
                "blocking_artifacts": [],
                "next_round_action": "",
            }
        ],
        "component:negative_outcome_enum",
        errors,
        must_mention="enum",
        must_keyword="enum",
    )

    # NEGATIVE 3 — map_prior_gating without branch (then.required[branch] fires).
    expect_invalid(
        v,
        [
            {
                "component": "map_prior_gating",
                "outcome": "deploy",
                "reason": "",
                "blocking_artifacts": [],
                "next_round_action": "",
            }
        ],
        "component:negative_map_prior_missing_branch",
        errors,
        must_mention="branch",
        must_keyword="required",
    )

    # NEGATIVE 4 — branch present on non-map-prior component (else.not.required fires).
    expect_invalid(
        v,
        [
            {
                "component": "hard_negative_mining",
                "outcome": "deploy",
                "reason": "",
                "blocking_artifacts": [],
                "next_round_action": "",
                "branch": "live",
            }
        ],
        "component:negative_branch_on_non_map_prior",
        errors,
        must_mention="branch",
        must_keyword="not",
    )

    # NEGATIVE 5 — blocking_artifacts contains free prose (not path nor sha256).
    expect_invalid(
        v,
        [
            {
                "component": "copy_paste_class_balanced_loss",
                "outcome": "drop",
                "reason": "ablation showed +0.0 mAP",
                "blocking_artifacts": ["just some prose, not a path"],
                "next_round_action": "abandoned",
            }
        ],
        "component:negative_artifact_prose",
        errors,
        must_keyword="anyOf",
    )

    # NEGATIVE 6 — leading-./ canonicalization (C3 minor).
    expect_invalid(
        v,
        [
            {
                "component": "copy_paste_class_balanced_loss",
                "outcome": "deploy",
                "reason": "",
                "blocking_artifacts": ["./runs/_r2_copy_paste_ablation.json"],
                "next_round_action": "",
            }
        ],
        "component:negative_path_leading_dot_slash",
        errors,
        must_keyword="anyOf",
    )

    # NEGATIVE 7 — double-slash empty segment.
    expect_invalid(
        v,
        [
            {
                "component": "copy_paste_class_balanced_loss",
                "outcome": "deploy",
                "reason": "",
                "blocking_artifacts": ["runs//ablation.json"],
                "next_round_action": "",
            }
        ],
        "component:negative_path_double_slash",
        errors,
        must_keyword="anyOf",
    )

    # NEGATIVE 8 — interior current-dir segment (C3 iter-2 minor).
    expect_invalid(
        v,
        [
            {
                "component": "copy_paste_class_balanced_loss",
                "outcome": "deploy",
                "reason": "",
                "blocking_artifacts": ["runs/./ablation.json"],
                "next_round_action": "",
            }
        ],
        "component:negative_path_interior_dot",
        errors,
        must_keyword="anyOf",
    )

    # NEGATIVE 9 — trailing slash (directory spelling, not artifact ref).
    expect_invalid(
        v,
        [
            {
                "component": "copy_paste_class_balanced_loss",
                "outcome": "deploy",
                "reason": "",
                "blocking_artifacts": ["runs/ablation/"],
                "next_round_action": "",
            }
        ],
        "component:negative_path_trailing_slash",
        errors,
        must_keyword="anyOf",
    )

    # POSITIVE — confirm hidden filename (leading dot in segment) still allowed.
    expect_valid(
        v,
        [
            {
                "component": "copy_paste_class_balanced_loss",
                "outcome": "deploy",
                "reason": "",
                "blocking_artifacts": ["runs/.gitkeep"],
                "next_round_action": "",
            }
        ],
        "component:positive_hidden_segment_allowed",
        errors,
    )


def test_audit_coverage(errors: list[str]) -> None:
    v = self_check(AUDIT_SCHEMA, errors)
    if v is None:
        return

    # POSITIVE — covered + low_power + construction_failed + insufficient.
    expect_valid(
        v,
        [
            {
                "class_id": 0,
                "class_name": "red",
                "full_val_support": 11601,
                "full_val_insufficient": False,
                "audit_support": 30,
                "audit_coverage_status": "covered",
                "audit_low_power": False,
            },
            {
                "class_id": 5,
                "class_name": "redRight",
                "full_val_support": 6,
                "full_val_insufficient": True,
                "audit_support": 2,
                "audit_coverage_status": "low_power",
                "audit_low_power": True,
            },
            {
                "class_id": 6,
                "class_name": "greenRight",
                "full_val_support": 4,
                "full_val_insufficient": True,
                "audit_support": 0,
                "audit_coverage_status": "construction_failed",
                "audit_low_power": True,
                "construction_failure_reason": "all 4 instances filtered by night-scenario stratification",
            },
        ],
        "audit:positive_three_states",
        errors,
    )

    # NEGATIVE 1 — full_val_insufficient=true but support=100 (cross-field rule).
    expect_invalid(
        v,
        [
            {
                "class_id": 1,
                "class_name": "yellow",
                "full_val_support": 100,
                "full_val_insufficient": True,
                "audit_support": 5,
                "audit_coverage_status": "covered",
                "audit_low_power": True,
            }
        ],
        "audit:negative_insufficient_flag_mismatch",
        errors,
        must_keyword="const",
    )

    # NEGATIVE 2 — status=construction_failed but audit_support=2.
    expect_invalid(
        v,
        [
            {
                "class_id": 2,
                "class_name": "green",
                "full_val_support": 12925,
                "full_val_insufficient": False,
                "audit_support": 2,
                "audit_coverage_status": "construction_failed",
                "audit_low_power": True,
                "construction_failure_reason": "wrong",
            }
        ],
        "audit:negative_status_support_mismatch",
        errors,
        must_keyword="const",
    )

    # NEGATIVE 3 — construction_failure_reason on non-failed class.
    expect_invalid(
        v,
        [
            {
                "class_id": 3,
                "class_name": "redLeft",
                "full_val_support": 3218,
                "full_val_insufficient": False,
                "audit_support": 10,
                "audit_coverage_status": "covered",
                "audit_low_power": True,
                "construction_failure_reason": "should not appear",
            }
        ],
        "audit:negative_failure_reason_on_covered",
        errors,
        must_keyword="not",
    )

    # NEGATIVE 4 — missing required field.
    expect_invalid(
        v,
        [
            {
                "class_id": 4,
                "class_name": "greenLeft",
                "full_val_support": 586,
                "full_val_insufficient": False,
                "audit_support": 12,
                "audit_coverage_status": "covered"
            }
        ],
        "audit:negative_missing_required",
        errors,
        must_mention="required",
        must_keyword="required",
    )

    # NEGATIVE 5 — wrong enum on status.
    expect_invalid(
        v,
        [
            {
                "class_id": 0,
                "class_name": "red",
                "full_val_support": 11601,
                "full_val_insufficient": False,
                "audit_support": 30,
                "audit_coverage_status": "ok",
                "audit_low_power": False,
            }
        ],
        "audit:negative_status_enum",
        errors,
        must_mention="enum",
        must_keyword="enum",
    )


def test_carry_forward(errors: list[str]) -> None:
    v = self_check(CARRY_SCHEMA, errors)
    if v is None:
        return

    # POSITIVE — blocked AND, blocked OR, single-token blocked, single-token + redundant
    # explicit any (spec accepts as no-op), scheduled.
    expect_valid(
        v,
        [
            {
                "item_id": "kd_first_cell",
                "status": "blocked",
                "blocked_on": ["deim_l_training", "hard_neg_manifest_hash", "r2_data_freeze"],
                "unblock_logic": "all",
                "next_entrypoint": "additional_components_plan §七.6",
            },
            {
                "item_id": "int8_qat",
                "status": "blocked",
                "blocked_on": ["sahi_b_c_measured", "tsm_phase_1b_passed"],
                "unblock_logic": "any",
                "next_entrypoint": "additional_components_plan §十 a-stage",
            },
            {
                "item_id": "tsm_phase_1a",
                "status": "blocked",
                "blocked_on": ["on_vehicle_replay_failure_modes"],
                "next_entrypoint": "components/temporal_shift_module/runners/concept_validation.py",
            },
            {
                "item_id": "single_token_redundant_any",
                "status": "blocked",
                "blocked_on": ["autonomy_team"],
                "unblock_logic": "any",
                "next_entrypoint": "additional_components_plan §八 a-stage",
            },
            {
                "item_id": "adaptive_roi",
                "status": "scheduled",
                "blocked_on": [],
                "next_entrypoint": "r3_latency_budget_release",
            },
        ],
        "carry:positive_five_shapes",
        errors,
    )

    # NEGATIVE 1 — status=blocked + blocked_on=[].
    expect_invalid(
        v,
        [
            {
                "item_id": "bad_blocked_empty",
                "status": "blocked",
                "blocked_on": [],
                "next_entrypoint": "x",
            }
        ],
        "carry:negative_blocked_no_tokens",
        errors,
        must_keyword="minItems",
    )

    # NEGATIVE 2 — status=scheduled + non-empty blocked_on.
    expect_invalid(
        v,
        [
            {
                "item_id": "bad_scheduled_with_token",
                "status": "scheduled",
                "blocked_on": ["r2_data_freeze"],
                "next_entrypoint": "x",
            }
        ],
        "carry:negative_scheduled_with_tokens",
        errors,
        must_keyword="maxItems",
    )

    # NEGATIVE 3 — status=scheduled + unblock_logic present.
    expect_invalid(
        v,
        [
            {
                "item_id": "bad_scheduled_with_logic",
                "status": "scheduled",
                "blocked_on": [],
                "unblock_logic": "all",
                "next_entrypoint": "x",
            }
        ],
        "carry:negative_scheduled_with_logic",
        errors,
        must_keyword="not",
    )

    # NEGATIVE 4 — blocked_on token outside closed enum.
    expect_invalid(
        v,
        [
            {
                "item_id": "bad_unknown_token",
                "status": "blocked",
                "blocked_on": ["mystery_blocker"],
                "next_entrypoint": "x",
            }
        ],
        "carry:negative_unknown_token",
        errors,
        must_mention="enum",
        must_keyword="enum",
    )

    # NEGATIVE 5 — missing next_entrypoint.
    expect_invalid(
        v,
        [
            {
                "item_id": "bad_no_entrypoint",
                "status": "scheduled",
                "blocked_on": [],
            }
        ],
        "carry:negative_missing_entrypoint",
        errors,
        must_mention="required",
        must_keyword="required",
    )

    # NEGATIVE 6 — wrong unblock_logic enum.
    expect_invalid(
        v,
        [
            {
                "item_id": "bad_logic_enum",
                "status": "blocked",
                "blocked_on": ["r2_data_freeze", "autonomy_team"],
                "unblock_logic": "either",
                "next_entrypoint": "x",
            }
        ],
        "carry:negative_logic_enum",
        errors,
        must_mention="enum",
        must_keyword="enum",
    )

    # NEGATIVE 7 — unblock_evidence_path with whitespace (B2 M1 fix verification).
    expect_invalid(
        v,
        [
            {
                "item_id": "bad_evidence_with_prose",
                "status": "blocked",
                "blocked_on": ["r2_data_freeze"],
                "unblock_evidence_path": "see planning doc for details",
                "next_entrypoint": "x",
            }
        ],
        "carry:negative_evidence_path_with_whitespace",
        errors,
        must_keyword="pattern",
    )


def test_uniqueness_guards(errors: list[str]) -> None:
    """JSON Schema draft-07's `uniqueItems` only catches byte-identical records.
    Spec-level 'unique by sub-key' rules (component / item_id / class_id /
    class_name) require runtime guards. C3 flagged this as MAJOR.
    """
    # Component: two records with same component but different outcomes.
    sub_errors: list[str] = []
    assert_unique_by(
        [
            {"component": "map_prior_gating", "outcome": "deploy", "reason": "", "blocking_artifacts": [], "next_round_action": "", "branch": "live"},
            {"component": "map_prior_gating", "outcome": "defer", "reason": "GPS deadline missed", "blocking_artifacts": [], "next_round_action": "R3 retry", "branch": "replay_only"},
        ],
        "component",
        "uniqueness:component_duplicate",
        sub_errors,
    )
    if not sub_errors:
        errors.append("[uniqueness:component_duplicate] expected uniqueness violation but assert_unique_by accepted duplicates")

    # Audit: two records with same class_id.
    sub_errors = []
    assert_unique_by(
        [
            {"class_id": 0, "class_name": "red", "full_val_support": 11601, "full_val_insufficient": False, "audit_support": 30, "audit_coverage_status": "covered", "audit_low_power": False},
            {"class_id": 0, "class_name": "shouldnt_repeat", "full_val_support": 100, "full_val_insufficient": False, "audit_support": 5, "audit_coverage_status": "covered", "audit_low_power": True},
        ],
        "class_id",
        "uniqueness:audit_class_id_duplicate",
        sub_errors,
    )
    if not sub_errors:
        errors.append("[uniqueness:audit_class_id_duplicate] expected uniqueness violation but assert_unique_by accepted duplicates")

    # Audit: two records with mismatched (class_id, class_name) composite — same class_id mapped to two names.
    sub_errors = []
    assert_unique_by(
        [
            {"class_id": 5, "class_name": "redRight", "full_val_support": 6, "full_val_insufficient": True, "audit_support": 0, "audit_coverage_status": "construction_failed", "audit_low_power": True, "construction_failure_reason": "x"},
            {"class_id": 5, "class_name": "redRight_dup", "full_val_support": 6, "full_val_insufficient": True, "audit_support": 0, "audit_coverage_status": "construction_failed", "audit_low_power": True, "construction_failure_reason": "y"},
        ],
        "class_id",
        "uniqueness:audit_id_with_name_drift",
        sub_errors,
    )
    if not sub_errors:
        errors.append("[uniqueness:audit_id_with_name_drift] expected uniqueness violation on class_id but accepted")

    # Carry-forward: two records with same item_id.
    sub_errors = []
    assert_unique_by(
        [
            {"item_id": "kd_first_cell", "status": "blocked", "blocked_on": ["r2_data_freeze"], "next_entrypoint": "x"},
            {"item_id": "kd_first_cell", "status": "scheduled", "blocked_on": [], "next_entrypoint": "y"},
        ],
        "item_id",
        "uniqueness:carry_item_id_duplicate",
        sub_errors,
    )
    if not sub_errors:
        errors.append("[uniqueness:carry_item_id_duplicate] expected uniqueness violation but accepted")

    # Positive control: distinct keys must NOT raise.
    sub_errors = []
    assert_unique_by(
        [
            {"component": "copy_paste_class_balanced_loss"},
            {"component": "hard_negative_mining"},
            {"component": "map_prior_gating"},
        ],
        "component",
        "uniqueness:component_distinct_positive",
        sub_errors,
    )
    if sub_errors:
        errors.extend(sub_errors)


def _decision_output_record(**overrides):
    """Helper: minimum-valid PrecisionDecisionOutput record with overrides applied."""
    base = {
        "detector": "yolo26_s",
        "ship_precision": "fp16",
        "decision_case": "B",
        "delta_mAP_0.5": {"point": 0.0, "ci_low": -0.001, "ci_high": 0.001},
        "delta_mAP_0.5_0.95": {"point": 0.0, "ci_low": -0.001, "ci_high": 0.001},
        "class_wise": [
            {
                "name": "red",
                "delta_AP": 0.0,
                "full_val_support": 11601,
                "full_val_insufficient": False,
                "support_tier": "high_block",
                "audit_support": 30,
                "audit_coverage_status": "covered",
                "audit_low_power": False,
            }
        ],
        "safety_class_exclusions": [],
        "diagnostics": {
            "num_safety_classes_regressed": 0,
            "worst_safety_delta_AP": 0.0,
            "support_of_worst_class": 0,
            "worst_class_name": "",
        },
        "investigation_required": False,
        "investigation_note": "",
        "all_thresholds_pass": {
            "loc_regression_pp": True,
            "safety_regression_pp": True,
            "build_map_variance_pp": True,
            "build_tdetect_variance_pct": True,
        },
        "sensitivity_sweep": {
            "one_at_a_time": {
                "loc_regression_pp": {"-0.1": "fp16"},
                "safety_regression_pp": {"0.5": "fp16"},
                "build_map_variance_pp": {"0.5": "fp16"},
                "build_tdetect_variance_pct": {"5.0": "fp16"},
            },
            "joint": {"strict": "fp16", "nominal": "fp16", "lenient": "fp16"},
        },
        "selected_artifact_path": "runs/yolo26_s/best.engine",
        "selected_artifact_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    }
    base.update(overrides)
    return base


def test_decision_output(errors: list[str]) -> None:
    decision = json.loads(DECISION_SCHEMA.read_text())
    try:
        Draft7Validator.check_schema(decision)
    except SchemaError as e:
        errors.append(f"[_r2_decision_schema.json] FAILED draft-07 self-check: {e.message}")
        return

    v = build_decision_validator("PrecisionDecisionsArray")

    # POSITIVE — 4 detectors, B case (FP16 ship), all thresholds pass.
    expect_valid(
        v,
        [
            _decision_output_record(detector="yolo26_s"),
            _decision_output_record(detector="yolov13_s"),
            _decision_output_record(detector="deim_dfine_s"),
            _decision_output_record(detector="deim_dfine_m"),
        ],
        "decision:positive_4_detectors_case_B",
        errors,
    )

    # POSITIVE — Case A (FP32 ship): ship_precision=fp32, decision_case=A.
    expect_valid(
        v,
        [
            _decision_output_record(
                detector="yolo26_s",
                ship_precision="fp32",
                decision_case="A",
                selected_artifact_path="runs/yolo26_s/best_fp32.engine",
            )
        ],
        "decision:positive_case_A_fp32",
        errors,
    )

    # POSITIVE — Case C with investigation_required=true + non-empty note.
    expect_valid(
        v,
        [
            _decision_output_record(
                detector="yolo26_s",
                decision_case="C",
                investigation_required=True,
                investigation_note="rare class redLeft regressed -0.6pp at full_val_support=42, support_tier=low_block",
            )
        ],
        "decision:positive_case_C_with_investigation",
        errors,
    )

    # POSITIVE — inconclusive_global escape with pre_committed_defer_outcome.
    expect_valid(
        v,
        [
            _decision_output_record(
                detector="yolo26_s",
                decision_case="inconclusive_global",
                pre_committed_defer_outcome="r2_fp16_default",
            )
        ],
        "decision:positive_inconclusive_with_defer",
        errors,
    )

    # NEGATIVE — case A but ship_precision=fp16 (cross-field rule).
    expect_invalid(
        v,
        [
            _decision_output_record(
                detector="yolo26_s",
                decision_case="A",
                ship_precision="fp16",
            )
        ],
        "decision:negative_caseA_fp16_mismatch",
        errors,
        must_keyword="const",
    )

    # NEGATIVE — case B but ship_precision=fp32.
    expect_invalid(
        v,
        [
            _decision_output_record(
                detector="yolo26_s",
                decision_case="B",
                ship_precision="fp32",
            )
        ],
        "decision:negative_caseB_fp32_mismatch",
        errors,
        must_keyword="const",
    )

    # NEGATIVE — case C but investigation_required=false.
    expect_invalid(
        v,
        [
            _decision_output_record(
                detector="yolo26_s",
                decision_case="C",
                ship_precision="fp16",
                investigation_required=False,
            )
        ],
        "decision:negative_caseC_no_investigation",
        errors,
        must_keyword="const",
    )

    # NEGATIVE — investigation_required=true but empty investigation_note.
    expect_invalid(
        v,
        [
            _decision_output_record(
                detector="yolo26_s",
                investigation_required=True,
                investigation_note="",
            )
        ],
        "decision:negative_empty_investigation_note",
        errors,
        must_keyword="minLength",
    )

    # NEGATIVE — pre_committed_defer_outcome on non-inconclusive case.
    expect_invalid(
        v,
        [
            _decision_output_record(
                detector="yolo26_s",
                decision_case="B",
                pre_committed_defer_outcome="r2_fp16_default",
            )
        ],
        "decision:negative_defer_outcome_on_caseB",
        errors,
        must_keyword="const",
    )

    # NEGATIVE — detector outside closed enum.
    expect_invalid(
        v,
        [_decision_output_record(detector="rtdetr_l")],
        "decision:negative_detector_enum",
        errors,
        must_keyword="enum",
    )

    # NEGATIVE — selected_artifact_path missing .engine suffix.
    expect_invalid(
        v,
        [_decision_output_record(selected_artifact_path="runs/yolo26_s/best.pt")],
        "decision:negative_artifact_not_engine",
        errors,
        must_keyword="pattern",
    )

    # NEGATIVE — class_wise item missing required audit field.
    bad_record = _decision_output_record()
    bad_record["class_wise"] = [
        {
            "name": "red",
            "delta_AP": 0.0,
            "full_val_support": 11601,
            "full_val_insufficient": False,
            "support_tier": "high_block",
            "audit_support": 30,
            "audit_coverage_status": "covered",
            # audit_low_power missing
        }
    ]
    expect_invalid(
        v,
        [bad_record],
        "decision:negative_class_wise_missing_audit_low_power",
        errors,
        must_keyword="required",
    )

    # === MAJOR-1 fixtures (support_tier ↔ full_val_support coupling) ===

    # NEGATIVE — full_val_insufficient=true but support_tier="high_block".
    bad_tier1 = _decision_output_record()
    bad_tier1["class_wise"] = [
        {
            "name": "redRight",
            "delta_AP": 0.0,
            "full_val_support": 6,
            "full_val_insufficient": True,
            "support_tier": "high_block",
            "audit_support": 0,
            "audit_coverage_status": "construction_failed",
            "audit_low_power": True,
            "construction_failure_reason": "absent",
        }
    ]
    expect_invalid(v, [bad_tier1], "decision:negative_tier_with_insufficient", errors, must_keyword="type")

    # NEGATIVE — support=50 (low_block range) but tier="high_block".
    bad_tier2 = _decision_output_record()
    bad_tier2["class_wise"] = [
        {
            "name": "yellow",
            "delta_AP": 0.0,
            "full_val_support": 50,
            "full_val_insufficient": False,
            "support_tier": "high_block",
            "audit_support": 30,
            "audit_coverage_status": "covered",
            "audit_low_power": False,
        }
    ]
    expect_invalid(v, [bad_tier2], "decision:negative_tier_50_high_block", errors, must_keyword="const")

    # NEGATIVE — support=200 (medium_block range) but tier="low_block".
    bad_tier3 = _decision_output_record()
    bad_tier3["class_wise"] = [
        {
            "name": "yellow",
            "delta_AP": 0.0,
            "full_val_support": 200,
            "full_val_insufficient": False,
            "support_tier": "low_block",
            "audit_support": 30,
            "audit_coverage_status": "covered",
            "audit_low_power": False,
        }
    ]
    expect_invalid(v, [bad_tier3], "decision:negative_tier_200_low_block", errors, must_keyword="const")

    # POSITIVE — boundary support=99 → low_block, 100 → medium_block, 499 → medium_block, 500 → high_block.
    for support_val, expected_tier in [(99, "low_block"), (100, "medium_block"), (499, "medium_block"), (500, "high_block")]:
        good = _decision_output_record()
        good["class_wise"] = [
            {
                "name": "boundary_test",
                "delta_AP": 0.0,
                "full_val_support": support_val,
                "full_val_insufficient": False,
                "support_tier": expected_tier,
                "audit_support": 30,
                "audit_coverage_status": "covered",
                "audit_low_power": False,
            }
        ]
        expect_valid(v, [good], f"decision:positive_tier_boundary_{support_val}_{expected_tier}", errors)

    # === MAJOR-2 fixtures (class_wise audit cross-field rules) ===

    # NEGATIVE — audit_support=0 + status="covered" (must be construction_failed).
    bad_audit1 = _decision_output_record()
    bad_audit1["class_wise"] = [
        {
            "name": "redRight",
            "delta_AP": 0.0,
            "full_val_support": 6,
            "full_val_insufficient": True,
            "support_tier": None,
            "audit_support": 0,
            "audit_coverage_status": "covered",  # WRONG
            "audit_low_power": True,
        }
    ]
    expect_invalid(v, [bad_audit1], "decision:negative_class_wise_audit_status_mismatch", errors, must_keyword="const")

    # NEGATIVE — audit_support=50 + audit_low_power=True (must be False since >= 30).
    bad_audit2 = _decision_output_record()
    bad_audit2["class_wise"] = [
        {
            "name": "red",
            "delta_AP": 0.0,
            "full_val_support": 11601,
            "full_val_insufficient": False,
            "support_tier": "high_block",
            "audit_support": 50,
            "audit_coverage_status": "covered",
            "audit_low_power": True,  # WRONG — must be False
        }
    ]
    expect_invalid(v, [bad_audit2], "decision:negative_class_wise_audit_low_power_mismatch", errors, must_keyword="const")

    # NEGATIVE — construction_failure_reason on covered class.
    bad_audit3 = _decision_output_record()
    bad_audit3["class_wise"] = [
        {
            "name": "red",
            "delta_AP": 0.0,
            "full_val_support": 11601,
            "full_val_insufficient": False,
            "support_tier": "high_block",
            "audit_support": 30,
            "audit_coverage_status": "covered",
            "audit_low_power": False,
            "construction_failure_reason": "should not be here",
        }
    ]
    expect_invalid(v, [bad_audit3], "decision:negative_class_wise_failure_reason_on_covered", errors, must_keyword="not")

    # === MAJOR-3 fixtures (escape-case ship_precision) ===

    # NEGATIVE — audit_disagreement + ship_precision=fp32.
    expect_invalid(
        v,
        [
            _decision_output_record(
                detector="yolo26_s",
                decision_case="audit_disagreement",
                ship_precision="fp32",
            )
        ],
        "decision:negative_audit_disagreement_fp32",
        errors,
        must_keyword="const",
    )

    # NEGATIVE — inconclusive_global + ship_precision=fp32.
    expect_invalid(
        v,
        [
            _decision_output_record(
                detector="yolo26_s",
                decision_case="inconclusive_global",
                ship_precision="fp32",
            )
        ],
        "decision:negative_inconclusive_fp32",
        errors,
        must_keyword="const",
    )

    # NEGATIVE — executor_error + ship_precision=fp32.
    expect_invalid(
        v,
        [
            _decision_output_record(
                detector="yolo26_s",
                decision_case="executor_error",
                ship_precision="fp32",
            )
        ],
        "decision:negative_executor_error_fp32",
        errors,
        must_keyword="const",
    )

    # NEGATIVE — investigation_required=true on Case B (reverse direction MINOR-4).
    expect_invalid(
        v,
        [
            _decision_output_record(
                detector="yolo26_s",
                decision_case="B",
                ship_precision="fp16",
                investigation_required=True,
                investigation_note="should not allow on case B",
            )
        ],
        "decision:negative_investigation_on_caseB",
        errors,
        must_keyword="const",
    )

    # === MINOR-2 fixtures (Diagnostics worst-class consistency) ===

    # NEGATIVE — num_safety_classes_regressed=0 but worst_class_name non-empty.
    bad_diag = _decision_output_record()
    bad_diag["diagnostics"] = {
        "num_safety_classes_regressed": 0,
        "worst_safety_delta_AP": 0.0,
        "support_of_worst_class": 0,
        "worst_class_name": "spurious",
    }
    expect_invalid(v, [bad_diag], "decision:negative_diag_spurious_worst_class", errors, must_keyword="maxLength")


def test_decision_input(errors: list[str]) -> None:
    """The Input shape inlines audit_records via $ref to
    _r2_audit_coverage_schema.json#/items. This test exercises the cross-file
    $ref resolution path — i.e. an audit record with self-inconsistent
    cross-field rules (e.g. audit_support=0 but status=covered) MUST be rejected
    even though the rejection rule lives in the audit schema, not the decision
    schema.
    """
    v = build_decision_validator("PrecisionDecisionsInput")

    valid_audit_record = {
        "class_id": 0,
        "class_name": "red",
        "full_val_support": 11601,
        "full_val_insufficient": False,
        "audit_support": 30,
        "audit_coverage_status": "covered",
        "audit_low_power": False,
    }

    base_input = {
        "detector": "yolo26_s",
        "eval_fp16_path": "runs/yolo26_s/eval_fp16.json",
        "eval_fp32_path": "runs/yolo26_s/eval_fp32.json",
        "eval_audit_fp16_path": "runs/yolo26_s/eval_audit_fp16.json",
        "eval_audit_fp32_path": "runs/yolo26_s/eval_audit_fp32.json",
        "timing_fp16_path": "runs/yolo26_s/timing_fp16.json",
        "timing_fp32_path": "runs/yolo26_s/timing_fp32.json",
        "build_variance_path": "runs/_r2_build_variance/yolo26_s.json",
        "audit_coverage_path": "runs/_r2_audit_coverage.json",
        "audit_coverage_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "val_manifest_sha256": "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
        "audit_records": [valid_audit_record],
    }

    # POSITIVE — well-formed input.
    expect_valid(v, base_input, "decision_input:positive_minimum", errors)

    # NEGATIVE — audit_records cross-file $ref propagates: status mismatch
    # (audit_support=0 but status=covered) is invalid per audit schema's
    # cross-field rule. If this fixture passes, the $ref isn't being resolved
    # and we have a load-bearing-but-ignored invariant.
    bad_input = json.loads(json.dumps(base_input))  # deep copy
    bad_input["audit_records"] = [
        {
            "class_id": 5,
            "class_name": "redRight",
            "full_val_support": 6,
            "full_val_insufficient": True,
            "audit_support": 0,
            "audit_coverage_status": "covered",  # WRONG — must be construction_failed
            "audit_low_power": True,
        }
    ]
    expect_invalid(
        v,
        bad_input,
        "decision_input:negative_audit_xref_status_mismatch",
        errors,
        must_keyword="const",
    )

    # NEGATIVE — audit_coverage_sha256 wrong format.
    bad_input2 = json.loads(json.dumps(base_input))
    bad_input2["audit_coverage_sha256"] = "not-a-hash"
    expect_invalid(
        v,
        bad_input2,
        "decision_input:negative_sha256_pattern",
        errors,
        must_keyword="pattern",
    )

    # NEGATIVE — missing required field.
    bad_input3 = json.loads(json.dumps(base_input))
    del bad_input3["audit_coverage_sha256"]
    expect_invalid(
        v,
        bad_input3,
        "decision_input:negative_missing_required",
        errors,
        must_keyword="required",
    )


def test_runtime_enforcement(errors: list[str]) -> None:
    """Verify the runtime-raising entrypoints in `_r2_schema_utils` actually
    raise. These functions are what writers (decide_precision/decide_cell/
    decide_phase) and verify (`_r2_verify.py`) must call before they persist
    or accept any `_r2_*.json` file. If they don't raise, the spec-level
    uniqueness rules are unenforced at write/verify time even though the
    test-driver's `assert_unique_by` flags them at fixture time.
    """
    # validate_records: schema violation must raise ValidationError.
    bad_record = [
        {
            "component": "ship",  # outside enum
            "outcome": "deploy",
            "reason": "",
            "blocking_artifacts": [],
            "next_round_action": "",
        }
    ]
    try:
        validate_records(bad_record, COMPONENT_SCHEMA)
        errors.append("[runtime:component_enum_should_raise] validate_records did NOT raise ValidationError")
    except ValidationError:
        pass

    # enforce_unique_by: collision must raise SchemaUniquenessError.
    try:
        enforce_unique_by(
            [{"component": "map_prior_gating"}, {"component": "map_prior_gating"}],
            "component",
        )
        errors.append("[runtime:component_dup_should_raise] enforce_unique_by did NOT raise on duplicate component")
    except SchemaUniquenessError:
        pass

    # enforce_unique_by: empty input must NOT raise.
    try:
        enforce_unique_by([], "component")
    except Exception as e:
        errors.append(f"[runtime:empty_input_should_pass] enforce_unique_by raised on empty list: {e!r}")

    # enforce_unique_by: single record must NOT raise.
    try:
        enforce_unique_by([{"component": "map_prior_gating"}], "component")
    except Exception as e:
        errors.append(f"[runtime:single_record_should_pass] enforce_unique_by raised on single-record list: {e!r}")

    # enforce_unique_by: composite key correctness — distinct on at least one component is OK.
    try:
        enforce_unique_by(
            [
                {"class_id": 0, "class_name": "red"},
                {"class_id": 0, "class_name": "redLeft"},  # different name, same id — TUPLE distinct
            ],
            ("class_id", "class_name"),
        )
    except SchemaUniquenessError as e:
        errors.append(f"[runtime:composite_distinct_should_pass] composite key wrongly flagged distinct pair: {e!r}")

    # enforce_unique_by: composite key collision when both components match.
    try:
        enforce_unique_by(
            [
                {"class_id": 0, "class_name": "red"},
                {"class_id": 0, "class_name": "red"},
            ],
            ("class_id", "class_name"),
        )
        errors.append("[runtime:composite_collision_should_raise] composite key did NOT raise on full match")
    except SchemaUniquenessError:
        pass

    # enforce_unique_by: missing-key precondition → fail-fast with KeyError, NOT
    # collapse to None-coincidence collision (C3 iter-2 minor: prior behaviour
    # was implicit pass-through via dict.get; now explicit).
    try:
        enforce_unique_by(
            [
                {"item_id": "a", "next_entrypoint": "x"},
                {"item_id": "b", "next_entrypoint": "y"},
            ],
            "missing_field",
        )
        errors.append("[runtime:missing_key_should_raise] missing-key precondition did NOT raise KeyError")
    except KeyError:
        pass
    except SchemaUniquenessError as e:
        errors.append(f"[runtime:missing_key_should_raise] expected KeyError, got SchemaUniquenessError: {e!r}")

    # enforce_unique_by: composite key with one component missing → KeyError.
    try:
        enforce_unique_by(
            [
                {"class_id": 0, "class_name": "red"},
                {"class_id": 1},  # missing class_name
            ],
            ("class_id", "class_name"),
        )
        errors.append("[runtime:composite_missing_key_should_raise] composite missing-key did NOT raise KeyError")
    except KeyError:
        pass

    # enforce_unique_by: wrong input type must raise TypeError, not silently pass.
    try:
        enforce_unique_by(["not a dict"], "component")
        errors.append("[runtime:non_dict_should_raise] enforce_unique_by accepted non-dict element")
    except TypeError:
        pass

    # validate_and_enforce: combined gate — schema-pass + uniqueness-fail must raise SchemaUniquenessError
    # AFTER schema validation succeeds.
    valid_records_dup_component = [
        {
            "component": "map_prior_gating",
            "outcome": "deploy",
            "reason": "",
            "blocking_artifacts": [],
            "next_round_action": "",
            "branch": "live",
        },
        {
            "component": "map_prior_gating",
            "outcome": "defer",
            "reason": "GPS deadline missed",
            "blocking_artifacts": [],
            "next_round_action": "R3 retry",
            "branch": "replay_only",
        },
    ]
    try:
        validate_and_enforce(valid_records_dup_component, COMPONENT_SCHEMA, ["component"])
        errors.append("[runtime:combined_gate_should_raise] validate_and_enforce passed despite duplicate component")
    except SchemaUniquenessError:
        pass

    # validate_and_enforce: schema-fail must raise ValidationError (uniqueness not even checked).
    schema_fail_records = [
        {
            "component": "ship",  # invalid enum
            "outcome": "deploy",
            "reason": "",
            "blocking_artifacts": [],
            "next_round_action": "",
        }
    ]
    try:
        validate_and_enforce(schema_fail_records, COMPONENT_SCHEMA, ["component"])
        errors.append("[runtime:combined_gate_schema_fail] validate_and_enforce passed despite schema violation")
    except ValidationError:
        pass

    # UNIQUENESS_KEYS mapping: ensure every R2 schema with array-shape output
    # is registered (§1.0 three + §2.1.1 decision schema).
    expected_files = {
        "_r2_component_decision_schema.json",
        "_r2_audit_coverage_schema.json",
        "_r2_carry_forward_schema.json",
        "_r2_decision_schema.json",
    }
    missing = expected_files - set(UNIQUENESS_KEYS.keys())
    if missing:
        errors.append(f"[runtime:uniqueness_keys_registry] missing entries: {sorted(missing)}")

    # Runtime: enforce_unique_by on detector array.
    try:
        enforce_unique_by(
            [
                {"detector": "yolo26_s"},
                {"detector": "yolo26_s"},
            ],
            "detector",
        )
        errors.append("[runtime:detector_dup_should_raise] enforce_unique_by accepted duplicate detector")
    except SchemaUniquenessError:
        pass

    # CRITICAL — cross-file $ref must be load-bearing through the runtime API
    # that production callers (_r2_decide_precision.py / _r2_verify.py b-stage)
    # hit. They call validate_records / validate_and_enforce WITHOUT building a
    # Registry by hand. If the default registry inside load_validator silently
    # dropped the cross-schema $ref, an audit-record self-inconsistency
    # (audit_support=0 + audit_coverage_status='covered') would slip past
    # runtime validation even though the audit schema rejects it.
    bad_decision_input_with_audit_xref_violation = {
        "detector": "yolo26_s",
        "eval_fp16_path": "runs/yolo26_s/eval_fp16.json",
        "eval_fp32_path": "runs/yolo26_s/eval_fp32.json",
        "eval_audit_fp16_path": "runs/yolo26_s/eval_audit_fp16.json",
        "eval_audit_fp32_path": "runs/yolo26_s/eval_audit_fp32.json",
        "timing_fp16_path": "runs/yolo26_s/timing_fp16.json",
        "timing_fp32_path": "runs/yolo26_s/timing_fp32.json",
        "build_variance_path": "runs/_r2_build_variance/yolo26_s.json",
        "audit_coverage_path": "runs/_r2_audit_coverage.json",
        "audit_coverage_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "val_manifest_sha256": "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
        "audit_records": [
            {
                "class_id": 5,
                "class_name": "redRight",
                "full_val_support": 6,
                "full_val_insufficient": True,
                "audit_support": 0,
                "audit_coverage_status": "covered",  # cross-field rule: must be construction_failed
                "audit_low_power": True,
            }
        ],
    }
    try:
        validate_records(
            bad_decision_input_with_audit_xref_violation,
            DECISION_SCHEMA,
            definition="PrecisionDecisionsInput",
        )
        errors.append(
            "[runtime:audit_xref_must_be_load_bearing] validate_records did NOT propagate "
            "the cross-file $ref violation — production callers would silently accept "
            "self-inconsistent audit records"
        )
    except ValidationError:
        pass

    # Confirm get_default_registry actually contains the audit schema URI so
    # the cross-file $ref has a target. Defense-in-depth assertion: the
    # registry's contents must include _r2_audit_coverage_schema.json by URI.
    registry = get_default_registry()
    try:
        registry.contents("_r2_audit_coverage_schema.json")
    except Exception as e:
        errors.append(
            f"[runtime:default_registry_missing_audit_schema] registry does NOT contain "
            f"_r2_audit_coverage_schema.json: {e!r}"
        )

    # Confirm validate_records with `definition=` argument works against
    # the Output shape too (full round-trip via the runtime API).
    valid_decisions_array = [_decision_output_record(detector="yolo26_s")]
    try:
        validate_records(valid_decisions_array, DECISION_SCHEMA, definition="PrecisionDecisionsArray")
    except ValidationError as e:
        errors.append(f"[runtime:output_shape_via_definition] valid output rejected: {e.message}")

    # Confirm definition-not-found raises ValueError (defensive API check).
    try:
        validate_records([], DECISION_SCHEMA, definition="DoesNotExist")
        errors.append("[runtime:definition_not_found] validate_records accepted unknown definition")
    except ValueError:
        pass

    # === C3 MAJOR-A: ConfidenceInterval ordering enforcement ===
    # Schema accepts the malformed CI; the runtime helper must reject it.
    malformed_ci_record = _decision_output_record()
    malformed_ci_record["delta_mAP_0.5"] = {"point": -0.10, "ci_low": 0.20, "ci_high": 0.30}
    try:
        enforce_ci_ordering([malformed_ci_record], CI_FIELDS_PER_DECISION_OUTPUT_RECORD)
        errors.append(
            "[runtime:ci_ordering_should_raise] enforce_ci_ordering accepted "
            "ci_low=0.20 > point=-0.10 (impossible CI)"
        )
    except ValueError:
        pass

    # ci_low > ci_high case
    bad_ci_record_2 = _decision_output_record()
    bad_ci_record_2["delta_mAP_0.5_0.95"] = {"point": 0.0, "ci_low": 0.5, "ci_high": -0.5}
    try:
        enforce_ci_ordering([bad_ci_record_2], CI_FIELDS_PER_DECISION_OUTPUT_RECORD)
        errors.append(
            "[runtime:ci_ordering_inverted_should_raise] enforce_ci_ordering accepted ci_low > ci_high"
        )
    except ValueError:
        pass

    # Well-formed CI passes.
    good_ci_record = _decision_output_record()
    try:
        enforce_ci_ordering([good_ci_record], CI_FIELDS_PER_DECISION_OUTPUT_RECORD)
    except Exception as e:
        errors.append(f"[runtime:ci_ordering_well_formed_should_pass] {e!r}")

    # Single-record (not a list) is also accepted.
    try:
        enforce_ci_ordering(good_ci_record, CI_FIELDS_PER_DECISION_OUTPUT_RECORD)
    except Exception as e:
        errors.append(f"[runtime:ci_ordering_single_record_should_pass] {e!r}")

    # Missing CI path raises KeyError.
    incomplete_record = {"detector": "yolo26_s"}
    try:
        enforce_ci_ordering([incomplete_record], CI_FIELDS_PER_DECISION_OUTPUT_RECORD)
        errors.append("[runtime:ci_ordering_missing_path_should_raise] enforce_ci_ordering accepted missing path")
    except KeyError:
        pass

    # === C3 MAJOR-B: registry registers $id alongside basename ===
    # The audit schema's $id is "scripts/_r2_audit_coverage_schema.json".
    # A future contributor might write `$ref: "scripts/_r2_audit_coverage_schema.json#/items"`
    # and expect resolution. Confirm both URIs resolve.
    registry = get_default_registry()
    for uri in (
        "_r2_audit_coverage_schema.json",
        "scripts/_r2_audit_coverage_schema.json",
        "_r2_decision_schema.json",
        "scripts/_r2_decision_schema.json",
    ):
        try:
            registry.contents(uri)
        except Exception as e:
            errors.append(f"[runtime:registry_uri_{uri}] {e!r}")

    # === C3 MINOR: validate_and_enforce raises TypeError on misuse ===
    # Single-object record + unique_keys = misuse → loud TypeError, not silent.
    valid_input = {
        "detector": "yolo26_s",
        "eval_fp16_path": "runs/yolo26_s/eval_fp16.json",
        "eval_fp32_path": "runs/yolo26_s/eval_fp32.json",
        "eval_audit_fp16_path": "runs/yolo26_s/eval_audit_fp16.json",
        "eval_audit_fp32_path": "runs/yolo26_s/eval_audit_fp32.json",
        "timing_fp16_path": "runs/yolo26_s/timing_fp16.json",
        "timing_fp32_path": "runs/yolo26_s/timing_fp32.json",
        "build_variance_path": "runs/_r2_build_variance/yolo26_s.json",
        "audit_coverage_path": "runs/_r2_audit_coverage.json",
        "audit_coverage_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "val_manifest_sha256": "fedcba9876543210fedcba9876543210fedcba9876543210fedcba9876543210",
        "audit_records": [
            {
                "class_id": 0,
                "class_name": "red",
                "full_val_support": 11601,
                "full_val_insufficient": False,
                "audit_support": 30,
                "audit_coverage_status": "covered",
                "audit_low_power": False,
            }
        ],
    }
    try:
        validate_and_enforce(
            valid_input,
            DECISION_SCHEMA,
            unique_keys=["detector"],
            definition="PrecisionDecisionsInput",
        )
        errors.append(
            "[runtime:misuse_unique_keys_on_object_should_raise] validate_and_enforce "
            "accepted unique_keys with non-list records"
        )
    except TypeError:
        pass

    # validate_and_enforce on non-list records WITHOUT unique_keys must work fine.
    try:
        validate_and_enforce(valid_input, DECISION_SCHEMA, definition="PrecisionDecisionsInput")
    except Exception as e:
        errors.append(f"[runtime:input_validation_no_unique_should_pass] {e!r}")

    # === C3 iter-2 finding: verifier scaffold contract pins CI ordering ===
    # _r2_verify.py::verify_decisions_artifact is currently NotImplementedError
    # but its docstring is the b-stage contract. C3 noted: without explicit
    # mention of enforce_ci_ordering in the verifier scaffold, a b-stage author
    # could miss the gate, leaving CI ordering load-bearing only at write-time
    # (not at round close). Pin the contract via meta-test.
    verify_path = SCRIPTS / "_r2_verify.py"
    verify_src = verify_path.read_text()
    if "enforce_ci_ordering" not in verify_src:
        errors.append(
            "[runtime:verify_scaffold_must_import_ci_ordering] _r2_verify.py "
            "does not import enforce_ci_ordering — round close would skip the gate"
        )

    # Extract verify_decisions_artifact's docstring and pin the three-gate
    # contract markers so a docstring drift surfaces here.
    import inspect
    import importlib.util
    spec = importlib.util.spec_from_file_location("_r2_verify_module", verify_path)
    if spec is not None and spec.loader is not None:
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            errors.append(f"[runtime:verify_scaffold_load] {e!r}")
        else:
            doc = inspect.getdoc(module.verify_decisions_artifact) or ""
            for marker in ("validate_and_enforce", "enforce_ci_ordering", "selected_artifact_path"):
                if marker not in doc:
                    errors.append(
                        f"[runtime:verify_scaffold_docstring_drift] "
                        f"verify_decisions_artifact docstring missing '{marker}' contract reference"
                    )


def main() -> int:
    errors: list[str] = []
    test_component_decision(errors)
    test_audit_coverage(errors)
    test_carry_forward(errors)
    test_decision_output(errors)
    test_decision_input(errors)
    test_uniqueness_guards(errors)
    test_runtime_enforcement(errors)
    if errors:
        print("FAIL — schema self-test surfaced issues:")
        for line in errors:
            print(f"  {line}")
        return 1
    print("OK — four R2 schemas pass draft-07 self-check + cross-file $ref + positive/negative fixtures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
