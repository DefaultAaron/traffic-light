"""Self-test for the three R2 shared schemas (§1.0 c-stage).

Validates each schema against JSON Schema draft-07 meta-schema, then runs
positive + negative fixtures asserting the conditional rules pinned in
docs/planning/pre_r2_kickoff_checklist.md v1.4 §1.0.

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

from _r2_schema_utils import (
    SchemaUniquenessError,
    UNIQUENESS_KEYS,
    enforce_unique_by,
    validate_and_enforce,
    validate_records,
)

SCRIPTS = Path(__file__).resolve().parent
COMPONENT_SCHEMA = SCRIPTS / "_r2_component_decision_schema.json"
AUDIT_SCHEMA = SCRIPTS / "_r2_audit_coverage_schema.json"
CARRY_SCHEMA = SCRIPTS / "_r2_carry_forward_schema.json"

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

    # UNIQUENESS_KEYS mapping: ensure every R2 §1.0 schema is registered.
    expected_files = {
        "_r2_component_decision_schema.json",
        "_r2_audit_coverage_schema.json",
        "_r2_carry_forward_schema.json",
    }
    missing = expected_files - set(UNIQUENESS_KEYS.keys())
    if missing:
        errors.append(f"[runtime:uniqueness_keys_registry] missing entries: {sorted(missing)}")


def main() -> int:
    errors: list[str] = []
    test_component_decision(errors)
    test_audit_coverage(errors)
    test_carry_forward(errors)
    test_uniqueness_guards(errors)
    test_runtime_enforcement(errors)
    if errors:
        print("FAIL — schema self-test surfaced issues:")
        for line in errors:
            print(f"  {line}")
        return 1
    print("OK — three R2 schemas pass draft-07 self-check + positive/negative fixtures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
