"""Runtime validation utilities for R2 §1.0 shared schemas.

Schema-level rules (types, enums, conditionals, patterns) are enforced by
draft-07 validation against the JSON Schema files. Spec-level rules that
draft-07 cannot express — sub-key uniqueness across array members — are
enforced here so they are load-bearing at write/verify time, not just in
the c-stage test fixture driver.

Importable by:
- scripts/_r2_schemas_test.py (c-stage fixture driver)
- scripts/_r2_decide_precision.py / _kd_decide_cell.py / _tsm_decide_phase.py
  (d-stage executors that WRITE these JSON files — must call validate_and_enforce
  before persisting)
- scripts/_r2_verify.py (round-close verifier that READS these files — must call
  validate_and_enforce before declaring the round well-formed)

All functions raise on contract violation. Callers decide policy.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence, Union

from jsonschema import Draft7Validator
from jsonschema.exceptions import ValidationError

UniqueKey = Union[str, Sequence[str]]


class SchemaUniquenessError(ValueError):
    """Spec-level 'unique by sub-key' violation; JSON Schema draft-07 cannot
    express this constraint structurally because uniqueItems compares whole
    records."""


def load_validator(schema_path: Path) -> Draft7Validator:
    """Load schema, run draft-07 meta-check, return a Validator.

    Raises:
        jsonschema.exceptions.SchemaError: if the schema itself is malformed.
    """
    with Path(schema_path).open() as f:
        schema = json.load(f)
    Draft7Validator.check_schema(schema)
    return Draft7Validator(schema)


def validate_records(records, schema_path: Path) -> None:
    """Validate records against schema. Raises ValidationError on first failure.

    Use iter_errors-style fixture drivers (test code) instead of this when you
    want to surface ALL errors at once. This function is for runtime gates that
    should refuse to proceed on any violation.
    """
    validator = load_validator(schema_path)
    errors = list(validator.iter_errors(records))
    if errors:
        raise errors[0]


def enforce_unique_by(records, key: UniqueKey) -> None:
    """Enforce sub-key uniqueness across array members.

    Precondition: every record MUST contain every key field. Missing fields
    raise KeyError immediately rather than collapsing to a None-coincidence
    "uniqueness violation" — that ambiguity was C3-flagged. Callers must
    schema-validate first (`validate_records` or `validate_and_enforce`) so
    required-field absence is caught by the schema's own `required` rule
    before this helper runs.

    Args:
        records: array (list of dicts) — the same shape JSON Schema validates
            with `type: array`.
        key: either a single field name or a tuple/list of field names for
            composite uniqueness.

    Raises:
        SchemaUniquenessError: with offending indices and key value.
        TypeError: if `records` is not iterable of dicts.
        KeyError: if any record is missing a key field (precondition failure;
            schema validation should have caught this earlier).
    """
    if isinstance(key, str):
        single_key = key

        def getter(r):
            if single_key not in r:
                raise KeyError(
                    f"enforce_unique_by precondition failure: record missing required field {single_key!r} "
                    f"— schema validation must run first"
                )
            return r[single_key]
        key_repr = key
    else:
        keys = tuple(key)

        def getter(r):
            for k in keys:
                if k not in r:
                    raise KeyError(
                        f"enforce_unique_by precondition failure: record missing required field {k!r} "
                        f"in composite key {keys!r} — schema validation must run first"
                    )
            return tuple(r[k] for k in keys)
        key_repr = "(" + ", ".join(keys) + ")"

    seen: dict = {}
    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise TypeError(
                f"enforce_unique_by expected dict at idx {idx}, got {type(rec).__name__}"
            )
        k = getter(rec)
        if k in seen:
            raise SchemaUniquenessError(
                f"uniqueness violation on {key_repr}: idx {seen[k]} and idx {idx} share key={k!r}"
            )
        seen[k] = idx


def validate_and_enforce(
    records,
    schema_path: Path,
    unique_keys: Iterable[UniqueKey] | None = None,
) -> None:
    """Combined gate: schema validation + uniqueness enforcement.

    Args:
        records: the JSON-decoded array.
        schema_path: path to the JSON Schema draft-07 file.
        unique_keys: optional iterable of keys (each a string or sequence of
            strings) — every key is checked for sub-array uniqueness.

    Raises:
        ValidationError: schema-level failure.
        SchemaUniquenessError: spec-level uniqueness violation.

    Calling order is: schema first (catches type/enum/required/conditional
    failures before uniqueness can be meaningfully evaluated), then each
    uniqueness key in the order given. First failure wins.
    """
    validate_records(records, schema_path)
    if unique_keys:
        for key in unique_keys:
            enforce_unique_by(records, key)


# Convention: each R2 §1.0 schema declares its uniqueness keys here so
# writers/verifiers can call validate_and_enforce(records, SCHEMA, KEYS) without
# rederiving the spec rules. Updating these requires bumping the schema's $id
# and updating callers.
UNIQUENESS_KEYS = {
    "_r2_component_decision_schema.json": ["component"],
    "_r2_audit_coverage_schema.json": ["class_id", "class_name"],
    "_r2_carry_forward_schema.json": ["item_id"],
}


__all__ = [
    "SchemaUniquenessError",
    "ValidationError",
    "UNIQUENESS_KEYS",
    "load_validator",
    "validate_records",
    "enforce_unique_by",
    "validate_and_enforce",
]
