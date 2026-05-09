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
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT7

UniqueKey = Union[str, Sequence[str]]

_SCRIPTS_DIR = Path(__file__).resolve().parent
_DEFAULT_REGISTRY: Registry | None = None


class SchemaUniquenessError(ValueError):
    """Spec-level 'unique by sub-key' violation; JSON Schema draft-07 cannot
    express this constraint structurally because uniqueItems compares whole
    records."""


def _build_default_registry() -> Registry:
    """Build the cross-schema registry by loading every R2 schema declared in
    UNIQUENESS_KEYS from this directory. The decision schema's cross-file
    `$ref` (`_r2_audit_coverage_schema.json#/items`) is only load-bearing
    when the validator has access to a registry that knows the referenced
    schema. Caching at module level prevents O(N) file reads per validation.

    Each schema is registered under BOTH its bare basename AND its declared
    `$id`. C3 found that registering by basename only would let a future
    contributor write `$ref: "scripts/_r2_audit_coverage_schema.json#..."`
    (matching the schema's own `$id`) and silently miss resolution. Registering
    both URIs eliminates that brittle convention mismatch.

    Schemas absent from disk are silently skipped so this module can still be
    imported in environments where only a subset of the schemas is present.
    """
    registry = Registry()
    for basename in UNIQUENESS_KEYS:
        path = _SCRIPTS_DIR / basename
        if path.exists():
            content = json.loads(path.read_text())
            resource = Resource(contents=content, specification=DRAFT7)
            registry = registry.with_resource(uri=basename, resource=resource)
            schema_id = content.get("$id")
            if schema_id and schema_id != basename:
                registry = registry.with_resource(uri=schema_id, resource=resource)
    return registry


def get_default_registry() -> Registry:
    """Return the lazily-built cross-schema registry; callers wanting an
    uncached fresh one should call _build_default_registry directly.
    """
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = _build_default_registry()
    return _DEFAULT_REGISTRY


def load_validator(
    schema_path: Path,
    definition: str | None = None,
    registry: Registry | None = None,
) -> Draft7Validator:
    """Load schema, run draft-07 meta-check, return a Validator with the
    cross-schema registry attached.

    Cross-file `$ref` (e.g. _r2_decision_schema.json's class_wise items pointing
    at _r2_audit_coverage_schema.json#/items) is load-bearing only when the
    validator has access to a registry that knows the referenced schema —
    which it does by default here.

    Args:
        schema_path: path to the JSON Schema draft-07 file.
        definition: optional name in the schema's `definitions/` block (e.g.
            'PrecisionDecisionsInput'). When provided, the returned validator
            validates against `#/definitions/{definition}` rather than the
            schema root. Required for multi-shape schemas like
            _r2_decision_schema.json that have no top-level type constraint.
        registry: optional override; defaults to `get_default_registry()`.

    Raises:
        jsonschema.exceptions.SchemaError: malformed schema.
        ValueError: definition not found in schema's definitions block.
    """
    with Path(schema_path).open() as f:
        schema = json.load(f)
    Draft7Validator.check_schema(schema)
    actual_registry = registry or get_default_registry()
    if definition is None:
        return Draft7Validator(schema, registry=actual_registry)
    defs = schema.get("definitions", {})
    if definition not in defs:
        raise ValueError(
            f"definition {definition!r} not found in {schema_path.name}; "
            f"available: {sorted(defs.keys())}"
        )
    return Draft7Validator(
        {"$ref": f"#/definitions/{definition}", "definitions": defs},
        registry=actual_registry,
    )


def validate_records(records, schema_path: Path, definition: str | None = None) -> None:
    """Validate records against schema. Raises ValidationError on first failure.

    Args:
        records: input to validate.
        schema_path: path to the JSON Schema draft-07 file.
        definition: optional definition name for multi-shape schemas. Same
            semantics as `load_validator`'s `definition` argument.

    Use iter_errors-style fixture drivers (test code) instead of this when you
    want to surface ALL errors at once. This function is for runtime gates that
    should refuse to proceed on any violation.
    """
    validator = load_validator(schema_path, definition=definition)
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
    definition: str | None = None,
) -> None:
    """Combined gate: schema validation + uniqueness enforcement.

    Args:
        records: the JSON-decoded array (or single object for non-array shapes).
        schema_path: path to the JSON Schema draft-07 file.
        unique_keys: optional iterable of keys (each a string or sequence of
            strings) — every key is checked for sub-array uniqueness. MUST
            be omitted when `records` is not a list; passing both is a misuse
            (uniqueness is undefined on a single object) and raises TypeError.
            For multi-shape schemas like _r2_decision_schema.json: pass
            unique_keys ONLY when validating the array shape (definition=
            "PrecisionDecisionsArray"); omit when validating the single-object
            shape (definition="PrecisionDecisionsInput").
        definition: optional definition name for multi-shape schemas. Same
            semantics as `load_validator`.

    Raises:
        ValidationError: schema-level failure.
        SchemaUniquenessError: spec-level uniqueness violation.
        TypeError: unique_keys passed alongside non-list records (loud failure
            for what would otherwise be a silent no-op or confusing crash).

    Calling order is: schema first (catches type/enum/required/conditional
    failures before uniqueness can be meaningfully evaluated), then each
    uniqueness key in the order given. First failure wins.
    """
    validate_records(records, schema_path, definition=definition)
    if unique_keys:
        if not isinstance(records, list):
            raise TypeError(
                f"validate_and_enforce: unique_keys passed but records is "
                f"{type(records).__name__}, not list. Uniqueness is undefined "
                f"on a single object — omit unique_keys for non-array shapes."
            )
        for key in unique_keys:
            enforce_unique_by(records, key)


def enforce_ci_ordering(
    records,
    ci_field_paths: Iterable[Sequence[str]],
) -> None:
    """Enforce ci_low <= point <= ci_high on every ConfidenceInterval-shaped
    sub-object reachable via the given field paths. JSON Schema draft-07
    cannot express cross-property numeric ordering, so this is the runtime
    gate that closes the gap.

    Args:
        records: list of records (or a single dict — handled both ways).
        ci_field_paths: each entry is a sequence of keys naming a nested
            CI object, e.g. ["delta_mAP_0.5"] or ["delta_mAP_0.5_0.95"]. The
            resolved sub-object MUST have keys 'point', 'ci_low', 'ci_high'.

    Raises:
        ValueError: any CI sub-object has ci_low > point or point > ci_high.
        KeyError: a path doesn't resolve to an existing dict (precondition
            failure; schema validation should have caught this earlier).
    """
    seq = records if isinstance(records, list) else [records]
    for idx, rec in enumerate(seq):
        if not isinstance(rec, dict):
            continue
        for path in ci_field_paths:
            cursor = rec
            for key in path:
                if not isinstance(cursor, dict) or key not in cursor:
                    raise KeyError(
                        f"enforce_ci_ordering precondition failure: record "
                        f"idx {idx} missing path {list(path)}"
                    )
                cursor = cursor[key]
            ci_low = cursor.get("ci_low")
            point = cursor.get("point")
            ci_high = cursor.get("ci_high")
            if ci_low is None or point is None or ci_high is None:
                raise KeyError(
                    f"enforce_ci_ordering: record idx {idx} path {list(path)} "
                    f"missing ci_low/point/ci_high"
                )
            if not (ci_low <= point <= ci_high):
                raise ValueError(
                    f"CI ordering violation at record idx {idx} path {list(path)}: "
                    f"ci_low={ci_low} point={point} ci_high={ci_high} "
                    f"(must satisfy ci_low <= point <= ci_high)"
                )


# Canonical CI field paths for _r2_decision_schema.json PrecisionDecisionOutput
# records. Production callers (_r2_decide_precision.py / _r2_verify.py b-stage)
# pass this constant to enforce_ci_ordering after schema validation passes.
CI_FIELDS_PER_DECISION_OUTPUT_RECORD = (
    ("delta_mAP_0.5",),
    ("delta_mAP_0.5_0.95",),
)


# Convention: each R2 schema declares its uniqueness keys here so writers/
# verifiers can call validate_and_enforce(records, SCHEMA, KEYS) without
# rederiving the spec rules. Updating these requires bumping the schema's $id
# and updating callers.
UNIQUENESS_KEYS = {
    "_r2_component_decision_schema.json": ["component"],
    "_r2_audit_coverage_schema.json": ["class_id", "class_name"],
    "_r2_carry_forward_schema.json": ["item_id"],
    # _r2_decision_schema.json's persisted output is PrecisionDecisionsArray
    # (one record per R2 detector). Input shape (PrecisionDecisionsInput) is
    # a single record, not an array, and has no array-level uniqueness rule.
    "_r2_decision_schema.json": ["detector"],
}


__all__ = [
    "SchemaUniquenessError",
    "ValidationError",
    "UNIQUENESS_KEYS",
    "CI_FIELDS_PER_DECISION_OUTPUT_RECORD",
    "load_validator",
    "get_default_registry",
    "validate_records",
    "enforce_unique_by",
    "enforce_ci_ordering",
    "validate_and_enforce",
]
