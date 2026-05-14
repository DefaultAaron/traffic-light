"""2-arm ablation aggregator: no_hn vs with_hn.

Drives the d-stage flow end-to-end:

  0. Cross-artifact equality (the runner is the only layer that holds
     all of: config YAML + frozen FP manifest file + per-arm eval JSONs).

     Note (B2 final-review I2 2026-05-10): ``config_yaml_sha256`` is
     a TOP-LEVEL ONLY field on the output artifact — single-source
     artifact, no per-cell mirror by design. Do NOT add a per-cell
     ``config_yaml_sha256`` "for symmetry"; the schema would reject it
     and the cross-block equality pattern doesn't apply (only one
     config is consumed per ablation run).

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
        # the silent-arm-collapse case.
        #
        # Codex stop-gate fix: compare canonical forms via Path.resolve()
        # so that lexically distinct paths pointing at the same file are
        # still detected — symlinks, relative-vs-absolute drift, and
        # ``..`` traversal all alias multiple lexical paths onto one
        # inode. Path equality alone (``Path("./a") == Path("/abs/a")``
        # is False) misses these. ``resolve(strict=False)`` handles
        # non-existent paths (e.g. output_json before write) without
        # error. Hard links are out of scope.
        canonical_by_field: list[tuple[str, Path, Path]] = []
        for name, value in path_fields:
            try:
                canonical = value.resolve(strict=False)
            except (OSError, RuntimeError) as e:
                raise ValueError(
                    f"AblationConfig path field {name}={value!r} could not "
                    f"be canonicalized via Path.resolve(strict=False): {e}"
                ) from e
            canonical_by_field.append((name, value, canonical))
        canonicals = [canon for _, _, canon in canonical_by_field]
        if len(set(canonicals)) != len(canonicals):
            seen: dict[Path, list[tuple[str, Path]]] = {}
            for name, value, canon in canonical_by_field:
                seen.setdefault(canon, []).append((name, value))
            duplicates = {
                str(canon): [
                    f"{name} (input={value!s})" for name, value in pairs
                ]
                for canon, pairs in seen.items()
                if len(pairs) > 1
            }
            raise ValueError(
                f"AblationConfig path fields must canonicalize to distinct "
                f"filesystem locations; got {len(canonicals)} fields "
                f"collapsing to {len(set(canonicals))} unique resolved "
                f"paths (duplicates by canonical form: {duplicates}). "
                f"Common offenders: output_json equal to any input path "
                f"(would clobber the input mid-write); no_hn_eval_json == "
                f"with_hn_eval_json — lexically OR via symlink / "
                f"relative-vs-absolute drift — would silently collapse the "
                f"two arms onto one eval source. The check uses "
                f"Path.resolve(strict=False) so symlinks and ``..`` "
                f"traversal are normalized."
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
    """
    import hashlib
    import json
    import os

    import jsonschema

    from components.hard_negative_mining._internals import is_hex_sha256
    from components.hard_negative_mining.config import (
        load_hard_negative_mining_yaml,
    )
    from components.hard_negative_mining.data.eval_manifest import (
        load_frozen_eval_manifest,
    )
    from components.hard_negative_mining.gates.ablation_gate import (
        compute_arm_metrics,
    )
    from components.hard_negative_mining.gates.decision_gate import (
        DecisionInputs,
        apply_decision_rule,
    )

    # ----- Step 0: load configs, manifest, and per-arm eval JSONs ----------
    cfg = load_hard_negative_mining_yaml(config.config_yaml)
    manifest = load_frozen_eval_manifest(config.frozen_manifest_json)

    config_yaml_sha = hashlib.sha256(
        config.config_yaml.read_bytes()
    ).hexdigest()

    def _load_eval(path: Path, arm_label: str) -> dict:
        if not path.exists():
            raise FileNotFoundError(
                f"{arm_label} eval JSON does not exist: {path}"
            )
        with path.open("r", encoding="utf-8") as fh:
            eval_data = json.load(fh)
        if not isinstance(eval_data, dict):
            raise ValueError(
                f"{path}: {arm_label} eval JSON top-level must be an object"
            )
        # Required-key check + light type validation (the dataclasses do the
        # rest at construction).
        required = {
            "fp_count": int,
            "real_light_recall": float,
            "total_mAP_at_0_5": float,
            "data_yaml_sha256": str,
            "eval_manifest_sha256": str,
            "fp_manifest_sha256": str,
            "map_no_regression": bool,
            "map_regression_tolerance_pp": float,
        }
        for key, expected_type in required.items():
            if key not in eval_data:
                raise ValueError(
                    f"{path}: {arm_label} eval JSON missing required key "
                    f"{key!r}"
                )
            value = eval_data[key]
            # Allow int for float fields (JSON serializes 0 as 0).
            if expected_type is float and isinstance(value, int) and not isinstance(value, bool):
                eval_data[key] = float(value)
                continue
            # Strict bool check first (Python bool is int).
            if expected_type is bool and not isinstance(value, bool):
                raise ValueError(
                    f"{path}: {arm_label} eval JSON key {key!r} must be bool; "
                    f"got {type(value).__name__}={value!r}"
                )
            if expected_type is int and (
                not isinstance(value, int) or isinstance(value, bool)
            ):
                raise ValueError(
                    f"{path}: {arm_label} eval JSON key {key!r} must be int; "
                    f"got {type(value).__name__}={value!r}"
                )
            if expected_type is float and not isinstance(value, float):
                raise ValueError(
                    f"{path}: {arm_label} eval JSON key {key!r} must be float; "
                    f"got {type(value).__name__}={value!r}"
                )
            if expected_type is str and not isinstance(value, str):
                raise ValueError(
                    f"{path}: {arm_label} eval JSON key {key!r} must be str; "
                    f"got {type(value).__name__}={value!r}"
                )
        # B2 review MINOR-8 fix: validate hex format AT the per-key
        # validation loop so the diagnostic names which eval JSON
        # produced the malformed hash (rather than surfacing later
        # as a generic "not valid hex" downstream of cross-arm checks).
        for hash_key in (
            "data_yaml_sha256", "eval_manifest_sha256", "fp_manifest_sha256",
        ):
            if not is_hex_sha256(eval_data[hash_key]):
                raise ValueError(
                    f"{path}: {arm_label} eval JSON key {hash_key!r} must "
                    f"be 64-char lowercase hex; got {eval_data[hash_key]!r}"
                )
        return eval_data

    no_hn = _load_eval(config.no_hn_eval_json, "no_hn")
    with_hn = _load_eval(config.with_hn_eval_json, "with_hn")

    # ----- Step 0: cross-artifact equality checks --------------------------
    # Manifest self-hash already verified inside load_frozen_eval_manifest;
    # cross-check that both eval JSONs reference that same hash.
    for arm_label, ev in (("no_hn", no_hn), ("with_hn", with_hn)):
        if ev["fp_manifest_sha256"] != manifest.manifest_sha256:
            raise ValueError(
                f"{arm_label} eval JSON's fp_manifest_sha256 "
                f"({ev['fp_manifest_sha256']!r}) does not match the loaded "
                f"frozen FP manifest's manifest_sha256 "
                f"({manifest.manifest_sha256!r}). The trainer must have "
                f"evaluated this arm against a different FP manifest than "
                f"the one the runner is consuming — §4.7 anti-gaming check "
                f"failed."
            )
    if no_hn["eval_manifest_sha256"] != with_hn["eval_manifest_sha256"]:
        raise ValueError(
            f"no_hn / with_hn disagree on eval_manifest_sha256 "
            f"({no_hn['eval_manifest_sha256']!r} vs "
            f"{with_hn['eval_manifest_sha256']!r}). Every cell must consume "
            f"the SAME frozen R2 val manifest."
        )
    if no_hn["data_yaml_sha256"] != with_hn["data_yaml_sha256"]:
        raise ValueError(
            f"no_hn / with_hn disagree on data_yaml_sha256 "
            f"({no_hn['data_yaml_sha256']!r} vs "
            f"{with_hn['data_yaml_sha256']!r}). Class-label drift across "
            f"arms silently corrupts AP comparisons."
        )
    if cfg.data_yaml_sha256 != no_hn["data_yaml_sha256"]:
        raise ValueError(
            f"config data_yaml_sha256 ({cfg.data_yaml_sha256!r}) does not "
            f"match the per-arm eval JSONs' data_yaml_sha256 "
            f"({no_hn['data_yaml_sha256']!r}). The YAML must point at the "
            f"SAME data.yaml the trainer evaluated against."
        )
    if no_hn["map_regression_tolerance_pp"] != with_hn["map_regression_tolerance_pp"]:
        raise ValueError(
            f"no_hn / with_hn disagree on map_regression_tolerance_pp "
            f"({no_hn['map_regression_tolerance_pp']} vs "
            f"{with_hn['map_regression_tolerance_pp']}). Tolerance is a "
            f"runner-side knob; the SAME value must stamp every cell."
        )
    if cfg.map_regression_tolerance_pp != no_hn["map_regression_tolerance_pp"]:
        raise ValueError(
            f"config map_regression_tolerance_pp "
            f"({cfg.map_regression_tolerance_pp}) does not match the per-arm "
            f"eval JSONs' map_regression_tolerance_pp "
            f"({no_hn['map_regression_tolerance_pp']}). The YAML's tolerance "
            f"is authoritative and must equal the upstream eval pipeline's."
        )
    if not no_hn["map_no_regression"]:
        raise ValueError(
            "no_hn eval JSON's map_no_regression is False — baseline is "
            "self-compared so the verdict is trivially True. Either the "
            "trainer pipeline mis-stamped the baseline cell or the wrong "
            "JSON was passed as --no-hn-eval."
        )

    eval_manifest_sha = no_hn["eval_manifest_sha256"]
    fp_manifest_sha = manifest.manifest_sha256
    data_yaml_sha = cfg.data_yaml_sha256
    tolerance = cfg.map_regression_tolerance_pp

    if not is_hex_sha256(eval_manifest_sha):
        raise ValueError(
            f"eval_manifest_sha256 from eval JSONs is not valid hex: "
            f"{eval_manifest_sha!r}"
        )

    # ----- Step 1: build ArmMetrics for both arms --------------------------
    baseline_arm = compute_arm_metrics(
        arm_id="no_hn",
        is_baseline_reference=True,
        baseline_fp_count=no_hn["fp_count"],
        candidate_fp_count=no_hn["fp_count"],
        baseline_real_light_recall=no_hn["real_light_recall"],
        candidate_real_light_recall=no_hn["real_light_recall"],
        baseline_total_map=no_hn["total_mAP_at_0_5"],
        candidate_total_map=no_hn["total_mAP_at_0_5"],
        eval_manifest_sha256=eval_manifest_sha,
        fp_manifest_sha256=fp_manifest_sha,
        map_no_regression=True,
        map_regression_tolerance_pp=tolerance,
        data_yaml_sha256=data_yaml_sha,
    )
    candidate_arm = compute_arm_metrics(
        arm_id="with_hn",
        is_baseline_reference=False,
        baseline_fp_count=no_hn["fp_count"],
        candidate_fp_count=with_hn["fp_count"],
        baseline_real_light_recall=no_hn["real_light_recall"],
        candidate_real_light_recall=with_hn["real_light_recall"],
        baseline_total_map=no_hn["total_mAP_at_0_5"],
        candidate_total_map=with_hn["total_mAP_at_0_5"],
        eval_manifest_sha256=eval_manifest_sha,
        fp_manifest_sha256=fp_manifest_sha,
        map_no_regression=with_hn["map_no_regression"],
        map_regression_tolerance_pp=tolerance,
        data_yaml_sha256=data_yaml_sha,
    )

    # ----- Step 2: apply decision rule on with_hn --------------------------
    # B2 review MAJOR-6 fix: source ``map_no_regression`` from the
    # ArmMetrics dataclass field, not the raw eval dict (single source of
    # truth — see CPB sister runner for full rationale).
    inputs = DecisionInputs(
        baseline=baseline_arm,
        candidate=candidate_arm,
        map_no_regression=candidate_arm.map_no_regression,
        map_regression_tolerance_pp=tolerance,
    )
    decision_result = apply_decision_rule(inputs)

    # ----- Step 3: assemble output dict per schema -------------------------
    def _arm_to_block(arm) -> dict:
        return {
            "fp_drop_frac": arm.fp_drop_frac,
            "real_light_recall_delta_pp": arm.real_light_recall_delta_pp,
            "total_map_delta_pp": arm.total_map_delta_pp,
            "is_baseline_reference": arm.is_baseline_reference,
            "map_no_regression": arm.map_no_regression,
            "map_regression_tolerance_pp": arm.map_regression_tolerance_pp,
            "eval_manifest_sha256": arm.eval_manifest_sha256,
            "fp_manifest_sha256": arm.fp_manifest_sha256,
            "data_yaml_sha256": arm.data_yaml_sha256,
        }

    candidate_block = _arm_to_block(candidate_arm)
    baseline_block = _arm_to_block(baseline_arm)
    artifact = {
        "schema_version": "1",
        "config_yaml_sha256": config_yaml_sha,
        "data_yaml_sha256": data_yaml_sha,
        "fp_manifest_sha256": fp_manifest_sha,
        "eval_manifest_sha256": eval_manifest_sha,
        "headline_arm": "with_hn",
        "headline_metrics": candidate_block,
        "headline_decision": decision_result.decision.value,
        "notes": (
            f"§四 hard-negative-mining ablation; anchor_arm=with_hn; "
            f"tolerance={tolerance}pp; manifest entries={len(manifest.entries)}"
        ),
        "no_hn": {
            "metrics": baseline_block,
            "notes": (
                f"no_hn baseline reference (self-comparison); "
                f"fp_count={no_hn['fp_count']}, "
                f"recall={no_hn['real_light_recall']:.6f}, "
                f"mAP={no_hn['total_mAP_at_0_5']:.6f}"
            ),
        },
        "with_hn": {
            "metrics": candidate_block,
            "decision": decision_result.decision.value,
            "notes": decision_result.notes,
        },
    }

    # ----- Step 4: schema-validate before assert_artifact_invariants -------
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "gates"
        / "_hard_negative_decision_schema.json"
    )
    with schema_path.open("r", encoding="utf-8") as fh:
        schema = json.load(fh)
    jsonschema.validate(instance=artifact, schema=schema)

    # ----- Step 5: cross-row invariants (the schema can't express) --------
    assert_artifact_invariants(artifact)

    # ----- Step 6: atomic write .tmp → final --------------------------------
    # B2 review MAJOR-3: atomic-write hygiene. Use a process-unique tmp
    # name (mkstemp) so concurrent runs sharing the output directory don't
    # race on the same .tmp filename. Unlink the tmp on any exception
    # between fdopen and replace so a partial / orphaned .tmp doesn't
    # pollute runs/.
    import tempfile

    out_path = config.output_json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=out_path.name + ".",
        suffix=".tmp",
        dir=str(out_path.parent),
    )
    # B2 iter-2 MINOR-2 fix: enter try IMMEDIATELY after mkstemp so an
    # asynchronous KeyboardInterrupt arriving between mkstemp and the
    # try block cannot leak both the fd and the .tmp file.
    try:
        tmp_path = Path(tmp_name)
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
            json.dump(artifact, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp_path, out_path)
    except BaseException:
        try:
            os.close(tmp_fd)
        except OSError:
            pass
        try:
            Path(tmp_name).unlink()
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    """CLI entry: parse args → build ``AblationConfig`` → ``run_ablation``.

    Argparse contract:
        --no-hn-eval FILE       baseline arm's eval JSON (must carry
                                 map_no_regression=True)
        --with-hn-eval FILE     candidate arm's eval JSON
        --config FILE           configs/hard_negative_mining.yaml path
        --frozen-manifest FILE  runs/_hard_negative_eval_manifest.json path
        --output FILE           runs/_hard_negative_decision.json path
        --anchor-arm {with_hn}  REQUIRED; single-value choice (with_hn).
                                 The single-element enum is intentional
                                 defense (mirrors the schema's
                                 headline_arm enum) — no_hn is rejected.

    Returns:
        0 — AGREED-CLEAN run; output JSON written.
        1 — ``executor_error`` in the headline row (output JSON still
            written but contains diagnostic notes).
        2 — internal contract violation (e.g. schema validation failure
            after ``assert_artifact_invariants`` passed; should be
            unreachable in practice).
    """
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(
        prog="hard_negative_mining.runners.ablation",
        description=(
            "§四 hard-negative-mining ablation d-stage runner. Aggregates "
            "no_hn / with_hn eval JSONs into runs/_hard_negative_decision.json "
            "per plan §4.7 (Codex stop-gate fix 2026-05-10: defer is "
            "mAP-agnostic; only deploy requires map_no_regression=True)."
        ),
    )
    parser.add_argument("--no-hn-eval", type=Path, required=True)
    parser.add_argument("--with-hn-eval", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--frozen-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--anchor-arm",
        choices=("with_hn",),
        required=True,
        help=(
            "Single-value enum {with_hn} mirroring the schema's "
            "headline_arm enum. no_hn is REJECTED — baseline carries no "
            "decision. Required at the CLI to make the constraint "
            "explicit."
        ),
    )
    args = parser.parse_args()

    try:
        config = AblationConfig(
            no_hn_eval_json=args.no_hn_eval,
            with_hn_eval_json=args.with_hn_eval,
            config_yaml=args.config,
            frozen_manifest_json=args.frozen_manifest,
            output_json=args.output,
            anchor_arm=args.anchor_arm,
        )
        run_ablation(config)
    except (FileNotFoundError, ValueError) as e:
        # AblationConfig.__post_init__ ValueErrors (distinct-path,
        # anchor-arm, etc.) AND run_ablation Step 0 hard fails BOTH
        # surface here. Both are pre-output-write conditions so exit 2
        # (no artifact produced); the runner does not write a partial
        # output JSON on these paths.
        print(f"hard_negative_mining ablation: {type(e).__name__}: {e}",
              file=sys.stderr)
        return 2

    # Re-read the output to determine the exit code from the headline
    # decision. The runner doesn't return a DecisionResult so this is the
    # canonical post-run check.
    with args.output.open("r", encoding="utf-8") as fh:
        artifact = json.load(fh)
    return 1 if artifact["headline_decision"] == "executor_error" else 0


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
      9. **Decision-vs-verdict consistency (Codex stop-gate fix
         2026-05-10)**: plan §4.7 deploy gate requires total mAP
         no-regression; plan §4.7 defer gate does NOT (defer is gated
         solely on fp_drop ∈ [20%, 50%) AND recall_delta ≥ −0.5 pp).
         Therefore a cell with ``decision == "deploy"`` AND
         ``metrics.map_no_regression == False`` is malformed; defer +
         regression is plan-legal. The §3.7 sister's parallel check
         covers both deploy and defer because §3.7's plan prose
         explicitly extends mAP no-regression to defer; do NOT
         carry that constraint into §四.

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

    # Codex stop-gate fix 2026-05-10: decision-vs-verdict consistency for
    # DEPLOY ONLY. Plan §4.7 deploy gate requires total mAP no-regression;
    # plan §4.7 defer gate does NOT (defer is gated solely on fp_drop in
    # [20%, 50%) AND recall_delta ≥ −0.5 pp). The earlier iter-1 check
    # rejected defer-with-regression too, mistakenly carrying the §3.7
    # sister's "defer requires mAP no-regression" constraint into §四
    # where the plan prose excludes it. drop is compatible with mAP
    # regression by design (drop is the catch-all).
    decision = with_hn["decision"]
    metrics = with_hn["metrics"]
    if decision == "deploy" and metrics["map_no_regression"] is not True:
        raise ValueError(
            f"with_hn.decision={decision!r} is incompatible with "
            f"metrics.map_no_regression={metrics['map_no_regression']!r} — "
            f"plan §4.7 deploy gate requires total mAP no-regression; a "
            f"cell that regressed total mAP MUST land on defer (if other "
            f"defer guards pass) or drop, not deploy (malformed decision "
            f"artifact)"
        )
    # Same check on the headline (it must mirror with_hn but we re-check
    # in case headline copy was correct but the verdict is still inconsistent).
    headline_metrics = artifact["headline_metrics"]
    headline_decision = artifact["headline_decision"]
    if headline_decision == "deploy" and headline_metrics["map_no_regression"] is not True:
        raise ValueError(
            f"headline_decision={headline_decision!r} is incompatible with "
            f"headline_metrics.map_no_regression="
            f"{headline_metrics['map_no_regression']!r}"
        )


if __name__ == "__main__":
    raise SystemExit(main())
