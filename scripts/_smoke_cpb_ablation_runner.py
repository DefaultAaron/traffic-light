"""End-to-end smoke for §三 copy-paste + class-balance ablation runner b-stage.

Synthesizes configs/copy_paste_balance.yaml, configs/data_R2_class_weights.yaml,
and 5 per-arm eval JSONs (no_aug, cp_only, cp_balanced × {0.99, 0.999, 0.9999}).
Drives the runner via subprocess and verifies the output
``runs/_copy_paste_decision.json`` headline matches expectation.

Run: ``uv run python scripts/_smoke_cpb_ablation_runner.py``
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


NUM_CLASSES = 10
CLASS_NAMES = [
    "red", "yellow", "green", "redLeft", "greenLeft",
    "redRight", "greenRight", "forwardRed", "forwardGreen", "armOn",
]
# Baseline supports (rare = support < 30): class IDs 7, 8, 9 are rare;
# class 8 (forwardGreen) is in the safety set.
BASE_SUPPORTS = [200, 150, 180, 80, 60, 40, 35, 25, 20, 10]
BASE_AP = [0.80, 0.70, 0.82, 0.75, 0.72, 0.60, 0.62, 0.40, 0.45, 0.30]
BASE_TOTAL_MAP = 0.65
BASE_RARE_FP = 50
SAFETY_CLASS_IDS = [0, 1, 2, 8]
RARE_THRESHOLD = 30
TOLERANCE = 0.2
DATA_YAML_SHA = "d" * 64
EVAL_MANIFEST_SHA = "e" * 64
FP_MANIFEST_SHA = "f" * 64


def _write_config(path: pathlib.Path) -> None:
    yaml_text = f"""num_classes: {NUM_CLASSES}
class_names:
{chr(10).join(f"  - {n}" for n in CLASS_NAMES)}
rare_class_threshold: {RARE_THRESHOLD}
safety_class_ids: {SAFETY_CLASS_IDS}
copy_paste:
  probability: 0.5
  y_center_max_frac: 0.6
  paste_source_class_ids: [7, 8, 9]
  min_per_batch_K: 0
  required_fliplr: 0.0
  required_mosaic_lock: false
class_balance:
  beta: 0.999
  apply_mode: ultralytics_class_weights
  max_weight_ratio: 50.0
  class_counts_path: null
map_regression_tolerance_pp: {TOLERANCE}
"""
    path.write_text(yaml_text)


def _write_weights(
    path: pathlib.Path, *, override_class_names: list | None = None
) -> None:
    names = override_class_names if override_class_names is not None else CLASS_NAMES
    yaml_text = f"""schema_version: "1"
data_yaml_sha256: "{DATA_YAML_SHA}"
train_split: train
num_classes: {NUM_CLASSES}
class_names:
{chr(10).join(f"  - {n}" for n in names)}
counts: {BASE_SUPPORTS}
rare_class_threshold: {RARE_THRESHOLD}
rare_class_ids: [7, 8, 9]
"""
    path.write_text(yaml_text)


def _write_eval(
    path: pathlib.Path,
    *,
    aps: list,
    total_map: float,
    rare_fp: int,
    map_no_regression: bool,
) -> None:
    payload = {
        "per_class_AP": [
            {"class_id": i, "class_name": CLASS_NAMES[i],
             "ap_at_0_5": aps[i], "full_val_support": BASE_SUPPORTS[i]}
            for i in range(NUM_CLASSES)
        ],
        "total_mAP_at_0_5": total_map,
        "rare_fp_count": rare_fp,
        "data_yaml_sha256": DATA_YAML_SHA,
        "eval_manifest_sha256": EVAL_MANIFEST_SHA,
        "fp_manifest_sha256": FP_MANIFEST_SHA,
        "map_no_regression": map_no_regression,
        "map_regression_tolerance_pp": TOLERANCE,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _ap_with_rare_boost(boost_pp: float) -> list:
    """Apply +boost_pp on rare classes 7,8,9; otherwise unchanged."""
    out = list(BASE_AP)
    for cid in (7, 8, 9):
        out[cid] = min(1.0, out[cid] + boost_pp / 100.0)
    return out


def _run_case(
    *,
    label: str,
    expected_decision: str,
    expected_exit: int,
    anchor_arm: str,
    anchor_beta: float | None,
    cp_only_boost: float,
    cp_only_map_delta: float,
    cp_only_fp_delta_frac: float,
    cp_only_safety_delta_pp: float,
    cp_balanced_boost: dict,
    cp_balanced_map_delta: float,
    cp_balanced_fp_delta_frac: float,
    cp_balanced_safety_delta_pp: float,
    expected_anchor_beta: float | None = None,
    cp_balanced_force_map_no_regression: bool | None = None,
    expected_stderr_substr: str | None = None,
) -> None:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"cpb_ablation_{label}_"))
    try:
        _write_config(tmp / "config.yaml")
        _write_weights(tmp / "weights.yaml")
        # Baseline eval.
        _write_eval(
            tmp / "no_aug.json",
            aps=list(BASE_AP), total_map=BASE_TOTAL_MAP,
            rare_fp=BASE_RARE_FP, map_no_regression=True,
        )
        # cp_only.
        cp_only_aps = _ap_with_rare_boost(cp_only_boost)
        cp_only_aps[8] = max(0.0, BASE_AP[8] + cp_only_safety_delta_pp / 100.0)
        _write_eval(
            tmp / "cp_only.json",
            aps=cp_only_aps,
            total_map=BASE_TOTAL_MAP + cp_only_map_delta / 100.0,
            rare_fp=int(round(BASE_RARE_FP * (1.0 + cp_only_fp_delta_frac))),
            map_no_regression=(cp_only_map_delta >= -TOLERANCE),
        )
        # cp_balanced × 3 β.
        cp_balanced_files: dict[float, pathlib.Path] = {}
        for beta in (0.99, 0.999, 0.9999):
            boost = cp_balanced_boost[beta]
            aps = _ap_with_rare_boost(boost)
            aps[8] = max(0.0, BASE_AP[8] + cp_balanced_safety_delta_pp / 100.0)
            p = tmp / f"cp_balanced_{beta}.json"
            _write_eval(
                p, aps=aps,
                total_map=BASE_TOTAL_MAP + cp_balanced_map_delta / 100.0,
                rare_fp=int(round(
                    BASE_RARE_FP * (1.0 + cp_balanced_fp_delta_frac)
                )),
                map_no_regression=(
                    cp_balanced_force_map_no_regression
                    if cp_balanced_force_map_no_regression is not None
                    else (cp_balanced_map_delta >= -TOLERANCE)
                ),
            )
            cp_balanced_files[beta] = p
        out = tmp / "decision.json"

        cmd = [
            "uv", "run", "python", "-m",
            "components.copy_paste_balance.runners.ablation",
            "--no-aug-eval", str(tmp / "no_aug.json"),
            "--cp-only-eval", str(tmp / "cp_only.json"),
            "--config", str(tmp / "config.yaml"),
            "--weights", str(tmp / "weights.yaml"),
            "--output", str(out),
            "--anchor-arm", anchor_arm,
        ]
        if anchor_beta is not None:
            cmd += ["--anchor-beta", str(anchor_beta)]
        for beta in (0.99, 0.999, 0.9999):
            cmd += ["--cp-balanced-eval", str(cp_balanced_files[beta]),
                    "--cp-balanced-beta", str(beta)]

        proc = subprocess.run(
            cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True
        )
        if proc.returncode != expected_exit:
            print(f"FAIL {label}: expected exit {expected_exit}, "
                  f"got {proc.returncode}")
            print(f"  stdout: {proc.stdout!r}")
            print(f"  stderr: {proc.stderr!r}")
            raise SystemExit(1)
        if expected_exit == 2:
            if expected_stderr_substr is not None and expected_stderr_substr not in proc.stderr:
                print(f"FAIL {label}: stderr should contain "
                      f"{expected_stderr_substr!r}; got: {proc.stderr!r}")
                raise SystemExit(1)
            print(f"  CPB {label:32s} OK -> exit 2 | stderr: "
                  f"{proc.stderr.strip()[:100]}")
            return
        with out.open("r", encoding="utf-8") as fh:
            artifact = json.load(fh)
        if artifact["headline_decision"] != expected_decision:
            print(f"FAIL {label}: expected {expected_decision}, "
                  f"got {artifact['headline_decision']!r}")
            print(f"  notes: {artifact['notes']!r}")
            print(f"  headline_metrics.rare_mean: "
                  f"{artifact['headline_metrics']['rare_class_mean_delta_AP_pp']}")
            raise SystemExit(1)
        # B2 review MAJOR-1/MAJOR-5 verification: when anchor_arm=cp_only,
        # deploy_anchor_beta should point at the best-decision sweep row
        # (deploy > defer > drop > executor_error), not arbitrary first.
        if expected_anchor_beta is not None:
            got_anchor = artifact["cp_balanced"]["deploy_anchor_beta"]
            if got_anchor != expected_anchor_beta:
                print(f"FAIL {label}: expected deploy_anchor_beta="
                      f"{expected_anchor_beta}, got {got_anchor}")
                raise SystemExit(1)
        print(f"  CPB {label:32s} OK -> "
              f"{artifact['headline_decision']:15s} | sweep: "
              f"{','.join(r['decision'] for r in artifact['cp_balanced']['sensitivity_sweep'])}"
              + (f" | anchor_beta={artifact['cp_balanced']['deploy_anchor_beta']}"
                 if expected_anchor_beta is not None else ""))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_path_collision_canonical_cases() -> None:
    """Verify the distinct-path check is canonical, not lexical.

    Codex stop-gate fix: ``Path("./a")`` and ``Path("/abs/a")`` compare
    UNEQUAL even when they point at the same file; ``Path("a")`` and a
    symlink ``a_link → a`` also compare unequal. ``Path.resolve()``
    canonicalizes both — the check must use resolved forms. Two
    sub-cases:
      (a) two eval paths colliding via SYMLINK (the symlink and its
          target are lexically different paths to the same inode);
      (b) two eval paths colliding via RELATIVE-vs-ABSOLUTE form
          combined with ``..`` traversal — both lexically distinct
          Paths but resolve to the same target.
    """
    # --- sub-case (a): symlink alias ---------------------------------------
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="cpb_ablation_symlink_"))
    try:
        _write_config(tmp / "config.yaml")
        _write_weights(tmp / "weights.yaml")
        _write_eval(
            tmp / "no_aug.json",
            aps=list(BASE_AP), total_map=BASE_TOTAL_MAP,
            rare_fp=BASE_RARE_FP, map_no_regression=True,
        )
        _write_eval(
            tmp / "cp_only.json",
            aps=_ap_with_rare_boost(6.0),
            total_map=BASE_TOTAL_MAP + 0.005,
            rare_fp=int(round(BASE_RARE_FP * 0.9)),
            map_no_regression=True,
        )
        # Create cp_balanced[0.99] as a SYMLINK to cp_only.json.
        # Lexically Path("cp_balanced_alias.json") != Path("cp_only.json"),
        # but resolve() follows the symlink and detects the collision.
        (tmp / "cp_balanced_alias.json").symlink_to(tmp / "cp_only.json")
        for beta in (0.999, 0.9999):
            _write_eval(
                tmp / f"cp_balanced_{beta}.json",
                aps=_ap_with_rare_boost(6.0),
                total_map=BASE_TOTAL_MAP + 0.005,
                rare_fp=int(round(BASE_RARE_FP * 0.9)),
                map_no_regression=True,
            )
        out = tmp / "decision.json"
        cmd = [
            "uv", "run", "python", "-m",
            "components.copy_paste_balance.runners.ablation",
            "--no-aug-eval", str(tmp / "no_aug.json"),
            "--cp-only-eval", str(tmp / "cp_only.json"),
            "--cp-balanced-eval", str(tmp / "cp_balanced_alias.json"),
            "--cp-balanced-beta", "0.99",
            "--cp-balanced-eval", str(tmp / "cp_balanced_0.999.json"),
            "--cp-balanced-beta", "0.999",
            "--cp-balanced-eval", str(tmp / "cp_balanced_0.9999.json"),
            "--cp-balanced-beta", "0.9999",
            "--config", str(tmp / "config.yaml"),
            "--weights", str(tmp / "weights.yaml"),
            "--output", str(out),
            "--anchor-arm", "cp_balanced",
            "--anchor-beta", "0.999",
        ]
        proc = subprocess.run(cmd, cwd=str(_REPO_ROOT),
                              capture_output=True, text=True)
        if proc.returncode != 2 or "canonicalize" not in proc.stderr:
            print(f"FAIL symlink-alias: expected exit 2 with 'canonicalize' "
                  f"in stderr; got exit {proc.returncode}, "
                  f"stderr={proc.stderr!r}")
            raise SystemExit(1)
        print(f"  CPB {'path symlink alias':32s} OK -> exit 2 | "
              f"caught via Path.resolve()")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --- sub-case (b): relative-vs-absolute drift ---------------------------
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="cpb_ablation_relabs_"))
    try:
        _write_config(tmp / "config.yaml")
        _write_weights(tmp / "weights.yaml")
        _write_eval(
            tmp / "no_aug.json",
            aps=list(BASE_AP), total_map=BASE_TOTAL_MAP,
            rare_fp=BASE_RARE_FP, map_no_regression=True,
        )
        _write_eval(
            tmp / "cp_only.json",
            aps=_ap_with_rare_boost(6.0),
            total_map=BASE_TOTAL_MAP + 0.005,
            rare_fp=int(round(BASE_RARE_FP * 0.9)),
            map_no_regression=True,
        )
        for beta in (0.99, 0.999, 0.9999):
            _write_eval(
                tmp / f"cp_balanced_{beta}.json",
                aps=_ap_with_rare_boost(6.0),
                total_map=BASE_TOTAL_MAP + 0.005,
                rare_fp=int(round(BASE_RARE_FP * 0.9)),
                map_no_regression=True,
            )
        out = tmp / "decision.json"
        # cp_only_eval is the absolute path; cp_balanced[0.99] is the
        # same file but expressed with a redundant `..` segment. Lexically
        # different Paths; resolve() canonicalizes both to the same target.
        abs_cp_only = str((tmp / "cp_only.json").resolve())
        aliased = str(tmp / "cp_balanced_0.99.json" / ".." / "cp_only.json")
        cmd = [
            "uv", "run", "python", "-m",
            "components.copy_paste_balance.runners.ablation",
            "--no-aug-eval", str(tmp / "no_aug.json"),
            "--cp-only-eval", abs_cp_only,
            "--cp-balanced-eval", aliased,
            "--cp-balanced-beta", "0.99",
            "--cp-balanced-eval", str(tmp / "cp_balanced_0.999.json"),
            "--cp-balanced-beta", "0.999",
            "--cp-balanced-eval", str(tmp / "cp_balanced_0.9999.json"),
            "--cp-balanced-beta", "0.9999",
            "--config", str(tmp / "config.yaml"),
            "--weights", str(tmp / "weights.yaml"),
            "--output", str(out),
            "--anchor-arm", "cp_balanced",
            "--anchor-beta", "0.999",
        ]
        proc = subprocess.run(cmd, cwd=str(_REPO_ROOT),
                              capture_output=True, text=True)
        if proc.returncode != 2 or "canonicalize" not in proc.stderr:
            print(f"FAIL rel-vs-abs: expected exit 2 with 'canonicalize' "
                  f"in stderr; got exit {proc.returncode}, "
                  f"stderr={proc.stderr!r}")
            raise SystemExit(1)
        print(f"  CPB {'path rel-vs-abs (..)':32s} OK -> exit 2 | "
              f"caught via Path.resolve()")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_truncated_per_class_ap_case() -> None:
    """C3 iter-4 NEW-MAJOR fix verification: if every arm omits the
    SAME non-empty subset of classes (truncated `per_class_AP`), the
    earlier cross-arm parity check passed silently. The new
    class-set-coverage check rejects per-arm per_class_AP that does
    not cover exactly ``range(cfg.num_classes)`` in order.

    Fixture: drop classes 8 and 9 (rare classes!) from every eval —
    coverage shrinks from 10 to 8 rows. Without the iter-4 fix this
    would silently produce a decision artifact computed on a truncated
    rare population. With the fix → exit 2.
    """
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="cpb_ablation_truncated_"))
    try:
        _write_config(tmp / "config.yaml")
        _write_weights(tmp / "weights.yaml")

        # Override _write_eval to produce only 8 rows (drop class 8, 9).
        def _write_eval_truncated(p: pathlib.Path, *, aps, total_map, rare_fp, map_no_regression):
            payload = {
                "per_class_AP": [
                    {"class_id": i, "class_name": CLASS_NAMES[i],
                     "ap_at_0_5": aps[i],
                     "full_val_support": BASE_SUPPORTS[i]}
                    for i in range(8)  # only 0..7, omitting rare classes 8, 9
                ],
                "total_mAP_at_0_5": total_map,
                "rare_fp_count": rare_fp,
                "data_yaml_sha256": DATA_YAML_SHA,
                "eval_manifest_sha256": EVAL_MANIFEST_SHA,
                "fp_manifest_sha256": FP_MANIFEST_SHA,
                "map_no_regression": map_no_regression,
                "map_regression_tolerance_pp": TOLERANCE,
            }
            with p.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh)

        _write_eval_truncated(
            tmp / "no_aug.json",
            aps=list(BASE_AP), total_map=BASE_TOTAL_MAP,
            rare_fp=BASE_RARE_FP, map_no_regression=True,
        )
        _write_eval_truncated(
            tmp / "cp_only.json",
            aps=_ap_with_rare_boost(6.0), total_map=BASE_TOTAL_MAP + 0.005,
            rare_fp=int(round(BASE_RARE_FP * 0.9)), map_no_regression=True,
        )
        for beta in (0.99, 0.999, 0.9999):
            _write_eval_truncated(
                tmp / f"cp_balanced_{beta}.json",
                aps=_ap_with_rare_boost(6.0),
                total_map=BASE_TOTAL_MAP + 0.005,
                rare_fp=int(round(BASE_RARE_FP * 0.9)),
                map_no_regression=True,
            )
        out = tmp / "decision.json"
        cmd = [
            "uv", "run", "python", "-m",
            "components.copy_paste_balance.runners.ablation",
            "--no-aug-eval", str(tmp / "no_aug.json"),
            "--cp-only-eval", str(tmp / "cp_only.json"),
            "--cp-balanced-eval", str(tmp / "cp_balanced_0.99.json"),
            "--cp-balanced-beta", "0.99",
            "--cp-balanced-eval", str(tmp / "cp_balanced_0.999.json"),
            "--cp-balanced-beta", "0.999",
            "--cp-balanced-eval", str(tmp / "cp_balanced_0.9999.json"),
            "--cp-balanced-beta", "0.9999",
            "--config", str(tmp / "config.yaml"),
            "--weights", str(tmp / "weights.yaml"),
            "--output", str(out),
            "--anchor-arm", "cp_balanced",
            "--anchor-beta", "0.999",
        ]
        proc = subprocess.run(
            cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True
        )
        if proc.returncode != 2:
            print(f"FAIL truncated per_class_AP: expected exit 2, got "
                  f"{proc.returncode}")
            print(f"  stderr: {proc.stderr!r}")
            raise SystemExit(1)
        if "must have exactly 10 rows" not in proc.stderr:
            print(f"FAIL truncated: stderr should mention 'must have exactly "
                  f"10 rows'; got: {proc.stderr!r}")
            raise SystemExit(1)
        print(f"  CPB {'truncated per_class_AP (8/10 cls)':32s} OK -> "
              f"exit 2 | caught truncation to 8 rows")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_weights_class_name_drift_case() -> None:
    """C3 iter-3 MAJOR-1 fix verification: stale weights YAML with
    different class_names than the config must be rejected at Step 0.

    The CPB runner now validates ``weights_raw["class_names"]`` against
    ``cfg.class_names`` directly. A weights file with a swapped class-name
    pair (e.g. class 5 ↔ class 6) but matching num_classes + rare_threshold
    + data_yaml_sha256 hash would previously pass through silently.
    """
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="cpb_ablation_class_drift_"))
    try:
        _write_config(tmp / "config.yaml")
        # Drift: swap class 5 ↔ class 6 names in the weights YAML.
        drifted_names = list(CLASS_NAMES)
        drifted_names[5], drifted_names[6] = drifted_names[6], drifted_names[5]
        _write_weights(tmp / "weights.yaml", override_class_names=drifted_names)
        _write_eval(
            tmp / "no_aug.json",
            aps=list(BASE_AP), total_map=BASE_TOTAL_MAP,
            rare_fp=BASE_RARE_FP, map_no_regression=True,
        )
        _write_eval(
            tmp / "cp_only.json",
            aps=_ap_with_rare_boost(6.0),
            total_map=BASE_TOTAL_MAP + 0.005,
            rare_fp=int(round(BASE_RARE_FP * 0.9)),
            map_no_regression=True,
        )
        for beta in (0.99, 0.999, 0.9999):
            _write_eval(
                tmp / f"cp_balanced_{beta}.json",
                aps=_ap_with_rare_boost(6.0),
                total_map=BASE_TOTAL_MAP + 0.005,
                rare_fp=int(round(BASE_RARE_FP * 0.9)),
                map_no_regression=True,
            )
        out = tmp / "decision.json"
        cmd = [
            "uv", "run", "python", "-m",
            "components.copy_paste_balance.runners.ablation",
            "--no-aug-eval", str(tmp / "no_aug.json"),
            "--cp-only-eval", str(tmp / "cp_only.json"),
            "--cp-balanced-eval", str(tmp / "cp_balanced_0.99.json"),
            "--cp-balanced-beta", "0.99",
            "--cp-balanced-eval", str(tmp / "cp_balanced_0.999.json"),
            "--cp-balanced-beta", "0.999",
            "--cp-balanced-eval", str(tmp / "cp_balanced_0.9999.json"),
            "--cp-balanced-beta", "0.9999",
            "--config", str(tmp / "config.yaml"),
            "--weights", str(tmp / "weights.yaml"),
            "--output", str(out),
            "--anchor-arm", "cp_balanced",
            "--anchor-beta", "0.999",
        ]
        proc = subprocess.run(
            cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True
        )
        if proc.returncode != 2:
            print(f"FAIL class-name drift: expected exit 2, got {proc.returncode}")
            print(f"  stderr: {proc.stderr!r}")
            raise SystemExit(1)
        if "class_names" not in proc.stderr or "divergence at index 5" not in proc.stderr:
            print(f"FAIL class-name drift: stderr should mention 'class_names' "
                  f"+ 'divergence at index 5'; got: {proc.stderr!r}")
            raise SystemExit(1)
        print(f"  CPB {'weights class_names drift':32s} OK -> exit 2 | "
              f"caught swapped names at index 5")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_path_collision_case() -> None:
    """Verify AblationConfig rejects duplicate path fields.

    Codex stop-gate fix: a CPB AblationConfig that pairs the same path
    for two distinct fields (e.g. cp_only_eval == cp_balanced_eval[0],
    or output_json == any input) would silently collapse two arms onto
    one eval source or clobber an input mid-write. The post_init
    distinct-path check (mirroring HN B2 review I4) rejects this at
    construction. Smoke exercises the runner CLI with a duplicate path
    and verifies exit code 2 (Step 0 hard fail propagated via main()).
    """
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="cpb_ablation_path_collide_"))
    try:
        _write_config(tmp / "config.yaml")
        _write_weights(tmp / "weights.yaml")
        _write_eval(
            tmp / "no_aug.json",
            aps=list(BASE_AP), total_map=BASE_TOTAL_MAP,
            rare_fp=BASE_RARE_FP, map_no_regression=True,
        )
        _write_eval(
            tmp / "cp_only.json",
            aps=_ap_with_rare_boost(6.0),
            total_map=BASE_TOTAL_MAP + 0.005,
            rare_fp=int(round(BASE_RARE_FP * 0.9)),
            map_no_regression=True,
        )
        for beta in (0.99, 0.999, 0.9999):
            _write_eval(
                tmp / f"cp_balanced_{beta}.json",
                aps=_ap_with_rare_boost(6.0),
                total_map=BASE_TOTAL_MAP + 0.005,
                rare_fp=int(round(BASE_RARE_FP * 0.9)),
                map_no_regression=True,
            )
        out = tmp / "decision.json"

        # Pass cp_only_eval and cp_balanced_eval[0.99] as the SAME file —
        # the post_init distinct-path check should reject this.
        cmd = [
            "uv", "run", "python", "-m",
            "components.copy_paste_balance.runners.ablation",
            "--no-aug-eval", str(tmp / "no_aug.json"),
            "--cp-only-eval", str(tmp / "cp_only.json"),
            # COLLIDING: cp_balanced[0.99] is the same path as cp_only.
            "--cp-balanced-eval", str(tmp / "cp_only.json"),
            "--cp-balanced-beta", "0.99",
            "--cp-balanced-eval", str(tmp / "cp_balanced_0.999.json"),
            "--cp-balanced-beta", "0.999",
            "--cp-balanced-eval", str(tmp / "cp_balanced_0.9999.json"),
            "--cp-balanced-beta", "0.9999",
            "--config", str(tmp / "config.yaml"),
            "--weights", str(tmp / "weights.yaml"),
            "--output", str(out),
            "--anchor-arm", "cp_balanced",
            "--anchor-beta", "0.999",
        ]
        proc = subprocess.run(
            cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True
        )
        if proc.returncode != 2:
            print(f"FAIL path-collision: expected exit 2, got "
                  f"{proc.returncode}")
            print(f"  stdout: {proc.stdout!r}")
            print(f"  stderr: {proc.stderr!r}")
            raise SystemExit(1)
        if "distinct" not in proc.stderr:
            print(f"FAIL path-collision: stderr should mention 'distinct'; "
                  f"got: {proc.stderr!r}")
            raise SystemExit(1)
        print(f"  CPB {'path collision (cp_only==cp_balanced[0])':32s} "
              f"OK -> exit 2 | stderr: {proc.stderr.strip()[:100]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("§三 copy-paste + class-balance runner end-to-end smoke:")
    # Deploy on cp_balanced[β=0.999]: rare +6pp avg, safety -0.5pp, fp -10%, mAP +0.5pp.
    # cp_only is also deployable in this fixture for simplicity.
    _run_case(
        label="deploy (cp_balanced 0.999)",
        expected_decision="deploy", expected_exit=0,
        anchor_arm="cp_balanced", anchor_beta=0.999,
        cp_only_boost=6.0, cp_only_map_delta=0.5,
        cp_only_fp_delta_frac=-0.10, cp_only_safety_delta_pp=-0.5,
        cp_balanced_boost={0.99: 4.5, 0.999: 6.0, 0.9999: 7.0},
        cp_balanced_map_delta=0.5,
        cp_balanced_fp_delta_frac=-0.10,
        cp_balanced_safety_delta_pp=-0.5,
    )
    # Headline cp_only — verify deploy_anchor_beta picks best-decision
    # sweep row. cp_balanced sweep cells get rare_avg=1.5pp → all defer,
    # so all 3 sweep rows have decision=defer. Best-decision picker
    # falls back to lowest beta tie-break → 0.99.
    _run_case(
        label="deploy (cp_only) anchor=best",
        expected_decision="deploy", expected_exit=0,
        anchor_arm="cp_only", anchor_beta=None,
        cp_only_boost=6.0, cp_only_map_delta=0.5,
        cp_only_fp_delta_frac=-0.10, cp_only_safety_delta_pp=-0.5,
        cp_balanced_boost={0.99: 1.5, 0.999: 1.5, 0.9999: 1.5},
        cp_balanced_map_delta=0.0,
        cp_balanced_fp_delta_frac=0.0,
        cp_balanced_safety_delta_pp=-0.3,
        expected_anchor_beta=0.99,
    )
    # B2 iter-2 MINOR-1 fix: cp_only headline with TRULY MIXED sweep
    # decisions (drop, defer, deploy) so the best-decision picker is
    # exercised against the full ordering, not just deploy-vs-defer.
    # Picker MUST pick 0.9999 (deploy). Without the MAJOR-1 fix the
    # runner picked first_beta=0.99 (a drop cell as the canonical
    # anchor — semantically meaningless).
    #
    # Boost math with cp_balanced_safety_delta_pp=+3.0 (no regression on
    # rare-safety class 8; class 8 gets +3pp regardless of boost):
    #   0.99 boost=3.0: classes 7+9 +3pp, class 8 +3pp → rare_avg=3pp
    #                   (≥2 → defer fails; max=3 → deploy fails) → middle-drop
    #   0.999 boost=1.0: classes 7+9 +1pp, class 8 +3pp → rare_avg≈1.67pp
    #                    (<2pp → defer; map OK) → defer
    #   0.9999 boost=6.0: classes 7+9 +6pp, class 8 +3pp → rare_avg=5,
    #                     max=6 → deploy
    _run_case(
        label="deploy (cp_only) anchor=best(mixed)",
        expected_decision="deploy", expected_exit=0,
        anchor_arm="cp_only", anchor_beta=None,
        cp_only_boost=6.0, cp_only_map_delta=0.5,
        cp_only_fp_delta_frac=-0.10, cp_only_safety_delta_pp=-0.5,
        cp_balanced_boost={0.99: 3.0, 0.999: 1.0, 0.9999: 6.0},
        cp_balanced_map_delta=0.5,
        cp_balanced_fp_delta_frac=-0.10,
        cp_balanced_safety_delta_pp=3.0,
        expected_anchor_beta=0.9999,
    )
    # Defer on cp_balanced[β=0.99]: rare +1.5pp, mAP OK, safety -0.5pp.
    _run_case(
        label="defer (cp_balanced 0.99)",
        expected_decision="defer", expected_exit=0,
        anchor_arm="cp_balanced", anchor_beta=0.99,
        cp_only_boost=0.5, cp_only_map_delta=0.0,
        cp_only_fp_delta_frac=0.0, cp_only_safety_delta_pp=-0.3,
        cp_balanced_boost={0.99: 1.5, 0.999: 1.5, 0.9999: 1.5},
        cp_balanced_map_delta=0.0,
        cp_balanced_fp_delta_frac=0.0,
        cp_balanced_safety_delta_pp=-0.5,
    )
    # Drop literal: rare improvement OK but mAP regress > 0.5pp.
    _run_case(
        label="drop literal (mAP regress)",
        expected_decision="drop", expected_exit=0,
        anchor_arm="cp_balanced", anchor_beta=0.999,
        cp_only_boost=6.0, cp_only_map_delta=-0.6,
        cp_only_fp_delta_frac=-0.10, cp_only_safety_delta_pp=-0.5,
        cp_balanced_boost={0.99: 6.0, 0.999: 6.0, 0.9999: 6.0},
        cp_balanced_map_delta=-0.6,
        cp_balanced_fp_delta_frac=-0.10,
        cp_balanced_safety_delta_pp=-0.5,
    )
    _run_path_collision_case()
    _run_path_collision_canonical_cases()
    _run_weights_class_name_drift_case()
    _run_truncated_per_class_ap_case()
    # C3 iter-2 MAJOR-1 fix: when all cp_balanced sweep rows are
    # executor_error AND headline=cp_only, the runner must refuse to
    # write the artifact rather than silently stamp an executor_error
    # cell as the canonical anchor. Force verdict ↔ delta inconsistency
    # on all 3 sweep cells by stamping map_no_regression=True while
    # delta=-0.5pp (well below the 0.2pp tolerance) → gate fires
    # EXECUTOR_ERROR for all 3 → runner raises ValueError → exit 2.
    _run_case(
        label="all sweep executor_error → refuse",
        expected_decision="UNUSED",  # exit 2 short-circuits decision check
        expected_exit=2,
        anchor_arm="cp_only", anchor_beta=None,
        cp_only_boost=6.0, cp_only_map_delta=0.5,
        cp_only_fp_delta_frac=-0.10, cp_only_safety_delta_pp=-0.5,
        cp_balanced_boost={0.99: 6.0, 0.999: 6.0, 0.9999: 6.0},
        cp_balanced_map_delta=-0.5,  # would normally produce map_no_regression=False
        cp_balanced_fp_delta_frac=-0.10,
        cp_balanced_safety_delta_pp=-0.5,
        cp_balanced_force_map_no_regression=True,  # FORCE inconsistency
        expected_stderr_substr="all cp_balanced sweep rows are executor_error",
    )
    # Drop middle-case: rare +3pp avg (uniform across all 3 rare classes
    # including safety class 8) — improvement insufficient for deploy
    # (need ≥ 5pp), mAP OK, no literal drop trigger. Cascade lands on
    # middle-case drop. Safety_delta=+3pp matches the boost so the rare
    # mean stays at 3pp (overriding class 8 to a smaller value would
    # pull mean below 2pp and silently switch the case to defer).
    _run_case(
        label="drop middle-case (rare=3pp)",
        expected_decision="drop", expected_exit=0,
        anchor_arm="cp_balanced", anchor_beta=0.9999,
        cp_only_boost=3.0, cp_only_map_delta=0.0,
        cp_only_fp_delta_frac=0.0, cp_only_safety_delta_pp=3.0,
        cp_balanced_boost={0.99: 3.0, 0.999: 3.0, 0.9999: 3.0},
        cp_balanced_map_delta=0.0,
        cp_balanced_fp_delta_frac=0.0,
        cp_balanced_safety_delta_pp=3.0,
    )
    print("\nALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
