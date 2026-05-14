"""End-to-end smoke for §四 hard-negative-mining ablation runner b-stage.

Synthesizes a frozen FP manifest (with canonical-SHA256 stamped), two
per-arm eval JSONs (no_hn baseline + with_hn candidate), and a
configs/hard_negative_mining.yaml; drives
``components.hard_negative_mining.runners.ablation.main`` via subprocess
and verifies the output ``runs/_hard_negative_decision.json`` matches
the expected case. Covers deploy / defer / drop / executor_error
outcomes.

Run: ``uv run python scripts/_smoke_hn_ablation_runner.py``
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _canonical_manifest_sha(payload: dict) -> str:
    """Reproduce the loader's manifest_sha256 protocol exactly."""
    canonical = {
        "confidence_threshold": payload["confidence_threshold"],
        "nms_iou_threshold": payload["nms_iou_threshold"],
        "entries": payload["entries"],
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _write_manifest(path: pathlib.Path) -> str:
    entries = [
        {
            "image_sha256": f"{i:064x}",
            "image_relpath": f"frames/{i:04d}.jpg",
            "label_source": "demo8" if i % 4 == 0 else (
                "demo11" if i % 4 == 1 else (
                    "demo13" if i % 4 == 2 else "real_light_set"
                )
            ),
            "has_real_light": i % 4 == 3,  # 1/4 of entries are real-light
        }
        for i in range(20)
    ]
    payload = {
        "confidence_threshold": 0.25,
        "nms_iou_threshold": 0.5,
        "entries": entries,
    }
    manifest_sha = _canonical_manifest_sha(payload)
    payload["manifest_sha256"] = manifest_sha
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    return manifest_sha


def _write_config(
    path: pathlib.Path, *, data_yaml_sha: str, tolerance: float
) -> None:
    yaml_text = f"""schema_version: "1"
num_classes: 10
class_names:
  - red
  - yellow
  - green
  - redLeft
  - greenLeft
  - redRight
  - greenRight
  - forwardRed
  - forwardGreen
  - armOn
data_yaml_sha256: "{data_yaml_sha}"
baseline_weights_path: null
mining_sources:
  - demo8
  - demo11
  - demo13
  - r2_self
output_candidates_path: null
min_sample_fraction: 0.10
max_true_positive_missed_rate: 0.10
frozen_manifest_path: null
map_regression_tolerance_pp: {tolerance}
"""
    path.write_text(yaml_text)


def _write_eval(
    path: pathlib.Path,
    *,
    fp_count: int,
    recall: float,
    total_map: float,
    data_yaml_sha: str,
    eval_manifest_sha: str,
    fp_manifest_sha: str,
    tolerance: float,
    map_no_regression: bool,
) -> None:
    payload = {
        "fp_count": fp_count,
        "real_light_recall": recall,
        "total_mAP_at_0_5": total_map,
        "data_yaml_sha256": data_yaml_sha,
        "eval_manifest_sha256": eval_manifest_sha,
        "fp_manifest_sha256": fp_manifest_sha,
        "map_no_regression": map_no_regression,
        "map_regression_tolerance_pp": tolerance,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _run_case(
    *,
    label: str,
    expected_decision: str,
    expected_exit: int,
    fp_count_with_hn: int,
    recall_with_hn: float,
    total_map_with_hn: float,
    map_no_regression_with_hn: bool,
    tolerance: float = 0.2,
) -> None:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix=f"hn_ablation_{label}_"))
    try:
        data_yaml_sha = "d" * 64
        eval_manifest_sha = "e" * 64
        fp_manifest_sha = _write_manifest(tmp / "manifest.json")
        _write_config(tmp / "config.yaml",
                      data_yaml_sha=data_yaml_sha, tolerance=tolerance)
        baseline_recall = 0.90
        baseline_map = 0.85
        baseline_fp = 100
        _write_eval(
            tmp / "no_hn.json",
            fp_count=baseline_fp, recall=baseline_recall,
            total_map=baseline_map,
            data_yaml_sha=data_yaml_sha,
            eval_manifest_sha=eval_manifest_sha,
            fp_manifest_sha=fp_manifest_sha,
            tolerance=tolerance, map_no_regression=True,
        )
        _write_eval(
            tmp / "with_hn.json",
            fp_count=fp_count_with_hn, recall=recall_with_hn,
            total_map=total_map_with_hn,
            data_yaml_sha=data_yaml_sha,
            eval_manifest_sha=eval_manifest_sha,
            fp_manifest_sha=fp_manifest_sha,
            tolerance=tolerance,
            map_no_regression=map_no_regression_with_hn,
        )
        out = tmp / "decision.json"

        cmd = [
            "uv", "run", "python", "-m",
            "components.hard_negative_mining.runners.ablation",
            "--no-hn-eval", str(tmp / "no_hn.json"),
            "--with-hn-eval", str(tmp / "with_hn.json"),
            "--config", str(tmp / "config.yaml"),
            "--frozen-manifest", str(tmp / "manifest.json"),
            "--output", str(out),
            "--anchor-arm", "with_hn",
        ]
        proc = subprocess.run(cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True)
        if proc.returncode != expected_exit:
            print(f"FAIL {label}: expected exit {expected_exit}, got "
                  f"{proc.returncode}")
            print(f"  stdout: {proc.stdout!r}")
            print(f"  stderr: {proc.stderr!r}")
            raise SystemExit(1)
        if expected_exit == 2:
            # Step 0 hard fail — no output written, expected_decision unused.
            print(f"  HN {label:30s} OK -> exit 2 (Step 0 fail) | "
                  f"stderr: {proc.stderr.strip()[:80]}")
            return
        with out.open("r", encoding="utf-8") as fh:
            artifact = json.load(fh)
        if artifact["headline_decision"] != expected_decision:
            print(f"FAIL {label}: expected decision {expected_decision}, "
                  f"got {artifact['headline_decision']!r}")
            print(f"  notes: {artifact['notes']!r}")
            raise SystemExit(1)
        print(f"  HN {label:30s} OK -> {artifact['headline_decision']:15s} | "
              f"with_hn.notes: {artifact['with_hn']['notes'][:80]}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    print("§四 hard-negative-mining runner end-to-end smoke:")
    # Deploy: fp 100→40 (60% drop), recall 0.90→0.896 (-0.4pp), mAP 0.85→0.849 (-0.1pp tol-ok).
    _run_case(
        label="deploy",
        expected_decision="deploy", expected_exit=0,
        fp_count_with_hn=40, recall_with_hn=0.896,
        total_map_with_hn=0.849, map_no_regression_with_hn=True,
    )
    # Defer: fp 100→70 (30% drop), recall 0.90→0.898 (-0.2pp), mAP regress -0.4pp.
    # mAP-AGNOSTIC defer per §4.7 — still defers even when mAP regressed.
    _run_case(
        label="defer (mAP-agnostic)",
        expected_decision="defer", expected_exit=0,
        fp_count_with_hn=70, recall_with_hn=0.898,
        total_map_with_hn=0.846, map_no_regression_with_hn=False,
    )
    # Drop literal: recall regressed past -0.5pp.
    _run_case(
        label="drop literal (recall)",
        expected_decision="drop", expected_exit=0,
        fp_count_with_hn=40, recall_with_hn=0.892,
        total_map_with_hn=0.849, map_no_regression_with_hn=True,
    )
    # Drop literal: fp_drop below 0.20.
    _run_case(
        label="drop literal (fp_drop)",
        expected_decision="drop", expected_exit=0,
        fp_count_with_hn=90, recall_with_hn=0.898,
        total_map_with_hn=0.849, map_no_regression_with_hn=True,
    )
    # Drop catch-all: fp ≥ 0.50, recall OK, mAP regress.
    _run_case(
        label="drop catch-all",
        expected_decision="drop", expected_exit=0,
        fp_count_with_hn=40, recall_with_hn=0.898,
        total_map_with_hn=0.844, map_no_regression_with_hn=False,
    )
    # Step 0 hard fail: with_hn carries an inconsistent verdict — upstream
    # stamped map_no_regression=True but tolerance vs delta says False.
    # Detected by the gate's verdict cross-check as EXECUTOR_ERROR (exit 1
    # — the artifact IS written for executor_error since it's a runtime
    # decision-rule outcome, not a Step 0 cross-artifact fail).
    _run_case(
        label="executor_error (bad verdict)",
        expected_decision="executor_error", expected_exit=1,
        fp_count_with_hn=40, recall_with_hn=0.898,
        total_map_with_hn=0.844, map_no_regression_with_hn=True,
    )
    print("\nALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
