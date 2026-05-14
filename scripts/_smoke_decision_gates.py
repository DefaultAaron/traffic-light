"""Smoke test for §三 + §四 decision_gate.apply_decision_rule b-stage drop.

Synthetic ArmMetrics → DecisionInputs → apply_decision_rule, exercising
deploy / defer / drop / executor_error paths on both gates. Confirms:
  * Each path returns the expected decision enum value.
  * notes carries the triggering condition (non-empty + recognisable).
  * Cross-validation invariants (ArmMetrics + DecisionInputs construction)
    accept legitimate inputs; rule body doesn't bypass them.

Drives the runner-equivalent surface area without the runner's I/O.
Run: ``uv run python scripts/_smoke_decision_gates.py``.
"""

from __future__ import annotations

import math
import pathlib
import sys

# Self-contained launch: when invoked as `python scripts/_smoke_decision_gates.py`,
# sys.path[0] is the script's directory (scripts/), not the project root, so
# `from components...` fails. Insert the repo root explicitly so the script is
# runnable without a PYTHONPATH=. prefix or `-m` form.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from components.copy_paste_balance.gates import ablation_gate as cpb_ag
from components.copy_paste_balance.gates import decision_gate as cpb_dg
from components.hard_negative_mining.gates import ablation_gate as hn_ag
from components.hard_negative_mining.gates import decision_gate as hn_dg


HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64


def _cpb_baseline() -> cpb_ag.ArmMetrics:
    return cpb_ag.ArmMetrics(
        arm_id="no_aug",
        is_baseline_reference=True,
        rare_class_mean_delta_AP_pp=0.0,
        rare_class_max_delta_AP_pp=0.0,
        rare_safety_min_delta_AP_pp=0.0,
        rare_related_fp_delta_frac=0.0,
        total_map_delta_pp=0.0,
        rare_class_ids=(5, 6, 7),
        rare_safety_class_ids=(5, 6),
        zero_support_rare_classes=(),
        eval_manifest_sha256=HEX_A,
        fp_manifest_sha256=HEX_B,
        map_no_regression=True,
        map_regression_tolerance_pp=0.2,
        data_yaml_sha256=HEX_C,
    )


def _cpb_candidate(
    *,
    arm_id: str,
    rare_avg: float,
    rare_max: float,
    rare_safety_min: float,
    rare_fp: float,
    total_map: float,
    map_no_regr: bool,
) -> cpb_ag.ArmMetrics:
    return cpb_ag.ArmMetrics(
        arm_id=arm_id,
        is_baseline_reference=False,
        rare_class_mean_delta_AP_pp=rare_avg,
        rare_class_max_delta_AP_pp=rare_max,
        rare_safety_min_delta_AP_pp=rare_safety_min,
        rare_related_fp_delta_frac=rare_fp,
        total_map_delta_pp=total_map,
        rare_class_ids=(5, 6, 7),
        rare_safety_class_ids=(5, 6),
        zero_support_rare_classes=(),
        eval_manifest_sha256=HEX_A,
        fp_manifest_sha256=HEX_B,
        map_no_regression=map_no_regr,
        map_regression_tolerance_pp=0.2,
        data_yaml_sha256=HEX_C,
    )


def _cpb_run(
    *,
    case: str,
    arm_id: str,
    beta: float | None,
    expect: cpb_dg.CopyPasteDecision,
    rare_avg: float,
    rare_max: float,
    rare_safety_min: float,
    rare_fp: float,
    total_map: float,
    map_no_regr: bool,
) -> None:
    cand = _cpb_candidate(
        arm_id=arm_id,
        rare_avg=rare_avg,
        rare_max=rare_max,
        rare_safety_min=rare_safety_min,
        rare_fp=rare_fp,
        total_map=total_map,
        map_no_regr=map_no_regr,
    )
    inputs = cpb_dg.DecisionInputs(
        baseline=_cpb_baseline(),
        candidate=cand,
        map_no_regression=map_no_regr,
        map_regression_tolerance_pp=0.2,
    )
    result = cpb_dg.apply_decision_rule(inputs, beta=beta)
    assert result.decision == expect, (
        f"CPB {case}: expected {expect.value}, got {result.decision.value} | "
        f"notes={result.notes!r}"
    )
    assert result.notes, f"CPB {case}: notes must be non-empty"
    assert result.arm_id.value == arm_id, (
        f"CPB {case}: arm_id mismatch — expected {arm_id}, "
        f"got {result.arm_id.value}"
    )
    assert result.beta == beta, (
        f"CPB {case}: beta mismatch — expected {beta!r}, got {result.beta!r}"
    )
    print(f"  CPB {case:34s} OK -> {result.decision.value:15s} | {result.notes[:100]}")


def smoke_cpb() -> None:
    print("§三 copy_paste_balance.apply_decision_rule:")
    # Deploy: rare avg ≥ 5pp, rare safety ≥ -1pp, rare_fp ≤ 0.10, mAP no-regr.
    _cpb_run(
        case="deploy (avg-path, cp_only)",
        arm_id="cp_only", beta=None,
        expect=cpb_dg.CopyPasteDecision.DEPLOY,
        rare_avg=6.0, rare_max=6.0,
        rare_safety_min=-0.5, rare_fp=0.05, total_map=-0.1,
        map_no_regr=True,
    )
    # Deploy: rare max ≥ 5pp via single-class alternative (cp_balanced).
    _cpb_run(
        case="deploy (max-path, cp_balanced 0.999)",
        arm_id="cp_balanced", beta=0.999,
        expect=cpb_dg.CopyPasteDecision.DEPLOY,
        rare_avg=1.5, rare_max=5.5,
        rare_safety_min=-0.8, rare_fp=0.08, total_map=0.0,
        map_no_regr=True,
    )
    # Deploy boundary: rare avg = 5pp exactly (non-strict ≥).
    _cpb_run(
        case="deploy boundary (rare_avg=5pp)",
        arm_id="cp_balanced", beta=0.99,
        expect=cpb_dg.CopyPasteDecision.DEPLOY,
        rare_avg=5.0, rare_max=5.0,
        rare_safety_min=-1.0, rare_fp=0.10, total_map=-0.2,
        map_no_regr=True,
    )
    # Defer: rare avg < 2pp, mAP no-regr, no literal drop fires.
    _cpb_run(
        case="defer (rare_avg=1.5pp)",
        arm_id="cp_only", beta=None,
        expect=cpb_dg.CopyPasteDecision.DEFER,
        rare_avg=1.5, rare_max=2.5,
        rare_safety_min=-0.5, rare_fp=0.05, total_map=-0.1,
        map_no_regr=True,
    )
    # Defer boundary: rare avg = 1.999pp.
    _cpb_run(
        case="defer boundary (rare_avg=1.999pp)",
        arm_id="cp_only", beta=None,
        expect=cpb_dg.CopyPasteDecision.DEFER,
        rare_avg=1.999, rare_max=1.999,
        rare_safety_min=-0.9, rare_fp=0.09, total_map=-0.15,
        map_no_regr=True,
    )
    # Drop literal: total_map regress > 0.5pp.
    _cpb_run(
        case="drop literal (mAP -0.6pp)",
        arm_id="cp_balanced", beta=0.9999,
        expect=cpb_dg.CopyPasteDecision.DROP,
        rare_avg=6.0, rare_max=6.0,
        rare_safety_min=-0.5, rare_fp=0.05, total_map=-0.6,
        map_no_regr=False,
    )
    # Drop literal: rare safety regress > 1pp.
    _cpb_run(
        case="drop literal (safety -1.5pp)",
        arm_id="cp_balanced", beta=0.999,
        expect=cpb_dg.CopyPasteDecision.DROP,
        rare_avg=6.0, rare_max=6.0,
        rare_safety_min=-1.5, rare_fp=0.05, total_map=-0.1,
        map_no_regr=True,
    )
    # Drop literal: rare_fp rise > 0.30.
    _cpb_run(
        case="drop literal (rare_fp 0.40)",
        arm_id="cp_only", beta=None,
        expect=cpb_dg.CopyPasteDecision.DROP,
        rare_avg=6.0, rare_max=6.0,
        rare_safety_min=-0.5, rare_fp=0.40, total_map=-0.1,
        map_no_regr=True,
    )
    # Drop middle-case: rare avg in [2, 5), safety -0.5pp (in [-1, -1] OK),
    # rare_fp = 0.05, mAP no-regr → improvement insufficient under deploy
    # guards (avg+max < 5), defer fails (rare_avg ≥ 2), no literal trigger.
    _cpb_run(
        case="drop middle (rare_avg=3pp)",
        arm_id="cp_balanced", beta=0.99,
        expect=cpb_dg.CopyPasteDecision.DROP,
        rare_avg=3.0, rare_max=3.5,
        rare_safety_min=-0.5, rare_fp=0.05, total_map=-0.1,
        map_no_regr=True,
    )
    # Drop middle: rare safety in (-1, -1.0001) — but safety_min = -1pp
    # exactly is deploy-OK; -0.5pp here, so deploy fails on rare improvement
    # alone (avg/max < 5).
    # Drop middle: mAP regression below tolerance but above -0.5pp drop trigger.
    _cpb_run(
        case="drop middle (mAP -0.4pp, no rare improvement)",
        arm_id="cp_only", beta=None,
        expect=cpb_dg.CopyPasteDecision.DROP,
        rare_avg=2.5, rare_max=2.5,
        rare_safety_min=-0.5, rare_fp=0.05, total_map=-0.4,
        map_no_regr=False,
    )
    # Executor error path: NaN — direct ArmMetrics construction would fail
    # at __post_init__, so we bypass via object.__setattr__ on a frozen
    # dataclass to simulate the "future refactor bypassed validation" case.
    cand_ok = _cpb_candidate(
        arm_id="cp_only",
        rare_avg=1.0, rare_max=1.0,
        rare_safety_min=-0.5, rare_fp=0.05, total_map=-0.1,
        map_no_regr=True,
    )
    object.__setattr__(cand_ok, "rare_class_mean_delta_AP_pp", math.nan)
    inputs_nan = cpb_dg.DecisionInputs.__new__(cpb_dg.DecisionInputs)
    object.__setattr__(inputs_nan, "baseline", _cpb_baseline())
    object.__setattr__(inputs_nan, "candidate", cand_ok)
    object.__setattr__(inputs_nan, "map_no_regression", True)
    object.__setattr__(inputs_nan, "map_regression_tolerance_pp", 0.2)
    result = cpb_dg.apply_decision_rule(inputs_nan, beta=None)
    assert result.decision == cpb_dg.CopyPasteDecision.EXECUTOR_ERROR, (
        f"CPB executor_error: expected EXECUTOR_ERROR, got {result.decision.value}"
    )
    assert "NaN/inf" in result.notes, f"CPB executor_error notes: {result.notes!r}"
    print(f"  CPB {'executor_error (NaN bypass)':34s} OK -> "
          f"{result.decision.value:15s} | {result.notes[:100]}")

    # Executor error path: verdict / delta inconsistency. Buggy upstream
    # stamps map_no_regression=True with total_map_delta_pp=-0.5pp,
    # which is below the 0.2pp tolerance. Without the defensive
    # cross-check the deploy guard fires (rare avg 6pp meets deploy
    # criterion + verdict says no-regression); WITH the check it
    # surfaces as EXECUTOR_ERROR instead of shipping as DEPLOY.
    cand_bad_verdict = _cpb_candidate(
        arm_id="cp_only",
        rare_avg=6.0, rare_max=6.0,
        rare_safety_min=-0.5, rare_fp=0.05, total_map=-0.5,
        map_no_regr=True,  # WRONG: -0.5pp regress is below the 0.2pp tolerance
    )
    inputs_bad = cpb_dg.DecisionInputs(
        baseline=_cpb_baseline(),
        candidate=cand_bad_verdict,
        map_no_regression=True,  # Matches candidate (DecisionInputs cross-check passes)
        map_regression_tolerance_pp=0.2,
    )
    result = cpb_dg.apply_decision_rule(inputs_bad, beta=None)
    assert result.decision == cpb_dg.CopyPasteDecision.EXECUTOR_ERROR, (
        f"CPB verdict-inconsistency: expected EXECUTOR_ERROR (deploy guard "
        f"would have fired without the cross-check), got "
        f"{result.decision.value} | notes={result.notes!r}"
    )
    assert "inconsistency" in result.notes
    print(f"  CPB {'executor_error (bad verdict)':34s} OK -> "
          f"{result.decision.value:15s} | {result.notes[:100]}")
    # Sanity-counterpart: inverse direction — upstream stamps False but
    # delta is actually no-regression. Without the cross-check we'd
    # fall to drop (deploy fails on map_no_regr=False); WITH the check
    # it still surfaces as executor_error.
    cand_bad_inverse = _cpb_candidate(
        arm_id="cp_only",
        rare_avg=6.0, rare_max=6.0,
        rare_safety_min=-0.5, rare_fp=0.05, total_map=-0.1,
        map_no_regr=False,  # WRONG: -0.1pp is within 0.2pp tolerance
    )
    inputs_bad_inverse = cpb_dg.DecisionInputs(
        baseline=_cpb_baseline(),
        candidate=cand_bad_inverse,
        map_no_regression=False,
        map_regression_tolerance_pp=0.2,
    )
    result = cpb_dg.apply_decision_rule(inputs_bad_inverse, beta=None)
    assert result.decision == cpb_dg.CopyPasteDecision.EXECUTOR_ERROR, (
        f"CPB verdict-inconsistency inverse: expected EXECUTOR_ERROR, got "
        f"{result.decision.value}"
    )
    print(f"  CPB {'executor_error (bad inverse)':34s} OK -> "
          f"{result.decision.value:15s} | {result.notes[:100]}")

    # Executor error: NaN tolerance. Codex stop-gate caught that the
    # original CPB defensive NaN/inf check iterated only over candidate
    # fields, missing inputs.map_regression_tolerance_pp. A NaN tolerance
    # combined with map_no_regression=False would silently match the
    # verdict cross-check (anything ≥ -NaN is False, == False matches
    # stamped False) and fall through to drop with no NaN warning. With
    # the fix, NaN tolerance surfaces as EXECUTOR_ERROR at the defensive
    # check.
    cand_for_nan_tol = _cpb_candidate(
        arm_id="cp_only",
        rare_avg=1.0, rare_max=1.0,
        rare_safety_min=-0.5, rare_fp=0.05, total_map=-0.1,
        map_no_regr=False,  # paired with NaN tolerance so the consistency
                            # check would also "match" silently without the fix
    )
    inputs_nan_tol = cpb_dg.DecisionInputs.__new__(cpb_dg.DecisionInputs)
    object.__setattr__(inputs_nan_tol, "baseline", _cpb_baseline())
    object.__setattr__(inputs_nan_tol, "candidate", cand_for_nan_tol)
    object.__setattr__(inputs_nan_tol, "map_no_regression", False)
    object.__setattr__(inputs_nan_tol, "map_regression_tolerance_pp", math.nan)
    result = cpb_dg.apply_decision_rule(inputs_nan_tol, beta=None)
    assert result.decision == cpb_dg.CopyPasteDecision.EXECUTOR_ERROR, (
        f"CPB NaN tolerance: expected EXECUTOR_ERROR (would have silently "
        f"dropped without the fix), got {result.decision.value} | "
        f"notes={result.notes!r}"
    )
    assert "map_regression_tolerance_pp" in result.notes, (
        f"CPB NaN tolerance notes must name the offending field; got "
        f"{result.notes!r}"
    )
    print(f"  CPB {'executor_error (NaN tolerance)':34s} OK -> "
          f"{result.decision.value:15s} | {result.notes[:100]}")


def _hn_baseline() -> hn_ag.ArmMetrics:
    return hn_ag.ArmMetrics(
        arm_id="no_hn",
        is_baseline_reference=True,
        fp_drop_frac=0.0,
        real_light_recall_delta_pp=0.0,
        total_map_delta_pp=0.0,
        map_no_regression=True,
        map_regression_tolerance_pp=0.2,
        eval_manifest_sha256=HEX_A,
        fp_manifest_sha256=HEX_B,
        data_yaml_sha256=HEX_C,
    )


def _hn_candidate(
    *,
    fp_drop: float,
    recall_delta: float,
    total_map: float,
    map_no_regr: bool,
) -> hn_ag.ArmMetrics:
    return hn_ag.ArmMetrics(
        arm_id="with_hn",
        is_baseline_reference=False,
        fp_drop_frac=fp_drop,
        real_light_recall_delta_pp=recall_delta,
        total_map_delta_pp=total_map,
        map_no_regression=map_no_regr,
        map_regression_tolerance_pp=0.2,
        eval_manifest_sha256=HEX_A,
        fp_manifest_sha256=HEX_B,
        data_yaml_sha256=HEX_C,
    )


def _hn_run(
    *,
    case: str,
    expect: hn_dg.HardNegativeDecision,
    fp_drop: float,
    recall_delta: float,
    total_map: float,
    map_no_regr: bool,
) -> None:
    cand = _hn_candidate(
        fp_drop=fp_drop,
        recall_delta=recall_delta,
        total_map=total_map,
        map_no_regr=map_no_regr,
    )
    inputs = hn_dg.DecisionInputs(
        baseline=_hn_baseline(),
        candidate=cand,
        map_no_regression=map_no_regr,
        map_regression_tolerance_pp=0.2,
    )
    result = hn_dg.apply_decision_rule(inputs)
    assert result.decision == expect, (
        f"HN {case}: expected {expect.value}, got {result.decision.value} | "
        f"notes={result.notes!r}"
    )
    assert result.notes, f"HN {case}: notes must be non-empty"
    print(f"  HN  {case:34s} OK -> {result.decision.value:15s} | {result.notes[:100]}")


def smoke_hn() -> None:
    print("§四 hard_negative_mining.apply_decision_rule:")
    # Deploy: fp_drop ≥ 0.50, recall ≥ -0.5, mAP no-regr.
    _hn_run(
        case="deploy (fp_drop=0.60)",
        expect=hn_dg.HardNegativeDecision.DEPLOY,
        fp_drop=0.60, recall_delta=-0.2, total_map=-0.1,
        map_no_regr=True,
    )
    # Deploy boundary: fp_drop = 0.50 exactly (non-strict ≥).
    _hn_run(
        case="deploy boundary (fp_drop=0.50)",
        expect=hn_dg.HardNegativeDecision.DEPLOY,
        fp_drop=0.50, recall_delta=-0.5, total_map=-0.2,
        map_no_regr=True,
    )
    # Defer: fp_drop ∈ [0.20, 0.50), recall ≥ -0.5. mAP-agnostic.
    _hn_run(
        case="defer (fp_drop=0.30, mAP OK)",
        expect=hn_dg.HardNegativeDecision.DEFER,
        fp_drop=0.30, recall_delta=-0.2, total_map=-0.1,
        map_no_regr=True,
    )
    # Defer: mAP-agnostic — defer even when mAP regressed.
    _hn_run(
        case="defer (mAP regress, defer-agnostic)",
        expect=hn_dg.HardNegativeDecision.DEFER,
        fp_drop=0.30, recall_delta=-0.2, total_map=-0.4,
        map_no_regr=False,
    )
    # Defer boundary: fp_drop = 0.20 exactly (inclusive lower).
    _hn_run(
        case="defer boundary (fp_drop=0.20)",
        expect=hn_dg.HardNegativeDecision.DEFER,
        fp_drop=0.20, recall_delta=-0.5, total_map=-0.1,
        map_no_regr=True,
    )
    # Defer boundary: fp_drop = 0.4999 (exclusive upper).
    _hn_run(
        case="defer boundary (fp_drop=0.4999)",
        expect=hn_dg.HardNegativeDecision.DEFER,
        fp_drop=0.4999, recall_delta=-0.5, total_map=-0.2,
        map_no_regr=True,
    )
    # Drop literal: recall < -0.5pp.
    _hn_run(
        case="drop literal (recall=-0.6pp)",
        expect=hn_dg.HardNegativeDecision.DROP,
        fp_drop=0.60, recall_delta=-0.6, total_map=-0.1,
        map_no_regr=True,
    )
    # Drop literal: fp_drop < 0.20.
    _hn_run(
        case="drop literal (fp_drop=0.10)",
        expect=hn_dg.HardNegativeDecision.DROP,
        fp_drop=0.10, recall_delta=-0.2, total_map=-0.1,
        map_no_regr=True,
    )
    # Drop catch-all: fp_drop ≥ 0.50, recall ≥ -0.5, mAP regress.
    _hn_run(
        case="drop catch-all (fp=0.60, mAP regress)",
        expect=hn_dg.HardNegativeDecision.DROP,
        fp_drop=0.60, recall_delta=-0.3, total_map=-0.4,
        map_no_regr=False,
    )
    # Drop literal both: recall < -0.5 AND fp_drop < 0.20.
    _hn_run(
        case="drop literal (both triggers)",
        expect=hn_dg.HardNegativeDecision.DROP,
        fp_drop=0.05, recall_delta=-0.8, total_map=-0.1,
        map_no_regr=True,
    )
    # Executor error: NaN bypass.
    cand_ok = _hn_candidate(
        fp_drop=0.30, recall_delta=-0.2, total_map=-0.1, map_no_regr=True,
    )
    object.__setattr__(cand_ok, "fp_drop_frac", math.nan)
    inputs_nan = hn_dg.DecisionInputs.__new__(hn_dg.DecisionInputs)
    object.__setattr__(inputs_nan, "baseline", _hn_baseline())
    object.__setattr__(inputs_nan, "candidate", cand_ok)
    object.__setattr__(inputs_nan, "map_no_regression", True)
    object.__setattr__(inputs_nan, "map_regression_tolerance_pp", 0.2)
    result = hn_dg.apply_decision_rule(inputs_nan)
    assert result.decision == hn_dg.HardNegativeDecision.EXECUTOR_ERROR, (
        f"HN executor_error: expected EXECUTOR_ERROR, got {result.decision.value}"
    )
    assert "NaN/inf" in result.notes, f"HN executor_error notes: {result.notes!r}"
    print(f"  HN  {'executor_error (NaN bypass)':34s} OK -> "
          f"{result.decision.value:15s} | {result.notes[:100]}")

    # Executor error: verdict / delta inconsistency. fp_drop ≥ 0.50 +
    # recall ≥ -0.5pp pass deploy guards; upstream stamps
    # map_no_regression=True but total_map_delta_pp=-0.5pp is BELOW
    # the 0.2pp tolerance. Without the defensive cross-check the
    # deploy guard would fire on a regressing arm (§4.7 has no literal
    # drop trigger on mAP alone — the verdict is the only mAP guard
    # against deploy). WITH the check, surface as EXECUTOR_ERROR.
    cand_bad_verdict = _hn_candidate(
        fp_drop=0.60, recall_delta=-0.2, total_map=-0.5,
        map_no_regr=True,  # WRONG: -0.5pp regress is below the 0.2pp tolerance
    )
    inputs_bad = hn_dg.DecisionInputs(
        baseline=_hn_baseline(),
        candidate=cand_bad_verdict,
        map_no_regression=True,
        map_regression_tolerance_pp=0.2,
    )
    result = hn_dg.apply_decision_rule(inputs_bad)
    assert result.decision == hn_dg.HardNegativeDecision.EXECUTOR_ERROR, (
        f"HN verdict-inconsistency: expected EXECUTOR_ERROR (deploy guard "
        f"would have fired without the cross-check), got "
        f"{result.decision.value} | notes={result.notes!r}"
    )
    assert "inconsistency" in result.notes
    print(f"  HN  {'executor_error (bad verdict)':34s} OK -> "
          f"{result.decision.value:15s} | {result.notes[:100]}")
    # Inverse direction: verdict says regression but delta is within
    # tolerance. The defer path is mAP-AGNOSTIC for §四, so without
    # the cross-check this would still defer (fp_drop ∈ [0.20, 0.50),
    # recall OK, map_no_regression=False is irrelevant to defer per
    # §4.7) — defer is the "wrong" outcome because the verdict is
    # inconsistent. Catch it as executor_error.
    cand_bad_inverse = _hn_candidate(
        fp_drop=0.30, recall_delta=-0.2, total_map=-0.1,
        map_no_regr=False,  # WRONG: -0.1pp is within 0.2pp tolerance
    )
    inputs_bad_inverse = hn_dg.DecisionInputs(
        baseline=_hn_baseline(),
        candidate=cand_bad_inverse,
        map_no_regression=False,
        map_regression_tolerance_pp=0.2,
    )
    result = hn_dg.apply_decision_rule(inputs_bad_inverse)
    assert result.decision == hn_dg.HardNegativeDecision.EXECUTOR_ERROR, (
        f"HN verdict-inconsistency inverse: expected EXECUTOR_ERROR, got "
        f"{result.decision.value} | notes={result.notes!r}"
    )
    print(f"  HN  {'executor_error (bad inverse)':34s} OK -> "
          f"{result.decision.value:15s} | {result.notes[:100]}")

    # Executor error: NaN tolerance — HN already covered this in its
    # initial defensive check (the field is in HN's NaN/inf tuple), but
    # exercise it for parity with the CPB NaN-tolerance fix so any
    # future refactor that drops the field from HN's tuple regresses
    # loudly here.
    cand_for_nan_tol = _hn_candidate(
        fp_drop=0.30, recall_delta=-0.2, total_map=-0.1,
        map_no_regr=False,
    )
    inputs_nan_tol = hn_dg.DecisionInputs.__new__(hn_dg.DecisionInputs)
    object.__setattr__(inputs_nan_tol, "baseline", _hn_baseline())
    object.__setattr__(inputs_nan_tol, "candidate", cand_for_nan_tol)
    object.__setattr__(inputs_nan_tol, "map_no_regression", False)
    object.__setattr__(inputs_nan_tol, "map_regression_tolerance_pp", math.nan)
    result = hn_dg.apply_decision_rule(inputs_nan_tol)
    assert result.decision == hn_dg.HardNegativeDecision.EXECUTOR_ERROR, (
        f"HN NaN tolerance: expected EXECUTOR_ERROR, got "
        f"{result.decision.value} | notes={result.notes!r}"
    )
    assert "map_regression_tolerance_pp" in result.notes, (
        f"HN NaN tolerance notes must name the offending field; got "
        f"{result.notes!r}"
    )
    print(f"  HN  {'executor_error (NaN tolerance)':34s} OK -> "
          f"{result.decision.value:15s} | {result.notes[:100]}")


def smoke_cpb_empty_rare_safety() -> None:
    """B2 review MAJOR-2 fix: empty rare_safety surfaces UserWarning.

    Two pathological cases that previously silently passed the deploy
    `≥ -1pp` safety guard (rare_safety_min = 0.0):
      (a) rare_safety_class_ids non-empty, all entries zero-support;
      (b) rare_safety_class_ids empty (safety has no rare overlap).
    Both must now emit UserWarning so the runner / operator sees it.
    """
    import warnings as _warnings

    print("\n§三 compute_arm_metrics empty-rare-safety warning:")
    safety_class_ids = (0, 1, 2, 8)  # class 8 is rare ∩ safety

    # Baseline with class 8 (the rare-safety) at FULL_VAL_SUPPORT = 20
    # but candidate with support = 0 → zero_support exclusion → empty
    # eligible_rare_safety.
    baseline = (
        cpb_ag.PerClassAP(class_id=0, class_name="red", ap_at_0_5=0.80, full_val_support=200),
        cpb_ag.PerClassAP(class_id=1, class_name="yellow", ap_at_0_5=0.70, full_val_support=150),
        cpb_ag.PerClassAP(class_id=2, class_name="green", ap_at_0_5=0.82, full_val_support=180),
        cpb_ag.PerClassAP(class_id=7, class_name="forwardRed", ap_at_0_5=0.40, full_val_support=25),
        cpb_ag.PerClassAP(class_id=8, class_name="forwardGreen", ap_at_0_5=0.45, full_val_support=20),
        cpb_ag.PerClassAP(class_id=9, class_name="armOn", ap_at_0_5=0.30, full_val_support=10),
    )
    candidate = tuple(
        cpb_ag.PerClassAP(
            class_id=p.class_id, class_name=p.class_name,
            ap_at_0_5=p.ap_at_0_5,
            full_val_support=0 if p.class_id == 8 else p.full_val_support,
        )
        for p in baseline
    )
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        arm = cpb_ag.compute_arm_metrics(
            arm_id="cp_only",
            baseline_per_class=baseline,
            candidate_per_class=candidate,
            baseline_total_map=0.65, candidate_total_map=0.65,
            baseline_rare_fp_count=50, candidate_rare_fp_count=50,
            rare_class_threshold=30,
            safety_class_ids=safety_class_ids,
            eval_manifest_sha256=HEX_A,
            fp_manifest_sha256=HEX_B,
            map_no_regression=True,
            map_regression_tolerance_pp=0.2,
            data_yaml_sha256=HEX_C,
        )
        warned = [str(wi.message) for wi in w if issubclass(wi.category, UserWarning)]
        assert any("rare-safety" in msg for msg in warned), (
            f"case (a): expected UserWarning about rare-safety; got {warned}"
        )
        assert arm.rare_safety_min_delta_AP_pp == 0.0
        assert arm.zero_support_rare_classes == (8,)
        print(f"  (a) zero-support rare-safety → UserWarning fired OK")

    # Case (b): rare_class_ids contains classes 7,8,9 but
    # safety_class_ids has no overlap (only 0,1,2).
    with _warnings.catch_warnings(record=True) as w:
        _warnings.simplefilter("always")
        arm = cpb_ag.compute_arm_metrics(
            arm_id="cp_only",
            baseline_per_class=baseline,
            candidate_per_class=baseline,  # self → all-zero
            baseline_total_map=0.65, candidate_total_map=0.65,
            baseline_rare_fp_count=50, candidate_rare_fp_count=50,
            rare_class_threshold=30,
            safety_class_ids=(0, 1, 2),  # no rare overlap (rare = 7,8,9)
            eval_manifest_sha256=HEX_A,
            fp_manifest_sha256=HEX_B,
            map_no_regression=True,
            map_regression_tolerance_pp=0.2,
            data_yaml_sha256=HEX_C,
        )
        warned = [str(wi.message) for wi in w if issubclass(wi.category, UserWarning)]
        assert any("zero overlap" in msg for msg in warned), (
            f"case (b): expected UserWarning about zero overlap; got {warned}"
        )
        assert arm.rare_safety_class_ids == ()
        print(f"  (b) safety∩rare empty   → UserWarning fired OK")


def smoke_cpb_end_to_end() -> None:
    """Drive §三 PerClassAP → compute_arm_metrics → apply_decision_rule.

    Confirms the c-stage helper and d-stage gate compose correctly. The
    baseline arm is built by self-comparison (same per_class iterable both
    sides); candidate arms are built against the baseline. The §3.7 rule
    is then applied to each candidate cell.
    """
    print("\n§三 end-to-end (compute_arm_metrics → apply_decision_rule):")
    # Per-class AP setup: 10 classes (R2 nc lower bound), 3 are "rare"
    # (support < 30): class IDs 7, 8, 9. Class 8 is in the safety set.
    safety_class_ids = (0, 1, 2, 8)  # red, yellow, green + rare-safety

    baseline_per_class = (
        cpb_ag.PerClassAP(class_id=0, class_name="red", ap_at_0_5=0.80, full_val_support=200),
        cpb_ag.PerClassAP(class_id=1, class_name="yellow", ap_at_0_5=0.70, full_val_support=150),
        cpb_ag.PerClassAP(class_id=2, class_name="green", ap_at_0_5=0.82, full_val_support=180),
        cpb_ag.PerClassAP(class_id=3, class_name="redLeft", ap_at_0_5=0.75, full_val_support=80),
        cpb_ag.PerClassAP(class_id=4, class_name="greenLeft", ap_at_0_5=0.72, full_val_support=60),
        cpb_ag.PerClassAP(class_id=5, class_name="redRight", ap_at_0_5=0.60, full_val_support=40),
        cpb_ag.PerClassAP(class_id=6, class_name="greenRight", ap_at_0_5=0.62, full_val_support=35),
        # Rare classes: support < 30.
        cpb_ag.PerClassAP(class_id=7, class_name="forwardRed", ap_at_0_5=0.40, full_val_support=25),
        cpb_ag.PerClassAP(class_id=8, class_name="forwardGreen", ap_at_0_5=0.45, full_val_support=20),
        cpb_ag.PerClassAP(class_id=9, class_name="armOn", ap_at_0_5=0.30, full_val_support=10),
    )

    # Baseline arm (no_aug): self-comparison → all-zero deltas.
    baseline_arm = cpb_ag.compute_arm_metrics(
        arm_id="no_aug",
        baseline_per_class=baseline_per_class,
        candidate_per_class=baseline_per_class,
        baseline_total_map=0.65,
        candidate_total_map=0.65,
        baseline_rare_fp_count=50,
        candidate_rare_fp_count=50,
        rare_class_threshold=30,
        safety_class_ids=safety_class_ids,
        eval_manifest_sha256=HEX_A,
        fp_manifest_sha256=HEX_B,
        map_no_regression=True,
        map_regression_tolerance_pp=0.2,
        data_yaml_sha256=HEX_C,
    )
    assert baseline_arm.is_baseline_reference is True
    assert baseline_arm.rare_class_ids == (7, 8, 9), (
        f"rare set should be class IDs 7,8,9; got {baseline_arm.rare_class_ids}"
    )
    assert baseline_arm.rare_safety_class_ids == (8,), (
        f"rare-safety should be (8,); got {baseline_arm.rare_safety_class_ids}"
    )
    assert baseline_arm.zero_support_rare_classes == ()
    assert baseline_arm.rare_class_mean_delta_AP_pp == 0.0
    assert baseline_arm.total_map_delta_pp == 0.0
    print(f"  baseline arm: rare={baseline_arm.rare_class_ids}, "
          f"rare_safety={baseline_arm.rare_safety_class_ids}, "
          f"zero_support={baseline_arm.zero_support_rare_classes} OK")

    # Candidate: rare improvement +6pp avg, rare_safety -0.5pp, FP -10%
    # → deploy.
    cand_per_class_deploy = tuple(
        cpb_ag.PerClassAP(
            class_id=p.class_id,
            class_name=p.class_name,
            ap_at_0_5=p.ap_at_0_5 + (0.06 if p.class_id in (7, 9) else
                                    -0.005 if p.class_id == 8 else
                                    0.0),
            full_val_support=p.full_val_support,
        )
        for p in baseline_per_class
    )
    cand_arm_deploy = cpb_ag.compute_arm_metrics(
        arm_id="cp_balanced",
        baseline_per_class=baseline_per_class,
        candidate_per_class=cand_per_class_deploy,
        baseline_total_map=0.65,
        candidate_total_map=0.66,  # +1pp mAP
        baseline_rare_fp_count=50,
        candidate_rare_fp_count=45,  # -10% FP
        rare_class_threshold=30,
        safety_class_ids=safety_class_ids,
        eval_manifest_sha256=HEX_A,
        fp_manifest_sha256=HEX_B,
        map_no_regression=True,
        map_regression_tolerance_pp=0.2,
        data_yaml_sha256=HEX_C,
    )
    # Sanity: rare mean ≈ (6 + -0.5 + 6) / 3 = 3.83pp, max = 6pp
    assert 3.5 < cand_arm_deploy.rare_class_mean_delta_AP_pp < 4.0
    assert abs(cand_arm_deploy.rare_class_max_delta_AP_pp - 6.0) < 1e-6
    assert abs(cand_arm_deploy.rare_safety_min_delta_AP_pp - (-0.5)) < 1e-6
    assert abs(cand_arm_deploy.rare_related_fp_delta_frac - (-0.10)) < 1e-6
    assert abs(cand_arm_deploy.total_map_delta_pp - 1.0) < 1e-6
    print(f"  candidate (deploy): rare_mean={cand_arm_deploy.rare_class_mean_delta_AP_pp:.3f}pp, "
          f"max={cand_arm_deploy.rare_class_max_delta_AP_pp:.3f}pp, "
          f"safety_min={cand_arm_deploy.rare_safety_min_delta_AP_pp:.3f}pp, "
          f"fp={cand_arm_deploy.rare_related_fp_delta_frac:.4f}, "
          f"map={cand_arm_deploy.total_map_delta_pp:.3f}pp OK")

    inputs = cpb_dg.DecisionInputs(
        baseline=baseline_arm,
        candidate=cand_arm_deploy,
        map_no_regression=True,
        map_regression_tolerance_pp=0.2,
    )
    # Rare max is 6pp ≥ 5pp → deploy via max-path.
    result = cpb_dg.apply_decision_rule(inputs, beta=0.999)
    assert result.decision == cpb_dg.CopyPasteDecision.DEPLOY
    print(f"  → apply_decision_rule: {result.decision.value} OK")

    # Zero-support behavior: class 9 has support 0 on candidate.
    zero_support_cand = tuple(
        cpb_ag.PerClassAP(
            class_id=p.class_id,
            class_name=p.class_name,
            ap_at_0_5=p.ap_at_0_5,
            full_val_support=0 if p.class_id == 9 else p.full_val_support,
        )
        for p in baseline_per_class
    )
    arm_zero = cpb_ag.compute_arm_metrics(
        arm_id="cp_only",
        baseline_per_class=baseline_per_class,
        candidate_per_class=zero_support_cand,
        baseline_total_map=0.65,
        candidate_total_map=0.65,
        baseline_rare_fp_count=50,
        candidate_rare_fp_count=50,
        rare_class_threshold=30,
        safety_class_ids=safety_class_ids,
        eval_manifest_sha256=HEX_A,
        fp_manifest_sha256=HEX_B,
        map_no_regression=True,
        map_regression_tolerance_pp=0.2,
        data_yaml_sha256=HEX_C,
    )
    assert arm_zero.zero_support_rare_classes == (9,), (
        f"class 9 should be in zero_support; got {arm_zero.zero_support_rare_classes}"
    )
    print(f"  zero-support handling: class 9 excluded from "
          f"rare deltas (zero_support={arm_zero.zero_support_rare_classes}) OK")


def smoke_hn_end_to_end() -> None:
    """Drive §四 compute_arm_metrics → apply_decision_rule end-to-end."""
    print("\n§四 end-to-end (compute_arm_metrics → apply_decision_rule):")
    # Baseline: 100 FPs on the §4.7 manifest, recall 0.90, mAP 0.85.
    baseline_arm = hn_ag.compute_arm_metrics(
        arm_id="no_hn",
        is_baseline_reference=True,
        baseline_fp_count=100,
        candidate_fp_count=100,
        baseline_real_light_recall=0.90,
        candidate_real_light_recall=0.90,
        baseline_total_map=0.85,
        candidate_total_map=0.85,
        eval_manifest_sha256=HEX_A,
        fp_manifest_sha256=HEX_B,
        map_no_regression=True,
        map_regression_tolerance_pp=0.2,
        data_yaml_sha256=HEX_C,
    )
    assert baseline_arm.is_baseline_reference is True
    assert baseline_arm.fp_drop_frac == 0.0
    assert baseline_arm.real_light_recall_delta_pp == 0.0
    print(f"  baseline arm: fp_drop=0, recall_delta=0, "
          f"map_delta=0 (self-comparison) OK")

    # Candidate: FPs 100 → 40 (60% drop), recall 0.90 → 0.896, mAP 0.85 → 0.849
    # → deploy (fp ≥ 0.50, recall_delta ≈ -0.4pp ≥ -0.5pp, mAP no-regr).
    # Note: deltas chosen to land clear of boundaries that IEEE 754 can't
    # represent exactly. e.g. 0.895 - 0.90 in float = -0.0050…0044, which
    # × 100 = -0.5000…0004pp — below the -0.5pp boundary. Real eval
    # pipelines won't hit boundaries by accident; the smoke fixture must.
    # Similarly the verdict ↔ delta cross-check at the gate uses
    # ``total_map_pp >= -tolerance_pp``, so total_map=-0.2pp exactly is
    # also fragile under (cand_abs - base_abs) * 100 rounding.
    cand_arm = hn_ag.compute_arm_metrics(
        arm_id="with_hn",
        is_baseline_reference=False,
        baseline_fp_count=100,
        candidate_fp_count=40,
        baseline_real_light_recall=0.90,
        candidate_real_light_recall=0.896,
        baseline_total_map=0.85,
        candidate_total_map=0.849,
        eval_manifest_sha256=HEX_A,
        fp_manifest_sha256=HEX_B,
        map_no_regression=True,
        map_regression_tolerance_pp=0.2,
        data_yaml_sha256=HEX_C,
    )
    assert abs(cand_arm.fp_drop_frac - 0.60) < 1e-6
    assert -0.5 < cand_arm.real_light_recall_delta_pp < -0.3, (
        f"recall_delta_pp = {cand_arm.real_light_recall_delta_pp}; "
        f"expected ~-0.4 (above the -0.5pp drop boundary)"
    )
    assert -0.2 < cand_arm.total_map_delta_pp < 0.0, (
        f"total_map_delta_pp = {cand_arm.total_map_delta_pp}; "
        f"expected ~-0.1pp (within the 0.2pp tolerance, clear of boundary)"
    )
    print(f"  candidate (deploy): fp_drop={cand_arm.fp_drop_frac:.4f}, "
          f"recall_delta={cand_arm.real_light_recall_delta_pp:.3f}pp, "
          f"map_delta={cand_arm.total_map_delta_pp:.3f}pp OK")

    inputs = hn_dg.DecisionInputs(
        baseline=baseline_arm,
        candidate=cand_arm,
        map_no_regression=True,
        map_regression_tolerance_pp=0.2,
    )
    result = hn_dg.apply_decision_rule(inputs)
    assert result.decision == hn_dg.HardNegativeDecision.DEPLOY
    print(f"  → apply_decision_rule: {result.decision.value} OK")

    # Cross-arm mismatch: (arm_id, is_baseline_reference) inconsistent.
    try:
        hn_ag.compute_arm_metrics(
            arm_id="with_hn",
            is_baseline_reference=True,  # mismatch
            baseline_fp_count=100, candidate_fp_count=40,
            baseline_real_light_recall=0.90, candidate_real_light_recall=0.895,
            baseline_total_map=0.85, candidate_total_map=0.849,
            eval_manifest_sha256=HEX_A, fp_manifest_sha256=HEX_B,
            map_no_regression=True, map_regression_tolerance_pp=0.2,
            data_yaml_sha256=HEX_C,
        )
    except ValueError as e:
        assert "is_baseline_reference mismatch" in str(e) or "is_baseline_reference" in str(e)
        print(f"  arm_id/is_baseline_reference mismatch: ValueError raised OK")
    else:
        raise AssertionError("expected ValueError on arm_id/is_baseline_reference mismatch")


def main() -> int:
    smoke_cpb()
    smoke_hn()
    smoke_cpb_end_to_end()
    smoke_hn_end_to_end()
    smoke_cpb_empty_rare_safety()
    print("\nALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
