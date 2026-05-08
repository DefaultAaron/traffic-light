"""KD-acceptance gate evaluators (helpers; ship-decision lives in scripts/_kd_decide_cell.py).

Planned helpers (per v1.2 §6):
    gate1_map_nondegrade.py  — A1 baseline CI + lower-CI > A1 floor
    gate2_safety_per_class.py— per-class AP delta on full_val_support ≥ 30 classes
    gate3_fp_no_growth.py    — R1 demo8/11/13 background-frame FP count
    gate4_cost_budget.py     — wall-clock < 2.0 × T_scratch_A1
    gate5_trt_engine.py      — engine eval-parity (0.01 pp) at R2 ship_precision

These helpers are pure functions over runs/<cell>/ artifacts. The decision-rule
executor scripts/_kd_decide_cell.py imports them and writes runs/_kd_decisions.json
against scripts/_kd_decision_schema.json.
"""
