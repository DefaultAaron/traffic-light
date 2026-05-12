"""Compute concrete field-by-field diff between old training-time
`runs/detect/deim_dfine_{s,m}-r1/eval/latest.pth` (in-loop last-epoch snapshot)
and new deployment-checkpoint `logs/deim_eval_{s,m}/eval.pth`
(--test-only on best_stg2.pth).

Output: durable JSON at logs/deim_eval/old_vs_new_diff.json that backs the
methodology-equivalence claim in docs/reports/phase_2_round_1_results.md
§DEIM-D-FINEs and §DEIM-D-FINEm footnotes.

Run once after _remote_deim_eval.sh has produced the new eval pickles AND
the old `runs/detect/deim_dfine_{s,m}-r1/eval/latest.pth` files are
present locally (synced from training rig).

This file is colocated under `docs/reports/r1_evidence/` alongside its
one-shot output `deim_eval_old_vs_new_diff.json`; the methodology-
equivalence claim only mattered at R1 closure and won't be re-audited.
Invoke from repo root: `python docs/reports/r1_evidence/_deim_eval_diff_audit.py`.
"""

import json
from pathlib import Path

import numpy as np
import torch


def parse(eval_pth: Path) -> dict:
    e = torch.load(eval_pth, map_location="cpu", weights_only=False)
    P = e["precision"]
    T0, A0, M2 = 0, 0, 2
    aps: list[float] = []
    ap50s: list[float] = []
    p_bestf1s: list[float] = []
    r_bestf1s: list[float] = []
    K = P.shape[2]
    r_curve = np.linspace(0.0, 1.0, 101)
    for k in range(K):
        pr50 = P[T0, :, k, A0, M2]
        valid = pr50 > -1
        ap50 = float(pr50[valid].mean()) if valid.any() else 0.0
        prall = P[:, :, k, A0, M2]
        valid = prall > -1
        ap = float(prall[valid].mean()) if valid.any() else 0.0
        p = P[T0, :, k, A0, M2]
        mask = p > -1
        if mask.any():
            pc, rc = p[mask], r_curve[mask]
            f1 = np.where((pc + rc) > 0, 2 * pc * rc / (pc + rc), 0.0)
            j = int(f1.argmax())
            p_bestf1s.append(float(pc[j]))
            r_bestf1s.append(float(rc[j]))
        else:
            p_bestf1s.append(0.0)
            r_bestf1s.append(0.0)
        ap50s.append(ap50)
        aps.append(ap)
    return {
        "overall_ap50": sum(ap50s) / K,
        "overall_ap": sum(aps) / K,
        "overall_p": sum(p_bestf1s) / K,
        "overall_r": sum(r_bestf1s) / K,
        "per_class_ap50": ap50s,
        "per_class_ap": aps,
        "per_class_p": p_bestf1s,
        "per_class_r": r_bestf1s,
    }


def diff(old: dict, new: dict, names: list[str]) -> dict:
    out: dict = {"aggregate": {}, "per_class": []}
    for k in ["overall_ap50", "overall_ap", "overall_p", "overall_r"]:
        out["aggregate"][k] = {"old": old[k], "new": new[k], "delta": new[k] - old[k]}
    for i, n in enumerate(names):
        out["per_class"].append({
            "class": n,
            "ap50":  {"old": old["per_class_ap50"][i], "new": new["per_class_ap50"][i], "delta": new["per_class_ap50"][i] - old["per_class_ap50"][i]},
            "ap":    {"old": old["per_class_ap"][i],   "new": new["per_class_ap"][i],   "delta": new["per_class_ap"][i] - old["per_class_ap"][i]},
            "p":     {"old": old["per_class_p"][i],    "new": new["per_class_p"][i],    "delta": new["per_class_p"][i] - old["per_class_p"][i]},
            "r":     {"old": old["per_class_r"][i],    "new": new["per_class_r"][i],    "delta": new["per_class_r"][i] - old["per_class_r"][i]},
        })
    all_deltas = [
        abs(out["aggregate"][k]["delta"]) for k in ["overall_ap50", "overall_ap", "overall_p", "overall_r"]
    ] + [
        abs(row[m]["delta"]) for row in out["per_class"] for m in ["ap50", "ap", "p", "r"]
    ]
    out["max_abs_delta_across_32_fields"] = max(all_deltas)
    return out


def main() -> None:
    names = ["red", "yellow", "green", "redLeft", "greenLeft", "redRight", "greenRight"]
    # __file__ at docs/reports/r1_evidence/_deim_eval_diff_audit.py
    # parents[0]=r1_evidence, [1]=reports, [2]=docs, [3]=repo_root
    repo = Path(__file__).resolve().parents[3]
    out_dir = repo / "docs" / "reports" / "r1_evidence"
    out_dir.mkdir(parents=True, exist_ok=True)
    results: dict = {
        "purpose": "Backing artifact for docs/reports/phase_2_round_1_results.md §DEIM-D-FINE{s,m} methodology-equivalence footnotes (old eval/latest.pth in-loop snapshot vs new --test-only on deployment best ckpt)",
        "generator": "docs/reports/r1_evidence/_deim_eval_diff_audit.py",
        "field_count_per_model": "32 = 4 aggregate (overall AP50, AP, P_bestF1, R_bestF1) + 7 classes × 4 per-class metrics (AP50, AP, P, R)",
        "models": {},
    }
    for size in ["s", "m"]:
        old_pth = repo / f"runs/detect/deim_dfine_{size}-r1/eval/latest.pth"
        new_pth = repo / f"logs/deim_eval_{size}/eval.pth"
        if not (old_pth.exists() and new_pth.exists()):
            results["models"][size] = {"status": "missing", "old": str(old_pth), "new": str(new_pth)}
            continue
        old = parse(old_pth)
        new = parse(new_pth)
        results["models"][size] = diff(old, new, names)
    out = out_dir / "deim_eval_old_vs_new_diff.json"
    out.write_text(json.dumps(results, indent=2))
    s_max = results["models"]["s"]["max_abs_delta_across_32_fields"]
    m_max = results["models"]["m"]["max_abs_delta_across_32_fields"]
    print(f"[OK] {out}")
    print(f"  DEIM-S max|Δ| = {s_max:.4f}")
    print(f"  DEIM-M max|Δ| = {m_max:.4f}")


if __name__ == "__main__":
    main()
