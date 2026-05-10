"""c-stage acceptance gate + d-stage decision rule for §四 hard-negative mining.

Sister precedent: ``components/copy_paste_balance/gates/`` (locked
iter-11 2026-05-09). Same boundary discipline: per-cell input
validation at dataclass construction; output validation against the
JSON Schema BEFORE atomic rename; cross-row invariants in
``runners/ablation.assert_artifact_invariants``.

The §四 decision rule is a 2-arm cascade (no_hn baseline vs with_hn
candidate) — much simpler than the §三 4-case rule because there's no
β sweep and no rare-class arithmetic. Plan §4.7 verbatim:

  * deploy : FP count drop ≥ 50% AND real-light recall delta ≥ −0.5 pp
              AND total mAP no-regression (default tol 0.2 pp).
  * defer  : 20% ≤ FP drop < 50% AND real-light recall delta ≥ −0.5 pp.
  * drop   : real-light recall delta < −0.5 pp OR FP drop < 20%.

Boundary semantics: drop triggers strict (>); deploy/defer guards
non-strict (≥). See ``decision_gate.py`` module docstring for the
full boundary table.
"""
