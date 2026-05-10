"""2-arm ablation runner for §四 hard-negative mining (a-stage scaffold).

Sister precedent: ``components/copy_paste_balance/runners/`` (locked
iter-11 2026-05-09). Runner consumes already-trained eval JSONs from
two arms (no_hn baseline + with_hn candidate), applies the §4.7
decision rule, validates the output against the JSON Schema, and
atomically writes ``runs/_hard_negative_decision.json``.
"""
