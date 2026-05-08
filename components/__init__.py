"""Pipeline components beyond the core detector workstream.

Each subpackage owns one optional add-on. Code here is independent of the main
training entrypoints in ``main.py`` — runners import from here only when the
corresponding R2/R3 cell or phase is scheduled.

Subpackages:
    knowledge_distillation/    KD pipeline (R2 in-round; cells A0-A7 across
                               YOLO + DEIM families). v1.3 plan at
                               docs/planning/knowledge_distillation_pipeline.md.
    temporal_shift_module/     TSM detector-level temporal optimization (R2/R3
                               optional track; phases 1-A / 1-B / 1-C). v1.0
                               plan at docs/planning/temporal_optimization_plan.md
                               §1. The §2 post-detector smoothers (HMM /
                               AdaEMA / GRU / Transformer) are a SEPARATE
                               temporal track and live under ``inference/
                               temporal/`` per plan §7 — not in this tree.
"""
