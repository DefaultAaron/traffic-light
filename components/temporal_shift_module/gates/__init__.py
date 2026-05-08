"""TSM phase-gate evaluators (one module per Gate-1A / Gate-1B / Gate-1C).

Each gate consumes the corresponding phase's output JSON (eval_*.json,
timing_*.json, sidecar verification) and returns an explicit
``(pass: bool, case: str, diagnostics: dict)`` triple. Decision-rule logic
lives ONLY here; the runner reports raw metrics, the gate decides pass/fail.

Files:
    concept_validation_gate.py        Gate-1A: small_target_recall_delta ≥ 2 pp
                                                OR mAP_delta ≥ 1 pp
    full_train_acceptance_gate.py     Gate-1B: small_target_recall ≥ 0.6
                                                AND end_to_end_latency_ms < 26
    streaming_export_gate.py          Gate-1C: ONNX clean AND trtexec exit 0
                                                AND cache_memory_mb < 5
                                                AND end_to_end_latency_ms < 26
                                      (lands when Phase 1-C is scheduled)

------------------------------------------------------------------------------
Gate executor invariants (parallels R2 precision plan + KD plan)
------------------------------------------------------------------------------
- Each gate evaluator MUST emit exactly ONE outcome per phase: pass / fail /
  inconclusive (insufficient_evidence). Zero outcomes or multiple outcomes →
  exit 2 with a malformed-input message.
- Bucketed-recall numbers (small / medium / large by bbox height) MUST come
  from a class-and-bucket-stratified eval — gates do NOT recompute deltas
  from raw class_probs. The runner is responsible for emitting a
  bucketed-eval JSON; the gate consumes and decides.
- Latency numbers (end_to_end_latency_ms) MUST carry the locked caveat
  columns input_source / resolution / tracker_mode / output_mode (parallels
  R2 demo FPS columns). Gate-1B refuses to consume timing JSON missing any
  caveat column.
- Phase 1-A "small target recall +2 pp OR mAP +1 pp" is an OR (either
  suffices). Phase 1-B "small recall ≥ 0.6 AND latency < 26" is an AND
  (both required). Wiring these wrong is the canonical scaffold-to-impl
  failure mode the plan §1.5 wording is trying to prevent.

------------------------------------------------------------------------------
Acceptance-gate philosophy
------------------------------------------------------------------------------
Gates are pre-committed at scaffold time (here, this module) so they cannot
be silently relaxed mid-phase. If the chosen detector's R2 small-target
recall is unusual (e.g. already > 0.6 single-frame, making Gate-1B trivial
to satisfy), the gate is RE-COMMITTED at Phase 1-B scheduling time, NOT
amended mid-flight. The "do not amend mid-flight" rule mirrors the R2
precision parity plan's build-determinism rule.
"""
