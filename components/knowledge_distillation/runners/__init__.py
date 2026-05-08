"""Per-cell training entrypoints (one module per A0-A7 cell).

Each runner is a stub until its cell is scheduled. Stubs raise
NotImplementedError to keep accidental invocation loud. Wrapper shell scripts
(``scripts/train_kd_*.sh``) will be added when the corresponding cell lands.

------------------------------------------------------------------------------
Expected uniform CLI (locked when the first cell lands; non-enforced contract)
------------------------------------------------------------------------------
The first runner to be implemented (likely A1) should declare argparse with
this shape; subsequent cells extend, never rename. Preventing CLI drift is
why this contract is pinned BEFORE any cell is implemented.

    --config <yaml>             student + teacher + KD hyperparams
    --teacher-ckpt <path>       fine-tuned teacher artifact (R2 nc range);
                                may be repeated for multi-teacher cells (A5),
                                consumed in declaration order as phase-1 /
                                phase-2 / ... — repeating is the documented
                                extension; do NOT introduce `--teacher-ckpts`
                                or any plural rename.
    --student-init {scratch,coco,r2_baseline}
    --output-dir runs/<cell_id>/
    --seed <int>                written to SEED.txt at run START
    --ci-method {bootstrap1000,seed5}   default bootstrap1000 per §6#1
    --resume <ckpt>             on resume: SEED.txt is NOT overwritten
    --assistant-ckpt <path>     A7 only: TAKD intermediate (M-tier) used to
                                bridge from L-tier teacher to S student.

------------------------------------------------------------------------------
Reproducibility contract (project-wide, see CLAUDE.md)
------------------------------------------------------------------------------
- Write SEED.txt to output-dir BEFORE invoking the trainer (survives crashes).
- On --resume: do NOT write SEED.txt. The original run dir already owns the
  correct seed; CLI default would corrupt the metadata. Source seed FROM
  SEED.txt and pass that value to the trainer (see scripts/train_deim.sh).
- exec the trainer call with NO trailing safety-net lines so failures
  propagate cleanly (see scripts/train_deim.sh, scripts/train_yolov13.sh).
- args.yaml is auto-emitted by Ultralytics / DEIM trainer — never write it
  manually.

------------------------------------------------------------------------------
KD-implementation contract (per v1.2 §5.4; classic mistakes pre-empted)
------------------------------------------------------------------------------
- Teacher MUST be in eval() mode + torch.no_grad() during student forward.
- Teacher forwards on the SAME augmented image as the student (Wang 2022;
  v1.2 §5.4). Re-running the teacher on un-augmented images is wrong.
- KD warmup: ramp KD weight from 0 to full over Stage-1 epochs; Stage-2
  (final 5-10% epochs) optionally KD-off + hard-target only.
- AMP scaler must scale KD loss consistently with hard-target loss; use
  one shared scaler.
- Teacher feature/logit cache (per §5.4: SSD cache enabled when throughput
  < 60% A1 baseline) MUST be augmentation-aware. Caching by image-id
  alone re-reads teacher outputs that were forwarded on a DIFFERENT
  augmented input, silently violating §5.4 + Wang 2022. Implementable
  cache key options, in order of preference:
    1. (image-id, sampled-transform-state hash, batch-composition hash)
       when the augmentation pipeline exposes the sampled state
       (RandomIoUCrop crop-box, Mosaic source-indices, RandomZoomOut
       fill, etc.) and the worker RNG seed.
    2. SHA256 of the post-transform tensor itself (full provenance
       fallback when the upstream pipeline doesn't expose its sampled
       state — true for parts of Ultralytics' built-in aug chain).
  In either case, also verify class-list / nc compatibility against the
  cache header before reuse.
- DDP unused-parameter risk: A6/A7 introduce training-only auxiliary
  branches (cross-arch projection MLP, TAKD assistant). If those branches
  are conditional per-batch (e.g., active only on certain phases or
  inputs), DDP's reducer can hit the same `unused parameter idx N` crash
  that R1 DEIM-M training did at ep40. Mitigations, in order of
  preference: (a) make the auxiliary branch participate every step (best
  — no DDP overhead, no silent-skip bugs); (b) explicitly mark it
  excluded when inactive (next best — DDP graph stays static per phase);
  (c) declare `find_unused_parameters: True` (tolerated fallback —
  matches the existing DEIM-M traffic_light config, but adds reducer
  overhead per step and masks future bugs where a parameter SHOULD have
  participated). Document the chosen path in the cell's KD report.

------------------------------------------------------------------------------
Cell-to-spec map (v1.2 §5.1)
------------------------------------------------------------------------------
Each runner docstring carries: cell ID, summary, plan §-anchors, trigger
condition. Treat the runner docstring as the on-disk contract; the plan
document is the upstream truth.
"""
