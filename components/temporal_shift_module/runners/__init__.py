"""Per-phase TSM training entrypoints (one module per Phase 1-A / 1-B / 1-C).

Each runner is a stub until its phase is scheduled. Stubs raise
NotImplementedError to keep accidental invocation loud. Wrapper shell scripts
(``scripts/train_tsm_*.sh``, ``scripts/export_tsm.sh``) will be added when the
corresponding phase lands.

------------------------------------------------------------------------------
Expected uniform CLI (locked when Phase 1-A lands; non-enforced contract)
------------------------------------------------------------------------------
The Phase 1-A runner declares argparse with this shape; 1-B / 1-C extend, never
rename. Pinning the contract BEFORE any phase is implemented prevents CLI drift
and keeps the cross-phase decision JSON well-formed.

    --config <yaml>             student detector + TSM hyperparams (clip_size,
                                shift_fraction, cache_stages)
    --base-detector {yolo26_s,yolo26_m,yolov13_s,deim_dfine_s,deim_dfine_m}
                                MUST match the main-track-selected detector;
                                runner validates against the patches/ module
                                TARGET_DETECTOR before patching.
    --pretrained-init {scratch}
                                ENFORCED to "scratch" — fine-tune from R1/R2
                                weights is invalid per §1.4 (channel-semantic
                                mismatch). The flag exists for forward-compat
                                only; passing any other value MUST exit 2 with
                                a §1.4 reference.

                                CRITICAL on DEIM (v1.1 hardening): the value
                                "scratch" is NECESSARY but NOT SUFFICIENT.
                                DEIM's HGNetv2 backbone defaults to
                                `pretrained=True` (`DEIM/engine/backbone/
                                hgnetv2.py:443`), which loads stage1
                                pretrained weights from upstream HuggingFace
                                regardless of the runner-level "scratch" flag.
                                The runner MUST also resolve the loaded DEIM
                                config and assert `HGNetv2.pretrained == False`
                                (or override it to False before model init);
                                exit 2 with a v1.1 hardening reference if the
                                resolved config still loads pretrained backbone
                                weights. Same guard for any Ultralytics
                                upstream that auto-loads COCO weights.
    --output-dir runs/<phase_id>/
    --seed <int>                written to SEED.txt at run START (per project
                                reproducibility contract; see CLAUDE.md +
                                scripts/train_deim.sh).
    --ci-method {bootstrap1000,seed5}   default bootstrap1000 (parallels KD §6#1)
    --resume <ckpt>             on resume: SEED.txt is NOT overwritten.
    --clip-size <int>           default 4. Phase 1-A locks this; 1-B / 1-C MUST
                                NOT vary it (cache shape + ONNX trace depend on
                                clip-size at train time).
    --shift-fraction <float>    default 0.125 (= 1/8). Phase 1-A locks this for
                                the same reason.
    --cache-stages <stage,...>  default "P3,P4,P5". Phase 1-C may drop P5 per
                                §1.6 if memory pressured; train-time and
                                inference-time MUST match.

------------------------------------------------------------------------------
Reproducibility contract (project-wide, see CLAUDE.md)
------------------------------------------------------------------------------
- Write SEED.txt to output-dir BEFORE invoking the trainer (survives crashes).
- On --resume: do NOT write SEED.txt. The original run dir already owns the
  correct seed; CLI default would corrupt the metadata. Source seed FROM
  SEED.txt and pass that value to the trainer (see scripts/train_deim.sh).
- exec the trainer call with NO trailing safety-net lines so failures
  propagate cleanly.
- args.yaml is auto-emitted by Ultralytics / DEIM trainer — never write it
  manually; TSM hyperparams ride along on the same args.yaml.

------------------------------------------------------------------------------
TSM-implementation contract (per v1.0 §1.4 + §1.5; classic mistakes pre-empted)
------------------------------------------------------------------------------
- Pretrained-init MUST be "scratch" (§1.4). Loading R1/R2 best.pt and shifting
  on top is invalid; the channels never saw the shifted layout. The runner
  enforces this via the --pretrained-init validator above.
- Augmentation MUST be clip-consistent (data/__init__.py contract) — SAME
  Mosaic source frames, SAME crop box, SAME flip flag across the T frames of
  one clip. This parallels Wang 2022 augmentation consistency in KD; here it's
  intra-student (no teacher) but the same data-pipeline correctness rule.
- Patches/<detector>.apply() MUST be called BEFORE the trainer instantiates
  the model and revert() MUST be called in a finally: block. Leaving the
  monkey-patch active across runs causes silent contamination of TSM-off
  ablation runs.
- Cache stages train-time and inference-time MUST match — train with
  cache_stages = ["P3", "P4", "P5"] and export with ["P3", "P4"] is a Phase
  1-C correctness bug. The export runner reads cache_stages from training
  args.yaml and refuses to drop stages without an explicit --override-stages
  flag (which writes a divergence note into the engine sidecar).
- Phase 1-A failure (gate 1A miss) MUST loop back to §0.2 problem-attribution
  re-evaluation — DO NOT iterate hyperparams hoping for a sub-2pp gain. The
  plan explicitly says "回到 §0.2 重新评估问题归因" because TSM not helping on
  10%-data PoC means TSM probably won't help on full data either.
- DDP unused-parameter risk (parallels KD scaffold contract): the shift op
  itself has no learnable parameters, so the standard KD A6/A7 risk does not
  apply here. BUT — if Phase 1-A is structured as a "shift on / shift off"
  ablation in a SINGLE training run (sharing the rest of the network with a
  conditional flag), then the un-shifted branch's parameters become unused on
  shift-on iterations and DDP's reducer crashes the same way R1 DEIM-M did at
  ep40. Mitigation: run shift-on and shift-off as TWO SEPARATE jobs. This is
  ALSO consistent with plan §1.4 方案 1 ("两个 weight 同时训练 (4090 同 GPU,
  分时或分卡), 对比报告"), which mandates two separate weights side-by-side.
  Plan §1.4 does not explicitly forbid a single-job runtime flag, but the
  scaffold-time enforcement here forbids it on top of §1.4 because of the
  R1 DEIM-M reducer-crash precedent. Document the chosen path in the phase
  report.

- Activation-gate tripwire (v1.1 hardening, v1.2 schema-enforced, v1.3
  three-way SHA + four-tag set + path-regex hardened):
  Every real runner MUST execute the following algorithm BEFORE invoking
  the trainer or exporter, and exit 2 on any failure with a §0.2
  reference:

    1. jsonschema-validate `runs/_tsm_activation.json` against
       `scripts/_tsm_activation_schema.json`. The schema enforces:
       schema_version pin ("1.1" — bumped from "1.0" at v1.4 for the
       v1.3 far_distance_miss enum extension; runner refuses any other
       schema_version value); selected_detector_artifact_sha256
       (64-hex); selected_detector_artifact_path (regex rejects absolute
       paths and '..' segments, requires '.engine' suffix);
       replay_evidence_path (regex rejects absolute and '..');
       approved_failure_mode_tags (closed four-tag enum
       {small_target_miss, far_distance_miss, occluded_miss,
       motion_blur} per plan §0.2 row 1).
    2. Assert selected_detector_artifact_path exists on disk.
    3. Compute computed_sha = SHA256(selected_detector_artifact_path).
    4. Locate sidecar at <selected_detector_artifact_path>.meta.json;
       assert it exists and load sidecar.engine_sha256.
    5. THREE-WAY SHA equality:
           activation_sha (from JSON) == sidecar.engine_sha256
                                      == computed_sha
       Pairwise-mismatch diagnostics (exit 2 with the matching message):
         - activation != computed && sidecar == computed
             → "stale activation tripwire — regenerate activation file"
         - sidecar != computed && activation == computed
             → "stale engine sidecar — regenerate sidecar via export
                script"
         - activation == sidecar != computed
             → "engine file changed without sidecar/activation update —
                rebuild engine"
         - all three differ
             → "full re-activation required"
    6. Assert replay_evidence_path exists on disk (no SHA needed; replay
       evidence is human-curated, not machine-stable).
    7. Assert approved_failure_mode_tags ⊆ this runner module's
       PHASE_FAILURE_MODE_SCOPE constant (each runner declares this
       module-level frozenset; v1.4-introduced — see
       concept_validation.py / full_dataset_train.py /
       streaming_engine_export.py PHASE_FAILURE_MODE_SCOPE). The check
       is set-subset, not docstring-parsing — a future per-phase scope
       narrowing replaces the constant, not the docstring. v1.4 sets
       all three constants to the full four-tag plan-§0.2-row-1 set.

  This converts the activation gate from comment-only policy to a
  runtime tripwire that survives stub-replacement — a future commit
  replacing `raise NotImplementedError` with real trainer code MUST
  keep the activation-gate validation; if it doesn't, code review
  catches the missing import.

------------------------------------------------------------------------------
DEIM-D-FINE-specific implementation notes (when --base-detector is deim_*)
------------------------------------------------------------------------------
- Patch target is `HG_Block.forward` (DEIM/engine/backbone/hgnetv2.py:275),
  NOT a YOLO BasicBlock. The runner MUST import the patch from
  `patches/deim_hg_block.py` and verify its `TARGET_DETECTOR == "deim_dfine"`
  before applying.
- Patched stage scope (UNAMBIGUOUS — three equivalent forms): config names
  `stage2 / stage3 / stage4` = code indices `HGNetv2.stages[1] / [2] / [3]`
  = the entries returned via `return_idx = [1, 2, 3]` in HGNetv2.__init__
  (matching DEIM-S traffic_light config line 32). HG_Block instances inside
  these three stages receive the patched class; StemBlock (line 125) and
  stage1 (`HGNetv2.stages[0]`, NOT in return_idx) do not. The runner asserts
  this before instantiating the model and exits 2 if the patched-stage list
  disagrees with config.
- HybridEncoder (`DEIM/engine/deim/hybrid_encoder.py`) and the D-FINE decoder
  are NEVER patched. The runner asserts the encoder and decoder modules are
  un-monkey-patched before training starts; if any encoder / decoder forward
  has been replaced, exit 2 with a §detector-applicability reference.
- GO-LSD wiring stays UNCHANGED — the runner does NOT toggle GO-LSD.
  Whatever the upstream DEIM config sets (GO-LSD on by default for D-FINE)
  is preserved. Comparing TSM-on vs TSM-off WITH GO-LSD on both sides is the
  apples-to-apples comparison.
- Dataloader path is `DEIM/engine/data/coco_dataset.py` upstream + the
  components/temporal_shift_module/data/clip_collator.py wrapper. The
  collator MUST sample DEIM's RandomIoUCrop / RandomZoomOut / RandomHorizontal
  Flip ONCE per clip and replay the same sampled state across the T frames
  — partial implementation that re-samples per frame silently destroys the
  temporal signal (parallels Wang 2022 augmentation consistency).
- DEIM's `find_unused_parameters: True` (set in the existing M traffic_light
  config) is preserved. TSM adds zero learnable parameters so this flag's
  cost remains the existing reducer overhead — no new unused-param surface.
- Engine sidecar: ENFORCEMENT OWNERSHIP (v1.1 clarification).
    write owner:    `scripts/export_yolo.sh` and `scripts/export_deim.sh`
                    emit `.meta.json` (existing engine-sidecar contract).
                    They will gain the four TSM fields when sidecar carry-
                    forward lands (see top-level __init__.py).
    validate owner: the Phase 1-C runner (`streaming_engine_export.py`)
                    re-reads the sidecar AFTER `trtexec` returns and asserts
                    the four TSM fields are present and match the training
                    args.yaml. The Phase 1-C runner does NOT write the
                    sidecar — only exports do — but the runner's exit-zero
                    contract requires the post-export validation pass.
  The training runners (Phase 1-A / 1-B) write only args.yaml; they don't
  touch the sidecar at all.

------------------------------------------------------------------------------
Phase-to-spec map (v1.0 §1.5)
------------------------------------------------------------------------------
Each runner docstring carries: phase ID, summary, plan §-anchors, gate
condition, trigger condition. Treat the runner docstring as the on-disk
contract; the plan document is the upstream truth.

Files in this subpackage:
    concept_validation.py            Phase 1-A — small-data PoC (10% data,
                                     20 epochs, scratch init, clip-size 4)
    full_dataset_train.py            Phase 1-B — full R2 dataset retrain,
                                     same hyperparams as main detector
    streaming_engine_export.py       Phase 1-C — ONNX (Slice+Concat shift) →
                                     TRT FP16 → per-camera per-stage cache
                                     wired into the C++ inference path
"""
