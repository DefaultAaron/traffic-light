"""Temporal Shift Module — detector-level temporal optimization (R2/R3 optional, v1.0).

Authoritative spec: ``docs/planning/temporal_optimization_plan.md`` §1 (TSM
recommended path) + §4 (data preparation) + §7 (file change list). The §2
post-detector smoothers (HMM / AdaEMA / GRU / Transformer) are a SEPARATE
optimization track and live under ``inference/temporal/`` per plan §7 — not in
this package.

Mechanism (§1.1):
    YOLO neck (C3 / C2f / equivalent) BasicBlock.forward injects a 1/8-channel
    forward shift along the time axis; remaining 7/8 channels keep the original
    spatial-only path. Streaming inference re-uses the previous frame's 1/8
    channel from a per-stage cache (no future frame access).

    feat_t = Conv(concat(prev_feat_t.c1, feat_t.c2, feat_t.c3))
        where split sizes = (1/8 C, 1/8 C reserved for offline bidirectional
        training, 6/8 C residual). Online inference uses forward-shift only.

Phases (§1.5):
    1-A  PoC, 1 week — small-data train (10% data, 20 epochs); gate on
         small-target recall +2 pp OR overall mAP +1 pp.
    1-B  Full train, 1 week — full R2 dataset, same epoch / patience /
         augmentation as the main detector; gate on small-target recall ≥ 0.6
         AND end-to-end Orin latency < 26 ms.
    1-C  Export + Orin verify, 3-5 days — ONNX (Slice + Concat for shift, no
         custom op), trtexec FP16, per-camera per-stage feature cache wired in
         the C++ inference path.

Acceptance gates (§1.5; phase-sequential, all required to ship TSM):
    Gate-1A  small_target_recall_delta ≥ 2 pp  OR  mAP_delta ≥ 1 pp
             (10% data, 20 epochs proxy; either condition suffices)
    Gate-1B  small_target_recall ≥ 0.6 AND end_to_end_latency_ms < 26
             (full R2 dataset, deployed config)
    Gate-1C  ONNX trace clean (no custom ops); trtexec FP16 build clean;
             per-stage feature cache memory < 5 MB / camera; Orin
             end_to_end_latency_ms < 26 (re-measured post-export)

Subpackages:
    modules/    nn.Module pieces — TemporalShift (the shift op itself) +
                FeatureCache (per-stage previous-frame container for streaming
                inference). Generic across detector families.
    data/       clip-of-N=4 dataloader collator wrapping the chosen detector's
                native dataloader; clip-internal frame order preserved, clip
                shuffle randomized.
    patches/    backbone-injection patches per detector family. Lands ONLY for
                the main-track-selected detector at Phase 1-A; no speculative
                patches for non-selected families.
    runners/    per-phase training entrypoints (concept_validation.py /
                full_dataset_train.py / streaming_engine_export.py) + uniform
                CLI / repro / TSM-implementation contracts (see
                runners/__init__.py).
    gates/      per-phase acceptance-gate evaluators
                (concept_validation_gate.py / full_train_acceptance_gate.py).

Detector applicability (v1.0; spec gap vs plan §1 — added at scaffold time):

    YOLO26 / YOLOv13 (Ultralytics-style):
        Inject TSM at C3 / C2f BasicBlock.forward in the neck (the original
        §1.1 surface). Channel split 1/8 + 1/8 + 6/8 is integer for n/s/m
        widths (multiples of 16). Dataloader: Ultralytics native, with
        Mosaic / MixUp augmentations clip-consistent.

    DEIM-D-FINE:
        Inject TSM at HG_Block.forward — `DEIM/engine/backbone/hgnetv2.py:275`.
        The shift acts on `x` at the start of forward, BEFORE the layer loop
        (`for layer in self.layers: x = layer(x)`). Channel split is on
        `in_chs` of the block. HGNetv2 stage in_chs values (typically 64 /
        128 / 256 / 512 / 1024) are all 8-divisible.

        Stages to patch: HG_Stage 1, 2, 3 (the deeper convolutional stages
        of HGNetv2). HGStem (single-conv entry, line 125 `StemBlock`) and
        HG_Stage 0 (very early features) are NOT patched — the small-target
        / occluded recall benefit accrues at mid-to-deep stages where the
        receptive field is still local enough for short-range temporal
        context to matter. HG_Stage exact patching range is re-confirmed
        per detector size at Phase 1-A scheduling.

        OUT-OF-SCOPE for TSM injection on DEIM:
          - HybridEncoder (`DEIM/engine/deim/hybrid_encoder.py`) — the
            transformer encoder with AIFI multi-head self-attention.
            Original TSM (Lin et al. ICCV 2019) is a convolutional-feature
            channel-shift; mixing with attention layers is unstudied for
            video object detection and the §1.6 risk row 4 ("don't expand
            scope speculatively") explicitly warns against this.
          - D-FINE decoder transformer layers — pure attention, not
            applicable.

        ORTHOGONAL TO GO-LSD: GO-LSD is a decoder-side self-distillation
        loss between adjacent decoder layers. TSM at backbone level does
        not interact with GO-LSD at the loss or gradient path. Both can
        stack without rewiring. (This also means: A0 ablation in the KD
        plan's `deim_baseline_golsd_off.py` is independent of TSM
        ablations; the two cells are addable.)

        Dataloader: DEIM uses COCO-style (`DEIM/engine/data/coco_dataset.py`)
        with RandomIoUCrop / RandomZoomOut / RandomHorizontalFlip — different
        from Ultralytics's Mosaic-centric pipeline. The clip collator must
        wrap THIS upstream. Augmentation clip-consistency rule is enforced
        the same way (data/__init__.py).

        DDP `find_unused_parameters: True`: the existing DEIM-M traffic_light
        config sets this. TSM has zero learnable parameters so adds no new
        unused-param risk on its own. The standard caveat still applies —
        do NOT structure Phase 1-A as a runtime "shift on/off" flag in one
        job (the un-shifted residual conv path becomes unused on shift-on
        iterations and the reducer crashes the same way R1 DEIM-M did at
        ep40).

    Phase IDs (1-A / 1-B / 1-C) and acceptance-gate thresholds are
    detector-agnostic; per-detector measured deltas land in the per-detector
    phase report at scheduling time.

Detector coupling (§1.4):
    TSM is structurally coupled to the main-track-selected detector — fine-tune
    from existing R1/R2 weights is INVALID (shift changes channel semantics;
    pretrained channels never saw the shifted layout). Phase 1-A trains from
    scratch on 10% data; Phase 1-B retrains from scratch on full R2 data. This
    forbids "load best.pt + shift on top" as a Phase 1-A shortcut.

Activation gate (§0.2):
    TSM scaffold lives here independent of main-track readiness, but the runners
    MUST NOT execute until BOTH:
      (a) main detector selection is final (R2 ship_precision row populated in
          parent table), AND
      (b) on-vehicle replay surfaces small-target / occluded miss failure
          modes (post-detector smoothers cannot recover detector-side misses).
    The runner stubs raise NotImplementedError until manually rewired — they do
    NOT auto-detect main-track readiness.

Deferred deliverables (NEW, not yet present):
    runs/_tsm_decisions.json                   per-phase decision records
    scripts/_tsm_decision_schema.json          schema (fields parallel to
                                               _r2_decision_schema.json /
                                               _kd_decision_schema.json — TBD
                                               at Phase 1-A scheduling time)
    scripts/_tsm_decide_phase.py               decision-rule executor
    scripts/build_pseudo_labels.py             ByteTrack pseudo-label pipeline
                                               (§4.2; shared with §2 smoothers)
    inference/cpp/include/temporal_cache.hpp   C++ per-camera per-stage cache
    inference/cpp/src/temporal_cache.cpp       cache implementation + lifecycle
    docs/integration/temporal_shift.md         deployment runbook (post-1-C)

Engine sidecar carry-forward (§1.5 Phase 1-C, parallels KD §6#5):
    scripts/export_yolo.sh and scripts/export_deim.sh sidecar (.meta.json) MUST
    grow new fields BEFORE Phase 1-C ships:
      - tsm_enabled: bool
      - tsm_shift_fraction: float (0.125 = 1/8)
      - tsm_clip_size_train: int (4 by default)
      - tsm_feature_cache_stages: list[str] (e.g. ["P3", "P4"])
    Sidecar gap is the canonical pre-Phase-1-C blocker.

Status v1.0:
    Scaffold only — every runner / module / patch / gate stub raises
    NotImplementedError. Plan v1.0 of this README; no §-anchor renumbering
    coordinated with temporal_optimization_plan.md yet.
"""
