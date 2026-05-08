"""Temporal Shift Module — detector-level temporal optimization (R2/R3 optional, v1.4).

Authoritative spec: ``docs/planning/temporal_optimization_plan.md`` §1 (TSM
recommended path; carries a 2026-05-09 v1.1 amendment in §1.1 noting the
causal-end-to-end deviation applied here) + §4 (data preparation) + §7 (file
change list). The §2 post-detector smoothers (HMM / AdaEMA / GRU / Transformer)
are a SEPARATE optimization track and live under ``inference/temporal/`` per
plan §7 — not in this package.

Where this README diverges from the plan, this README wins (and the plan
carries a dated amendment pointing at it). Ground-truth precedence:
README §-anchors > plan §1.1 v1.1-amended sections > plan v1 prose.

Mechanism (§1.1, with v1.1 causal-end-to-end hardening):
    YOLO neck (C3 / C2f / equivalent) BasicBlock.forward injects a 1/8-channel
    forward shift along the time axis; remaining 7/8 channels keep the original
    spatial-only path. Streaming inference re-uses the previous frame's 1/8
    channel from a per-stage cache (no future frame access).

    feat_t = Conv(concat(prev_feat_t.c1, zeros_like(c2), feat_t.c3))
        where split sizes = (1/8 C, 1/8 C, 6/8 C residual).

    v1.1 causal hardening (deviation from plan §1.1 explicit, applied here):
        Plan §1.1 shows c2 ← next_feat_t.c2 (backward shift) for offline
        bidirectional training, with the caveat "在线推理只用前向". Following
        that literally would train with future-frame `c2` and zero-pad `c2`
        in streaming inference — a 1/8-channel train/inference distribution
        split that cleanly satisfies the project's highest-risk class:
        Python ORT vs C++ TRT post/export parity divergence.

        v1.1 enforces causal end-to-end: c2 is ZEROED in BOTH train and
        inference. The slot is preserved for structural compatibility with
        any future bidirectional-offline ablation, but DISABLED in v1.x.
        This is original online-TSM design (Lin 2019); the ICCV-2019 paper
        itself recommends single-direction shift for streaming deployment.

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

        Stages to patch (UNAMBIGUOUS identifiers — three equivalent forms):
            config name:     stage2 / stage3 / stage4
            code indexing:   HGNetv2.stages[1] / [2] / [3]
            return mapping:  return_idx = [1, 2, 3] in HGNetv2.__init__,
                             which is the DEIM-S traffic_light config default
                             (`DEIM/configs/deim_dfine/
                             deim_hgnetv2_s_traffic_light.yml:32`)
        These are the three stages whose features are returned to the
        HybridEncoder. The stem (`StemBlock`, line 125) and stage1
        (`HGNetv2.stages[0]`, NOT a returned feature) are NOT patched — the
        receptive field is too small for short-range temporal context to
        matter, and stage1's output is not consumed downstream.

        OUT-OF-SCOPE for TSM injection on DEIM:
          - HybridEncoder (`DEIM/engine/deim/hybrid_encoder.py`) — the
            transformer encoder with AIFI multi-head self-attention.
            Original TSM (Lin et al. ICCV 2019) is a convolutional-feature
            channel-shift; mixing with attention layers is unstudied for
            video object detection and the §1.6 risk row 4 ("don't expand
            scope speculatively") explicitly warns against this.
          - D-FINE decoder transformer layers — pure attention, not
            applicable.

        GO-LSD wiring stays UNCHANGED: GO-LSD is a decoder-side self-
        distillation loss between adjacent decoder layers. TSM at backbone
        level does NOT require any GO-LSD toggle, rewire, or hyperparam
        change — keep GO-LSD fixed across TSM-on and TSM-off comparisons
        for an apples-to-apples ablation.

        IMPORTANT — TSM is not "loss-path independent" of GO-LSD: changing
        backbone features changes the encoder/decoder activations that GO-LSD
        consumes, and gradients from GO-LSD do flow back through the patched
        backbone. The correct phrasing is "no separate wiring required",
        NOT "they don't interact". Stacking with the KD A0 cell
        (`deim_baseline_golsd_off.py`) is therefore a JOINT ablation, not a
        sum of independent effects — design ablations accordingly.

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
    runs/_tsm_activation.json                  activation-gate tripwire
                                               (REQUIRED — see Activation gate
                                               section below). Schema:
                                               `scripts/_tsm_activation_schema.
                                               json` (LANDED v1.2, hardened
                                               v1.3). Required fields:
                                               schema_version (=="1.1";
                                               bumped from "1.0" at v1.4
                                               for the v1.3 enum extension);
                                               selected_detector_artifact_
                                               sha256 (64-hex); selected_
                                               detector_artifact_path (regex-
                                               enforced repo-relative, no
                                               '..' segments, must end in
                                               '.engine' NOT '.pt'); replay_
                                               evidence_path (regex-enforced
                                               repo-relative, no '..');
                                               approved_failure_mode_tags
                                               (closed enum subset of
                                               {small_target_miss,
                                               far_distance_miss,
                                               occluded_miss, motion_blur} —
                                               four-tag set per plan §0.2 row
                                               1; minItems=1, uniqueItems);
                                               activation_timestamp (ISO 8601
                                               UTC). Three-way SHA equality
                                               enforced: activation_sha ==
                                               sidecar.engine_sha256 ==
                                               computed_sha (runner re-
                                               computes from file path AND
                                               loads sidecar at
                                               <path>.meta.json); pairwise
                                               mismatch yields a targeted
                                               stale-source diagnostic per
                                               schema description.
    runs/_tsm_decisions.json                   per-phase decision records
    scripts/_tsm_decision_schema.json          schema (fields parallel to
                                               _r2_decision_schema.json /
                                               _kd_decision_schema.json — TBD
                                               at Phase 1-A scheduling time)
    scripts/_tsm_decide_phase.py               decision-rule executor
    scripts/build_pseudo_labels.py             ByteTrack pseudo-label pipeline
                                               (§4.2; shared with §2 smoothers)
    inference/cpp/include/temporal_cache.hpp   C++ per-camera per-stage cache
                                               (path TBC against plan §7
                                               关键文件改动清单 at Phase 1-C
                                               scheduling)
    inference/cpp/src/temporal_cache.cpp       cache implementation + lifecycle
                                               (path TBC same as above)
    docs/integration/temporal_shift.md         deployment runbook (post-1-C)

Engine sidecar carry-forward (§1.5 Phase 1-C, parallels KD §6#5):
    scripts/export_yolo.sh and scripts/export_deim.sh sidecar (.meta.json) MUST
    grow new fields BEFORE Phase 1-C ships:
      - tsm_enabled: bool
      - tsm_shift_fraction: float (0.125 = 1/8)
      - tsm_clip_size_train: int (4 by default)
      - tsm_feature_cache_stages: list[str] (e.g. ["P3", "P4"])
    Sidecar gap is the canonical pre-Phase-1-C blocker.

Status v1.4:
    Scaffold only — every runner / module / patch / gate stub raises
    NotImplementedError. v1.4 closes the C3 iter-4 ADDITIONAL-FINDINGS
    against v1.3:
      (α) schema_version bumped 1.0 → 1.1 to reflect the v1.3 enum
          extension (added far_distance_miss). The schema's own contract
          said "adding a new tag requires schema_version bump"; v1.3
          forgot to actually do the bump. v1.4 corrects it. Activation
          files written against v1.0 schema must be regenerated.
      (β) PHASE_FAILURE_MODE_SCOPE module-level frozenset added to each
          runner (concept_validation.py / full_dataset_train.py /
          streaming_engine_export.py). Activation tripwire step 7 now
          performs a deterministic set-subset check against this
          constant, NOT prose-docstring parsing. v1.4 sets all three
          constants to the full four-tag plan-§0.2-row-1 set; future
          per-phase narrowing is a one-line constant edit.
      (γ) Path regex tightened — also rejects backslash / Windows drive
          prefixes (C:\\...), not only leading '/' and '..'. Aligns the
          schema with the "repo-relative POSIX" stated convention.

    v1.3 closed C3 iter-3 ADDITIONAL-FINDINGS against v1.2:
      (i)   Activation schema path semantics ENFORCED in JSON Schema, not
            prose: selected_detector_artifact_path regex rejects absolute
            paths and '..' segments and requires '.engine' suffix;
            replay_evidence_path regex rejects absolute and '..'.
      (ii)  Plan §0.2 row 1 four-tag set restored — `far_distance_miss`
            added to approved_failure_mode_tags enum (was missing in
            v1.2). Tag mapping documented in schema description: 小目标=
            small_target_miss, 远距离=far_distance_miss, 遮挡=occluded_
            miss, 运动模糊=motion_blur.
      (iii) THREE-WAY SHA equality contract spelled out: activation_sha
            == sidecar.engine_sha256 == computed_sha. Pairwise-mismatch
            diagnostics in schema description (stale activation / stale
            sidecar / changed engine / full re-activation).
      (iv)  Empty fenced ``` pair at plan §1.1 line 79-80 removed (NIT).

    v1.2 closed C3 iter-2 ADDITIONAL-FINDINGS against v1.1:
      (a) Plan-authority conflict closed — plan §1.1 carries a dated
          2026-05-09 v1.1 amendment supersesing the bidirectional `c2`
          line; README header declares precedence (README > amended
          plan §1.1 > plan v1 prose).
      (b) Activation tripwire schema-enforced via
          `scripts/_tsm_activation_schema.json` (LANDED).
      (c) Residual `HG_Stage 3` wording in modules/__init__.py replaced
          with `config stage4 / HGNetv2.stages[3]`.
      (d) Stray `apples-to-\napples` line break in runners/__init__.py
          rejoined.

    v1.1 hardened four MAJOR scaffold issues caught by C3 iter-1:
      (1) c2 zeroed in both train and inference (causal end-to-end);
      (2) DEIM stage scope rewritten with three unambiguous identifiers
          (config stage2/3/4, self.stages[1]/[2]/[3], return_idx=[1,2,3]);
      (3) GO-LSD claim rewritten — "no separate wiring required",
          NOT "loss/gradient-path independent";
      (4) DEIM `HGNetv2.pretrained=True` upstream default explicitly
          countered in the runner contract — `--pretrained-init scratch`
          alone is insufficient for DEIM TSM.
    Plus three MINOR: DEIM-S relabeled HGNetv2-B0 (per current configs);
    sidecar enforcement ownership clarified (export script writes,
    Phase 1-C runner validates post-export); activation gate gets a
    tripwire artifact (`runs/_tsm_activation.json` — schema landed at v1.2).

    Plan stays at v1 with a v1.1 §1.1 amendment; this README at v1.4.
    Activation schema at v1.1 (was v1.0 in v1.3 README; bumped at v1.4).
"""
