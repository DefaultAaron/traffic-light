"""Clip-of-N dataloader collator (TSM-specific dataloader interface).

Wraps the main-track-selected detector's native dataloader (Ultralytics YOLO
or DEIM) into clip-of-N=4 batches: clip-internal frame order preserved, clip
boundaries randomized. Only one collator file because the wrapping is generic
— the underlying dataloader is detector-specific (DEIM uses COCO-style with
its own augmentation pipeline; Ultralytics uses Mosaic + its own).

------------------------------------------------------------------------------
Dataloader contract (when collator lands — Phase 1-A scheduling)
------------------------------------------------------------------------------
Input:
    Native (per-frame) dataloader yielding (image, targets, image_id, ...)
    R2 video manifest grouping image_ids by source video (continuous 30 fps,
    track IDs persist across frames).

Output:
    Clip batch of shape (B, T, C, H, W) where T = clip_size (default 4) and
    intra-clip frames are CONSECUTIVE in the source video. Clip-level shuffle
    is independent of intra-clip order.

Constraints (§1.4 + §4.1):
    - Clips MUST originate from a single source video — no clip spanning two
      videos.
    - Clip-batch shrinks effective batch_size by T → §1.6 risk row 3 says
      scale lr by sqrt(bs) and accept longer epoch. The collator does NOT
      compensate; that's the runner's hyperparam responsibility.
    - Augmentation MUST be applied CONSISTENTLY across the T frames of a
      clip — sampling the same crop box, the same source-frame indices, the
      same flip flag, the same color-jitter sample. This is the same Wang
      2022 augmentation-consistency rule that KD enforces teacher-side; here
      it's enforced clip-internally for student-only training. Augmentation
      that breaks clip consistency silently destroys the temporal signal TSM
      is trying to learn.

      Detector-specific augmentation sources to clamp:
        Ultralytics (YOLO26 / YOLOv13): Mosaic, MixUp, HSV-jitter, flip,
            RandomAffine. Mosaic source-frame indices and the per-tile crop
            transform are the easy-to-miss surfaces.
        DEIM (`DEIM/engine/data/transforms/`): RandomIoUCrop, RandomZoomOut,
            RandomHorizontalFlip, ConvertPILImage, plus the optional
            ColorJitter / Resize chain. RandomIoUCrop's IoU-conditioned crop
            box and RandomZoomOut's fill-pad coordinates are the
            easy-to-miss surfaces.
      The collator's clip-consistency hook MUST cover both pipelines when
      `--base-detector` selects the corresponding upstream.
    - val / test loader MUST NOT clip-shuffle and MUST NOT augment — clip
      ordering is the clip itself, intra-clip frames are consecutive in
      arrival order. Eval clip selection is deterministic.

Train / val split (§1.4 + §4.1):
    - Split by VIDEO FILE, not by clip. Cross-video temporal pattern leakage
      is the canonical R2 mistake to avoid here.

Single file in this subpackage:
    clip_collator.py   collator wrapping the native dataloader
"""
