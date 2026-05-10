"""Hard-negative candidate miner — config + mine() stub (a-stage).

Plan §4.1 mechanism (verbatim):

    "用 R1 baseline 在 demo8 / 11 / 部分 demo13 跑推理，收集 FP 帧
    (yellow / green 框打在非交通灯目标上的帧), 加入训练集作为纯背景图
    (无标注) 或带 ignore 区域. Ultralytics 已支持 bg/ 目录格式; DEIM
    走 COCO-style empty-image 写入."

This module owns the candidate-collection step ONLY. Outputs a typed
``MiningCandidateManifest`` (one row per FP frame) which then feeds
the ``verifier`` module's human-verification protocol. The frozen eval
manifest in ``data/eval_manifest.py`` is a SEPARATE downstream artifact
— it's the post-verification, post-curation fixed evaluation set.

Scope fence (load-bearing):
  * NEVER bundle mined candidates directly into the training bg/ dir
    without human verification (plan §4.5 risk: mis-classifying a real
    but hard-to-detect light as a hard negative would silently destroy
    recall on that class).
  * NEVER recompute the eval manifest from mined candidates after
    training has started (plan §4.7 anti-gaming: the FP-denominator
    must be locked at fit time).

Scaffold (a-stage): API only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from components.hard_negative_mining._internals import (
    PLAN_LOCKED_CONFIDENCE_THRESHOLD,
    PLAN_LOCKED_NMS_IOU_THRESHOLD,
    PLAN_MINING_SOURCES,
)


@dataclass(frozen=True)
class MiningSourceVideo:
    """One demo video / R2-self capture to run R1 baseline on.

    The source list is plan-pinned for R2 ablation (demo8/11/13 + R2
    self-collected hard scenes). Adding a new source requires a fresh
    §四 adversarial loop because new sources change the candidate
    population and could silently game the FP-drop denominator if
    introduced post-fit.

    Note (B2 review I8 2026-05-10): ``_ALLOWED_SOURCE_LABELS`` is a
    STRICT SUBSET of ``FrozenEvalManifestEntry._ALLOWED_LABEL_SOURCES``.
    The manifest superset additionally includes ``"real_light_set"``
    — the recall-denominator population — which is INTENTIONALLY
    excluded here because real_light_set frames are not mined; they
    feed the manifest's recall computation and are separately
    annotated upstream of this pipeline.
    """

    video_path: Path
    source_label: str

    _ALLOWED_SOURCE_LABELS: ClassVar[tuple[str, ...]] = PLAN_MINING_SOURCES

    def __post_init__(self) -> None:
        if not isinstance(self.video_path, Path):
            raise ValueError(
                f"video_path must be Path; got "
                f"{type(self.video_path).__name__}={self.video_path!r} "
                f"(loader / CLI must coerce str → pathlib.Path before construction)"
            )
        if not isinstance(self.source_label, str):
            raise ValueError(
                f"source_label must be str; got "
                f"{type(self.source_label).__name__}"
            )
        if self.source_label not in self._ALLOWED_SOURCE_LABELS:
            raise ValueError(
                f"source_label must be one of {self._ALLOWED_SOURCE_LABELS}; "
                f"got {self.source_label!r} (plan §4.1 source set is "
                f"locked — adding a new source requires re-running the "
                f"§四 adversarial loop)"
            )


@dataclass(frozen=True)
class HardNegativeMinerConfig:
    """Configuration for the candidate-mining pass.

    Plan §4.1 thresholds (FROZEN — same values as the eval manifest's
    locked thresholds; mirroring keeps fit-time and eval-time
    consistent):
      * ``confidence_threshold = 0.25``
      * ``nms_iou_threshold = 0.5``

    These are intentionally NOT exposed via YAML; they are part of the
    plan-locked contract. Loader rejects deviation.

    ``baseline_weights_path`` is the R1 baseline checkpoint used for
    inference. ``output_candidates_path`` is the path the mining pass
    writes to (NOT an eval manifest — a downstream candidate manifest
    which then feeds verification).

    Field naming alignment (B2 review I5 2026-05-10 + C3 iter-1 PROBLEM
    2026-05-10): ``output_candidates_path`` matches
    ``HardNegativeMiningYamlConfig.output_candidates_path`` (both fields
    use the SAME name post-rename). The YAML loader maps the legacy
    ``candidates_output_path`` key (if present) for a one-version
    transition; b-stage will drop the old key.
    """

    sources: tuple[MiningSourceVideo, ...]
    baseline_weights_path: Path
    output_candidates_path: Path
    confidence_threshold: float
    nms_iou_threshold: float
    # Plan §4.5 risk: limit y-center is for copy_paste, not hard-neg —
    # hard-neg has no spatial filter (anywhere a baseline FP appears is
    # by construction a "this is not a light" example).

    _LOCKED_CONFIDENCE_THRESHOLD: ClassVar[float] = PLAN_LOCKED_CONFIDENCE_THRESHOLD
    _LOCKED_NMS_IOU_THRESHOLD: ClassVar[float] = PLAN_LOCKED_NMS_IOU_THRESHOLD

    def __post_init__(self) -> None:
        if not isinstance(self.sources, tuple):
            raise ValueError(
                f"sources must be tuple; got {type(self.sources).__name__} "
                f"(loader must coerce list → tuple before construction)"
            )
        if not self.sources:
            raise ValueError(
                "sources must be non-empty (a mining run with no input "
                "videos cannot produce candidates)"
            )
        for i, src in enumerate(self.sources):
            if not isinstance(src, MiningSourceVideo):
                raise ValueError(
                    f"sources[{i}] must be MiningSourceVideo; got "
                    f"{type(src).__name__}"
                )
        # Reject duplicate (video_path, source_label) pairs — running R1
        # twice on the same video would silently double-count its FP
        # contribution if the dedupe at output-write time were skipped.
        pairs = [(s.video_path, s.source_label) for s in self.sources]
        if len(set(pairs)) != len(pairs):
            raise ValueError(
                f"sources must contain no duplicate (video_path, source_label) "
                f"pairs; got {pairs}"
            )
        if not isinstance(self.baseline_weights_path, Path):
            raise ValueError(
                f"baseline_weights_path must be Path; got "
                f"{type(self.baseline_weights_path).__name__}="
                f"{self.baseline_weights_path!r}"
            )
        if not isinstance(self.output_candidates_path, Path):
            raise ValueError(
                f"output_candidates_path must be Path; got "
                f"{type(self.output_candidates_path).__name__}="
                f"{self.output_candidates_path!r}"
            )
        # B2 review I4 2026-05-10: reject path collisions across input/output
        # fields. ``output_candidates_path == baseline_weights_path`` would
        # clobber the model checkpoint mid-write; ``output_candidates_path``
        # equal to any source video path would clobber the input.
        if self.output_candidates_path == self.baseline_weights_path:
            raise ValueError(
                f"output_candidates_path must differ from baseline_weights_path; "
                f"got {self.output_candidates_path} (writing the candidates "
                f"JSON over the model checkpoint would clobber the baseline)"
            )
        for i, src in enumerate(self.sources):
            if self.output_candidates_path == src.video_path:
                raise ValueError(
                    f"output_candidates_path must differ from every source "
                    f"video path; got collision with sources[{i}].video_path="
                    f"{src.video_path}"
                )
        # Locked thresholds — bool-exclusion + finite + equality-with-pin.
        if not isinstance(self.confidence_threshold, float) or isinstance(
            self.confidence_threshold, bool
        ):
            raise ValueError(
                f"confidence_threshold must be float; got "
                f"{type(self.confidence_threshold).__name__}="
                f"{self.confidence_threshold!r}"
            )
        if not math.isfinite(self.confidence_threshold):
            raise ValueError(
                f"confidence_threshold must be finite; got "
                f"{self.confidence_threshold!r}"
            )
        if self.confidence_threshold != self._LOCKED_CONFIDENCE_THRESHOLD:
            raise ValueError(
                f"confidence_threshold must equal "
                f"{self._LOCKED_CONFIDENCE_THRESHOLD} (plan §4.1 lock); "
                f"got {self.confidence_threshold} — fit-time and eval-time "
                f"thresholds must be identical to keep the FP comparison "
                f"meaningful"
            )
        if not isinstance(self.nms_iou_threshold, float) or isinstance(
            self.nms_iou_threshold, bool
        ):
            raise ValueError(
                f"nms_iou_threshold must be float; got "
                f"{type(self.nms_iou_threshold).__name__}="
                f"{self.nms_iou_threshold!r}"
            )
        if not math.isfinite(self.nms_iou_threshold):
            raise ValueError(
                f"nms_iou_threshold must be finite; got "
                f"{self.nms_iou_threshold!r}"
            )
        if self.nms_iou_threshold != self._LOCKED_NMS_IOU_THRESHOLD:
            raise ValueError(
                f"nms_iou_threshold must equal "
                f"{self._LOCKED_NMS_IOU_THRESHOLD} (plan §4.1 lock); "
                f"got {self.nms_iou_threshold}"
            )


def mine_candidates(config: HardNegativeMinerConfig) -> Path:
    """Run R1 baseline on configured sources; write candidate manifest.

    Output schema (b-stage; one row per FP frame; B2 review I3 2026-05-10
    added ``data_yaml_sha256`` to lock class-set provenance, closing
    the §4.7 anti-gaming surface where data.yaml v1 mining was applied
    to v2 training):

        {
          "schema_version": "1",
          "baseline_weights_sha256": str,   # 64-char lowercase hex
          "data_yaml_sha256": str,          # 64-char lowercase hex; class-set provenance
          "confidence_threshold": 0.25,
          "nms_iou_threshold": 0.5,
          "candidates": [
            {
              "image_sha256": str,
              "image_relpath": str,
              "source_label": "demo8" | "demo11" | "demo13" | "r2_self",
              "baseline_detection_class": str,    # "yellow" / "green" / ...
              "baseline_detection_score": float,
              "baseline_detection_bbox_xyxy": [x1, y1, x2, y2]
            },
            ...
          ]
        }

    The output is INPUT to the verifier module — every row must be
    confirmed by a human reviewer before it can be added to the
    training bg/ directory. The runner cross-validates
    ``data_yaml_sha256`` against every per-arm eval JSON's
    ``data_yaml_sha256`` to prevent class-set drift between mining
    time and training time.

    Args:
        config: ``HardNegativeMinerConfig`` resolved from CLI / YAML.

    Returns:
        Path to the written candidates JSON
        (== ``config.output_candidates_path``).

    Raises:
        FileNotFoundError: any source video / baseline weights missing.
        NotImplementedError: a-stage scaffold.
    """
    raise NotImplementedError("b-stage")
