"""Copy-paste augmentation module.

Implements the bbox-level copy-paste of rare-class instances onto any
background image with a y-center mask constraint. Plan §3.1 calls this
out as a forced per-batch guarantee: at least K instances of each
rare class per batch, drawn from a pre-curated source pool.

Plan-pinned constraints (load-bearing — read before changing):

* **y-center mask** (plan §3.5 risk): the paste position's bbox y-center
  MUST lie in the upper ``y_center_max_frac`` of the frame (default
  0.33, i.e. upper third). Without this, copy-paste produces
  "lights floating in the sky" samples that hurt validation more than
  they help. b-stage MAY tighten the constraint per dataset analysis;
  loosening REQUIRES re-running the §三 adversarial loop.

* **fliplr=0 lock**: arrows (``redLeft`` / ``greenLeft`` / ``redRight`` /
  ``greenRight`` / ``forwardRed`` / ``forwardGreen``) cannot be horizontally
  flipped. Per-project memory `feedback_augmentation`. The runner verifies
  the active training config has ``fliplr: 0.0`` BEFORE training starts;
  mismatch is a hard fail rather than a silent degradation.

* **Mosaic interaction**: copy-paste applied AFTER mosaic stitching
  produces unstable instance density. Default ``required_mosaic_lock``
  is False (allow Ultralytics default mosaic); b-stage measures and may
  tighten if instability surfaces.

Scaffold (a-stage): public API surface frozen here as type-checked
``NotImplementedError`` stubs. b-stage replaces bodies with the actual
paste op (Ultralytics callback or DEIM dataloader hook); no API change
permitted without re-running the §三 adversarial loop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np  # noqa: F401  # b-stage uses np in apply()


@dataclass(frozen=True)
class CopyPasteConfig:
    """Pre-committed knobs for copy-paste augmentation.

    Mirrors the ``copy_paste:`` block of ``configs/copy_paste_balance.yaml``
    1:1 so a YAML load → dataclass build is a one-liner once b-stage adds
    the loader. Frozen so any accidental mid-run mutation surfaces as
    ``FrozenInstanceError`` rather than a silent augmentation shift.

    YAML→dataclass adapter note: ``paste_source_class_ids`` is YAML
    ``list[int]`` and the loader MUST coerce via ``tuple(int(x) for x in
    raw)`` before constructing this dataclass.
    """

    num_classes: int
    probability: float                            # Ultralytics `copy_paste=`
    y_center_max_frac: float | None              # plan §3.5 mask; None disables
    paste_source_class_ids: tuple[int, ...]      # rare-class pool
    min_per_batch_K: int                         # 0 = no per-batch guarantee
    required_fliplr: float = 0.0                 # plan §3.5 hard pin
    required_mosaic_lock: bool = False           # b-stage may tighten

    def __post_init__(self) -> None:
        # System-boundary validation — same discipline as
        # HmmYamlConfig.__post_init__ / TransitionConfig.__post_init__:
        # type-with-bool-exclusion + finite-check + range-check across all
        # numeric fields, even when the default looks safe. YAML load → typed
        # construction is the only validation gate before b-stage consumes the
        # config; defending here covers direct-construction tests too.
        if self.num_classes is None:
            raise ValueError("num_classes must be set explicitly; got None")
        if not isinstance(self.num_classes, int) or isinstance(self.num_classes, bool):
            raise ValueError(
                f"num_classes must be int; got "
                f"{type(self.num_classes).__name__}={self.num_classes!r}"
            )
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be > 0; got {self.num_classes}")
        # `True in (0.0, 1.0)` is True (bool is int subclass) — exclude bool.
        if not isinstance(self.probability, float) or isinstance(self.probability, bool):
            raise ValueError(
                f"probability must be float; got "
                f"{type(self.probability).__name__}={self.probability!r}"
            )
        if not math.isfinite(self.probability):
            raise ValueError(f"probability must be finite; got {self.probability!r}")
        if not (0.0 <= self.probability <= 1.0):
            raise ValueError(
                f"probability must be in [0, 1]; got {self.probability}"
            )
        if self.y_center_max_frac is not None:
            if not isinstance(self.y_center_max_frac, float) or isinstance(self.y_center_max_frac, bool):
                raise ValueError(
                    f"y_center_max_frac must be None or float; got "
                    f"{type(self.y_center_max_frac).__name__}={self.y_center_max_frac!r}"
                )
            if not math.isfinite(self.y_center_max_frac):
                raise ValueError(
                    f"y_center_max_frac must be finite; got {self.y_center_max_frac!r}"
                )
            if not (0.0 < self.y_center_max_frac <= 1.0):
                raise ValueError(
                    f"y_center_max_frac must be in (0, 1]; got {self.y_center_max_frac}"
                )
        if not isinstance(self.paste_source_class_ids, tuple):
            raise ValueError(
                f"paste_source_class_ids must be tuple; got "
                f"{type(self.paste_source_class_ids).__name__}={self.paste_source_class_ids!r} "
                f"(loader must coerce YAML list → tuple before construction)"
            )
        for cid in self.paste_source_class_ids:
            if not isinstance(cid, int) or isinstance(cid, bool):
                raise ValueError(
                    f"paste_source_class_ids entries must be int; got "
                    f"{type(cid).__name__}={cid!r}"
                )
            if not (0 <= cid < self.num_classes):
                raise ValueError(
                    f"paste_source_class_id {cid} out of range "
                    f"[0, {self.num_classes})"
                )
        # C3 iter-3 NEW-MAJOR (paste-pool dedup) 2026-05-09: reject
        # duplicates so a YAML typo doesn't silently pile per-class paste
        # pressure (e.g. ``paste_source_class_ids: [3, 3, 5]`` would
        # double-bias class 3 in any sampling that respects multiplicity).
        if len(set(self.paste_source_class_ids)) != len(self.paste_source_class_ids):
            raise ValueError(
                f"paste_source_class_ids must contain no duplicates; got "
                f"{self.paste_source_class_ids}"
            )
        if not isinstance(self.min_per_batch_K, int) or isinstance(self.min_per_batch_K, bool):
            raise ValueError(
                f"min_per_batch_K must be int; got "
                f"{type(self.min_per_batch_K).__name__}={self.min_per_batch_K!r}"
            )
        if self.min_per_batch_K < 0:
            raise ValueError(
                f"min_per_batch_K must be >= 0; got {self.min_per_batch_K}"
            )
        # required_fliplr is the value the runner verifies against the active
        # training config. Plan §3.5 pins this to 0.0; any non-zero value is
        # a contract violation against feedback_augmentation memory.
        if not isinstance(self.required_fliplr, float) or isinstance(self.required_fliplr, bool):
            raise ValueError(
                f"required_fliplr must be float; got "
                f"{type(self.required_fliplr).__name__}={self.required_fliplr!r}"
            )
        if self.required_fliplr != 0.0:
            raise ValueError(
                f"required_fliplr is plan-pinned to 0.0 (arrows can't be horizontally "
                f"flipped — see feedback_augmentation memory); got {self.required_fliplr}"
            )
        if not isinstance(self.required_mosaic_lock, bool):
            raise ValueError(
                f"required_mosaic_lock must be bool; got "
                f"{type(self.required_mosaic_lock).__name__}={self.required_mosaic_lock!r}"
            )
        # C3 iter-3 NEW-MAJOR (active-but-empty pool) 2026-05-09: an empty
        # paste_source_class_ids tuple combined with probability > 0 OR
        # min_per_batch_K > 0 makes copy-paste a silent no-op (the
        # augmenter has nothing to paste from), which would let a
        # cp_only / cp_balanced arm look legitimate without exercising
        # the planned rare-class injection. Reject the inconsistent
        # combination at construction.
        if (self.probability > 0.0 or self.min_per_batch_K > 0) and not self.paste_source_class_ids:
            raise ValueError(
                f"paste_source_class_ids must be non-empty when copy-paste is "
                f"active (probability={self.probability}, "
                f"min_per_batch_K={self.min_per_batch_K}); an empty source pool "
                f"with active augmentation is a silent no-op that would "
                f"corrupt the ablation arm without surfacing as a config error"
            )


class CopyPasteAugment:
    """Stateful copy-paste augmenter.

    Construction binds a ``CopyPasteConfig`` and an instance source pool
    (rare-class crops from an offline-curated bank). ``apply()`` is called
    per-image (or per-batch in the per-batch-K mode) and returns a paired
    ``(image, labels)`` with synthetic instances stamped according to the
    config's mask + probability rules.

    Source-pool note: b-stage decides whether the source pool is loaded
    eagerly (in-memory crops) or lazily (filesystem-pathed). a-stage
    leaves the constructor signature minimal so b-stage can choose without
    re-running the adversarial loop on the public API.

    Scope fence: this class is ablation-only. ``inference.cpp`` /
    ``inference.tracker`` / ``inference.demo`` MUST NOT import it. The
    augmenter runs at training time only.
    """

    def __init__(self, config: CopyPasteConfig):
        self._config = config

    @property
    def config(self) -> CopyPasteConfig:
        return self._config

    def apply(
        self,
        *,
        image: np.ndarray,
        labels: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply copy-paste augmentation to one (image, labels) pair.

        b-stage will implement; signature reserved here so the trainer
        callback / dataloader hook can be written against a stable API.

        Args:
            image: ``(H, W, 3)`` uint8 RGB image (Ultralytics convention)
                or ``(C, H, W)`` torch.Tensor (DEIM convention) — b-stage
                MAY add a runtime check + dispatch on ndim/dtype.
            labels: ``(N, 5)`` array ``[class_id, cx, cy, w, h]`` with
                cx/cy/w/h normalized to ``[0, 1]`` (YOLO convention).
            rng: numpy random Generator for deterministic augmentation
                (training-callback-passed, NOT a fresh ``np.random.default_rng()``
                — that would mask seed-determinism bugs).

        Returns:
            ``(augmented_image, augmented_labels)`` — labels grow by however
            many synthetic instances were stamped (0 if probability rolled
            against the augmentation, possibly more than 0 in per-batch-K
            mode).

        Raises:
            ValueError: shape/range violations on inputs, OR mask constraint
                produced no valid paste position (b-stage logs at WARNING
                and returns the unmodified pair rather than raising — the
                augmenter is best-effort, not a correctness gate).
            NotImplementedError: a-stage scaffold.
        """
        raise NotImplementedError("b-stage")
