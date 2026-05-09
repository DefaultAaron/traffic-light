"""Per-class instance counts → ``configs/data_R2_class_weights.yaml``.

Walks a YOLO/DEIM data manifest (``data.yaml`` + label files) and tallies
the number of instances per class on the TRAIN split only (per Cui 2019
§3 — weights are computed from the training distribution, not validation).

Output schema for ``configs/data_R2_class_weights.yaml``:

    schema_version: "1"
    data_yaml_sha256: <hex>     # source-of-truth hash of the data.yaml
    train_split: "train"        # which split was tallied
    num_classes: <int>
    class_names: [<str>, ...]   # length = num_classes
    counts: [<int>, ...]        # length = num_classes; per-class instance count
    rare_class_threshold: 30    # plan default; classes below this are "rare"
    rare_class_ids: [<int>, ...]  # derived: indices where counts[i] < threshold

The YAML is a fitter output, not a config knob — it's regenerated whenever
the training manifest changes. ``ClassBalanceWeights.from_counts`` consumes
the ``counts`` array from this file.

Scaffold (a-stage): API surface only. b-stage implements the walk +
serialize.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np  # noqa: F401  # b-stage uses np in to_numpy() / fitter


@dataclass(frozen=True)
class ClassCountsTable:
    """Tallied per-class instance counts from one data manifest split.

    Validation (``__post_init__``): every ``counts[i] >= 0``, length matches
    ``num_classes``, ``len(class_names) == num_classes``, ``rare_class_ids``
    is a subset of ``range(num_classes)``.
    """

    num_classes: int
    class_names: tuple[str, ...]
    counts: tuple[int, ...]
    train_split: str
    rare_class_threshold: int
    rare_class_ids: tuple[int, ...]
    data_yaml_sha256: str

    def __post_init__(self) -> None:
        if self.num_classes is None:
            raise ValueError("num_classes must be set explicitly; got None")
        if not isinstance(self.num_classes, int) or isinstance(self.num_classes, bool):
            raise ValueError(
                f"num_classes must be int; got "
                f"{type(self.num_classes).__name__}={self.num_classes!r}"
            )
        if self.num_classes <= 0:
            raise ValueError(f"num_classes must be > 0; got {self.num_classes}")
        if not isinstance(self.class_names, tuple):
            raise ValueError(
                f"class_names must be tuple; got {type(self.class_names).__name__}"
            )
        if len(self.class_names) != self.num_classes:
            raise ValueError(
                f"class_names length ({len(self.class_names)}) must equal "
                f"num_classes ({self.num_classes})"
            )
        for i, name in enumerate(self.class_names):
            if not isinstance(name, str):
                raise ValueError(
                    f"class_names[{i}] must be str; got {type(name).__name__}={name!r}"
                )
        if not isinstance(self.counts, tuple):
            raise ValueError(f"counts must be tuple; got {type(self.counts).__name__}")
        if len(self.counts) != self.num_classes:
            raise ValueError(
                f"counts length ({len(self.counts)}) must equal num_classes "
                f"({self.num_classes})"
            )
        for i, c in enumerate(self.counts):
            if not isinstance(c, int) or isinstance(c, bool):
                raise ValueError(
                    f"counts[{i}] must be int; got {type(c).__name__}={c!r}"
                )
            if c < 0:
                raise ValueError(f"counts[{i}] must be >= 0; got {c}")
        # C3 iter-3 NEW-MAJOR (train-only) 2026-05-09: Cui 2019 §3 mandates
        # weights computed from the TRAIN split only — using val/test
        # produces a plausible but wrong weight vector that silently
        # changes the ablation arm behavior. The module contract +
        # estimate_from_dataset() default already pin "train"; mirror
        # the constraint here so a hand-edited weights file with
        # train_split="val" cannot ship.
        if not isinstance(self.train_split, str):
            raise ValueError(
                f"train_split must be str; got {type(self.train_split).__name__}"
            )
        if self.train_split != "train":
            raise ValueError(
                f"train_split must be 'train'; got {self.train_split!r} "
                f"(class-balance weights are derived from the train split per "
                f"Cui 2019 §3 — using val/test would silently corrupt the "
                f"ablation arm with the wrong distribution)"
            )
        if not isinstance(self.rare_class_threshold, int) or isinstance(self.rare_class_threshold, bool):
            raise ValueError(
                f"rare_class_threshold must be int; got "
                f"{type(self.rare_class_threshold).__name__}={self.rare_class_threshold!r}"
            )
        if self.rare_class_threshold < 0:
            raise ValueError(
                f"rare_class_threshold must be >= 0; got {self.rare_class_threshold}"
            )
        if not isinstance(self.rare_class_ids, tuple):
            raise ValueError(
                f"rare_class_ids must be tuple; got {type(self.rare_class_ids).__name__}"
            )
        for i, cid in enumerate(self.rare_class_ids):
            if not isinstance(cid, int) or isinstance(cid, bool):
                raise ValueError(
                    f"rare_class_ids[{i}] must be int; got "
                    f"{type(cid).__name__}={cid!r}"
                )
            if not (0 <= cid < self.num_classes):
                raise ValueError(
                    f"rare_class_ids[{i}]={cid} out of range [0, {self.num_classes})"
                )
        # C3 iter-3 NEW-MAJOR (rare-id derivation) 2026-05-09: rare_class_ids
        # MUST equal the derived set ``{i : counts[i] < rare_class_threshold}``
        # in ascending order. Without this check, a stale or hand-edited
        # rare_class_ids could disagree with `counts` and corrupt downstream
        # copy-paste source selection + rare-class decision reporting.
        # Naturally rejects duplicates (the derived set is sorted, unique).
        expected_rare = tuple(
            i for i, c in enumerate(self.counts) if c < self.rare_class_threshold
        )
        if self.rare_class_ids != expected_rare:
            raise ValueError(
                f"rare_class_ids must equal the derived set "
                f"{{i : counts[i] < rare_class_threshold}} in ascending order; "
                f"got rare_class_ids={self.rare_class_ids}, expected={expected_rare} "
                f"(rare set drift would corrupt downstream copy-paste source "
                f"selection and rare-class decision reporting)"
            )
        # data_yaml_sha256 is a 64-char LOWERCASE hex string (sha256 of the
        # source data.yaml). Empty string is rejected here — the runner MUST
        # compute the hash; a missing one would let a stale weights file ship.
        # C3 iter-2 NEW-MINOR 2026-05-09: lowercase enforcement aligns this
        # check with ArmMetrics's manifest-hash check (`_is_hex_sha256`) and
        # the schema's `^[0-9a-f]{64}$` pattern. Without lowercase pinning,
        # an uppercase hex would pass here but fail downstream equality
        # checks against ``hashlib.sha256(...).hexdigest()`` (which always
        # returns lowercase).
        if not isinstance(self.data_yaml_sha256, str):
            raise ValueError(
                f"data_yaml_sha256 must be str; got {type(self.data_yaml_sha256).__name__}"
            )
        if len(self.data_yaml_sha256) != 64:
            raise ValueError(
                f"data_yaml_sha256 must be 64 hex chars; got len={len(self.data_yaml_sha256)}"
            )
        try:
            int(self.data_yaml_sha256, 16)
        except ValueError as exc:
            raise ValueError(
                f"data_yaml_sha256 must be valid hex; got {self.data_yaml_sha256!r}"
            ) from exc
        if self.data_yaml_sha256 != self.data_yaml_sha256.lower():
            raise ValueError(
                f"data_yaml_sha256 must be LOWERCASE hex; got {self.data_yaml_sha256!r} "
                f"(hashlib.sha256().hexdigest() returns lowercase; uppercase would "
                f"fail downstream equality checks)"
            )

    def to_numpy(self) -> np.ndarray:
        """Return ``counts`` as an ``int64`` numpy array of shape (C,).

        Convenience for ``ClassBalanceWeights.from_counts`` consumers; b-stage
        implements.

        Raises:
            NotImplementedError: a-stage scaffold.
        """
        raise NotImplementedError("b-stage")


def estimate_from_dataset(
    *,
    data_yaml_path: Path,
    train_split: str = "train",
    rare_class_threshold: int = 30,
) -> ClassCountsTable:
    """Walk a YOLO/DEIM data manifest and tally per-class instance counts.

    Args:
        data_yaml_path: path to the active ``data/traffic_light.yaml`` (or
            DEIM equivalent). Loader reads ``names`` (class list) +
            ``train`` (split path) keys.
        train_split: which split key in ``data.yaml`` to consume. Defaults
            to ``"train"`` per Cui 2019 §3 convention. Loader rejects any
            other value silently producing a non-train tally would corrupt
            the weight vector.
        rare_class_threshold: instance count below which a class is flagged
            "rare" for the §三 decision rule. Plan default 30 (matches the
            R2 precision plan's ``full_val_support`` floor).

    Returns:
        ``ClassCountsTable`` ready to be serialized into
        ``configs/data_R2_class_weights.yaml``.

    Raises:
        FileNotFoundError: ``data_yaml_path`` or any referenced label
            directory does not exist.
        ValueError: malformed YAML, mismatched class IDs, etc.
        NotImplementedError: a-stage scaffold.
    """
    raise NotImplementedError("b-stage")
