"""Frozen evaluation manifest dataclass for §四 hard-negative mining.

Plan §4.7 anti-gaming requirement (verbatim):

    "冻结评估 manifest: 训练前冻结 runs/_hard_negative_eval_manifest.json
    (demo8/11/13 + R2 自采难场景帧 + 真实灯标注帧的混合 manifest), 含
    image SHA256 + 标注源 + 评估阈值 (confidence ≥ 0.25, NMS IoU = 0.5;
    与 R2 精度奇偶 eval-parity gate 同款). 评估时不得改 manifest 或阈值
    -- 保护 FP 下降率 denominator 不被换 frame 子集 game."

This dataclass is the typed in-memory representation. The on-disk file
remains the canonical artifact; this dataclass is built ONLY by the
loader stub in this module. The d-stage ablation runner reads the
manifest exactly once per ablation run and validates that the
``manifest_sha256`` recorded in every per-arm eval JSON matches the
file's actual hash — this closes the FP-denominator gaming loop the
plan §4.7 prose describes.

Sister-file precedent (internal B2 audit trail; b-stage authors can
ignore): ``components/copy_paste_balance/data/class_weights.py``
(locked iter-11 2026-05-09). Same coercion-at-loader-only discipline:
``__post_init__`` does pure value checks, NO filesystem I/O; the
loader is responsible for SHA256 computation and existence checks.

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
    PLAN_MANIFEST_LABEL_SOURCES,
    is_hex_sha256,
)


@dataclass(frozen=True)
class FrozenEvalManifestEntry:
    """One image-level row in ``runs/_hard_negative_eval_manifest.json``.

    The manifest is mixed: real-light frames (used for recall computation)
    + background frames (used for FP computation). The ``has_real_light``
    flag separates the two populations at eval time so the runner can
    compute recall on real-light frames only and FP on background frames
    only WITHOUT re-reading per-image labels.

    Note on ``label_source``: includes the value ``"real_light_set"``
    that ``MiningSourceVideo._ALLOWED_SOURCE_LABELS`` does NOT — those
    frames feed the manifest's recall denominator and are separately
    annotated, not mined.
    """

    image_sha256: str
    image_relpath: str
    label_source: str            # one of PLAN_MANIFEST_LABEL_SOURCES
    has_real_light: bool

    _ALLOWED_LABEL_SOURCES: ClassVar[tuple[str, ...]] = PLAN_MANIFEST_LABEL_SOURCES

    def __post_init__(self) -> None:
        if not is_hex_sha256(self.image_sha256):
            raise ValueError(
                f"image_sha256 must be 64-char lowercase hex; got "
                f"{self.image_sha256!r}"
            )
        if not isinstance(self.image_relpath, str):
            raise ValueError(
                f"image_relpath must be str; got "
                f"{type(self.image_relpath).__name__}={self.image_relpath!r}"
            )
        if not self.image_relpath:
            raise ValueError("image_relpath must be non-empty")
        # B2 review C3 (null-byte injection) 2026-05-10: a manifest path
        # with an embedded null byte would silently break the SHA256
        # round-trip an attacker could exploit if they had write access
        # to the manifest file. The §4.7 manifest is the FP-denominator
        # anti-gaming artifact, so this is treated as CRITICAL even
        # though the likelihood is low.
        if "\x00" in self.image_relpath:
            raise ValueError(
                f"image_relpath must not contain a null byte; got "
                f"{self.image_relpath!r} (the §4.7 manifest is the "
                f"FP-denominator anti-gaming artifact; null-byte injection "
                f"would silently break the SHA256 round-trip)"
            )
        # Reject absolute paths — the manifest is intended to be relocatable
        # across machines (workstation vs Orin) and absolute paths break that.
        if Path(self.image_relpath).is_absolute():
            raise ValueError(
                f"image_relpath must be relative; got absolute path "
                f"{self.image_relpath!r}"
            )
        if not isinstance(self.label_source, str):
            raise ValueError(
                f"label_source must be str; got "
                f"{type(self.label_source).__name__}={self.label_source!r}"
            )
        if self.label_source not in self._ALLOWED_LABEL_SOURCES:
            raise ValueError(
                f"label_source must be one of {self._ALLOWED_LABEL_SOURCES}; "
                f"got {self.label_source!r} (plan §4.7 manifest sources are "
                f"frozen — adding a new source requires re-running the "
                f"§四 adversarial loop)"
            )
        if not isinstance(self.has_real_light, bool):
            raise ValueError(
                f"has_real_light must be bool; got "
                f"{type(self.has_real_light).__name__}={self.has_real_light!r}"
            )


@dataclass(frozen=True)
class FrozenEvalManifest:
    """Typed wrapper around ``runs/_hard_negative_eval_manifest.json``.

    Locked thresholds (plan §4.7 prose; mirrored on R2 eval-parity gate):
      * ``confidence_threshold`` — FIXED at 0.25.
      * ``nms_iou_threshold``    — FIXED at 0.5.

    These are enforced by equality with the module-level constants in
    ``_internals.py``; loader rejects any deviation from the YAML / JSON
    to prevent silent threshold drift between fit time and eval time.
    ``manifest_sha256`` is the SHA256 of the canonical serialization
    (sorted keys, no whitespace) — recorded into every per-arm eval
    JSON's ``fp_manifest_sha256`` field by the trainer pipeline. The
    runner cross-verifies.

    Cross-knob invariants enforced in ``__post_init__`` (pure value
    checks; NO filesystem I/O — the loader handles existence + hashing):
      * ``confidence_threshold`` and ``nms_iou_threshold`` equal the
        plan-pinned ClassVars.
      * ``entries`` is a tuple of ``FrozenEvalManifestEntry``; YAML/JSON
        list must be coerced to tuple by the loader.
      * ``entries`` is non-empty.
      * ``entries`` contain at least one ``has_real_light=True`` (recall
        denominator) AND at least one ``has_real_light=False`` (FP
        denominator). A manifest missing either population would silently
        produce divide-by-zero or undefined recall/FP delta.
      * ``image_sha256`` values are unique across entries (no
        accidental duplicates that would double-count an image's
        contribution to FP/recall).
      * ``manifest_sha256`` is 64-char lowercase hex.
    """

    manifest_sha256: str
    confidence_threshold: float
    nms_iou_threshold: float
    entries: tuple[FrozenEvalManifestEntry, ...]

    _LOCKED_CONFIDENCE_THRESHOLD: ClassVar[float] = PLAN_LOCKED_CONFIDENCE_THRESHOLD
    _LOCKED_NMS_IOU_THRESHOLD: ClassVar[float] = PLAN_LOCKED_NMS_IOU_THRESHOLD

    def __post_init__(self) -> None:
        if not is_hex_sha256(self.manifest_sha256):
            raise ValueError(
                f"manifest_sha256 must be 64-char lowercase hex; got "
                f"{self.manifest_sha256!r}"
            )
        # Locked-threshold hard-pin. Bool-exclusion before equality check
        # because ``True == 1.0`` would silently alias the conf threshold.
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
                f"{self._LOCKED_CONFIDENCE_THRESHOLD} (plan §4.7 hard-pin); "
                f"got {self.confidence_threshold} — deviation is a contract "
                f"violation (FP denominator would silently change)"
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
                f"{self._LOCKED_NMS_IOU_THRESHOLD} (plan §4.7 hard-pin); "
                f"got {self.nms_iou_threshold}"
            )
        if not isinstance(self.entries, tuple):
            raise ValueError(
                f"entries must be tuple; got {type(self.entries).__name__} "
                f"(loader must coerce JSON list → tuple before construction)"
            )
        if not self.entries:
            raise ValueError(
                "entries must be non-empty (a frozen manifest with no "
                "images cannot drive an FP / recall computation)"
            )
        for i, entry in enumerate(self.entries):
            if not isinstance(entry, FrozenEvalManifestEntry):
                raise ValueError(
                    f"entries[{i}] must be FrozenEvalManifestEntry; got "
                    f"{type(entry).__name__}"
                )
        # Population-coverage invariant: at least one of each side. Without
        # both populations the recall denominator OR the FP denominator is
        # zero, which silently turns the §4.7 metrics into undefined values.
        has_real = any(e.has_real_light for e in self.entries)
        has_bg = any(not e.has_real_light for e in self.entries)
        if not has_real:
            raise ValueError(
                "entries must contain at least one has_real_light=True row "
                "(plan §4.7 recall denominator); got none"
            )
        if not has_bg:
            raise ValueError(
                "entries must contain at least one has_real_light=False row "
                "(plan §4.7 FP denominator); got none"
            )
        # image_sha256 uniqueness: an accidental duplicate would double-count
        # the image's contribution to the FP / recall numerator and corrupt
        # the deploy/defer/drop comparison.
        sha_set = {e.image_sha256 for e in self.entries}
        if len(sha_set) != len(self.entries):
            raise ValueError(
                f"image_sha256 must be unique across entries; got "
                f"{len(self.entries)} entries with only {len(sha_set)} "
                f"distinct hashes"
            )


def load_frozen_eval_manifest(path: str | Path) -> FrozenEvalManifest:
    """Parse ``runs/_hard_negative_eval_manifest.json`` into a typed dataclass.

    Coercion contract (b-stage MUST honor ALL of these):
      * ``entries`` (JSON list[dict]) → ``tuple(FrozenEvalManifestEntry(**raw)
        for raw in items)``.
      * ``confidence_threshold`` and ``nms_iou_threshold`` are read from
        the JSON and validated against the locked ClassVars; deviation
        is a hard fail.
      * ``manifest_sha256`` MUST be precomputed and present in the JSON
        (the loader does NOT silently rehash; the canonical hash is the
        SAME value the trainer recorded in each eval JSON's
        ``fp_manifest_sha256``).
      * **B2 review I6 verification contract** (2026-05-10): the loader
        MUST re-derive the SHA256 of the canonical (sorted-keys,
        no-whitespace) serialization of
        ``{confidence_threshold, nms_iou_threshold, entries}`` and
        verify it equals the precomputed ``manifest_sha256`` field.
        Mismatch is a hard fail. Without this re-derivation, a tampered
        manifest could ship with an arbitrary ``manifest_sha256`` and
        the field becomes meaningless. The trainer pipeline uses the
        SAME canonical-serialization protocol when computing the hash
        recorded into each per-arm eval JSON's ``fp_manifest_sha256``.
      * If the path does not exist or the JSON is malformed, raise
        ``FileNotFoundError`` / ``ValueError`` with a structured message
        — the runner converts these into ``decision: "executor_error"``
        rows.

    Args:
        path: filesystem path to a ``_hard_negative_eval_manifest.json`` file.

    Returns:
        ``FrozenEvalManifest`` — fully validated, ready for downstream
        consumption by the c-stage gate.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: schema / range / threshold-deviation / sha-mismatch
            violations.
    """
    import hashlib as _hashlib
    import json as _json

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    with path.open("r", encoding="utf-8") as fh:
        raw = _json.load(fh)
    if not isinstance(raw, dict):
        raise ValueError(
            f"{path}: top-level JSON must be an object; got "
            f"{type(raw).__name__}"
        )
    for required_key in (
        "manifest_sha256",
        "confidence_threshold",
        "nms_iou_threshold",
        "entries",
    ):
        if required_key not in raw:
            raise ValueError(
                f"{path}: required key {required_key!r} is missing"
            )

    # B2 review I6: re-derive the canonical SHA256 and compare against the
    # precomputed value. Canonical form: sort_keys=True, no whitespace,
    # over only {confidence_threshold, nms_iou_threshold, entries}.
    canonical_payload = {
        "confidence_threshold": raw["confidence_threshold"],
        "nms_iou_threshold": raw["nms_iou_threshold"],
        "entries": raw["entries"],
    }
    canonical_bytes = _json.dumps(
        canonical_payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    expected_sha = _hashlib.sha256(canonical_bytes).hexdigest()
    declared_sha = raw["manifest_sha256"]
    if expected_sha != declared_sha:
        raise ValueError(
            f"{path}: manifest_sha256 mismatch. Declared "
            f"{declared_sha!r}, re-derived {expected_sha!r} from "
            f"canonical serialization (sorted-keys, no-whitespace) of "
            f"the {{confidence_threshold, nms_iou_threshold, entries}} "
            f"subset. A tampered manifest cannot ship — the §4.7 frozen "
            f"manifest is the FP-denominator anti-gaming artifact and the "
            f"trainer pipeline uses the same protocol to stamp the hash "
            f"into per-arm eval JSONs."
        )

    raw_entries = raw["entries"]
    if not isinstance(raw_entries, list):
        raise ValueError(
            f"{path}: entries must be a JSON list; got "
            f"{type(raw_entries).__name__}"
        )
    entries: list[FrozenEvalManifestEntry] = []
    for i, entry_raw in enumerate(raw_entries):
        if not isinstance(entry_raw, dict):
            raise ValueError(
                f"{path}: entries[{i}] must be a JSON object; got "
                f"{type(entry_raw).__name__}"
            )
        try:
            entries.append(
                FrozenEvalManifestEntry(
                    image_sha256=entry_raw.get("image_sha256", ""),
                    image_relpath=entry_raw.get("image_relpath", ""),
                    label_source=entry_raw.get("label_source", ""),
                    has_real_light=entry_raw.get("has_real_light"),
                )
            )
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"{path}: entries[{i}] invalid — {e}"
            ) from e

    return FrozenEvalManifest(
        manifest_sha256=declared_sha,
        confidence_threshold=raw["confidence_threshold"]
        if not isinstance(raw["confidence_threshold"], bool)
        else raw["confidence_threshold"],
        nms_iou_threshold=raw["nms_iou_threshold"]
        if not isinstance(raw["nms_iou_threshold"], bool)
        else raw["nms_iou_threshold"],
        entries=tuple(entries),
    )
