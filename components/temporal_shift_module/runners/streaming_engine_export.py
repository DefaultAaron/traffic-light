"""Phase 1-C: Export + Orin verify — ONNX (Slice+Concat shift) → TRT FP16 → Orin.

Phase 1-C closes out TSM by:
    1. ONNX export with the shift op decomposed into Slice + Concat (no custom
       op; required by §1.5 Phase 1-C).
    2. trtexec FP16 build on Orin (or workstation + scp; sidecar records build
       host per the engine sidecar contract).
    3. Per-camera per-stage feature cache wired into inference/cpp/src/
       trt_pipeline.cpp.
    4. Re-measured end-to-end Orin latency < 26 ms with the cache active and
       the ROS2 node holding cache state across frames.

Gate condition (§1.5 + gates — TBD; Phase 1-C gate file lands here when
implementation lands):
    ONNX trace clean (no custom ops detected by onnx.checker)
    AND trtexec FP16 build exit 0
    AND per-camera cache memory < 5 MB
    AND end_to_end_orin_latency_ms < 26 (same locked-caveat columns as 1-B)

Trigger condition:
    Phase 1-B passed Gate-1B. NO direct entry.

Sidecar fields (REQUIRED before this runner can ship — see
components/temporal_shift_module/__init__.py "Engine sidecar carry-forward"):
    tsm_enabled, tsm_shift_fraction, tsm_clip_size_train, tsm_feature_cache_stages.
The runner refuses to write to a sidecar that lacks these fields.

Spec: docs/planning/temporal_optimization_plan.md §1.5 Phase 1-C + §1.6 risk
row 2 (memory budget).
Status: scaffold — lands when Phase 1-C is scheduled.
"""

from __future__ import annotations

import sys

_PHASE = "1-C"
_SUMMARY = "Export + Orin verify: ONNX (Slice+Concat) → TRT FP16 → cache wiring"


def main() -> int:
    raise NotImplementedError(
        f"TSM phase {_PHASE} ({_SUMMARY}) is a scaffold stub. "
        "Implementation lands when scheduled — "
        "see docs/planning/temporal_optimization_plan.md §1.5 + §1.6."
    )


if __name__ == "__main__":
    sys.exit(main())
