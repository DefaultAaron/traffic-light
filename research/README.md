# Research / Paper-Track Material

Surveys, contribution candidates, and feasibility studies kept **outside** `docs/` so that `docs/` stays focused on project execution and `research/` stays focused on paper writing.

| Folder | Contents | When to read |
|---|---|---|
| [`surveys/`](surveys/) | Literature surveys + alt-architecture / enhancement / feasibility studies | Designing an ablation, framing related work, evaluating a new direction |
| [`contributions/`](contributions/) | Field-gap analysis + candidate contributions of this project's R2 data | Drafting paper claims, prioritizing what's worth releasing |

## surveys/

| File | Status | Summary |
|---|---|---|
| [`alt_detector_architectures.md`](surveys/alt_detector_architectures.md) | Used for R1 alt-track selection | Comparison of YOLO26 alternatives (YOLOv13, DEIM-D-FINE, DETR family, TL-domain methods); selection rationale + exclusion criteria; deployment-side TRT 8.5 constraints |
| [`detection_enhancements.md`](surveys/detection_enhancements.md) | Recommendations integrated into planning | Beyond-the-three-plans methods (training aug, hard-neg mining, KD, SAHI, map-prior, HDR camera, multi-camera, planner-prior); priority recommendations with R1 demo痛点匹配 |
| [`depth_estimation.md`](surveys/depth_estimation.md) | On hold pending team-leader approval | Stereo (Fast-FoundationStereo / HITNet) vs monocular (DA v3 / Metric3D v2) feasibility on Orin; latency / accuracy / license trade-offs |

## contributions/

| File | Summary |
|---|---|
| [`field_gaps_and_contributions.md`](contributions/field_gaps_and_contributions.md) | TL-detection field-wide failure modes; public-dataset coverage gaps; this project's contribution candidates from R2 self-collected data (CN-gantry benchmark, TL × barrier joint, hard-condition slice, engineering refs) |

---

## Conventions

- **Language:** Chinese prose with inline English technical terms (matches `docs/` convention).
- **Status banner:** Each file's top quote-block declares whether it is *active* (still informing decisions), *used / archived* (decisions already made), *on hold* (awaiting external approval), or *recommendations integrated* (content has been split into `docs/planning/`).
- **Cross-refs to `docs/`:** Use `../../docs/<sub>/<file>.md` from inside `research/<sub>/`.
- **No execution plans here.** Anything that becomes an execution plan moves to `docs/planning/`. The original survey stays here as paper-writing source material.

## See also

- [`../docs/`](../docs/) — project execution (planning / data / reports / integration / ops)
- [`../README.md`](../README.md) — top-level project overview
