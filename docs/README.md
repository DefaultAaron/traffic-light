# Documentation Index

Project documentation organized by purpose. Pick the folder that matches what you're trying to do.

| Folder | For | Lifecycle |
|---|---|---|
| [`planning/`](planning/) | Living plans and schedule — what we're doing and when | Updated frequently |
| [`data/`](data/) | Dataset design, conversion, annotation | Updated per data round |
| [`reports/`](reports/) | Frozen snapshots at end of each phase / round | Append-only after phase end |
| [`integration/`](integration/) | External-facing contracts: ROS 2 messages, TRT pipeline | API-stable, revise only on breaking changes |
| [`ops/`](ops/) | Operations: scripts reference, team roster, runbooks | Updated when scripts/tooling change |
| [`templates/`](templates/) | Shared per-round / per-deliverable shells | Stable; bump version when shape changes |

> Academic / paper-track material lives **outside** `docs/` under [`../research/`](../research/) — surveys, contribution candidates, feasibility studies. `docs/` is reserved for project execution; `research/` is reserved for paper writing source material.

---

## planning/

| File | Contents |
|---|---|
| [`development_plan.md`](planning/development_plan.md) | Master plan: candidate models, datasets, phases, milestones, risks |
| [`timeline.md`](planning/timeline.md) | Week-by-week schedule through the 2026-05-15 deadline |
| [`temporal_optimization_plan.md`](planning/temporal_optimization_plan.md) | **R2/R3 optional optimization track**, parallel to main detector selection. Recommended: TSM (detector-level, zero-overhead). Alternatives: post-detector smoothers (HMM → AdaEMA → GRU → Transformer). Plan A (tracker + EMA) already landed; further work activation depends on baseline + replay surfacing specific failure modes |
| [`cross_detection_reasoning_plan.md`](planning/cross_detection_reasoning_plan.md) | **R3 candidate, deferred** — same-frame multi-light co-occurrence reasoning (Bayesian post-processing → CRF → Relation Network). Premise validation must run on R2 self-collected data (R1 data is foreign-domain and gets discarded). Activation gated behind temporal track + persistent edge-class confusion + a 3-gate empirical check on R2 data |
| `任务计划-吴正日.xlsx` | Personal tracking sheet |

## data/

| File | Contents |
|---|---|
| [`class_distribution.md`](data/class_distribution.md) | R1 7-class sample counts per dataset; R2 scope-change banner |
| [`data_conversion.md`](data/data_conversion.md) | S2TLD / BSTLD / LISA → unified YOLO format conversion |
| [`annotation_tool.md`](data/annotation_tool.md) | How to use the project's XML review / edit tools |
| [`r2_data_collection_sop.md`](data/r2_data_collection_sop.md) | **R2 multimodal data collection + annotation SOP** — dual 8MP cameras (normal + wide) + LiDAR; sync, calibration, site/time coverage, 10–14 class taxonomy, hard-case slices, LiDAR-aided distance GT + vibration diagnostics + cross-modal hard-neg mining, site-based splits, release prep |

## reports/

| File | Contents |
|---|---|
| [`phase_1_report.md`](reports/phase_1_report.md) | Phase 1 3-class baseline — model selection narrative (historical) |
| [`phase_1_results.md`](reports/phase_1_results.md) | Raw eval tables for Phase 1 (RT-DETR-L, YOLO11/26 n/s/m, 3-class) |
| [`phase_2_round_1_report.md`](reports/phase_2_round_1_report.md) | R1 7-class: YOLO26 n/s/m training + Orin deployment (25 ms/frame @ 1280 FP16) + demo diagnosis (engine imgsz + xyxy postprocess fixes) + alt-track launch + R2 scope lock (10–14 classes) — **living doc** |
| [`phase_2_round_1_results.md`](reports/phase_2_round_1_results.md) | Raw eval tables for R1 (YOLO26 n/s/m populated; YOLOv13-s and DEIM-D-FINE-S/M pending) |

## integration/

| File | Contents |
|---|---|
| [`ros2_contract.md`](integration/ros2_contract.md) | `Detection2DArray` contract, topic name, class_id strings, for planning-module consumers |
| [`trt_quickstart.md`](integration/trt_quickstart.md) | **Start here if you're integrating** — minimal Orin setup, three integration modes, common gotchas |
| [`trt_deployment.md`](integration/trt_deployment.md) | End-to-end Orin deployment: sync → environment → CMake → build → ONNX strip → trtexec → run |
| [`tracker.md`](integration/tracker.md) | ByteTrack + per-track EMA voting (Python + C++ parity); landed as flicker P0 mitigation |

## ops/

| File | Contents |
|---|---|
| [`scripts_reference.md`](ops/scripts_reference.md) | Index + flag reference for everything under `scripts/` (demo sweep, training, dataset prep, model export, flicker validation, network workarounds) |
| [`team_roster.md`](ops/team_roster.md) | 12-specialist roster (4 clusters) + dispatch protocol; read before delegating non-trivial work |
| [`tailscale_runbook.md`](ops/tailscale_runbook.md) | Quick runbook: campus-network long-poll failures + which `scripts/tailscale_*.sh` workaround applies |

---

## See also

- [`../research/`](../research/) — surveys, contribution candidates, feasibility studies (paper-track)
- [`../README.md`](../README.md) — top-level project overview
