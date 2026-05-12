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
| [`development_plan.md`](planning/development_plan.md) | Master plan: candidate models, datasets, **two-stage delivery — Stage 1 (R2 close = performance + stability, no latency gate) → Stage 2 (Orin FP16 p95 < 33 ms with bounded quality regression)**. Evidence-bounded round close. 2026-05-15 deadline retired. |
| [`additional_components_plan.md`](planning/additional_components_plan.md) | Per-component WHAT spec: §三 copy-paste, §四 hard-negative, §六 SAHI (per-camera adaptive), §七 KD v2 cell matrix, §八 multi-camera fusion (heterogeneous Cam-W + Cam-T, dual baseline) |
| [`temporal_optimization_plan.md`](planning/temporal_optimization_plan.md) | **Optimization track**, parallel to main detector selection. Recommended: TSM (detector-level, zero-overhead). Alternatives: post-detector smoothers (HMM → AdaEMA → GRU → Transformer). Plan A (tracker + EMA) already landed; further work activation depends on baseline + replay surfacing specific failure modes |
| [`cross_detection_reasoning_plan.md`](planning/cross_detection_reasoning_plan.md) | **R3 candidate, deferred** — same-frame multi-light co-occurrence reasoning (Bayesian post-processing → CRF → Relation Network). Premise validation must run on R2 self-collected data (R1 data is foreign-domain and gets discarded). Activation gated behind temporal track + persistent edge-class confusion + a 3-gate empirical check on R2 data |
| [`pre_deploy_AGV_integration.md`](planning/pre_deploy_AGV_integration.md) | R4+ carry-forward parking stub: runtime camera switching, deploy-tuning triggers (post Cam-W feasibility / always-on fusion outcomes). Activation by evidence, not date. |
| [`kd_a6_design_spike.md`](planning/kd_a6_design_spike.md) | A6 cross-arch DEIM-M → YOLO26-s design spike output; path γ selected (FDR Integral collapse → L1+GIoU bbox KD + cls KL + PKD projection) |
| _Retired_ | `timeline.md` + `任务计划-吴正日.xlsx` (2026-05-12 deadline pivot) + `pre_r2_kickoff_checklist.md` (R2 in-flight) — all moved to [`_archive/`](_archive/) |

## data/

| File | Contents |
|---|---|
| [`r2_data_collection_sop.md`](data/r2_data_collection_sop.md) | **R2 multimodal data collection + annotation SOP** — heterogeneous dual cameras (Cam-W = SG3S 3MP wide w/ LFM + Cam-T = SG8S 8MP tele no-LFM) at dual baselines (~50mm + ~250mm) + LiDAR; sync, per-baseline calibration + freshness, site/time coverage, 10–14 class taxonomy, hard-case slices, LiDAR-aided distance GT + vibration diagnostics + cross-modal hard-neg mining, site-based splits, release prep |

> R1 dataset prep docs (`class_distribution.md`, `data_conversion.md`, `annotation_tool.md`) are retired — see [`_archive/`](_archive/). R1 datasets (LISA / BSTLD / S2TLD) are no longer part of R2/R3 training or evidence base.

## reports/

| File | Contents |
|---|---|
| [`phase_2_round_1_report.md`](reports/phase_2_round_1_report.md) | R1 7-class: YOLO26 n/s/m/l + YOLOv13-s + DEIM-D-FINE-S/M/L; Orin deployment (25 ms/frame @ 1280 FP16); demo diagnosis (engine imgsz + xyxy postprocess fixes); R2 scope lock (10–14 classes) — **CLOSED 2026-05-12** |
| [`phase_2_round_1_results.md`](reports/phase_2_round_1_results.md) | Raw eval tables for R1 — all 7 trained models populated (YOLO26 n/s/m/l, YOLOv13-s, DEIM-D-FINE-S/M/L); DEIM family on unified deployment-checkpoint methodology (see `r1_evidence/`) |
| [`r1_evidence/`](reports/r1_evidence/) | R1 closure durable audit: DEIM per-class JSONs + old-vs-new methodology diff + one-shot reproducer script |

> Phase 1 historical reports (`phase_1_report.md`, `phase_1_results.md`) and the early KD A2a stub (`ablation_results.md`) are retired — see [`_archive/`](_archive/). R1 7-class results fully supersede the P1 3-class baseline.

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
