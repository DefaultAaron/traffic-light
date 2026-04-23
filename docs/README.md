# Documentation Index

Project documentation organized by purpose. Pick the folder that matches what you're trying to do.

| Folder | For | Lifecycle |
|---|---|---|
| [`planning/`](planning/) | Living plans and schedule — what we're doing and when | Updated frequently |
| [`data/`](data/) | Dataset design, conversion, annotation | Updated per data round |
| [`reports/`](reports/) | Frozen snapshots at end of each phase / round | Append-only after phase end |
| [`proposals/`](proposals/) | Research / feasibility investigations | Revisited if / when promoted to a plan |
| [`integration/`](integration/) | External-facing contracts: ROS 2 messages, TRT pipeline | API-stable, revise only on breaking changes |

---

## planning/

| File | Contents |
|---|---|
| [`development_plan.md`](planning/development_plan.md) | Master plan: candidate models, datasets, phases, milestones, risks |
| [`timeline.md`](planning/timeline.md) | Week-by-week schedule through the 2026-05-15 deadline |
| `任务计划-吴正日.xlsx` | Personal tracking sheet |

## data/

| File | Contents |
|---|---|
| [`class_distribution.md`](data/class_distribution.md) | R1 7-class sample counts per dataset; R2 scope-change banner |
| [`data_conversion_plan.md`](data/data_conversion_plan.md) | S2TLD / BSTLD / LISA → unified YOLO format conversion |
| [`annotation_tool_guide.md`](data/annotation_tool_guide.md) | How to use the project's XML review / edit tools |

## reports/

| File | Contents |
|---|---|
| [`phase_1_report.md`](reports/phase_1_report.md) | Phase 1 3-class baseline — model selection narrative (historical) |
| [`phase_1_results.md`](reports/phase_1_results.md) | Raw eval tables for Phase 1 (RT-DETR-L, YOLO11/26 n/s/m, 3-class) |
| [`phase_2_round_1_report.md`](reports/phase_2_round_1_report.md) | R1 7-class: YOLO26 n/s/m training + Orin deployment + demo diagnosis + alt-track launch + R2 scope lock (10–14 classes) — **living doc** |
| [`phase_2_round_1_results.md`](reports/phase_2_round_1_results.md) | Raw eval tables for R1 (YOLO26 n/s/m populated; YOLOv13-s and DEIM-D-FINE-S/M pending) |

## proposals/

| File | Status |
|---|---|
| [`yolo26_alternatives_survey.md`](proposals/yolo26_alternatives_survey.md) | **In training (2026-04-22)** — YOLOv13-s + DEIM-D-FINE-S/M as R1 alt tracks |
| [`depth_estimation_feasibility.md`](proposals/depth_estimation_feasibility.md) | **On hold**, awaiting team leader approval |
| [`temporal_encoder_feasibility.md`](proposals/temporal_encoder_feasibility.md) | **Deferred**, revisit after 2026-05-15 deadline |

## integration/

| File | Contents |
|---|---|
| [`ros2_integration_guide.md`](integration/ros2_integration_guide.md) | `Detection2DArray` contract, topic name, class_id strings, for planning-module consumers |
| [`trt_pipeline_guide.md`](integration/trt_pipeline_guide.md) | End-to-end Orin deployment: sync → environment → CMake → build → ONNX strip → trtexec → run |
