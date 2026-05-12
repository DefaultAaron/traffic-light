# 附加组件计划 v2.0

v1.1 LOCK 2026-05-10

## 状态

| 项 | 当前值 |
|---|---|
| 范围 | R2 / R3 训练侧、推理侧、系统集成侧附加组件 |
| R2 in-round hot path | §三 Copy-paste、§四 硬负样本、§七 KD P0 cells |
| R2 持续调度 | §六 SAHI（gated；R2 真机回放暴露小目标 long-tail FN 后启动 a-stage）、§八 多相机融合（[阻塞] 自动驾驶团队 — 外参 / time-sync 锁定后启动）|
| R3+ deferred | §五 地图先验、§九 自适应推理 / ROI、§十 INT8 QAT、§十一 规划器先验 |
| v2.0 规则 | WHAT-only；本文件定义共享五步生命周期 |

| 组件 | v2.0 状态 | 输出 |
|---|---|---|
| §三 Copy-paste + 类平衡 | a-stage LANDED；R2 freeze 后 b/c/d | `runs/_copy_paste_decision.json` |
| §四 硬负样本挖掘 | a-stage LANDED；pre-R2 rehearsal 可启动 | `runs/_hard_negative_decision.json` |
| §五 地图先验门控 | DEFERRED → R3+ | `runs/_map_prior_decision.json` |
| §六 SAHI 切片推理 | gated；a-stage 未启动（trigger：R2 真机回放暴露小目标 long-tail FN）| `runs/_sahi_decision.json` |
| §七 KD | scaffold v1.3 LANDED；P0 cells 待启动 | `runs/_kd_decisions.json` |
| §八 多相机融合 | [阻塞] 自动驾驶团队 | `runs/_multi_camera_decision.json` |
| §九 自适应推理 / ROI | DEFERRED → R3+ | `runs/_adaptive_inference_decision.json` |
| §十 INT8 QAT | DEFERRED → R3+ | `runs/_int8_qat_decision.json` |
| §十一 规划器先验融合 | DEFERRED → R3+ | `runs/_planner_prior_decision.json` |

## Deferred

| 组件 | blocked_on | unblock_logic | 解锁入口 |
|---|---|---|---|
| §五 地图先验门控 | `gps_topic_5_12`, `r2_raw_video_negative_controls` | all | §五 |
| §九 自适应推理 / ROI | `map_prior_landed`, `r3_inference_budget_window` | all | §九 |
| §十 INT8 QAT | `sahi_b_c_measured`, `tsm_phase_1b_passed` | any | §十 |
| §十一 规划器先验融合 | `planning_team` | all | §十一 |

R2 phase report `coverage-gaps` 必须逐项列出以上 deferred 组件、blocked_on、unblock evidence path、next entrypoint。

## 一、生命周期定义

| 阶段 | 名称 | 验收 |
|---|---|---|
| a | 脚手架 Scaffold | 文件存在；`bash -n` / `python -m py_compile` 通过；如适用，B2 + C3 AGREED |
| b | 实现 Implementation | stub / `NotImplementedError` 替换为可运行代码；fixture / micro dataset 跑通 |
| c | 消融 Ablation | baseline A/B；预承诺指标 + CI；冻结 manifest / threshold |
| d | 决策 Decision | deploy / defer / drop 三选一；写入对应 `runs/_*_decision*.json` |
| e | 报告 Report | phase report 子节落地；defer 必填 `next_round_action`；drop 必填 `reason` 字段 |

R2 round 关帧前必须完成 a-c：§三、§四、§七 P0 cells。deploy 候选必须完成 d。所有启动项必须完成 e。未触发 / deferred 项不计 incomplete，但必须进入 coverage-gaps / carry-forward。

## 二、行动顺序

| 顺序 | 组件 | 依赖 / 阻塞 | 日历 | R2 要求 |
|---:|---|---|---|---|
| 1 | §三 Copy-paste + 类平衡 | R2 freeze | R2 训练前 | a-c 必须 |
| 2 | §四 硬负样本挖掘 | R1 baseline / demo8/11/13 / R2 难场景 | R2 训练前 | a-c 必须 |
| 3 | §七 KD | R2 选型胜者 + teacher | R2 训练同步 | P0 必须 |
| 4 | §六 SAHI | R2 replay 暴露小目标 long-tail FN | replay 触发后启动 a-stage | gated |
| 5 | §八 多相机融合 | `autonomy_team` | 待对齐 | blocked |
| 6 | §五 地图先验 | deferred 条件 | R3+ | carry-forward |
| 7 | §九 自适应推理 | §五 + R3 预算窗口 | R3+ | carry-forward |
| 8 | §十 INT8 QAT | §六 / TSM latency band | R3+ | carry-forward |
| 9 | §十一 规划器先验 | `planning_team` + cross-detection | R3+ | carry-forward |

## 三、Copy-paste 增强 + 类平衡损失

### 行动项

- [x] a. `components/copy_paste_balance/` LANDED；commits `b6670a1` + `d8b3c02`；C3 AGREED-CLEAN。
- [ ] b. R2 freeze 后启用 YOLO `copy_paste` / DEIM dataloader hook；保持 `fliplr=0`；paste y-center 约束在上画面区域。
- [ ] c. 三臂消融：无 copy-paste / copy-paste / copy-paste + class-balanced。
- [ ] d. 写 `runs/_copy_paste_decision.json`。
- [ ] e. R2 phase report 子节。

### 决策规则

| outcome | condition | JSON write |
|---|---|---|
| deploy | 稀有类 AP 平均 +5 pp OR 至少 1 个稀有类 +5 pp；且所有稀有安全类 AP delta ≥ -1 pp；稀有类相关 FP 上升 ≤ 10%；总 mAP ≥ -0.2 pp | `outcome="deploy"` |
| defer | 稀有类改善 < 2 pp；总 mAP ≥ -0.2 pp | `outcome="defer"`, `next_round_action` 写 R3 copy-paste 策略 |
| drop | 总 mAP 退化 > 0.5 pp OR 任一稀有安全类 AP 退化 > 1 pp OR 稀有类 FP 上升 > 30% | `outcome="drop"` |

FP 计数使用 §四 frozen manifest；confidence ≥ 0.25；NMS IoU = 0.5。

## 四、硬负样本挖掘

### 行动项

- [x] a. `components/hard_negative_mining/` LANDED；commits `e802250` + `58bf859` + `7428f88` + `57e281b`；C3 final AGREED-CLEAN。
- [ ] b. R1 baseline 跑 demo8/11/13 + R2 难场景；构建 `bg/` / empty-image；人工核验 ≥10%。
- [ ] c. A/B：无硬负 vs 有硬负；记录 FP、真实灯 recall、总 mAP。
- [ ] d. 写 `runs/_hard_negative_decision.json`。
- [ ] e. R2 phase report 子节。

### 决策规则

| outcome | condition | JSON write |
|---|---|---|
| deploy | demo8 类背景 FP 下降 ≥ 50%；真实灯 recall delta ≥ -0.5 pp；总 mAP ≥ -0.2 pp | `outcome="deploy"` |
| defer | FP 下降 20-50%；真实灯 recall delta ≥ -0.5 pp | `outcome="defer"` |
| drop | 真实灯 recall delta < -0.5 pp OR FP 下降 < 20% | `outcome="drop"` |

冻结评估 manifest：`runs/_hard_negative_eval_manifest.json` + `.sha256`；评估时不得改 frame 子集、confidence、NMS IoU。

## 五、地图先验门控

### 状态

DEFERRED → R3+。R2 in-round 不要求 a-stage。

### 行动项

- [ ] a. 与定位团队锁定 ROS2 GPS topic；落 `MapPriorGate` 接口。
- [ ] b. OSM / 路网 polygon 匹配；未匹配到路口时不抑制检测。
- [ ] c. A/B：无门控 / 路口外抑制 / 路口外抑制 + 路口内 SAHI。
- [ ] d. 写 `runs/_map_prior_decision.json`。
- [ ] e. R3 phase report 子节。

### 决策规则

| outcome | condition | JSON write |
|---|---|---|
| deploy | 路口外 FP -80%；路口内 recall ≥ -0.5 pp；负面控制 missed safety-critical = 0 | `outcome="deploy"`, `branch="live"` |
| defer | 路口外 FP -50%；路口内 recall 退化 0.5-2 pp；负面控制 missed safety-critical = 0 | `outcome="defer"` |
| drop | 路口内 recall 退化 > 2 pp OR 负面控制 missed safety-critical ≥ 1 | `outcome="drop"` |

负面控制场景：施工 / 临时信号灯、GPS jitter ±20 m、map-missing intersection。

## 六、SAHI 切片推理

### 自适应配置（per-camera；与 §八 同步，2026-05-12 锁定）

异构双相机 Cam-W + Cam-T 在 pixel-on-target 上差异 ~7×（§八 hardware 配置）；统一 4-tile slicing 在 Cam-T 上是无效成本。Per-camera 切片网格：

| 相机 | 原生分辨率 | 100 m TL pixel-on-target（估算）| Slice 网格 | Overlap | 默认 |
|---|---|---|---|---|---|
| Cam-W | 1920×1536 | ~8 px | **c0 precheck 后冻结**（4×4 vs 3×3 vs 2×4 三选一）| 20 % | 4×4 候选；c0 决定 |
| Cam-T | 3840×2160 | ~60 px | **1×1** 默认；2×2 仅 c3 触发 | 20 % | 1×1（c3 条件：frozen val ≥ 50 个 150 m+ 实例时启动 2×2 ablation）|

**ROI mask 契约**（calibrated image coords，非 tile ID）：
- **Cam-T-primary core**：Cam-W 校准图中 Cam-T 视场投影内的中心区域，Cam-W SAHI 不切。
- **Fusion-overlap margin**：core 边界 ±20 % 缓冲带，Cam-W SAHI 切但 fusion 时优先信任 Cam-T。
- **Cam-W-primary outer ring**：margin 之外，Cam-W SAHI 独占。
- ROI mask 由 per-baseline calibration（§八）派生，存 `runs/_sahi_roi_mask_<baseline>mm.yaml`。
- **Slice detection remap contract**：所有 slice 内检测必须 remap 至 native camera coords，再喂 §八 calibration / WBF / fusion 阶段。Batch 调度为实现细节，必须保留 `camera_id` + `baseline_id` tag。

### 行动项

- [ ] **c0. Grid precheck（pre-implementation）**：Orin 合成张量 dry run，测 1× / 5× / 17× / 22× forward count FP16 latency；若 17× forward 推断 end-to-end > 80 ms 则放弃 Cam-W 4×4，回退 3×3 或 2×4。frozen val 子集（≥ 200 帧）对比 Cam-W 4×4 / 3×3 / 2×4 小目标 recall × latency，冻结一个网格。
- [ ] a. `inference/cpp/include/sahi.hpp` 接口（per-camera grid 配置 + ROI mask）；与 `trt_pipeline.cpp` + §八 fusion 兼容性 review。
- [ ] b. Python 验证 + C++ 实现 + per-camera batch engine（Cam-W ROI-gated batch + Cam-T native）；slice→native coord remap 单元测试。
- [ ] c1. A/B：无 SAHI / Cam-W ROI-gated SAHI / Cam-W full-frame SAHI；post-fusion 小目标 recall 按 `baseline_id × camera_id × distance_bin`（near/mid/far per SOP §7.1）stratify 报告 + per-stage latency breakdown（detector forward / slicing overhead / remap / fusion / WBF）。**Case A deploy gate 强制 per-bin 无退化**（见决策表 A 行）；任一 bin lower-CI 退化 > 2 pp 自动 fall-through 至 Case B / C / D（aggregate 不够 +5 pp 时）或 `executor_error`（aggregate ≥ +5 pp 但 per-bin gate 失败时，blocks deployment）。
- [ ] c2. A/B：Cam-W ROI-gated vs full-frame SAHI（boundary-bin reporting：ROI margin 内对象单独报告 recall 与重复率）。
- [ ] c3. **(conditional)** Cam-T 2×2 vs 1×1：仅当 frozen val ≥ 50 个 150 m+ TL 实例时启动；否则 Cam-T 默认 1×1。
- [ ] c4. **Alternatives check**：no-SAHI + §三 small-object copy-paste 单独对照行（不计入主决策，但 phase report 必报）；super-resolution 路径 explicitly deferred（不在本 round）。
- [ ] d. 写 `runs/_sahi_decision.json`（含 per-camera grid + ROI mask path + per-stage latency breakdown）。
- [ ] e. phase report 子节。

### 决策规则（保留 4-case；评估指标 = post-fusion 小目标 recall，Cam-W-only recall 为诊断字段）

| Case | condition | outcome | JSON / carry-forward |
|---|---|---|---|
| A | post-fusion 小目标 recall lower-CI ≥ +5 pp AND **no `baseline_id × camera_id × distance_bin` lower-CI regression > 2 pp**（per-bin 结构性 gate；空 bin support < 30 时 lower-CI 计算结果作 `insufficient`，不阻塞 deploy）AND end-to-end FP16 latency（含 §八 fusion overhead） < 50 ms | deploy | `outcome="deploy"` |
| B | post-fusion 小目标 recall lower-CI ≥ +2 pp AND end-to-end FP16 latency ∈ [50, 80) ms | defer-to-R3-INT8-evaluation | 写 `runs/_sahi_decision.json`；登记 `item_id="sahi_int8_retest"`，`blocked_on=["sahi_b_c_measured"]` |
| C | post-fusion 小目标 recall lower-CI ∈ [+2, +5) pp AND end-to-end FP16 latency < 50 ms | defer-to-R3-recall-marginal | 登记 `item_id="sahi_recall_marginal_retest"`，`blocked_on=["r3_inference_budget_window"]` |
| D | post-fusion 小目标 recall lower-CI < +2 pp OR end-to-end FP16 latency ≥ 80 ms | drop | `outcome="drop"` |

求值顺序：D → A → B → C；exactly-one outcome；0 或多匹配写 `decision_case="executor_error"` 并 **block deployment**（不进 Gate #5 sidecar）。

**与项目二阶段交付（`development_plan.md` §Stage 1 / Stage 2 close gate）reconcile**：本节 4-case 中的 `< 50 ms` / `[50, 80) ms` / `≥ 80 ms` 是 SAHI 内部的 recall × latency 分流阈值，**仅用于 R2 / Stage 1 阶段**判断 SAHI 是否值得继续投入（Case A = 内部 deploy 候选；Case B/C = defer 至 Stage 2；Case D = drop）。Stage 1 通过的 SAHI 候选属于"性能优先"产物，**不**即时部署；Stage 2 latency 优化阶段再用 `development_plan.md` §Stage 2.A（p95 < 33 ms on Orin FP16）做最终 ship 决策。SAHI Case A 通过 Stage 1 但 Stage 2.A 未过 → 进 phase report carry-forward，登记 `item_id="sahi_deploy_blocked_on_stage2_latency"`。

边界（按 lower-CI / end-to-end latency 解读）：`+5 pp, 49 ms → A`；`+5 pp, 50.0 ms → B`；`+5 pp, 80.0 ms → D`；`+2 pp, 49 ms → C`；`+2 pp, 50.0 ms → B`；`+1.999 pp, 30 ms → D`。numeric comparison 使用 ≥ / < 严格匹配上表 condition；exactly-equal 边界值（50.0, 80.0, +2.000, +5.000）走表中显式标记的 case。

**Latency breakdown 报告契约**：phase report 子节必须列 Cam-W only / Cam-W + Cam-W SAHI / Cam-W + Cam-T fusion / Cam-W SAHI + Cam-T fusion 四档的 detector forward / slicing / remap / fusion / WBF 分段时延，以便区分 SAHI cost vs fusion cost 的责任归属（fusion overhead 在所有 SAHI variants 中等成本，不应让 SAHI 因 fusion 被错误降级）。

## 七、知识蒸馏 KD

### 部署侧约束（2026-05-11 锁定）

- **学生侧只能是 S**：DEIM-D-FINE-S + YOLO26s **双轨同时作为部署候选**；M / L 仅作教师 / 上限基准。
- **部署侧参数调优时机：R2 close 后按需触发**，**不阻塞** KD 消融周期或 ship-decision。R1 demo 视检观察 DEIM-S/M 稳定性弱于 YOLO（同物体置信度逐帧抖动 + 长尾 bbox 漂移 + demo8 假阳拒识弱于 YOLO26m）；YOLO 侧亦可能在 R2 数据下出现新的稳定性 / 召回失衡。调优在 §pre-deploy AGV 阶段执行，调优维度：conf 阈值、NMS、（DEIM 专属）FDR `reg_max`、训练 mixup / mosaic `stop_epoch`、（YOLO 专属）anchor sensitivity / `cls_pw`。具体触发逻辑见 §KD 验收门 Gate #6。

### 行动项

- [x] a1. `components/knowledge_distillation/` scaffold v1.3 LANDED；B2 + C3 AGREED。
- [ ] a2. 首个 KD cell 前落地：`scripts/_kd_decision_schema.json`、`scripts/_kd_decide_cell.py`、export sidecar 字段（含 `student_arch ∈ {yolo26_s, deim_dfine_s}`、`burst_jitter_demo_4_10_12_15_max_pp`、`hard_neg_fp_rate_demo_8_11_13` — 后两项为推理 baseline 记录字段，不进 KD ship-decision，供 §pre-deploy AGV `pre_deploy_AGV_integration.md` 的 deploy-tuning trigger 比较）。
- [ ] b. 替换 runner stubs；按 cell 矩阵分批推进。
- [ ] c. A0-A7 消融；应用 6 项验收门。
- [ ] d. 写 `runs/_kd_decisions.json`。
- [ ] e. R2 phase report KD 子节。

### 教师规模规则

| 项 | 规则 |
|---|---|
| 主路径 | M 教师：YOLO26-m / DEIM-D-FINE-M |
| A7 | L tier：YOLO26-l / **DEIM-D-FINE-L**（in training，预计 R2 启动前 `runs/detect/deim_dfine_l-r1/best_stg2.pth` 落地）|
| X/XL | 排除 |
| A7 触发 | DEIM-L 训练完成（自动解锁）OR 稀有安全类 AP 未达 R3 门槛 |
| A7 桥接 | TAKD / ESKD / 投影 MLP 三选二 |
| YOLO26-l 教师状态 | **不合格**：R1 best mAP50=0.850 < YOLO26-m 0.869；只用 DEIM-L 作为 L tier 教师 |

容量结论：DEIM-L OK；YOLO26-l 在 R1 状态下被禁用（R2 重训若超过 m 则解禁）；X/XL excluded。

### KD 验收门

| Gate | 要求 |
|---|---|
| #1 总 mAP | KD lower-CI > A1_CI_low AND KD lower-CI > A1 point - 0.5 pp；ship-decision 强制 `seed5` |
| #2 安全类 AP | 每个 `full_val_support ≥ 30` 安全类 AP delta ≥ -0.5 pp |
| #3 FP | demo8/11/13 背景帧 FP 不上升；与 §四 manifest 共享 |
| #4 成本 | 每 cell wall-clock < `T_scratch_A1 × 2.0`（**Stage 1 training-cost gate only**；KD ship 候选另需通过 `development_plan.md` §Stage 2.A latency + §Stage 2.B quality regression cap）|
| #5 TRT + sidecar | engine 通过 eval-parity；sidecar 含 `kd_cell_id` / `kd_method` / `kd_teacher_artifact_sha256` |
| #6 部署稳定性 trigger | **不阻塞 KD Stage 1 ship-candidate decision**。R2 close 后对 ship-flagged 学生（YOLO26-s 或 DEIM-D-FINE-S）跑 demo4/10/12/15 burst 抖动 + demo8/11/13 假阳率测量；若任一指标差于 R1 baseline（DEIM 基线由 A2b rehearsal 测得，YOLO 基线由 A2a rehearsal 测得）则在 `docs/planning/pre_deploy_AGV_integration.md` 注册 deploy-tuning 任务，调优结果回写 `runs/_pre_deploy_tuning_decisions.json`。**KD cells 自身 Stage 1 ship-candidate decision 依据 Gate #1-#5；最终 ship 仍须通过 `development_plan.md` §Stage 2.A latency + §Stage 2.B quality regression** |

### Cell 矩阵

学生侧锁定 S（YOLO26-s 或 DEIM-D-FINE-S）。互补家族 = YOLO 学生取 DEIM 教师 / 反之。

| Cell | 学生 | 教师 | 方法栈 | 触发 | 优先级 |
|---|---|---|---|---|---|
| A0 | DEIM-D-FINE-S | — | GO-LSD off；no external KD | DEIM 路径 baseline | P0 |
| A1 | YOLO26-s + DEIM-D-FINE-S | — | scratch | always；双路径并行（S deployees）| P0 |
| A2a | YOLO26-s | YOLO26-m | cls-logit KL | YOLO 家族 | P0 |
| A2b | DEIM-D-FINE-S | DEIM-D-FINE-M | LD on FDR + cls-logit KL | DEIM 家族 | P0 |
| A3 | S 双路径 | 同家族 M | PKD feature-level | always | P0 |
| A4 | S 双路径 | 同家族 M | A2 + A3 | max(A2/A3) lower-CI > A1 point；安全类 delta ≥ -0.5 pp | P1 |
| A5 | S 双路径 | 同家族 M → 互补家族 M | 渐进 2 教师 | A4 通过全部 6 gate | P2 |
| A6 | YOLO26-s | DEIM-D-FINE-M | 跨架构 cls-logit KL + DEIM FDR Integral 坍缩 → scalar bbox L1+GIoU KD + PKD FPN 投影 conv（path γ；A6 spike 选定）| A4 通过全部 6 gate；**优先级提升**：DEIM 长尾教学信号 → YOLO 稳定推理路径 | P1 |
| A7 | DEIM-D-FINE-S | **DEIM-D-FINE-L**（in training） | TAKD / ESKD / projection MLP | DEIM-L 训练完成自动解锁；仅 DEIM 学生（YOLO26-l 不合格） | P1 |

**Drawdown 顺序更新**：先丢 A5，再丢 A4（如非触发），保留 A6 / A7（独立高价值轨道）。P0 + A6 + A7 不可同时丢。

**A6 优先级上调说明（2026-05-11；spike 完成 2026-05-12 措辞同步）**：跨架构 DEIM-M → YOLO26s 是同时利用 "DEIM 长尾召回强 + YOLO 推理稳定" 两端的唯一路径。1-day 投影层设计 spike 已完成（`docs/planning/kd_a6_design_spike.md`，`runs/rehearsal_kd_A6_design_spike.json`），选定 path γ：DEIM FDR Integral 坍缩 → scalar bbox L1+GIoU KD + cls-logit KL + PKD FPN 投影 conv（YOLO26 `reg_max=1` 无原生 DFL，直接 FDR↔DFL 分布对齐不可行）。1-week PoC 验证长尾 recall ≥ +5 pp。PoC fail → R3 候选诊断：pseudo-label bridge（DEIM teacher pred → confidence-filtered top-k → YOLO target assignment）作为替代路径，**不在 R2 范围**。

### Runner 映射

| Cell | runner |
|---|---|
| A0 | `components.knowledge_distillation.runners.deim_baseline_golsd_off` |
| A1 | `components.knowledge_distillation.runners.scratch_baseline` |
| A2a | `components.knowledge_distillation.runners.yolo_logit_kd` |
| A2b | `components.knowledge_distillation.runners.deim_logit_localization_kd` |
| A3 | `components.knowledge_distillation.runners.pearson_feature_kd` |
| A4 | `components.knowledge_distillation.runners.logit_plus_feature_kd` |
| A5 | `components.knowledge_distillation.runners.progressive_multi_teacher` |
| A6 | `components.knowledge_distillation.runners.cross_arch_feature_kd` |
| A7 | `components.knowledge_distillation.runners.takd_large_teacher` |

统一 CLI：`--config`、`--teacher-ckpt`、`--student-init {scratch,coco,r2_baseline,r1_rehearsal}`、`--output-dir`、`--seed`、`--ci-method {bootstrap1000,seed5}`、`--resume`、`--rehearsal-on-r1`（pre-R2 rehearsal 标志，强制 `rehearsal_` 前缀输出）。Resume 不覆写 `SEED.txt`。

## 八、多相机融合

### 硬件配置（2026-05-12 锁定）

- **Cam-W**（wide）：森云 SG3S-ISX031C-GMSL2F（Sony ISX031, 2.95 MP @ 1920×1536, 3.0 µm pixel, HDR + LFM）；水平 FOV ~38°。
- **Cam-T**（tele）：森云 SG8S-AR0820C-5300-G2A（OnSemi AR0820, 8.3 MP @ 3840×2160, 2.1 µm pixel, HDR only, no LFM）；水平 FOV ~10°。
- **基线配置**：双相机水平排列，两套基线 ~50 mm + ~250 mm（不同车型）。基线影响 fusion 外参，不改单帧图像内容。

### 单相机运行（fusion fault / pre-deploy fallback）

- 运行 fallback = **Cam-W**（pending range-bucket recall validation against R2 val）。Cam-T-only 仅在 ODD = 高速 / 直线 + TL 前向 cone 内时启用。
- Cam-W-only 性能数据登记为 §六（SAHI）/ §三（copy-paste-balance）trigger 证据，**不在 §八 内激活上述 section**。
- Fusion runtime fault（topic loss / sync drift / calib reprojection 超阈值）→ 系统自动 fallback Cam-W-only；fault event 写 `runs/_multi_camera_runtime_faults.json`。

### 模型拓扑（双相机部署时）

- **单一共享模型权重**（R2 选型胜出者）；训练数据合并 wide + tele 帧。
- Per-camera adapter head 列入 R3 contingency action item（§八 行动项 e2），不进决策表 outcome。
- 训练数据按 `camera_id ∈ {wide, tele}` 分层；aspect-ratio-aware augmentation；train/val 按 station 切分（与 SOP §3 一致）。
- 推理：TRT engine `batch=2`（每相机一帧）；单 engine / 单 sidecar / 单 eval-parity check。
- 评测必须按 `baseline_id × camera_id` stratify：每条 fusion 决策行对应 `{50mm, 250mm} × {wide, tele}` 四象限指标。

### 基线对性能的影响

- 检测模型权重：baseline-shared；两基线共用同一权重。
- Fusion accuracy：依赖 baseline + 内参 + 外参；每个 baseline 独立 `runs/_camera_calib_<baseline>mm.yaml`。
- Fusion overlap 几何：50 mm vs 250 mm 在 forward overlap 大小 / 视差余量上不同；R2 fusion A/B 必须分别报告两基线指标。

### 行动项

- [ ] a. [阻塞] 与自动驾驶团队锁定 per-baseline 相机外参 + time-sync 验证（SOP §2.3 硬件 target drift < 1 ms；fusion fault gate drift > 5 ms 持续，见决策表 `drop` 行）+ ROS2 topic 命名 + calibration reprojection error threshold。
- [ ] b. 晚期融合：投影 + WBF；per-baseline calibration YAML 加载契约；fusion runtime fault → Cam-W-only fallback 路径。
- [ ] c. A/B：(c1) Cam-W only vs 双相机融合（按 `baseline_id` stratify）；(c2) 同模型在 50 mm vs 250 mm fusion accuracy 差异；遮挡 / 远距 / 横向桶单独报告。**c1 outcome 法定有效**：若双相机融合未按本节四行决策规则优于 Cam-W-only，本轮采用 Cam-W-only 作为 **Stage 1 in-field feasibility baseline**（plan §八 已有 fusion-fault fallback 即 Cam-W-only；该 outcome 为 valid R2 / Stage 1 close 状态，非"defeat"；**但不直接 Stage 2 ship-eligible**——仍须通过 `development_plan.md` §Stage 2.A latency + §Stage 2.B quality regression），双相机本身进入后续轮次。**JSON 编码**：该 outcome 走决策表 `defer` 行，但 `runs/_multi_camera_decision.json` 必须额外写 `selected_baseline="cam_w_only"` + `stage1_scope="feasibility_baseline"` 区分于普通 `defer`（普通 `defer` 不指定 selected_baseline，等 R3 重测）。
- [ ] c5. **Diagnostic ablation**：跨相机 IoU-NMS（projected to shared frame）作为 WBF 低成本对照，仅用于验证 WBF 复杂度是否带来稳定收益；不新增验收门槛，结果进 phase report appendix。
- [ ] c6. **Offline-only diagnostic ablation**：Cam-T 远距 / 低置信触发（asymmetric far-range trigger）— 用日志回放估计 recall-latency 权衡；**R2 不作为 online 部署模式**，online 评估（含 hysteresis、engine residency、handoff failure policy）deferred to `pre_deploy_AGV_integration.md` 运行时切换条目。
- [ ] d. 写 `runs/_multi_camera_decision.json` + `runs/_camera_calib_{50mm,250mm}.yaml` + `runs/_multi_camera_runtime_faults.json`。
- [ ] e. R2 / R3 phase report 子节。
- [ ] e2. **(R3 contingency)** Per-camera adapter head — 触发条件：wide-only OR tele-only per-camera AP@0.5:0.95 lower-CI 降幅 > 2 pp OR point 降幅 > 4 pp（matched seed-pair, n ≥ 5），相对单相机训练版本。激活时 write `runs/_multi_camera_per_camera_ap.json` 并进入独立 a-stage scaffold。

### 决策规则

| outcome | condition | JSON write |
|---|---|---|
| deploy | （两基线**均**满足）遮挡 recall lower-CI +10 pp OR 远距 recall lower-CI +5 pp，且两基线 fusion accuracy（mAP@0.5 lower-CI）相对 Cam-W only 任一基线均不退化 > 2 pp | `outcome="deploy"` |
| defer | 单基线提升 / 单维度提升不显著；或一基线满足但另一基线退化 > 2 pp lower-CI；与 deploy / drop 冲突时取 defer | `outcome="defer"` |
| drop | 相机配置不允许：外参未锁定、time-sync drift > 5 ms 持续、calib reprojection error > 2 px、ROS2 topic 失联 ≥ 5 s | `outcome="drop"` |

**优先级**（drawdown 顺序）：calibration + time-sync 验证 第一；fusion A/B 第二；per-camera adapter R3 contingency 最后。SAHI / copy-paste 按各自 section 现有 trigger，不在 §八 抢占。

**Calibration 更新**：每 session 起点引用 `calib/<session_id>.yaml`；月度全标定 + 任何镜头干预 / 重装后必标（与 SOP §2.4 一致）。Calibration freshness > 30 天 OR reprojection error > 2 px → calib `stale=true`，对应 session fusion 评测降权。

## 九、自适应推理 / ROI

### 状态

DEFERRED → R3+。解锁：§五 a-stage LANDED + R3+ 推理预算评估窗口。

### 行动项

- [ ] a. `inference/cpp/` frequency-control + ROI 接口。
- [ ] b. GPS-based frequency switch + ROI crop。
- [ ] c. A/B：全帧 30 fps vs 自适应。
- [ ] d. 写 `runs/_adaptive_inference_decision.json`。
- [ ] e. R3 phase report 子节。

### 决策规则

| outcome | condition |
|---|---|
| deploy | 端到端释放 ≥ 5 ms；recall ≥ -0.5 pp |
| defer | 释放 ≤ 5 ms OR recall 退化 0.5-2 pp |
| drop | recall 退化 > 2 pp |

## 十、INT8 QAT

### 状态

DEFERRED → R3+。R2 in-round 不启动 INT8 校准 / export 改造。

### 探针

| 来源 | R2 行为 | R3+ 行为 |
|---|---|---|
| SAHI FP16 latency ∈ [50, 80) ms | 只登记 `sahi_int8_retest` | 启动 INT8 校准 + 重测 SAHI |
| TSM Phase 1-B passed 且 latency ∈ [26, 35] ms | 只登记 `tsm_int8_retest` | 启动 INT8 校准 + 重测 TSM |

TSM 探针预检查：`deploy_eligible == true`；`pretrained_init_mode == "scratch"`；Phase 1-B 精度门通过（小目标桶 recall ≥ 0.6）。

### 行动项

- [ ] a. `scripts/export_yolo.sh` / `scripts/export_deim.sh` 增加 `INT8=1`；校准集 manifest。
- [ ] b. QAT / calibration pipeline；TRT INT8 engine；sidecar `precision: int8`。
- [ ] c. A/B：FP16 vs INT8；mAP delta + speedup。
- [ ] d. 写 `runs/_int8_qat_decision.json`。
- [ ] e. R3 phase report 子节。

### 决策规则

| outcome | condition |
|---|---|
| deploy | mAP delta ≥ -0.5 pp AND speedup ≥ 1.5x |
| defer | mAP delta -0.5 至 -1.5 pp 且 SAHI 不依赖 QAT 预算 |
| drop | mAP delta < -1.5 pp |

## 十一、规划器先验融合

### 状态

DEFERRED → R3+。解锁：规划团队 topic 契约 + cross-detection framework b-stage。

### 行动项

- [ ] a. [阻塞] 与规划团队锁定 `route_intent` / `expected_signals` ROS2 topic。
- [ ] b. 在 cross-detection mean-field / CRF 框架接入 planner prior。
- [ ] c. A/B：无 planner-prior vs 有 planner-prior。
- [ ] d. 写 `runs/_planner_prior_decision.json`。
- [ ] e. R3 phase report 子节。

### 决策规则

| outcome | condition |
|---|---|
| deploy | demo10 类持续误分类率 -50%；总 mAP 不退化 |
| defer | 误分类率 -20-50% |
| drop | prior 反向 push 导致 mAP 退化 > 0.5 pp |

## 十二、排除项

| 项 | 状态 |
|---|---|
| HDR 相机 / 多曝光融合 | PM / 硬件决策 |
| DINOv2 / MAE 自监督预训练 | 依赖 R2 raw video SOP；不在本计划执行 |
| Fleet-based 主动学习 | 部署后运维 |
| 多任务 / 辅助头 | R3 候选 |
| TTA | 离线伪标签工具；部署排除 |
| 检测器集成 | 离线工具；部署排除 |
| 图像超分预处理 | R3 候选 only |
| 合成数据 / CARLA | 仅 R2 实采触达不了关键条件时回看 |
| ISP ROI AE | HDR 不可得时 fallback |

## 衔接

- `development_plan.md`：主检测器、R2 类别、部署 gate。
- `_archive/pre_r2_kickoff_checklist.md (2026-05-12 归档)`：R2 close gate、schema、carry-forward 登记格式。
- `temporal_optimization_plan.md`：TSM / HMM；§十 TSM INT8 探针同步。
- `cross_detection_reasoning_plan.md`：共现先验与 planner-prior 接入框架。
- `scripts/_r2_carry_forward_schema.json`：13-token closed enum。
- `scripts/_r2_component_decision_schema.json`：组件决策记录 schema。
