# 交通信号灯识别 — 开发计划 v2.1

## 项目目标（2026-05-12 锁定 — 二阶段交付）

**交付可在 Jetson Orin 上稳定运行的实时检测模型**，按以下二阶段推进：

### Stage 1：性能优先（performance-first，no latency gate）

目标：**最佳可达模型性能 + 稳定性**，不考虑推理延迟。

- 在不受 latency 约束的情况下争取最佳 detection quality（per-class recall、安全类 AP、红→绿零容忍）+ temporal stability（flip-flop、burst jitter、dwell time、illegal transition）。
- 训练 imgsz / 模型 size / SAHI 全开等所有手段允许；只要在 Stage 1 evidence-bounded gate 上通过即可。
- 输出：**最佳性能候选模型** + frozen eval evidence。
- 当前阶段（R2 round）锁定为 Stage 1；R2 close 走 Stage 1 evidence gate。

### Stage 2：实时优化（real-time optimization to ship）

目标：**Stage 1 最佳模型 → Orin FP16 / TensorRT 上 p95 end-to-end < 33 ms**，作为最终交付。

- 在 Stage 1 性能基础上做压缩 / 加速：KD（已在 §七 plan）、INT8 QAT（§十 deferred）、模型尺寸下调、SAHI tuning、TRT engine optimization、batch / scheduling 优化。
- Stage 2 允许牺牲 Stage 1 mAP 至预承诺下限（见 §Stage 2 quality-floor gate），换取 latency 满足。
- 阶段切换条件：Stage 1 通过其 gate 后启动；不并行。

### 共同基线

- **feasibility-first**：先证明部署可行（field test 通过），再追绝对最优。
- **2026-05-15 deadline retired**：周级排期不再驱动决策；轮次（R2 / R3）按 evidence-bounded close gate 推进。`docs/planning/timeline.md` 已归档至 `docs/_archive/`。

## 状态

| 项 | 当前值 |
|---|---|
| 部署平台 | NVIDIA Tegra Orin 64GB |
| 输入 | 异构双相机 Cam-W SG3S 1920×1536 + Cam-T SG8S 3840×2160（详 `additional_components_plan.md` §八）|
| 输出 | `vision_msgs/Detection2DArray` |
| 当前阶段 | R2：10-14 类联合检测准备 / 执行（evidence-bounded） |
| 当前 gate | §Stage 1 close gate + R2 close evidence（见下方决策规则）；Stage 2 ship gate deferred to R3 / pre-deploy |
| v2.1 规则 | WHAT-only；生命周期定义见 `additional_components_plan.md` §一；2026-05-12 retire deadline + add deployment readiness section |

| 里程碑 | 状态 / 交付物 |
|---|---|
| P1：3 类基线 | ✅ 完成；YOLO26s 选定；见 `docs/reports/phase_1_report.md` |
| R1 主力：7 类 YOLO26 n/s/m | ✅ 完成；Orin 1280 FP16 约 25 ms/帧；`xyxy` 后处理修复 |
| R1 备选：YOLOv13-s / DEIM-D-FINE-S/M | 🚧 训练中 |
| R1 跟踪：ByteTrack + EMA | ✅ Python + C++ 落地；`inference/tracker/`、`inference/cpp/src/tracker.cpp` |
| R1 决策 | ⏳ 待三轨训练完成后应用规则 |
| R2 范围 | ✅ 下限 10 类锁定；上限 14 类按 PM / 数据触发 |
| R2 数据 | ⏳ 自采数据 + SOP；R1 数据集退役 |
| R2 训练增强 | ⏳ copy-paste、硬负样本、KD P0 cells |
| R2 close | ⏳ Stage 1 close gate（性能 + 稳定，不含 latency）|
| Stage 2 ship | 待 R2 close 后启动；Orin FP16 / TRT p95 end-to-end < 33 ms |
| R2 时序优化 | gated；baseline replay 暴露失败模式后启动 |
| R3+ carry-forward | 地图先验、自适应推理、INT8 QAT、规划器先验 |

## Deferred

| 项 | 状态 | 入口 |
|---|---|---|
| 地图先验门控 | DEFERRED → R3+ | `additional_components_plan.md` §五 |
| 自适应推理 / ROI | DEFERRED → R3+ | `additional_components_plan.md` §九 |
| INT8 QAT | DEFERRED → R3+ | `additional_components_plan.md` §十 |
| 规划器先验融合 | DEFERRED → R3+ | `additional_components_plan.md` §十一 |
| 跨检测共现推理 | R3 可选 | `cross_detection_reasoning_plan.md` |

## 行动项

### 候选模型

| 模型 | 参数量 | 免NMS | 许可证 | 定位 |
|---|---:|---|---|---|
| YOLO26-n | ~2.5M | 是 | AGPL-3.0 | P1 基线；R1 证实容量不足，不部署 |
| YOLO26-s | ~9M | 是 | AGPL-3.0 | R1 部署基线 |
| YOLO26-m | ~20M | 是 | AGPL-3.0 | R1 上限基准 |
| YOLOv13-s | ~9M | 是 | AGPL-3.0 | R1 备选轨道 |
| DEIM-D-FINE-S | ~10M | 是 | Apache-2.0 | R1 / R2 Apache-2.0 备选 |
| DEIM-D-FINE-M | ~19M | 是 | Apache-2.0 | R1 / R2 精度上限 |

### 数据与类别

| 数据集 / 来源 | 状态 | R2 用途 |
|---|---|---|
| S2TLD / BSTLD / LISA | R1 后整体退役 | 不再作为 R2/R3 训练、评估、设计证据 |
| R2 自采 | 唯一 R2 基础 | 训练、验证、replay、栏杆与新增灯型 |
| raw video | 强制保留 ≥6 个月 | SSL、主动学习、时序 / 共现 / 负面控制 |

R1 7 类：

| ID | 类别 | ID | 类别 |
|---:|---|---:|---|
| 0 | `red` | 4 | `greenLeft` |
| 1 | `yellow` | 5 | `redRight` |
| 2 | `green` | 6 | `greenRight` |
| 3 | `redLeft` |  |  |

R2 目标：

| 组 | 下限 | 上限 / 条件 |
|---|---|---|
| 交通灯 | R1 7 类 + `forwardGreen` + `forwardRed` = 9 | 再加 ≤3 类；PM / 自采数据触发 |
| 栏杆 | `barrier` = 1 | `armOn` / `armOff`；每态 ≥500 实例且多样性达标 |
| 总 `nc` | 10 | 14 |

### R2 热路径

| 项 | 任务 | 状态 |
|---|---|---|
| Copy-paste + 类平衡 | R2 freeze 后 b/c/d；写 `runs/_copy_paste_decision.json` | a-stage LANDED |
| 硬负样本挖掘 | demo8/11/13 + R2 难场景；写 `runs/_hard_negative_decision.json` | a-stage LANDED |
| KD P0 cells | A1 + A3 + A2a/A2b；DEIM 路径另跑 A0 | scaffold v1.3 LANDED |
| R2 精度奇偶 | FP16/FP32 sidecar + eval-parity + decision JSON | plumbing scaffold LANDED |

### R2 采集 / 标注 / 训练

- [ ] PM 锁定最终交通灯上限类清单。
- [ ] PM / 标注负责人锁定一页标注 SOP。
- [ ] R2 自采 raw video + 抽帧数据冻结。
- [ ] 冻结 `runs/_r2_val_manifest.txt`、`runs/_r2_audit_coverage.json`。
- [ ] 冻结 `runs/_hard_negative_eval_manifest.json`。
- [ ] 应用 §R2 训练 imgsz 决策规则 → 写 `runs/_r2_train_config.json`（`imgsz` / `multi_scale` / `bbox_width_p50` / `frac_lt_0.03`）。
- [ ] 训练 10-14 类联合检测模型；单一模型输出交通灯 + 栏杆。
- [ ] **(Stage 1 optional diagnostic)** 导出 Orin TensorRT FP16 / 选定 precision engine — 仅为 Stage 1 candidate freeze 时 latency baseline 记录；**不进 R2 close gate**。
- [ ] **(Stage 1 optional diagnostic)** 完整 demo + 5-min Orin soak — 同上；Stage 2 阶段重做并升级为强制。
- [ ] Stage 1 候选冻结：写 `runs/_stage1_candidate_freeze.json`（checkpoint sha256、class map、train/eval manifest hash、preprocessing config、tracker/EMA config、calibration scope、Stage 1 metrics、owner timestamp）。
- [ ] 写 `docs/reports/phase_2_round_*.md`，coverage-gaps 列出 deferred / blocked 项，Stage 2 待启动条目按 `pre_deploy_AGV_integration.md` 接收。

## 决策规则

### R1 三轨部署

1. YOLO26s/m 在部署域评估集 mAP50 ≥ 0.60 → 备选轨道降为监控，进入 R2。
2. YOLOv13-s 相对 YOLO26s +≥ 3 pp mAP50 → 主力切换为 YOLOv13-s。
3. DEIM-D-FINE 相对 YOLO26 最佳 +≥ 5 pp mAP50 → 主力切换为 DEIM（Stage 1 性能优先；latency 由 Stage 2 评估）。（R1 历史 gate 为 ≤ 50 ms/帧，R1 决策已应用）
4. 三者差距 < 2 pp → 按许可证成本排序：DEIM > YOLOv13 > YOLO26。

### R2 训练 imgsz 决策规则（2026-05-11 锁定，替代之前的"R2 锁 imgsz=1280"无条件项）

R1 1280-训练实验显示同域 BSTLD/S2TLD/LISA val 上 s/m 在 1280 训练反降 −4.4 / −11.3 pp、仅 n 受益 +6.5 pp；该结论受限于 R1 数据分布（>80% 标签宽度 < 3%），**不能外推到 R2 部署域数据**（竖屏 / 手机 / 国内路况 + 新增 barrier 类预计 bbox 中位数显著更大）。R2 训练分辨率由数据分布决定，在 R2 manifest freeze 后执行：

```
输入：runs/_r2_val_manifest.txt + runs/_r2_train_manifest.txt
计算：
  bbox_width_p50  = R2 train+val 归一化 bbox 宽度中位数
  frac_lt_0.03    = 宽度 < 3% 的标签占比

规则：
  if frac_lt_0.03 >= 0.50:
      imgsz = 1280              # 小目标主导，与 R1 同档但数据已换 → 1280 训练
  elif frac_lt_0.03 <= 0.25 and bbox_width_p50 > 0.04:
      imgsz = 640               # 部署域 bbox 更大，省 3-4× GPU 时长
  else:
      imgsz = 960, multi_scale = True   # Ultralytics 随机 0.5-1.5× 缩放
```

输出：`runs/_r2_train_config.json`，字段 `{imgsz, multi_scale, bbox_width_p50, frac_lt_0.03, rule_branch}`。所有 R2 训练 wrapper 必须 read 该文件，不接受 hardcode 的 imgsz。

**DEIM 注意**：R1 仅有 YOLO26 的 1280-训练数据；DEIM-S/M 在 1280 训练的行为未知。R2 启动时若决策规则选 1280，DEIM 训练前 1 epoch wall-clock 必须 sanity-check（若发散即降回 960 + multi_scale）。

### Stage 1 close gate（performance-only；R2 round 走此门）

**Stage 1 = R2 round 的 ship-decision gate。无 latency 约束**。所有 ship-decision（KD A-cell、SAHI Case A、§八 fusion deploy、Cam-W-only feasibility）必须独立通过 Stage 1 A + B 方可成为 Stage 2 候选。

#### Stage 1.A — 稳定门（temporal stability）

| 指标 | 定义 | 门 |
|---|---|---|
| flip-flop rate | 每条 tracker ID 上 state 在 N 连续帧内变化次数 / 帧数（N=30，即 1 s）| < 0.05（即 1 s 内 state flip 不超过 1.5 次的轨道占比 ≥ 95%） |
| burst jitter conf delta | 同一 track 相邻两帧 detector confidence 绝对差 | p95 < 0.15 |
| min stable dwell time | 进入 `state ∈ {red, yellow, green}` 后 tracker 持续报告同 state 的最短时长 | p50 ≥ 0.5 s |
| illegal transition rate | tracker state 违反交通灯状态机（如 red → green 跳过 yellow）次数 / TL 总观测数 | < 0.001 |

测量基础：固定路线 ≥ 3 段 session（覆盖白天 / 黄昏 / 夜晚至少两档），每段 ≥ 5 min；每段单独报告，全过 + aggregate 报告。Tracker / EMA / TSM / HMM 配置须与候选模型推理 pipeline 一致；**Stage 1 不要求该 pipeline 满足 latency 约束**。

#### Stage 1.B — 质量门（detection quality floor）

| 类 | 要求 |
|---|---|
| `red` / `yellow` / `green` 主灯 | per-class recall ≥ 0.85 on field-rep eval（Cam-W in-field 数据 or R2 val 子集，二选一冻结）|
| 安全类（含 `redLeft`、`greenLeft`、其他 arrow / `barrier-up`、`barrier-down`）| recall ≥ 0.70（support ≥ 30 instance 时硬门；< 30 标 `insufficient` 不阻塞）|
| Hard-negative FP | demo8/11/13 类背景帧上 false-positive 帧率 ≤ R1 baseline + 0.02 |
| 红→绿误判 | **零容忍**：任一帧 `red → green` 跨类误判 → block ship |

Stage 1.A + Stage 1.B 都通过 → 候选进入 Stage 2 优化 pipeline；记录 frozen Stage 1 metrics 作为 Stage 2 quality-floor 参考（见下）。

### Stage 2 close gate（real-time optimization；最终交付门）

**Stage 2 = 把 Stage 1 通过的候选优化至 Orin FP16 实时可部署。Stage 2 启动于 Stage 1 candidate frozen 之后。**

**Stage 1 → Stage 2 transition artifact**：必须先冻结 `runs/_stage1_candidate_freeze.json`（字段见 §R2 采集/标注/训练 行动项）；Stage 2 优化流程仅消费冻结后的 candidate，不允许重训 Stage 1。任何 Stage 2.B 退化判定使用 freeze 文件中的 `frozen_candidate_metric` 作为参考点。

#### Stage 2.A — 实时门（latency）

| 字段 | 要求 |
|---|---|
| 平台 | Jetson AGX Orin 64GB MAXN 模式（power profile 记录于 sidecar）|
| 引擎 | TensorRT FP16（INT8 candidate 由 §十 触发后并行测）+ 对应 `*.engine.meta.json` sidecar |
| 测量协议 | warm-up ≥ 200 帧后采样 ≥ 1000 帧；report median + **p95** + p99 |
| 测量边界（INCLUDE）| GMSL2 capture / decode → image preprocess → detector forward → postprocess（NMS + decode）→ tracker update + EMA |
| 测量边界（EXCLUDE）| 视频文件读 I/O、文件写 / log、可视化绘制、ROS2 publish、外部下游 |
| 门 | **p95 end-to-end < 33 ms**（= 30 FPS）；median 单独报告但不构成门 |
| 持续性 | 5-min Orin soak；soak 全程 p95 不退化 > 10 %（防 throttling） |
| 双相机 | `batch=2` TRT 推理时延单独测；Cam-W-only mode 时延单独测；§八 c1 outcome 决定哪一个进 ship |

#### Stage 2.B — 质量回退下限（quality-floor regression cap）

Stage 2 优化（KD 压缩、INT8、SAHI tuning、TRT optim）允许损失 Stage 1 metrics，但每项指标 Stage 2 实测值不得越过 **`max(Stage 1 absolute floor − 0.05, frozen_candidate_metric − 0.05)`**（即不仅 absolute floor，**也**不允许从冻结的 Stage 1 candidate 实测值再降 0.05 以上；防止 Stage 1 强候选被压成 floor-just-passes 弱模型）：

| 指标 | Stage 1 floor | Stage 2 实测下限 |
|---|---|---|
| `red` / `yellow` / `green` 主灯 recall | ≥ 0.85 | `max(0.80, frozen − 0.05)` |
| 安全类 recall | ≥ 0.70 | `max(0.65, frozen − 0.05)` |
| Stage 1.A 稳定指标 | 见上 | 各指标退化 **≤ 25 %**（flip-flop < 0.0625；burst jitter p95 < 0.1875；dwell p50 ≥ 0.375 s；illegal transition < 0.00125）;  退化 25-50 % **不自动通过**：必须写 `runs/_stage2_temporal_escape_evidence.json`（schema：`frozen_manifest_hash` + 三组 before/after 数值字段：`red_to_green_misclass_rate {before, after}`、`track_handoff_continuity {before, after}`（=断 track 数/总 track）、`night_pwm_stability {before, after}`（夜间 session 上 conf-std on red bulb），每组 `passed_check: true/false`，三者全 `true` 方可通过；任一 `false` → block ship。退化 > 50 % 一律 block ship 无 escape |
| Hard-negative FP | ≤ R1 baseline + 0.02 | ≤ R1 baseline + 0.03 |
| 红→绿误判零容忍 | block ship | **不放宽**；track-level 违反（red→unknown→green 跨多帧）同算违反 |

通过 Stage 2.A + Stage 2.B → 进 pre_deploy_AGV_integration.md / 实车测试。

#### 失败行为（fail-loud，两阶段共用）

任一 Stage 1 / Stage 2 门失败：

1. 该候选不进入下一阶段；写 `runs/_stage<N>_readiness_fail.json`（candidate / stage / failed_gates / measured_values / time-window）。
2. 失败 cell 进 phase report 的 carry-forward 章节，按 `scripts/_r2_carry_forward_schema.json` 13-token enum 登记。
3. **不允许通过手动 cherry-pick 时段或子集来通过 gate**；test data manifest + measurement window 必须冻结写 sidecar。

### R2 close gate（evidence-bounded；R2 = Stage 1 close）

**R2 round 的 close gate ≡ Stage 1 close gate**。Stage 2 不在 R2 范围内（R3 / pre-deploy 阶段执行）。

R2 close = 以下证据全集 frozen + decision JSON 写完，**与日期无关**：

| Gate | 要求 |
|---|---|
| 数据 | R2 train / val / audit manifests frozen + sha256；frozen manifest 路径 + hash 写 phase report |
| 类别 | `nc` 在 10-14；类别映射与 frozen manifest 一致 |
| 训练 imgsz | 由 §R2 训练 imgsz 决策规则 派生，写 `runs/_r2_train_config.json` |
| 组件 | Copy-paste / hard-negative / KD P0 cells 各自决策 JSON 写出（deploy / defer / drop 任一即合规）|
| 精度奇偶 | `runs/_r2_precision_decisions.json` 完整；engine 通过 eval-parity gate；sidecar 完整 |
| **Stage 1 close** | 选定 candidate 通过 §Stage 1.A + Stage 1.B；**若双相机 fusion candidate 失败但 Cam-W-only candidate 通过 §Stage 1.A + Stage 1.B**，§八 c1 Cam-W-only feasibility outcome 是合法 R2 / Stage 1 close 状态（`selected_baseline="cam_w_only"`、`stage1_scope="feasibility_baseline"`）；若 Cam-W-only 也未过 §Stage 1.A + 1.B → 无合法 close，写 `runs/_stage1_no_viable_candidate.json` 并 carry-forward。**注意**：feasibility-baseline outcome 不是 Stage 2 ship-eligible，仍须通过 §Stage 2.A + 2.B |
| 报告 | decision JSON + coverage-gaps + carry-forward JSON 完整；Stage 2 待启动条目写入 `pre_deploy_AGV_integration.md` |

## 资源

| 环境 | 硬件 | 用途 |
|---|---|---|
| 开发 | MacBook M4 Pro 24GB | 本地开发、快速验证、CoreML 导出 |
| 训练 | GPU 服务器 / 4090 D | 完整训练、消融、KD |
| 部署 | NVIDIA Tegra Orin 64GB | TensorRT / ROS2 / 实车验证 |

## 衔接

- `pre_deploy_AGV_integration.md`：R4+ 运行时切换 / 部署调优 carry-forward 停车场。
- `additional_components_plan.md`：训练 / 推理 / 集成组件；五步生命周期定义。
- `temporal_optimization_plan.md`：TSM / HMM / GRU；仅 replay 暴露失败模式后启动。
- `cross_detection_reasoning_plan.md`：R3 同帧共现 / planner-prior 框架。
- `scripts/_r2_carry_forward_schema.json`：carry-forward 13-token closed enum。
- `scripts/_r2_component_decision_schema.json`：组件 deploy / defer / drop 结构化记录。
