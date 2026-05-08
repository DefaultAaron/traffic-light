# 时序优化计划（可选）— 与主检测器选型并行

> **状态（2026-04-27）**：R2/R3 可选优化轨道。本计划**不属于主检测器选型工作**，与之**并行推进**，互不阻塞。
>
> **推荐首选实验**：[§1 TSM](#1-推荐路径tsm--detector-level-时序零开销整合进检测器训练) —— 零参数、零额外推理开销、与主检测器训练流程耦合最深、文献预期收益覆盖本项目主要痛点（小目标 / 遮挡漏检）。
>
> **何时启动**：仅在主检测器（R2 baseline）实车 replay 出现具体可测量的失败模式后启动；不在选型阶段提前介入。
>
> **状态（2026-05-09）**：TSM 执行面已落地为 [`components/temporal_shift_module/`](../../components/temporal_shift_module/__init__.py) 脚手架（README v1.5；激活契约 schema [`scripts/_tsm_activation_schema.json`](../../scripts/_tsm_activation_schema.json) v1.1），在 B2 + C3 对抗审查 6 轮后达成 AGREED。runners / modules / patches / data / gates 子包均为 `NotImplementedError` 占位 —— 本计划为设计意图层，**on-disk 脚手架是实施契约**；当二者出现冲突，README 优先级高于本计划 v1 散文（README header 显式标注 §-anchor 优先级）。Phase 1-A 启动前必须写入 `runs/_tsm_activation.json` 通过 schema v1.1 校验（**6 字段必填**：`schema_version` / `selected_detector_artifact_sha256` / `selected_detector_artifact_path` / `replay_evidence_path` / `approved_failure_mode_tags` / `activation_timestamp`；其中 `approved_failure_mode_tags` 取自 **4 标签闭枚举**；并满足 SHA 三方一致性 = activation_sha == sidecar.engine_sha256 == 文件计算值）。
>
> 本文是时序优化的**可执行级集成计划**，覆盖 detector-level（§1 TSM、§3 StreamYOLO）+ post-detector（§2 HMM/AdaEMA/GRU/Transformer）两类方案。Plan A（tracker + EMA）已在 R1 落地（[`../integration/tracker.md`](../integration/tracker.md)），是 §2 路径的基线。开发计划中的轨道定位见 [`development_plan.md`](development_plan.md) §"R2 时序优化（可选并行轨道）"。

---

## 0. 定位与启动判定

### 0.1 双轨开发模型

R2 阶段开发分为两条独立轨道，互不阻塞：

| | Track 1（主线） | Track 2（本计划） |
|---|---|---|
| 目标 | 确定 R2 部署用的**检测模型** | 在确定的检测模型上**叠加时序优化** |
| 范围 | YOLO26 / YOLOv13 / DEIM-D-FINE 选型；R2 数据集（10–14 类）训练；Orin 部署 | TSM detector-level 改造、post-detector smoother（HMM/GRU/...） |
| 输出 | R2 production engine | 可选优化模块（启动后追加） |
| 时间表 | 2026-05-15 截止前完成 | **截止后**（且仅当主线就绪后实测出明确问题）才启动 |
| 数据依赖 | R2 自采 + 现有合并数据 | 与主线**同一份连续视频数据**（共享数据流） |

**关键并行性**：本计划不为主线让路，也不阻塞主线：
- 主线决定"用哪个检测器、训练参数、部署路径" —— 不需要等本计划任何决策
- 本计划只在主线产出 baseline、且 baseline 实车 replay 暴露了**特定失败模式**时启动
- 数据采集 SOP（连续 30 fps、含 track ID）由主线推进时一并落实，本计划复用

### 0.2 启动判定（按实测失败模式驱动，非按计划提前启动）

在 R2 主检测器完成训练 + Orin 部署 + 实车 replay 之前，**本计划完全不启动**。replay 之后按观测到的失败模式选择路径：

| 实测问题 | 推荐路径 | 决策依据 |
|---|---|---|
| 小目标 / 远距离 / 遮挡 / 运动模糊导致漏检 | **§1 TSM** | post-detector smoother 救不回 detector 没看到的目标 —— 必须改检测器 |
| 已有 EMA 但仍有边缘类别跳变（边缘置信度场景 EMA 锁错类） | §2.2 HMM 起步 | 检测器看到了，只是分类不稳；smoother 是对的工具 |
| 出现非法状态转移（`green → red` 无黄过渡等） | §2.2 HMM 起步 | HMM 的转移矩阵显式编码合法转移，比 GRU 更对症 |
| 上述两类问题同时存在 | **§1 + §2 组合** | TSM 在底层提升检测；smoother 在上层平滑分类 |

| 不启动条件 | 原因 |
|---|---|
| 主检测器选型未完成 | 本计划只能在 baseline 之上叠加，不能替代选型 |
| 实车 replay 未到位 | 启动判定必须基于实测数据，不基于推测 |
| `redRight` / `greenRight` 召回 = 0 | 这是训练数据问题（19 / 13 样本），时序方案不能造样本 → 走 R2 标注 |

### 0.3 数据共享（与主线一次性对齐）

无论选 §1 还是 §2，都需要**连续视频 + track ID** 数据 —— 这一约束在主线 R2 数据采集 SOP 中**应当一次性落实**，而非两次重做：

- 30 fps 连续录制（不稀疏抽帧）
- 跨地点 / 跨时段平衡（避免单一路口主导）
- 主检测器 + ByteTrack 跑伪标签 → 人工抽检 ~200 段（拒收率应 < 10%）
- 切分按视频文件，避免泄漏

数据准备细节见 [§4](#4-数据准备两条路径共用)。

---

## 1. 推荐路径：TSM — detector-level 时序，零开销，整合进检测器训练

### 1.1 机制

YOLO neck（C3 / C2f / 等价模块）的 forward 中，每个 BasicBlock 内对 1/8 channel 沿时间维度做 ±1 shift。剩余 7/8 channel 走原 spatial-only 路径：

```
原:  feat_t = Conv(feat_t)
TSM: c1, c2, c3 = split(feat_t, [1/8, 1/8, 6/8])
     c1 ← prev_feat_t.c1   (前向 shift)
     c2 ← next_feat_t.c2   (后向 shift；在线推理只用前向)
     feat_t = Conv(concat(c1, c2, c3))
```

> **2026-05-09 v1.1 修订（与 `components/temporal_shift_module/__init__.py` 一致）**：
> 上文"在线推理只用前向"会让 `c2` 在训练期吃 next_feat_t.c2、在流式推理期被零填，**1/8 channel 看到不同分布** —— 命中本项目最高风险类（Python ORT 与 C++ TRT 后处理 / 导出 parity 漂移）。本计划 v1 的位运算保留作为结构槽位，但 v1.1 起 **`c2` 在训练和推理两侧都置零**（causal end-to-end），与原始 online TSM (Lin 2019) 单向 shift 一致。受 v1.1 约束的实现细节以 `components/temporal_shift_module/__init__.py` 为准，本节落地时遵循 v1.1。

### 1.2 文献证据（VOD benchmarks）

- **TSM (ICCV 2019)**：零参数、零 FLOPs。Jetson Nano 上视频识别 76 fps（13 ms），shift 模块本身 < 1 ms 开销。在 detection 任务上 "similar or higher performance than FGFA, much smaller latency per frame"。
- **FGFA (ICCV 2017)**：ImageNet VID 上 ResNet-101 单帧 73.4 → 76.3 mAP（**+2.9%**）；fast-moving 子集 +6.2%。光流 warp 邻帧特征。**已因 FlowNet 15–30 ms 延迟排除 Orin 部署**（详见 §3.2）。
- **StreamYOLO (CVPR 2022)**：Argoverse-HD sAP +4.7% / VsAP +8.2%。详见 [§3](#3-备选-detector-level-streamyolo更重)。

通用 VOD 结论：时序聚合在整体 mAP 上 **+2–5%**，困难子集（小目标 / 遮挡 / 模糊）**+5–8%**，清晰大目标接近 0 增益。

### 1.3 TL 场景预期增益（重要校准）

文献数字不能直接套到本项目 —— TL 与 VOD benchmark 在关键维度有差异：

| 维度 | VOD benchmark | 本项目（TL on Orin） |
|---|---|---|
| 目标运动 | 多为运动物体 | TL 世界系**完全静止**；只有 ego-motion 平移 |
| 目标尺寸 | 中大占比高 | **小目标主导**（远距 TL 仅 10–20 px） |
| 遮挡 | 中等 | **频繁**（前车 / 树叶 / 行人） |

**校准后的预期**：

- 整体 mAP：**+1–3%**（不是 +6%；TL 是 image-plane 的 slow-moving）
- 小目标桶（< 20 px）recall：**+3–5%** —— 直接关联 50 m+ 制动距离
- 遮挡桶 recall：**+5–10%** —— 文献最高增益区间
- 大 / 近 / 清晰 TL：≈ 0（已饱和）

**收益最大的子集（小目标 + 遮挡）正是 Phase 1 demo 暴露的痛点** —— 这是采纳 TSM 的主要论据。

### 1.4 与主检测器训练的整合

TSM 与主线选型工作**深度耦合**：

- **不能从已训权重 fine-tune**：shift 改变了 channel 语义；R1 / R2 baseline 权重在 shift 后的 channel 上没学过对应模式 → **必须从头重训**（脚手架 `runners/__init__.py` 在 `--pretrained-init` 与 DEIM `HGNetv2.pretrained` 上声明契约要求；当前 v1.5 runner 仍为 `NotImplementedError`，Phase 1-A 实现时必须将契约落成运行时校验。详见下文 §1.4.1）。
- **训练框架与注入点因检测器而异**：YOLO 路径（YOLO26 / YOLOv13）在 Ultralytics neck 的 `C3 / C2f BasicBlock.forward` 注入 shift；DEIM 路径在 HGNetv2 backbone 的 `HG_Block.forward` 注入（详见 §1.4.1）。两条路径都**不需要自写训练循环**（与 §2.4 GRU 不同），但 dataloader 包装与 patch apply/revert 由脚手架 runner 负责。
- **数据**：与主线 R2 训练**同一份 dataset**，但 dataloader 改成 clip-of-N（默认 N=4）；clip 内顺序保持，clip 间随机。Clip 内增强必须**采样一致**（Wang 2022 原则）—— Mosaic / RandomIoUCrop / RandomZoomOut 等的采样状态在 clip 内必须复用一次，否则时序信号被破坏。脚手架 `data/__init__.py` 列出 Ultralytics 与 DEIM 两套增强管线的 clip-consistency clamp 表。
- **整合时机**：
  - **方案 1（推荐）**：主线选定基线模型后，再启动 TSM 改造作为"基线 + 时序"的 A/B 对照。两个 weight 同时训练（4090 同 GPU，分时或分卡），对比报告。
  - **方案 2**：主线训练完成、replay 验证后再启动。简单但损失 1.5 周再训时间。

#### 1.4.1 DEIM-D-FINE 适用性（2026-05-09 添加；与脚手架 v1.5 同步）

TSM 适用于 DEIM-D-FINE 的卷积 backbone，**不适用于** transformer 编码器或解码器。具体边界：

| DEIM 网络层 | TSM 注入 | 理由 |
|---|---|---|
| HGNetv2 backbone `HG_Block.forward`（`DEIM/engine/backbone/hgnetv2.py:275`） | **是** —— 在 forward 起点对 `x` 做 channel 切分 | 卷积特征，与 YOLO BasicBlock 结构同构；`in_chs` 在 HGNetv2 各 size 下都是 8 的倍数（B0 配置：64/128/256/512/1024） |
| 配置 `stage2 / stage3 / stage4`（即 `self.stages[1] / [2] / [3]`；由 `return_idx=[1,2,3]` 输出至编码器） | **是** —— 三种等价标识符 | 这三个 stage 是返回给 `HybridEncoder` 的特征源；中深层感受野适合短程时序信号 |
| HGStem（`StemBlock`，line 125）+ 配置 `stage1`（即 `self.stages[0]`；不在 `return_idx` 中） | 否 | 单 conv 入口 / 早期特征感受野过小 |
| `HybridEncoder`（`DEIM/engine/deim/hybrid_encoder.py`，含 AIFI 多头自注意力） | 否 | 注意力层；原始 TSM (Lin 2019) 仅针对卷积特征，VOD 上下文未研究混合形态 |
| D-FINE decoder transformer 层 | 否 | 纯注意力，不适用 |

**与 GO-LSD 的关系**：GO-LSD 是 decoder 自蒸馏 loss，TSM 在 backbone 层。TSM **不需要单独的 GO-LSD 开关或 wiring**，但二者并非"loss path 独立"—— TSM 改变的 backbone 特征会影响 encoder/decoder 激活，GO-LSD 梯度会回流到 patched backbone。比较 TSM-on vs TSM-off 时**两侧保持主线选定 config 的 GO-LSD 状态不变**（若主线 baseline 走 D-FINE 默认 GO-LSD-on，则两侧 on；若主线 baseline 已是 GO-LSD-off，则两侧 off）。叠加 KD A0 cell（`deim_baseline_golsd_off.py`）属于联合消融，不是独立效应相加。

**预训练权重风险**：DEIM 的 `HGNetv2.pretrained` 默认 `True`（`DEIM/engine/backbone/hgnetv2.py:443`），即使 runner 传 `--pretrained-init scratch`，HGNetv2 仍会加载 stage1 上游 HuggingFace 权重 —— 违反 §1.4 第一条 "必须从头重训"。脚手架 `runners/__init__.py` 契约要求在 model init 之前 resolve config 并断言 `HGNetv2.pretrained == False`，否则 exit 2 —— Phase 1-A 实现时必须落成此运行时校验（v1.5 runner 仍为 `NotImplementedError`）。

**Dataloader**：DEIM 走 COCO-style（`DEIM/engine/data/coco_dataset.py`）+ 自家增强管线（RandomIoUCrop / RandomZoomOut / RandomHorizontalFlip 等），与 Ultralytics 的 Mosaic-centric 不同。Clip-collator 必须包装这套上游，并对每条 clip 内的所有 T 帧复用同一组采样状态。

**DDP**：DEIM-M 的 traffic_light config 已设 `find_unused_parameters: True`。TSM 自身无可学习参数，不引入新的 unused-param 风险；但**禁止**用单 job 内 runtime flag 切换 shift on/off 的方式做消融（会触发 R1 DEIM-M ep40 同款 reducer 崩溃）—— 必须按 §1.4 方案 1 跑两个独立 weight。

### 1.5 实施步骤（首选 TSM 路径）

> **2026-05-09 同步说明**：每个 Phase 的运行入口、gate 评估器、激活契约均已在 `components/temporal_shift_module/` 脚手架声明（详见 §7 LANDED 表）。下文 checklist 为设计意图；实际运行时由对应 runner 文件的 `main()` 执行（v1.5 仍为 `NotImplementedError`，激活时再实现）。三个 Phase 共享同一激活契约：runner 启动前必须 jsonschema-validate `runs/_tsm_activation.json` 并通过 7 步算法（含 SHA 三方一致性、replay 路径存在性、`approved_failure_mode_tags ⊆ PHASE_FAILURE_MODE_SCOPE`）—— 详见脚手架 `runners/__init__.py` 与 [`scripts/_tsm_activation_schema.json`](../../scripts/_tsm_activation_schema.json)。

**Phase 1-A: 概念验证（1 周）** —— runner: [`concept_validation.py`](../../components/temporal_shift_module/runners/concept_validation.py)；gate: [`concept_validation_gate.py`](../../components/temporal_shift_module/gates/concept_validation_gate.py)

- [ ] Fork 主线选定的检测器（YOLO26 / YOLOv13 / DEIM 之一）；定位 neck 中 BasicBlock 或 HGNetv2 中 HG_Block（详见 §1.4.1）
- [ ] 实现 `TemporalShift` 模块：c1 = 1/8 channel 前向 shift（来自前帧）；c2 = 1/8 channel 训练与推理两侧均置零（**v1.1 修订 causal end-to-end，详见 §1.1**）；c3 = 6/8 channel 直通；边界 zero-pad
- [ ] dataloader 改成"clip of N=4 frames"；clip 内顺序保持，clip 内增强采样状态复用一次（Wang 2022）
- [ ] 小规模训练（10% 数据 + 20 epochs）验证 loss 下降、不 NaN
- [ ] 在小 val 集上对比 TSM-on / TSM-off 的 mAP（按 bbox 高度分桶）

**通过判定**：小目标桶 recall +2% **或** 总体 mAP +1%（脚手架 Gate-1A 实现，OR 条件）。任一达成即进入 Phase 1-B；否则回到 §0.2 重新评估问题归因（**禁止 hyperparam iteration**——失败意味着 TSM 不是对症工具）。

**诊断要求（与 activation tags 对齐 — 2026-05-09 添加）**：若 `runs/_tsm_activation.json` 的 `approved_failure_mode_tags` 包含 `far_distance_miss` / `occluded_miss` / `motion_blur` 中任一，Phase 1-A 报告必须列出对应 bucket recall delta。Gate-1A 仍按 OR 执行不变，但**不得只用 small-target recall / 总体 mAP 解释已激活的失败模式** —— 已激活但未改善的 bucket 是 §0.2 重新评估问题归因的强信号，不是 OR 条件已通过的对照。

**Bucket 定义（Phase 1-A 调度时锁定，录入激活文件 `notes` 字段，落地后不得改）**：

| Tag | 推荐定义 | 数据来源 |
|---|---|---|
| `far_distance_miss` | bbox 高度分桶 `< 20 px` / `20–40 px` / `> 40 px`（默认；640 输入。若标注含相机内参 + 平地假设，可用标定距离 `> 50 m` / `25–50 m` / `< 25 m` 等价桶） | R2 标注 bbox 高度 |
| `occluded_miss` | 遮挡比分桶 `> 70%` / `30–70%` / `< 30%`（若 R2 标注无显式遮挡比，回退到二元遮挡标志或 ByteTrack 关联失败比作为代理） | R2 遮挡标注（如有）；否则启动前由调度人指定回退源 |
| `motion_blur` | variance-of-Laplacian 分桶（阈值在 R2 静止帧子样本上分位数定，<P25 / P25–P75 / >P75）；EXIF 快门信息为辅助核对，不作主信号 | raw frame；R2 视频抽帧 |

每个激活的 tag 必须在激活文件 `notes` 字段写入选定的桶定义（含数值阈值或 fallback 来源），**Phase 1-A 启动后不得修改** —— 否则 Phase 1-A 报告与 Phase 1-B 全量训练的 bucket 对比失去基线。

**Phase 1-B: 全量训练 + 验证（1 周）** —— runner: [`full_dataset_train.py`](../../components/temporal_shift_module/runners/full_dataset_train.py)；gate: [`full_train_acceptance_gate.py`](../../components/temporal_shift_module/gates/full_train_acceptance_gate.py)

- [ ] 全 R2 数据 + 与主线同样的 epoch / patience / 增强策略
- [ ] 评估 TSM vs 单帧 baseline：总体 mAP + 按 bbox 高度分桶 recall + 遮挡 recall + Orin 实测延迟（带锁定 caveat columns：input_source / resolution / tracker_mode / output_mode）
- [ ] 通过判定：小目标桶 recall ≥ 0.6（§0.2 启动阈值）+ 整机延迟 < 26 ms（脚手架 Gate-1B 实现）

**Phase 1-C: ONNX/TRT 导出 + Orin 验证（3–5 天）** —— runner: [`streaming_engine_export.py`](../../components/temporal_shift_module/runners/streaming_engine_export.py)；gate 文件待 Phase 1-C 调度时落地（暂定名 `streaming_export_gate.py`）

- [ ] 推理路径中 TSM shift 退化为"复用上一帧 channel-1/8 特征"
- [ ] 在 ROS2 节点持有 per-stage 上一帧特征缓存（每相机一份；YOLO26-s 约 90 KB，DEIM-S/B0 约 720 KB —— 假设 640 输入、fp16、缓存全部返回 stage；DEIM-M (B2) / DEIM-L (B4) 或修改 `cache_stages` 时需重算；< 5 MB / 相机预算 per §1.6）
- [ ] ONNX export 处理 shift 算子：`Slice` + `Concat` 即可表达（无需自定义算子）
- [ ] Orin trtexec FP16 + 端到端实测延迟 < 26 ms
- [ ] 引擎 sidecar（`scripts/export_yolo.sh` / `scripts/export_deim.sh` 写出的 `<engine>.meta.json`）必须包含四个新字段 `tsm_enabled` / `tsm_shift_fraction` / `tsm_clip_size_train` / `tsm_feature_cache_stages`（**Phase 1-C 启动前需 carry-forward 到 export 脚本**）。Phase 1-C runner 不写 sidecar，只在 trtexec 返回后**校验** sidecar 存在并匹配 `args.yaml`。

#### 1.5.4 TSM 生命周期 rollup（与附加组件计划五步约定对齐）

> 与 [`additional_components_plan.md`](additional_components_plan.md) §一 五步生命周期约定一致；Phase 1-A/1-B/1-C checklist 是消融阶段（c）的细分。

- [ ] **a. 脚手架** — [`../../components/temporal_shift_module/`](../../components/temporal_shift_module/__init__.py) **LANDED v1.5，B2 + C3 已 AGREED**（详见 scaffold README + 6 轮对抗审查记录）；激活 schema [`../../scripts/_tsm_activation_schema.json`](../../scripts/_tsm_activation_schema.json) **LANDED v1.1**
- [ ] **b. 实现** — runners (`concept_validation.py` / `full_dataset_train.py` / `streaming_engine_export.py`) 替换 `NotImplementedError` 为真实代码；patches/`<detector>_block.py` 按主线选型胜者落地（v1.5 暂未落任何 patch 文件）；`runs/_tsm_activation.json` 由审批人写入并通过 7 步 schema + 三方 SHA + bucket 定义验证
- [ ] **c. 消融** — Phase 1-A 概念验证 → Gate-1A（小目标 recall +2 pp OR 总体 mAP +1 pp，OR 条件）+ §1.5 诊断要求；Phase 1-B 全量训练 → Gate-1B（小目标桶 recall ≥ 0.6 AND 整机延迟 < 26 ms，AND 条件）；Phase 1-C 导出 → 引擎 sidecar 四字段 carry-forward + Orin 实测延迟
- [ ] **d. 决策** — 三选一写入 `runs/_tsm_decision.json`（schema TBD，与 `_kd_decisions.json` / `_r2_precision_decisions.json` ≥ 80% 字段重用）：
    - **deploy**：Phase 1-A 通过 Gate-1A + Phase 1-B 通过 Gate-1B + Phase 1-C ONNX/TRT 验证通过 + Orin 延迟 < 26 ms；
    - **defer**：Phase 1-A 通过但 Phase 1-B 延迟 26–35 ms 范围（INT8 QAT 落地后重评，详见 [`additional_components_plan.md`](additional_components_plan.md) §十）；
    - **drop**：Phase 1-A 失败（小目标 recall +< 2 pp 且 mAP +< 1 pp）—— §1.5 明确要求**回到 §0.2 重新评估问题归因，禁止 hyperparam iteration**。
- [ ] **e. 报告** — `phase_R2.md`（或 R3 phase report，按时序优化轨道实际启动时间归属）TSM 子节落地，含 Phase 1-A/B/C 通过状态、bucket recall delta、对应 activation tag 解释、引擎 sidecar 验证。

### 1.6 风险

| 风险 | 缓解 |
|---|---|
| TSM 在 TL 数据上无显著增益（与 fast-moving VOD 不同） | Phase 1-A 小规模验证作为 go/no-go gate；不直接全量训练 |
| Orin 上前帧特征缓存写穿带宽 | 缓存仅 1/8 channel × 各 stage（< 5 MB / 相机）；若仍紧张可只缓存 P3/P4 不缓存 P5 |
| Clip-batch 让有效 batch_size 减半 | 学习率按 √(bs) 缩放；同 GPU 下接受更长 epoch |
| 主检测器选型仍未定，提前启动 TSM | **不要提前启动** —— TSM 与具体 backbone 强耦合，主线未定时启动 = 返工 |
| 跨视频差异让时序 pattern 不泛化 | 多场景 / 多时段训练；cross-video 验证集 |
| TSM 改造的 fork 与主线 upstream 漂移 | 用 patch / overlay 形式维护 shift 注入，避免 fork 整库 |
| **脚手架与本计划版本漂移**（2026-05-09 添加） | README header 显式声明优先级；本计划重大修订时同步更新；脚手架 README 状态字段记录每次 sync commit。**PR 检查项**（任何修改这三个文件之一的 PR 必须运行）：(1) `docs/planning/temporal_optimization_plan.md` §0 状态注的 README 版本号与 `components/temporal_shift_module/__init__.py` header 版本号一致；(2) §0 状态注与 §7 LANDED 表中 schema 版本号与 `scripts/_tsm_activation_schema.json` 的 `schema_version.const` 一致。手工 review 即可；如未来引入 CI，添加 grep-based 测试。 |
| **Phase runner 实现时遗漏 activation tripwire**（2026-05-09 添加） | tripwire 不是 prose-only：schema 文件 + 7 步算法已落地；Phase 1-A 实现 PR 必须 import jsonschema 并调用契约定义的 7 步；code review 必查这一步 |
| **export sidecar 写入者与 Phase 1-C 校验者职责分裂**（2026-05-09 添加） | 写入责任在 `scripts/export_yolo.sh` / `scripts/export_deim.sh`；校验责任在 Phase 1-C runner（exit 2 on mismatch）；二者通过 sidecar 字段集合（`tsm_*` 四个）解耦；scaffold 顶层 README 与 §1.5 同步声明 |

---

## 2. 备选路径：post-detector smoothers（仅当问题在分类稳定性，非检测精度）

> **何时选这条路**：检测器看到了目标，只是分类边缘抖动；或出现非法状态转移。**漏检问题这条路救不回来**（必须走 §1）。

### 2.1 选择哲学

所有 §2 方案吃的都是 detector 已经吐出的 `class_probs` 序列。共同前提：tracker（[已落地的 ByteTrack](../integration/tracker.md)）提供稳定 ID + 关联，本节只决定"对每条轨迹的类别序列做什么"。

按工作量从小到大：

```
Plan A (固定 EMA, 已落地)
   ↓ 不够
HMM (< 1 天)         ← §2.2 强烈推荐先试
   ↓ 不够
自适应 EMA (1-2 天)  ← §2.3
   ↓ 不够
Per-track GRU (1.5-2 周)  ← §2.4
   ↓ 不够
Transformer (在 GRU 基础上换头)  ← §2.5
```

### 2.2 HMM（隐马尔可夫模型）— 推荐先试

**机制**：每条轨迹的类别序列视为 Markov 过程。
- 转移矩阵 $A_{ij} = P(y_t = j \mid y_{t-1} = i)$（C×C 个参数；C=7 时 49 个，C=14 时 196 个）
- 观测：detector softmax 概率
- 推理：forward-backward 或 Viterbi

| 维度 | 评价 |
|---|---|
| 参数量 | ~50–200（C 取决于主线 nc） |
| 训练数据 | **不需要伪标签** —— 转移矩阵从主线训练数据 + ByteTrack 多数投票轨迹即可估 |
| Orin 延迟 | < 0.01 ms |
| TRT 复杂度 | **零** —— 不用 ONNX；纯 C++ 矩阵乘 |
| 表达能力 | 仅一阶 Markov |
| 可解释性 | 极高（转移矩阵可直接打印） |
| 实施工作量 | **< 1 天** |

**为什么先试 HMM**：
- TL 状态本来就接近一阶 Markov（red 长时停留、yellow 短暂、green 长时停留、合法转移有限）
- 上手最快、可解释性最高、没有 ONNX/TRT 风险
- 把"非法转移"问题（§0.2 第 3 行）直接吃掉

如果 HMM 已经把闪烁干到目标值，§2.3 / §2.4 都不必做。

#### 2.2.1 HMM 生命周期 rollup（2026-05-09 添加；与附加组件计划五步约定对齐）

> 与 [`additional_components_plan.md`](additional_components_plan.md) §一 五步生命周期约定一致。HMM 仅在 §0.2 启动判定行 2/3（边缘类别跳变 / 非法状态转移）实测出现后启动；与 §1 TSM 触发条件互不相同。

- [ ] **a. 脚手架** — `inference/temporal/__init__.py`（按 §7 文件改动清单 deferred）+ `inference/temporal/hmm.py` 转移矩阵 + Viterbi 接口（NEW，deferred）；C++ 集成接口 `inference/cpp/include/temporal.hpp` 与 `temporal.cpp`（deferred）。配置：在 `configs/` 中新增 `temporal_hmm.yaml` 声明 `transition_matrix_path` / `viterbi_window` 等。**当前未落地**（HMM 触发条件未达成）。
- [ ] **b. 实现** — 转移矩阵估计：从主线训练数据 + ByteTrack 多数投票轨迹离线统计（C×C 频率归一化 + Laplace 平滑）；Python 实现 forward-backward / Viterbi；C++ 推理（纯矩阵乘，无 ONNX/TRT 风险）；接入 `TrackedDetection.class_probs` 字段（与现有 EMA 二选一开关）。
- [ ] **c. 消融** — A/B 对照（Plan A 固定 EMA vs Plan A + HMM）：
    - 类别跳变率（每条 track 的 argmax-change 频率）；
    - 非法转移计数（如 `green → red` 无黄过渡）；
    - 总 mAP / 关键类别 AP 不退化（与 §1.6 同款 caveat：locked input_source / resolution / tracker_mode）；
    - 实测 Orin 延迟（HMM 应 < 0.01 ms，应在测量噪声以内）。
- [ ] **d. 决策** — 三选一写入 `runs/_hmm_decision.json`（schema TBD，与 `_kd_decisions.json` / `_r2_precision_decisions.json` 字段格式对齐）：
    - **deploy**：类别跳变率 −50% 且非法转移计数 → 0 且总 mAP / 安全类 AP 不退化（−0.2 pp 容忍）；
    - **defer**：跳变率仅降 20–50% 或非法转移仍偶发，留作 §2.3 自适应 EMA 升级时回看；
    - **drop**：跳变率 < 20% 改善（HMM 一阶 Markov 表达不足，需升级到 §2.4 GRU 或 §2.5 Transformer 路径）。
- [ ] **e. 报告** — 对应 phase report（HMM 启动时所属 round）子节，含转移矩阵可视化（矩阵打印为 §2.2 列出的可解释性强项）+ 跳变率对照 + 非法转移消除证据 + 是否进 §2.3+ 升级路径的明示理由。

### 2.3 自适应 EMA α — 最小可行可学习时序模型

**机制**：固定 α 换成由小 MLP 预测：

$$\alpha_t = \sigma(\mathrm{MLP}([\mathrm{conf}_t, H(\hat p_{t-1}), \mathrm{age}_t, \mathrm{argmax\_change}_t]))$$
$$\hat p_t = \alpha_t \cdot \mathrm{onehot}(c_t) + (1 - \alpha_t) \cdot \hat p_{t-1}$$

| 维度 | 评价 |
|---|---|
| 参数量 | ~50（4-dim 输入 → 8-dim hidden → 1-dim α） |
| 训练数据 | 与 §2.4 GRU 同（伪标签轨迹），但收敛更快 |
| Orin 延迟 | < 0.01 ms |
| 实施工作量 | 1–2 天 |

适合"想比固定 EMA 强一点但不引入 RNN"的场景。表达能力仍是一阶递归，比 §2.4 GRU 弱。

### 2.4 Per-track GRU — 文献验证最充分的路径

**前置**：HMM 与自适应 EMA 都试过仍不够。预算 1.5–2 周。

**文献支撑**：Müller & Dietmayer IV 2023 在 YOLOv5 + GRU 头 / DriveU 上报告 +2.3% state accuracy，类别跳变 −60%。

#### 2.4.1 架构

```python
# inference/temporal/gru_head.py
class GRUClassHead(nn.Module):
    """Per-track temporal smoother for class logits.

    Input  per timestep: class_probs (C,) + raw_confidence (1,) + age_bucket (1,)
                         → input_dim = C + 2
    Hidden: 32–64
    Output: smoothed class_probs (C,)

    Stateful inference: caller passes h_prev, gets h_new back.
    """
    def __init__(self, num_classes: int, hidden_dim: int = 32):
        super().__init__()
        self.gru = nn.GRU(num_classes + 2, hidden_dim, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, h0=None):
        out, h = self.gru(x, h0)
        return self.head(out), h
```

设计选择（理由不再展开，详见前版本 git history）：单层、hidden=32 起步、GRU 而非 LSTM、`input_dim = C + 2`。

#### 2.4.2 训练（必须脱离 Ultralytics）

Ultralytics 训练循环假设逐帧无状态 + 图像级 shuffle + mosaic —— 与 GRU 的 clip-内有序 + 跨 clip shuffle + BPTT 完全冲突。**自写 PyTorch 循环约 100 行**：

```python
# scripts/train_gru_head.py（核心循环）
for x, y, lens in train_loader:                 # x: (B, T, C+2), y: (B,), lens: (B,)
    logits, _ = model(x)                        # (B, T, C)
    loss_ce = sequence_ce(logits, y, lens)      # mask by valid length
    loss_tc = temporal_consistency(logits, lens)  # KL between consecutive frames
    loss = loss_ce + tc_weight * loss_tc
    loss.backward(); optim.step()
```

**损失**：sequence CE（每帧推向轨迹级伪标签）+ temporal-consistency KL（抑制相邻帧概率剧烈跳变）。

**超参数起始**：batch=64、lr=1e-3 AdamW、cosine schedule、epochs=50–80、tc_weight=0.1–0.3、clip max_len=30、hidden=32。

#### 2.4.3 ONNX / TRT 导出

固定时间展开 T=1 的单步引擎，hidden state 由调用方逐帧推进：

```python
# scripts/export_gru_head.py
dummy_x = torch.zeros(1, 1, num_classes + 2)
dummy_h = torch.zeros(1, 1, 32)
torch.onnx.export(model, (dummy_x, dummy_h), out, opset_version=14,
                  input_names=["x", "h_prev"], output_names=["logits", "h_next"])
```

```bash
trtexec --onnx=gru_head.onnx --saveEngine=gru_head_fp16.engine --fp16 \
        --shapes=x:1x1xN,h_prev:1x1x32 --noTF32
```

预期延迟：FP16 单步 < 0.05 ms / track；20 条并发 ≈ 1 ms。

#### 2.4.4 ROS2 节点状态管理

每相机一个 `TemporalSmoother`，持有 `unordered_map<tracking_id, hidden_state>`。GC：超过 `max_gap_frames` 未更新的 hidden 丢弃。NaN 保护：单条轨迹出 NaN 时回退到 raw class_probs，不影响其他轨迹。

边界情况：
- 新轨迹：h0 = zeros
- 中断重现（< max_gap）：复用 hidden（tracker 已决定 ID 重用）
- 中断 ≥ max_gap：丢弃旧 hidden
- `reset()` / 视频切换：states 清空

#### 2.4.5 与 tracker / EMA 的关系

GRU 不替代 tracker（仍需 ID 分配 + 关联 + 生命周期管理）；只替代 EMA 这一步。启用 GRU 时 `TrackSmoother.alpha = 1.0`（直接吐 raw），下游 GRU 接管。`TrackedDetection` 接口不变；`class_probs` 字段承载 GRU 输出。

### 2.5 Per-track Transformer Encoder（小）— 略强于 GRU

1–2 层 Transformer encoder，causal mask，每条轨迹一份。

| 维度 | vs GRU |
|---|---|
| 参数量 | ~3× |
| Orin 延迟 | 略高（attention O(T²)；T ≤ 8 仍 < 0.1 ms） |
| TRT 复杂度 | 中（attention 在 TRT 8.5 上 reshape 需谨慎；FP16 数值需测） |
| 表达能力 | 长程依赖更好，短序列优势不明显 |

**结论**：T ≤ 8 时 attention 相比 GRU 没有理论优势，徒增导出复杂度。仅在 §2.4 GRU 已落地但效果不够时再上。

### 2.6 已排除

- **Particle filter / 贝叶斯滤波**：与 HMM 数学等价，是 sampling 实现。HMM forward 算法在离散状态下已是最优解，没必要用 PF。
- **State-space models (S4 / Mamba)**：T ≤ 8、C ≤ 14 的问题用 SSM 是过度设计；TRT 8.5 不支持核心算子。

### 2.7 对比总表

| 方案 | 训练数据 | 参数量 | Orin 延迟 | TRT 复杂度 | 表达能力 | 何时选 |
|---|---|---|---|---|---|---|
| 固定 EMA（Plan A） | 无 | 0 | < 0.01 ms | 0 | 弱 | **已落地** |
| HMM | 转移矩阵估计 | ~50–200 | < 0.01 ms | 0 | 中（一阶 Markov） | **§2 路径首选** |
| 自适应 EMA | 伪标签 | ~50 | < 0.01 ms | 极低 | 中 | HMM 之后或并列 |
| Per-track GRU | 伪标签 | ~10 k | < 1 ms | 低 | 强（多阶非线性） | HMM/AdaEMA 不够时 |
| Per-track Transformer | 伪标签 | ~30 k | < 0.1 ms | 中 | 强 | GRU 不够时 |
| Particle Filter | 无 | 0 | < 0.01 ms | 0 | = HMM | 重复方案，不推荐 |
| SSM (S4/Mamba) | 伪标签 | ~50 k | 未知 | 高 | 强（长程） | 不在路线图 |

---

## 3. 备选 detector-level：StreamYOLO（更重）

### 3.1 何时选

§1 TSM 概念验证未达 +2% 小目标 recall **或** TSM 增益不稳定（跨视频泛化差）时考虑 StreamYOLO。预算 2–3 周。

**机制**：YOLO 头之前插入 Dual Flow Perception (DFP) 模块，输入 `(feat_{t-1}, feat_t)`，输出 `feat_t_enhanced`。Trend Aware Loss 训练时按物体速度加权。

**对本项目契合度**：
- 自动驾驶专用，Argoverse-HD 训练 / 评估（含红绿灯）
- DFP 是轻量 conv module（~5–10 ms 在 Orin 预期）
- 比 TSM 表达力强，但实现复杂度高
- **TAL 对静态 TL 收益少** —— 该损失奖励对 fast-moving 目标的预测精度，TL 几乎不动

**代价**：
- 同样从头重训，连续帧数据要求同 §1 TSM
- 比 TSM 多 ~5–10 ms 推理 —— Orin 预算紧但可承受
- TRT 导出复杂度中等：训练时"预测下一帧"逻辑必须在推理时退化为单帧 forward

### 3.2 已排除（确认排除，不再评估）

| 方案 | 排除原因 |
|---|---|
| FGFA | FlowNet 15–30 ms 吃光预算 |
| ConvLSTM backbone | TRT 8.5 RNN op 支持不稳；需自定义训练框架 |
| TransVOD / Selsa / MEGA | Heavy attention，2× detector 延迟；TL 静态不需要这种长程 |

### 3.3 候选对比（detector-level）

| 方案 | Orin 延迟 | 参数增量 | 训练框架 | 预期 TL mAP 增益 | 推荐 |
|---|---|---|---|---|---|
| **TSM** | **+ < 1 ms** | **0** | Ultralytics + patch | **+1–3% 总体 / +3–5% 小目标** | **§1 首选**（时序路径） |
| **SAHI**（单帧切片，详见 [`../../research/surveys/detection_enhancements.md`](../../research/surveys/detection_enhancements.md) §3.7） | × 2–4 推理（按需启用） | 0 | 不改训练 | +5–15% **小目标** AP（dashcam 远距） | **TSM 互补** —— 不需要连续视频；适合"未到时序数据 SOP 但已有小目标痛点"时先上 |
| StreamYOLO | + 5–10 ms | ~1 M | 上游 codebase | +1–3% / +3–5% 小目标 | TSM 失败时 |
| FGFA | + 15–30 ms | + ResNet-FlowNet | 自定义 | +2–3% | ❌ 排除（延迟） |
| ConvLSTM backbone | + 3–6 ms | ~1–3 M | 自定义 | +1.5–3% | ❌ 排除（TRT op） |
| TransVOD / Selsa | + 5–10 ms | ~5–10 M | 上游 codebase | +3–5% | ❌ 排除（attention 不稳） |

**TSM vs SAHI 选路**：

- TSM 改训练框架但**零推理增量**，依赖**连续 30 fps 视频**——主线 R2 SOP 落实后才有触发；
- SAHI 不改训练但**推理延迟 2–4×**——需要预算（INT8 QAT 解放或地图先验门控按需启用）；
- 二者目标痛点重合（远距 / 小目标），原则上**择一**先上；若 TSM 落地后小目标桶仍不够 → 再叠加 SAHI 仅在接近路口启用。

---

## 4. 数据准备（两条路径共用）

### 4.1 R2 自采视频要求（Track 1 已对齐）

无论选 §1 还是 §2，数据要求一致：

- **30 fps 连续录制**（不稀疏抽帧）
- **每段 clip ≥ 30 帧**（≥ 1 s @ 30 fps；覆盖一次状态过渡）
- **跨地点 / 跨时段平衡**
- **训练 / 验证按视频文件切分**，不按 clip
- **raw video 必须保留**（不只留标注子集）—— 也是 SSL 预训练（[`../../research/surveys/detection_enhancements.md`](../../research/surveys/detection_enhancements.md) §3.5）和主动学习闭环（同文 §4.3）的数据来源

这一约束在主线 R2 数据采集 SOP 中**应当一次性落实**。

### 4.2 伪标签生成流水线

```
R2 自采连续视频（未标注）
    │
    ▼ 主检测器（R2 baseline）逐帧推理
[per-frame Detection]
    │
    ▼ ByteTrack 关联（生产路径，已落地）
[per-track sequence: (frame_idx, class_id, conf, class_probs)]
    │
    ▼ 路径 §2: 多数投票 → 轨迹级伪标签
    ▼ 路径 §1: 校验后逐帧 bbox 仍用人工 / 主检测器输出
    │
    ▼ 人工抽检 ~200 段（拒收率 < 10%）
[clean training set]
```

**抽检不可省略**：主检测器若有系统性偏差（如远距离倾向把 redLeft 识别为 red），多数投票伪标签会**固化偏差**进 §2 GRU 或 §1 TSM 改造。每 ~200 条轨迹至少抽检 20 条比对人工标注。

### 4.3 与主检测器训练数据流的关系

- 主线 R2 训练用静态图（合并数据集 + R2 自采抽帧）—— **逐帧 bbox 标注**
- §1 TSM 改造用主线**同一份**数据，dataloader 改 clip 模式 —— **不需要新标注**
- §2 post-detector 路径用**主线产出的 baseline engine** + R2 连续视频 + ByteTrack 伪标签 —— **不需要 bbox 标注**

数据流共享 = 一次采集服务三处用途。

### 4.4 输出文件结构

```
data/
├── merged/                       # 主线 R2 训练（已存在）
│   └── images/{train,val}/
└── temporal/                     # 本计划专用（启动后建立）
    ├── train/
    │   ├── video_001/
    │   │   ├── tracks.jsonl       # 每行: {frame, tracks: [{tracking_id, class_probs, ...}]}
    │   │   └── pseudo_labels.json # {tracking_id: y_track, ...}（仅 §2 用）
    │   └── ...
    └── val/                       # 同结构
```

---

## 5. 决策流程

### 5.1 推荐尝试顺序（按实测问题驱动）

```
主检测器选型完成 + 实车 replay
    │
    ▼ §0.2 失败模式判定
        ├── 漏检（小 / 遮挡 / 模糊）   → §1 TSM        （Phase 1-A 1 周）
        ├── 边缘类别跳变 / 非法转移  → §2 post-detector （HMM 1 天起）
        └── 两者都有                 → §1 + §2 组合
              │
              ▼ 各路径内按 §1.5 / §2.1 阶梯升级
                ├── §1: TSM → StreamYOLO → 升级到 R3 + JetPack（评估 ConvLSTM/TransVOD）
                └── §2: HMM → AdaEMA → GRU → Transformer → 承认问题在 detector 层（转 §1）
```

### 5.2 与主检测器选型的协同

| 主线进度 | 本计划应当做什么 |
|---|---|
| 选型未定（YOLO26 vs YOLOv13 vs DEIM 仍在跑） | **不启动**。可以提前调研 §1 TSM 在选型候选上的可行性（patch 难度评估） |
| 选型已定，训练中 | **不启动**。可以预备 §4 数据采集 SOP（与主线一次对齐） |
| 训练完成，Orin 部署中 | **不启动**。等实车 replay |
| Replay 完成，问题已实测 | **按 §0.2 启动对应路径** |
| Plan A + 主检测器已合规 | **永久搁置**（不为优化而优化） |

### 5.3 关键判断点

- 主检测器选型决策**优先**于本计划任一选项
- §1 TSM 与主检测器架构强耦合 —— 必须在选型确定后才能改造
- §2 post-detector 与主检测器架构无关 —— 但仍需 baseline 来跑伪标签
- 两条路径**共享 R2 自采视频数据** —— 数据采集 SOP 一次到位即可

---

## 6. 风险与开放项

| 风险 | 缓解 |
|---|---|
| 主检测器选型未定时提前启动 §1 | 严格遵守 §0.1：本计划在主线就绪前不启动；只允许调研 |
| 主检测器系统性偏差被伪标签固化 | §4.2 强制人工抽检 ≥ 10%；偏差 > 10% 时手工纠正部分轨迹 |
| R2 数据稀疏抽帧（破坏连续帧前提） | 数据采集 SOP **必须**明确连续 30 fps；稀疏数据本计划无法启动 |
| Orin 上 TSM 前帧特征缓存写穿带宽 | 缓存仅 1/8 channel × 各 stage（< 5 MB / 相机）；紧张时只缓存 P3/P4 |
| §2 GRU hidden state ROS2 节点崩溃后重启丢失 | 接受现状 —— 重启后所有轨迹按新轨迹处理；前 N 帧降级 raw |
| TRT 8.5 GRU 算子在 hidden ≠ 32 时不稳定 | 固定 hidden=32；导出后 ORT vs TRT 数值 diff |
| 多相机部署 hidden / 特征缓存污染 | 每相机一个 smoother / 缓存实例 |
| 启用 §2 GRU 后某条轨迹出 NaN | 单条轨迹回退到 raw class_probs；不影响其他轨迹 |
| Plan A + 主检测器已够用，本计划投入未见回报 | §0.2 决策门严格 —— 必须有实测证据才启动 |

**开放项**（启动前需确认）：

- [ ] R2 数据采集 SOP 文档化（连续 30 fps，跨地点 / 时段平衡）
- [ ] 团队 leader 同意"5/15 后追加 1.5–3 周可选优化"
- [ ] 主检测器最终选型 → 决定 §1 TSM 改造的 fork 目标
- [ ] DriveU 公开数据集 phase 标注用于 HMM 转移矩阵先验（许可证查证）
- [ ] Orin 上时序模型与主 detector engine 是否独立加载（推荐独立，便于热回退）

---

## 7. 关键文件改动清单（启动后）

### §1 TSM 路径

> **2026-05-09 同步**：原表为启动后规划；本节按"已落地（LANDED）/ 待落地（DEFERRED）"重组，反映脚手架 v1.5 实际状态。

#### LANDED（已落地至仓库；6 次 B2+C3 审查后 AGREED）

| 文件 / 路径 | 内容 | 落地版本 |
|---|---|---|
| `components/temporal_shift_module/__init__.py` | 顶层契约：机制（§1.1，含 v1.1 causal-end-to-end 修订）、phases、acceptance gates、subpackage map、detector 适用性（YOLO + DEIM）、coupling、激活 gate、deferred deliverables、sidecar carry-forward | README v1.5 |
| `components/temporal_shift_module/modules/{__init__.py, temporal_shift.py, feature_cache.py}` | TemporalShift 算子契约 + per-stage 前帧 cache 容器；YOLO26-s（90 KB） / DEIM-S B0（720 KB）单相机内存估算 | scaffold-only |
| `components/temporal_shift_module/data/{__init__.py, clip_collator.py}` | clip-of-N=4 dataloader 契约；Ultralytics + DEIM 增强 clip-consistency clamp 表 | scaffold-only |
| `components/temporal_shift_module/patches/__init__.py` | per-detector backbone 注入契约；DEIM HG_Block 标识符三元组（config / 代码索引 / return_idx）；HybridEncoder + decoder 显式 OUT-OF-SCOPE | scaffold-only |
| `components/temporal_shift_module/runners/{__init__.py, concept_validation.py, full_dataset_train.py, streaming_engine_export.py}` | 三个 phase runner stub（NotImplementedError）+ 统一 CLI 契约 + repro 契约 + TSM-implementation 契约 + DEIM-specific notes + 7 步激活算法 | scaffold-only |
| `components/temporal_shift_module/gates/{__init__.py, concept_validation_gate.py, full_train_acceptance_gate.py}` | Gate-1A（OR 条件）+ Gate-1B（AND 条件）评估器 stub | scaffold-only |
| `scripts/_tsm_activation_schema.json` | 激活 tripwire JSON Schema：**6 字段必填**（`schema_version` / `selected_detector_artifact_sha256` / `selected_detector_artifact_path` / `replay_evidence_path` / `approved_failure_mode_tags` / `activation_timestamp`）+ 4 标签闭枚举 + path regex（仓库相对、POSIX、`.engine` 后缀） | schema v1.1 |
| `docs/planning/temporal_optimization_plan.md` §1.1 | v1.1 修订（c2 训练与推理两侧均置零；causal end-to-end） | 本文件 |

#### DEFERRED（按 phase 调度时落地；不在 v1.5 范围内）

| 文件 / 路径 | 触发时机 | 内容 |
|---|---|---|
| `components/temporal_shift_module/patches/{yolo26_basicblock.py / yolov13_basicblock.py / deim_hg_block.py}` | Phase 1-A 调度时（按主线最终选型 1 个 file） | 选定检测器的 backbone 注入实现，含 idempotent `apply()` / `revert()` |
| `components/temporal_shift_module/{modules, data, runners, gates}/*.py` 实现 | 各 phase 调度时 | 替换 `raise NotImplementedError` 为真实实现 |
| `runs/_tsm_activation.json` | Phase 1-A 启动前由审批人写入 | 激活 tripwire 数据文件（不入库） |
| `scripts/build_pseudo_labels.py` | §4.2 触发（与 §2 共用） | 主检测器 + ByteTrack 跑伪标签轨迹 |
| `inference/cpp/include/temporal_cache.hpp` / `inference/cpp/src/temporal_cache.cpp` | Phase 1-C | C++ per-camera per-stage cache 容器 + 生命周期 |
| `inference/cpp/src/trt_pipeline.cpp` | Phase 1-C | 接入 per-stage 前帧 cache |
| `scripts/export_yolo.sh` / `scripts/export_deim.sh` 新增 sidecar 字段 | Phase 1-C 启动前 carry-forward | `tsm_enabled` / `tsm_shift_fraction` / `tsm_clip_size_train` / `tsm_feature_cache_stages` |
| `tests/fixtures/temporal/` | Phase 1-A | clip-level 训练 fixture |
| `docs/integration/temporal_shift.md` | Phase 1-C 完成后 | 部署 runbook（与 [`tracker.md`](../integration/tracker.md) 同级） |

### §2 post-detector 路径

| 文件 | 改动 |
|---|---|
| `inference/temporal/__init__.py` | 新建包 |
| `inference/temporal/hmm.py` | 转移矩阵 + Viterbi（§2.2） |
| `inference/temporal/adaptive_ema.py` | 自适应 α MLP（§2.3） |
| `inference/temporal/gru_head.py` | GRUClassHead（§2.4） |
| `inference/temporal/state.py` | Per-track state container（Python） |
| `scripts/train_gru_head.py` | 训练循环（§2.4.2） |
| `scripts/export_gru_head.py` | ONNX 导出（§2.4.3） |
| `inference/cpp/include/temporal.hpp` | C++ 接口 |
| `inference/cpp/src/temporal.cpp` | TRT engine wrapper + per-track state map |
| `inference/cpp/src/demo.cpp` | `--gru-engine path` 选项；启用时旁路 EMA |
| `tests/test_gru_head.py` | Python forward vs C++ TRT 数值 diff |
| `docs/integration/temporal_smoothing.md` | 启动后产出（与 `tracker.md` 同级） |

数据：`data/temporal/{train,val}/`（不入仓库；服务器侧持有）。

---

## 8. 与现有计划的关系

- **R1**：完全不影响 R1 落地。
- **R2 主线（10–14 类联合模型 + Orin 部署）**：本计划与主线**正交并行**：
  - 主线推进时不等本计划任何决策
  - 本计划在主线 baseline + replay 之后才有触发条件
  - 数据采集 SOP 由主线推进时落实，本计划复用
- **2026-05-15 截止**：本计划**预期不在 5/15 截止前落地**。优先级低于主线。
  - 5/15 前 Plan A 已合规 → 本计划永久搁置
  - 5/15 后实测有问题 → 按 §0.2 选路径启动

---

## 9. 参考资料

### 检测器级时序（§1 / §3）

1. Lin et al., "TSM: Temporal Shift Module for Efficient Video Understanding," ICCV 2019, [arXiv:1811.08383](https://arxiv.org/abs/1811.08383) — TSM 原文；零参数、零计算、Jetson 部署友好
2. Lin et al., "TSM for Edge Devices," 2021, [arXiv:2109.13227](https://arxiv.org/abs/2109.13227) — Jetson Nano 76 fps，<1 ms 开销
3. Yang et al., "Real-time Object Detection for Streaming Perception (StreamYOLO)," CVPR 2022, [arXiv:2203.12338](https://arxiv.org/abs/2203.12338) — DFP + TAL；Argoverse-HD sAP +4.7%
4. Zhu et al., "Flow-Guided Feature Aggregation for Video Object Detection," ICCV 2017, [arXiv:1703.10025](https://arxiv.org/abs/1703.10025) — FGFA（已排除，定量参考）
5. [TSM official repo](https://github.com/mit-han-lab/temporal-shift-module) — §1 实施参考
6. [StreamYOLO official repo](https://github.com/yancie-yjr/StreamYOLO) — §3 实施参考

### Post-detector 平滑（§2）

7. Müller & Dietmayer, "Detecting Traffic Lights by Single Shot Detection," IEEE IV 2023 — YOLOv5 + GRU 头，DriveU 上 +2.3% state acc，跳变 −60%（§2.4 唯一定量依据）
8. ByteTrack, ECCV 2022 — 上游 tracker（§2 复用）
9. Apollo Perception Traffic Light Module — Plan A 事实标准
10. DriveU Traffic Light Dataset — 含 phase 标注，HMM 转移矩阵先验来源
11. Rabiner, "A Tutorial on Hidden Markov Models," Proc. IEEE 1989 — §2.2 HMM 实现参考
12. TensorRT 8.5 Support Matrix — Recurrent Layers — §2.4 GRU 导出依据

