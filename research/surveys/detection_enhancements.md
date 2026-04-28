# 交通灯检测增强方法可行性调研

> **状态（2026-04-27）**：研究报告。在已有三条计划（[`development_plan.md`](../../docs/planning/development_plan.md) 主检测器选型、[`temporal_optimization_plan.md`](../../docs/planning/temporal_optimization_plan.md) 时序优化、[`cross_detection_reasoning_plan.md`](../../docs/planning/cross_detection_reasoning_plan.md) 跨检测推理）之外，系统性地识别能进一步增强交通灯检测的方法。
>
> **目的**：(1) 完整列出超出现有计划范围的候选方法（模型层 + 系统层）；(2) 用 R1 实测痛点对每条做契合度评估；(3) 给出进入 R2/R3 路线图的优先级建议。
>
> **结论速览**：四条高优先级建议——稀有类 copy-paste + 类不平衡损失（训练侧）、硬负样本挖掘阶段（训练侧）、地图先验门控（集成侧）、SAHI 切片推理（推理侧，按需）；外加 HDR 相机选型 / 多相机融合两项需 PM / 自动驾驶团队决策。详见 [§5 推荐进入路线图](#5-推荐进入路线图)。

---

## 0. 调研背景与方法

### 0.1 为什么做这次调研

R1 demo replay 暴露了多类失败模式（详见 [`../reports/phase_2_round_1_report.md`](../../docs/reports/phase_2_round_1_report.md) §视检结论）：远距小目标漏检、逆光 / 黄昏漏检、卡车遮挡瞬时丢失、非正面朝向漏检、稀有类（`redRight` / `greenRight` 各 10–20 样本）模型未学到、demo8 类背景误检（黄三角警示牌、绿色厂房墙）、demo10 类持续稳定的误分类（gantry 绿灯被一致误判）。

已有三条计划各自覆盖一部分失败模式，但**仍有几个清晰的缺口**：

- **物理层缺口**：逆光 / 黄昏 / 强光斑场景没有 ML 路径能根治，需走相机硬件侧；
- **持续性误分类缺口**：稳定地错误（demo10）— 时序 / 共现都救不回来；
- **稀有类生存缺口**：R2 即便补标，`redRight` 类样本量仍可能远低于多数类，需训练侧专门保护；
- **背景误检缺口**：demo8 警示牌 / 厂房墙 — 报告中已标注但无正式计划。

调研目标：识别能补上这些缺口的方法，区分 (a) 可直接进入 R2/R3 路线图，(b) 需要 PM / 上游团队配合，(c) 已不适配当前约束的应排除项。

### 0.2 约束条件

| 约束 | 当前状态 |
|---|---|
| 部署平台 | NVIDIA Jetson AGX Orin 64GB（与完整自动驾驶栈共享 GPU） |
| 推理预算 | YOLO26-s @ 1280 实测 25 ms/帧（FP16），上限 50 ms/帧 |
| 主时间表 | 2026-05-15 实车测试截止 |
| R2 数据 | 自采（连续 30 fps），R1 三公开集（LISA / BSTLD / S2TLD）将整体废弃 |
| 类别 | R2 联合检测：10–14 类（9–12 灯型 + 1–2 栏杆），下限锁定 10 |
| 许可 | YOLO26 / YOLOv13 AGPL-3.0；DEIM Apache-2.0；公开数据集仅研究用 |

### 0.3 调研方法

- **覆盖范围**：模型层（架构 / 训练 / 损失 / 增强 / 推理时方法）+ 系统层（相机 / ISP / 多传感器 / 地图 / 主动学习 / 上下游契约）；
- **评估维度**：契合 R1 实测痛点的程度、对 Orin 预算的影响、所需数据 / 标注代价、与已有计划的兼容性；
- **不评估**：研究前沿但 5/15 截止前不可能落地的项（如端到端 VLM、事件相机硬件迁移、扩散模型大批量合成），仅在 §4 简单交代排除原因。

### 0.4 与已有计划的关系

本报告**不是**新计划。计划级别的承诺仍在 [`../planning/`](../../docs/planning/) 中。本报告的产出形式：

1. 列出可入选项；
2. 给出推荐排序；
3. 推荐项**插入到对应的现有计划**（不另开计划文档），保持路线图集中。

---

## 1. 已有计划覆盖范围（速览）

| 计划 | 覆盖维度 | 已落地 / 未落地 |
|---|---|---|
| [`development_plan.md`](../../docs/planning/development_plan.md) | 主检测器选型（YOLO26 / YOLOv13 / DEIM）、数据集、阶段目标 | R1 完成；R2 等数据 |
| [`temporal_optimization_plan.md`](../../docs/planning/temporal_optimization_plan.md) | 时间维度（跨帧）：tracker+EMA、TSM、HMM、AdaEMA、GRU、Transformer、StreamYOLO | tracker+EMA 已落地；其余按 replay 失败模式启动 |
| [`cross_detection_reasoning_plan.md`](../../docs/planning/cross_detection_reasoning_plan.md) | 空间维度（同帧）：贝叶斯共现 / CRF / Relation Network | 计划阶段；启动门待 R2 数据 |

**不在三计划范围内的方向**（本报告调研对象）：

- 训练阶段增强（augmentation、损失、采样、蒸馏、自监督预训练）；
- 推理阶段增强（切片 SAHI、TTA、集成、INT8 量化）；
- 物理 / 硬件层（HDR 相机、ISP 调优、多相机融合）；
- 上下游契约（地图先验、规划模块预期信号、主动学习闭环）；
- 数据合成 / 域随机化（CARLA、GAN）。

---

## 2. R1 实测痛点 × 已有计划覆盖矩阵

| 失败模式（R1 demo） | 已有计划覆盖 | 残余缺口 |
|---|---|---|
| 远距 / 小目标漏检（demo1, 9, 13, 14） | TSM（detector-level 时序） | 时序需连续视频；单帧侧的小目标增强（如 SAHI、tile 推理）**未列入** |
| 卡车遮挡瞬时丢失（demo3, 12） | tracker+EMA 已掩蔽；recovery ≤1 帧 | 已足够 |
| 逆光 / 黄昏 / 光斑（demo9, 11） | 无 | **物理层缺口**：HDR 相机 / ISP / 多曝光融合**未列入** |
| 非正面朝向（demo13） | 部分 TSM；部分 R2 自采补样本 | 多相机视角融合**未列入** |
| 持续稳定误分类（demo10 gantry green） | 共现推理或可救（如该灯与同帧其它灯的关系反常） | 时序方案无效；共现仅在分布支持时有效；**稳定误分类硬约束**：训练侧硬负样本 / KD 教师审校 |
| 稀有类零学习（`redRight` 19 / `greenRight` 13） | R2 重标注会增加样本但仍可能远低于多数类 | **训练侧稀有类保护机制（copy-paste、class-balanced loss）未列入** |
| 背景误检（demo8 黄警示牌 / 绿厂房墙） | 报告 §视检结论 标注 | **硬负样本挖掘流程未正式列入计划** |
| 边缘类别抖动（demo15） | EMA + HMM | 已足够 |

**四个明确缺口**：
1. 物理层（HDR / ISP / 多相机）— 系统侧；
2. 训练侧稀有类保护 — 训练侧；
3. 训练侧硬负样本挖掘 — 训练侧；
4. 单帧侧小目标增强（与 TSM 互补，但作用机制不同）— 推理侧。

后文 §3 / §4 围绕填补这四个缺口构造方法目录。

---

## 3. 模型 / 训练侧候选方法

按工作量从小到大排列；每条标注**契合的痛点**与**与已有计划的关系**。

### 3.1 Copy-paste 增强（稀有类专用）

**机制**：训练时以 bbox-level 把稀有类实例（`redRight`、`greenRight`、`forwardRed`、`forwardGreen` 等）粘贴到任意背景图，强制保证每 batch 至少出现 K 次。

**文献**：Ghiasi et al., "Simple Copy-Paste is a Strong Data Augmentation Method," CVPR 2021，COCO long-tail +6.1 AP；Ultralytics YOLO11/YOLO26 内置 `copy_paste=` 标志。

**契合痛点**：稀有类零学习（缺口 ②）。

**Orin 影响**：零（仅训练时）。

**工作量**：1–2 天（调标志 + 验证类不平衡损失协同）。

**风险**：
- copy-paste 会破坏 TL 周围合理上下文（灯通常在杆上），可能引入"飘在天上"的灯样本；
- 与 mosaic 的交互需注意，mosaic 关闭后 copy-paste 仍生效；
- 与 fliplr=0（箭头不可水平翻）兼容。

**推荐度**：⭐⭐⭐⭐⭐（直接补缺口 ②，几乎零成本）。

### 3.2 类不平衡损失（class-balanced / focal 调权）

**机制**：损失函数对每类按"有效样本数"加权（Cui et al. 2019），或对低置信度样本加 focal-α/γ 提高梯度。Ultralytics 通过 `cls=` 类权重支持，DEIM 默认 focal。

**文献**：Cui et al., "Class-Balanced Loss Based on Effective Number of Samples," CVPR 2019。

**契合痛点**：稀有类零学习（缺口 ②）。

**Orin 影响**：零。

**工作量**：1 天（计算每类权重 + 跑对照实验）。

**风险**：权重过大会让稀有类 FP 暴增；通常与 §3.1 copy-paste 联合使用，先把样本量做大再用较温和的权重。

**推荐度**：⭐⭐⭐⭐⭐（与 §3.1 是天然组合）。

### 3.3 硬负样本挖掘（hard-negative mining）

**机制**：用 R1 baseline 在 demo8 / 11 / 部分 demo13 跑推理，收集 FP 帧（特别是 `yellow` / `green` 框打在非交通灯目标上的帧），加入训练集作为**纯背景图（无标注）**或带 `ignore` 区域。Ultralytics 已支持 `bg/` 目录格式。

**文献**：经典 OHEM (Shrivastava et al., 2016)；YOLO 系列长期实践；Müller & Dietmayer 2023 在交通灯任务上明确报告硬负挖掘对 P 的提升。

**契合痛点**：背景误检（缺口 ④，按本节编号见 §2）。

**Orin 影响**：零。

**工作量**：3–5 天（挖帧 + 标注 + 重训）。

**风险**：误把"真实但难分的灯"当作硬负 → 召回掉。挖帧后必须人工核验。

**推荐度**：⭐⭐⭐⭐⭐（直接补缺口 ④，报告 §视检结论已暗示要做）。

### 3.4 知识蒸馏（M → S）

**机制**：用 DEIM-M 或 YOLO26-m 作为 teacher，对 s 档 student 加蒸馏损失（KL on logits + L2 on neck feature maps）。R1 已经训出 m 档（仅作上限对照不部署），蒸馏让 m 的精度向 s 转移而不增加 s 的推理成本。

**文献**：Hinton et al. 2015；FitNets (Romero 2015)；YOLO 系列对应 LD (Logit Distillation)，文献 +1–3 pp mAP 不增加 latency。

**契合痛点**：所有痛点（精度普遍提升）；尤其是边界混淆（提高决策余量）和持续误分类（teacher 提供修正信号）。

**Orin 影响**：零（仅训练时多一份 teacher forward）。

**工作量**：1–2 周（接入蒸馏损失 + R2 同步训练）。

**风险**：teacher 自身在难场景的偏差会传给 student；建议蒸馏权重不宜过高（α=0.3–0.5）。

**推荐度**：⭐⭐⭐⭐（"免费午餐"，与 R2 训练同步即可，不延后主线）。

### 3.5 自监督预训练（DINOv2 / MAE on R2 raw video）

**机制**：R2 自采的连续视频，绝大多数帧不需要 / 不会标注。用未标注流做 backbone 自监督预训练（DINOv2 师生自蒸馏 / MAE 图像重建），再在标注子集上 fine-tune。

**文献**：DINOv2 (Oquab 2023)；MAE (He 2021)。Domain-specific SSL 在长尾 / OOD 上 +1–3 pp mAP。

**契合痛点**：稀有类（间接，通过更好的特征表示）+ 域适应（直接）。

**Orin 影响**：零。

**工作量**：~1 周 SSL 预训练 + 主训练时间。

**风险**：
- 仅在未标注数据 ≥ 100K 帧时优于 ImageNet 预训练，否则收益小；
- 预训练 backbone 需与主检测器兼容（YOLO26 / YOLOv13 是 CSPDarknet 系，DEIM 是 HGNetv2，DINOv2 主要做 ViT —— 需谨慎选 SSL 方法）。

**推荐度**：⭐⭐⭐（**关键依赖**：R2 数据采集 SOP 必须保留 raw video，不只标注后丢弃。这是一个 SOP 决策点）。

### 3.6 多任务 / 辅助头（color × arrow 解耦）

**机制**：把单一 7/14 类分类头换成两个并行头：(a) color ∈ {R, Y, G, off}，(b) shape ∈ {round, L, R, F, none}。最终类由两头联合决定（bipartite mapping）。

**文献**：ATLAS Traffic Light Dataset (2024) 引用 pictogram-aware detection；多任务学习一般理论。

**契合痛点**：边缘类别混淆 + 持续稳定误分类（demo10 类）—— color 和 shape 用不同特征，错一项不必错另一项。

**Orin 影响**：head 略大但可忽略（< 0.5 ms）。

**工作量**：2–3 周（改 head + 重训 + label remap）。

**风险**：
- 联合头改造与主检测器架构强耦合，每个候选检测器（YOLO26 / YOLOv13 / DEIM）都要单独 patch；
- 与现有 7-class label 不直接兼容，R2 标注 SOP 要同时给 color × shape 两个标签字段（多 1× 标注成本）。

**推荐度**：⭐⭐（高潜力但工作量大；R3 候选）。

### 3.7 SAHI（Slicing Aided Hyper Inference）

**机制**：把 1280×720 输入切成 4 个 640×640（或更多）有重叠 tile，逐 tile 推理，全图 NMS-merge。**不需要重训**，仅推理时启用。

**文献**：obss/sahi 开源仓；多个 dashcam 远距小目标场景报告 +5–15 pp 小目标 AP。

**契合痛点**：远距 / 小目标漏检（缺口 ④' 推理侧补充）。

**Orin 影响**：单帧延迟 ×（tile 数 / 并发因子）。4 tile 串行 ≈ 100 ms（超预算），但若 batch=4 一次 forward 可压到 ~40 ms；与 TSM 不同，SAHI 在**单帧**上工作。

**工作量**：1 周（接入 sahi 库 + Orin 端 batch 优化 + 与已有 trtexec 引擎兼容性测试）。

**风险**：
- TRT 引擎通常 batch=1 编译；改 batch 需重新构建；
- tile 边界目标可能重复检测，依赖 NMS 兜底；
- 与 TSM 二选一更经济：TSM 改训练框架但零推理增量，SAHI 不改训练但 ~2–4× 推理延迟。

**推荐度**：⭐⭐⭐⭐（非常对症远距小目标；强烈建议**按需启用**——见 §4.2 地图先验门控）。

### 3.8 测试时增强（TTA）

**机制**：单帧多尺度（640+1280+1536）推理结果加权融合；可选水平翻转（**仅对颜色类，arrow 类禁用**因 fliplr 会翻 L/R）。

**文献**：YOLO 系列基本能力；通用 +0.5–1.5 pp。

**契合痛点**：边界混淆（弱）；不直接补强缺口。

**Orin 影响**：3× 推理延迟。

**工作量**：< 1 天（Ultralytics 内置）。

**推荐度**：⭐⭐（部署不可行；仅作为 R2 自采**伪标签生成时**的工具）。

### 3.9 检测器集成（YOLO26-s + DEIM-S）

**机制**：两个不同架构同时推理，输出 bbox 用 WBF（Weighted Box Fusion）合并，分类 logits 加权平均。

**文献**：Solovyev 2019；ensembling 通用结论 +1–3 pp。

**契合痛点**：所有维度的微提升；不补强任何一个缺口。

**Orin 影响**：~2× 推理 + 双倍模型内存。**部署不可行**（已超 50 ms 预算）。

**工作量**：~3 天。

**推荐度**：⭐（仅作为**离线** ceiling probe 或伪标签生成器；不进入部署路径）。

### 3.10 INT8 量化感知训练（QAT）

**机制**：当前 FP16；QAT 校准后 INT8 在 Orin 上 ~1.5–1.8× 推理加速（取决于算子覆盖）。释放预算给 SAHI / TSM 叠加。

**文献**：TensorRT QAT 工具链文档；YOLO 系列 INT8 校准成熟。

**契合痛点**：间接 — 释放预算允许叠加其他方法。

**Orin 影响**：负向（更快），但精度可能掉 0.5–1.5 pp。

**工作量**：~1 周（校准集准备 + 校准 + 验证精度 delta）。

**风险**：DEIM 的 deformable attention 在 INT8 下可能数值不稳；先在 YOLO26 上验证。

**推荐度**：⭐⭐⭐（**条件性推荐**——只有当 SAHI 或 TSM 落地需要释放预算时才做）。

### 3.11 图像超分预处理（小目标 crop SR）

**机制**：粗检测后对低置信度 + 小尺寸 bbox 做 crop → SR（Real-ESRGAN-tiny ~2 ms/crop）→ 二次分类。

**文献**：Real-ESRGAN (Wang 2021)；近年自动驾驶 long-range detection 偶有引用。

**契合痛点**：远距小目标（弱）。

**Orin 影响**：可变 — 取决于每帧 crop 数；多 crop 时成本累加。

**推荐度**：⭐（被 SAHI 在大多数场景上 dominated；仅在 SAHI 不可行时考虑）。

### 3.12 合成数据 / 域随机化（CARLA + 自定义信号渲染器）

**机制**：在 CARLA 里渲染 R2 实采难触达的稀有条件（强逆光、黄昏特定角度、暴雨中的灯）；标签自然来自渲染器。

**文献**：Domain randomization (Tobin 2017)；自动驾驶仿真训练大量文献。+0.5–2 pp 当真实数据稀疏。

**契合痛点**：物理层缺口的 ML-side 部分缓解（不完全补强）。

**Orin 影响**：零。

**工作量**：2–4 周（搭仿真环境 + 渲染信号灯模型 + 数据 pipeline）。

**风险**：sim-to-real gap；合成域过拟合。

**推荐度**：⭐⭐（仅当 R2 实采无法覆盖关键稀有条件时；优先实采）。

---

## 4. 系统 / 集成侧候选方法

非 ML 改造，但补强已有计划无法覆盖的缺口。

### 4.1 HDR 相机 / 多曝光融合（物理层根治）

**机制**：典型车规相机动态范围 ~70 dB；HDR 自动相机（Sony IMX490 等）120 dB+，sun + signal 同框时不饱和。或软件层做相邻帧多曝光融合（需 ISP 暴露 raw 控制）。

**契合痛点**：逆光 / 黄昏 / 光斑（缺口 ①），demo9 / 11 类全场景。

**Orin 影响**：硬件改造；ML 侧零变更。

**工作量**：决策成本高（影响整车硬件方案），实施成本中等。

**风险**：
- 现有自动驾驶硬件方案是否允许换相机；
- 软件多曝光融合需 ISP 协作。

**推荐度**：⭐⭐⭐⭐⭐（**单点最大收益**，但需 PM / 硬件团队拍板；R2 数据采集前必须确定）。

### 4.2 地图先验门控（map-prior gating）

**机制**：用 OpenStreetMap / 路网图 / 已有 HD 地图的"接近路口"信号触发：
- 接近路口 → 提高 TL 类置信度门限的合理性，触发 SAHI 或 1536 高分辨率推理；
- 远离路口 → 抑制 TL 检测（直接拒识），免除 demo8 警示牌等场景的 FP；
- 路口模板携带"该路口存在哪种灯"（基础朝向 / 是否有左右转）— 进一步过滤极不合理输出。

**文献**：Possatti 2019（HD 地图过滤候选）；Apollo perception 实践。被 [`cross_detection_reasoning_plan.md`](../../docs/planning/cross_detection_reasoning_plan.md) §5 排除是因为它解决"灯关联"而非"分类消歧"——但**作为门控信号**这条路被完全忽视。

**契合痛点**：背景误检（缺口 ④，根治 demo8 类）+ 间接给 SAHI 提供启用时机。

**Orin 影响**：~0（CPU 上的查询逻辑）。

**工作量**：~1 周（接入 GPS + OSM 查询 + 决策逻辑）。

**风险**：
- 地图未及时更新会漏路口；fallback 必须保留默认推理路径；
- 需自动驾驶团队确认 OSM / 路网信号在 ROS2 上可获得。

**推荐度**：⭐⭐⭐⭐⭐（零硬件成本，重大 FP 削减；与 §3.7 SAHI 天然搭配）。

### 4.3 主动学习闭环（fleet-based active learning）

**机制**：部署后 ROS2 节点持续记录 `top1_conf < 0.6` 或 `top1 - top2 < 0.1` 的检测，每周批次送标注 → 重训。

**文献**：Sener & Savarese 2018；Tesla Shadow Mode 实践讨论。

**契合痛点**：持续稳定误分类（缺口 ③）+ 长尾分布持续改善。

**Orin 影响**：~0（仅写日志）。

**工作量**：~1 周建立流水线 + 持续运营成本。

**风险**：标注预算 / 节奏；隐私 / 数据合规。

**推荐度**：⭐⭐⭐⭐（**长期 ROI 极高**；与 R2 数据 SOP 天然衔接）。

### 4.4 多相机融合

**机制**：自动驾驶车通常已有 front-narrow + front-wide。两路独立检测后晚期融合（同一物体的多视角投影 + score 联合）。

**契合痛点**：遮挡（一路被挡另一路可能可见）+ 远距（窄视场看小灯更清楚）+ 非正面朝向。

**Orin 影响**：~2× 检测计算（除非主路单独占线程，副路低频共享）。

**工作量**：取决于现有相机配置 — 需自动驾驶团队确认。

**风险**：相机外参标定；时间同步。

**推荐度**：⭐⭐⭐⭐（**需自动驾驶团队对齐**；可能已在他们的路线图上，确认即可）。

### 4.5 自适应推理频率 / ROI 裁剪

**机制**：当前管线 30 fps 全帧推理。改为：
- 高速无路口 → 5 fps 或完全跳过 TL 推理；
- 接近路口 → 30 fps 全帧 + SAHI 启用；
- 路口附近且可见灯 → 上半帧 ROI 推理（典型 dashcam 灯在上 1/3）。

**契合痛点**：间接 — 释放预算给其他方法叠加；同时降低非路口的 FP（部分场景下连推理都不跑就不会出 demo8 类误检）。

**Orin 影响**：负向（释放预算）。

**工作量**：~3 天管线改造。

**推荐度**：⭐⭐⭐（**预算解放器**，与 SAHI / TSM / 集成方案的预算需求互补）。

### 4.6 ISP 自动曝光的 ROI 加权

**机制**：相机 ISP 默认按全帧亮度算 AE，TL 在画面中的占比小，强光斑场景下被全局亮度策略压暗。把 AE 权重锁到画面上 1/3（典型 TL 区域）可以改善信号灯亮度保留。

**契合痛点**：逆光 / 强光斑（缺口 ①，与 §4.1 HDR 互补；非 HDR 时的次优方案）。

**Orin 影响**：~0（ISP 层）。

**工作量**：~1 周（依赖 V4L2 / GMSL ISP 控制是否暴露）。

**风险**：相机 / ISP 厂商接口差异大；可能需专属 SDK。

**推荐度**：⭐⭐⭐（**HDR 不可得时的次优**；如选 HDR 则可省）。

### 4.7 与规划模块的预期信号上下文（late fusion）

**机制**：规划模块知道当前路径意图（"下一路口左转"），可以告知感知模块"期望看到 redLeft / greenLeft"。把规划的 prior 输入到检测后处理（类似共现先验，但是 planner-conditioned）。

**文献**：Apollo perception 中有类似实现。

**契合痛点**：边界混淆（弱辅助）+ 持续误分类（demo10 — 若规划知道这条路有直行需求，gantry 上识别成 round-green 而非 forward-green 的概率被惩罚）。

**Orin 影响**：~0。

**工作量**：~1 周接口设计 + 实施。

**关键观察**：[`cross_detection_reasoning_plan.md`](../../docs/planning/cross_detection_reasoning_plan.md) 的贝叶斯框架可以**直接吸收 planner-prior**作为先验来源 — 不需要单独写一套实现。同一套 mean-field / CRF 数学，输入换成 P(class | route_intent) 即可。

**推荐度**：⭐⭐⭐（**复用 cross-detection 计划框架**；如该计划启动，加个 prior 来源极便宜）。

---

## 5. 推荐进入路线图

### 5.1 优先级排序

按"补缺口程度 × 工作量 × 5/15 时间窗"综合排序：

| 排名 | 方法 | 类型 | 补缺口 | 工作量 | 是否赶上 5/15 |
|---|---|---|---|---|---|
| 1 | Copy-paste + 类不平衡损失（§3.1+§3.2） | 训练 | 稀有类 ② | 2–3 天 | ✅ 必须，R2 训练前 |
| 2 | 硬负样本挖掘（§3.3） | 训练 | 背景误检 ④ | 3–5 天 | ✅ R2 训练前 |
| 3 | 地图先验门控（§4.2） | 集成 | 背景误检 ④ + SAHI 触发器 | ~1 周 | ✅ |
| 4 | SAHI 切片推理（§3.7） | 推理 | 远距小目标 ④' | ~1 周 | ⚠️ 5/15 后启动可接受 |
| 5 | HDR 相机决策（§4.1） | 系统 | 物理层 ① | 决策成本 | ⚠️ R2 采集前必须拍板 |
| 6 | 知识蒸馏 M→S（§3.4） | 训练 | 普遍 | 1–2 周 | ✅ R2 训练同步 |
| 7 | 主动学习闭环（§4.3） | 集成 | 持续误分类 ③ | ~1 周设置 | ⚠️ 部署后启动 |
| 8 | 多相机融合（§4.4） | 系统 | 遮挡 / 朝向 | 取决于现状 | ⚠️ 需对齐 |
| 9 | SSL 预训练（§3.5） | 训练 | 长尾 / OOD | ~1 周 + 训练 | ✅ 仅当 R2 raw 保留 |
| 10 | 自适应推理频率 / ROI（§4.5） | 集成 | 预算解放 | ~3 天 | ✅ |

5/15 之前**必须落地**的：1, 2, 3。
5/15 之前**应当决策**的：5（HDR 相机选型 — R2 采集前要敲定）、9（R2 raw video 保留 — SOP 决策点）。
5/15 之后**按 replay 启动**的：4, 6, 7, 8, 10。

### 5.2 与已有计划的衔接

| 推荐项 | 落入哪个计划 | 集成方式 |
|---|---|---|
| §3.1 Copy-paste | `development_plan.md` | 在 §三 R2 训练策略中明确启用 |
| §3.2 类不平衡损失 | `development_plan.md` | 同上 |
| §3.3 硬负样本挖掘 | `development_plan.md` | §五.1 数据问题 + §六里程碑加 R2-硬负 步骤 |
| §3.4 知识蒸馏 | `development_plan.md` | §三 R2 训练策略中加 KD 选项 |
| §3.5 SSL 预训练 | `temporal_optimization_plan.md` §0.3 数据共享 + `development_plan.md` 数据 SOP | 主要约束是 raw video 必须保留 |
| §3.7 SAHI | `temporal_optimization_plan.md` §3 备选 detector-level 路径 | 与 TSM 并列为单帧 / 时序两条小目标增强路线 |
| §4.1 HDR 相机 | `development_plan.md` §五 风险 | PM 决策项标注 |
| §4.2 地图先验门控 | `development_plan.md` §五 + `cross_detection_reasoning_plan.md` §6 关系 | 作为独立集成层 |
| §4.3 主动学习 | `development_plan.md` §六 生产化 | 长期运营路径 |
| §4.4 多相机 | `development_plan.md` §五 风险 | 自动驾驶团队对齐项 |
| §4.7 Planner-prior | `cross_detection_reasoning_plan.md` §2.2(b) 或 §6 | 作为先验来源扩展 |

### 5.3 不推荐进入路线图

| 方法 | 不推荐理由 |
|---|---|
| §3.6 多任务 / 辅助头 | 工作量数量级大，与主检测器架构强耦合，R3 候选 |
| §3.8 TTA | 部署不可行；仅作伪标签生成工具，不入路线图 |
| §3.9 检测器集成 | 部署超预算；离线工具，不入路线图 |
| §3.11 图像超分预处理 | 被 §3.7 SAHI 主导，仅在 SAHI 不可行时再考虑 |
| §3.12 合成数据 | 优先 R2 实采；仅在实采无法触达时回看 |
| §4.6 ISP ROI AE | 仅在 §4.1 HDR 不可得时的次优 |

### 5.4 已排除路径（不在路线图，研究价值有限）

| 方法 | 排除原因 |
|---|---|
| 端到端 VLM（CLIP / GroundingDINO / OWLv2） | 不在 Orin 实时预算内；开放词汇能力对固定 14 类无溢价 |
| 事件相机（DVS） | 硬件迁移成本过高；超出 R2/R3 范围 |
| 扩散模型大批量合成 | 训练成本 + sim-to-real gap；优先 §3.12 CARLA |
| 强化学习 / 主动推理 | 训练数据 / 标注假设不支持 |
| Scene-graph / 全场景 GNN | 工程复杂度过高；信噪比远低于 §4.2 地图先验 |
| 在线适应 / 测试时训练 | Orin 部署稳定性优先于精度提升；未到该问题级别 |

---

## 6. 决策点（需 PM / Lead 拍板）

按紧迫性排序：

### 6.1 R2 数据采集 SOP（5/15 倒推必须立即决策）

| 决策项 | 影响 | 推荐 |
|---|---|---|
| **保留 raw 连续视频，不仅留标注子集** | §3.5 SSL 预训练可行性；§4.3 主动学习闭环数据来源 | ✅ 强烈建议保留 |
| **数据采集是否包含逆光 / 黄昏 / 光斑场景** | §4.1 HDR 决策 / §3.12 合成数据决策 | 强烈建议主动覆盖 |
| **是否引入 color × shape 双标签**（§3.6） | 多任务头可行性 | 默认不引入，等 R3 |

### 6.2 硬件选型（R2 采集前拍板）

| 决策项 | 影响 | 推荐 |
|---|---|---|
| **HDR 相机 / 普通相机** | 物理层缺口能否根治 | 协商 — 高优先级 |
| **多相机配置** | §4.4 融合可行性 | 与自动驾驶团队对齐 |
| **ISP raw 控制是否暴露** | §4.6 ROI AE 与软件多曝光 | 视厂商 SDK |

### 6.3 上下游契约（5/15 后落地，但需现在标注接口需求）

| 决策项 | 影响 | 推荐 |
|---|---|---|
| **GPS / 路网 / OSM 信号在 ROS2 上是否可获取** | §4.2 地图先验门控可行性 | 与定位团队对齐 |
| **规划模块发布"路径意图 / 期望信号"主题** | §4.7 planner-prior 可行性 | 与规划团队对齐 |

---

## 7. 方法间相互影响

简略矩阵 — 仅标"组合显著优于单独使用"和"互斥"两种关系。

```
              §3.1  §3.2  §3.3  §3.4  §3.5  §3.7  §3.10  §4.1  §4.2  §4.3  §4.4  §4.5  §4.7
§3.1 copy-paste  -    ++    +     +     +    .     .     .    .     .    .    .     .
§3.2 cls-bal    ++    -     +     +     .    .     .     .    .     .    .    .     .
§3.3 hard-neg   +     +     -     +     .    .     .     .    +     ++   .    .     .
§3.4 KD M→S     +     +     +     -     +    .     .     .    .     .    .    .     .
§3.5 SSL        +     .     .     +     -    .     .     .    .     .    .    .     .
§3.7 SAHI       .     .     .     .     .    -     ++    .    ++    .    .    +     .
§3.10 INT8 QAT  .     .     .     .     .    ++    -     .    .     .    .    +     .
§4.1 HDR camera .     .     .     .     .    .     .     -    .     .    .    .     .
§4.2 map-prior  .     .     +     .     .   ++     .     .    -     ++   +    +     +
§4.3 active-L   .     .    ++     .     .    .     .     .    ++    -    .    .     .
§4.4 multi-cam  .     .     .     .     .    .     .     .    +     .    -    .     .
§4.5 adaptive   .     .     .     .     .    +     +     .    +     .    .    -     .
§4.7 planner-π  .     .     .     .     .    .     .     .    +     .    .    .     -
```

`++`：组合显著优于任一单独；`+`：互补；`.`：基本独立；无 `--` 互斥项（所有推荐方法可叠加）。

**关键组合**：
- §3.1 + §3.2：稀有类保护双保险（先扩样本再调权重）；
- §3.7 SAHI + §3.10 INT8：SAHI 的预算需求由 INT8 回收一部分；
- §3.7 SAHI + §4.2 map-prior：地图门控 SAHI 启用时机，避免不必要的延迟；
- §4.2 + §4.3：地图先验帮助识别"违反先验的检测"，也是主动学习闭环的天然挖掘信号；
- §4.5 自适应推理 + §3.7 SAHI / §4.2 map-prior：预算解放与按需推理协同。

---

## 8. 与已有研究 / 计划的差异

| 已有计划 / 提案 | 与本调研的关系 |
|---|---|
| [`alt_detector_architectures.md`](alt_detector_architectures.md) | 仅讨论检测器架构选型；本报告补充训练侧 / 推理侧 / 系统侧增强 |
| [`depth_estimation.md`](depth_estimation.md) | 深度估计是另一个可选增强方向（已 on hold）；本报告未重复评估 |
| [`temporal_optimization_plan.md`](../../docs/planning/temporal_optimization_plan.md) | 仅时间维度；本报告 §3.7 SAHI 提供单帧维度的小目标增强补充 |
| [`cross_detection_reasoning_plan.md`](../../docs/planning/cross_detection_reasoning_plan.md) | 仅同帧空间维度；本报告 §4.7 planner-prior 是该框架的先验来源扩展 |

---

## 9. 参考资料

### 训练增强
1. Ghiasi et al., "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation," CVPR 2021, [arXiv:2012.07177](https://arxiv.org/abs/2012.07177)
2. Cui et al., "Class-Balanced Loss Based on Effective Number of Samples," CVPR 2019, [arXiv:1901.05555](https://arxiv.org/abs/1901.05555)
3. Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017, [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)
4. Shrivastava et al., "Training Region-based Object Detectors with Online Hard Example Mining," CVPR 2016, [arXiv:1604.03540](https://arxiv.org/abs/1604.03540)
5. Hinton et al., "Distilling the Knowledge in a Neural Network," 2015, [arXiv:1503.02531](https://arxiv.org/abs/1503.02531)
6. Romero et al., "FitNets: Hints for Thin Deep Nets," ICLR 2015, [arXiv:1412.6550](https://arxiv.org/abs/1412.6550)
7. Zheng et al., "Localization Distillation for Object Detection," CVPR 2022, [arXiv:2102.12252](https://arxiv.org/abs/2102.12252)
8. Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision," 2023, [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)
9. He et al., "Masked Autoencoders Are Scalable Vision Learners," CVPR 2022, [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)

### 推理增强
10. Akyon et al., "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection," ICIP 2022, [arXiv:2202.06934](https://arxiv.org/abs/2202.06934) — SAHI 原文 + 开源仓
11. Solovyev et al., "Weighted Boxes Fusion: Ensembling Boxes from Different Object Detection Models," 2019, [arXiv:1910.13302](https://arxiv.org/abs/1910.13302)
12. NVIDIA TensorRT QAT Toolkit (Pytorch-Quantization)

### 系统 / 集成
13. Possatti et al., "Traffic Light Recognition Using Deep Learning and Prior Maps," IJCNN 2019, [arXiv:1906.11886](https://arxiv.org/abs/1906.11886)
14. Behrendt et al., "A Deep Learning Approach to Traffic Lights: Detection, Tracking, and Classification," ICRA 2017
15. Sener & Savarese, "Active Learning for Convolutional Neural Networks: A Core-Set Approach," ICLR 2018, [arXiv:1708.00489](https://arxiv.org/abs/1708.00489)
16. Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World," IROS 2017, [arXiv:1703.06907](https://arxiv.org/abs/1703.06907)
17. Apollo Perception Traffic Light Module（开源代码）

### 数据集
18. ATLAS Traffic Light Dataset, 2024, [arXiv:2504.19722](https://arxiv.org/abs/2504.19722) — pictogram 细粒度标签

### 项目内部
19. [`../reports/phase_2_round_1_report.md`](../../docs/reports/phase_2_round_1_report.md) — R1 实测痛点来源
20. [`../planning/development_plan.md`](../../docs/planning/development_plan.md) — 主路线图
21. [`../planning/temporal_optimization_plan.md`](../../docs/planning/temporal_optimization_plan.md) — 时序优化轨道
22. [`../planning/cross_detection_reasoning_plan.md`](../../docs/planning/cross_detection_reasoning_plan.md) — 跨检测共现推理
