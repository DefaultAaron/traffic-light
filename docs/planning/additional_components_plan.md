# 附加组件计划（Additional Components Plan）

**状态**：v1.0（2026-05-09；前身 `knowledge_distillation_pipeline.md` v1.3 LOCKED 内容保留为 §七；新增 8 个附加组件按推荐开发顺序排列；统一生命周期 checklist 约定。）

**适用范围**：本计划跟踪 R2 / R3 阶段在主检测器选型 + 时序优化轨道（[`temporal_optimization_plan.md`](temporal_optimization_plan.md)）+ 跨检测推理（[`cross_detection_reasoning_plan.md`](cross_detection_reasoning_plan.md)）之外的附加组件。覆盖训练侧、推理侧、系统集成侧。每个组件携带"脚手架 → 实现 → 消融 → 部署/搁置"全周期 checklist。

**排除项**（[`../../research/surveys/detection_enhancements.md`](../../research/surveys/detection_enhancements.md) §5.1 顶 10 中**不纳入本计划**）：

| 排除项 | 原因 | 决策归属 |
|---|---|---|
| HDR 相机 / 多曝光融合（survey §4.1） | 硬件决策；影响整车方案 | PM / 自动驾驶团队 |
| DINOv2 / MAE 自监督预训练（survey §3.5） | 关键依赖 R2 raw video 保留 | R2 数据采集 SOP |
| Fleet-based 主动学习闭环（survey §4.3） | 长期运营路径；部署后启动 | post-deploy 运维 |

排除项的根因属于硬件选型 / 数据 SOP / 部署后运维，单独跟踪。研究侧已排除路径详见 §十二.

---

## 一、生命周期约定

每个附加组件按以下五步生命周期推进：

| 阶段 | 含义 | 验收 |
|---|---|---|
| **a. 脚手架（Scaffold）** | 组件契约 / 配置 / runner stub 落地至 `components/` 或对应目录 | 文件存在 + `bash -n` / `python -m py_compile` 通过；如适用，B2 + C3 对抗审查至 AGREED |
| **b. 实现（Implementation）** | 占位符 (`NotImplementedError` / `[ ]`) 替换为可运行代码，能在 R2 数据上跑出输出 | 单元 fixture / 微 dataset 跑通 |
| **c. 消融（Ablation）** | 与 baseline（无该组件）做 A/B 对照，记录预承诺指标 + 95% CI | A/B 数据 + CI + 决策规则结果 |
| **d. 决策（Decision）** | 三选一：**deploy**（落地到 ship-decision）/ **defer**（推迟到 R3+ 重评，含触发条件）/ **drop**（不再追） | `runs/_<component>_decisions.json` 写入 |
| **e. 报告（Report）** | round / phase 报告子节落地，含 deploy 路径 / defer 触发条件 | phase report 关帧前完成 |

每个组件 §"清单"小节列出五步状态。

**完成度要求（按组件分类，2026-05-09 conflictor iter-1 amendment）**：
- **R2 round 关帧前必须完成 a-c**：§三 Copy-paste、§四 硬负样本、§五 地图先验、§七 KD（仅 P0 cells）；
- **完成 a 即可，b-c 视触发条件 / 外部接口落地**：§六 SAHI（5/15+，可 5/15 后启动）、§八 多相机融合（[阻塞] 自动驾驶团队）、§九 自适应推理（5/15 之前可落地，但若推理预算未紧张可推迟）、§十 INT8 QAT（条件性）、§十一 规划器先验（[阻塞] 规划团队）。这类组件 R2 round 关帧时只需提供 (1) scaffold 落地（a），(2) 当前阻塞项 / 触发条件清单，(3) defer/drop 判定依据。
- **deploy 候选须完成 d**：仅当 c 消融数据支持 deploy 决策时；report e 总是需要。

未触发的组件不属于"未完成"，应在 phase report coverage-gaps 中明确列出阻塞 / 触发条件。

---

## 二、推荐执行顺序总表

按推荐执行顺序排列（以 [`../../research/surveys/detection_enhancements.md`](../../research/surveys/detection_enhancements.md) §5.1 priority 为主依据，同时考虑依赖阻塞与日历截止）。R2 in-round **必须**完成 §三-§五；§六-§十一 按 5/15 截止节奏 / replay 反馈 / 外部依赖落地启动。

| 排名 | 章节 | 组件 | survey priority rank | 依赖 / 阻塞 | 日历约束 | 推荐级 | 工作量 |
|---|---|---|---|---|---|---|---|
| 1 | §三 | Copy-paste + 类平衡损失 | 1 | — | R2 训练前**必须** | ⭐⭐⭐⭐⭐ | 2-3 天 |
| 2 | §四 | 硬负样本挖掘 | 2 | R1 baseline 已落地 | R2 训练前**必须** | ⭐⭐⭐⭐⭐ | 3-5 天 |
| 3 | §五 | 地图先验门控 | 3 | **[阻塞]** 定位团队 GPS topic | 5/15 之前 | ⭐⭐⭐⭐⭐ | ~1 周 |
| 4 | §六 | SAHI 切片推理 | 4 | 与 TSM 二选一 | 5/15+ 可接受 | ⭐⭐⭐⭐ | ~1 周 |
| 5 | §七 | 知识蒸馏（KD） | 6（survey 排 6；HDR=5 已排除） | R2 选型胜者 + scaffold v1.3 LANDED | R2 训练同步 | ⭐⭐⭐⭐ | 1-2 周 |
| 6 | §八 | 多相机融合 | 8（survey 排 8；active-learning=7 已排除） | **[阻塞]** 自动驾驶团队 | 待对齐 | ⭐⭐⭐⭐ | 视相机配置 |
| 7 | §九 | 自适应推理 / ROI | 10 | §五 GPS 信号 | 5/15 之前可落地 | ⭐⭐⭐ | ~3 天 |
| 8 | §十 | INT8 QAT | 3.10（survey 条件性） | 仅当 §六 SAHI / TSM 需释放预算 | 触发后 ~1 周 | ⭐⭐⭐ | ~1 周 |
| 9 | §十一 | 规划器先验融合 | 4.7（survey 集成层） | **[阻塞]** 规划团队 | 与 cross-detection 计划同步 | ⭐⭐⭐ | ~1 周 |

**5/15 之前必须落地**：§三、§四、§五。
**5/15 同步并行**：§七 KD（与主线 R2 训练同 GPU；时间预算见 §七.9，不是字面"免费"，是 wall-clock < 2× scratch 的成本约束）。
**5/15 之后 by replay**：§六、§九；§八、§十、§十一视外部依赖 / 触发条件落地。

> **注释**：survey rank 6/7 对应原 survey §5.1 列表中 KD（rank 6）与 active-learning（rank 7，已排除）；本计划重排后 KD 为本表 rank 5。survey rank 5（HDR）已排除（§十二）。

**5/15 之前必须落地**：§三、§四、§五。
**5/15 同步并行**：§七 KD（与主线 R2 训练同 GPU）。
**5/15 之后 by replay**：§六、§九、§十、§十一；§八视相机配置已有/将有定。

---

## 三、Copy-paste 增强 + 类平衡损失（R2 训练前必须）

### 3.1 机制
- **Copy-paste**：训练时以 bbox-level 把稀有类实例（`redRight` / `greenRight` / `forwardRed` / `forwardGreen` 等）粘贴到任意背景图，强制保证每 batch 至少出现 K 次。Ultralytics YOLO11/YOLO26 内置 `copy_paste=` 标志；DEIM 走自定义 dataloader hook。
- **类平衡损失**：每类按"有效样本数"加权（Cui et al. 2019），或对低置信度样本加 focal-α/γ。Ultralytics 通过 `cls=` 类权重支持，DEIM 默认 focal。

### 3.2 契合痛点
稀有类零学习（R1 `redRight` 19 / `greenRight` 13 样本，模型未学到）。

### 3.3 文献
- Ghiasi et al., "Simple Copy-Paste is a Strong Data Augmentation Method," CVPR 2021，COCO long-tail +6.1 AP；
- Cui et al., "Class-Balanced Loss Based on Effective Number of Samples," CVPR 2019。

### 3.4 Orin 影响 / 工作量
- 推理零影响（仅训练时）；
- 1-2 天（copy-paste 调标志）+ 1 天（类平衡权重计算 + 对照实验） ≈ 2-3 天。

### 3.5 风险
- copy-paste 破坏 TL 周围合理上下文（灯通常在杆上），可能引入"飘在天上"的灯样本 —— 通过 mask 约束粘贴位置（限制 y-中心在画面上 1/3）减轻；
- 与 mosaic 的交互需测试；
- 必须保持 `fliplr=0`（箭头不可水平翻）；
- 类平衡权重过大会让稀有类 FP 暴增 —— 与 copy-paste 联合使用，先扩样本再用温和权重。

### 3.6 衔接
- 落入 [`development_plan.md`](development_plan.md) §三 R2 训练策略 + R2 训练 hyperparams；
- 与 §四 硬负样本挖掘**互补**（copy-paste 增 TP，硬负减 FP）；
- 与 §七 KD **互补**（KD teacher forward 在 copy-paste-augmented 图上需保持 Wang 2022 一致性）。

### 3.7 决策规则（pre-committed）
- **指标**：稀有类（`full_val_support < 30` 类）AP delta；总 mAP 不退化（参考 §七.6#1 约定）；稀有类相关 FP 计数。
- **deploy**：稀有类 AP 平均 +5 pp **或** 至少 1 个稀有类 +5 pp **AND** 所有稀有安全类 AP delta ≥ −1 pp（即不为单一类拉升而牺牲其他稀有类）**AND** 稀有类相关 FP 上升 ≤ 10%（防止"飘在天上"样本引入 FP 暴增）**AND** 总 mAP 不退化（−0.2 pp 容忍）。
- **defer**：稀有类无显著改善（< 2 pp），但总 mAP 不退化 → R3 再试更激进 copy-paste 策略（启用条件：R3 round 启动时 by R2 ship-decision 报告中明示 defer 触发，或在 phase report coverage-gaps 中作为待办项列出）。
- **drop**：总 mAP 退化 > 0.5 pp **OR** 任一稀有安全类 AP 退化 > 1 pp **OR** 稀有类相关 FP 上升 > 30%。

### 3.8 清单
- [ ] **a. 脚手架**：训练 config 增加 `copy_paste` / `cls_weight` 字段；R2 nc 范围确定后计算每类有效样本数表写入 `configs/data_R2_class_weights.yaml`
- [ ] **b. 实现**：YOLO Ultralytics 直接启用；DEIM 需 dataloader hook（参考 §3.5 风险中的 mask 约束）
- [ ] **c. 消融**：A/B 对照（无 copy-paste vs 仅 copy-paste vs copy-paste + class-balanced），记录稀有类 AP 与总 mAP（95% CI 由 §七.6#1 共用工具）
- [ ] **d. 决策**：按 §3.7 规则三选一，写入 `runs/_copy_paste_decision.json`
- [ ] **e. 报告**：phase R2 子节落地

---

## 四、硬负样本挖掘（R2 训练前必须）

### 4.1 机制
用 R1 baseline 在 demo8 / 11 / 部分 demo13 跑推理，收集 FP 帧（`yellow` / `green` 框打在非交通灯目标上的帧），加入训练集作为**纯背景图（无标注）**或带 `ignore` 区域。Ultralytics 已支持 `bg/` 目录格式；DEIM 走 COCO-style empty-image 写入。

### 4.2 契合痛点
背景误检（demo8 黄警示牌、绿厂房墙；demo11 误检；demo13 一部分误检）。R1 报告 §视检结论已暗示要做。

### 4.3 文献
- Shrivastava et al., "OHEM: Online Hard Example Mining," CVPR 2016；
- Müller & Dietmayer, IV 2023（交通灯任务上明确报告硬负挖掘对 P 提升）。

### 4.4 Orin 影响 / 工作量
- 推理零影响；
- 3-5 天（挖帧 + 人工核验 + 重训）。

### 4.5 风险
- 误把"真实但难分的灯"标记为硬负 → 召回掉。**挖帧后必须人工核验**，建议每 200 帧抽检 ≥ 20。

### 4.6 衔接
- 落入 [`development_plan.md`](development_plan.md) §五.1 数据问题 + §六 里程碑加 R2-硬负 步骤；
- 与 §五 地图先验门控**互补**（地图先验 prevents 路口外推理；硬负减少推理时的 FP）；
- 与 §七.6#3 KD 验收门**共用 manifest**（demo8/11/13 背景帧）。

### 4.7 决策规则（pre-committed）
- **冻结评估 manifest**：训练前冻结 `runs/_hard_negative_eval_manifest.json`（demo8/11/13 + R2 自采难场景帧 + 真实灯标注帧的混合 manifest），含 image SHA256 + 标注源 + 评估阈值（confidence ≥ 0.25，NMS IoU = 0.5；与 R2 精度奇偶 eval-parity gate 同款）。**评估时不得改 manifest 或阈值**——保护 FP 下降率 denominator 不被换 frame 子集 game。
- **指标**：在冻结 manifest 上的 FP 数；真实灯 recall delta；总 mAP 不退化。
- **deploy**：FP 数下降 ≥ 50%（demo8 类背景帧）**AND** 真实灯 recall delta ≥ −0.5 pp（关键安全 floor，防止挖错样本）**AND** 总 mAP 不退化（−0.2 pp 容忍）。
- **defer**：FP 仅降 20-50% **AND** 真实灯 recall delta ≥ −0.5 pp，留作 R3 重评（启用条件：R2 报告 coverage-gaps 列出，R3 round 启动时回看）。
- **drop**：真实灯 recall delta < −0.5 pp（即真实灯 recall 退化 > 0.5 pp） **OR** FP 下降 < 20%（挖掘没效）。

### 4.8 清单
- [ ] **a. 脚手架**：硬负挖掘脚本 `scripts/_mine_hard_negatives.py`（NEW，deferred）；输入 R1 baseline + demo 视频，输出候选 FP 帧 manifest
- [ ] **b. 实现**：人工核验流程（≥ 10% 抽检）；R2 训练集 `bg/` 目录（YOLO）/ DEIM empty-image 写入
- [ ] **c. 消融**：A/B 对照（无硬负 vs 有硬负），demo8/11/13 FP 计数 + 真实灯 recall
- [ ] **d. 决策**：按 §4.7 规则三选一，写入 `runs/_hard_negative_decision.json`
- [ ] **e. 报告**：phase R2 子节落地

---

## 五、地图先验门控（5/15 之前）

### 5.1 机制
用 OpenStreetMap / 路网图 / 已有 HD 地图的"接近路口"信号触发：
- 接近路口 → 触发 SAHI 或 1536 高分辨率推理；
- 远离路口 → 抑制 TL 检测（直接拒识），免除 demo8 警示牌等场景的 FP；
- 路口模板携带"该路口存在哪种灯"（基础朝向 / 是否有左右转）—— 进一步过滤极不合理输出。

### 5.2 契合痛点
背景误检（根治 demo8 类）+ 给 SAHI 提供启用时机。

### 5.3 文献
- Possatti et al., "Traffic Light Recognition Using Deep Learning and Prior Maps," IJCNN 2019；
- Apollo Perception 实践。

### 5.4 Orin 影响 / 工作量
- ~0（CPU 上的查询逻辑）；
- ~1 周（接入 GPS + OSM 查询 + 决策逻辑）。

### 5.5 风险
- 地图未及时更新会漏路口 —— **fallback 必须保留默认推理路径**（即未匹配到路口时不抑制）；
- 需自动驾驶团队确认 OSM / 路网信号在 ROS2 上可获得（**[需对齐]**）；
- **GPS drift / OSM stale / 隧道 / 城市峡谷 / 临时信号灯**：地图 fallback 仅覆盖"未匹配到路口"情况，不覆盖 GPS 抖动 + 路口数据陈旧 + 临时信号 = "GPS 在路网中找到错误路口"的情况。决策规则需配套负面控制（§5.7）。
- **依赖锁定 deadline**：若 5/12 之前 GPS / topic 契约未与定位团队锁定，则 §五 实现进入 replay-only fallback：使用 R1 / R2 已采集视频附带的 GPS 录制 + 静态路线 annotation，**不能宣称生产环境 deploy** —— SAHI 触发与自适应推理（§六、§九）只能进 R2 phase report 的 "replay-evidence-only" 子节，不能落生产 ship-decision。

### 5.6 衔接
- 落入 [`development_plan.md`](development_plan.md) §五 + [`cross_detection_reasoning_plan.md`](cross_detection_reasoning_plan.md) §6 关系；作为独立集成层；
- 与 §六 SAHI **天然搭配**（地图门控 SAHI 启用时机，避免不必要的延迟）；
- 与 §四 硬负样本挖掘**互补**。

### 5.7 决策规则（pre-committed）
- **冻结评估场景**：评估必须包含负面控制场景：
  - (a) 施工 / 临时信号灯路段（地图未标）；
  - (b) GPS jitter 注入（±20 m 偏移模拟弱信号）；
  - (c) map-missing intersection（OSM 未标记的实际路口）。
  这些控制场景的安全关键漏检事件**不得为零以上**——任何 missed safety-critical light 在控制场景中即触发 drop。
- **指标**：路口外 FP 数；路口内 recall（按 GPS 标定）；负面控制场景的 missed safety-critical light 计数。
- **deploy**：路口外 FP −80% **AND** 路口内 recall 不退化（−0.5 pp 容忍）**AND** 负面控制场景 missed safety-critical = 0。
- **defer**：路口外 FP −50% **AND** 路口内 recall 退化 0.5-2 pp **AND** 负面控制场景 missed safety-critical ≤ 1 per replay hour，留作 R3 重评门限。
- **drop**：路口内 recall 退化 > 2 pp **OR** 负面控制场景出现 ≥ 1 missed safety-critical light（GPS drift / 临时信号触发 silent recall failure 是 drop 强信号；不接受 silent failure mode）。

### 5.8 清单
- [ ] **a. 脚手架**：与定位团队对齐 ROS2 GPS topic 契约（**[需对齐]**）；`inference/cpp/include/map_prior_gate.hpp`（NEW，deferred）声明 `MapPriorGate` 接口；fallback 行为定义
- [ ] **b. 实现**：OSM 查询 + 路口 polygon 匹配；`MapPriorGate.is_near_intersection(gps)` 返回 bool
- [ ] **c. 消融**：A/B 对照（无门控 vs 路口外抑制 vs 路口外抑制 + 路口内 SAHI 触发）
- [ ] **d. 决策**：按 §5.7 规则三选一，写入 `runs/_map_prior_decision.json`
- [ ] **e. 报告**：phase R2 子节落地

---

## 六、SAHI 切片推理（5/15+ 可接受）

### 6.1 机制
把 1280×720 输入切成 4 个 640×640（或更多）有重叠 tile，逐 tile 推理，全图 NMS-merge。**本计划仅评估 inference-only SAHI**（不需要重训），与 §六.7 的小目标 recall 决策门绑定。SAHI 原文也提供 slicing-aided fine-tuning 选项；若 inference-only SAHI 在 tile 边界 / scale shift 上效果不足、但 slicing-aided fine-tuning 可救，作为单独的 R3 候选另开（**当前不默认纳入**，理由：fine-tuning 路径已被 §1 TSM 单帧时序占据预算）。

### 6.2 契合痛点
远距 / 小目标漏检（与 TSM 互补：TSM 时序、SAHI 单帧）。

### 6.3 文献
- Akyon et al., "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection," ICIP 2022。

### 6.4 Orin 影响 / 工作量
- 单帧延迟 ×（tile 数 / 并发因子）。4 tile 串行 ≈ 100 ms（超预算），但若 batch=4 一次 forward 可压到 ~40 ms。**与地图先验门控按需启用**减轻；
- ~1 周（接入 sahi 库 + Orin 端 batch 优化 + 与 trtexec 引擎兼容性测试）。

### 6.5 风险
- TRT 引擎通常 batch=1 编译；改 batch 需重新构建（与 R2 精度奇偶 sidecar 兼容性需测）；
- tile 边界目标可能重复检测，依赖 NMS 兜底；
- **与 TSM 二选一更经济**：TSM 改训练框架但零推理增量，SAHI 不改训练但 ~2-4× 推理延迟。两者均落地时按 §五 地图先验只在路口附近启用 SAHI。

### 6.6 衔接
- 落入 [`temporal_optimization_plan.md`](temporal_optimization_plan.md) §3 备选 detector-level 路径（与 TSM 并列为单帧 / 时序两条小目标增强路线）；
- 与 §五 地图先验门控**天然搭配**；
- 与 §十 INT8 QAT **互补**（SAHI 的预算需求由 INT8 回收一部分）。

### 6.7 决策规则（pre-committed）
- **指标**：小目标桶（bbox 高度 < 20 px）recall delta；端到端 Orin 延迟（locked caveat columns 同 R2 精度奇偶 + temporal）。
- **deploy**：小目标 recall +5 pp + 在路口接近门控下端到端延迟 < 50 ms（上限）；
- **defer**：小目标 recall +2-5 pp 但延迟超 50 ms，留作 INT8 QAT 落地后重评；
- **drop**：小目标 recall +2 pp 以下 OR 延迟 > 60 ms 不可恢复。

### 6.8 清单
- [ ] **a. 脚手架**：`inference/cpp/include/sahi.hpp`（NEW，deferred）声明 4-tile 切片接口；与现有 `trt_pipeline.cpp` 兼容性 review；与 §五 `MapPriorGate` 触发链接
- [ ] **b. 实现**：sahi 库接入（Python 验证）+ C++ 端实现 + batch=4 引擎构建
- [ ] **c. 消融**：A/B 对照（无 SAHI vs 路口内 SAHI vs 全程 SAHI），分别测远距桶 recall + 延迟
- [ ] **d. 决策**：按 §6.7 规则三选一，写入 `runs/_sahi_decision.json`
- [ ] **e. 报告**：phase R2 / R3 子节落地

---

## 七、知识蒸馏（KD）— Round R2 in-round 主任务

> **本节内容继承 v1.3 LOCKED**（前身 `knowledge_distillation_pipeline.md`），保留完整决策规则 / cell 矩阵 / 验收门 / 4090 D 容量分析 / runner 映射。Scaffold 已落地至 [`../../components/knowledge_distillation/`](../../components/knowledge_distillation/__init__.py)（README v1.3，B2 + C3 已 AGREED）。本节内部 §1-§14 编号沿用前身（避免内部交叉引用断链）；本计划文档级 §-anchor 为 §七，其下子节使用 §七.1、§七.2 等。

### 七.1 背景与触发条件

[`development_plan.md`](development_plan.md) §R2 训练增强 第 133 行有一行条目（"知识蒸馏 M→S 条件性 — DEIM-M / YOLO26-m 作 teacher，KL on logits + 浅层 feature L2"）。本节将其展开为完整流水线，并显式回答 PM 三个核心问题：

1. **Q1**：教师是否应扩展至更大 (L/XL)？
2. **Q2**：是否需要多教师，还是单教师足够？
3. **Q3**：是否做跨架构迁移（DEIM ↔ YOLO）？

### 七.2 Q1：教师规模 — 是否应扩展至 L/XL?

**证据要点**：
- Cho & Hariharan, 2019 (ICCV)：分类领域"大教师不必然更好"；缓解为 ESKD 早停教师。
- Mirzadeh et al., 2020 (AAAI, TAKD)：中等规模 TA 桥接大跨度 capacity gap。
- Cao et al., 2022 (NeurIPS, PKD)：MaskRCNN-Swin → R-50 RetinaNet/FCOS +4.1/+4.8 pp（异构+跨度同时成立）。
- Cao et al., 2023 (ICML, MTPD)：Transformer → R-50 RetinaNet 36.5→42.0 AP。
- Peng et al., 2025 (ICLR Spotlight, D-FINE)：内置 GO-LSD 双向定位自蒸馏。

**张力分析**：分类的"大教师有害"在检测领域被 PKD/MTPD **部分推翻** —— Swin/Transformer 大教师可稳定提升小 CNN 学生，前提是有 Pearson 归一化 / 渐进式 / 投影 MLP 这类**桥**。

#### 七.2.3 决策

- **主路径**：M 教师（YOLO26-m / DEIM-D-FINE-M）。
- **矩阵补充**：cell A7 走 **L 教师**（X/XL 由 §七.2.4 排除），独立触发任一（A5 完成不是前置条件，避免与 §七.5.1 / §七.7 P2 "每 cell 独立触发" 冲突）：
  - (a) 稀有安全类（forwardRed / forwardGreen / barrier 双态 / 行人灯）AP 未达 R3 部署门槛；
  - (b) 4090 D capacity 允许且训练时间预算足够。
- **不纳入**：L/XL 直接对接 S 学生且无任何桥接（Cho 2019 已警示的高风险设计）。
- **缓解措施**（A7 触发时强制）：TAKD 中介 / ESKD 早停教师 / 投影 MLP，**三选二同时启用**。

#### 七.2.4 4090 D 容量分析（X/XL 排除依据）

**硬件**：4090 D（中国版）24 GB GDDR6X；CUDA 核 14592（标 4090 16384，少 ~10%）；FP32 ~73 TFLOPS。算力低标 4090 ~12%，VRAM 完全相同。

**KD 训练时 VRAM 估算**（imgsz=1280，FP32 mixed-prec，AdamW，单卡）：

| 组合（教师 no_grad + 学生反向 + KD 损失 + projection） | bs | 估算 VRAM | headroom | 评估 |
|---|---|---|---|---|
| YOLO-m → YOLO-s（A2a/A3/A4/A5） | 8 | ~10–14 GB | ~10–14 GB | ✓ 充裕 |
| **YOLO-l → YOLO-s（A7-L）** | 8 | **~12–17 GB** | **~7–12 GB** | **✓ 可行** |
| YOLO-x → YOLO-s | 8 | ~17–23 GB | ~1–7 GB | ⚠ 不足 2 GB |
| YOLO-x → YOLO-s | 4 | ~12–16 GB | ~8–12 GB | ⚠ wall-clock 翻倍 |
| DEIM-M → DEIM-S（A2b/A3/A4/A5） | 4 | ~9–13 GB | ~11–15 GB | ✓ |
| **DEIM-L → DEIM-S（A7-L）** | 4 | **~12–16 GB** | **~8–12 GB** | **✓ 可行** |
| DEIM-X → DEIM-S | 4 | ~14–20 GB | ~4–10 GB | ⚠ deformable attention OOM |
| DEIM-X → DEIM-S | 2 | ~11–15 GB | ~9–13 GB | ⚠ wall-clock 4× |

**教师 fine-tune（前置成本，全反向）**：YOLO26-l bs=8 ~16-20 GB ✓；YOLO26-x bs=4 ~18-22 GB ⚠ headroom ≤6 GB；DEIM-D-FINE-L bs=4 ~12-16 GB ✓；DEIM-D-FINE-X bs=2 ~16-22 GB ⚠ wall-clock 极慢。

**结论**：
1. **L 可行**（YOLO26-l / DEIM-D-FINE-L），主流 bs 留 ≥7 GB headroom，fine-tune 也 ✓。
2. **X/XL 不纳入** —— headroom < 2 GB（YOLO-x）或 OOM（DEIM-X）；降 bs 解 OOM 但 wall-clock 翻倍至 4×，违反 §七.6#4 (< 2× scratch)。
3. **A7 cell scope = L tier only**（YOLO26-l 或 DEIM-D-FINE-L）。
4. **OOM 兜底**：首 epoch 实测峰值 > 22 GB → 自动降 bs（YOLO 8→6，DEIM 4→3）；仍 > 22 GB → A7 降级为 M 教师 cell。

### 七.3 Q2：单教师 vs 多教师?

**证据要点**：
- Cao et al., 2023 (MTPD)：顺序渐进式多教师显著优于平均/同步；约束"所有教师须检测同类"。
- Liu, Yueying et al., 2020 (AMTML-KD, Neurocomputing 415)：分类证据，**不作检测器多教师 KD 的支撑**。
- Furlanello et al., 2018 (Born-Again Networks)：同架构师生连续 KD 可超越教师；最低代价"多教师"。
- 项目预算 [I]：4 教师同步前向 = 4× forward + cache，4090 D 24 GB 紧张。

#### 七.3.2 决策

- **baseline**：单教师 KD（A2a / A2b / A3 / A4）。
- **高级 cell**：渐进式 2 教师序列（cell A5），按 MTPD 范式。教师 1 = 同家族 M；教师 2 = 互补家族 M；触发：A4 通过全部 §七.6 验收门。
- **不纳入**：同步 4 教师（成本 × 4，跨 venv 开销，Cao 2023 同类约束）。
- **约束**：多教师须类集合完全一致；R2 nc 范围 10-14 锁定后，所有候选教师须先在同一类清单上 fine-tune。

### 七.4 Q3：跨架构 KD — DEIM ↔ YOLO?

**证据要点**：
- Cao et al., 2022 (PKD)：Pearson + 通道归一化 → 架构无关，明确支持异构 backbone。
- Wang et al., 2024 (CrossKD)：学生特征送入教师 head；GFL R-50 +3.5 pp；异构 backbone 可行（head 兼容是前提）。
- Yang et al., 2022 (MGD / FGD)：feature-level 架构无关；FGD 依赖共同空间网格。
- Chang et al., 2023 (DETRDistill) + Wang et al., 2024 (KD-DETR)：DETR 专用，仅 DEIM↔DEIM。

#### 七.4.2 关键限制

DEIM 是 DETR-style set-prediction（无 NMS / anchor / 含 decoder query）；YOLO 是 dense head + 网格 + C++ NMS。直接 DETR query → YOLO grid 在结构上**无自然对齐**：(a) Hungarian 匹配 vs grid 标签分配冲突；(b) 100-300 query vs 几千 anchor 密度不匹配；(c) C++ pipeline letterbox xyxy 解码与 DETR cxcywh+sigmoid 不兼容。

→ 跨架构 KD 走**架构无关 feature-level 通道（PKD / MGD / FGD）**；head/decoder/query 直接对齐**原则上不纳入**。

#### 七.4.3 决策（cell A6 实施规格）

- **方向**：teacher = 互补家族 M（YOLO 学生 → DEIM-M；DEIM 学生 → YOLO26-m），student = R2 选型胜者 (S)。
- **特征层级**：FPN 三层 stride 8/16/32（imgsz=1280 → 160/80/40）。DEIM 侧 encoder 多尺度 reshape 后 spatial map。
- **Spatial resample**：±1 偏移时 bilinear 对齐到学生维度。
- **通道投影**：学生侧 1×1 conv MLP（学生 channels → 教师 channels）；KD 损失在投影后空间计算。**导出 TRT 时移除该 projection**（仅训练辅助）。
- **损失**：首选 PKD；备选 MGD（慢 20-30%）；FGD foreground mask 跨架构未充分验证，不引入。
- **掩码**：A6 不引入 fg/bg 掩码；纯 spatial+channel feature mimic。

仅 DEIM↔DEIM cells 才使用 DETRDistill / KD-DETR。

### 七.5 综合流水线设计

#### 七.5.1 实验矩阵 — pre-committed cells

| Cell | 学生 | 教师 | 方法栈 | 触发条件 | 优先级 | 适用学生家族 |
|---|---|---|---|---|---|---|
| **A0** | DEIM-D-FINE-S | — | GO-LSD off, no external KD | always — DEIM 路径 baseline 净分离对照 | P0 | DEIM only |
| **A1** | 选型胜者 | — | scratch (DEIM 路径 GO-LSD on) | always — control + KD-acceptance gate 的 wall-clock 锚点 | P0 | 全部 |
| **A2a** | YOLO26-s | YOLO26-m | cls-logit KL only（YOLO26 已移除 DFL） | always；R2 in-round 即可启动 | P0 | YOLO 家族 |
| **A2b** | DEIM-D-FINE-S | DEIM-D-FINE-M | LD on FDR + cls-logit KL | always；R2 in-round 即可启动 | P0 | DEIM 家族 |
| **A3** | 选型胜者 (S) | 同家族 M | PKD feature-level | always | P0 | 全部 |
| **A4** | 选型胜者 (S) | 同家族 M | A2a/A2b + A3 组合 | max(A2a/A2b, A3) **mAP@0.5:0.95 lower-CI > A1 point** AND 无安全类 AP delta < −0.5pp | P1 | 全部 |
| **A5** | 选型胜者 (S) | 同家族 M → 互补家族 M | MTPD 渐进 2-教师 | A4 通过 §七.6 全部 5 项验收门 | P2 | 全部 |
| **A6** | 选型胜者 (S) | 互补家族 M | PKD 跨架构（投影 MLP + spatial resample，§七.4.3 规格） | A4 通过 §七.6 + 团队余力 | P2 | 全部 |
| **A7** | 选型胜者 (S) | 同家族 **L**（YOLO26-l 或 DEIM-D-FINE-L；TA 桥 = M） | TAKD via M + ESKD checkpoint + 投影 MLP（三选二） | (a) 稀有安全类 AP 未达 R3 门槛；OR (b) 4090 D capacity 允许（§七.2.4） | P2 | 全部 |

**A4 触发的精确语义**：
- 主指标 mAP@0.5:0.95；CI 由 §七.6#1 pre-committed 估计法计算（默认方法 b：1000× bootstrap）。
- 比较：候选 cell lower-CI > A1 point estimate（95% 置信"超过 A1 中位"）。
- 安全类否决：每个 `full_val_support ≥ 30` 的安全类 AP delta ≥ −0.5 pp。
- 平局：A2a / A3 / A2b 主指标差距 ≤ 0.1 pp 且都通过安全类时，按 wall-clock 选更便宜者。
- 计算者：`scripts/_kd_decide_cell.py` (NEW)；与 `_r2_decide_precision.py` 共用 CI 工具；接受 `--ci-method {seed5, bootstrap1000}`。

**Canonical drawdown order**（2 周 R2 in-round 预算不足时按以下顺序削减）：
1. 先丢 **A7**（capacity-conditional）；
2. 再丢 **A6**（cross-arch，证据 detection 转移弱）；
3. 最后丢 **A5**（detection 文献支持但项目数据集无验证）。

P0 cells（A0 仅 DEIM 路径 / A1 / A2a-按家族 / A2b-按家族 / A3）不可丢——丢则 round 不闭环。**A4（P1）仅在其触发条件成立时不可丢**（即 A2/A3 满足 §七.5.1 lower-CI > A1 point + 安全类阈值）；触发条件不成立时 A4 不必运行，round 仍可闭环。

#### 七.5.2 损失函数族

| 族 | 代表方法 | 适用 |
|---|---|---|
| 响应/logit | LD (Zheng 2022) | 仅 DFL-bearing 检测器；YOLO26 退化为 cls-logit KL |
| 特征/hint | PKD / MGD / FGD | 通用；PKD 异构友好；MGD dense generation；FGD 前景失衡 |
| DETR-专用 | DETRDistill / KD-DETR / GO-LSD（内置） | 仅 DEIM↔DEIM |
| Head/cross-task | CrossKD | 仅 head 兼容时（同家族） |
| 自蒸馏 | Born-Again Networks / GO-LSD（内置） | 默认对照 |

#### 七.5.3 提示层选择

| 学生 | hint 层 |
|---|---|
| YOLO26-s | neck P3-P5 + head pre-logit |
| DEIM-D-FINE-S（同家族 KD） | encoder memory + decoder intermediate（DETRDistill / KD-DETR 路径） |
| DEIM-D-FINE-S（A6 cross-arch） | stride 8/16/32 encoder feature pyramid，§七.4.3 规格 resample |

#### 七.5.4 训练时间表 + 数据增强一致性

```
Stage 0 (warm start, 10–20 epochs):  COCO 预训练 → R2 数据 fine-tune（仅硬目标）。
Stage 1 (KD-on, 80–150 epochs):       KD 损失逐步升温到完整权重；与 GT 共训。
Stage 2 (KD-off final, 5–10% 末尾):    可选；仅硬目标 + 强增强收尾。
```

**数据增强一致性**（Wang et al., 2022, *Inconsistent KD with Data Aug*）：教师在与学生**相同的增强后图像**上重新前向；不能用未增强图像作教师 logit。

**缓存模式**：默认即时计算（no cache）；训练吞吐 < 60% A1 baseline 时按 epoch 切换至 SSD 缓存（≤ 200 GB scratch quota），不切 RAM。缓存键须包含增强采样状态（参见 [`../../components/knowledge_distillation/runners/__init__.py`](../../components/knowledge_distillation/runners/__init__.py) 实现契约）。

#### 七.5.5 与 D-FINE 内置 GO-LSD 的交互

D-FINE GO-LSD 是双向定位自蒸馏。DEIM-D-FINE-S 学生 baseline 训练时已接收内置自蒸馏的定位信号 → DEIM 学生加外部 LD 功能重叠，**预期边际收益小**。

**A2b 的角色**：A2b（LD on FDR + cls-logit KL）是该重叠的**实证 ablation**，不是"避开定位通道"原则的反例。三 cell 组合 A0 (GO-LSD off, no KD) + A1 (scratch, GO-LSD on) + A2b (GO-LSD on, +外部 LD + cls-KL) 给出净分离证据：
- A1 − A0 = GO-LSD 内置自蒸馏的定位贡献；
- A2b − A1 = 在 GO-LSD 之上**外部 LD + cls-KL 是否还有边际收益**。

若 A2b − A1 ≤ §七.6#1 噪声阈值，证实 §七.5.5 的预期重叠假设；后续 DEIM 学生的 P1+ 优先 classification / encoder feature 通道（A3 / A4）。这一净分离逻辑独立于 A2b 是否通过 §七.6 验收门。

### 七.6 预承诺验收门（KD-acceptance gate）

5 项 pre-commit gate；本计划经 PM 批准后即作为 round-binding 规则。未全部通过者**不进入 KD ship-decision**（保留为研究记录）。

#### 七.6#1 总 mAP 不退化

- **A1 baseline 角色**：A1 训练完成后，CI 由两种估计法**视决策状态选择**（2026-05-09 conflictor iter-1 amendment）：
  - 方法 a：5 个种子重复 + mAP 95% CI 由 mean ± 1.96 × SD/√5；优 = 标准统计（捕获训练方差）；劣 = 5× wall-clock。
  - 方法 b：1 次训练 + 1000× bootstrap on per-image preds（与 R2 精度奇偶共用工具）；优 = 1× wall-clock；劣 = 低估种子方差，幸运 seed 可能假阳性通过。
  - **CI 方法选择规则**：
    - **生产 deploy 候选**（cell 拟入 ship-decision）：**强制 `seed5`**——bootstrap1000 在 single-seed 上低估训练方差，对 deploy 决策门不够稳；
    - **defer / drop 分类**或 field-test 候选（不入 ship-decision，仅作研究记录）：可用 `bootstrap1000` 节省 wall-clock。
  - CI 记录为 `A1_CI_low / A1_point / A1_CI_high` + `ci_method` 字段。`scripts/_kd_decide_cell.py` 在 `ship-decision` 路径上断言 `ci_method == "seed5"`，否则 exit 2。
- **KD cell 验收**（A2a/A2b/A3/A4/A5/A6/A7）：要求 `KD_cell_lower_CI_bound > A1_CI_low` AND `KD_cell_lower_CI_bound > A1_point − 0.5pp`（即 95% 置信不退化超过 0.5pp）。
- A1 自身不通过此门 — 它是参照。
- A0 按 A1 同样规则建立独立 CI；仅用于 §七.5.5 净分离分析，不直接进入 ship-decision。

#### 七.6#2 安全类逐类 AP

red / yellow / green / 所有箭头类 / barrier-up / barrier-down / 行人信号灯 — 每个 `full_val_support ≥ 30` 的类 AP delta ≥ −0.5 pp。`full_val_support < 30` 的类不阻塞。

#### 七.6#3 无新型 FP 增长

R1 demo8 / 11 / 13 类背景帧上 FP 数不上升（用 R2 hard-negative mining 同源帧检验）。**与本计划 §四 硬负挖掘共用 manifest**。

#### 七.6#4 训练成本预算

- A1 完成后记录其实测 wall-clock `T_scratch_A1`；A2+ cell 的成本验收**必须等 A1 完成后才可评估**。
- 门：单 cell wall-clock < `T_scratch_A1 × 2.0`。
- 超出者：cell 训练完成（用作未来参考）但不进入 ship-decision；记 `cost_gate_failed=true`。

#### 七.6#5 导出 TRT 引擎 + sidecar 验收

- **dtype 选取**：KD 学生导出 TRT 引擎并通过 R2 精度奇偶 eval-parity gate（0.01 pp）；导出 dtype 由 R2 精度奇偶针对该学生选定的 `ship_precision` 决定：
  - FP16：仅 FP16 引擎验收；
  - FP32：仅 FP32 引擎验收；
  - R2 ship_precision 仍未定（KD 与 R2 真正并行）：双 dtype 导出，ship-decision 暂挂回填。
- KD 收益**在 PyTorch checkpoint 上 ≠ TRT 引擎上**，建议用导出引擎复测。
- Sidecar：与 `scripts/export_yolo.sh` / `scripts/export_deim.sh` 同源。**KD 学生 sidecar 须额外三字段** `kd_cell_id` / `kd_method` / `kd_teacher_artifact_sha256`。两个 export 脚本当前**未输出**这些字段；首个 KD cell 落地前必须先完成 export-script 扩展（添加 env/CLI 入口接受这三个值，写入 sidecar；若任一 KD 字段缺失则 sidecar 视为不完整，引擎拒绝纳入 ship-decision）。属于已识别 carry-forward，不在本计划当前实现范围。

未达任一条款者：cell 不进入 ship-decision，失败原因记入 `runs/_kd_decisions.json`（schema `scripts/_kd_decision_schema.json` NEW，与 `_r2_decision_schema.json` 字段兼容率 ≥ 80%）。预期 ≥ 30% cells 拒绝落地是正常的（基于 Cho 2019 + Wang 2022 经验）；不据此调整门槛。

### 七.7 与现有 R2 / R3 计划的衔接

KD 优先 R2 in-round 内嵌；时间允许时启动尽可能多 cells；时间不足时按 §七.5.1 优先级 (P0 > P1 > P2) 由下往上削减。

| 阶段 | KD cells | 触发 |
|---|---|---|
| R2 in-round（默认 P0） | A0（仅 DEIM）+ A1 + A2a/A2b（按家族二选一）+ **A3** | 选型胜者一旦确定即可启动；不阻塞主线 R2 训练**报告初稿**。A3 为 P0/Always（§七.5.1 + §七.12 一致）。**KD 子节 / round closure 仍要求 P0 完成**（§七.5.1 + §七.12）；未完成则转入下行 KD 降级 round，并在 R2 报告 coverage-gaps 中列明。 |
| R2 in-round (P1) | + A4 | A1 完成 + max(A2a/A2b, A3) lower-CI > A1 point + 安全类阈值（§七.5.1 A4 触发） |
| R2 in-round (P2) — **每 cell 独立触发** | + A5 | A4 通过 §七.6 **全部 5 项**验收门（§七.5.1 A5 触发） |
|  | + A6 | A4 通过 §七.6 全部 5 项 + 团队余力（§七.5.1 A6 触发） |
|  | + A7 (L tier only) | (a) 稀有安全类（forwardRed/forwardGreen/barrier 双态/行人灯）AP 未达 R3 门槛；**OR** (b) 4090 D capacity 允许（§七.5.1 A7 触发；§七.2.4 容量分析锁定 L tier，X/XL 排除） |
| KD 降级 round | 残留未完成 cells | R2 phase report 关帧时若 P0 未跑完，残留推到独立 round；A4/A5/A6/A7 触发条件不成立时不入降级 round（其触发本身即是闭环判据） |
| R3 / pre-deploy | 无 | A7 已在 R2 内闭环；KD 维度无 R3 carry-forward |

**与 R2 精度奇偶（`~/.claude/plans/elegant-sauteeing-quail.md`）共用**：frozen R2 val manifest + bootstrap CI 工具 + eval-parity gate（0.01 pp）+ sidecar 契约 + `_kd_decisions.json` 与 `_r2_precision_decisions.json` ≥ 80% 字段重用。

### 七.8 风险与缓解（Top 6）

| 风险 | 缓解 |
|---|---|
| Capacity-gap：L 教师劣化 S 学生 | TAKD + ESKD + 投影 MLP **三选二同时启用**（A7 触发时） |
| 跨架构对齐失败 | 限 PKD/MGD feature-level；强制投影 MLP + spatial resample（A6） |
| GO-LSD 与外部 KD 重复 | A2b 故意保留外部 LD on FDR + cls-logit KL 以对此重叠做**实证 ablation**（A0 + A1 + A2b 三者的 mAP delta 差给出净分离证据，§七.5.5）；若 A2b − A1 ≤ §七.6#1 噪声阈值，**DEIM 学生 P1+ cells（A3 / A4）优先 cls/encoder feature 通道**避开定位重叠 |
| 强增强破坏教师监督 | 教师在增强后图像上**重新前向**（Wang 2022） |
| 训练成本爆炸 | 缓存教师特征/logit；不启用同步 4 教师；§七.6#4 wall-clock < 2× scratch |
| 4090 D OOM | §七.2.4 已排除 X/XL；首 epoch 峰值 > 22 GB 自动降 bs；仍 OOM → A7 降级为 M 教师 |

辅助：FP16 量化破坏 KD 收益（验收用 TRT 引擎而非 PyTorch ckpt）；Cross-arch 学生权重 license 传递性（A6，commercial-deploy 阶段重新激活，field-test 不阻塞）；教师 fine-tune 前置成本（~6-12h × N_teachers）。

### 七.9 资源与时间预算

[planning estimate]：A1 完成后回填 `T_scratch_A1` 实测值并重算。

| 项目 | 预估 |
|---|---|
| 单 cell 训练时间 | 8-24h（KD feature-level 上限 +40%） |
| 完整 P0 矩阵（A0 + A1 + A2a/A2b + A3） | 24-72h |
| 完整 P0+P1+P2（+ A4 + A5 + A6 + A7-L） | 88-192h |
| 教师 fine-tune 前置 | ~6-12h × N_teachers + 2-8 GB checkpoint × N_teachers |
| 教师特征缓存空间 | 0-200 GB（默认 0；< 60% A1 baseline 切 SSD） |
| 隐性运营成本 | ~3-5 day per round（demo 复评 + 法务 + report） |

**预算红线**：完整 P0+P1+P2 矩阵在 R2 in-round 时间分配下 ≤ **2 周墙钟时间**（含人监督 + 隐性运营）；超出按 §七.5.1 canonical drawdown 削减。R2 phase report 关帧时若 P0 未完成，残留推到"KD 降级 round"。

### 七.10 Pre-committed 决策（无 open issue）

| 项目 | 决策 |
|---|---|
| Cell A0（GO-LSD 关 + 无外部 KD）DEIM 路径 baseline 净分离 | ✅ §七.5.1 矩阵 P0 row |
| 教师特征缓存模式 | ✅ 默认即时计算；< 60% A1 baseline 切 SSD（≤ 200 GB），不切 RAM |
| A4 触发选择规则 | ✅ 候选 lower-CI > A1 point + 安全类 AP delta ≥ −0.5 pp |
| A5 触发 | ✅ A4 通过 §七.6 全部 5 项验收门（§七.5.1 / §七.7） |
| A6 触发 | ✅ A4 通过 §七.6 全部 5 项 + 团队余力（§七.5.1 / §七.7） |
| A7 触发 | ✅ 独立于 A4/A5：(a) 稀有安全类 AP 未达 R3 门槛 **OR** (b) 4090 D capacity 允许；L tier only，X/XL 排除（§七.2.3 / §七.5.1 / §七.7 / §七.12） |
| KD round 归属（PM #1） | ✅ R2 in-round 内嵌；时间不足时降级 round 不延阻 R2 关帧 |
| GPU 独占性（PM #2） | ✅ 4090 D 24 GB 用户独占，无抢占 |
| 教师 pool 含 L tier（PM #3） | ✅ YOLO26-l / DEIM-D-FINE-L 纳入；X/XL 由 §七.2.4 排除 |
| 法务 gate 时机（PM #4） | ✅ field-test 阶段不阻塞；commercial-deploy 阶段重新激活 |
| Conflictor-loop 终止（PM #5） | ✅ until AGREED，无 iter 硬上限（详见 §十四） |

### 七.11 使用方式（runner 模块映射）

每个 cell 一个 Python 入口模块，位于 [`../../components/knowledge_distillation/runners/`](../../components/knowledge_distillation/runners/__init__.py)。文件名描述其方法栈，cell ID 在 docstring 中作为规范引用。所有 runner 当前为 stub（`NotImplementedError`），将在所属 cell 调度时落地。

| Cell | runner 模块 | 调用形式（落地后） |
|---|---|---|
| A0 | `deim_baseline_golsd_off.py` | `python -m components.knowledge_distillation.runners.deim_baseline_golsd_off --config <cfg>` |
| A1 | `scratch_baseline.py` | `python -m components.knowledge_distillation.runners.scratch_baseline --config <cfg>` |
| A2a | `yolo_logit_kd.py` | `python -m components.knowledge_distillation.runners.yolo_logit_kd --teacher-ckpt <pt> --config <cfg>` |
| A2b | `deim_logit_localization_kd.py` | `python -m components.knowledge_distillation.runners.deim_logit_localization_kd --teacher-ckpt <pth> --config <cfg>` |
| A3 | `pearson_feature_kd.py` | `python -m components.knowledge_distillation.runners.pearson_feature_kd --teacher-ckpt <ckpt> --config <cfg>` |
| A4 | `logit_plus_feature_kd.py` | `python -m components.knowledge_distillation.runners.logit_plus_feature_kd --teacher-ckpt <ckpt> --config <cfg>` |
| A5 | `progressive_multi_teacher.py` | `python -m components.knowledge_distillation.runners.progressive_multi_teacher --teacher-ckpt <t1> --teacher-ckpt <t2> --config <cfg>`（重复 `--teacher-ckpt`，按出现顺序对应 phase 1 / phase 2；这是单数契约的 nargs 扩展，不是改名）|
| A6 | `cross_arch_feature_kd.py` | `python -m components.knowledge_distillation.runners.cross_arch_feature_kd --teacher-ckpt <other-family> --config <cfg>` |
| A7 | `takd_large_teacher.py` | `python -m components.knowledge_distillation.runners.takd_large_teacher --teacher-ckpt <L> --assistant-ckpt <M> --config <cfg>` |

**统一 CLI 契约**（在 `runners/__init__.py` 中文字记录，第一个 cell 落地时锁定）：
```
--config <yaml>             student + teacher + KD 超参
--teacher-ckpt <path>       已 fine-tune 的教师 artifact (R2 nc range)
--student-init {scratch, coco, r2_baseline}
--output-dir runs/<cell_id>/
--seed <int>                训练开始前写入 SEED.txt
--ci-method {bootstrap1000, seed5}    默认 bootstrap1000（§七.6#1）
--resume <ckpt>             resume 时不覆写 SEED.txt（须沿用原 SEED）
```

**辅助子包**：
- `losses/` — KD 损失模块（cls_logit_kl / ld_fdr / pkd / mgd / projection_mlp / kd_weight_ramp 标量）；仅 per-batch KD 信号。
- `schedules/` — 多阶段 / 多教师编排（kd_phase_runner / mtpd_progressive / takd_assistant / eskd_loader / golsd_toggle）；非损失函数。
- `gates/` — §七.6 验收门评估器（gate1-5）。

**Sidecar 与决策**：导出 TRT 引擎沿用 `scripts/export_yolo.sh` / `scripts/export_deim.sh`。**两个 export 脚本当前不输出 KD sidecar 字段**（`kd_cell_id` / `kd_method` / `kd_teacher_artifact_sha256`），首个 KD cell 落地前必须先扩展 export 脚本写入这三字段（详见 §七.6#5）；这是已识别 carry-forward，不在本 scaffold 范围。Ship-decision 由 `scripts/_kd_decide_cell.py`（NEW，deferred）写入 `runs/_kd_decisions.json`，schema 校验由 `scripts/_kd_decision_schema.json`（NEW，deferred）。

### 七.12 执行清单（cell-by-cell checkbox）

按 §七.5.1 优先级与 §七.7 阶段顺序勾选。所有 P0 必须完成 round 才闭环；P1/P2 按预算与触发条件决定。

#### P0（强制；R2 in-round 默认 — 按选型胜者家族勾选）

**Always:**
- [ ] **A1 / `scratch_baseline.py`** —— scratch 训练 + 记录 `T_scratch_A1` + 计算 A1 CI（`bootstrap1000` 默认）。**全部 KD cell 验收依赖此**。
- [ ] **A3 / `pearson_feature_kd.py`** —— 同家族 M 教师，PKD feature-level。

**If R2 winner = YOLO 家族（A0/A2b N/A）:**
- [ ] **A2a / `yolo_logit_kd.py`** —— cls-logit KL only。

**If R2 winner = DEIM 家族（A2a N/A）:**
- [ ] **A0 / `deim_baseline_golsd_off.py`** —— GO-LSD 关 + 无外部 KD baseline，§七.5.5 净分离分析输入。
- [ ] **A2b / `deim_logit_localization_kd.py`** —— LD on FDR + cls-logit KL（**A2b 是 GO-LSD-vs-外部-LD 重叠的实证 ablation**；§七.5.5 预测边际收益小，A2b + A0 + A1 三者结合给出净分离证据）。

#### P1（A1 + A2/A3 完成后；时间余力）

- [ ] **A4 / `logit_plus_feature_kd.py`** —— 触发：max(A2a/A2b, A3) lower-CI > A1 point + 安全类 AP delta ≥ −0.5 pp（§七.5.1）。

#### P2（按 §七.5.1 / §七.7 各 cell 独立触发；canonical drawdown 顺序）

- [ ] **A5 / `progressive_multi_teacher.py`** —— MTPD 渐进 2 教师。**触发：A4 通过 §七.6 全部 5 项验收门**。
- [ ] **A6 / `cross_arch_feature_kd.py`** —— 跨架构 PKD（投影 MLP + spatial resample）。**触发：A4 通过 §七.6 全部 5 项 + 团队余力**。
- [ ] **A7 / `takd_large_teacher.py`** —— L 教师 + TAKD/ESKD/projection 三选二桥接。**触发**（独立于 A4/A5）：(a) 稀有安全类 AP 未达 R3 门槛 **OR** (b) 4090 D capacity 允许（§七.5.1 / §七.2.3 / §七.2.4）。

#### 全 round-level 验收

- [ ] §七.6#1 总 mAP 不退化：所有候选 cell `lower-CI > A1_CI_low` AND `lower-CI > A1_point − 0.5pp`。
- [ ] §七.6#2 安全类逐类 AP：每个 `full_val_support ≥ 30` 安全类 AP delta ≥ −0.5 pp。
- [ ] §七.6#3 无新型 FP 增长：R1 demo8/11/13 背景帧 FP 数不上升。
- [ ] §七.6#4 训练成本：每 cell wall-clock < `T_scratch_A1 × 2.0`。
- [ ] §七.6#5 TRT 引擎 + sidecar 验收：导出引擎通过 0.01 pp eval-parity，sidecar 含 `kd_cell_id` / `kd_method` / `kd_teacher_artifact_sha256`。
- [ ] `runs/_kd_decisions.json` 写入完整记录，`scripts/_kd_decide_cell.py` 不报 schema 错误。
- [ ] `phase_R2.md` KD 子节落地（含 cell 通过/拒绝表 + 拒绝原因）。

### 七.13 KD 生命周期 rollup（与本计划 §一 五步约定对齐）

- [ ] **a1. 脚手架（已落地部分）** — [`../../components/knowledge_distillation/`](../../components/knowledge_distillation/__init__.py) **LANDED v1.3，B2 + C3 已 AGREED**（详见 scaffold README，cells A0-A7 / runners / losses / schedules / gates 子包均落地）
- [ ] **a2. 脚手架（待补部分）** —— **首个 KD cell 落地前必须落地**：
    - `scripts/_kd_decision_schema.json`（NEW，schema 与 `_r2_decision_schema.json` ≥ 80% 字段重用）；
    - `scripts/_kd_decide_cell.py`（NEW，§七.5.1 决策规则执行器）；
    - `scripts/export_yolo.sh` + `scripts/export_deim.sh` 新增 sidecar 字段 `kd_cell_id` / `kd_method` / `kd_teacher_artifact_sha256`（§七.6#5 carry-forward）。**a 阶段未完整落地前，§七 不得进 b 阶段**。
- [ ] **b. 实现** — 占位符（runner stubs / `NotImplementedError`）替换为真实代码，按 §七.5.1 cell 矩阵分批推进；**首个 cell 落地前必须扩展 export 脚本 sidecar 字段**（§七.6#5）
- [ ] **c. 消融** — A0-A7 cell A/B（按 §七.5.1 决策规则 + §七.6 验收门）；预承诺验收门 5 项
- [ ] **d. 决策** — 每个 cell 写入 `runs/_kd_decisions.json`（含通过/失败状态 + 拒绝原因）
- [ ] **e. 报告** — `phase_R2.md` KD 子节（含 cell 通过/拒绝表 + 拒绝原因）

---

## 八、多相机融合（需自动驾驶团队对齐）

### 8.1 机制
自动驾驶车通常已有 front-narrow + front-wide。两路独立检测后晚期融合（同一物体的多视角投影 + score 联合）。

### 8.2 契合痛点
遮挡（一路被挡另一路可能可见）+ 远距（窄视场看小灯更清楚）+ 非正面朝向。

### 8.3 文献
- Behrendt et al., "A Deep Learning Approach to Traffic Lights: Detection, Tracking, and Classification," ICRA 2017（多视角晚期融合先例）。

### 8.4 Orin 影响 / 工作量
- ~2× 检测计算（除非主路单独占线程，副路低频共享）；
- 取决于现有相机配置 — **需自动驾驶团队确认**（**[需对齐]**）。

### 8.5 风险
- 相机外参标定；
- 时间同步；
- 副路相机延迟可能不一致。

### 8.6 衔接
- 落入 [`development_plan.md`](development_plan.md) §五 风险（自动驾驶团队对齐项）；
- 可能已在他们的路线图上，确认即可。

### 8.7 决策规则（pre-committed）
- **指标**：遮挡场景 recall；远距场景 recall；总延迟。
- **deploy**：遮挡场景 recall +10 pp 或远距 recall +5 pp；
- **defer**：单维度提升不显著，留作硬件升级后重评；
- **drop**：相机配置不允许多相机融合。

### 8.8 清单
- [ ] **a. 脚手架**：**[阻塞]** 与自动驾驶团队对齐相机配置 + ROS2 topic 契约
- [ ] **b. 实现**：晚期融合逻辑（投影 + WBF）
- [ ] **c. 消融**：A/B 对照（单相机 vs 多相机）
- [ ] **d. 决策**：按 §8.7 规则，写入 `runs/_multi_camera_decision.json`
- [ ] **e. 报告**：phase R2/R3 子节

---

## 九、自适应推理 / ROI 裁剪（5/15 之前可落地）

### 9.1 机制
当前管线 30 fps 全帧推理。改为：
- 高速无路口 → 5 fps 或完全跳过 TL 推理；
- 接近路口 → 30 fps 全帧 + SAHI 启用；
- 路口附近且可见灯 → 上半帧 ROI 推理（典型 dashcam 灯在上 1/3）。

### 9.2 契合痛点
间接 — 释放预算给其他方法叠加（§六 SAHI / §十 INT8 QAT）；同时降低非路口 FP（部分场景下连推理都不跑就不会出 demo8 类误检）。

### 9.3 文献
- 通用工程实践；无专门文献支撑必要。

### 9.4 Orin 影响 / 工作量
- **负向**（释放预算）；
- ~3 天管线改造。

### 9.5 风险
- ROI 裁剪可能漏掉非典型位置的灯（如低位 pedestrian 灯）—— 决策规则需测试。

### 9.6 衔接
- 落入 [`development_plan.md`](development_plan.md) 推理管线优化；
- 与 §五 地图先验门控**协同**（路口判定共用 GPS 信号）；
- 与 §六 SAHI / §十 INT8 QAT 互补。

### 9.7 决策规则（pre-committed）
- **指标**：端到端延迟；遗漏 recall。
- **deploy**：端到端 5 ms+ 解放且 recall 不退化（−0.5 pp 容忍）；
- **defer**：解放 ≤ 5 ms 或 recall 退化 0.5-2 pp，R3 重评；
- **drop**：recall 退化 > 2 pp。

### 9.8 清单
- [ ] **a. 脚手架**：`inference/cpp/` 推理管线 frequency-control + ROI 接口（NEW，deferred）
- [ ] **b. 实现**：GPS-based frequency switch + ROI 裁剪
- [ ] **c. 消融**：A/B 对照（全帧 30 fps vs 自适应）
- [ ] **d. 决策**：按 §9.7，写入 `runs/_adaptive_inference_decision.json`
- [ ] **e. 报告**：phase R2/R3 子节

---

## 十、INT8 QAT（条件性，仅当 SAHI / TSM 需要释放预算）

### 10.1 机制
当前 FP16；QAT 校准后 INT8 在 Orin 上 ~1.5-1.8× 推理加速。释放预算给 SAHI / TSM 叠加。

### 10.2 契合痛点
间接 — 释放预算允许叠加其他方法。

### 10.3 文献
- TensorRT QAT Toolkit (PyTorch-Quantization)。

### 10.4 Orin 影响 / 工作量
- **负向**（更快），但精度可能掉 0.5-1.5 pp；
- ~1 周（校准集准备 + 校准 + 验证精度 delta）。

### 10.5 风险
- DEIM 的 deformable attention 在 INT8 下可能数值不稳；先在 YOLO26 上验证。

### 10.6 衔接
- 落入 [`development_plan.md`](development_plan.md) 推理优化；
- **条件性 + 预探测**：原则上 §十 仅在 §六 SAHI 或 TSM Phase 1-C 需要释放预算时启动。但避免与 §六 形成循环死锁（SAHI 候选 defer 等 INT8；INT8 仅在 SAHI 触发后启动），加入 **pre-SAHI latency probe**：若 §六 inference-only SAHI 实测 FP16 端到端 latency `> 50 ms` 但 `< 80 ms`（即 deploy 阈值之外、drop 阈值之内），**立即在同 round 启动 §十 INT8 校准与 export 改造**，不等待 §六 决策；§六 在 INT8 引擎上重测决定 deploy / defer / drop。
- 与 R2 精度奇偶（plan `~/.claude/plans/elegant-sauteeing-quail.md`）共用 sidecar / eval-parity gate；INT8 引擎须通过同一 0.01 pp gate。

### 10.7 决策规则（pre-committed）
- **指标**：精度 delta（mAP@0.5:0.95）；推理速度提升。
- **deploy**：mAP delta ≥ −0.5 pp + 速度 ≥ 1.5×；
- **defer**：mAP delta −0.5 ~ −1.5 pp 且 SAHI 解决方案不依赖 QAT 解放预算，留作 R3 重评；
- **drop**：mAP delta < −1.5 pp（无法补回的精度损失）。

### 10.8 清单
- [ ] **a. 脚手架**：`scripts/export_yolo.sh` / `scripts/export_deim.sh` 增加 `INT8=1` 选项（NEW，deferred）；校准集 manifest
- [ ] **b. 实现**：PyTorch-Quantization 校准 pipeline；TRT INT8 引擎构建；sidecar 增加 `precision: int8`
- [ ] **c. 消融**：A/B 对照（FP16 vs INT8 QAT），mAP delta + 速度
- [ ] **d. 决策**：按 §10.7，写入 `runs/_int8_qat_decision.json`
- [ ] **e. 报告**：phase R3 子节（部署优化）

---

## 十一、规划器先验融合（cross-detection 框架扩展）

### 11.1 机制
规划模块知道当前路径意图（"下一路口左转"），告知感知模块"期望看到 redLeft / greenLeft"。把规划的 prior 输入到检测后处理（类似共现先验，但是 planner-conditioned）。

### 11.2 契合痛点
边界混淆（弱辅助）+ 持续误分类（demo10 — 若规划知道这条路有直行需求，gantry 上识别成 round-green 而非 forward-green 的概率被惩罚）。

### 11.3 文献
- Apollo Perception 中类似实现。

### 11.4 Orin 影响 / 工作量
- ~0；
- ~1 周接口设计 + 实施。

### 11.5 风险
- 规划意图错误时 prior 反向 push（fallback：低置信度的 prior 不应用）；
- 需规划团队发布"路径意图 / 期望信号"主题（**[需对齐]**）。

### 11.6 衔接
- **关键观察**：[`cross_detection_reasoning_plan.md`](cross_detection_reasoning_plan.md) 的贝叶斯框架可以**直接吸收 planner-prior**作为先验来源 —— 不需要单独写一套实现。同一套 mean-field / CRF 数学，输入换成 P(class | route_intent) 即可；
- 落入 [`cross_detection_reasoning_plan.md`](cross_detection_reasoning_plan.md) §2.2(b) 或 §6 作为先验来源扩展。

### 11.7 决策规则（pre-committed）
- **指标**：demo10-类持续误分类率；总 mAP 不退化。
- **deploy**：持续误分类率 −50% + 总 mAP 不退化；
- **defer**：误分类率 −20-50%，留作 R3 重评；
- **drop**：意图错误反向 push 导致 mAP 退化 > 0.5 pp。

### 11.8 清单
- [ ] **a. 脚手架**：**[阻塞]** 与规划团队对齐 ROS2 topic 契约（"route_intent" / "expected_signals"）
- [ ] **b. 实现**：cross-detection 框架插入 planner-prior 输入；fallback 行为
- [ ] **c. 消融**：A/B 对照（无 planner-prior vs 有）
- [ ] **d. 决策**：按 §11.7，写入 `runs/_planner_prior_decision.json`
- [ ] **e. 报告**：phase R3 子节

---

## 十二、已排除路径

### 12.1 在本计划范围外但仍可能落地（决策外移）
- **HDR 相机 / 多曝光融合**（survey §4.1）—— PM / 硬件团队决策，非 ML 计划；
- **DINOv2 / MAE 自监督预训练**（survey §3.5）—— 关键依赖 R2 raw video SOP；
- **Fleet-based 主动学习**（survey §4.3）—— 长期运营路径，部署后启动。

### 12.1.5 范围相邻但其他计划跟踪（不在本计划，但需相互引用）
- **TSM 时序优化**：跟踪于 [`temporal_optimization_plan.md`](temporal_optimization_plan.md) §1（scaffold v1.5 LANDED），与本计划 §六 SAHI 互补（时序 vs 单帧）；
- **R2 精度奇偶（FP16 vs FP32）**：跟踪于 `~/.claude/plans/elegant-sauteeing-quail.md`（locked），与本计划 §七.6#5 / §十 共用 sidecar + eval-parity gate；
- **跨检测共现推理**：跟踪于 [`cross_detection_reasoning_plan.md`](cross_detection_reasoning_plan.md)，与本计划 §十一 规划器先验为先验来源扩展关系；
- **主检测器选型**：跟踪于 [`development_plan.md`](development_plan.md)，本计划全部组件依赖其选型胜者作为 student / 注入目标。

### 12.2 研究侧已排除（survey §5.4）
端到端 VLM、事件相机 DVS、扩散模型大批量合成、强化学习 / 主动推理、Scene-graph / 全场景 GNN、在线适应 / 测试时训练。理由详见 survey §5.4。

### 12.3 推理 / 训练侧已排除（survey §5.3）
- **多任务 / 辅助头**（survey §3.6）—— 工作量数量级大；与三套候选检测器（YOLO26 / YOLOv13 / DEIM）头部架构强耦合，每家族要单独 patch + R2 标注 SOP 要 1× 增量（color × shape 双标签）。R3 候选 only；
- **TTA**（survey §3.8）—— 部署不可行（3× 推理延迟）；仅作 R2 自采伪标签生成工具；
- **检测器集成**（survey §3.9）—— 部署超预算（2× 推理 + 双倍模型内存超 50 ms 上限）；离线工具；
- **图像超分预处理**（survey §3.11）—— R2 数据上**未实测**被 SAHI dominated；本计划接受 survey 排除论据但留 R3 重评开关：若 §六 SAHI 在 deploy 决策中失败且 §十 INT8 已用尽预算，SR crop 路径作为最后单帧补丁可重新评估。R3 候选 only；
- **合成数据 / CARLA**（survey §3.12）—— 优先 R2 实采；仅当 R2 实采无法触达关键稀有条件（逆光 / 黄昏 / 暴雨）时回看；
- **ISP ROI AE**（survey §4.6）—— 仅当 HDR 不可得时的次优；本项目 HDR 决策由 §十二.1 PM 拍板，本路径作为 fallback 不积极推进。

---

## 十三、参考资料

### KD 相关（继承自 v1.3）

#### 跨架构 / 异构 KD
- Cao, Y., Zhang, Y., et al. (2022). *PKD: General Distillation Framework for Object Detectors via Pearson Correlation Coefficient*. NeurIPS 2022. [arXiv:2207.02039]
- Wang, J., Chen, Y., et al. (2024). *CrossKD: Cross-Head Knowledge Distillation for Object Detection*. CVPR 2024, pp. 16520–16530.
- Yang, Z., Li, Z., et al. (2022). *Masked Generative Distillation*. ECCV 2022.
- Yang, Z., Li, Z., et al. (2022). *Focal and Global Knowledge Distillation for Detectors*. CVPR 2022.

#### DETR-specific KD
- Chang, J., et al. (2023). *DETRDistill: A Universal Knowledge Distillation Framework for DETR-families*. ICCV 2023. [arXiv:2211.10156]
- Wang, J., et al. (2024). *KD-DETR: Knowledge Distillation for Detection Transformer with Consistent Distillation Points Sampling*. CVPR 2024.
- Peng, Z., Yang, P., et al. (2025). *D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement*. ICLR 2025 Spotlight. (含 GO-LSD)

#### 多教师 / 渐进式 / Capacity gap
- Cao, S., et al. (2023). *Learning Lightweight Object Detectors via Multi-Teacher Progressive Distillation*. ICML 2023. [arXiv:2308.09105] — detection-specific 主证据
- Mirzadeh, S.I., et al. (2020). *Improved Knowledge Distillation via Teacher Assistant*. AAAI 2020. [arXiv:1902.03393]
- Cho, J.H. & Hariharan, B. (2019). *On the Efficacy of Knowledge Distillation*. ICCV 2019. [arXiv:1910.01348]
- Liu, Yueying, et al. (2020). *Adaptive Multi-Teacher Multi-Level Knowledge Distillation*. Neurocomputing 415: 106–113. (期刊；分类证据，**不作检测器主支撑**)
- Furlanello, T., et al. (2018). *Born-Again Neural Networks*. ICML 2018. [arXiv:1805.04770]

#### 检测器 KD 基础
- Zheng, Z., Ye, R., Wang, P., et al. (2022). *Localization Distillation for Dense Object Detection*. CVPR 2022, pp. 9407–9416. (TPAMI 2023 扩展)

#### 失败模式 / 数据增强相互作用
- Wang, T., et al. (2022). *Exploring Inconsistent Knowledge Distillation for Object Detection with Data Augmentation*. [arXiv:2209.09841]

#### KD 综述
- Klempau, M., et al. (2026). *Knowledge Distillation in Object Detection: A Survey from CNN to Transformer*. Sensors 26(1):292, MDPI.
- Liu, Z., et al. (2023). *When Object Detection Meets Knowledge Distillation: A Survey*. IEEE TPAMI.

### 训练增强（survey 来源）
- Ghiasi et al., "Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation," CVPR 2021, [arXiv:2012.07177]
- Cui et al., "Class-Balanced Loss Based on Effective Number of Samples," CVPR 2019, [arXiv:1901.05555]
- Lin et al., "Focal Loss for Dense Object Detection," ICCV 2017, [arXiv:1708.02002]
- Shrivastava et al., "Training Region-based Object Detectors with Online Hard Example Mining," CVPR 2016, [arXiv:1604.03540]

### 推理增强 + 系统集成（survey 来源）
- Akyon et al., "Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection," ICIP 2022, [arXiv:2202.06934]
- Solovyev et al., "Weighted Boxes Fusion: Ensembling Boxes from Different Object Detection Models," 2019, [arXiv:1910.13302]
- Possatti et al., "Traffic Light Recognition Using Deep Learning and Prior Maps," IJCNN 2019, [arXiv:1906.11886]
- Behrendt et al., "A Deep Learning Approach to Traffic Lights: Detection, Tracking, and Classification," ICRA 2017
- NVIDIA TensorRT QAT Toolkit (Pytorch-Quantization)
- Apollo Perception Traffic Light Module（开源代码）

### 部署平台 / 法规
- NVIDIA (2023). JetPack 5.1 Release Notes. (TensorRT 8.5.2 / CUDA 11.4 / cuDNN 8.6.0)
- AQSIQ / SAC (2011). GB 14887-2011 *道路交通信号灯*.
- AQSIQ / SAC (2016). GB 14886-2016 *道路交通信号灯设置与安装规范*.

### 项目内部
- [`../../research/surveys/detection_enhancements.md`](../../research/surveys/detection_enhancements.md) — 本计划新组件方法目录来源
- [`development_plan.md`](development_plan.md)、[`temporal_optimization_plan.md`](temporal_optimization_plan.md)、[`cross_detection_reasoning_plan.md`](cross_detection_reasoning_plan.md) — 衔接计划
- [`../../components/knowledge_distillation/`](../../components/knowledge_distillation/__init__.py) — KD scaffold（v1.3 LOCKED）
- [`../../components/temporal_shift_module/`](../../components/temporal_shift_module/__init__.py) — TSM scaffold（v1.5 AGREED）
- [`../reports/phase_2_round_1_report.md`](../reports/phase_2_round_1_report.md) — R1 实测痛点来源

---

## 十四、Conflictor-loop 终止条件

**适用范围**：本计划任一组件的 scaffold 阶段（B2 + C3）/ 决策规则修订 / 配置变更，均按本节终止条件运作。

**终止条件**：直到 AGREED；不设 iter 硬上限。

**Lock 条件**（同时满足）：当 iter 零 CRITICAL（决策规则歧义 / citation 错 / license 法务）+ HIGH 全部 amend + verdict ∈ {AGREED, AGREED-WITH-AMENDMENTS} + 仅剩 MEDIUM/LOW 非阻塞。

**Reopen criteria**（KD 部分继承自 v1.3）：§七.6 验收门执行歧义、§七.5.1 cell 触发循环依赖、§七.5.5 GO-LSD 论证错、§七.4.2 DETR↔YOLO 论证错、§七.8 license 收到反向法务意见、§十三 citation 错（参考文献节）、§七.2.4 容量分析被实测推翻（VRAM 偏差 > 30%）。其他发现进入 transcript 附录，不触发 amendment 循环。

非 KD 组件（§三-§六、§八-§十一）的 reopen criteria 视该组件特性独立设定；通用规则：决策规则歧义 / 关键 citation 错 / 工作量被实测推翻 ≥ 50%。
