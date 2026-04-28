# 跨检测共现推理计划（同帧多目标关系）

> **状态（2026-04-27）**：计划文档处于早期阶段（可行性已论述、实施细节已落到天级粒度，但前提验证须在 R2 自采数据上完成才能正式启动）。**未列入路线图**，作为 R3 可选优化轨道的候选。
>
> **核心问题**：能否利用同一帧内多个交通灯之间的"内在关联"来消歧难分类别？例如：若检测到一个高置信度的圆灯红灯，另一个置信度边缘的检测在 `red` / `redLeft` 之间摇摆，能否用同帧关系给出倾向？
>
> **重要前提**：本计划的核心机制是"用训练数据估出的类别共现先验改写检测概率"。R1 训练集（LISA / BSTLD / S2TLD）属于异域数据，将在 R2 自采落地后**整体废弃**——因此 R1 数据上的任何共现统计**不能**作为对部署场景前提的验证、也不能作为预期增益的依据。本文不引用 R1 数据上的共现数字；前提验证（§1）和先验估计（§2.2(b)）均要求在 R2 自采数据上做。
>
> **关键结论（提前给出）**：
> 1. 用户提出的"同帧多灯典型互不相同 / 一圆一向"**前提目前不可证伪**——必须等 R2 自采数据到位后，在目标场景上重新统计才能判定；
> 2. 方法本身（贝叶斯共现后处理 / CRF / Relation Network）在文献上成立，但其在本项目的增益**完全取决于** R2 自采数据上重估的先验是否携带可用信号；
> 3. 启动顺序（按风险/收益排序）：**主检测器选型 → 时序优化轨道 → R2 数据收集 → 本计划在新数据上重估先验后再决定是否启动**；
> 4. 本计划**不解决**主要痛点（小目标 / 遮挡漏检），那是检测器或时序工作的事；只对"同帧多灯、边缘类别置信度抖动"这一窄场景可能有效。

---

## 0. 定位与启动条件

### 0.1 本计划想解决什么问题

**目标问题**：同一帧内有多个交通灯检测，部分检测的类别概率边缘（多个类置信度接近）。能否利用其他高置信度检测的类别**统计共现关系**给出倾向，降低分类抖动？

**机制类型**：空间维度（同帧内）的关系推理。与已有 / 计划的工作正交：

| 维度 | 方法 | 状态 |
|---|---|---|
| 时间维度（跨帧） | tracker + EMA / TSM / HMM | Plan A 已落地；TSM/HMM 在 [`temporal_optimization_plan.md`](temporal_optimization_plan.md) |
| **空间维度（同帧）** | **共现先验 / CRF / Relation Net** | **本计划** |
| 单目标维度 | 检测器本身 | 主线 R2 选型 |

### 0.2 本计划**不**解决的问题

| 痛点 | 是否本计划能救 | 原因 |
|---|---|---|
| 小目标漏检 / 远距离漏检 | ❌ | 检测器没看到 → 无概率分布可用于推理 |
| 遮挡漏检 | ❌ | 同上 |
| 任意类的极少样本问题 | ❌ | 先验估计需要 pair 计数；样本少的类，先验本身不可靠（甚至负贡献）—— 该问题应回到数据采集 / 增强而非共现 |
| 类别边缘抖动（*已有时间序列* + EMA 后仍存） | ⚠️ 可能 | 见 §0.3 启动条件 |
| 同色不同型（`red` vs `redLeft`）混淆 | ⚠️ 可能 | 取决于 R2 数据上的实际共现分布；不可预判 |
| 不同色（`red` vs `green`）混淆 | ⚠️ 可能 | 同上 |

### 0.3 启动条件

**严格不启动**：
- R1 / R2 主检测器未完成；
- 时序优化轨道（[`temporal_optimization_plan.md`](temporal_optimization_plan.md)）未跑过；
- 实车 replay 未发现"剩余的边缘类别混淆"是显著痛点；
- **R2 自采数据未到位**——没有目标场景的 pair 统计就启动 = 用 R1 异域先验污染部署。

**可启动**：以上四条全部满足。然后再做 §1 描述的方法验证步骤：在 R2 自采数据上重估先验，看是否携带"超过随机猜测"的信号。

---

## 1. 前提验证（必须在 R2 自采数据上做）

### 1.1 为什么不能用 R1 数据验证

R1 训练集（LISA + BSTLD + S2TLD）将随 R2 自采到位**整体废弃**：
- LISA 美国路口、BSTLD 德国路口、S2TLD 早期中国路口——**没有一个**对应当前部署场景的中国路口配置；
- 同帧多灯的组合模式（多少灯 / 灯型组合 / 颜色组合）强依赖路口工程：信号相位编排、灯杆数量、方向灯使用习惯都由当地交通法规和工程实践决定；
- 因此 R1 数据上的共现统计是**异域先验**，对部署场景无证据效力。

→ 任何"基于 R1 数据 X% 帧含两个圆灯，所以本计划假设成立 / 不成立"的论述都是**错误的**。

### 1.2 R2 自采到位后必须做的验证步骤（启动前的 hard gate）

```
R2 自采视频（含 track ID）
    │
    ▼ 主检测器（R2 baseline）逐帧推理 + 抽样人工核验
[per-frame Detection]
    │
    ▼ 按 §2.2(a) 规则做同帧聚类
[per-cluster pair list]
    │
    ▼ 以下三道闸验证前提
        ① 多灯帧占比是否够高（建议 ≥ 50%）
        ② 类对共现分布是否显著非均匀（KL(prior || uniform) ≥ 阈值）
        ③ 边缘子集（top-1 conf < 0.7）上，先验 + 似然的联合判断 vs 仅似然，
           在抽样人工验证集上是否有可测正增益（pp）
    │
    ├── 三闸全过 → 启动 §2 实施
    ├── ① / ② 不过 → 部署场景多数单灯 / 共现弱，方法不适用，关闭计划
    └── ③ 不过 → 即便共现非均匀也不带分类有效信号，关闭计划
```

**关键**：上述三闸的具体阈值不在本文预设——必须在 R2 数据到位、看到分布形状后由团队定。本文不预设"应当大概是 +X%"。

### 1.3 用户原假设的处理方式

> 原假设：*"in a single frame, two traffic lights are typically different. if one detected round, the other would be likely to be directional."*

按 §1.1：**目前不可证伪，也不可证实**。该假设的真假取决于部署场景路口的工程配置，必须以 R2 自采数据为准。

需要注意的设计风险（无论假设方向如何）：
- 若实际分布是"同类同色为主"，则先验起的是**颜色一致性正则**作用——对"red vs green"边缘混淆有效；
- 若实际分布是"互补为主"（用户假设），则先验起的是**图标互补提示**作用——对"round vs arrow"边缘混淆有效；
- 两种情况都是同一套 §2 后处理框架，**不需要为不同假设写两套实现**——先验矩阵自动反映分布。

→ 因此本文 §2 的方法设计**不依赖**对假设方向的预判；只依赖"R2 数据上确实有非均匀共现分布"这一弱前提，由 §1.2 的闸 ② 检验。

### 1.4 期望增益估计

**不预估**——本计划增益的所有数值依赖部署场景共现分布形状，必须在 §1.2 闸 ③ 的抽样验证中实测。在 R2 数据到位之前给出 mAP 增益数字属于不诚实的预测。

定性而言：
- 上限：文献中相似机制（关系推理 / CRF + 后处理）在 COCO 等通用场景上 +1–2pp mAP；TL 类间关系比 COCO 简单，理论上限不会更高；
- 下限：0 或负值（先验若不能稳定区分难分类组合，会造成抖动）；
- 与同期可选优化项相比，**预期上限低于时序工作**（TSM 救漏检 = 检测端，本计划仅救分类端）。

---

## 2. 推荐路径：贝叶斯共现后处理

### 2.1 机制

每帧检测后、tracker 之前，对每个检测的 `class_probs` 做联合调整：

```
输入：  D = {d_1, ..., d_K}，每个 d_i 有原始 class_probs p_i ∈ R^C
先验：  π(c_i, c_j)  =  P(类别 c_i 与类别 c_j 同帧)（C×C 矩阵，从训练集统计）
聚类：  cluster(d_i)  =  与 d_i "共属同一交通灯组" 的检测集合
更新：  对每个 d_i：
        p_i'(c) ∝ p_i(c) · ∏_{d_j ∈ cluster(d_i)} (∑_{c_j} p_j(c_j) · π(c, c_j) / Z)
```

数学上等价于一阶 mean-field on a fully-connected pairwise CRF，unary = detector confidence，pairwise = co-occurrence prior。

### 2.2 关键设计选择

#### (a) 聚类规则（同一"交通灯组"的判定）

**重要**：不能对全帧所有检测一视同仁——可能有多个路口的灯同时出现。

候选规则（从简到繁）：

1. **全帧聚类**（最简单）：所有检测视为一组；
2. **图像 y-坐标桶**：高度差 < 0.1 × image_height 视为同组；
3. **空间距离阈值**：欧氏距离 < threshold（e.g. bbox 高度的 5 倍）；
4. **聚类算法**：DBSCAN over bbox centers；
5. **学习的聚类**：小型 GNN 给检测对打"是否同组"分。

**推荐起步：规则 2 或 3**——可解释、零训练、易调；如果效果不够再上 4 / 5。

#### (b) 先验估计

从 **R2 自采数据**（不是 R1）的标注 + Laplace smoothing 直接计数：

```python
# scripts/build_cooccurrence_prior.py（启动后产出）
# 输入：R2 自采标注（YOLO 格式）；输出：weights/cooccurrence_prior_r2.npy (C×C)
counts = collections.Counter()
total = 0
for label_file in r2_labels:                       # 仅 R2 自采，禁用 R1 数据
    classes = sorted(read_classes(label_file))
    for i in range(len(classes)):
        for j in range(len(classes)):
            counts[(classes[i], classes[j])] += 1
            total += 1
prior = np.zeros((C, C))
for (a, b), n in counts.items():
    prior[a, b] = (n + 1) / (total + C * C)        # Laplace
prior = 0.5 * (prior + prior.T)                    # 对称化更稳定
```

注意：
- **训练数据来源严格限定为 R2 自采**——R1 异域先验会污染部署判断；
- Laplace smoothing 必须，否则任何零计数组合会让乘法清零；
- 验证集与训练集同源（都来自 R2 自采的不同视频切分）；
- 部署若跨城市 / 跨路口形态，建议**按地理 / 路口类型分别估出多份先验**而非全局一份。

#### (c) 集成位置

```
detector → [postprocess: NMS + class_probs 输出] → ★共现后处理（本计划）★
                                                        ↓
                                                    ByteTrack 关联
                                                        ↓
                                                    EMA 投票（已落地）→ 输出
```

**关键**：共现后处理在 NMS 之后、tracker 之前。tracker 复用刷新后的 `class_probs`。

#### (d) 集成模式与计算预算

- 推理：单帧 K 个检测，C 类时复杂度 O(K² · C²)。K=10、C=14 → 19,600 次浮点乘法 → < 0.1 ms。
- 不需要 ONNX / TRT 改造（纯 C++ 矩阵运算）。
- 不修改检测器训练流程；仅 inference 端加模块。

#### (e) 先验来源扩展（planner-prior / 路由意图）

本节的贝叶斯 mean-field 框架对**先验来源不敏感** —— 只要先验是 C×C 矩阵或可表示为类条件概率，就能直接接入。除 §2.2(b) 的"R2 自采共现统计"之外，可叠加 / 替换的先验来源：

- **规划模块路径意图 prior**：规划知道下一路口将"左转 / 直行 / 右转"，对应期望灯型集合不同（左转期望 `redLeft / greenLeft`；直行期望 `forwardRed / forwardGreen` 或圆灯）。把 P(class | route_intent) 编码为 C 维 prior，与共现 prior 加权融合。
- **HD 地图 / 路网模板 prior**：路口模板携带"该路口存在哪些灯型"（基础朝向、是否有左右转分车道灯），不存在的类直接概率压低。
- **几何 / 上下文 prior**：检测在画面中的位置（dashcam 上 1/3 vs 下 2/3）、bbox 尺度（远距 vs 近距）作为类条件先验。

集成方式（优先级）：

```
log p_i'(c) ∝ log p_i(c)                          # 检测器似然
           + α₁ · log Σ_j π_cooc(c, c_j) · p_j(c_j)  # §2.2(b) 共现先验
           + α₂ · log π_planner(c | route_intent)    # 路径意图先验（如可得）
           + α₃ · log π_geom(c | bbox_pos, scale)    # 几何先验（如启用）
```

α₁/α₂/α₃ 由各自 prior 在 R2 验证集上的可靠性决定（弱信号 → 小权重 / 直接关闭）；任一来源的接入**不需要**重写 §2 / §3 数学，只新增一条 prior 项。

**Planner-prior 是与本计划合作最自然的扩展**（理由）：
- 与共现先验同为"对类别加偏置"的语义；
- 集成位置、计算预算、ROS2 接线点完全相同；
- 启动条件几乎不变 —— 只多一个"规划模块发布路径意图主题"的接口前置条件（详见 [`development_plan.md`](development_plan.md) §R2 推理 / 集成增强 决策项）。

### 2.3 实施步骤（启动后，预算 3–5 天；前提：R2 自采到位 + §1.2 三闸已过）

**Day 1**：

- [ ] `scripts/build_cooccurrence_prior.py` —— 从 R2 自采标注输出 `weights/cooccurrence_prior_r2.npy`（C×C）
- [ ] 离线 sanity check：先验矩阵打印 + 与 §1.2 闸 ② 估算的 KL 一致性核对

**Day 2**：

- [ ] `inference/spatial/cooccurrence.py` —— Python 实现，规则 2 聚类（y-坐标桶）+ mean-field 单步迭代
- [ ] 单元测试 + R2 自采验证集 mAP 对比（baseline vs +cooccurrence）

**Day 3**：

- [ ] 验证集分桶分析：仅在"边缘置信度子集"（top-1 conf < 0.7 且 top-2 conf > 0.4）上看增益
- [ ] 决策门：边缘子集上有可观测正增益（具体阈值由团队在看到分布后定，不在本文预设）才继续

**Day 4–5**（仅决策门通过）：

- [ ] C++ 端口（`inference/cpp/src/cooccurrence.cpp` ~150 行）
- [ ] 端到端 demo 视频对比（开 / 关）
- [ ] 文档化：`docs/integration/cooccurrence_guide.md`（参照 `tracker.md` 格式）

### 2.4 风险

| 风险 | 缓解 |
|---|---|
| 先验"过强"——把检测器看到的少数类压成多数类 | confidence-gated 应用：仅当原始 top-1 conf < 阈值（e.g. 0.6）才注入先验 |
| 不同路口的灯被错误聚类 | 起步用 y-bucket（同水平带）+ 距离阈值；不行再上 DBSCAN |
| R2 自采尚不充分时启动，先验欠采样 | 启动前 §1.2 闸 ② 强制检验先验稳定性（bootstrap CI 宽度），不达标不启动 |
| 任何小样本类（在 R2 上仍少的类）被先验惩罚 | Laplace smoothing 强度调高；或对这些类绕过先验（仅用原始概率） |
| 边缘场景增益不显著 | 决策门拦下（Day 3），不进入 C++ 端口 |
| 跨城市 / 跨路口形态部署，单一先验不通用 | §2.2(b) 的"按地理 / 路口类型多份先验"分桶部署 |

---

## 3. 备选路径：CRF 联合 MAP

### 3.1 何时选

§2 mean-field 单步迭代不收敛 / 边缘子集仍抖动时，升级到联合 MAP 推理：

```
argmax_{c_1, ..., c_K}  Σ_i log p_i(c_i) + Σ_{i<j} log π(c_i, c_j)
```

**何时可用**：
- K ≤ 5（绝大多数帧）：暴力枚举 14^5 ≈ 5 × 10^5 → < 5 ms 可接受；
- K > 5：用 belief propagation 或 graph cut（仅二值时）。

**优势**：解最优而非近似；可解释性高（先验 + 似然加权显式）。

**劣势**：
- 计算量随 K 指数增长（虽然实际 K 多 ≤ 5）；
- 仍是离散决策——失去 class_probs 的连续语义，下游 tracker EMA 无法继续利用概率。

**结论**：仅当 §2 mean-field 经实测不够时考虑；优先级低于 §2 + 自适应聚类。

---

## 4. 备选路径：Relation Network 模型改造

### 4.1 何时选

§2 / §3 后处理收益封顶（架构上限：先验只能弯曲已有概率，不能凭空补出新信息）。如果想突破这层，必须把"检测之间的关系"在**特征级**建模——即模型改造（model surgery）。

### 4.2 文献参考

- **Hu et al., "Relation Networks for Object Detection" CVPR 2018** ([arXiv:1711.11575](https://arxiv.org/abs/1711.11575))：在 Faster R-CNN 头部插入 object relation module（geometry + appearance attention，无外部监督）。COCO mAP +1.5–2pp。
- **RGRN: Relation-aware Graph Reasoning Network** (Springer 2023)：GCN over detection 节点 + 共现矩阵作为先验图，refine classification 与 bbox。
- **DETR self-attention 已天然包含 query-to-query attention**：但 stock DETR 不显式以共现为目标训练；扩展可加 auxiliary loss。

### 4.3 对本项目的实际门槛

- **必须重训检测器**——relation module 改变了 head 行为，直接在 R2 baseline 上叠加无效；
- **与主检测器架构强耦合**——YOLO26 / YOLOv13 / DEIM 的 head 结构差异较大，每个都要单独 patch；
- **训练复杂度上升**——批内多目标的 attention 增加显存和耗时（小 batch 时 OK，但 4090 24GB 跑 1280 + relation 可能 OOM）；
- **TRT 8.5 上 attention 算子需谨慎**（参见时序计划 §2.5 同样的风险）；
- **预期增益**：文献 +1.5–2pp，本项目预期更低（TL 类间关系比 COCO 简单）。

**结论**：相比 §2 后处理，工作量数量级更高（数周）、收益不明显更高。**不推荐除非 §2 已落地且仍有显著边缘类别痛点**。

### 4.4 如果硬要做的简化版

不做完整 relation module，**只做轻量化版本**：

- 在主检测器 head 之后加一个小 MLP，输入 `[own_class_probs (C) + max_neighbor_probs (C)]`，输出 refined `class_probs (C)`；
- "neighbor" 由空间规则定义（不可学习）；
- 训练数据：同 R2 主线 dataset；loss = CE on labels；
- 参数量 ~1k。

仍需主检测器在 inference 时同时吐 K 个检测的 features —— 这点 batch-mode 自然支持。预算：~1 周。

---

## 5. 已排除路径

| 方案 | 排除原因 |
|---|---|
| 全场景 GNN（T2SG / SCENE 风格） | 重量级；为整个交通场景建模过度；与"灯之间消歧"问题不匹配 |
| HD 地图先验（Bosch / Possatti 2019） | 解决的是"哪个灯与本车道相关"的 relevance estimation，不是分类消歧；需先期建图能力 |
| 端到端 LLM 判断 | 不在 Orin 实时预算内；仅适合云端后处理 |
| 整批检测送 Transformer 联合分类 | 训练数据要求大；本质上是 §4 的扩展，性价比反降 |
| 强化学习 / 主动推理 | 训练数据不支持 |

---

## 6. 与现有计划的关系

| 计划 | 关系 |
|---|---|
| R1 落地 | 不影响 |
| R2 主检测器选型 | 不影响、不阻塞、不被阻塞 |
| 时序优化轨道（[`temporal_optimization_plan.md`](temporal_optimization_plan.md)） | **优先于**本计划。时序救漏检 + 跨帧抖动；本计划只救同帧分类抖动；前者覆盖面更广 |
| Plan A（tracker + EMA） | 本计划集成在 EMA 之**前**；不替代、不冲突 |
| R2 数据采集 SOP | 本计划复用主线连续视频数据；不对采集提额外要求 |
| 2026-05-15 截止 | 本计划**预期不在 5/15 前启动**——优先级低于主线 + 时序轨道 |
| 检测增强调研（[`../../research/surveys/detection_enhancements.md`](../../research/surveys/detection_enhancements.md)） | §2.2(e) planner-prior / 几何 prior 是该调研 §4.7 的具体落地形式 —— 一旦规划模块在 ROS2 上发布"路径意图"主题，本计划框架可直接吸收（无需新写） |

---

## 7. 决策流程

```
主检测器选型完成（5/15 截止）
    │
    ▼
R2 自采数据到位
    │
    ▼
实车 replay
    │
    ▼ §0.3 启动条件
        ├── 漏检为主              → 时序优化 §1（TSM）
        ├── 类别抖动为主          → 时序优化 §2（HMM/AdaEMA）
        └── 上述两类问题已大致解决，
            但同帧多灯仍有边缘混淆 → §1.2 三闸验证（在 R2 数据上）
                │
                ├── 任一闸不过 → 关闭计划，回到时序 / 数据 / 检测器侧
                └── 三闸全过   → ★ 本计划 §2 启动 ★
                                    │
                                    ▼ Day 3 决策门（在 R2 验证集上）
                                        ├── 边缘子集有可观测正增益 → 进 C++ 端口（Day 4–5）
                                        └── 无显著增益              → 关闭计划
```

---

## 8. 风险（综合）

| 风险 | 缓解 |
|---|---|
| 用 R1 异域数据预判部署场景前提（本文初稿犯过此错） | §1.1 / §1.2 强约束——前提验证必须在 R2 自采数据上做；R1 数据禁止用于先验估计或前提判定 |
| 启动过早，与时序工作冲突 | §0.3 严格守门——时序轨道未跑过本计划不启动 |
| R2 自采尚不充分时启动，先验欠采样 | §1.2 闸 ② 检验先验稳定性（bootstrap CI 宽度），不达标不启动 |
| 部署场景跨城市 / 跨路口形态，单一先验不通用 | §2.2(b) 多份先验分桶部署 |
| 任何小样本类被先验压垮 | confidence-gated + Laplace smoothing；必要时绕过这些类 |
| 工作量看似小但 C++ 端口隐藏成本 | 决策门严格——Day 3 无显著增益不进 C++ |
| 与 tracker / EMA 顺序错乱导致重复平滑 | 集成位置固定：detector → cooccurrence → tracker → EMA；接口在 `inference/spatial/cooccurrence.py` 单点维护 |

---

## 9. 参考资料

### 关系推理 / 共现先验
1. Hu et al., "Relation Networks for Object Detection," CVPR 2018, [arXiv:1711.11575](https://arxiv.org/abs/1711.11575) —— object relation module 原文
2. RGRN: "Relation-aware graph reasoning network for object detection," Neural Comput. Appl. 2023 —— 共现矩阵作 GCN 图先验
3. "Detecting Objects with Graph Priors and Graph Refinement" —— 从共现统计推断图先验
4. Carion et al., "End-to-End Object Detection with Transformers (DETR)," ECCV 2020, [arXiv:2005.12872](https://arxiv.org/abs/2005.12872) —— object query 间 self-attention（本计划 §4.4 简化版的灵感来源）

### CRF / 结构化预测
5. Krähenbühl & Koltun, "Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials," NeurIPS 2011 —— mean-field 高效推理
6. Zheng et al., "Conditional Random Fields as Recurrent Neural Networks," ICCV 2015, [arXiv:1502.03240](https://arxiv.org/abs/1502.03240) —— CRF 端到端训练

### 交通灯领域
7. Behrendt et al., "A Deep Learning Approach to Traffic Lights: Detection, Tracking, and Classification," ICRA 2017 —— Bosch BSTLD 原文，含 RNN 时序上下文
8. Possatti et al., "Traffic Light Recognition Using Deep Learning and Prior Maps," IJCNN 2019, [arXiv:1906.11886](https://arxiv.org/abs/1906.11886) —— HD map 先验过滤候选（本计划 §5 排除参考）
9. ATLAS Traffic Light dataset, 2024, [arXiv:2504.19722](https://arxiv.org/abs/2504.19722) —— 含 pictogram 细粒度标注，先验估计的潜在外部数据源

### 项目内部
10. [`temporal_optimization_plan.md`](temporal_optimization_plan.md) —— 时序轨道；本计划在其之**后**启动
11. [`../integration/tracker.md`](../integration/tracker.md) —— Plan A tracker + EMA；本计划在 EMA 之**前**集成
