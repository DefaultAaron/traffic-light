# 跨检测共现推理计划 v2.0

## 状态

| 项 | 当前值 |
|---|---|
| 定位 | R3 可选优化轨道；同帧空间维度关系推理 |
| 当前状态 | 未列入 R2 路线图；R2 自采数据到位后才可验证 |
| 前提 | R1 数据不得用于先验估计或部署场景前提判断 |
| 启动顺序 | 主检测器 → 时序优化 → R2 自采数据 → 共现验证 |
| 生命周期 | 见 `additional_components_plan.md` §一 |

| 目标问题 | 是否本计划处理 |
|---|---|
| 同帧多灯、边缘类别置信度抖动 | 是 |
| `red` vs `redLeft` / round vs arrow 边缘混淆 | 条件性 |
| 小目标 / 远距离 / 遮挡漏检 | 否 |
| 任意小样本类本身 | 否 |
| 非法跨帧状态转移 | 否；见 `temporal_optimization_plan.md` §2 |

## Deferred

| 项 | 状态 | 入口 |
|---|---|---|
| 共现先验后处理 | blocked on R2 data + three-gate validation | §1 / §2 |
| CRF 联合 MAP | blocked on §2 mean-field insufficient | §3 |
| Relation Network | blocked on §2 / §3 insufficient and persistent issue | §4 |
| Planner-prior 接入 | blocked on planning topic + framework b-stage | `additional_components_plan.md` §十一 |

## 行动项

### 0. 启动条件

全部满足才启动：

- [ ] R2 主检测器完成训练、导出、Orin replay。
- [ ] `temporal_optimization_plan.md` 对应路径已跑过或明确不适用。
- [ ] replay 中仍有同帧多灯边缘类别混淆。
- [ ] R2 自采数据可用于 pair 统计。
- [ ] R2 标注 / track / replay artifacts 可 hash-pin。

严格不启动：

- [ ] 使用 R1 / LISA / BSTLD / S2TLD 共现统计作为部署先验。
- [ ] 漏检是主痛点。
- [ ] 时序跳变是主痛点且 HMM / EMA 未跑。

### 1. 前提验证

```text
R2 自采视频 + track ID
  -> R2 baseline 推理 + 抽样人工核验
  -> per-frame detections
  -> 同帧聚类
  -> pair list
  -> 三道闸
```

三道闸：

| Gate | 检查 | outcome |
|---|---|---|
| ① | 多灯帧占比够高 | 不够 → close plan |
| ② | 类对共现分布显著非均匀 | 不够 → close plan |
| ③ | 边缘子集上 prior + likelihood 优于 likelihood-only | 不够 → close plan |

边缘子集默认：top-1 conf < 0.7 且 top-2 conf > 0.4。阈值在 R2 数据 freeze 后预承诺并写入验证 manifest。

### 2. 贝叶斯共现后处理

#### 2.1 机制

```text
输入：D = {d_1 ... d_K}; 每个 d_i 有 class_probs p_i ∈ R^C
先验：π(c_i, c_j) = P(c_i 与 c_j 同帧)
聚类：cluster(d_i) = 与 d_i 同交通灯组的 detection 集合
更新：p_i'(c) ∝ p_i(c) * Π_j Σ_cj p_j(c_j) * π(c, c_j) / Z
```

集成位置：

```text
detector -> NMS / postprocess -> cooccurrence -> ByteTrack -> EMA / smoother -> output
```

#### 2.2 行动项

- [ ] a. `scripts/build_cooccurrence_prior.py`：R2 自采标注 → `weights/cooccurrence_prior_r2.npy`。
- [ ] a. prior metadata：R2 manifest sha、class map sha、Laplace α、聚类规则。
- [ ] b. `inference/spatial/cooccurrence.py`：y-bucket 或 distance threshold 聚类 + mean-field 单步。
- [ ] b. 禁用 R1 数据输入。
- [ ] c. baseline vs +cooccurrence；全量 + 边缘子集分桶。
- [ ] c. Day 3 gate：边缘子集正增益才进入 C++。
- [ ] d. 写 `runs/_cooccurrence_decision.json`。
- [ ] e. phase report 子节。

#### 2.3 聚类规则

| 优先级 | 规则 | 状态 |
|---|---|---|
| 1 | y-coordinate bucket | 起步 |
| 2 | bbox center distance threshold | 起步备选 |
| 3 | DBSCAN over centers | R3+ if needed |
| 4 | learned pair classifier / GNN | R3+ if needed |

不得默认全帧所有 detection 属同组，除非 validation manifest 显式锁定该规则。

#### 2.4 prior 来源

| 来源 | 状态 |
|---|---|
| R2 自采共现统计 | 主路径 |
| planner route intent | R3+；见 `additional_components_plan.md` §十一 |
| HD map / road template | R3+ |
| geometry / bbox position | R3+ |
| R1 数据 | 禁用 |

### 3. CRF 联合 MAP

仅 §2 mean-field 不够时启动。

```text
argmax_{c_1 ... c_K} Σ_i log p_i(c_i) + Σ_{i<j} log π(c_i, c_j)
```

- [ ] K ≤ 5：可暴力枚举。
- [ ] K > 5：belief propagation 或其他近似。
- [ ] 输出应保留 class probability 或记录 tracker EMA 接口变化。
- [ ] 写入 `runs/_cooccurrence_decision.json` branch=`crf_map`。

### 4. Relation Network / 轻量 MLP

仅 §2 / §3 已落地且边缘类别痛点仍显著时启动。

最小版：

- [ ] 输入：`own_class_probs(C) + max_neighbor_probs(C)`。
- [ ] 输出：refined `class_probs(C)`。
- [ ] neighbor 由固定空间规则定义。
- [ ] loss = CE on labels。
- [ ] 参数量约 1k。
- [ ] 需要主检测器 inference 同时吐检测 features 时再评估。

## 决策规则

### 三闸验证

| outcome | condition | next |
|---|---|---|
| start §2 | Gate ① / ② / ③ 全过 | implement cooccurrence |
| close single-light / weak prior | Gate ① 或 ② 不过 | 回检测器 / 数据 / 时序 |
| close no useful signal | Gate ③ 不过 | 回检测器 / 数据 / 时序 |

### §2 Day 3

| outcome | condition | JSON |
|---|---|---|
| deploy-candidate | 边缘子集有预承诺正增益；总 mAP / 安全类 AP 不退化 | `outcome="deploy_candidate"` |
| defer | 正增益不稳定；需更大 held-out / 多 seed | `outcome="defer"` |
| drop | 无增益或负增益 | `outcome="drop"` |

### 风险 gates

| 风险 | gate |
|---|---|
| prior 过强压少数类 | confidence-gated：仅低置信度应用 |
| 错误聚类 | y-bucket + distance threshold；聚类规则写入 metadata |
| R2 样本不足 | bootstrap CI 宽度 gate |
| 小样本类被先验惩罚 | Laplace α sweep；小样本类 bypass |
| C++ hidden cost | Day 3 无正增益不进 C++ |
| 跨城市 / 路口形态 | prior 按地理 / 路口类型分桶 |

## 文件清单

| 路径 | 阶段 |
|---|---|
| `scripts/build_cooccurrence_prior.py` | §2 a-stage |
| `weights/cooccurrence_prior_r2.npy` | §2 a-stage artifact |
| `inference/spatial/cooccurrence.py` | §2 b-stage |
| `inference/cpp/src/cooccurrence.cpp` | §2 deploy-candidate after Day 3 |
| `runs/_cooccurrence_decision.json` | §2 d-stage |
| `docs/integration/cooccurrence_guide.md` | deploy-candidate report / integration |

## 衔接

- `development_plan.md`：R3 可选轨道定位；planner-prior 决策项。
- `temporal_optimization_plan.md`：本计划在时序路径之后启动。
- `additional_components_plan.md` §十一：planner-prior 融合走同一 prior 接口。
- `pre_r2_kickoff_checklist.md` §4.2：planner-prior carry-forward blocked_on `planning_team`。
- `docs/data/r2_data_collection_sop.md`：R2 raw video / track / negative controls 数据入口。
