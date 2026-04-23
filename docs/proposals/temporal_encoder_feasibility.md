# 时序编码器（LSTM/GRU 等）可行性分析

> **状态（2026-04-23）**：
> - **方案 A 已落地**（Python + C++ 两端，fixture 驱动单测通过，Orin 集成待触发）。实现与调参见 [`../integration/tracker_voting_guide.md`](../integration/tracker_voting_guide.md)。
> - **方案 B（per-track GRU）** 保留为 R3 候选（5/15 之后），`TrackedDetection.class_probs` 已作为 GRU 的 soft 输入接口暴露。
> - 方案 C/D/E 不在路线图上。
>
> 本文档仍作为"为什么选 Plan A 而非 LSTM/GRU/ConvLSTM"的依据保留；实施细节以 tracker_voting_guide.md 为准。

---

## 背景

当前检测是**逐帧独立**推理：每帧图像送入 YOLO26 → 输出 `Detection2DArray`。现实中信号灯状态在秒级内是稳定的（30–60 s 循环），但单帧置信度会因远距离、眩光、遮挡、抖动出现漂移，引发"闪烁"（连续帧里 `red → red → nothing → red → red`）以及偶发误分类（`red` 被误识为 `redLeft`）。

**想法**：在检测头之上（或骨干特征之上）增加一个轻量时序编码器（LSTM / GRU / 小型时序注意力），利用前 3–10 帧的信息消除闪烁、提升稳定性。

**关键判据**：
- **目的**：短期平滑、稳定性 > 新增 mAP
- **平台**：NVIDIA Jetson AGX Orin 64GB，TensorRT 8.5 / JetPack 5.1
- **预算**：R1 约 25 ms@1280，总预算 50 ms，剩余 ~25 ms
- **数据**：R1 数据集是静态图；仅 R2 自采数据会有连续视频帧

---

## 架构方案对比

### 方案 A：跟踪 + 类别投票（Tracker + EMA Voting）— **P0**

**机制**：每帧独立检测 → Kalman + Hungarian（ByteTrack / OC-SORT）跨帧关联 → 每条轨迹维护一个类别概率的 EMA 或多数投票缓冲（窗口 N=5–10）。

| 项目 | 值 |
|------|-----|
| 额外训练 / 数据 | **无** |
| Orin 额外延迟 | < 1 ms（CPU 侧 ByteTrack） |
| TRT 部署复杂度 | 无（纯后处理） |
| mAP 提升（文献） | +1–3 |
| 闪烁降低（文献） | 50–70% |

**依据**：Apollo、Autoware、Bosch BSTLD 参考实现均采用此方案，**不使用 LSTM**。

**优点**：投入最小；R1 权重即可受益；ByteTrack 运行在 CPU 不占 GPU 预算；与现有 `Detection2DArray` 兼容（追加 `tracking_id` 字段即可）。

**局限**：投票作用于"硬"决策，全帧置信度都低时救不回来；无法建模状态转移先验（`green → yellow → red` 合法等）。

---

### 方案 B：逐轨迹 GRU/LSTM 分类头 — **P1（R3）**

**机制**：检测器逐帧推理；跟踪器关联轨迹；每条轨迹维护一个小 GRU（hidden=32–64），输入为最近 N 帧的分类 logits，输出平滑后类别概率。

| 项目 | 值 |
|------|-----|
| 额外训练 | GRU 头（~10k 参数），**半监督**：R1 模型在未标注连续视频上生成伪标签再训 GRU |
| 额外数据 | R2 自采视频即可，不需额外标注 |
| Orin 额外延迟 | < 0.2 ms（20 条轨迹以内几乎免费） |
| TRT 部署复杂度 | 低（独立 TRT 子引擎，或 CPU onnxruntime） |
| mAP 提升 | +1–2；**分类稳定性显著改善** |

**依据**：Müller & Dietmayer（IV 2023，DriveU）在 YOLOv5 + GRU 上报告 **+2.3% state accuracy，帧间类别跳变降低约 60%** — 唯一明确对比"单帧 vs LSTM/GRU"的 TL 论文。

**优点**：直接针对我们的失败模式（单帧置信度边缘时融合 soft logits + 时序先验）；能学到状态转移先验；与检测模型解耦（R1 权重不动，只加小尾巴）。

**局限**：ROS2 下需维护 hidden state（TRT 引擎无状态，state 由节点持有）；伪标签训练若 R1 有系统性偏差会被 GRU 固化 — 需人工校验 ~100 段伪标签。

**半监督训练路径（推荐）**：
1. R1 模型对 R2 未标注连续视频跑检测 + ByteTrack
2. 每条轨迹取多数投票作为伪标签
3. 训练 GRU 预测该伪标签，加一致性损失（相邻帧输出应相似）
4. 采样 ~100 条轨迹人工校验，避免 R1 偏差被固化

---

### 方案 C/D/E（不在路线图）

| 方案 | 机制 | 关键问题 | 预期 |
|------|------|---------|------|
| C. ConvLSTM backbone | 在 backbone → neck 间插入 ConvLSTM 块（Liu & Zhu CVPR 2018） | 要求连续帧、逐帧标注、一致 track ID 的视频数据（BSTLD / LISA / S2TLD 均非）；非 ONNX 原生算子，TRT 8.5 融合不稳 | +1.5–3 mAP，3–6 ms 延迟 |
| D. 时序注意力 / TransVOD | 最近 N=3–5 帧特征上做交叉注意力 | 数据约束同 C；整机延迟 ≈ 2× detector | +3–5 mAP，5–10 ms 延迟 |
| E. FGFA 光流融合 | FlowNet / RAFT 把相邻帧 warp 到当前帧 | 光流网络本身 15–30 ms 吃光预算；TRT 需定制插件 | 排除 |

**共同结论**：训练数据约束（连续视频逐帧标注）+ TRT 8.5 算子支持问题使性价比差。R3 若有 DriveU 级标注视频 + JetPack 升级到 TRT 10，可重新评估 C / D。

---

## 方案对比总表

| 方案 | 额外延迟 | 训练要求 | 数据要求 | TRT 复杂度 | 预期增益 | 推荐优先级 |
|------|---------|---------|---------|-----------|---------|-----------|
| A. 跟踪 + 投票 | <1 ms (CPU) | 无 | 无 | 无 | +1–3 mAP / 闪烁 -60% | **P0（若需要 mitigation）** |
| B. 逐轨迹 GRU 头 | <0.2 ms | GRU 小头，半监督 | 未标注视频 OK | 低 | +1–2 / 状态跳变 -60% | **P1（R3）** |
| C. ConvLSTM backbone | 3–6 ms | 端到端重训 | 连续标注视频 | 中高 | +1.5–3 mAP | 不在路线图 |
| D. 时序注意力 | 5–10 ms | 端到端重训 | 连续标注视频 | 中 | +3–5 mAP | 不在路线图 |
| E. FGFA 光流融合 | +15–30 ms | 端到端重训 | 连续标注视频 | 高 | +3 mAP | ❌ 排除 |

---

## 交通灯场景特定证据

- **Bosch BSTLD 原文**：Kalman + IoU 跟踪 + 类别投票，未使用 LSTM
- **Apollo Perception**：HD Map ROI → 单帧 CNN 分类 → 5–10 帧投票缓冲。中国主流自动驾驶开源栈，方案 A 是事实标准
- **Autoware Universe**：同样使用 ring buffer 多数投票
- **Müller & Dietmayer IV 2023**：YOLOv5 + GRU 分类头，DriveU 上 +2.3% state accuracy，帧间跳变 -60%

**统一发现**：TL 时序工作的主要收益是**稳定性/闪烁降低**（50–70%），而非 mAP 大涨（+1–3%）。

---

## TensorRT 部署要点（JetPack 5.1 / TRT 8.5，若启用方案 B）

- **LSTM/GRU ONNX 导出**：opset 14+ 原生支持；TRT 8.5 将 RNN 视为 monolithic block，丢失与相邻算子融合；双向 LSTM FP16 数值不稳，单向 LSTM 正常。
- **固定序列展开**（推荐）：导出时把 N=5 展开为普通图，绕过 RNN 算子边界情况。
- **状态管理**：TRT 引擎无状态。hidden state 必须由 ROS2 节点持有，作为 input/output 进出引擎；轨迹消失或帧间隔 >200 ms 时重置。
- **动态序列长度**：避免；按需固定 N 构建多引擎。

---

## 推荐路径

### 方案 A — 已落地（2026-04-23）

```
YOLO26 检测（现有）
  → ByteTrack 关联（Python + C++ 两端自写，共享 JSON fixtures）
  → 每轨迹 EMA 类别缓冲（α=0.3，min_hits=3，track_buffer=30）
  → 带 tracking_id 的输出（Python TrackedDetection / C++ tl::TrackedDetection）
```

- 落地情况：`inference/tracker/*.py`（vendored ByteTrack MIT）+ `inference/cpp/{include,src}/tracker.{hpp,cpp}`；CLI 开关 `--track`
- 超参数：R2 实车 replay 数据到位前不锁定，以 fixture 金标 + `scripts/measure_flicker.py` 持续迭代
- 实施 / 调参 / 决策门：[`../integration/tracker_voting_guide.md`](../integration/tracker_voting_guide.md)

### R3（5/15 之后）：方案 B

- 用 R1 + ByteTrack 生成 R2 自采视频的伪标签轨迹
- 训练小 GRU（hidden=64，N=5 展开），导出独立 TRT 子引擎
- 集成进 ROS2 节点（state 由节点维护）
- 工作量：~1–2 周；预期：在方案 A 之上再降 30–50% 分类跳变，+1–2% state accuracy

---

## 待确认事项（若启用）

- [ ] R2 自采视频是否保留**连续帧**（而非稀疏抽帧），以支撑方案 B 半监督训练
- [ ] 规划模块是否接受新增 `tracking_id` 字段（`vision_msgs/Detection2D.tracking_id` 原生支持）
- [ ] 跟踪器选型（ByteTrack vs OC-SORT）在 Orin 上延迟对比
- [ ] 阈值设计：EMA α、投票窗口 N、轨迹最大断联帧数

---

## 参考资料

1. [Apollo Perception - Traffic Light Module](https://github.com/ApolloAuto/apollo/tree/master/modules/perception) — 方案 A 事实标准参考
2. [Müller & Dietmayer, IEEE IV 2023](https://arxiv.org/abs/2303.09503) — YOLOv5 + GRU on DriveU，方案 B 定量依据
3. [ByteTrack, ECCV 2022](https://arxiv.org/abs/2110.06864) — 方案 A / B 的推荐跟踪器
4. [TensorRT 8.5 Support Matrix — Recurrent Layers](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-856/support-matrix/index.html)
