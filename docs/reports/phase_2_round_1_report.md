# 第二阶段 · 第一轮训练报告（R1）— 方向性检测（7类）

> **文档状态**：**已关闭（CLOSED 2026-05-12）** — DEIM-D-FINE-L 完成后 R1 七模型全部齐备；后续仅追加 R1 期间遗留的 carry-forward 完成证据（per-class DEIM-L AP / Orin Gate 实测），不再调整决策与结论。
> **后续路径**：R2 训练侧入口见 [`../planning/development_plan.md`](../planning/development_plan.md) §Stage 1；R1→R2 之间的预 R2 消化项见 §R1→R2 桥接（预 R2 可执行消化项）。
> **历史更新策略**（R1 进行期间适用）：新增结果追加在既有章节下方；重大修订在"变更记录"中登记，不覆盖历史结论。

---

## 概述

第二阶段第一轮训练在第一阶段（3类颜色）基础上扩展到 **7 类方向性检测**（red / yellow / green / redLeft / greenLeft / redRight / greenRight）。R1 的目标：

1. 在现有合并数据集（S2TLD + BSTLD + LISA 的方向重标注子集）上验证 7 类训练可行性，取得基准指标。
2. 打通从训练到 Orin 端 TensorRT C++ 推理的端到端链路。
3. 识别性能瓶颈并为 R2 制定改进方案。

**训练硬件**：RTX 4090 D（远程 GPU 服务器）  
**部署硬件**：NVIDIA Jetson AGX Orin 64GB（JetPack 5.1.2 / TRT 8.5.2）  
**框架**：Ultralytics（AGPL-3.0）

---

## 训练配置

沿用第一阶段超参数，主要变化在数据（7 类）和早停：

| 参数 | 值 | 备注 |
|------|-----|------|
| 轮次 | 100 | 实际触发早停 |
| `patience` | 20 | 早停耐心窗口 |
| 图像尺寸 | **640×640** | 与 P1 一致 |
| 批次大小 | 自动 | |
| 优化器 | Auto（SGD） | |
| `fliplr` | **0.0** | 禁用水平翻转（箭头方向语义反转，保留图像但标签未相应翻转） |
| `flipud` | 0.0 | 同第一阶段 |
| Mosaic | 1.0 | 第10轮后关闭 |
| HSV | H=0.015, S=0.7, V=0.4 | |
| 预训练权重 | COCO（Ultralytics 官方） | |

**数据集路径**：`data/merged/` — 训练 / 验证划分沿用第一阶段策略，按主要类别分层抽样 80/20。

---

## 数据分布（训练集，来自 `runs/detect/yolo26n-r1/labels.jpg`）

| 类别 | 实例数 | 占比 | 评估 |
|------|-------|------|------|
| green | 51,661 | 44.4% | 充足 |
| red | 46,483 | 40.0% | 充足 |
| redLeft | 12,983 | 11.2% | 充足 |
| yellow | 3,141 | 2.7% | 偏少（延续 P1 问题） |
| greenLeft | 2,604 | 2.2% | 偏少 |
| **redRight** | **19** | **0.02%** | **几乎为零** |
| **greenRight** | **13** | **0.01%** | **几乎为零** |

**边界框形态特征**：
- 宽度分布峰值在 0.01–0.03（即 `imgsz=640` 下 6–20 像素宽）— 典型小目标。
- 位置分布集中在 x≈0.55–0.7、y≈0.3–0.4（左舵车驾驶视角的上方右侧）。

---

## R1 训练结果

> 完整六模型 + 容量上限对照指标见 §R1 跨架构综合分析 → 整体指标对照。本节只保留 P1 → R1 桥接审计。

### 与第一阶段（3类）对比

| 模型 | P1 mAP50 | R1 mAP50 | P1 mAP50-95 | R1 mAP50-95 | 说明 |
|------|---------|---------|------------|------------|------|
| YOLO26n | 0.866 | 0.720 | 0.601 | 0.484 | **-14.6 / -11.7 pp** |
| YOLO26s | 0.893 | 0.849 | 0.628 | 0.608 | **-4.4 / -2.0 pp** |

**观察**：
- YOLO26s 从 3 类扩到 7 类仅下降 2 pp mAP50-95，可接受。
- YOLO26n 下降显著（>10 pp），在右转 / greenLeft 等小样本类别上严重失衡，印证"容量不足的模型无法补偿类别不均衡"。
- R1 `patience=20` 在多类不均衡任务中可能偏紧（R2 改 `patience=40`）。

### R1 备选架构（并行训练）

为降低单一架构风险 + 探明精度上限，R1 期间并行训练两条备选轨道。选型依据与环境配置见 [`../../research/surveys/alt_detector_architectures.md`](../../research/surveys/alt_detector_architectures.md)。

| 模型 | 许可证 | 角色 | 训练轮次 | 最佳 mAP50 | 最佳 mAP50-95 | Precision | Recall | 状态 |
|------|-------|------|---------|-----------|--------------|-----------|--------|------|
| YOLOv13-s | AGPL-3.0 | YOLO26 同架构家族低风险对照 | 100 / 100 | 0.815 | 0.580 | 0.836 | 0.767 | 完整跑完，未触发早停 |
| DEIM-D-FINE-S | Apache-2.0 | 商用许可证备选 + 小目标精度探底 | 132 / 132 | 0.848 | 0.602 | 0.919 | 0.846 | **完成**，决策规则 4 触发（见 §DEIM-D-FINE 专项评估） |
| DEIM-D-FINE-M | Apache-2.0 | 精度上限基准 | 102 / 102（best=97）| 0.859 | 0.613 | 0.951 | 0.847 | **完成**（2026-05-11，1d 7:50:21）|
| DEIM-D-FINE-L | Apache-2.0 | DEIM 家族容量上限 + 潜在 KD 教师 | 102 / 102（best=72）| 0.857 | 0.611 | — | — | **完成**（2026-05-12；best_stg2 缺省按设计 — 见 §DEIM-D-FINE 专项评估 → DEIM-L 节）|
| YOLO26l-r1 | AGPL-3.0 | YOLO26 容量上限基准 | 48 / 100（best=28）| 0.850 | 0.619 | 0.918 | 0.703 | 早停收敛，9.51 h |

**三条轨道的分工**：
- **主力（YOLO26 s/m）**：部署路径最成熟，Orin TRT 管线已打通 — 即便备选轨道胜出，数据处理 / 部署流程可复用。
- **YOLOv13-s**：`HyperACE` / `DSC3k2` 自定义模块相对 YOLO26 为增量改动，作为"YOLO26 出 bug 时的快速替换"。与 YOLO26 同属 AGPL，替换不改变合规策略。
- **DEIM-D-FINE-S/M**：Apache-2.0 从根本上解决 AGPL 商用问题；同时 D-FINE 的 FDR（Fine-grained Distribution Refinement）回归头在小目标定位上有 paper 级优势，对交通灯这种 bbox 宽度 < 3% 的目标理论收益大。N 规格按 `research/surveys/alt_detector_architectures.md §六` 评估不具备数据性价比，R1 不训。

**评估口径统一**：三条轨道共用同一合并数据集 `data/merged/`（YOLO 格式 + COCO JSON 双出）、同一 80/20 分层切分、同一 7 类定义 — 所有指标可直接横向对比。YOLOv13 走独立 venv（`yolov13/.venv`），DEIM 走主 `uv` 的 `--extra deim` 可选组，均不污染主力训练环境。

**增强口径对齐**：三条轨道均禁用水平翻转（箭头方向语义反转），其余光学 / 几何增强保持各框架默认。DEIM 默认数据管线继承 `base/dataloader.yml` 包含 `RandomHorizontalFlip`，在 `deim_hgnetv2_{s,m}_traffic_light.yml` 中显式覆盖 `transforms.ops` 剔除（2026-04-23）；Mosaic / RandomPhotometricDistort / RandomZoomOut / RandomIoUCrop / mixup 保留，由 `stop_epoch` 在尾段关闭以避免稀疏类（`redRight` / `greenRight`）样本被破坏。YOLO26 已在 R1 初版以 `fliplr=0` 锁定，YOLOv13 沿用。

**R1 决策点**（三轨完成后执行）：
1. 若主力 YOLO26s/m 在部署域评估集上已达 R2 验收标准（mAP50 ≥ 0.60），直接进入 R2 数据侧工作，备选轨道降为监控组。
2. 若 YOLOv13-s 相对 YOLO26s **+≥ 3 pp mAP50**，切换主力为 YOLOv13-s（合规策略不变）。
3. 若 DEIM-D-FINE 相对 YOLO26 最佳 **+≥ 5 pp mAP50**，且 Orin 端 FP16 延迟 ≤ 50 ms/帧，启动主力切换 — 需提前验证 Orin JetPack 5.1 / TRT 8.5 上 `MultiscaleDeformableAttnPlugin_TRT` 可从 OSS 构建（风险项，见 survey §三）。
4. 若三者持平（差异 < 2 pp），按许可证成本排序：**DEIM > YOLOv13 > YOLO26**，优先 Apache-2.0。

---

## R1 跨架构综合分析（同域 val）

> 本节基于 `runs/detect/{yolo26n,yolo26s,yolo26m,yolo26l,yolov13s,deim_dfine_s,deim_dfine_m}-r1/` 与 `yolo26{n,s,m}-r1-1280/` 训练曲线、混淆矩阵与逐类 PR 表（详见 [`phase_2_round_1_results.md`](./phase_2_round_1_results.md)）。**所有结论限同域 BSTLD/S2TLD/LISA 横屏 dashcam val 集**；部署域评估集尚未建立。六个完成模型 + 三条 1280 训练对照齐备 — 详见 §DEIM-D-FINE 专项评估 与 §Demo 视频诊断 → 分辨率训练扫描。

### 整体指标对照

| 模型 | mAP50 | mAP50-95 | P | R | F1 | 训练轮次 | 角色 |
|------|-------|---------|----|---|----|---------|------|
| YOLO26n-r1 | 0.720 | 0.484 | 0.940 | 0.689 | 0.795 | 39 / 100 | 容量瓶颈；640 训练不可跨尺度推理 |
| **YOLO26s-r1** | 0.849 | 0.608 | 0.930 | 0.670 | 0.778 | 46 / 100 | **R1 现役部署模型**（Orin TRT 上线中） |
| YOLO26m-r1 | 0.869 / 0.858 † | 0.646 † | 0.934 | 0.712 | 0.808 | 72 / 100 | YOLO 上限基准；†best mAP50@21 / mAP50-95@52 |
| YOLO26l-r1 | 0.850 | 0.619 | 0.918 | 0.703 | 0.795 | 48 / 100 | 容量再上一档 mAP50 < m，不进入部署候选 |
| YOLOv13-s | 0.815 | 0.580 | 0.836 | **0.767** | 0.800 | 100 / 100 | 备选监控：召回 +9.8 pp / 精度 -9.4 pp vs YOLO26s |
| DEIM-D-FINE-S | 0.848 | 0.602 | 0.919 | 0.846 | 0.881 | 132 / 132 | Apache-2.0 备选；决策规则 4 触发 vs YOLO26s |
| **DEIM-D-FINE-M** | **0.859** | **0.613** | **0.955** | **0.843** | **0.896** | 102 / 102 (best=97) | **R1 主力候选**（待 Orin TRT plugin Gate）— F1 R1 最高 |
| DEIM-D-FINE-L | 0.857 | 0.611 | 0.953 | 0.847 | 0.897 | 102 / 102 (best=72) | DEIM 容量上限 — 同域 val 与 M 持平（-0.2 / -0.2 pp），不进入部署候选；保留为 KD A6 / A7 教师候选 |

> †YOLO26m-r1 的 `best.pt` 取自 Ultralytics fitness 综合最优（mAP50 在 ep21 达到 0.869）；若以 mAP50-95 为单一准绳，ep52 略胜（0.646）。两者部署影响差异 < 1 pp，沿用 ep21 best.pt。

### 关键观察

**1. YOLO26 容量拐点在 s↔n 之间，不在 s↔m / m↔l 之间。**
- n→s：mAP50 +12.9 pp / mAP50-95 +12.4 pp（**质变**）。
- s→m：+2.0 / +3.8 pp；m→l：mAP50 倒挂 -1.9 pp（YOLO26l 训练 9.51 h 但 best=ep28）。Orin 1280 FP16 上 m 推理时间约为 s 的 1.6× — ROI 不利。
- YOLO 内部仍以 **YOLO26s-r1** 为现役部署，**主力候选已升级为 DEIM-D-FINE-M**（见 §DEIM-D-FINE 专项评估），待 Orin Gate 验证后切换。

**2. YOLOv13-s vs YOLO26s 是"召回换精度"的非帕累托权衡。**
- YOLOv13-s 整体 mAP50 **低 3.4 pp**、mAP50-95 **低 2.8 pp** → **未触发决策规则 2 的主力切换条件**（要求 +3 pp）。
- 但 YOLOv13-s 的 Recall 反而 **+9.8 pp**（0.767 vs 0.670），代价是 Precision **-9.4 pp**（0.836 vs 0.930）。
- 反映在背景假阳上：YOLOv13-s 的 background→FP 各类率 13–17%（混淆矩阵 background 行），而 YOLO26s 仅 3–7%。
- **R1 决策**：YOLOv13-s 不替换主力，但保留为监控组 — 其高召回属性在 R2 部署域评估上若能保持，可在"宁可误报不漏检"的安全策略下变为有效备选。

**3. 右转箭头类（redRight / greenRight）的真实状况比表面指标更糟。**

| 模型 | redRight R | redRight 混淆矩阵主分布 | greenRight R | greenRight 混淆矩阵主分布 |
|------|-----------|----------------------|-------------|------------------------|
| YOLO26n-r1 | 0 | 67% → 漏检（background）；预测端无 redRight 输出 | 0 | 100% → 误判为 green |
| YOLO26s-r1 | 0.167 | 50% → 漏检；33% → 误判为 red | 0 | 100% → 误判为 green |
| YOLO26m-r1 | 0.333 | 67% → 漏检；33% → 误判为 red | 0 | 75% → 误判为 green；25% → 漏检 |
| YOLOv13-s | 0.333 | 50% → 漏检；33% 正确分类 | **0.655** | 25% 正确分类；25% → 误判为 greenLeft |

- YOLO26 系列的 greenRight `mAP50` 在 s/m 上分别为 **0.875 / 0.823**，**这是统计假象**：仅 4 个 GT 实例，少数高置信预测拉高 AP，但 R=0 表示模型在常用置信阈值下完全不报 greenRight 类。混淆矩阵证实"预测端 greenRight 行整行为空"。
- YOLOv13-s 是**唯一**在 greenRight 类有非零召回（R=0.655）且预测端非空的模型；归因于 HyperACE 的特征聚合更易于在长尾少样本类形成正样本激活。
- **结论**：R1 期间不能基于现有 right-arrow 指标做任何模型决策；样本量（19/13 实例）已低于统计可靠下限。R2 必须靠"反向 fliplr 合成 + 国内场景实采"补齐到每类 ≥ 500 实例，否则该类继续无意义。

**4. 训练效率：YOLOv13-s 显著更慢。**
- YOLO26s-r1：46 epochs × ~210 s/epoch ≈ **2.7 h**（早停于 ep26 best）。
- YOLOv13-s：100 epochs × ~340 s/epoch ≈ **9.4 h**（无早停，全程跑完，测试时间起伏来自 ep47 周围 close_mosaic + resume，参见 results.csv）。
- 同卡 4090，YOLOv13-s 单 epoch 慢 60% 主要源自 HyperACE 注意力路径；R2 升至 1280 训练后差距还会扩大。这把 YOLOv13-s 的"备选轨道"角色定调为"YOLO26 出 bug 时再启用"，平时不主用。

**5. 训练曲线形态：mosaic close 是关键拐点。** YOLO26 全系列在 ep2–10 跳到 0.85+ 平台后稳定渐进；YOLOv13-s 在 ep47 close_mosaic 触发 P↓R↑ 反转、ep83 二次反转 — R2 可考虑提前关 mosaic 挤出收益，并放宽 patience（YOLOv13 在 patience=20 下会过早停在 ep45）。

**6. DEIM-D-FINE 家族在两个右转箭头类同时取得有意义召回，且 DEIM-M 完成后家族整体上探。**
- redRight：DEIM-S R=0.500 / mAP50=0.505、**DEIM-M R=0.500 / mAP50=0.505**（mAP50-95 +2.9 pp 优于 S，0.316 vs 0.287，**n=6 噪声内，仅作记录**）— 仍显著高于 YOLO26m（R=0.333 / mAP50=0.409）和 YOLOv13-s（R=0.333 / mAP50=0.351）；DEIM 优于 YOLO 的**真正信号**是召回率的"非零 vs 零"二元差异（DEIM 给出预测、YOLO 在常用置信阈值下不报），AP magnitude 在 n=6 下高方差不可独立用作决策。
- greenRight：DEIM-S R=0.750 / mAP50=0.693、**DEIM-M R=0.750 / mAP50=0.754**（+6.1 pp mAP50 优于 S，**n=4 噪声内，仅作记录**）— YOLOv13-s 次优（R=0.655）；所有 YOLO26 系列 R=0（YOLO26s greenRight mAP50=0.875 是 R=0 下的统计假象，详见 §DEIM-D-FINE 专项评估 † 注）。同上，"非零 vs 零"是结构性差异，AP magnitude 不应独立用作决策。
- 警告：右转箭头 GT 仅 6 / 4 实例，远低于统计可信下限。指标置信度有限，但**形态差异（"完全不报" vs "偶尔正确"）+ DEIM-M vs DEIM-S 内部一致提升趋势是可读的**。
- 推断：DEIM 的 D-FINE 回归头（FDR / fine-grained distribution refinement）+ 多尺度可变形注意力对极少样本类的位置先验更鲁棒，是 Apache-2.0 商用优势之外的工程意义"长尾保险"。该结论在 R2 数据补齐到每类 ≥500 实例后必须复测；R1 期间不构成单一决策因子。

### 类别分布偏置的传导路径（混淆矩阵证据）

YOLO26 系列 + YOLOv13-s 一致表现出以下**结构性偏置**，源于训练集 redLeft >>> redRight（12,983 vs 19）的极端不平衡（DEIM-S/M 是例外，详见 §关键观察 #6）：

- `red` ↔ `redLeft`：YOLOv13-s 上 17% 的真 red 被误判为 redLeft（YOLO26 上仅 1–4%）— YOLOv13 更易"过度方向化"。
- `green` ↔ `greenRight`：所有 YOLO26 模型把 greenRight 的 75–100% 误判为 green — 类别先验"被吞并"，模型从未建立 greenRight 决策面。
- `redRight` 主漏检方向是 background（漏检），不是 redLeft / red 混淆 — 小目标 + 极少样本 + 低置信被 NMS 滤掉。

R2 修复路径：redLeft↔redRight 靠 fliplr 合成（R2 §数据侧 第 2 项 A 方案），greenRight 漏检必须实采（fliplr greenLeft 会引入分布偏差 — 部分 greenLeft 是非镜像模式）。

### DEIM-D-FINE 专项评估

#### 训练完成度

- **DEIM-S**: 132 / 132 epochs 完整训练（stage_2 = epochs 120–131，含 12 个 no-aug epoch）。**`best_stat` 在 epoch 131 创全局最高，最终 epoch 132 仅 ≤ 0.001 漂移** → 训练曲线接近饱和但未完全锁定 — 详见下文 §DEIM 收敛性核查。**部署用 checkpoint = `best_stg2.pth`**（epoch 131）；R1 关闭时已通过 `--test-only` 重跑校核（`logs/deim_eval_s/eval.pth`），AR@100=0.709 与 epoch-131 训练日志一致（旧报告表格 AR@100=0.736 为转录错误，已纠正）。
- **DEIM-M**: **102 / 102 epochs 完整训练**（2026-05-11 完成，总时长 1d 7:50:21；best epoch=97，最终一致性已由 `runs/detect/deim_dfine_m-r1/log.txt` 第 102 行 `test_coco_eval_bbox[0]=0.6131894…` 与 `logs/deim-d-fine_m.log` 末尾 `best_stat: {'epoch': 97, 'coco_eval_bbox': 0.6134163…}` 双重确认）。先前 R1 报告所述 ep40 DDP `unused parameter idx 358` 崩溃在 trainer 加 `find_unused_parameters=True` 包装后已解决，从 `last.pth` 续训完剩余 ~61 ep。**最终指标** mAP50=0.859 / mAP50-95=0.613 / P=0.951 / R=0.847 / F1=0.896 — DEIM 家族最优，亦为 R1 全部六个完成模型中 F1 最高者。
- **DEIM-L**: **102 / 102 epochs 完整训练**（2026-05-12 完成；best epoch=72；指标快照源 `logs/deim-d-fine_l.log` epoch-72 COCO 评估块 + `runs/detect/deim_dfine_l-r1/eval/best_coco_summary.json` + 远端 `--test-only` 重跑产物 `logs/deim_eval_l/eval.pth`）。**最终聚合指标** AP@0.5=0.857 / AP@0.5:0.95=0.611 / AP@0.75=0.694 / AP_small=0.526 / AR@100=0.726；per-class 见 [`phase_2_round_1_results.md`](./phase_2_round_1_results.md) §DEIM-D-FINEl。
  - **`best_stg2.pth` 缺省 — 按 DEIM 设计**：epoch-72（stage-1）全局最优 mAP=0.6113 在剩余 30 个 epoch（含 stage-2 / no-aug 末段 epochs 90–101）未被超越；`DEIM/engine/solver/det_solver.py:130-149` 仅在 `epoch >= stop_epoch` 时**且**产生**新的**全局最优时落盘 `best_stg2.pth`，否则不写。**`best_stg1.pth` 即可部署模型**（epoch-72 权重）。
  - **本地 checkpoint 缺口（说明而非错误）**：本地 `runs/detect/deim_dfine_l-r1/` 仅含部分 stage-1 编号 ckpt（3..31）+ `best_stg1.pth`；缺 `last.pth` 与 epoch ≥ 35 的编号 ckpt — 同源于 `det_solver.py:99` 在 `epoch < stop_epoch=90` 才写 `last.pth` 与编号 ckpt，加上远端 → 本地权重同步当前仍在进行（rsync `.checkpoint00XX.pth.XXXXXX` 部分传输文件可见）。`last.pth` 在 epoch-89 冻结，整 R1 决策不依赖该文件。
  - **per-class AP 已补齐（2026-05-12 R1 关闭后追加）**：通过 `scripts/eval/_remote_deim_eval.sh` 在远端 GPU 重跑 `DEIM/train.py --test-only -r best_stg1.pth`，由 `scripts/eval/_parse_deim_per_class.py` 从 `logs/deim_eval_l/eval.pth` 解析 7 类 P / R / AP@50 / AP@50:95；DEIM-S / DEIM-M 同步以**部署用 `best_stg2.pth`** 重跑统一口径（替换原 `eval/latest.pth` 末 epoch in-loop snapshot；详见 §产物清单 + 变更记录）。R1→R2 桥接消化项 C 完成。

#### DEIM 收敛性核查（inter-DEIM IoU 扫，2026-05-12 R1 关闭追加）

> **口径**：本表仅在 DEIM 家族内部横向比较 — 三模型共用 DEIM `CocoEvaluator` 同一评估流水（pycocotools 同语义，maxDets=100，eval_spatial_size=640²）。**与 YOLO 跨家族不可直接套用** — 跨家族 selection 仍仅以同域 val AP@0.5 / AP@0.5:0.95 为准（见 §整体指标对照；YOLO 走 Ultralytics `model.val()` max_det=300 + NMS-free head；DEIM 走 DETR set-prediction top-K，两者在 AP@0.5 / AP@0.5:0.95 上 pycocotools-aligned 但不输出 AP_small / AP_medium / AP_large）。AP_small / AP_medium / AP_large 跨家族补齐被列为 §R1→R2 桥接外 carry-forward（需 GPU 时段重跑 YOLO val 走 pycocotools-direct）。

**每模型 saved-checkpoint 指标（epoch = best_stat 选出 = AP@0.5:0.95 最大）**：

| 模型 | 保存 ckpt | best epoch | total ep | stage 划分 | AP@0.5 | AP@0.5:0.95 | AP@0.75 | AP_small | AP_medium | AP_large | AR@100 |
|------|-----------|-----------|----------|-----------|--------|-------------|---------|----------|-----------|----------|--------|
| DEIM-S | best_stg2.pth | **131 / 131** | 132 | stop_epoch=120；stage-2=120..131（12 ep no-aug） | 0.848 | 0.602 | 0.664 | 0.517 | 0.757 | 0.774 | 0.709 |
| DEIM-M | best_stg2.pth | **97 / 101**  | 102 | stop_epoch=90；stage-2=90..101 | 0.859 | 0.613 | 0.640 | 0.530 | 0.770 | 0.783 | 0.698 |
| DEIM-L | best_stg1.pth | **72 / 101**  | 102 | stop_epoch=90；stage-2=90..101 | 0.857 | 0.611 | 0.694 | 0.526 | 0.772 | 0.815 | 0.726 |

> AR@100 三模型均由 2026-05-12 R1 关闭后 `--test-only` 重跑 `logs/deim_eval_{s,m,l}/eval.pth` 提供；DEIM-S 修正旧报告 0.736 转录错误（实际 0.709，与训练日志 epoch-131 块完全一致），DEIM-M 补齐先前 "—" 占位。

**收敛性判定**：

| 模型 | best 后剩余 epoch | 训练曲线尾段判定 | 工程意义 |
|------|-------------------|------------------|----------|
| DEIM-S | **0**（best = 最终 epoch 131） | **未饱和** — AP@0.5:0.95 在最后一个 epoch 仍创新高（ep 127→131 提升 +0.001）；stage-2 EMA refresh + no-aug 仍带来收益 | 若下一轮（R2 / R3）再训 DEIM-S，建议 +20 epochs 验证拐点位置；R1 当前 ckpt 仍是该数据下可获得的 DEIM-S 上限 |
| DEIM-M | 4（best=97，final=101） | **已收敛** — 最后 4 epoch 平台无新高 | best_stg2 信号可靠 |
| DEIM-L | 29（best=72，final=101） | **饱和早**（stage-1 内已锁定最佳）— stage-2 30 epoch + no-aug 末段全部无新高 | 数据约束达顶（DEIM-L vs DEIM-M 同域 val 持平 ±0.2 pp）→ R1 7-class data 对 DEIM 家族的有效容量上限位于 DEIM-M 容量等级；DEIM-L 不进入部署，作 KD 教师 |

**AP@0.5 极值检验（IoU=0.5 是否需独立挑 ckpt？）**：每个 DEIM 模型用其全部 epoch 的 AP@0.5 做扫描，找到 max AP@0.5 epoch，对比该 epoch 的 AP@0.5:0.95：

- DEIM-S：max AP@0.5 = 0.848 在 ep 131（与 best_stat 同 epoch），其余前几名 ep 127=0.843、ep 119=0.844 — best_stat 选 ep 131 与 AP@0.5 峰值一致，**无需调整**。
- DEIM-M：max AP@0.5 = 0.860 在 ep 81 / 92 / 94（AP@0.5:0.95 分别 0.611 / 0.604 / 0.605），best_stat ep 97 AP@0.5 = 0.859。**Δ AP@0.5 = +0.001，AP@0.5:0.95 -0.002~0.009**，整体净亏 — 保持 ep 97（best_stg2）**无需调整**。
- DEIM-L：max AP@0.5 = 0.859 在 ep 67（AP@0.5:0.95 = 0.611），best_stat ep 72 AP@0.5 = 0.857。**Δ AP@0.5 = +0.002，AP@0.5:0.95 持平**，但 ep 67 不是 best_stat 选中的 ckpt；`checkpoint0067.pth` 按 `checkpoint_freq=4` 落盘（67+1=68=4·17），即理论上可用。**实际差距在 ±0.002 噪声水平 — 保持 ep 72（best_stg1）作为 deployable，无需调整**。后续若有针对 R1 数据的 IoU=0.5 单一精度 deploy 场景，可重 eval 一次 ep 67 ckpt 做 sanity check（成本 ~1 GPU 分钟）。

**结论（R1 关闭时点）**：

1. **DEIM-S / DEIM-M / DEIM-L 当前 saved checkpoint 报告的 AP@0.5 / AP@0.5:0.95 与训练日志全 epoch 扫描结果一致**（均在 ±0.002 内），**不需调整 IoU 阈值或重新选 ckpt**。
2. **DEIM-S 未完全饱和** — R1 该数据集下"训长一点"可能再加 +0.5~1 pp，但不影响 R1 关闭时的决策（DEIM-M / DEIM-S 均通过决策规则 4）。R2 / R3 数据冻结后 DEIM-S 再训时需注意延长 epochs 上限。
3. **DEIM-L 与 DEIM-M 在 R1 数据下持平** → R1 7 类 + 同域 val 对 DEIM 家族的有效容量上限位于 DEIM-M 级别；继续放大模型（L→X）在 R1 数据上不会有 mAP 边际收益。R2 数据扩充后该结论需复核（**carry-forward** — 不属于 R1 决策范围）。
4. **YOLO 侧 four 模型 (n/s/m/l) 均收敛**（`results.csv` tail 趋势：早停的 n=ep39/best=15、s=ep46/best=26、m=ep72/best=21、l=ep48/best=28；patience=20 在 s/m/l 上正常触发）— 无 DEIM-S 式"最后 epoch 仍创新高"情形。

#### 与主力候选 YOLO26s 直接对比（同 val、同切分、同 7 类）

| 类别 | YOLO26s R | DEIM-S R | ΔR | YOLO26s mAP50 | DEIM-S mAP50 | ΔmAP50 |
|------|-----------|----------|----|---------------|--------------|--------|
| red        | 0.949 | 0.960 | +1.1 pp | 0.973 | 0.971 | -0.2 pp |
| yellow     | 0.848 | 0.880 | +3.2 pp | 0.904 | 0.914 | +1.0 pp |
| green      | 0.927 | 0.960 | +3.3 pp | 0.964 | 0.966 | +0.2 pp |
| redLeft    | 0.937 | 0.960 | +2.3 pp | 0.965 | 0.975 | +1.0 pp |
| greenLeft  | 0.858 | 0.910 | +5.2 pp | 0.931 | 0.914 | -1.7 pp |
| **redRight**   | 0.167 | **0.500** | **+33.3 pp** | 0.333 | **0.505** | **+17.2 pp** |
| **greenRight** | 0.000 | **0.750** | **+75.0 pp** | 0.875 † | 0.693 | -18.2 pp † |
| **overall（等权）** | 0.669 | **0.846** | **+17.7 pp** | 0.849 | 0.848 | **-0.06 pp** |

> † YOLO26s greenRight mAP50=0.875 是统计假象 — R=0 表示常用置信阈值下完全不报 greenRight；仅少数高置信预测拉高 AP。DEIM-S R=0.750 + mAP50=0.693 是**真实可工作**的 4 实例预测。

**核心读法（DEIM-S）**：在 7 类中**全部 R 不低于 YOLO26s**（最大领先 +75 pp，最小 +1.1 pp），整体 mAP50 / mAP50-95 均在 ±1 pp 内，Pareto 前沿上至少持平且严格更优。

#### 与上限基准 YOLO26m 直接对比（DEIM-M sister 表）

| 类别 | YOLO26m R | DEIM-M R | ΔR | YOLO26m mAP50 | DEIM-M mAP50 | ΔmAP50 |
|------|-----------|----------|----|---------------|--------------|--------|
| red        | 0.973 | 0.960 | -1.3 pp | 0.981 | 0.971 | -1.0 pp |
| yellow     | 0.907 | 0.890 | -1.7 pp | 0.928 | 0.922 | -0.6 pp |
| green      | 0.969 | 0.950 | -1.9 pp | 0.973 | 0.967 | -0.6 pp |
| redLeft    | 0.965 | 0.970 | +0.5 pp | 0.975 | 0.975 |  0.0 pp |
| greenLeft  | 0.945 | 0.910 | -3.5 pp | 0.930 | 0.918 | -1.2 pp |
| **redRight**   | 0.333 | **0.500** | **+16.7 pp** | 0.409 | **0.505** | **+9.6 pp** |
| **greenRight** | 0.000 | **0.750** | **+75.0 pp** | 0.823 † | **0.754** | -6.9 pp † |
| **overall（等权）** | 0.728 | **0.847** | **+11.9 pp** | 0.860 | 0.859 | -0.1 pp |

> † YOLO26m greenRight 0.823 同样是 R=0 统计假象；DEIM-M 在 R=0.750 下达到 0.754 是真实可工作的指标。

**核心读法（DEIM-M）**：YOLO26m 在 5 个"主流类"上 R 略胜 1–3 pp，但 DEIM-M 在两个长尾右转类拉开决定性优势（+16.7 / +75.0 pp R）；overall mAP50 持平、F1 +8.8 pp、Precision +1.7 pp。整体上 DEIM-M 是更工程友好的 Pareto 点。

#### 决策规则触发判断

DEIM-M 完成后两个候选都进入决策窗口（DEIM-S 同尺寸对 YOLO26s、DEIM-M 同尺寸对 YOLO26m）：

- **规则 3**（DEIM ≥ 最佳 YOLO + 5 pp mAP50 + Orin FP16 ≤ 50 ms/帧）：**未触发**。最佳 DEIM (DEIM-M, mAP50=0.859) vs 最佳 YOLO (YOLO26m, mAP50=0.869) = **-1.0 pp**，未达 +5 pp 阈值。
- **规则 4**（差异 < 2 pp 视为持平 → 按许可证优先级 DEIM > YOLOv13 > YOLO26）：**两条对位均触发**。
  - DEIM-S vs YOLO26s：mAP50 差 -0.06 pp、mAP50-95 差 -0.65 pp，两项均在 ±2 pp 内。
  - DEIM-M vs YOLO26m：mAP50 差 -1.04 pp、mAP50-95 差 -2.7/-3.3 pp（†以 ep21 best 或 ep52 best 取数）；mAP50 在 ±2 pp 内、mAP50-95 略超 — 但 Recall +13.5 pp（0.847 vs 0.712）、F1 +8.8 pp（0.896 vs 0.808）、Precision +1.7 pp（0.951 vs 0.934），综合优于 YOLO26m。许可证优先级 + 综合指标 → **DEIM-M 上位为主力候选**，DEIM-S 作为更轻量化的备选保留（≤50 ms Orin 预算下若 DEIM-M 延迟超标可降级）。
- **DEIM-L 不进入主力候选**：相对 DEIM-M 同域 val mAP50 -0.2 pp / mAP50-95 -0.2 pp（**未触发**任何升级规则），但参数量与延迟显著上升 — 不满足"许可证持平时取较小尺寸"的工程惯例。DEIM-L 的 R1 价值在于**容量探顶证据**（DEIM 家族在 R1 7 类同域 val 上的有效容量上限约 ~0.61 mAP@0.5:0.95 — DEIM-M 与 DEIM-L 持平 → 上限受限于数据而非模型）+ **KD 教师资格**（A6 cross-arch / A7 same-family — 见 [`../planning/additional_components_plan.md`](../planning/additional_components_plan.md) §七）。

#### 主力切换前的 Orin 验证 Gate（必须先全部通过）

DEIM-S / DEIM-M 在 R1 期间**不替换部署模型**（YOLO26s @ 1280 FP16 引擎仍是产线模型）。切换的前置条件（**DEIM-M 与 DEIM-S 各自独立通过**，以延迟达标的最大尺寸优先）：

- [ ] OSS `MultiscaleDeformableAttnPlugin_TRT` 在 Jetson AGX Orin / JetPack 5.1.2 / TRT 8.5.2 上构建成功（含 D-FINE 自定义算子链：可变形注意力 + 分布回归头）。详见 `research/surveys/alt_detector_architectures.md §三` 风险评估
- [ ] DEIM-{S,M} ONNX 导出 + `trtexec --fp16` 引擎构建成功
- [ ] 1280×1280 FP16 在 Orin 实测延迟 ≤ 50 ms / 帧（与 YOLO26s 同等预算）
- [ ] Demo 视检（与 YOLO26s 同样本同方法）：demo8 警示三角假阳是否同样发生 / demo10 横向龙门是否仍漏检 / demo15 小目标是否仍抖动

四项全部通过后，**优先以 DEIM-M 替换 YOLO26s 成为部署主力**；若 DEIM-M 在 Orin 上延迟超标则降级至 DEIM-S；两者均未达标则维持 YOLO26s。

---

## Demo 视频实际表现观察

> R1 期间在 `demo/demo{1..15}.mp4`（共 15 段，含横屏 dashcam 与竖屏手机视频）上分别用 yolo26{n,s,m}-r1 的 `best.pt` 做了三种分辨率（640 / 1280 / 1536）的离线推理，结果存放于 `demo/yolo26{n,s,m}-r1/{best,best_1280,best_1536}/`。

### 推理覆盖矩阵

| 模型 | 640 | 1280 | 1536 | 总计 demo 段 |
|------|-----|------|------|------|
| yolo26n-r1 | 15/15 | 15/15 | 15/15 | 45 |
| yolo26s-r1 | 15/15 | 15/15 + 5(/best-1280 增量集) | 15/15 | 50 |
| yolo26m-r1 | 15/15 | 15/15 | 15/15 | 45 |

`yolo26s-r1/best-1280/` 含 demo1–5 的早期增量产物（diagnosis 阶段），与 `best_1280/` 全量集互不冲突。

### 视检方法

2026-04-26 以 ffmpeg 抽帧对 `yolo26s-r1/best_1280/demo{1..15}.mp4` 做了 9 帧均匀采样（3×3 montage）+ 4 段稳定性突击（demo4/10/12/15 各取 8 帧 0.25s 步长，共 2 秒滑窗），对照原始 demo 帧人工辨识 bbox 位置 / 类别。难场景（demo8/10/11/13）追加 yolo26m-r1 同时间戳 montage 做横向对比。所有 montage 缓存在 `/tmp/demo_inspect/`，本节结论均来自该次视检，分三个维度记录。

### 漏检（miss detection）

| Demo | 场景 | 漏检表现 | s vs m |
|------|------|----------|--------|
| demo1 | 远距离路口 | 远端小目标常常未被框出 | 两者均漏 |
| demo3 | 卡车遮挡 | 灯被前车短暂遮挡时立即丢失，不复现到再出现 | 两者均漏 |
| demo9 | 黄昏 + 龙门信号 | 低光下龙门灯几乎完全无检测 | 两者均漏 |
| demo11 | 强逆光（太阳直射） | 中段画面（远 / 中距离）龙门灯丢失，仅近距离能检出 | 两者均漏，但 m 在最后两帧能同时框出红绿 |
| demo13 | 卡车列队 + 龙门 | 大量帧龙门灯未被框出（侧向角度，灯朝向偏） | 两者均严重漏，m 略好但仍不及格 |
| demo14 | 港口排队 | 远端龙门信号几乎不被检出 | 两者均漏 |
| demo10 | 工业区横向龙门 | s 仅检出靠近的少数灯 | **m 显著优于 s**：每帧多检 1–3 个龙门灯 |

漏检的共同规律：**远距离 / 小目标 + 弱光 / 逆光 + 非正面朝向**三类条件叠加时，模型几乎无能为力。这与 §R1 跨架构综合分析中的"640 训练分辨率难以覆盖竖屏 / 远距小灯"诊断一致。

### 误检（wrong detection）

分两种：

**(a) 灯被框对但分类错误**
- demo10：龙门交通灯在多个帧中被一致地标为 `green`，但目测灯形偏暗（疑似不发光或为红，难以从画质判断）。s / m 均做出相同判定 — 这是模型层面的偏置，非偶发抖动。
- demo15：路口闸控指示灯 / 检查站指示灯被识别为交通灯类别（红或绿）。属于"语义边界外但视觉接近"的误判，本数据集未明确将此类样本纳入负样本，模型行为符合预期。

**(b) 非交通灯目标被误判为交通灯**（背景误报，FP）
- demo8（s-r1）：黄色三角警示牌被多帧识别为 `yellow`；卡车驾驶舱的黄灯也被吃成 `yellow`；绿色厂房墙面也曾出现 `green` 框 — **这是 s-r1 在该 demo 上最严重的失稳来源**。
- demo8（m-r1）：上述大部分错误大幅减少，但仍能在中段画面看到龙门桁架顶部出现 `green` 小框（背景结构误激活）。
- demo11：太阳光斑 / 警示牌区域偶发 `yellow` / 红框（视检不能 100% 确定属于警示牌还是真实灯，但与混淆矩阵中 `yellow` 行 background→FP 较高的趋势一致）。

总体上 m 的背景假阳显著少于 s，对应同域 val P 从 0.93 → 0.88，但 R 和 mAP50 都升高 — m 的容量主要花在"减少背景误激活"而非"提升远端召回"。

### 检测稳定性（同一物体的逐帧一致性）

抽样了 4 段 8 帧 / 0.25s 步长的突击：

| Demo | 场景 | 稳定性 |
|------|------|--------|
| demo4 | 近距离单体绿灯，画质干净 | **8/8 帧均稳定检出**，类别一致，无闪烁。理想场景下行为正常。 |
| demo10 | 中距离龙门绿灯（疑似类别误判） | 8/8 帧持续输出同一 `green` 框，**类别错误也是稳定的**（不是抖动） |
| demo12 | 卡车从前方驶离，露出龙门红灯 | 第 1 帧被卡车遮挡 → 无检；第 2–8 帧灯进入视野后立即出现红框并稳定保留。**遮挡转可见的恢复时间 ≤ 1 帧**。 |
| demo15 | 远端 / 小目标闸控灯 | **明显闪烁**：第 1, 3, 4 帧出现绿框，第 2, 5–8 帧无 — 在置信度阈值附近边缘震荡。 |

分类：
- 单体大目标（demo4） → 稳定。
- 单体中目标且有正确特征（demo12 红灯靠近过程） → 稳定。
- 持续误分类（demo10） → "稳定地错误"，反而比抖动更难自动识别为问题。
- 小目标 / 弱信号（demo15） → 抖动严重；Plan A（tracker + EMA 投票）在分类边缘场景已落地缓解，若 baseline 仍漏检（小目标 recall 低）则按 [`../planning/temporal_optimization_plan.md`](../planning/temporal_optimization_plan.md) §1 上 detector-level TSM。

### 视检结论（对部署 / R2 的指导）

1. **现役部署仍是 YOLO26s，主力候选已升级为 DEIM-M（待 Orin Gate）**：m 在 demo10 横向龙门有可见提升但其他场景与 s 接近、demo8 背景误检亦未根除，YOLO 内部 +2 pp mAP50 / +1.5× 时延的 ROI 不利。DEIM-M 在指标层全面追平 / 长尾领先 YOLO26m（见 §DEIM-D-FINE 专项评估）— 主力切换条件是 Orin Gate 通过。
2. **R2 需补的训练样本**：远距离龙门信号 / 黄昏 + 弱光 / 逆光强光斑 / 卡车密集队列 + 龙门 — 这些场景在当前训练集中覆盖不足，是漏检的主因。
3. **R2 时序聚合双轨**（详见 [`../planning/temporal_optimization_plan.md`](../planning/temporal_optimization_plan.md)）：(a) 分类抖动 → Plan A（tracker + EMA，已落地）→ HMM / GRU；(b) 漏检（小目标 / 遮挡） → detector-level TSM（推荐）。两路径互不阻塞，按部署域评估集结果分别启动。
4. **demo8 假阳的根因在数据**：警示三角 / 厂房绿墙这类硬负样本若未在训练集出现，模型当然学不到拒识。R2 主动从 demo 中挖出这些误报帧补回训练集做硬负学习。
5. **备选轨道 demo 评估必要性**：
   - **YOLOv13-s — 不必需**。决策规则 2 未触发 (-3.4 pp mAP50)，其唯一独特价值 greenRight R=0.655 已被 DEIM 家族超越。
   - **DEIM-D-FINE-S/M — 必需**，但被 Orin Gate 阻断。两步走：(a) GPU 服务器 torch 端 demo（不依赖 TRT plugin，立即可做）；(b) Orin TRT plugin 构建后端到端引擎 demo 验延迟一致性。DEIM-M 优先（主力候选），DEIM-S 作为延迟降级备选同步跑。

### 视检材料路径

- s-r1 montages: `/tmp/demo_inspect/montages/s1280_demo{1..15}.jpg`（3×3 9 帧采样）
- m-r1 难场景对照: `/tmp/demo_inspect/montages/m1280_demo{8,10,11,13}.jpg`
- 稳定性突击: `/tmp/demo_inspect/stability/s_demo{4,10,12,15}_burst.jpg`（8 帧 0.25s 步长）

> 这些产物为临时缓存，R2 启动时应将关键示例（尤其是 demo8 假阳帧 / demo13 漏检帧）提取为 `data/r2_hard_examples/` 永久样本，并标注后回灌训练集。

### 演示覆盖现状（demo coverage gaps）

R1 期间 demo 视检由 PM 手工完成，结论已落入上文 §Demo 视频实际表现观察 + §视检结论。但 `demo/_review/ledger.json` 自动账目从未建立 — 全部 480 个 `demo/<run>/<engine>/*.mp4` 在 D1 ledger 中状态为 `never`。每次报告 Write 均通过 `.gate_override` sentinel 绕过 demo-coverage 门控，绕过本身不引入 demo 内容回归（更新的是训练指标 + 决策，与 demo 输出正交）。

**R2 启动时必须补齐**：

- [ ] PM 手工视检结论补录至 ledger（D1 demo-reviewer 一次性追扫 yolo26{n,s,m}-r1 的 best / best_1280 / best_1536 + DEIM-S/M 当前 best_stg2 / best_stg2_fp32）
- [ ] DEIM-{S,M} Orin TRT plugin 通过后，端到端 demo 由 D1 自动入账
- [ ] YOLOv13-s 现有 demo 输出若不再投入 R2 直接清理

---

## 部署链路验证

R1 首次完成端到端部署链路：训练 → ONNX 导出 → Head 裁剪（`scripts/export/strip_yolo26_head.py`）→ Orin 上 `trtexec --fp16` 构建 → C++ 推理。`inference/cpp/build/tl_demo` 在 Orin 上成功运行，1280 / 1536 两种分辨率引擎均已验证。

**部署端 latency（2026-04-21 Orin 实测，yolo26s-r1，FP16）**：
- `imgsz=1280` → ~25 ms/帧（~39 FPS）
- `imgsz=1536` → ~28 ms/帧（~34.5 FPS）

均远低于 50 ms 预算上限。部署和引擎构建的全流程见 [`../integration/trt_deployment.md`](../integration/trt_deployment.md)。

---

## Demo 视频诊断（总结）

R1 初版 C++ 推理在竖屏 `demo/demo.mp4` 出现"多数帧无检测 + 偶有大框"，定位到两个并发 bug 已全部修复：(1) 导出 ONNX 时漏传 `--imgsz` 导致引擎固化 640、C++ 管线静默回退（`trt_pipeline.cpp:296-300`）— 重新导出 1280 / 1536 引擎；(2) postprocess 误把 4 通道当 `cxcywh`，实际为 `xyxy` — 修复 `inference/cpp/src/trt_pipeline.cpp` 与 `inference/trt_pipeline.py`，引擎无需重建。修复后 demo 输出与 `.pt` 原生一致。

### 分辨率训练扫描（同域 val）

**A. 640 训练模型，1280 推理（YOLO26s-r1）**：

```
imgsz=640:  mAP50=0.8497  mAP50-95=0.6073
imgsz=1280: mAP50=0.7023  mAP50-95=0.4359
```

模型在 640 训练，stride / anchor 先验锁定 640；1280 推理时小目标先验错位，mAP50 下降 14.7 pp。

**B. 1280 独立训练对照**（`runs/detect/yolo26{n,s,m}-r1-1280/`，同切分同数据）：

| 变体 | 训练 imgsz | mAP50 | mAP50-95 | P | R | 训练轮次 | vs 640 训练同模型 |
|------|-----------|-------|----------|----|---|---------|--------------------|
| YOLO26n-r1-1280 | 1280 | **0.785** | 0.561 | 0.936 | 0.705 | 100 / 100 | +6.5 / +7.7 pp（容量受益） |
| YOLO26s-r1-1280 | 1280 | 0.805 | 0.576 | 0.934 | 0.646 | 52 / 100 | **-4.4** / -3.2 pp（同域 val 反降） |
| YOLO26m-r1-1280 | 1280 | 0.747 | 0.526 | 0.788 | 0.643 | 54 / 100 | **-11.3** / **-11.7** pp |

**跨尺度推理能力**（demo 实测）：YOLO26n-640 在 1280 / 1536 demo 上 0 detection；YOLO26s-640 在三个分辨率均能检测，1536 conf 最高 (0.58)。n-1280 训练后跨尺度先验对齐 — 上表 +6.5 pp 与 demo 跨尺度失败同源（n 容量不足以同时覆盖训练与推理两个尺度），s 不存在此问题。

**结论**：
1. 提高部署 `imgsz` 对同域数据反向、对 OOD demo 正向 — R1 的 1280 / 1536 引擎是 OOD hotfix。
2. **R2 不能照搬"1280 训练 + 同域 val 验收"** — 否则会得出"YOLO26m 1280 差 11 pp"的误导性结论。R2 验收**必须绑定部署域评估集**（竖屏 / 手机 / 国内路况），同域 val 仅作训练收敛信号。
3. R1 现役部署沿用 **YOLO26s-640 训练 + 1280 / 1536 引擎**；YOLO26n 不部署；1280-trained YOLO26n 作为延迟极限受限的回退方案。

### 剩余 R1 数据缺陷（不会被部署修补）

- `redRight` / `greenRight` 样本各 10–20 条，模型实际未学到。
- 验证集全部是 BSTLD 横屏 dashcam，**未覆盖部署域**（竖屏 / 手机 / 国内路况）。
- 训练标签 >80% 宽度 < 3%（imgsz=640 下 <20 px）→ 位置先验集中在上右四分之一（dashcam 倾向）。

以上三项由 R2（`imgsz=1280` 训练 + 部署域采集 + 类别补齐）解决。

---

## 第二轮（R2）计划

R2 目标：**关闭部署域与训练域的差距 + 扩展类别（联合模型）**。

### 部署侧（R1 已完成，供 R2 接力）

- [x] 1280 / 1536 引擎在 Orin 上构建与验证
- [x] xyxy postprocess 修复
- [ ] 部署域评估集：从实际部署摄像头采集 100–300 帧，标注（含 R2 最终类别），作为独立 `data/eval_deployment/`。所有后续对比以该集为准，不再只看同域 val。

### 数据侧（R2 前必须完成）

1. **方向重标注**（延续 P1 行动计划第 5 项）：
   - BSTLD 测试集 8,334 张 → 方向重标注（优先级：高）
   - S2TLD `normal_2` 4,168 张 → 方向重标注
   - S2TLD `normal_1` 796 张 → 方向重标注
2. **右转箭头数据补齐**：
   - 方案 A（短期）：利用 `fliplr` 反向增强将 `redLeft`/`greenLeft` 合成为 `redRight`/`greenRight`，图像与类别映射同时翻转。
   - 方案 B（长期）：国内道路场景实采包含右转箭头的真实数据。
3. **评估"去除极稀疏类"**：若 R2 前无法把 `redRight`/`greenRight` 各自补至 >500 条，考虑合并为 `rightArrow` 单一类或暂时移出训练目标。

### 训练侧

4. **`imgsz=1280` 训练**（非 640）：与部署对齐，减轻小目标信息损失。预估训练时间 3–4×。
5. **`patience=40`**：小样本类别收敛更慢。
6. **类别加权 / 过采样**：对 `yellow`、`greenLeft` 过采样 2–3×。
7. **域适应性增强**：启用 `degrees=5.0`、`perspective=0.0005`；自定义增强管线加入 JPEG 压缩 / motion blur。**不开启** `fliplr`（箭头方向会与标签不一致）。
8. **主力模型**：分两路并行 — (a) **DEIM-D-FINE-M**（R1 主力候选，Orin Gate 通过后即切换部署）+ DEIM-D-FINE-S（延迟降级备选）；(b) **YOLO26s**（R1 现役 fallback）+ YOLO26m（上限基准）。YOLO26n / l / YOLOv13-s 在 R2 暂缓。

### 验收标准（R2 退出条件）

- [ ] 部署域评估集上 mAP50 ≥ 0.60（含所有 R2 类别整体）
- [ ] 黄灯召回率 ≥ 0.70（不依赖过采样后的高 precision 陷阱）
- [ ] 方向类（redLeft / greenLeft）mAP50 ≥ 0.55
- [ ] Demo 视频上主观检测率（人工抽查）≥ 80%，无明显过大框误报
- [ ] Orin 端 1280 × 1280 引擎 FP16 延迟 ≤ 50 ms / 帧

---

## R2 范围扩展（PM 确认事项）

2026-04-21 PM 会议新增两项诉求。两者均为"加类别"类型的扩展，与现有 R2 计划可在**同一次训练**中完成 — 不会额外引入轮次，但改变数据准备的优先级和规模。

### 扩展一：栏杆 / 道闸检测（分层需求）

| 层级 | 类别 | 触发条件 |
|------|------|---------|
| **MVP（最低要求）** | **`barrier`**（单类：检测栏杆本体，不分状态）| 必达 |
| **最佳实践（可选）** | **`armOn`**（抬起 / 放行）+ **`armOff`**（放下 / 阻挡）| 两种状态**各自**达到质量阈值（≥500 实例且覆盖足够场景多样性） |

- **架构可行性**：`nc` 从 7 提到 8（MVP）或 9（best-practice），仅 head 输出通道差异。可**先按 MVP 标注**，若采集后两态都达标再于标注阶段批量子标签（CVAT / LabelMe 支持）。
- **数据可行性**：S2TLD / BSTLD / LISA **均不含栏杆**，必须从零采集。
  - 首选：**部署现场拍摄**（与部署域评估集合并采集，一次出差覆盖多件事）。
  - 备选：Roboflow Universe（"boom gate" / "railway barrier" / "parking barrier"）— 商业许可须逐条核对。
- **数据预算**：MVP ≥500 实例（目标 2–5K）；双态模式每类 ≥500 且数量比例差不超 3×。
- **类别定义必须先敲定再标注**：SOP 需明确"损坏 / 折断栏杆"、"部分可见"、"地锁 / 升降柱（非横杆式）" 的归类。模糊定义会在 R3 细分时暴雷。

### 扩展二：交通灯类别扩展（已确认 9 类，最多 12 类）

- **已确认新增 2 类**：`forwardGreen`（直行绿灯）、`forwardRed`（直行红灯）— 国内红绿灯配置中"直行箭头"与"满盘灯"分属独立灯位。
- **交通灯类别最低锁定为 9 类**（原 7 + forwardGreen + forwardRed）。
- **最多扩展到 12 类**（再加 3 类）— 具体清单待 PM 按部署场景观测结果出。候选方向：
  - 行人信号：`pedRed` / `pedGreen` / `pedYellow`
  - 方向类补全：`yellowLeft` / `yellowRight` / `forwardYellow`
  - 黄闪警示、熄灯、故障状态

**`forwardRed` / `forwardGreen` 数据来源**：BSTLD / LISA / S2TLD 的直行箭头样本**可能已存在**但在 R1 被折叠为 `red` / `green`。R2 前用脚本复扫 + 人工复核，不足部分与栏杆同批实采。

### 合并策略：一个 10–14 类模型，而非多模型

| 组 | 最小 | 最大 |
|----|------|------|
| 交通灯 | 9（已确认）| 12 |
| 栏杆 | 1（单类 MVP）| 2（armOn + armOff）|
| **总 `nc`** | **10** | **14** |

理由：
1. **共享主干的正则效应**：联合训练通常比两个独立模型更鲁棒；栏杆与交通灯语义差异大，互相干扰低。
2. **部署成本**：Orin 单引擎 1280 分辨率现 ~25 ms / 帧，延迟预算 50 ms；拆两模型 → ~50 ms，逼近上限。
3. **维护成本**：一份 ONNX → 一份引擎 → 一套后处理 → 一份 class map。

**唯一例外**：若栏杆与交通灯在不同摄像头 / 不同频率下工作，则拆分更合理。此点须 PM 确认部署拓扑。

### 代码与配置变更预清单（待类别锁定后一次性修改）

| 文件 | 变更 |
|------|------|
| `data/traffic_light.yaml` | `nc` 和 `names` |
| `scripts/export/strip_yolo26_head.py` | 调用处 `--num-classes` |
| `inference/cpp/include/trt_pipeline.hpp` | `kClassNames` 数组及 `std::array<.., N>` 的 N |
| `inference/cpp/src/demo.cpp` | `kColors` BGR 调色板 |
| `inference/trt_pipeline.py` | `CLASS_NAMES` dict |
| `configs/*.yaml` | 硬编码类别引用（如有） |

R2 启动前只需 PM 提供**最终类别清单**，以上 6 处改动 <0.5 天可完成。

### 对 R2 数据计划的优先级重排

在 §数据侧原有项基础上追加两项并与现有项共享出差机会：

1. **现场采集专项**（合并三目标）：部署域评估集 + `armOn` / `armOff` 训练数据 + PM 清单中的新交通灯类别。
2. **PM 清单锁定**（阻塞项）：数据采集开始**之前**出一页 SOP，锁定最终类别清单 + 每类视觉定义 + 边界情况处理。SOP 缺失是 R1 重标注返工的第一原因。

---

## 产物清单（R1）

| 产物 | 路径 | 说明 |
|------|------|------|
| 训练权重 | `runs/detect/yolo26{n,s,m}-r1/weights/best.pt` | n / s / m 变体最佳 |
| ONNX（640）| `runs/detect/yolo26{n,s}-r1/weights/stripped.onnx` | 640 导出 + 裁头 |
| ONNX（1280 / 1536）| `runs/detect/yolo26{n,s}-r1/weights/stripped_{1280,1536}.onnx` | 高分辨率导出 + 裁头（部署选 s） |
| TRT 引擎 | `runs/detect/yolo26{n,s}-r1/weights/best*.engine` | Orin 端构建 |
| Demo 回放 | `runs/diagnose/{n,s}-pt-{640,1280,1536}/demo.mp4`、`demo/s-r1-{1280,1536}.mp4` | 诊断用；批量扫描脚本 `scripts/run_demos.sh` |
| 训练指标 | `runs/detect/yolo26{n,s,m}-r1/results.csv` | 逐轮记录 |
| 跟踪库 | `inference/tracker/*.py`、`inference/cpp/{include,src}/tracker.{hpp,cpp}` | Python + C++ 两端；fixtures 驱动的 parity 单测 |
| DEIM-D-FINE-S 训练权重 | `runs/detect/deim_dfine_s-r1/best_stg2.pth` | 132 / 132 ep 完整训练 |
| DEIM-D-FINE-M 训练权重 | `runs/detect/deim_dfine_m-r1/best_stg2.pth` | **102 / 102 ep（best=97）完整训练** |
| DEIM-D-FINE-L 训练权重 | `runs/detect/deim_dfine_l-r1/best_stg1.pth` | **102 / 102 ep（best=72）完整训练**；`best_stg2.pth` 按 DEIM 设计缺省（见 §DEIM-D-FINE 专项评估 → DEIM-L） |
| DEIM-L COCO eval 快照 | `runs/detect/deim_dfine_l-r1/eval/best_coco_summary.{json,txt}` | epoch-72 COCO 12-tuple 聚合 + 说明 |
| DEIM 统一口径 per-class eval（S/M/L） | `logs/deim_eval_{s,m,l}/{eval.pth,per_class.json,per_class.txt,test_only.log}`（runtime / 大文件，**gitignored 在 `logs/`**，需 `scripts/eval/_remote_deim_eval.sh` 重跑或 rsync 复现）| 2026-05-12 R1 关闭后用 `scripts/eval/_remote_deim_eval.sh` + `scripts/eval/_parse_deim_per_class.py` 在远端 GPU `--test-only` 部署 ckpt（S/M=`best_stg2.pth`，L=`best_stg1.pth`）；统一替换 S / M 原 `eval/latest.pth`（末 epoch in-loop snapshot）口径 |
| DEIM per-class eval 持久审计副本（tracked） | `docs/reports/r1_evidence/deim_eval_{s,m,l}_per_class.json` | 与上述 logs/ per_class.json 同源同内容；tracked 副本用于在 `logs/` 重置后仍能审计 [`phase_2_round_1_results.md`](./phase_2_round_1_results.md) §DEIM-D-FINE 表 |
| DEIM 方法学等价性 diff 审计（tracked）| `docs/reports/r1_evidence/deim_eval_old_vs_new_diff.json` + 同目录 reproducer `_deim_eval_diff_audit.py` | 由其同目录的一次性 reproducer 生成；记录旧 `eval/latest.pth` vs 新 `--test-only on best_stg2.pth` 的 32 字段 old / new / Δ（4 aggregate + 7 类 × 4 metrics）；S max|Δ|=0.0000，M max|Δ|=0.0100 |
| YOLO26l-r1 训练权重 | `runs/detect/yolo26l-r1/weights/best.pt` | 48 / 100 ep（best=28），9.51 h，容量上限基准 |
| YOLO26{n,s,m}-r1-1280 训练权重 | `runs/detect/yolo26{n,s,m}-r1-1280/weights/best.pt` | imgsz=1280 独立训练实验，同域 val 与 640 训练对比见 §1280 训练 vs 640 训练 |
| DEIM 训练日志 | `runs/detect/deim_dfine_{s,m}-r1/log.txt` | 逐轮 COCO 12 项 JSONL |
| DEIM 评估快照（历史） | `runs/detect/deim_dfine_{s,m}-r1/eval/latest.pth` | faster_coco_eval 序列化训练 in-loop 末 epoch snapshot；2026-05-12 后被统一口径产物（前述 `logs/deim_eval_*/`）取代作为 per-class P/R 表来源 |
| 部署文档 | [`../integration/trt_deployment.md`](../integration/trt_deployment.md)、[`../integration/tracker.md`](../integration/tracker.md) | |

---

## 变更记录

| 日期 | 变更 |
|------|------|
| 2026-04-21 | 初版：YOLO26n/s-r1 完成、Orin TRT 部署链路打通、demo 诊断、R2 计划 |
| 2026-04-21 | TRT postprocess `xyxy` 解码修复（误解为 cxcywh 致 67% 面积大框）；同域 val 分辨率扫描揭示 OOD demo 与同域精度反向关系 |
| 2026-04-21 | Orin 延迟实测 s-r1 @ 1280=~25 ms / @ 1536=~28 ms（远低于 50 ms 预算）|
| 2026-04-21 | R2 范围扩展（PM）：交通灯 9–12 类（+forwardGreen/Red）、栏杆 1–2 类（MVP barrier / 最佳 armOn+armOff），联合 `nc` 10–14 |
| 2026-04-22 | YOLO26m-r1 完成：mAP50=0.869 / mAP50-95=0.635；相对 s 仅 +2 pp，部署仍首选 s |
| 2026-04-22 | R1 备选轨道立项：YOLOv13-s + DEIM-D-FINE-S/M（依据 `research/surveys/alt_detector_architectures.md`）|
| 2026-04-23 | DEIM 训练配置修正：覆盖 `transforms.ops` 剔除继承的 `RandomHorizontalFlip` 防 left/right 语义反转 |
| 2026-04-26 | YOLOv13-s 完成 100/100 ep：mAP50=0.815，未达决策规则 2 阈值；保留为监控组（greenRight R=0.655 是亮点）。新增 §R1 跨架构综合分析 与 §Demo 视频实际表现观察 |
| 2026-04-26 | Demo 人工视检（s/m-r1）：漏检主因 远距 + 弱光 + 非正面；demo10 横向龙门 m 优于 s；demo8 假阳 m 缓解未根除；部署仍选 s。关键样本应在 R2 启动时提取至 `data/r2_hard_examples/` |
| 2026-05-05 | DEIM-D-FINE-S 完成 132/132 ep：mAP50=0.848，长尾 redRight R=0.500 / greenRight R=0.750 显著领先；**决策规则 4 触发**，待 Orin TRT plugin Gate。DEIM-M 在 ep40 触发 DDP `unused parameter idx 358` 崩溃 |
| 2026-05-11 | **DEIM-D-FINE-M 完成** 102/102 ep（best=97，1d 7:50:21；DDP 修复 `find_unused_parameters=True` 后 resume）：mAP50=0.859 / F1=0.896 — DEIM 家族最优、R1 F1 最高；**主力候选升级为 DEIM-M**，DEIM-S 转为延迟降级备选。**YOLO26l-r1 完成** 48/100 ep（best=28，9.51 h）：mAP50=0.850 < m，不进入部署候选。**1280 独立训练实验** YOLO26{n,s,m}-r1-1280：n +6.5 pp、s/m 同域 val 反降 -4.4 / -11.3 pp，R2 验收须绑定部署域评估集 |
| 2026-05-11 | 报告 v2 trim：删除冗余 §指标汇总；合并 §同域 val 分辨率扫描 / §1280 训练 / §模型容量观察 → §分辨率训练扫描；压缩 §变更记录；§整体指标对照 与 §视检结论 对齐 DEIM-M 主力候选决策；新增 DEIM-M vs YOLO26m sister 对照表 |
| **2026-05-12** | **R1 关闭** — DEIM-D-FINE-L 完成 102/102 ep（best=72）：mAP50=0.857 / mAP50-95=0.611，与 DEIM-M 同域 val 持平（±0.2 pp）→ **不进入部署候选**，保留为 KD A6/A7 教师候选。`best_stg2.pth` 缺省按 DEIM 设计（stage-2 未超过 stage-1 全局最优）。R1 七模型齐备；后续仅追加 carry-forward 完成证据，不再调整决策。新增 §R1→R2 桥接（预 R2 可执行消化项）|
| — | **carry-forward**：DEIM-{S,M,L} Orin TRT plugin 构建 + FP16 延迟实测（DEIM-M 优先）— Stage 2 范畴，见 [`../planning/pre_deploy_AGV_integration.md`](../planning/pre_deploy_AGV_integration.md) |
| — | **carry-forward**：部署域评估集建立后的基准指标 — R2 manifest 冻结后启动 |
| — | **carry-forward**：YOLO 家族 AP_small / AP_medium / AP_large 跨家族 parity（与 DEIM CocoEvaluator 对齐）— 触发于 R2 manifest 冻结后 GPU 时段；输出 `runs/detect/yolo26{n,s,m,l}-r1/eval/coco_compat.json`（pycocotools-direct，maxDets=100）+ 新增对照表回插 §DEIM 收敛性核查 footnote；当前 R1 决策不依赖跨家族 small/medium/large 比较（仅 AP@0.5 / AP@0.5:0.95 即 pycocotools-aligned 足够）|
| **2026-05-12** | **DEIM per-class AP 统一口径完成（消化项 C 关闭）** — 通过 `scripts/eval/_remote_deim_eval.sh` 在远端 GPU 重跑 DEIM-S / DEIM-M (`best_stg2.pth`) + DEIM-L (`best_stg1.pth`) 的 `--test-only`，由 `scripts/eval/_parse_deim_per_class.py` 解析 `logs/deim_eval_{s,m,l}/eval.pth` 生成 7 类 P/R/AP@50/AP@50:95；统一替换原 S/M `eval/latest.pth` 末 epoch in-loop snapshot 口径。**最终决策不变**：DEIM-M 仍为 DEIM 家族部署候选；DEIM-L 与 DEIM-M 同域持平（M=0.613 / L=0.611），保留为 KD 教师。修正 §收敛性核查 表 DEIM-S AR@100 转录错误（0.736 → 0.709）+ 补齐 DEIM-M AR@100=0.698。详见 [`phase_2_round_1_results.md`](./phase_2_round_1_results.md) §DEIM-D-FINE{s,m,l} 更新后表 + footnote |

---

## R1→R2 桥接（预 R2 可执行消化项）

R1 已关闭，R2 manifest 冻结仍在数据侧推进（hardware lock 已 2026-05-12 完成，见 [`../planning/additional_components_plan.md`](../planning/additional_components_plan.md) §八 + [`../data/r2_data_collection_sop.md`](../data/r2_data_collection_sop.md)）。本节列出**在 R2 数据 freeze 之前**即可推进的消化项 — 全部跑在 R1 现有数据 / scaffold 上，输出可直接接入 R2 训练或 Stage 2 部署调优。

### 处理原则

- **不依赖 R2 数据**：用 `data/merged/`（R1 7 类）或现有 demo 视频；R2 训练流程开启后这些 a/b/c-stage 产物保持向后兼容。
- **不改变 R2 决策规则**：所有 ablation 结果是"R2 manifest freeze 后用真实数据再跑一遍"的预演与脚手架验证，不锁定 R2 入选 / 落选。
- **AGREED scaffold 优先**：已通过 B2 + codex-review-conflictor 的 a-stage scaffold 直接进入 c-stage runtime；未通过的不在本清单中。

### 消化项清单

| ID | 消化项 | 输入 / 数据 | 输出 / 产物 | 状态门 | 阻塞 | 参考 |
|----|--------|------------|------------|--------|------|------|
| **A** | KD A1+A2a+A2b rehearsal 复盘 + A6 path γ 设计 spike | R1 `data/merged/` + DEIM-M 当前 best_stg2 教师 | `runs/rehearsal_kd_A{1,2a,2b}_R1.json` + `runs/rehearsal_kd_A6_design_spike.json` 二次验证 | KD AGREED，runners 已落地 | 无 — 直接可重跑 | memory: `project_kd_rehearsal_scaffolding` + `project_kd_upgrade_review` |
| **B** | KD A7 same-family DEIM-L → DEIM-M / DEIM-S 教师试运行 | DEIM-L best_stg1.pth 教师 + `data/merged/` | A7 cell matrix R1-data spike（不锁 R2 决策）| AGREED scaffolding 在 `components/knowledge_distillation/` | DEIM-L 远端权重同步完成（用户已确认进行中）| `components/knowledge_distillation/integration/deim_kd_launch.py` |
| ~~**C**~~ | ~~DEIM-L per-class AP 提取~~ | DEIM-L `best_stg1.pth` | `logs/deim_eval_l/{eval.pth,per_class.json,per_class.txt}` + [`phase_2_round_1_results.md`](./phase_2_round_1_results.md) §DEIM-D-FINEl 表格已填齐；DEIM-S / M 同步重跑统一口径 | **✅ 已完成 2026-05-12** | — | `scripts/eval/_remote_deim_eval.sh` + `scripts/eval/_parse_deim_per_class.py` |
| **D** | Copy-paste + class-balance β-sweep（3-arm，R1 7 类）| `data/merged/` + `--copy-paste` / `--cls-weight` 默认 OFF 旁路 | `runs/copy_paste_balance/no_aug` / `cp_only` / `cp_balanced` × YOLO26s | c-stage AGREED-CLEAN（iter-11） | 训练机器 GPU | memory: `project_copy_paste_balance_scaffold` — `components/copy_paste_balance/` |
| **E** | Hard-negative mining FP harvest（2-arm，R1）| Demo 视频假阳帧 + `data/merged/` | `runs/hard_negative_mining/{no_hn, with_hn}` × YOLO26s | a-stage AGREED-CLEAN（iter-2）；data-prep-time 流程 | 一次性 FP 流水 + 训练机器 GPU | memory: `project_hard_negative_mining_scaffold` — `components/hard_negative_mining/` |
| **F** | TSM scaffold 激活 tripwire 与因果性校验 | `components/temporal_shift_module/` + 合成时序 chunk | tripwire schema v1.1 实测命中率 + DEIM 适配脚手段 | scaffolding AGREED iter-6 | 无 — 单机 CPU 可跑 | memory: `project_tsm_scaffold` — `scripts/_tsm_activation_schema.json` |
| **G** | `_r2_decide_precision.py` + `_r2_verify.py` b-stage 合成 fixtures | a-stage scaffold + JSON schemas | b-stage 实现 + `scripts/_r2_schemas_test.py` 扩展 | 决策规则 v1 锁定（见 active plan）+ B2/C3 loop | 无 — 单机 CPU 可跑 | active plan §1.0 §2.1.1 |
| **H** | `export_yolo.sh` + `export_deim.sh` sidecar 离线校验 | 现有 R1 ONNX / 引擎 | sidecar 解析 + `engine_sha256` 校验脚本 | Engine Sidecar Contract 已落地 | 无 — 单机可跑（dry-run）；Orin 端真延迟为 Stage 2 范畴 | memory: `project_engine_sidecar_contract` |
| **I** | SAHI c0 grid precheck（R1 demo 视频）| Cam-W spec + 现有 demo 视频近似分辨率 | c0 grid 候选 + 缩放策略 dry-run | §六 SAHI per-camera adaptive AGREED | 无 — 单机 CPU 可跑（不需 GPU 推理）| [`additional_components_plan.md §六`](../planning/additional_components_plan.md) |

### 排序与触发条件

- **立刻可跑（无 GPU 依赖）**：F, G, H, I — 可在本地开发机推进。
- **等待 GPU 时段**：A, D, E — 训练机器空闲窗口插队（每项预计 < 1 GPU-day）。
- **等待权重同步**：B — 仅 DEIM-L 远端 → 本地权重同步完成后启动。
- **已完成**：C — DEIM per-class AP 统一口径完成（2026-05-12，见 §变更记录）。

任何消化项**不允许**修改 R2 决策规则、active plan 已锁条款、或 R2 训练侧 `_r2_train_config.json` 入参。如发现需要调整，按 `precision_parity_plan.md` 的 reopen criteria 走标准 reopen 流程。

### 与 R2 关系

- 消化项 **A / B** → 进入 R2 §七 KD cell matrix 当作 R1-data 预演证据，**不**作为 R2 KD 实测结果。
- 消化项 **C**（已完成 2026-05-12）→ DEIM 家族 7 类 P/R/AP per-class 表（`logs/deim_eval_{s,m,l}/`）作为 R2 §七 KD A6/A7 cell matrix 教师 / 学生 per-class baseline reference，不再列为待办。
- 消化项 **D / E** → 进入 R2 §三 copy-paste / §四 hard-negative 的 R2-data 实测前的"管线就绪 + 入参合理性"证据。
- 消化项 **F / G / H** → 直接 carry into R2 round 工具链（TSM / decide / verify / sidecar），不再为 R2 独立验证。
- 消化项 **I** → 进入 §六 SAHI 的 R2 c0 grid 候选短名单。

---

## 决策口径补丁（2026-05-12 R1 关闭后追加）

> **注**：本节为 R1 关闭后追加的口径说明，**不修改任何已落地决策与结论**。来源 codex-report-conflictor 3-pass 评审遗留的 process-polish 项（用户授权 R1 关闭后追加补丁，见 commit 7bbf752 之后的 follow-up）。

### 决策规则 4 的 tie metric 口径

R1 决策规则 4（同质性时按 license 优先级择优）的 tie 判定**以 mAP50 为准**——基于以下源文本推断（非原文显式）：`docs/planning/development_plan.md §决策规则` 的 R1 三轨规则 1 / 2 / 3 全部以 mAP50 为阈值（"mAP50 ≥ 0.60" / "+≥ 3 pp mAP50" / "+≥ 5 pp mAP50"），规则 4 文本仅写 "三者差距 < 2 pp"未显式指名 metric；按规则 1-3 上下文一致性推断，规则 4 的"差距"沿用 mAP50 作为 tie metric。

DEIM-M vs YOLO26m 在 mAP50 差为 -1.0 pp（0.859 vs 0.869，**在 ±2 pp 内**），触发 tie；按 license 优先级 DEIM (Apache-2.0) > YOLO26 (AGPL-3.0)，DEIM-M 上位为主力候选。

mAP50-95 在 R1 决策规则中为**辅助参考指标**（差为 -2.7 / -3.3 pp，超出 2 pp），其作用是当 tie metric 触发 license 偏好后，复核选中模型是否在更严格的 IoU 范围下显著退化。R1 评估时该差距被以下 3 项证据补偿：
1. Recall +13.5 pp（0.847 vs 0.712）
2. F1 +8.8 pp（0.896 vs 0.808）
3. 长尾召回（redRight / greenRight）"非零 vs 零"的结构性差异（DEIM 给出预测，YOLO26 在常用置信阈值下不报）

后续 R2 / R3 round 如需修正 tie metric 口径（改用 mAP50-95 或两者 AND-clause），按 `precision_parity_plan.md` reopen criteria 流程，**不允许**在 R1 关闭后追溯修改本轮决策。

### redRight / greenRight 复测触发条件

R1 redRight n=6 / greenRight n=4 远低于 precision parity plan `support ≥ 30` 统计基线，单类 AP / P / R 在该 support 下为高方差噪声占主导（详见 `phase_2_round_1_results.md` §DEIM-D-FINEl 稀有类样本量警示 footnote）。**正式复测触发**：

- **触发条件**：R2 manifest 冻结后，新 val 集 redRight + greenRight **任一类**的实例数（in val） ≥ 30，**或** R2 close gate 校验时长尾类样本达成 SOP 目标支撑度（见 [`docs/data/r2_data_collection_sop.md`](../data/r2_data_collection_sop.md) §R2 训练 imgsz 决策规则 与 §决策规则）。
- **复测内容**：在新支撑度下重跑 `scripts/eval/_remote_deim_eval.sh` + `scripts/eval/_parse_deim_per_class.py`；输出 `runs/detect/<r2_run>/eval/per_class_AP.json`；将新 P/R/AP 与 R1 同表对照（R1 数字保持原值不变作为 baseline）。
- **目标**：将长尾类指标从"记录用"升级为"决策用"，验证 DEIM-M 长尾召回优势在足够支撑度下是否保留 / 反转。
- **负责人 / 时机**：R2 round B0（主训练）后 7 天内由检测组完成；进度跟踪进 R2 round 报告 §变更记录。

### Demo 覆盖率门（demo coverage gate）现状说明

R1 关闭与本补丁追加期间，对 `docs/reports/phase_2_round_1_*.md` 的多次 Write/Edit 操作经由 `.claude/hooks/check_demo_coverage.py` 触发 demo-reviewer ledger gate；当前 ledger 状态：

- `demo/_review/ledger.json` 内容为空（`{"version":1,"updated_at":"","entries":{}}`，从未有任何条目落盘）。`.claude/hooks/check_demo_coverage.py` 在每次 Write/Edit `docs/reports/phase_*.md` 时**实时扫描** `demo/<run>/<engine>/demo*.mp4` 文件系统并与（空）ledger 比对，**hook 当前报告 ~480 个 `(run × engine)` group `never` 状态**（覆盖 R1 全部 10 个 run 的 best / best-1280 / best_1280 / best_1536 引擎 × 上下游 tracker 模式）；这是 hook 计算的 coverage 缺口，**不是 ledger 文件存储的条目**。
- 本 R1 关闭 + 后续 polish 编辑均通过 **`demo/_review/.gate_override`** 一次性 sentinel 旁路（每次 Edit 消费，每次重置；参考 `CLAUDE.md` 项目段 "Demo coverage hook"）。
- 跨越 e1c5dc5 → 7bbf752 → 本补丁 commit 的所有编辑均为**数值表格 / 方法学 footnote / banner / 跨引用修复 / 口径补丁文本**，**不依赖**也**不修改** demo 视频证据。

**碳排放（process debt）**：demo-reviewer ledger 大规模 backfill 列入 **R2 retrospective parking lot**（不阻塞 R2 启动）：

1. R2 部分启动时（demo 重跑后），ledger 中 R1 的 `never` 条目应同步刷新为 `current`（如 demo 视频也重新生成）或保持 `never` 但标注"R1 archived; 不再 backfill"（按 R2 demo-reviewer SOP 走标准 PR）。
2. 若 R2 round 报告（`phase_2_round_2_report.md`）写入前 ledger 状态未改善，应在该报告 §coverage-gaps 节显式列出延期 group 与延期原因（CLAUDE.md 已要求此项）。
3. 不允许的反模式：长期使用 `.gate_override` 旁路而不补完 ledger — sentinel 设计为"临时单次"用途。


