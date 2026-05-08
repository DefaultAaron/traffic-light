# 知识蒸馏（Knowledge Distillation, KD）流水线设计 — 最终模型生产环节

**状态**：v1.2（2026-05-08，PM 五项答复 + R1 决策点 + 4090 D 容量分析全部融入；conflictor-loop AGREED）。

**适用范围**：
- **学生候选**：YOLO26-s（R1 主力 baseline）+ DEIM-D-FINE-S（主力候选，pending Orin TRT plugin 验证）。
  - YOLOv13-s 不在候选：R1 决策规则 2 未触发（−3.4 pp mAP50）；保留为监控组应对 YOLO26 系统性失败。
  - DEIM-D-FINE-M 暂缓：R1 训练 ep40 因 DDP `unused parameter` 崩溃（rank 0 idx 358），41/102 ep 不构成可信评估；trainer 修复 + resume 完成后再评。
- **教师候选**：M（默认）+ L（条件性，按 §2.4 capacity 分析）；同/跨家族均可。X/XL **不纳入**（§2.4 4090 D 容量分析排除）。
- **部署目标**：Jetson Orin 64GB，TRT 8.5 / JetPack 5.1，FP16 或 FP32 引擎。
- **训练资源**：单 4090 D 工作站（24 GB VRAM；用户独占，无抢占）+ 独立 DEIM venv。

---

## 一、背景与触发条件

`development_plan.md` §R2 训练增强 第 133 行已有一行条目：

> **知识蒸馏 M→S**（条件性） — DEIM-M / YOLO26-m 作为 teacher，KL on logits + 浅层 feature L2；R2 训练同步进行，不延后主线。

本文档将该一行条目展开为完整的 KD 流水线设计，并显式回答 PM 提出的三个核心问题：

1. **Q1**：教师是否应扩展至更大 (L/XL)？
2. **Q2**：是否需要多教师，还是单教师足够？
3. **Q3**：是否做跨架构迁移（DEIM ↔ YOLO）？

每个问题给出 cited 证据 + 决策建议；问题之外补全完整流水线设计。

---

## 二、Q1：教师规模 — 是否应扩展至 L/XL?

### 2.1 已知证据

| 来源 | 标签 | 关键发现 |
|---|---|---|
| Cho & Hariharan, 2019, ICCV, *On the Efficacy of Knowledge Distillation* | [E] | "更大模型不必然是更好教师"；小学生**无法模仿**过大教师；TAKD 类多步顺序在他们的实验中**也无效**；缓解：早停教师（ESKD） |
| Mirzadeh et al., 2020, AAAI, *Improved KD via Teacher Assistant* (TAKD) | [E] | 中等规模 TA 显著缓解大跨度 capacity gap；与 Cho 2019 形成证据张力 |
| Cao et al., 2022, NeurIPS, *PKD* | [E] | MaskRCNN-Swin 教师 → ResNet-50 RetinaNet/FCOS 学生：+4.1pp / +4.8pp mAP（异构 + 容量大跨度同时成立） |
| Cao et al., 2023, ICML, *MTPD* | [E] | Transformer 教师 → CNN 学生：ResNet-50 RetinaNet 36.5 → 42.0 AP（+5.5pp） |
| Peng et al., 2025, ICLR Spotlight, *D-FINE* | [E] | D-FINE 已内置 GO-LSD（Global Optimal Localization Self-Distillation）；DEIM-D-FINE-S 学生本身已通过自蒸馏吸收浅层定位知识 |
| 项目内部观察 (R1) | [project-prior, R1 single-run] | R1 单跑中 YOLO26-s vs YOLO26-m 在 R1 数据集上 mAP@0.5 差距约 +2pp；DEIM-S vs DEIM-M 量级相近。R1 数据集已退役、非部署域、未做种子重复 — 只作项目侧附议，不作通用结论 |

### 2.2 张力分析

经典分类领域（Cho 2019）的"大教师有害"在检测领域被 PKD/MTPD **部分推翻**：Swin/Transformer 大教师可以稳定提升小 CNN 学生，前提是有 Pearson 归一化 / 渐进式 / 投影 MLP 这类**桥**。换言之"capacity gap 阻止 KD"不是普适命题，而是"没有桥的 capacity gap 阻止 KD"。

[I] 项目特定风险（数据 moderate + 算力受限）：
- (a) 过拟合本地背景 / 安装姿态 / 类先验，传递脆弱伪决策面到 S 学生；
- (b) 大教师训练成本本身吃掉 KD 实验矩阵的 GPU 时间预算。

### 2.3 决策建议

**主路径**：M 教师为默认（YOLO26-m / DEIM-D-FINE-M）。

**矩阵补充**：保留 cell A7 走 **L 教师**（容量分析见 §2.4 排除 X/XL），触发条件任一：
- (a) A2–A5 cells 完成后稀有安全类（forwardRed / forwardGreen / barrier 双态 / 行人灯）AP 未达 R3 部署门槛；
- (b) 4090 D 24 GB VRAM 余量允许且训练时间预算足够。

L 教师候选：**YOLO26-l (~32M) / DEIM-D-FINE-L (~31M)**。无论 (a) 或 (b) 触发，应叠加 capacity-gap 缓解：TAKD 中介 / ESKD 早停教师 checkpoint / 投影 MLP **三选二同时启用**。

**不纳入**：以 L/XL 直接对接 S 学生且无任何缓解措施 — 证据上属于 Cho 2019 已警示的高风险设计。

### 2.4 4090 D 容量分析（决定 L vs X/XL 的取舍）

**硬件约束**：4090 D（NVIDIA RTX 4090 D，中国市场版本）— 24 GB GDDR6X VRAM；CUDA 核 14592（标准 4090 为 16384，约少 10%）；FP32 ~73 TFLOPS（标准 4090 ~83）；同显存带宽。**算力较标准 4090 低 ~12%，VRAM 完全相同。**

**KD 训练时 VRAM 估算**（imgsz=1280，FP32 mixed-precision，AdamW，单卡）：

| 组合（教师 no_grad + 学生反向 + KD 损失 + projection） | bs | 估算 VRAM | 4090 D 24 GB headroom | 评估 |
|---|---|---|---|---|
| YOLO-s 学生（A1 baseline，无 KD） | 8 | ~6–10 GB | ~14–18 GB | ✓ |
| YOLO-m 教师 → YOLO-s 学生（A2a / A3 / A4 / A5） | 8 | ~10–14 GB | ~10–14 GB | ✓ 充裕 |
| **YOLO-l 教师 → YOLO-s 学生（A7-L）** | 8 | **~12–17 GB** | **~7–12 GB** | **✓ 可行** |
| YOLO-x 教师 → YOLO-s 学生 | 8 | ~17–23 GB | ~1–7 GB | ⚠ 不足 2 GB headroom |
| YOLO-x 教师 → YOLO-s 学生 | 4 | ~12–16 GB | ~8–12 GB | ⚠ 但 wall-clock 翻倍 |
| DEIM-S 学生（A1 baseline，无 KD；deformable attention 内存重） | 4 | ~5–7 GB | ~17–19 GB | ✓ |
| DEIM-M 教师 → DEIM-S 学生（A2b / A3 / A4 / A5） | 4 | ~9–13 GB | ~11–15 GB | ✓ |
| **DEIM-L 教师 → DEIM-S 学生（A7-L）** | 4 | **~12–16 GB** | **~8–12 GB** | **✓ 可行** |
| DEIM-X 教师 → DEIM-S 学生 | 4 | ~14–20 GB | ~4–10 GB | ⚠ 紧张，deformable attention OOM 风险 |
| DEIM-X 教师 → DEIM-S 学生 | 2 | ~11–15 GB | ~9–13 GB | ⚠ 但 wall-clock 4× |

**教师 fine-tune（前置成本，全反向图）**：
- YOLO26-l fine-tune bs=8 imgsz=1280：~16–20 GB ✓
- YOLO26-x fine-tune bs=4 imgsz=1280：~18–22 GB ⚠ headroom ≤ 6 GB
- DEIM-D-FINE-L fine-tune bs=4 imgsz=1280：~12–16 GB ✓
- DEIM-D-FINE-X fine-tune bs=2 imgsz=1280：~16–22 GB ⚠ wall-clock 极慢

**结论**：
1. **L 教师可行**（YOLO26-l / DEIM-D-FINE-L）— 主流 bs（8 / 4 imgsz=1280）下均留 ≥ 7 GB headroom；fine-tune 也 ✓。
2. **X/XL 教师不纳入** — 主流 bs 下 headroom < 2 GB（YOLO-x）或会 OOM（DEIM-X）；降 bs 解决 OOM 但 wall-clock 翻倍至 4×，违反 §6#4 训练成本预算 (< 2× scratch)；fine-tune 也只能 bs=2，不实用。
3. **A7 cell scope 锁定为 L tier 唯一**（YOLO26-l 或 DEIM-D-FINE-L，按学生家族选取）。原 v1.1 的"X/XL 待 capacity 实测核定"作废；§2.4 容量分析即为该实测的替代（基于参数量 + 文献 backbone activation 估算）。
4. **OOM 兜底**：训练首 epoch 实测 VRAM 峰值 > 22 GB → 自动降 bs（YOLO 8→6，DEIM 4→3）；仍 > 22 GB → A7 cell 降级为 M 教师 cell，记录 capacity 失败。

---

## 三、Q2：单教师 vs 多教师?

### 3.1 已知证据

| 来源 | 标签 | 关键发现 |
|---|---|---|
| Cao et al., 2023, ICML, MTPD | [E, detection-specific] | 顺序渐进式多教师显著优于平均/同步多教师；约束："所有教师须检测同类" |
| Liu, Yueying et al., 2020, *AMTML-KD*, **Neurocomputing 415: 106–113** | [E, classification-only] | 每实例自适应教师权重 — 证据是分类，不作检测器多教师 KD 的支撑 |
| Furlanello et al., 2018, ICML, *Born-Again Networks* | [E, classification-primary] | 同架构师生连续 KD 可超越教师；最低代价"多教师"形式 |
| 项目计算预算 | [I] | 4 教师同步前向 = 4× forward + cache，4090 D 24 GB 紧张 |

### 3.2 决策建议

- **baseline**：单教师 KD（A2a / A2b / A3 / A4 cells）。
- **高级 cell（exploratory）**：渐进式 2-教师序列（cell A5），按 MTPD 范式：
  - 教师 1 = 同家族 M（稳定特征）；教师 2 = 互补家族 M；
  - 触发：A4 已通过 §6 全部 5 项验收门。
- **不纳入**：同步 4 教师训练（成本 × 4，跨 venv 开销，Cao 2023 同类约束限制收益）。
- **约束**：多教师须类集合完全一致；R2 nc 范围 10–14 锁定后，所有候选教师须在同一类清单上 fine-tune 后方可作为教师。

---

## 四、Q3：跨架构 KD — DEIM ↔ YOLO?

### 4.1 已知证据

| 来源 | 标签 | 关键发现 |
|---|---|---|
| Cao et al., 2022, NeurIPS, PKD | [E] | Pearson 相关 + 通道归一化使方法架构无关；明确支持异构 backbone |
| Wang et al., 2024, CVPR, *CrossKD* | [E] | 学生特征送入教师 head，回避 GT vs 教师监督冲突；GFL R-50 +3.5pp；异构 backbone 可行（head 兼容性是前提） |
| Yang et al., 2022, ECCV, *MGD* | [E] | 学生特征掩码 → 生成式恢复教师特征；架构无关 |
| Yang et al., 2022, CVPR, *FGD* | [E] | 前/背景掩码 + 全局关系；feature-level，依赖共同空间网格 |
| Chang et al., 2023, ICCV, *DETRDistill* | [E] | DETR 家族专用：仅适合 DEIM↔DEIM |
| Wang et al., 2024, CVPR, *KD-DETR* | [E] | DETR 专用，consistent distillation points；同上仅 DEIM↔DEIM |

### 4.2 关键限制

[I] DEIM 是 DETR-style set-prediction（无 NMS、无 anchor、decoder query）；YOLO 是 dense head + 网格 + C++ NMS。直接 DETR decoder query → YOLO grid 在结构上**不存在自然对齐**：
- (a) Hungarian 匹配 vs grid 标签分配冲突；
- (b) 100–300 query vs 几千 anchor 的稀疏密度不匹配；
- (c) C++ pipeline 的 letterbox xyxy 解码假设与 DETR cxcywh + sigmoid 输出不兼容。

跨架构 KD **走架构无关 feature-level 通道（PKD / MGD / FGD）**；head/decoder/query 直接对齐**原则上不纳入**。

### 4.3 决策建议（cell A6 实施规格）

- **方向**：teacher = 互补家族 M（YOLO26 学生时 = DEIM-D-FINE-M；DEIM 学生时 = YOLO26-m）；student = R2 选型胜者 (S 级)。
- **特征层级**：三层 FPN-style feature pyramid：
  - YOLO 学生：neck P3/P4/P5（stride 8/16/32）；
  - DEIM 学生：encoder 多尺度对应 stride 8/16/32 reshape 后 spatial map；
  - 二者 stride 维度自然对齐（imgsz=1280 → 160/80/40）。
- **Spatial resample**：±1 偏移时 bilinear 对齐到学生维度。
- **通道投影**：学生侧 1×1 conv MLP（学生 channels → 教师 channels）；KD 损失在投影后空间计算。**导出 TRT 时移除该 projection**（仅训练辅助）。
- **损失方法**：首选 PKD（Pearson 通道相关）；备选 MGD（masked feature regeneration，慢 20–30%）；FGD foreground mask 跨架构未充分验证，不引入。
- **掩码策略**：A6 不引入 fg/bg 掩码；纯 spatial+channel feature mimic。

仅 DEIM↔DEIM cells（A2b / A3 当 R2 选 DEIM 学生时）才使用 DETRDistill / KD-DETR。

---

## 五、综合流水线设计

### 5.1 实验矩阵 — pre-committed cells

| Cell | 学生 | 教师 | 方法栈 | 触发条件 | 优先级 | 适用学生家族 |
|---|---|---|---|---|---|---|
| **A0** | DEIM-D-FINE-S | — | GO-LSD off, no external KD | always — DEIM 路径 baseline 净分离对照 | P0 | DEIM only |
| **A1** | 选型胜者 | — | scratch (DEIM 路径 GO-LSD on) | always — control + KD-acceptance gate 的 wall-clock 锚点 | P0 | 全部 |
| **A2a** | YOLO26-s | YOLO26-m | cls-logit KL only（YOLO26 已移除 DFL，LD 不适用） | always；R2 in-round 即可启动 | P0 | YOLO 家族 |
| **A2b** | DEIM-D-FINE-S | DEIM-D-FINE-M | LD on FDR + cls-logit KL | always；R2 in-round 即可启动 | P0 | DEIM 家族 |
| **A3** | 选型胜者 (S) | 同家族 M | PKD feature-level | always | P0 | 全部 |
| **A4** | 选型胜者 (S) | 同家族 M | A2a/A2b + A3 组合 | max(A2a/A2b, A3) **mAP@0.5:0.95 lower-CI > A1 point** AND 无安全类 AP delta < −0.5pp | P1 | 全部 |
| **A5** | 选型胜者 (S) | 同家族 M → 互补家族 M | MTPD 渐进 2-教师 | A4 通过 §6 全部 5 项验收门 | P2 | 全部 |
| **A6** | 选型胜者 (S) | 互补家族 M | PKD 跨架构（投影 MLP + spatial resample，§4.3 规格） | A4 通过 §6 + 团队余力 | P2 | 全部 |
| **A7** | 选型胜者 (S) | 同家族 **L**（YOLO26-l 或 DEIM-D-FINE-L；TA 桥 = M） | TAKD via M + ESKD checkpoint | (a) 稀有安全类 AP 未达 R3 门槛；OR (b) 4090 D capacity 允许（§2.4 核定 L tier 可行；X/XL 已排除） | P2 | 全部 |

**A4 触发的精确语义**：
- 主指标 mAP@0.5:0.95，95% CI 由 §6#1 pre-committed 估计法计算（默认方法 b：1000× bootstrap on per-image preds；可选方法 a：5-seed）；
- 比较：候选 cell 的 lower-CI > A1 baseline 的 point estimate（即 95% 置信"超过 A1 中位"）；
- 安全类否决：候选 cell 在每个 `full_val_support ≥ 30` 的安全类上 AP delta ≥ −0.5 pp（与 R2 精度奇偶共用阈值）；
- 平局处理：A2a / A3 / A2b 主指标差距 ≤ 0.1pp 且都通过安全类时按 wall-clock 排序选更便宜者代入 A4；
- 计算者：`scripts/_kd_decide_cell.py`（NEW；与 R2 精度奇偶 `_r2_decide_precision.py` 共用 CI 工具；接受 `--ci-method {seed5, bootstrap1000}`）。

**Canonical drawdown order**（2-week R2 in-round 预算不足时按以下顺序削减；§9 budget red line 引用本节，不重复声明）：

1. 先丢 **A7**（capacity-conditional，仅 capacity 触发而稀有类未达门槛时首批削减）；
2. 再丢 **A6**（cross-arch，证据 detection 转移弱）；
3. 最后丢 **A5**（detection 文献支持但项目数据集无验证）。

A4 / A3 / A2 / A1 / A0 不可丢——P0/P1，丢则 round 不闭环。

### 5.2 损失函数族

| 族 | 代表方法 | 适用 |
|---|---|---|
| 响应/logit | LD (Zheng 2022) | 仅 DFL-bearing 检测器（DEIM-D-FINE / GFocal）；YOLO26 已移除 DFL，故 YOLO 学生退化为 cls-logit KL only |
| 特征/hint | PKD / MGD / FGD | 通用；PKD 异构友好；MGD dense generation；FGD 前景失衡 |
| DETR-专用 | DETRDistill / KD-DETR / GO-LSD（内置） | 仅 DEIM↔DEIM |
| Head/cross-task | CrossKD | 仅 head 兼容时（同家族） |
| 自蒸馏 | Born-Again Networks / GO-LSD（内置） | 默认对照 |

### 5.3 提示层选择

| 学生 | hint 层 | 备注 |
|---|---|---|
| YOLO26-s | P3-P5 neck/FPN + head pre-logit | 与现有 trt_pipeline.cpp 解码兼容 |
| DEIM-D-FINE-S（同家族 KD） | encoder memory + decoder intermediate | DETRDistill / KD-DETR 路径 |
| DEIM-D-FINE-S（cell A6 cross-arch） | stride 8/16/32 encoder feature pyramid（§4.3 规格 resample） | 必须叠加 spatial resample + channel projection MLP |

**A6 方向**：teacher = 互补家族 M；student = 同家族 S。Projection MLP 在学生侧（学生 channels → 教师 channels）；KD loss 在投影后空间计算；**导出 TRT 时移除**。

### 5.4 训练时间表

```
Stage 0  (warm start, 10–20 epochs)
  └ COCO 预训练 → R2 数据 fine-tune（仅硬目标），与 scratch 路径并行复用同一 baseline
Stage 1  (KD-on, 80–150 epochs)
  └ KD 损失逐步升温到完整权重；与 GT 损失共训
  └ 与 fliplr=0 / copy-paste / hard-negative mining 兼容
Stage 2  (KD-off final tuning, 5–10% 末尾 epochs，可选)
  └ 关闭 KD，仅硬目标 + 强增强收尾；减少教师偏置传递
```

[I] **数据增强一致性**（Wang et al. 2022, *Inconsistent KD with Data Aug*）：教师在与学生相同的增强后图像上重新前向；不能用未增强图像作教师 logit。

**缓存模式**：默认即时计算（no cache）；如训练吞吐 < 60% A1 baseline，按 epoch 切换至 SSD 缓存（≤ 200 GB scratch quota），不切 RAM。

### 5.5 与 D-FINE 内置 GO-LSD 的交互

[E] D-FINE GO-LSD 是双向定位自蒸馏（精细分布层 → 浅层）。DEIM-D-FINE-S 学生 baseline 训练时已接收内置自蒸馏的定位信号。

[I] 推论：
- DEIM 学生加外部 LD（也是定位 KD）功能重叠，边际收益小；
- 应优先在 DEIM 学生上加 classification 通道 / encoder feature 通道外部 KD（避开定位通道）；
- A0 cell（GO-LSD 关 + 无外部 KD）净分离外部 KD 的真实贡献。

---

## 六、预承诺验收门（KD-acceptance gate）

5 项 pre-commit gate；本计划经 PM 批准后即作为 round-binding 规则。未全部通过者**不进入 KD ship-decision**（保留为研究记录）。

### #1. 总 mAP 不退化

- **A1 baseline 角色**：A1 训练完成后，CI 由以下两种估计法**任选其一**：
  - 方法 a：5 个种子重复 + mAP 95% CI 由 mean ± 1.96 × SD/√5；优 = 标准统计；劣 = 5× wall-clock。
  - 方法 b：1 次训练 + 1000× bootstrap on per-image preds（与 R2 精度奇偶共用工具）；优 = 1× wall-clock；劣 = 低估种子方差。
  - **默认方法 b**（与 R2 精度奇偶 CI 一致）；team 余力允许时切方法 a。
  - CI 记录为 `A1_CI_low / A1_point / A1_CI_high`。
- **KD cell 验收**（A2a / A2b / A3 / A4 / A5 / A6 / A7）：要求 `KD_cell_lower_CI_bound > A1_CI_low` AND `KD_cell_lower_CI_bound > A1_point − 0.5pp`（即 95% 置信不退化超过 0.5pp）。
- A1 自身不通过此门 — 它是参照，不与自己比较。
- A0（GO-LSD off baseline，DEIM 学生专用）按 A1 同样规则建立独立 CI；仅用于 §5.5 净分离分析，不直接进入 KD ship-decision。

### #2. 安全类逐类 AP

red / yellow / green / 所有箭头类 / barrier-up / barrier-down / 行人信号灯 — 每个 `full_val_support ≥ 30` 的类 AP delta ≥ −0.5 pp（与 R2 精度奇偶共用阈值）。`full_val_support < 30` 的类不阻塞。

### #3. 无新型 FP 增长

R1 demo8 / 11 / 13 类背景帧上 FP 数不上升（用 R2 hard-negative mining 同源帧检验）。

### #4. 训练成本预算

- A1 baseline 完成后记录其实测 wall-clock `T_scratch_A1`；A2+ cell 的成本验收**必须等 A1 完成后才可评估**。
- 门：单 cell wall-clock < `T_scratch_A1 × 2.0`。
- 超出者：cell 训练完成 (用作未来参考) 但不进入 KD ship-decision；记 `cost_gate_failed=true`。

### #5. 导出 TRT 引擎 + sidecar 验收

- **dtype 选取**：KD 学生导出 TRT 引擎并通过 R2 精度奇偶的 eval-parity gate（0.01 pp）；导出 dtype 由 R2 精度奇偶针对该学生选定的 `ship_precision` 决定：
  - FP16：仅 FP16 引擎验收；
  - FP32：仅 FP32 引擎验收；
  - R2 ship_precision 仍未定（KD 与 R2 真正并行）：双 dtype 导出，ship-decision 暂挂回填。
- KD 收益**在 PyTorch checkpoint 上的提升 ≠ TRT 引擎上的提升**，建议用导出引擎复测。
- Sidecar：与 `scripts/export_yolo.sh` / `scripts/export_deim.sh` 同源；KD 学生 sidecar 须额外字段 `kd_cell_id` / `kd_method` / `kd_teacher_artifact_sha256`。

---

未达任一条款者：cell 不进入 KD ship-decision（保留为研究记录，不发布、不部署），失败原因记入 `runs/_kd_decisions.json`（schema 文件 `scripts/_kd_decision_schema.json` NEW，与 `_r2_decision_schema.json` 字段兼容率 ≥ 80%）。

[I] 基于 Cho 2019 + Wang 2022 经验，预期 ≥ 30% cells 拒绝落地是正常的；流水线接受这一拒绝率，不据此调整门槛。

---

## 七、与现有 R2 / R3 计划的衔接

**整体路径**：KD 优先 R2 in-round 内嵌；时间允许时启动尽可能多 cells；时间不足时按 §5.1 优先级 (P0 > P1 > P2) 由下往上削减。

| 阶段 | KD cells | 触发 |
|---|---|---|
| R2 in-round（默认） | A0（DEIM 路径才有）+ A1 + A2a/A2b（按家族二选一） | 选型胜者一旦确定即可启动；与精度奇偶并行；不阻塞 R2 phase report |
| R2 in-round (P1 — 时间余力) | + A3 + A4 | A1 完成 + 精度奇偶不消耗超出预期的 GPU 时间 |
| R2 in-round (P2 — 时间余力 + capacity 余力) | + A5 + A6 + A7 (L tier only) | P1 完成 + 4090 D capacity 允许（A7） + 团队余力 |
| KD 降级 round（仅 R2 时间不足时启动） | 残留未完成 cells | R2 phase report 关帧时若 P0 矩阵未跑完，残留推到独立 round（不延阻 R2 关帧） |
| R3 / pre-deploy | 无 | A7 已在 R2 内闭环；KD 维度无 R3 carry-forward |

**与 R2 精度奇偶（`~/.claude/plans/elegant-sauteeing-quail.md`）的衔接**：
- 共用 frozen R2 val manifest + bootstrap CI 工具；
- 共用 eval-parity gate（0.01 pp）；
- 共用 sidecar 契约（`scripts/export_yolo.sh` / `scripts/export_deim.sh`）；
- `runs/_kd_decisions.json` schema 与 `_r2_precision_decisions.json` 共用 class-wise AP / safety-class / bootstrap CI 字段（重用率 ≥ 80%）。

**与其他规划文档**：
- `R3_precision_reproducibility.md`：互不阻塞；如 R2 build 决定性问题导致 R3 启动，KD 等 R3 复现性确认完成后再走专项，避免 build variance 噪声。
- `temporal_optimization_plan.md`：互补；TSM 改输入，KD 改损失，可叠加。
- `cross_detection_reasoning_plan.md`：解耦；共现推理是 post-detector 推理增强，KD 是检测器训练增强。

---

## 八、风险与缓解

| 风险 | 缓解措施 | 触发条件 |
|---|---|---|
| Capacity-gap：L 教师劣化 S 学生 | TAKD 桥 + ESKD 早停 + 投影 MLP（三选二同时启用） | 仅 A7 触发时 |
| 多教师协同冲突 | 渐进式而非同步；只在已观测互补错误时启用 A5 | A5 启动前用教师在 frozen val 上 per-class disagreement 作筛选 |
| 跨架构对齐失败 | 限 PKD/MGD feature-level；head-level 跨架构原则上不纳入；强制投影 MLP + spatial resample | A6 启动时 |
| GO-LSD 与外部 KD 重复 | DEIM 学生外部 KD 不走定位通道；优先 cls/encoder feature；A0 baseline 净分离 | DEIM 学生所有 cells |
| KD 不如 scratch | 验收门强制对照 baseline；接受拒绝（预计 ≥ 30%） | 所有 cells |
| 强增强（mosaic / copy-paste）破坏教师监督 | 教师在增强后图像上重新前向（Wang 2022） | 所有 cells |
| 训练成本爆炸 | 缓存教师特征/logit；不启用同步 4-教师；单 cell wall-clock ≤ 24h（适用 P0/P1/P2；A7 多阶段 TAKD 16–48h 例外） | 所有 cells（A7 例外） |
| FP16 量化破坏 KD 收益 | 验收用导出 TRT 引擎而非 PyTorch checkpoint；按 §6#5 dtype 规则 | 所有 cells |
| Cross-arch 学生权重 license 传递性 | [U, legal-untested] 当前 field-test 阶段不阻塞；A6 学生权重限内部使用；blocking gate 延后到 future-commercial-deploy 阶段 | 当前阶段 advisory；commercial-deploy 阶段重新激活 |
| 教师 fine-tune 前置成本 | 教师须先在 R2 nc 范围（10–14）上 fine-tune；估算：~6–12h × N_teachers + 2–8 GB checkpoint × N_teachers；列入 §9 budget | 所有 cells |
| 4090 D OOM | §2.4 已排除 X/XL；首 epoch 实测峰值 > 22 GB → 自动降 bs；仍 OOM → A7 cell 降级为 M 教师 cell | A7 启动时 |
| 隐性运营成本 | (a) 教师特征/logit 缓存 SSD ops + 200 GB scratch；(b) per-cell 验收审计与 demo 复评（≤ 0.5 day/cell）；(c) 法务复核（A6 学生权重对外发布前 1–3 day，blocking；当前阶段非 blocking）；(d) round report 撰写 + conflictor-loop（≤ 2 day at round close） | 全 round |

---

## 九、资源与时间预算

[planning estimate]：所有数值在 A1 完成后回填 `T_scratch_A1` 实测值并重算。

| 项目 | 预估 | 备注 |
|---|---|---|
| 单 cell 训练时间 | 8–24h | 基于 R1 YOLO26-s 单跑 ~6h / DEIM-S ~12h 同尺寸外推；KD feature-level 上限 +40% |
| 完整 P0 矩阵（A0 + A1 + A2a/A2b + A3） | 24–72h | DEIM 路径含 A0；A1 完成是 §6#1/#4 验收门激活前提 |
| 完整 P0+P1+P2 矩阵（+ A4 + A5 + A6 + A7-L） | 88–192h | R2 in-round 内（PM #1）；含 A6 跨 venv 推理 + A7 TAKD（capacity-conditional） |
| 教师 fine-tune 前置 | ~6–12h × N_teachers + 2–8 GB checkpoint × N_teachers | 教师在 R2 nc 范围（10–14）上 fine-tune；与 R2 训练共用 4090 D 基础设施；不计入 R2 budget；N_teachers：单教师路径=1，A5=2，A6=1 额外，A7=1 L 教师 |
| 教师特征缓存空间 | 0–200 GB | 默认即时计算（0 GB），仅当吞吐 < 60% baseline 切 SSD 缓存 |
| DEIM venv / YOLO venv | 独立 | A6 须用两个环境分别推理 + 训练 |
| 隐性运营成本 | ~3–5 day per round | 见 §8 隐性运营成本行 |

**预算红线**：完整 P0+P1+P2 矩阵在 R2 in-round 时间分配下不超过 **2 周墙钟时间**（含人监督 + 隐性运营）；超出按 §5.1 canonical drawdown 顺序削减（A7 → A6 → A5）。R2 phase report 关帧时若 P0 cells 仍未完成，残留推到独立"KD 降级 round"（不延阻 R2 关帧）。

---

## 十、Pre-committed 决策（无 open issue）

PM 五项答复 + R1 决策点应用后，原 §10.2 全部清空。下表覆盖所有曾经 open 的事项：

| 项目 | 决策 |
|---|---|
| Cell A0（GO-LSD 关 + 无外部 KD）作 DEIM 路径 baseline 净分离 | ✅ pre-commit；§5.1 矩阵 P0 row |
| 教师特征缓存模式 | ✅ 默认即时计算（no cache）；吞吐 < 60% A1 baseline 时切 SSD 缓存（≤ 200 GB），不切 RAM；§5.4 |
| A4 触发选择规则 | ✅ 候选 cell lower-CI > A1 point + 安全类 AP delta ≥ −0.5 pp（CI 估计法由 §6#1 pre-commit）；§5.1 + §6#1 |
| A5 / A6 触发 | ✅ A4 通过 §6 全部 5 项验收门即可；不再叠加额外 CI 比较；§5.1 |
| KD round 归属（PM #1） | ✅ R2 in-round 内嵌；时间不足时降级 round 不延阻 R2 关帧；§7 |
| GPU 独占性（PM #2） | ✅ 4090 D 24 GB 用户独占，无抢占；R2 in-round 时间假设可成立；§9 |
| 教师 pool 含 L tier（PM #3） | ✅ YOLO26-l / DEIM-D-FINE-L 纳入；X/XL 由 §2.4 容量分析排除；A7 = L tier only；P2 优先级 |
| 法务 gate 时机（PM #4） | ✅ 当前 field-test 阶段不阻塞；commercial-deploy 阶段重新激活；§8 license 行 |
| Conflictor-loop 终止（PM #5） | ✅ until AGREED，无 iter 硬上限；§13 |

---

## 十一、与 round 模板的对齐

| `docs/templates/round_template.md` 章节 | 本计划对应 |
|---|---|
| 范围 | §一 + §五.1 + §七 |
| 预承诺决策规则 | §六（5 项）+ §五.1 cell 触发条件 |
| 训练轨道 | §五 完整流水线 |
| 评估矩阵 | §五.1 + §六 |
| 决策应用 | §六（通过者可纳入 KD ship-decision；未通过者建议不纳入；门槛不宜中途调整） |
| Demo 评审 | KD 学生导出引擎后走 `scripts/run_demos.sh`，复用 demo-reviewer |
| 结转下一轮 | §七（KD 维度无 R3 carry-forward） |
| 报告 | KD cells 完成情况内嵌于 `docs/reports/phase_R2.md` 的 KD 子节；仅当残留 cells 推到"KD 降级 round"时才写独立 `phase_R2_kd.md`（无 phase_R3_kd 默认） |
| 团队签字 | conflictor-loop AGREED + PM 已确认 §十 |
| 反模式自检 | §八 风险表 + §六 验收门 |

---

## 十二、参考文献

### 跨架构 / 异构 KD
- Cao, Y., Zhang, Y., et al. (2022). *PKD: General Distillation Framework for Object Detectors via Pearson Correlation Coefficient*. NeurIPS 2022. [arXiv:2207.02039]
- Wang, J., Chen, Y., et al. (2024). *CrossKD: Cross-Head Knowledge Distillation for Object Detection*. CVPR 2024, pp. 16520–16530.
- Yang, Z., Li, Z., et al. (2022). *Masked Generative Distillation*. ECCV 2022.
- Yang, Z., Li, Z., et al. (2022). *Focal and Global Knowledge Distillation for Detectors*. CVPR 2022.

### DETR-specific KD
- Chang, J., et al. (2023). *DETRDistill: A Universal Knowledge Distillation Framework for DETR-families*. ICCV 2023. [arXiv:2211.10156]
- Wang, J., et al. (2024). *KD-DETR: Knowledge Distillation for Detection Transformer with Consistent Distillation Points Sampling*. CVPR 2024.
- Peng, Z., Yang, P., et al. (2025). *D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement*. ICLR 2025 Spotlight. (含 GO-LSD)

### 多教师 / 渐进式 / Capacity gap
- Cao, S., et al. (2023). *Learning Lightweight Object Detectors via Multi-Teacher Progressive Distillation*. ICML 2023. [arXiv:2308.09105] — detection-specific 主证据
- Mirzadeh, S.I., et al. (2020). *Improved Knowledge Distillation via Teacher Assistant*. AAAI 2020. [arXiv:1902.03393]
- Cho, J.H. & Hariharan, B. (2019). *On the Efficacy of Knowledge Distillation*. ICCV 2019. [arXiv:1910.01348]
- Liu, Yueying, et al. (2020). *Adaptive Multi-Teacher Multi-Level Knowledge Distillation*. Neurocomputing 415: 106–113. (期刊；证据为分类，**不作检测器主支撑**)
- Furlanello, T., et al. (2018). *Born-Again Neural Networks*. ICML 2018. [arXiv:1805.04770]

### 检测器 KD 基础
- Zheng, Z., Ye, R., Wang, P., et al. (2022). *Localization Distillation for Dense Object Detection*. CVPR 2022, pp. 9407–9416. (TPAMI 2023 扩展)

### 失败模式 / 数据增强相互作用
- Wang, T., et al. (2022). *Exploring Inconsistent Knowledge Distillation for Object Detection with Data Augmentation*. [arXiv:2209.09841]

### 综述
- Klempau, M., et al. (2026). *Knowledge Distillation in Object Detection: A Survey from CNN to Transformer*. Sensors 26(1):292, MDPI.
- Liu, Z., et al. (2023). *When Object Detection Meets Knowledge Distillation: A Survey*. IEEE TPAMI.

### 部署平台
- NVIDIA (2023). JetPack 5.1 Release Notes. (TensorRT 8.5.2 / CUDA 11.4 / cuDNN 8.6.0)

### 法规背景（China traffic light，间接相关）
- 国家质量监督检验检疫总局 (AQSIQ) / 国家标准化管理委员会 (SAC) (2011). GB 14887-2011 *道路交通信号灯*.
- 国家质量监督检验检疫总局 (AQSIQ) / 国家标准化管理委员会 (SAC) (2016). GB 14886-2016 *道路交通信号灯设置与安装规范*.

---

## 十三、Conflictor-loop 终止条件

**终止条件：直到 AGREED 为止**，不设 iter 硬上限。同时按"未解决 issue 状态"评估。

**Lock 条件**（同时满足）：
- (a) 当 iter 中零 CRITICAL 发现关于：决策规则歧义（§六）、citation 错误（§十二）、license 法务（§八 cross-arch 行 — 当前阶段 advisory 仍记录）；
- (b) 当 iter HIGH 发现已全部 amend；
- (c) 当 iter verdict = AGREED OR AGREED-WITH-AMENDMENTS；
- (d) 仅剩 MEDIUM/LOW 发现且非阻塞 round 启动。

**Reopen criteria**（lock 后再开）：
- §六 验收门存在某输入组合零或多个 case 同时匹配的执行歧义；
- §五.1 cells 触发条件相互依赖循环；
- §五.5 GO-LSD 与外部 KD 交互论证错误；
- §四.2 DETR↔YOLO 不可对齐论证错误；
- §八 license 法务 gate 收到与本计划相反的书面法务意见；
- §十二 citation 错误（venue / author / year / 论点支撑性）；
- §2.4 容量分析被实测推翻（4090 D 实际 VRAM 行为与表格估算偏差 > 30%）。

非以上 reopen criteria 的发现进入 transcript 附录，不触发 amendment 循环。
