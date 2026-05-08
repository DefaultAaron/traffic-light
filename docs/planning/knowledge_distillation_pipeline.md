# 知识蒸馏（Knowledge Distillation, KD）流水线设计 — 最终模型生产环节

**状态**：v1.3（2026-05-08；v1.2 LOCKED 内容保留 + runner 文件改名 + 新增使用方式与执行清单 + 章节精简）。

**适用范围**：
- **学生候选**：YOLO26-s（R1 主力 baseline）+ DEIM-D-FINE-S（主力候选，pending Orin TRT plugin 验证）。
  - YOLOv13-s 不在候选：R1 决策规则 2 未触发（−3.4 pp mAP50）；保留为监控组应对 YOLO26 系统性失败。
  - DEIM-D-FINE-M 暂缓：R1 训练 ep40 因 DDP `unused parameter` 崩溃，需 trainer 修复 + resume 完成后再评。
- **教师候选**：M（默认）+ L（条件性，按 §2.4 容量分析）；同/跨家族均可。X/XL **不纳入**。
- **部署目标**：Jetson Orin 64GB，TRT 8.5 / JetPack 5.1，FP16 或 FP32 引擎。
- **训练资源**：单 4090 D 工作站（24 GB VRAM；用户独占）+ 独立 DEIM venv。

---

## 一、背景与触发条件

`development_plan.md` §R2 训练增强 第 133 行有一行条目（"知识蒸馏 M→S 条件性 — DEIM-M / YOLO26-m 作 teacher，KL on logits + 浅层 feature L2"）。本文档将其展开为完整流水线，并显式回答 PM 三个核心问题：

1. **Q1**：教师是否应扩展至更大 (L/XL)？
2. **Q2**：是否需要多教师，还是单教师足够？
3. **Q3**：是否做跨架构迁移（DEIM ↔ YOLO）？

---

## 二、Q1：教师规模 — 是否应扩展至 L/XL?

**证据要点**：
- Cho & Hariharan, 2019 (ICCV)：分类领域"大教师不必然更好"；缓解为 ESKD 早停教师。
- Mirzadeh et al., 2020 (AAAI, TAKD)：中等规模 TA 桥接大跨度 capacity gap。
- Cao et al., 2022 (NeurIPS, PKD)：MaskRCNN-Swin → R-50 RetinaNet/FCOS +4.1/+4.8 pp（异构+跨度同时成立）。
- Cao et al., 2023 (ICML, MTPD)：Transformer → R-50 RetinaNet 36.5→42.0 AP。
- Peng et al., 2025 (ICLR Spotlight, D-FINE)：内置 GO-LSD 双向定位自蒸馏。

**张力分析**：分类的"大教师有害"在检测领域被 PKD/MTPD **部分推翻** —— Swin/Transformer 大教师可稳定提升小 CNN 学生，前提是有 Pearson 归一化 / 渐进式 / 投影 MLP 这类**桥**。

### 2.3 决策

- **主路径**：M 教师（YOLO26-m / DEIM-D-FINE-M）。
- **矩阵补充**：cell A7 走 **L 教师**（X/XL 由 §2.4 排除），独立触发任一（A5 完成不是前置条件，避免与 §5.1 / §7 P2 "每 cell 独立触发" 冲突）：
  - (a) 稀有安全类（forwardRed / forwardGreen / barrier 双态 / 行人灯）AP 未达 R3 部署门槛；
  - (b) 4090 D capacity 允许且训练时间预算足够。
- **不纳入**：L/XL 直接对接 S 学生且无任何桥接（Cho 2019 已警示的高风险设计）。
- **缓解措施**（A7 触发时强制）：TAKD 中介 / ESKD 早停教师 / 投影 MLP，**三选二同时启用**。

### 2.4 4090 D 容量分析（X/XL 排除依据）

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
2. **X/XL 不纳入** —— headroom < 2 GB（YOLO-x）或 OOM（DEIM-X）；降 bs 解 OOM 但 wall-clock 翻倍至 4×，违反 §6#4 (< 2× scratch)。
3. **A7 cell scope = L tier only**（YOLO26-l 或 DEIM-D-FINE-L）。
4. **OOM 兜底**：首 epoch 实测峰值 > 22 GB → 自动降 bs（YOLO 8→6，DEIM 4→3）；仍 > 22 GB → A7 降级为 M 教师 cell。

---

## 三、Q2：单教师 vs 多教师?

**证据要点**：
- Cao et al., 2023 (MTPD)：顺序渐进式多教师显著优于平均/同步；约束"所有教师须检测同类"。
- Liu, Yueying et al., 2020 (AMTML-KD, Neurocomputing 415)：分类证据，**不作检测器多教师 KD 的支撑**。
- Furlanello et al., 2018 (Born-Again Networks)：同架构师生连续 KD 可超越教师；最低代价"多教师"。
- 项目预算 [I]：4 教师同步前向 = 4× forward + cache，4090 D 24 GB 紧张。

### 3.2 决策

- **baseline**：单教师 KD（A2a / A2b / A3 / A4）。
- **高级 cell**：渐进式 2 教师序列（cell A5），按 MTPD 范式。教师 1 = 同家族 M；教师 2 = 互补家族 M；触发：A4 通过全部 §6 验收门。
- **不纳入**：同步 4 教师（成本 × 4，跨 venv 开销，Cao 2023 同类约束）。
- **约束**：多教师须类集合完全一致；R2 nc 范围 10-14 锁定后，所有候选教师须先在同一类清单上 fine-tune。

---

## 四、Q3：跨架构 KD — DEIM ↔ YOLO?

**证据要点**：
- Cao et al., 2022 (PKD)：Pearson + 通道归一化 → 架构无关，明确支持异构 backbone。
- Wang et al., 2024 (CrossKD)：学生特征送入教师 head；GFL R-50 +3.5 pp；异构 backbone 可行（head 兼容是前提）。
- Yang et al., 2022 (MGD / FGD)：feature-level 架构无关；FGD 依赖共同空间网格。
- Chang et al., 2023 (DETRDistill) + Wang et al., 2024 (KD-DETR)：DETR 专用，仅 DEIM↔DEIM。

### 4.2 关键限制

DEIM 是 DETR-style set-prediction（无 NMS / anchor / 含 decoder query）；YOLO 是 dense head + 网格 + C++ NMS。直接 DETR query → YOLO grid 在结构上**无自然对齐**：(a) Hungarian 匹配 vs grid 标签分配冲突；(b) 100-300 query vs 几千 anchor 密度不匹配；(c) C++ pipeline letterbox xyxy 解码与 DETR cxcywh+sigmoid 不兼容。

→ 跨架构 KD 走**架构无关 feature-level 通道（PKD / MGD / FGD）**；head/decoder/query 直接对齐**原则上不纳入**。

### 4.3 决策（cell A6 实施规格）

- **方向**：teacher = 互补家族 M（YOLO 学生 → DEIM-M；DEIM 学生 → YOLO26-m），student = R2 选型胜者 (S)。
- **特征层级**：FPN 三层 stride 8/16/32（imgsz=1280 → 160/80/40）。DEIM 侧 encoder 多尺度 reshape 后 spatial map。
- **Spatial resample**：±1 偏移时 bilinear 对齐到学生维度。
- **通道投影**：学生侧 1×1 conv MLP（学生 channels → 教师 channels）；KD 损失在投影后空间计算。**导出 TRT 时移除该 projection**（仅训练辅助）。
- **损失**：首选 PKD；备选 MGD（慢 20-30%）；FGD foreground mask 跨架构未充分验证，不引入。
- **掩码**：A6 不引入 fg/bg 掩码；纯 spatial+channel feature mimic。

仅 DEIM↔DEIM cells 才使用 DETRDistill / KD-DETR。

---

## 五、综合流水线设计

### 5.1 实验矩阵 — pre-committed cells

| Cell | 学生 | 教师 | 方法栈 | 触发条件 | 优先级 | 适用学生家族 |
|---|---|---|---|---|---|---|
| **A0** | DEIM-D-FINE-S | — | GO-LSD off, no external KD | always — DEIM 路径 baseline 净分离对照 | P0 | DEIM only |
| **A1** | 选型胜者 | — | scratch (DEIM 路径 GO-LSD on) | always — control + KD-acceptance gate 的 wall-clock 锚点 | P0 | 全部 |
| **A2a** | YOLO26-s | YOLO26-m | cls-logit KL only（YOLO26 已移除 DFL） | always；R2 in-round 即可启动 | P0 | YOLO 家族 |
| **A2b** | DEIM-D-FINE-S | DEIM-D-FINE-M | LD on FDR + cls-logit KL | always；R2 in-round 即可启动 | P0 | DEIM 家族 |
| **A3** | 选型胜者 (S) | 同家族 M | PKD feature-level | always | P0 | 全部 |
| **A4** | 选型胜者 (S) | 同家族 M | A2a/A2b + A3 组合 | max(A2a/A2b, A3) **mAP@0.5:0.95 lower-CI > A1 point** AND 无安全类 AP delta < −0.5pp | P1 | 全部 |
| **A5** | 选型胜者 (S) | 同家族 M → 互补家族 M | MTPD 渐进 2-教师 | A4 通过 §6 全部 5 项验收门 | P2 | 全部 |
| **A6** | 选型胜者 (S) | 互补家族 M | PKD 跨架构（投影 MLP + spatial resample，§4.3 规格） | A4 通过 §6 + 团队余力 | P2 | 全部 |
| **A7** | 选型胜者 (S) | 同家族 **L**（YOLO26-l 或 DEIM-D-FINE-L；TA 桥 = M） | TAKD via M + ESKD checkpoint + 投影 MLP（三选二） | (a) 稀有安全类 AP 未达 R3 门槛；OR (b) 4090 D capacity 允许（§2.4） | P2 | 全部 |

**A4 触发的精确语义**：
- 主指标 mAP@0.5:0.95；CI 由 §6#1 pre-committed 估计法计算（默认方法 b：1000× bootstrap）。
- 比较：候选 cell lower-CI > A1 point estimate（95% 置信"超过 A1 中位"）。
- 安全类否决：每个 `full_val_support ≥ 30` 的安全类 AP delta ≥ −0.5 pp。
- 平局：A2a / A3 / A2b 主指标差距 ≤ 0.1 pp 且都通过安全类时，按 wall-clock 选更便宜者。
- 计算者：`scripts/_kd_decide_cell.py` (NEW)；与 `_r2_decide_precision.py` 共用 CI 工具；接受 `--ci-method {seed5, bootstrap1000}`。

**Canonical drawdown order**（2 周 R2 in-round 预算不足时按以下顺序削减）：
1. 先丢 **A7**（capacity-conditional）；
2. 再丢 **A6**（cross-arch，证据 detection 转移弱）；
3. 最后丢 **A5**（detection 文献支持但项目数据集无验证）。

P0 cells（A0 仅 DEIM 路径 / A1 / A2a-按家族 / A2b-按家族 / A3）不可丢——丢则 round 不闭环。**A4（P1）仅在其触发条件成立时不可丢**（即 A2/A3 满足 §5.1 lower-CI > A1 point + 安全类阈值）；触发条件不成立时 A4 不必运行，round 仍可闭环。

### 5.2 损失函数族

| 族 | 代表方法 | 适用 |
|---|---|---|
| 响应/logit | LD (Zheng 2022) | 仅 DFL-bearing 检测器；YOLO26 退化为 cls-logit KL |
| 特征/hint | PKD / MGD / FGD | 通用；PKD 异构友好；MGD dense generation；FGD 前景失衡 |
| DETR-专用 | DETRDistill / KD-DETR / GO-LSD（内置） | 仅 DEIM↔DEIM |
| Head/cross-task | CrossKD | 仅 head 兼容时（同家族） |
| 自蒸馏 | Born-Again Networks / GO-LSD（内置） | 默认对照 |

### 5.3 提示层选择

| 学生 | hint 层 |
|---|---|
| YOLO26-s | neck P3-P5 + head pre-logit |
| DEIM-D-FINE-S（同家族 KD） | encoder memory + decoder intermediate（DETRDistill / KD-DETR 路径） |
| DEIM-D-FINE-S（A6 cross-arch） | stride 8/16/32 encoder feature pyramid，§4.3 规格 resample |

### 5.4 训练时间表 + 数据增强一致性

```
Stage 0 (warm start, 10–20 epochs):  COCO 预训练 → R2 数据 fine-tune（仅硬目标）。
Stage 1 (KD-on, 80–150 epochs):       KD 损失逐步升温到完整权重；与 GT 共训。
Stage 2 (KD-off final, 5–10% 末尾):    可选；仅硬目标 + 强增强收尾。
```

**数据增强一致性**（Wang et al., 2022, *Inconsistent KD with Data Aug*）：教师在与学生**相同的增强后图像**上重新前向；不能用未增强图像作教师 logit。

**缓存模式**：默认即时计算（no cache）；训练吞吐 < 60% A1 baseline 时按 epoch 切换至 SSD 缓存（≤ 200 GB scratch quota），不切 RAM。缓存键须包含增强采样状态（参见 `components/knowledge_distillation/runners/__init__.py` 实现契约）。

### 5.5 与 D-FINE 内置 GO-LSD 的交互

D-FINE GO-LSD 是双向定位自蒸馏。DEIM-D-FINE-S 学生 baseline 训练时已接收内置自蒸馏的定位信号 → DEIM 学生加外部 LD 功能重叠，**预期边际收益小**。

**A2b 的角色**：A2b（LD on FDR + cls-logit KL）是该重叠的**实证 ablation**，不是"避开定位通道"原则的反例。三 cell 组合 A0 (GO-LSD off, no KD) + A1 (scratch, GO-LSD on) + A2b (GO-LSD on, +外部 LD + cls-KL) 给出净分离证据：
- A1 − A0 = GO-LSD 内置自蒸馏的定位贡献；
- A2b − A1 = 在 GO-LSD 之上**外部 LD + cls-KL 是否还有边际收益**。

若 A2b − A1 ≤ §6#1 噪声阈值，证实 §5.5 的预期重叠假设；后续 DEIM 学生的 P1+ 优先 classification / encoder feature 通道（A3 / A4）。这一净分离逻辑独立于 A2b 是否通过 §6 验收门。

---

## 六、预承诺验收门（KD-acceptance gate）

5 项 pre-commit gate；本计划经 PM 批准后即作为 round-binding 规则。未全部通过者**不进入 KD ship-decision**（保留为研究记录）。

### #1. 总 mAP 不退化

- **A1 baseline 角色**：A1 训练完成后，CI 由两种估计法**任选其一**：
  - 方法 a：5 个种子重复 + mAP 95% CI 由 mean ± 1.96 × SD/√5；优 = 标准统计；劣 = 5× wall-clock。
  - 方法 b：1 次训练 + 1000× bootstrap on per-image preds（与 R2 精度奇偶共用工具）；优 = 1× wall-clock；劣 = 低估种子方差。
  - **默认方法 b**；team 余力允许时切方法 a。CI 记录为 `A1_CI_low / A1_point / A1_CI_high`。
- **KD cell 验收**（A2a/A2b/A3/A4/A5/A6/A7）：要求 `KD_cell_lower_CI_bound > A1_CI_low` AND `KD_cell_lower_CI_bound > A1_point − 0.5pp`（即 95% 置信不退化超过 0.5pp）。
- A1 自身不通过此门 — 它是参照。
- A0 按 A1 同样规则建立独立 CI；仅用于 §5.5 净分离分析，不直接进入 ship-decision。

### #2. 安全类逐类 AP

red / yellow / green / 所有箭头类 / barrier-up / barrier-down / 行人信号灯 — 每个 `full_val_support ≥ 30` 的类 AP delta ≥ −0.5 pp。`full_val_support < 30` 的类不阻塞。

### #3. 无新型 FP 增长

R1 demo8 / 11 / 13 类背景帧上 FP 数不上升（用 R2 hard-negative mining 同源帧检验）。

### #4. 训练成本预算

- A1 完成后记录其实测 wall-clock `T_scratch_A1`；A2+ cell 的成本验收**必须等 A1 完成后才可评估**。
- 门：单 cell wall-clock < `T_scratch_A1 × 2.0`。
- 超出者：cell 训练完成（用作未来参考）但不进入 ship-decision；记 `cost_gate_failed=true`。

### #5. 导出 TRT 引擎 + sidecar 验收

- **dtype 选取**：KD 学生导出 TRT 引擎并通过 R2 精度奇偶 eval-parity gate（0.01 pp）；导出 dtype 由 R2 精度奇偶针对该学生选定的 `ship_precision` 决定：
  - FP16：仅 FP16 引擎验收；
  - FP32：仅 FP32 引擎验收；
  - R2 ship_precision 仍未定（KD 与 R2 真正并行）：双 dtype 导出，ship-decision 暂挂回填。
- KD 收益**在 PyTorch checkpoint 上 ≠ TRT 引擎上**，建议用导出引擎复测。
- Sidecar：与 `scripts/export_yolo.sh` / `scripts/export_deim.sh` 同源。**KD 学生 sidecar 须额外三字段** `kd_cell_id` / `kd_method` / `kd_teacher_artifact_sha256`。两个 export 脚本当前**未输出**这些字段；首个 KD cell 落地前必须先完成 export-script 扩展（添加 env/CLI 入口接受这三个值，写入 sidecar；若任一 KD 字段缺失则 sidecar 视为不完整，引擎拒绝纳入 ship-decision）。属于已识别 carry-forward，不在本计划当前实现范围。

未达任一条款者：cell 不进入 ship-decision，失败原因记入 `runs/_kd_decisions.json`（schema `scripts/_kd_decision_schema.json` NEW，与 `_r2_decision_schema.json` 字段兼容率 ≥ 80%）。预期 ≥ 30% cells 拒绝落地是正常的（基于 Cho 2019 + Wang 2022 经验）；不据此调整门槛。

---

## 七、与现有 R2 / R3 计划的衔接

KD 优先 R2 in-round 内嵌；时间允许时启动尽可能多 cells；时间不足时按 §5.1 优先级 (P0 > P1 > P2) 由下往上削减。

| 阶段 | KD cells | 触发 |
|---|---|---|
| R2 in-round（默认 P0） | A0（仅 DEIM）+ A1 + A2a/A2b（按家族二选一）+ **A3** | 选型胜者一旦确定即可启动；不阻塞主线 R2 训练**报告初稿**。A3 为 P0/Always（§5.1 + §12 一致）。**KD 子节 / round closure 仍要求 P0 完成**（§5.1 + §12）；未完成则转入下行 KD 降级 round，并在 R2 报告 coverage-gaps 中列明。 |
| R2 in-round (P1) | + A4 | A1 完成 + max(A2a/A2b, A3) lower-CI > A1 point + 安全类阈值（§5.1 A4 触发） |
| R2 in-round (P2) — **每 cell 独立触发** | + A5 | A4 通过 §6 **全部 5 项**验收门（§5.1 A5 触发） |
|  | + A6 | A4 通过 §6 全部 5 项 + 团队余力（§5.1 A6 触发） |
|  | + A7 (L tier only) | (a) 稀有安全类（forwardRed/forwardGreen/barrier 双态/行人灯）AP 未达 R3 门槛；**OR** (b) 4090 D capacity 允许（§5.1 A7 触发；§2.4 容量分析锁定 L tier，X/XL 排除） |
| KD 降级 round | 残留未完成 cells | R2 phase report 关帧时若 P0 未跑完，残留推到独立 round；A4/A5/A6/A7 触发条件不成立时不入降级 round（其触发本身即是闭环判据） |
| R3 / pre-deploy | 无 | A7 已在 R2 内闭环；KD 维度无 R3 carry-forward |

**与 R2 精度奇偶（`~/.claude/plans/elegant-sauteeing-quail.md`）共用**：frozen R2 val manifest + bootstrap CI 工具 + eval-parity gate（0.01 pp）+ sidecar 契约 + `_kd_decisions.json` 与 `_r2_precision_decisions.json` ≥ 80% 字段重用。

---

## 八、风险与缓解（Top 6）

| 风险 | 缓解 |
|---|---|
| Capacity-gap：L 教师劣化 S 学生 | TAKD + ESKD + 投影 MLP **三选二同时启用**（A7 触发时） |
| 跨架构对齐失败 | 限 PKD/MGD feature-level；强制投影 MLP + spatial resample（A6） |
| GO-LSD 与外部 KD 重复 | A2b 故意保留外部 LD on FDR + cls-logit KL 以对此重叠做**实证 ablation**（A0 + A1 + A2b 三者的 mAP delta 差给出净分离证据，§5.5）；若 A2b − A1 ≤ §6#1 噪声阈值，**DEIM 学生 P1+ cells（A3 / A4）优先 cls/encoder feature 通道**避开定位重叠 |
| 强增强破坏教师监督 | 教师在增强后图像上**重新前向**（Wang 2022） |
| 训练成本爆炸 | 缓存教师特征/logit；不启用同步 4 教师；§6#4 wall-clock < 2× scratch |
| 4090 D OOM | §2.4 已排除 X/XL；首 epoch 峰值 > 22 GB 自动降 bs；仍 OOM → A7 降级为 M 教师 |

辅助：FP16 量化破坏 KD 收益（验收用 TRT 引擎而非 PyTorch ckpt）；Cross-arch 学生权重 license 传递性（A6，commercial-deploy 阶段重新激活，field-test 不阻塞）；教师 fine-tune 前置成本（~6-12h × N_teachers）。

---

## 九、资源与时间预算

[planning estimate]：A1 完成后回填 `T_scratch_A1` 实测值并重算。

| 项目 | 预估 |
|---|---|
| 单 cell 训练时间 | 8-24h（KD feature-level 上限 +40%） |
| 完整 P0 矩阵（A0 + A1 + A2a/A2b + A3） | 24-72h |
| 完整 P0+P1+P2（+ A4 + A5 + A6 + A7-L） | 88-192h |
| 教师 fine-tune 前置 | ~6-12h × N_teachers + 2-8 GB checkpoint × N_teachers |
| 教师特征缓存空间 | 0-200 GB（默认 0；< 60% A1 baseline 切 SSD） |
| 隐性运营成本 | ~3-5 day per round（demo 复评 + 法务 + report） |

**预算红线**：完整 P0+P1+P2 矩阵在 R2 in-round 时间分配下 ≤ **2 周墙钟时间**（含人监督 + 隐性运营）；超出按 §5.1 canonical drawdown 削减。R2 phase report 关帧时若 P0 未完成，残留推到"KD 降级 round"。

---

## 十、Pre-committed 决策（无 open issue）

| 项目 | 决策 |
|---|---|
| Cell A0（GO-LSD 关 + 无外部 KD）DEIM 路径 baseline 净分离 | ✅ §5.1 矩阵 P0 row |
| 教师特征缓存模式 | ✅ 默认即时计算；< 60% A1 baseline 切 SSD（≤ 200 GB），不切 RAM |
| A4 触发选择规则 | ✅ 候选 lower-CI > A1 point + 安全类 AP delta ≥ −0.5 pp |
| A5 触发 | ✅ A4 通过 §6 全部 5 项验收门（§5.1 / §7） |
| A6 触发 | ✅ A4 通过 §6 全部 5 项 + 团队余力（§5.1 / §7） |
| A7 触发 | ✅ 独立于 A4/A5：(a) 稀有安全类 AP 未达 R3 门槛 **OR** (b) 4090 D capacity 允许；L tier only，X/XL 排除（§2.3 / §5.1 / §7 / §12） |
| KD round 归属（PM #1） | ✅ R2 in-round 内嵌；时间不足时降级 round 不延阻 R2 关帧 |
| GPU 独占性（PM #2） | ✅ 4090 D 24 GB 用户独占，无抢占 |
| 教师 pool 含 L tier（PM #3） | ✅ YOLO26-l / DEIM-D-FINE-L 纳入；X/XL 由 §2.4 排除 |
| 法务 gate 时机（PM #4） | ✅ field-test 阶段不阻塞；commercial-deploy 阶段重新激活 |
| Conflictor-loop 终止（PM #5） | ✅ until AGREED，无 iter 硬上限 |

---

## 十一、使用方式（runner 模块映射）

每个 cell 一个 Python 入口模块，位于 `components/knowledge_distillation/runners/`。文件名描述其方法栈，cell ID 在 docstring 中作为规范引用。所有 runner 当前为 stub（`NotImplementedError`），将在所属 cell 调度时落地。

| Cell | runner 模块 | 调用形式（落地后） |
|---|---|---|
| A0 | `deim_baseline_golsd_off.py` | `python -m components.knowledge_distillation.runners.deim_baseline_golsd_off --config <cfg>` |
| A1 | `scratch_baseline.py` | `python -m components.knowledge_distillation.runners.scratch_baseline --config <cfg>` |
| A2a | `yolo_logit_kd.py` | `python -m components.knowledge_distillation.runners.yolo_logit_kd --teacher-ckpt <pt> --config <cfg>` |
| A2b | `deim_logit_localization_kd.py` | `python -m components.knowledge_distillation.runners.deim_logit_localization_kd --teacher-ckpt <pth> --config <cfg>` |
| A3 | `pearson_feature_kd.py` | `python -m components.knowledge_distillation.runners.pearson_feature_kd --teacher-ckpt <ckpt> --config <cfg>` |
| A4 | `logit_plus_feature_kd.py` | `python -m components.knowledge_distillation.runners.logit_plus_feature_kd --teacher-ckpt <ckpt> --config <cfg>` |
| A5 | `progressive_multi_teacher.py` | `python -m components.knowledge_distillation.runners.progressive_multi_teacher --teacher-ckpt <t1> --teacher-ckpt <t2> --config <cfg>` （重复 `--teacher-ckpt`，按出现顺序对应 phase 1 / phase 2；这是单数契约的 nargs 扩展，不是改名）|
| A6 | `cross_arch_feature_kd.py` | `python -m components.knowledge_distillation.runners.cross_arch_feature_kd --teacher-ckpt <other-family> --config <cfg>` |
| A7 | `takd_large_teacher.py` | `python -m components.knowledge_distillation.runners.takd_large_teacher --teacher-ckpt <L> --assistant-ckpt <M> --config <cfg>` |

**统一 CLI 契约**（在 `runners/__init__.py` 中文字记录，第一个 cell 落地时锁定）：
```
--config <yaml>             student + teacher + KD 超参
--teacher-ckpt <path>       已 fine-tune 的教师 artifact (R2 nc range)
--student-init {scratch, coco, r2_baseline}
--output-dir runs/<cell_id>/
--seed <int>                 训练开始前写入 SEED.txt
--ci-method {bootstrap1000, seed5}    默认 bootstrap1000（§6#1）
--resume <ckpt>             resume 时不覆写 SEED.txt（须沿用原 SEED）
```

**辅助子包**：
- `losses/` — KD 损失模块（cls_logit_kl / ld_fdr / pkd / mgd / projection_mlp / kd_weight_ramp 标量）；仅 per-batch KD 信号。
- `schedules/` — 多阶段 / 多教师编排（kd_phase_runner / mtpd_progressive / takd_assistant / eskd_loader / golsd_toggle）；非损失函数。
- `gates/` — §6 验收门评估器（gate1-5）。

**Sidecar 与决策**：导出 TRT 引擎沿用 `scripts/export_yolo.sh` / `scripts/export_deim.sh`。**两个 export 脚本当前不输出 KD sidecar 字段**（`kd_cell_id` / `kd_method` / `kd_teacher_artifact_sha256`），首个 KD cell 落地前必须先扩展 export 脚本写入这三字段（详见 §6#5）；这是已识别 carry-forward，不在本 scaffold 范围。Ship-decision 由 `scripts/_kd_decide_cell.py`（NEW，deferred）写入 `runs/_kd_decisions.json`，schema 校验由 `scripts/_kd_decision_schema.json`（NEW，deferred）。

---

## 十二、执行清单（cell-by-cell checkbox）

按 §5.1 优先级与 §7 阶段顺序勾选。所有 P0 必须完成 round 才闭环；P1/P2 按预算与触发条件决定。

### P0（强制；R2 in-round 默认 — 按选型胜者家族勾选）

**Always:**
- [ ] **A1 / `scratch_baseline.py`** —— scratch 训练 + 记录 `T_scratch_A1` + 计算 A1 CI（`bootstrap1000` 默认）。**全部 KD cell 验收依赖此**。
- [ ] **A3 / `pearson_feature_kd.py`** —— 同家族 M 教师，PKD feature-level。

**If R2 winner = YOLO 家族（A0/A2b N/A）:**
- [ ] **A2a / `yolo_logit_kd.py`** —— cls-logit KL only。

**If R2 winner = DEIM 家族（A2a N/A）:**
- [ ] **A0 / `deim_baseline_golsd_off.py`** —— GO-LSD 关 + 无外部 KD baseline，§5.5 净分离分析输入。
- [ ] **A2b / `deim_logit_localization_kd.py`** —— LD on FDR + cls-logit KL（**A2b 是 GO-LSD-vs-外部-LD 重叠的实证 ablation**；§5.5 预测边际收益小，A2b + A0 + A1 三者结合给出净分离证据）。

### P1（A1 + A2/A3 完成后；时间余力）

- [ ] **A4 / `logit_plus_feature_kd.py`** —— 触发：max(A2a/A2b, A3) lower-CI > A1 point + 安全类 AP delta ≥ −0.5 pp（§5.1）。

### P2（按 §5.1 / §7 各 cell 独立触发；canonical drawdown 顺序）

- [ ] **A5 / `progressive_multi_teacher.py`** —— MTPD 渐进 2 教师。**触发：A4 通过 §6 全部 5 项验收门**。
- [ ] **A6 / `cross_arch_feature_kd.py`** —— 跨架构 PKD（投影 MLP + spatial resample）。**触发：A4 通过 §6 全部 5 项 + 团队余力**。
- [ ] **A7 / `takd_large_teacher.py`** —— L 教师 + TAKD/ESKD/projection 三选二桥接。**触发**（独立于 A4/A5）：(a) 稀有安全类 AP 未达 R3 门槛 **OR** (b) 4090 D capacity 允许（§5.1 / §2.3 / §2.4）。

### 全 round-level 验收

- [ ] §6#1 总 mAP 不退化：所有候选 cell `lower-CI > A1_CI_low` AND `lower-CI > A1_point − 0.5pp`。
- [ ] §6#2 安全类逐类 AP：每个 `full_val_support ≥ 30` 安全类 AP delta ≥ −0.5 pp。
- [ ] §6#3 无新型 FP 增长：R1 demo8/11/13 背景帧 FP 数不上升。
- [ ] §6#4 训练成本：每 cell wall-clock < `T_scratch_A1 × 2.0`。
- [ ] §6#5 TRT 引擎 + sidecar 验收：导出引擎通过 0.01 pp eval-parity，sidecar 含 `kd_cell_id` / `kd_method` / `kd_teacher_artifact_sha256`。
- [ ] `runs/_kd_decisions.json` 写入完整记录，`scripts/_kd_decide_cell.py` 不报 schema 错误。
- [ ] `phase_R2.md` KD 子节落地（含 cell 通过/拒绝表 + 拒绝原因）。

---

## 十三、参考文献

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
- Liu, Yueying, et al. (2020). *Adaptive Multi-Teacher Multi-Level Knowledge Distillation*. Neurocomputing 415: 106–113. (期刊；分类证据，**不作检测器主支撑**)
- Furlanello, T., et al. (2018). *Born-Again Neural Networks*. ICML 2018. [arXiv:1805.04770]

### 检测器 KD 基础
- Zheng, Z., Ye, R., Wang, P., et al. (2022). *Localization Distillation for Dense Object Detection*. CVPR 2022, pp. 9407–9416. (TPAMI 2023 扩展)

### 失败模式 / 数据增强相互作用
- Wang, T., et al. (2022). *Exploring Inconsistent Knowledge Distillation for Object Detection with Data Augmentation*. [arXiv:2209.09841]

### 综述
- Klempau, M., et al. (2026). *Knowledge Distillation in Object Detection: A Survey from CNN to Transformer*. Sensors 26(1):292, MDPI.
- Liu, Z., et al. (2023). *When Object Detection Meets Knowledge Distillation: A Survey*. IEEE TPAMI.

### 部署平台 / 法规
- NVIDIA (2023). JetPack 5.1 Release Notes. (TensorRT 8.5.2 / CUDA 11.4 / cuDNN 8.6.0)
- AQSIQ / SAC (2011). GB 14887-2011 *道路交通信号灯*.
- AQSIQ / SAC (2016). GB 14886-2016 *道路交通信号灯设置与安装规范*.

---

## 十四、Conflictor-loop 终止条件

**终止条件**：直到 AGREED；不设 iter 硬上限。

**Lock 条件**（同时满足）：当 iter 零 CRITICAL（决策规则歧义 / citation 错 / license 法务）+ HIGH 全部 amend + verdict ∈ {AGREED, AGREED-WITH-AMENDMENTS} + 仅剩 MEDIUM/LOW 非阻塞。

**Reopen criteria**：§六 验收门执行歧义、§五.1 cell 触发循环依赖、§五.5 GO-LSD 论证错、§四.2 DETR↔YOLO 论证错、§八 license 收到反向法务意见、§十三 citation 错（参考文献节）、§2.4 容量分析被实测推翻（VRAM 偏差 > 30%）。其他发现进入 transcript 附录，不触发 amendment 循环。
