# A6 跨架构 KD 设计 spike — DEIM-D-FINE-M → YOLO26-s (2026-05-11)

## 状态

| 项 | 值 |
|---|---|
| spike kind | pre-R2 R1 rehearsal；GPU-free；paper + 1-batch shape probe |
| 触发 | `pre_r2_kickoff_checklist.md` §2.5 KD rehearsal 表 A6 行 |
| 决策影响 | `additional_components_plan.md` §七 A6 优先级（P1 / P2）；A6 方法栈选项 |
| 输出 | `runs/rehearsal_kd_A6_design_spike.json` + 本文档 |
| spike 脚本 | `components/knowledge_distillation/spikes/a6_design_spike.py` |
| rehearsal_kind | `synthetic_fixture`（1-batch 随机张量探针，无 R1 标注依赖） |

## 一、问题陈述

§七 v2 A6 行声明跨架构 KD 方法栈 = "logit + **FDR↔DFL 分布对齐** + projection MLP"。spike 任务是验证三件事：

1. 教师端 DEIM-D-FINE-M 的 FDR 头输出形状 / 路径可读出
2. 学生端 YOLO26-s 是否存在原生 DFL 头可作为分布对齐目标
3. projection MLP 的具体设计（DETR query embed ↔ YOLO grid 的映射可行性）

## 二、核心发现

### 2.1 YOLO26 学生侧无 DFL 头（与 YOLOv8 / v11 不同）

YOLO26-s R1 ckpt（`runs/detect/yolo26s-r1/weights/best.pt`）经 `ultralytics.nn.modules.head.Detect` 实测：

| 属性 | 值 |
|---|---|
| `reg_max` | **1** |
| `nc` | 7 |
| `no` | 11 = nc(7) + 4×1 |
| `cv2`（回归头） | `Conv2d(32, 4, k=1)` × 3 尺度 |
| `cv3`（分类头） | `Conv2d(128, 7, k=1)` × 3 尺度 |
| `stride` | [8, 16, 32] |

YOLO26 的 Detect 头 `reg_max=1` 等价 DFL 关闭：每锚点回归输出 **4 个直接坐标通道**，没有 4×17 bin 分布的 DFL 设计。这与 YOLOv8 / YOLOv11 默认 `reg_max=16` 的 DFL 路径**不同**。

### 2.2 DEIM-D-FINE-M 教师侧 FDR 头是 132-logit 分布

DEIM 源码 `DEIM/engine/deim/dfine_decoder.py:506-507` + base config `DEIM/configs/base/dfine_hgnetv2.yml:61`：

| 属性 | 值 |
|---|---|
| `reg_max` | **32** |
| `num_classes` | 7 |
| 回归头 | `MLP(hidden_dim, hidden_dim, 4×(reg_max+1)=132, 3)` |
| 分类头 | `Linear(hidden_dim, num_classes=7)` |
| query 数 | ~300（DEIM 默认 num_queries） |
| Integral | softmax over 33-bin per side → 4 标量坐标 |

### 2.3 §七 v2 A6 的"FDR↔DFL 对齐"在当前 YOLO26 形态下**不可直接实施**

学生侧没有 DFL 头 → 无目标分布可与教师 FDR 做 KL；§七 v2 立项时未核查 YOLO26 Detect 头实际 `reg_max`，假设了 YOLOv8 式 DFL 结构。spike 修正此前提。

### 2.4 DETR query embed ↔ YOLO grid 映射 — 复审仍维持 OUT-OF-SCOPE

`cross_arch_feature_kd.py` v1.2 docstring：

> DETR set-prediction has no natural mapping to YOLO grid + NMS

spike 复审：DEIM 是 set-prediction（query 数固定，匹配靠 Hungarian），YOLO 是 grid+anchor（每锚点固定空间位置）。query-grid 直接映射会丢失 set-prediction 的去重特性。**A6 的 "projection MLP" 应特指 FPN 特征金字塔的 1×1 投影 conv，而非 query embedding 投影**。§七 A6 行的 "projection MLP" 在 spike pass 后建议改写为 "FPN 投影 conv"。

## 三、三条恢复路径

| 路径 | 方法栈 | 实现复杂度 | 长尾信号保留 | 1-week PoC 可行性 |
|---|---|---|---|---|
| β | 训练期辅助 DFL 头（YOLO 学生新增 4×17-bin 分布头）+ DEIM FDR 监督 KL（reg_max 重映射 32→16）+ PKD + cls-logit KL | 高（+30-40% 训练 step） | 强 | 低-中 |
| γ | DEIM FDR 经 Integral 坍缩成 4 标量坐标 → L1 + GIoU bbox KD + PKD + cls-logit KL | 中（与 A2a 同档） | 中（保留 Hungarian 选出的长尾正样本） | 高 |
| δ | PKD feature-level + cls-logit KL；放弃分布 KD（回到 `cross_arch_feature_kd.py` v1.2 spec） | 低（runner stub 改实现） | 弱 | 极高 |

### 3.1 路径 β — 训练期辅助分布头

学生 YOLO26-s 在三尺度 head 之后挂载 4 路辅助 `Linear(32→4×17)` 分布头，仅在训练期前向。监督来自 DEIM-M FDR：DEIM 的 33-bin 分布需重映射到学生的 17-bin（线性 / 学习投影皆可）。主回归 loss 仍用 YOLO26-s 原生 4-coord 输出。

- **导出形态**：辅助头不进 ONNX export（在 `forward()` 中以 `training` 分支控制）。TRT engine 与现有 YOLO26-s engine 完全一致 → 部署侧零变化
- **核心风险**：DEIM 33-bin → YOLO 17-bin 的重映射没有现成参考；学生辅助头初始化敏感；训练 step 时间增加 30-40%
- **A6 立项卖点保留度**：强 — 完整保留 DEIM 分布信号

### 3.2 路径 γ — Integral 坍缩

直接复用 DEIM 自己的 `Integral` 模块（`dfine_decoder.py:246-268`）：

```
softmax(FDR_132_logits) → reshape (B, L, 4, 33) → ⊙ project_weights → (B, L, 4) box coords
```

教师输出 4-coord，学生输出 4-coord，直接做 L1 + GIoU bbox KD（无需任何分布对齐）。配合 cls-logit KL（学生 / 教师 num_classes=7 一致，直接 KL）+ PKD feature。

- **卖点保留度**：中等 — Integral 坍缩后丢失了分布峰锐度信息，但保留了 Hungarian matcher 选出的长尾正样本与中心位置。等价于"用 DEIM 当一个更好的回归 + 分类 oracle"
- **实现成本**：与 A2a 同档；无新增可训练参数
- **PoC 最快**

### 3.3 路径 δ — 弃分布 KD

直接落地 `cross_arch_feature_kd.py` v1.2 docstring 已写好的 PKD feature-only 设计 + 增补 cls-logit KL（与 A2a 同款）。

- 与 § 七 v2 A6 立项理由（"DEIM 长尾教学信号"）**关联最弱**
- A6 与 A4（A2 + A3 同家族）的差异化窗口主要靠 "DEIM-M 的长尾召回强 → 训 YOLO 学生"；δ 路径几乎只剩"用另一家族当教师"，长尾信号靠 Hungarian + 损失而非分布

## 四、投影 MLP（β / γ / δ 共用）设计

特征金字塔投影 conv（**非** query embedding 投影）：

| 项 | 配置 |
|---|---|
| 学生 FPN 通道 | YOLO26-s P3/P4/P5 = [128, 256, 512]（实测） |
| 教师 FPN 通道 | DEIM-M hybrid_encoder 输出 hidden_dim=256（三尺度同维） |
| 投影 conv | `Conv2d(C_student → 256, k=1)` × 3 尺度（学生侧） |
| 空间对齐 | 学生 / 教师 stride 不同 → bilinear resample 学生侧到教师空间 |
| PKD loss | MSE(student_proj, teacher_pyramid)；MGD 备份 |
| FGD | 不启用（跨架构前景 / 背景语义不一致已在 v1.2 标定） |
| 部署 | 投影 conv 不导出 ONNX；TRT engine 仅含 YOLO26-s 原始头 |

## 五、Spike 1-batch shape sanity 计划

`components/knowledge_distillation/spikes/a6_design_spike.py` 实测项（无需 DEIM ckpt，DEIM 走源码静态分析）：

1. 加载 YOLO26-s R1 ckpt（`runs/detect/yolo26s-r1/weights/best.pt`）
2. 读出 `Detect` 头 `(reg_max, nc, no, cv2.channels, cv3.channels, stride)`
3. 模拟三尺度 P3/P4/P5 特征张量（B=1, C=[128,256,512], H/W=80/40/20 @ imgsz=640）
4. 构造 PKD 投影 conv `Conv2d(C_s → 256, k=1)`；前向，验证输出 shape (B, 256, H, W)
5. 模拟 DEIM 教师 FPN（B=1, C=256, H=80/40/20）；逐尺度计算 MSE 张量值（验证 loss 可数值化）
6. 模拟 DEIM FDR 张量 (B=1, num_queries=300, 132)；调用 `Integral.softmax + project` 公式（不依赖 DEIM 源码 import；脚本内复刻 Integral 数学）输出 (B, 300, 4)
7. 模拟 cls-logit (B=1, 300, 7) 与 YOLO cls-logit (B=1, 8400, 7)；演示 KL（需先做 Hungarian 匹配 / topk 选择，spike 中只演示 shape 对齐）
8. 写 `runs/rehearsal_kd_A6_design_spike.json`，schema：

```json
{
  "rehearsal_kind": "synthetic_fixture",
  "spike_date": "2026-05-11",
  "yolo26s_head": {"reg_max": 1, "nc": 7, "no": 11, "stride": [8, 16, 32], "p_channels": [128, 256, 512]},
  "deim_dfine_m_head_from_source": {"reg_max": 32, "num_classes": 7, "fdr_logits_per_query": 132, "hidden_dim": 256},
  "incompatibility_found": "yolo26_no_dfl_head",
  "projection_conv_shape_check": "pass" | "fail",
  "pkd_loss_numerical_check": "pass" | "fail",
  "integral_collapse_shape_check": "pass" | "fail",
  "selected_path": "gamma" | "beta" | "delta" | "blocked",
  "selected_path_rationale": "...",
  "a6_priority_recommendation": "P1_with_path_gamma" | "P2_demote" | "blocked",
  "next_step": "1-week PoC with selected path on R1 data" | "...",
  "spike_pass": true | false
}
```

## 六、Pass / fail（pre-committed）

| 信号 | pass | fail |
|---|---|---|
| YOLO26 head 形状探针 | reg_max / nc / channels 读出无异常 | crash 或字段缺失 |
| DEIM FDR 形状（源码） | reg_max=32 / 132 logits / Integral 路径推得 | 源码与 base config 不一致 |
| 投影 MLP shape check | 三尺度 MSE 可数值化 | shape 不匹配 |
| Integral 坍缩 shape check | 输出 (B, L, 4) 形状正确 | 数值溢出 / 形状错 |
| 选定路径 | β / γ / δ 任一明确 | 三路径全部不可行 |

**全部 pass → A6 维持 P1（建议方法栈 = γ，cls-logit KL + L1/GIoU bbox KD + PKD feature；§七 A6 行 "FDR↔DFL 分布对齐 + projection MLP" 改写为 "DEIM FDR Integral 坍缩 + bbox KD + PKD FPN 投影 conv"）**
**任何 fail → A6 demoted P2**，移交 §七 retrospective。

## 七、推荐（spike 默认）

**γ 优先 + δ 兜底**：
- γ 与 A6 立项理由（"DEIM 长尾教学信号"）连接通过 Hungarian-matched 长尾正样本 + Integral 坐标 + cls KL，比 δ 多保留信号
- 实现成本与 A2a 同档，可与 A2a 共用 cls-logit KL 损失模块
- 1-week PoC 路径：runner 改实，长尾 recall 抬升 ≥ +5 pp → A6 P1 持仓；不达标 → β 补救（额外 1d 设计 budget）OR A6 → P2

**β 留作 fallback**：γ PoC fail 后启动；不在当前 spike 落地。

## 八、对 §七 / kickoff 的回写

spike pass 后建议改动（不在此 spike commit 中执行，留给后续 §七 amendment）：

1. `additional_components_plan.md` §七 A6 行 "方法栈" 列 → "logit + Integral 坍缩 bbox KD + PKD FPN 投影 conv"（去掉 "FDR↔DFL 分布对齐" 与 "projection MLP" 表述）
2. §七 A6 优先级说明（2026-05-11） → 追加 spike-decision 引用 + γ 路径选定
3. `pre_r2_kickoff_checklist.md` §2.5 A6 spike 行状态：`gated on A6 投影层设计` → `complete (path=γ)`
4. `cross_arch_feature_kd.py` runner 实装：γ 路径优先；docstring 与 §七 v2 修订同步

## 九、衔接

- `components/knowledge_distillation/spikes/a6_design_spike.py`：spike 执行脚本
- `runs/rehearsal_kd_A6_design_spike.json`：spike 决策 JSON
- `additional_components_plan.md` §七 A6：方法栈最终态在 spike 后写定
- `pre_r2_kickoff_checklist.md` §2.5 A6 行：spike 完成回写
- `research/surveys/detection_enhancements.md` §3.4：cross-arch KD survey
