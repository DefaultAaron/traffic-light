# 第 N 轮 — <轮次名称>

> **使用说明**：将本模板复制到 `docs/planning/round_<N>_plan.md`（计划阶段）和
> `docs/reports/phase_<N>.md`（报告阶段），保留所有标题，逐节填写。**任何标题
> 不得删除**：留空时写 `n/a` 或 `none`，不要省略——空白本身就是信号。

---

## 范围（Scope）

- **类别（nc）**：<列出全部类别名，如 `red, yellow, green, redLeft, ...`>
- **数据集快照**：`data/<round>/` — 提交 ID 或 freeze tag <填入>
- **备选轨道（alt-tracks）**：<列出参与本轮的所有架构 / 模型尺寸>
- **本轮排除**：<本轮明确不做的事，例如 "不调整数据增强、不引入时序模块"——
  把"不在范围"写出来，避免范围漂移>

## 预承诺决策规则（Pre-committed decision rules）

> **关键**：本节必须在训练**开始前**写完并提交。训练完成后**只允许**应用、不
> 允许修改。允许修改的唯一路径是开下一轮。

按优先级编号：

1. **主阈值**：<例 "如 YOLO26s/m 在部署域 mAP50 ≥ 0.60，备选轨道降级为监控">
2. **平局规则**：<例 "差距 < 2 pp 时，按许可证成本：DEIM > YOLOv13 > YOLO26">
3. **门槛条件**：<例 "DEIM 必须先在 Orin FP16 ≤ 50 ms/帧，否则规则 3 失效">

## 训练轨道（Tracks）

| 轨道 | 启动脚本 | venv | 备注 |
|---|---|---|---|
| <YOLO26s> | <main.py train yolo26s> | <.venv (uv)> | <主力> |
| <YOLOv13-s> | <scripts/train_yolov13.sh s> | <yolov13/.venv> | <hedge: AGPL-3.0 同许可> |
| <DEIM-D-FINE-S> | <NPROC=1 scripts/train_deim.sh s> | <DEIM/.venv> | <hedge: Apache-2.0> |

> **每条轨道的运行目录都必须有 `args.yaml` + `SEED.txt`**（轮次复现性硬约束）。

## 评估矩阵（Eval matrix）

每条轨道完成后，**完整**填写下列三列；缺哪条说明哪条为什么没做。

| 指标 | 说明 | 是否完成 |
|---|---|---|
| 域内 mAP（冻结 split） | mAP50 / mAP50-95 / 每类 recall | <Y / N + 链接到 results> |
| 部署域 mAP | 实地采集片段上的定量 + demo-reviewer 定性 | <Y / N> |
| Orin 延迟 | FP16，目标 imgsz，端到端 ms/帧 | <Y / N> |

特别关注的低样本类（如 `redRight` / `greenRight`）：<列出本轮实际样本数 +
每类 recall>。

## 决策应用（Decision applied）

- **入选基线**：<最终选定的模型>
- **触发的规则**：<规则 1 / 2 / 3 中的具体哪一条，用一句话引用规则原文>
- **被淘汰的轨道**：<列出 + 一句话原因>

如果**没有**轨道达标：<明确写"无入选；触发回退路径 X"，不要装作没事>。

## Demo 评审（Demo review）

- **覆盖率**：所有 (run × engine) 组的 ledger 状态 = `current`（不允许 `never`
  / `stale`）
- **demo-reviewer 关键发现摘要**：<3-5 条 bullet，引用 `demo/_review/summary.md`>

## 结转下一轮（Carry-forward to Round N+1）

> **不允许写 "TBD"**。每条都必须可执行、可量化。

- [ ] <项目 1：例 "redRight 类样本数 < 50，下轮收集至少 200 帧"，包含验收标准>
- [ ] <项目 2>

## 报告（Report）

- 计划文档：`docs/planning/round_<N>_plan.md`
- 阶段报告：`docs/reports/phase_<N>.md`
- 变更记录：在阶段报告内"变更记录"章节追加，不允许覆盖历史

## 团队签字（Team sign-off）

| 角色 | 必经环节 | 完成 |
|---|---|---|
| 主会话（PM） | 完整走完计划→训练→评估→决策→报告链条 | [ ] |
| `codex-plan-conflictor` | 计划阶段 ExitPlanMode 前对抗式评审 | [ ] |
| `data-scientist` | 训练前数据集 EDA + 训练后每类 recall 分析 | [ ] |
| `demo-reviewer` | 所有 (run × engine) 完成评审 | [ ] |
| `codex-report-conflictor` | 报告提交前对抗式评审 | [ ] |
| `superpowers:code-reviewer` | 涉及代码改动时（推理 / 训练脚本）对照计划评审 | [ ] |
| 你（人类 PM） | 阶段报告签字 → carry-forward 列表确认 → 进入 N+1 | [ ] |

---

## 反模式自检（每次填到这里之前停下来对一遍）

- [ ] 决策规则**在训练前**就写在了计划里
- [ ] 没有"事后调整规则以匹配结果"
- [ ] 每个 `runs/<variant>/` 都有 `args.yaml` + `SEED.txt`
- [ ] 报告里每条结论都能在 `runs/`、`results.csv`、`demo/_review/` 中追溯到证据
- [ ] carry-forward 列表是**具体的、可验收的**，不是"探索 X"
- [ ] 本轮没有引入"范围扩展"——计划里写明的范围之外的工作进了 carry-forward，
  不在本轮塞进来
