# 报告/建议书写作流水线

> 适用范围：面向管理层 / 外部机构 / 投资人的**叙事性文档**——例如本仓库的 `docs/proposals/` 目录、外部沟通的白皮书、董事会备忘录。
>
> **不适用**：检测项目本身的轮次报告（`docs/reports/phase_<round>.md`）。轮次报告走的是另一条流水线（`paper-researcher` + `superpowers:code-reviewer` (B2) + `codex-report-conflictor`），有 demo-coverage hook 把守，与本流水线**互不干涉**。
>
> 本流水线脱胎于 `~/Documents/Projects/book-creator`（长篇书籍生成框架）的 7-phase 设计，但裁剪到 4 个阶段，去除 per-section deal-loop / per-chapter overview / writer 分配 worktree 等针对长文的机制，仅保留对一次性提案/白皮书最有价值的"并行研究 + 对抗式审稿"两环。

---

## 1. 团队（4 个角色）

| 角色 | 运行时 | 是否写盘 | 适用阶段 |
|---|---|---|---|
| **主会话**（本 Claude Code 会话） | Claude Code | 是——唯一负责改 `docs/proposals/` 文档 | Phase 0/1/2/3 全程协调 |
| **`report-gemini-researcher`** | Gemini CLI（`--approval-mode plan`） | 否（只读） | Phase 1 RESEARCH |
| **`report-codex-collaborator` MODE: RESEARCH** | codex-companion（无 `--write`） | 否（只读） | Phase 1 RESEARCH |
| **`report-codex-collaborator` MODE: CONFLICT** | codex-companion（无 `--write`） | 否（只读） | Phase 3 CONFLICT |

> **与检测流水线的边界**：本流水线的两个 agent 文件（`.claude/agents/report-*.md`）与检测流水线的 `codex-report-conflictor` / `codex-plan-conflictor` / `codex-review-conflictor` / `paper-researcher` 完全独立。命名前缀 `report-` 区分用途；触发场景在各自 frontmatter 的 `description` 字段中已声明，不会互相错触。

---

## 2. 流水线（4 阶段）

### Phase 0 — 选题与提纲（一次性）

来源通常是用户口头需求或一份初稿。主会话先：

1. 把用户需求拆成"中心论点 + 章节列表 + 每节核心主张"。
2. 对每条主张做诚实的来源体检：
   - **已知/可考的事实** → 后续 Phase 1 用研究流水线核实最新数据。
   - **数值/百分比/排行** → 必须列入 Phase 1 必查清单。
   - **个人观点/价值判断** → 标记为"作者立场"，不进入研究环节。
3. 输出一个**研究清单**（research checklist），形如：

```text
- claim-A1: Tesla FSD shadow miles 累计数据 (2025-12 截止值)
- claim-A2: Apollo Go 累计单数 vs Waymo 对比 (2025-Q2)
- claim-B1: 自然资源部 2024-07 测绘通知具体编号 + 适用范围
- claim-B2: MIIT 2026-01-01 三项国标编号 + 强制等级
- claim-C1: 主流厂商内部 AI 渗透率统计 (Microsoft/Google/腾讯/字节)
- claim-D1: 2026 主流 AI 订阅与 API 价目（国内外）
```

### Phase 1 — 三路并行研究

主会话**在同一条消息里**同时触发以下三路（`Agent` 工具的多次并发调用）：

```text
1. report-gemini-researcher
   payload = 研究清单的全部条目（或分批），可附"已知不要重复"的上下文
2. report-codex-collaborator MODE: RESEARCH
   同上
3. 主会话自己用 WebSearch / WebFetch
   重点查 Phase 0 标记的"硬数值"
```

并发是关键——三路独立观察 + 主会话整合 = 单一模型偏差被稀释。三路之间**不互相喂结果**（避免 collusion）。

整合产物存在主会话的工作上下文里，或落地为一份临时 markdown（`/tmp/<proposal>-research-notes.md`），不进 git。如果某条 claim 三路都没找到可靠来源 → **必须从草稿中删除或显式标注 unverified**，不允许保留为修辞性陈述。

### Phase 2 — 撰写或重写

主会话在 `docs/proposals/<name>.md` 里直接落笔（中文，技术英文术语行内保留）。声音风格按**建议而非命令**的原则：

| 命令式（避免） | 建议式（采用） |
|---|---|
| 我们建议同时启动两件事 | 这里有两个我们觉得值得管理层考虑的方向 |
| 强制推动全员使用 AI | 想分享一些关于 AI 工具采用的观察，供管理层斟酌 |
| 需要为车队加装相机 | 一种可选的实施路径是为车队加装相机…… |
| 建议方案 | 一种思路 / 我们的看法 |
| 必须 / 应当 | 或可考虑 / 视情况可以 |

**Phase 2 完成的判据**：
- 每条数值都有 Phase 1 找到的可靠来源。
- 没有任何"建议同时启动 / 强制 / 必须 / 应当"等命令式语气残留。
- 每个章节都有"如果管理层有意推进，那么……"或同义句式。
- 末尾保留作者签名格式（`> 提交对象 / 提交人 / 日期`）。

### Phase 3 — 对抗式审稿（CONFLICT 循环）

主会话调度 `report-codex-collaborator MODE: CONFLICT ROUND: 1`，传入：

- `{{DRAFT_OR_PLAN}}`：完整草稿。
- `{{MAIN_SESSION_REASONING}}`：本次稿件的设计理由（为什么这么排版、为什么这么定调、哪些数据已被 Phase 1 三路核实过）。

收到 codex 回复后，按"convergence protocol"读尾行：

- `AGREED:` → 进入 Phase 3 收敛，主会话提交（除非用户另有交代）。
- `STILL DISAGREEING:` → 主会话决定如何回应：
  - **接受**：直接修改草稿，下一轮带 `RESUME: true` + `ROUND: 2` 给 codex 复审。
  - **反驳**：使用 `CONTESTED: <critique-id> — <category>: <one-line>`，类别从 `already-satisfied / technically-wrong / pedagogically-worse / out-of-scope / over-budget / chapter-context` 选一。codex 下一轮**必须先回应被反驳的点**才能引入新意见。

每轮后主会话在 chat 中向用户简报：第几轮、本轮采纳/反驳了哪些、下一轮要去查什么新材料。

**Phase 3 终止条件**（任一）：
- codex 返回 `AGREED:`。
- 连续 2 轮 codex 回复中无新论点（即所有意见都在被反驳后没有新证据）。
- 用户在 chat 中显式叫停（"够了"/"直接交付"）。

---

## 3. 与本仓库其他流水线的隔离

| 资源 | 是否新增 | 影响范围 |
|---|---|---|
| `.claude/agents/report-gemini-researcher.md` | 新增 | gitignored（per-machine），不进 git |
| `.claude/agents/report-codex-collaborator.md` | 新增 | gitignored（per-machine），不进 git |
| `.claude/settings.json`（项目共享） | **不动** | 无 |
| `.claude/settings.local.json`（per-machine） | 可能需要补一条 `Bash(gemini -p:*)` 的 allow（首次调用会触发授权） | 仅本机 |
| `.claude/hooks/` | **不动** | 无；本流水线不挂任何 hook |
| 检测侧 codex conflictors（`codex-plan-conflictor` / `codex-report-conflictor` / `codex-review-conflictor`） | **不动** | 各自仍按 CLAUDE.md 既定职责工作 |
| `paper-researcher` | **不动** | 仍服务 `research/` 目录的学术调研 |
| docs 语种约定 | 沿用 | 提案文档继续用中文（行内保留必要英文术语） |

> **环境变量约定**：
> - `REPORT_PIPELINE_GEMINI_MODEL`（可选）—— 覆盖默认 Gemini 模型。默认值见 `report-gemini-researcher.md` frontmatter 下方说明，预览模型每年都会换名，过期就改这个变量。
> - codex 模型由 agent 内部 pin 在 `gpt-5.5`，与检测侧 conflictor 一致；如要换模型，改 agent 文件本体而非环境变量。

---

## 4. 不在本流水线范围内的事

- **代码生成**：本流水线只产中文叙事文档。任何代码改动走 `superpowers:code-reviewer` (B2) + `codex-review-conflictor`。
- **检测项目轮次报告**：走 `paper-researcher` + B2 + `codex-report-conflictor` + demo-coverage hook，不混用。
- **学术论文**：走 `paper-researcher`，材料落 `research/`。
- **plan 阶段对抗**（`ExitPlanMode` 之前）：走 `codex-plan-conflictor`，不用本流水线的 `report-codex-collaborator MODE: CONFLICT`。

---

## 5. 提交策略

写作过程中**不**自动提交。终稿 + Phase 3 AGREED 之后，主会话在 chat 中向用户提议一次单 commit，列出 staged 文件 + 草拟 message，**等用户一字回复"commit"才执行**——这条规则继承自 CLAUDE.md 的 git policy，本流水线不破例。

---

## 6. 设计来源

完整 7-phase 长篇书籍流水线见 `~/Documents/Projects/book-creator/_workflow/pipeline_design.md`（外部仓库）。本流水线裁掉了：

- per-chapter / per-section 切分（提案是单文件，无需 DAG 调度）。
- writer 分配 + sacrificial worktree（提案不并行多写手）。
- `gemini-researcher` 的"内容风险触发再核实"规则（提案是一次性产出，不需要写作期间反复核实）。
- TOC + chapter overview 维护（提案没有跨章节的目录骨架）。

保留并强化的：

- 三路并行研究（Phase 1）。
- 对抗式 conflict 循环（Phase 3，含 `CONTESTED` 反驳协议）。
- 主会话 = 唯一写盘者（避免 race condition）。
- 命名空间隔离（`report-` 前缀让本流水线与检测流水线零交集）。
