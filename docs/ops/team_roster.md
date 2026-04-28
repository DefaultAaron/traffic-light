# 团队成员名册 — 红绿灯检测项目

> **状态：生效中**（自 2026-04-28 起）。
> 4 个 wshobson 插件已安装（`machine-learning-ops` 1.2.1、`systems-programming` 1.2.2、`comprehensive-review` 1.3.0、`debugging-toolkit` 1.2.0）；
> 4 个新建自定义 agent 已落入 `.claude/agents/`：`codex-plan-conflictor`、`codex-report-conflictor`、`codex-review-conflictor`、`paper-researcher`；
> `demo-reviewer`（已存在）经验证保持完好；
> `superpowers:code-reviewer` 与 `codex:codex-rescue`（已存在）按"主动否决/复用"表中的角色定位接入。
> 本文档自此变为团队正式手册。

## 组建该团队的原因

你将我（主 Claude 会话）任命为项目经理，要求我组建团队。项目自身的特征决定了
约束条件：

- 单人开发（你本人）负责红绿灯检测组件
- 处于 MVP 前研发阶段，目标是为明泰自动驾驶 AGV 系统交付组件
- 多轨道训练（YOLO26 / YOLOv13 / DEIM），共用一份冻结数据集
- 边缘部署到 Jetson AGX Orin（CUDA Linux ARM SoC，**非** Cortex-M）
- 轮次迭代方法论（参见 `~/.claude/plans/elegant-sauteeing-quail.md`）
- 研究驱动：有论文产出意图，因此需要学术严谨性

团队过大变成形式主义；过小则留下盲点。目标是 **12 名专家分 4 个集群 + 我作为
队长**，分工清晰、不重叠。

---

## 集群 A — 实施核心（4 名专家）

负责实际构建：训练、脚本、推理代码、模型导出。来源：`wshobson/agents`
市场（`claude-code-workflows`）。

### A1. ml-engineer

| | |
|---|---|
| **来源** | `machine-learning-ops` 插件 |
| **领域** | PyTorch / Ultralytics / ONNX / TensorRT / 边缘推理 |
| **职责** | 训练轨道（YOLO26、YOLOv13、DEIM）、ONNX 导出、head-strip 脚本、Orin engine 构建 |
| **何时调用** | 启动新训练；新增导出流程；出现延迟 / 精度回归；混合精度调优 |
| **预期输入** | 训练配置、数据集路径、目标 backbone、延迟预算 |
| **预期输出** | 训练完成的 checkpoint + 指标 + 导出的 ONNX/engine + 回归原因分析 |
| **项目关联文件** | `scripts/train_yolov13.sh`, `scripts/train_deim.sh`, `inference/trt_pipeline.py`, `weights/`, R1 报告 Chapter A |

### A2. data-scientist

| | |
|---|---|
| **来源** | `machine-learning-ops` 插件 |
| **领域** | 探索性数据分析（EDA）、统计建模、数据集分析、消融实验设计 |
| **职责** | 类别不平衡分析、数据集轮次对比、评估指标解读 |
| **何时调用** | R-N 范围锁定前（数据集统计）；训练后（每类 recall 分析）；需要做消融时；低样本类（如 `redRight=19`、`greenRight=13`）引发关注时 |
| **预期输入** | 数据集路径或 COCO/YOLO 标注、训练 results CSV、待验证的假设 |
| **预期输出** | notebook 风格分析、图表、统计置信度陈述、数据修复建议 |
| **项目关联文件** | `data/merged/`, `runs/<variant>/results.csv`, R1 报告 §validation, R2 范围扩展记忆 |

### A3. mlops-engineer

| | |
|---|---|
| **来源** | `machine-learning-ops` 插件 |
| **领域** | ML 流水线、实验追踪、模型注册、MLOps 工具栈 |
| **职责** | 何时 / 如何升级到 W&B / MLflow / DVC — 解决方法论计划中的 "defer until" 行 |
| **何时调用** | 实验数量超出文本日志承载；数据集回滚开始变痛；多模型同时部署；CI/CD 流水线变得值得搭建 |
| **预期输入** | 当前痛点（例：「`runs/` 已有 30 个目录，轮次报告难以维护」） |
| **预期输出** | 推荐工具、集成方案、成本-收益分析、迁移步骤 |
| **项目关联文件** | 方法论计划 §"Lightweight MLOps", `runs/`, `weights/` |

### A4. cpp-pro

| | |
|---|---|
| **来源** | `systems-programming` 插件 |
| **领域** | 现代 C++（C++17/20/23）、RAII、模板、性能、嵌入式 Linux C++ |
| **职责** | `inference/cpp/` 中的 C++ 推理流水线；TRT 插件构建（如 DEIM 在 JetPack 5.1 上需要的 `MultiscaleDeformableAttnPlugin_TRT`） |
| **何时调用** | C++ 流水线出现 bug；新模型架构需要 C++ 后处理（DEIM 的 DETR 风格 ≠ YOLO grid-decode）；性能优化；TRT 插件构建失败 |
| **预期输入** | 具体的目标文件或函数；问题描述；engine 规格 |
| **预期输出** | 带原理说明的 C++ 补丁、构建验证、内存安全检查 |
| **项目关联文件** | `inference/cpp/src/trt_pipeline.cpp`, `inference/cpp/CMakeLists.txt`, R1 Chapter A 中的 xyxy bug 修复 |

---

## 集群 B — 质量 / 评审（3 名专家）

不负责构建，只负责审阅他人构建的产物。来源：wshobson。

### B1. architect-review

| | |
|---|---|
| **来源** | `comprehensive-review` 插件 |
| **领域** | 系统设计、架构模式、结构完整性 |
| **职责** | 大方向变更的设计层评审 |
| **何时调用** | 在做出结构性决定之前：单一联合 TL+barrier 模型、检测后时序轨道（TSM 还是 GRU 还是 ByteTrack+EMA）、决策规则结构变更、与 AGV 栈的集成边界 |
| **预期输入** | 拟议设计（文字 + 图示如有）、现有约束、已考虑过的备选方案 |
| **预期输出** | 评判结论 + 2-3 个备选设计排名 + 风险点 + 建议的决策准则 |
| **项目关联文件** | `research/surveys/`, `research/contributions/`, 方法论计划, R2 范围扩展记忆 |

### B2. superpowers:code-reviewer（已安装，无需新装）

| | |
|---|---|
| **来源** | `superpowers` 插件（项目启动时即已安装） |
| **领域** | 阶段完成后的对照评审 — 把完工的代码与"原始计划 + 规范"对齐 |
| **职责** | 在某个轮次步骤完成后（例如某条训练轨道跑完、某个 bug 修复落地），对照本轮 plan / 决策规则 / 项目规范做评审 |
| **何时调用** | 完成轮次中的某个明确步骤后；落地一处非平凡修复（如 R1 的 xyxy 修复）之后；提交涉及多文件的改动到 `main` 之前 |
| **预期输入** | 已完成的步骤 / diff + 对应的 plan 或决策规则引用 |
| **预期输出** | 对照清单：本步骤是否完成所有计划项、是否引入计划外内容、是否符合项目规范 |
| **项目关联文件** | `inference/`, `scripts/`, `.claude/hooks/`, `~/.claude/plans/`, `docs/reports/` |
| **为何选它而不是 `comprehensive-review:code-reviewer`** | 两者均为代码评审，重复。`superpowers:code-reviewer` 的"完工后对照计划"框架与本项目的轮次方法论天然契合（每轮都有预承诺的决策规则）；`comprehensive-review` 那个偏重安全 / 性能 / 可靠性，对本项目（无 API、无服务、无 PII）针对性弱，且 ML 正确性类 bug 由 B3（debugger）+ C3（codex-review-conflictor）覆盖。 |

### B3. debugger

| | |
|---|---|
| **来源** | `debugging-toolkit` 插件 |
| **领域** | 错误、测试失败、异常行为的根因分析 |
| **职责** | "X 为什么坏了" 类调查 |
| **何时调用** | NaN loss；engine 输出垃圾；性能悬崖；测试失败但不暴露原因 |
| **预期输入** | 错误信息 + 调用栈 + 复现步骤 |
| **预期输出** | 定位的根因、最小修复、验证步骤。该 agent 本可加速发现的 bug 类型：R1 的 xyxy 后处理 + engine 输入尺寸双 bug 组合 |
| **项目关联文件** | `logs/` 下的日志、训练 stderr、Orin 运行时错误 |

---

## 集群 C — Codex 反对者 + PM 分解器（4 名包装专家）

来自 codex 插件（GPT 系列模型）的对抗式第二意见。它们的职责是 **质疑** 计划、
报告、代码评审，对冲主会话的确认偏误。每位都是对 codex-rescue 运行时的薄包装，
并预置角色化的提示模板。

> **与已安装的 `codex:codex-rescue` 的关系**：`codex-rescue` 是底层运行时 + 通用
> 转发器，**保留**作为非结构化"急救"的逃生口（例如调试一个我自己卡住的问题、
> 临时让 Codex 跑一段独立调研）。下面 C1/C2/C3 是对**同一**运行时的角色化包装，
> 用于结构化、有特定输出 schema 的场景（计划 / 报告 / 代码评审三类）。两者**互
> 补不重复**：runtime + escape hatch（`codex-rescue`）vs. 纪律性入口（C1-C3）。

### C1. codex-plan-conflictor

| | |
|---|---|
| **来源** | 新建 — 包装 codex-rescue 运行时 |
| **领域** | 计划批准前的对抗式评审 |
| **职责** | 抓出我遗漏的备选架构、隐藏风险、决策规则漏洞 |
| **何时调用** | 任何非平凡计划在 ExitPlanMode 之前；锁定一轮范围之前；提交决策规则之前 |
| **预期输入** | 计划文本 + 决策上下文（约束、被排除的备选方案及原因） |
| **预期输出** | 结构化批评：claim → evidence → counter-evidence → verdict（接受 / 接受但需修订 / 拒绝 + 理由） |
| **项目关联文件** | `~/.claude/plans/`, 方法论计划, R2 范围计划（撰写后） |

### C2. codex-report-conflictor

| | |
|---|---|
| **来源** | 新建 — 包装 codex-rescue 运行时 |
| **领域** | 轮次 / 阶段报告的对抗式评审 |
| **职责** | 抓出无依据的论断、缺失的评估角度、决策规则前后不一致 |
| **何时调用** | `docs/reports/phase_<round>.md` 提交前；下一轮 carry-forward 签字确认前 |
| **预期输入** | 报告草稿 + 数据来源（results CSV、计划中的决策规则、demo-reviewer 笔记） |
| **预期输出** | 按章节的批评：claim → evidence-on-disk reference → counter-evidence → verdict；标注 "无依据" 段落 |
| **项目关联文件** | `docs/reports/phase_2_round_1_report.md`（既有样例） |

### C3. codex-review-conflictor（含脚本评审）

| | |
|---|---|
| **来源** | 新建 — 包装 codex-rescue 运行时 |
| **领域** | 关键变更的代码评审第二意见 |
| **职责** | 抓出 `code-reviewer`（B2）漏掉的问题；起到决断 / 复核作用 |
| **何时调用** | B2 完成评审后，如果变更涉及：后处理（xyxy decode、NMS）、追踪器集成（ByteTrack + EMA voting）、决策规则应用逻辑、hook 网关逻辑、**以及 `scripts/` 下的训练 wrapper / demo sweep / 脚本合并**（2026-04-28 起明确入域）|
| **预期输入** | diff + B2 的评审发现 + 复现或测试用例 |
| **预期输出** | 独立结论（同意 / 不同意 / 补充发现）+ 具体例子 |
| **项目关联文件** | 同 B2；尤其是 inference + tracker 代码路径 |

### C4. codex-pm-planner（新建，2026-04-28）

| | |
|---|---|
| **来源** | 新建自定义 agent — `.claude/agents/codex-pm-planner.md` |
| **领域** | PM 侧的请求分解 — 把用户原始请求转换为"专家管线"调度方案 |
| **职责** | 在主会话碰任何团队成员之前，读取请求 + 项目上下文（CLAUDE.md / team_roster / 当前 plan / `git log`），输出结构化方案：目标 / 约束 / 证据收集 / 调度顺序 / 对抗回路触发点 / 范围围栏 / 验收门 |
| **何时调用** | 非平凡请求触发 ≥2 个专家域 / 单域但高利害（训练、导出、推理 / 后处理 / tracker 改动、hook gating、决策规则改动）/ 含模糊范围词（"清理"、"重构"、"审查全部"）/ 触发对抗回路（plan → C1；report → C2；critical code → B2 + C3）|
| **何时跳过** | 平凡请求（commit / 状态查询 / 单文件明确改动）；用户已显式指定 agent 与顺序；纯信息查询无落盘副作用 |
| **预期输入** | 用户原始请求（逐字）、当前 plan 路径（如有，由 dispatcher 显式传入而非硬编码）、最近 commit 概览、用户当轮显式约束 |
| **预期输出** | FULL schema（≥3 调度 / 触发对抗回路 / 多领域）：Goal / Constraints / Evidence / Pipeline / Adversarial fires / Out-of-scope / Verification；COMPACT schema（≤2 调度且单域无对抗）：Goal / Dispatches / Out-of-scope / Verification |
| **项目关联文件** | `CLAUDE.md`, `docs/ops/team_roster.md`, 当前 `~/.claude/plans/<id>.md`, 既有 codex-* conflictor 体例 |
| **关键约束** | 1) 不递归调度自己；不在生成的 pipeline 里调度 C4；2) 不杜撰事实 / 不发明 agent / 不分配越权写权限；3) 不扩展用户的 deliverable；4) 输出仅作主会话顾问参考，不作为 plan 走 C1 二次评审；5) round/phase report 走 C2（仅当报告引用代码时才追加 B2），别把 B2 默认拉进来 |
| **写权限** | **无**。frontmatter 仅 `tools: Bash`；通过 `codex-companion.mjs` 转发，禁用 `--write`。所有方案以响应文本形式返回，由主会话决定是否执行 |

**关于 codex agents 的说明**：它们通过 `codex-rescue` 运行时调用 `codex-companion.mjs`。
输出风格和推理强度（effort）是路由控制；包装器会强制把模式钉在 "评审 / 只读"
（不带 `--write`），这样反对者绝不可能改文件，只能批评。C4 codex-pm-planner 同样
read-only，但角色不同：它在主会话**调度团队之前**做请求分解，不在事后做对抗式评审。

---

## 集群 D — 自定义（4 名专家）

### D1. demo-reviewer

| | |
|---|---|
| **来源** | 已存在 — `.claude/agents/demo-reviewer.md` |
| **领域** | 生成的 demo 视频的定性评审 |
| **职责** | Q1-Q4 质量轴（正确性 / 漏检 / 平滑性 / 其他）、单视频笔记、滚动汇总、持久化 ledger |
| **何时调用** | 跑完 `scripts/run_demos.sh` 后；以及每次写阶段报告前由 `.claude/hooks/check_demo_coverage.py` 自动触发 |
| **预期输入** | 待评审的具体 (run × engine) 组（由主会话根据覆盖率网关传入） |
| **预期输出** | 每视频笔记、ledger 更新、汇总增量 |
| **项目关联文件** | `demo/`, `demo/_review/`, `.claude/hooks/check_demo_coverage.py` |
| **状态** | 已接入并生效；无需变动 |

### D2. paper-researcher（新建）

| | |
|---|---|
| **来源** | 新建自定义 agent |
| **领域** | 学术 CV/ML 文献调研、相关工作梳理、论文写作伙伴 |
| **职责** | 找相关论文、组织 related-work 章节、批评消融实验表的完整性、引用规范、起草论文片段 |
| **何时调用** | 考虑某架构选择 → "前人怎么做？"；设计消融 → "覆盖完整吗？"；写论文章节 → "起草并附引用"；投稿前 → "引用是否完整" |
| **预期输入** | 主题 / 问题 / 草稿章节 + 拟投会刊风格（如 CVPR vs IROS vs IV） |
| **预期输出** | 加注释的参考文献（尽量带 BibTeX）、带行内引用的章节草稿、消融表批评、缺口分析 |
| **项目关联文件** | `research/surveys/alt_detector_architectures.md`（既有的调研体例 — paper-researcher 沿用）；未来的 `docs/paper/` |
| **关键约束** | paper-researcher 是 **协作伙伴**，不是预言机。它读不到没给它的论文；它必须引用 **具体的论文**（带年份 + 会刊），不能编造结果。该角色最大的风险是幻觉，需要在提示词中明确约束。 |
| **写权限** | **无**。frontmatter 仅授予 `Read` / `WebSearch` / `WebFetch` / `Glob` / `Grep`，未授予 `Write` / `Edit` / `Bash`。所有产出以响应文本形式返回，由主会话决定是否落盘以及落盘到哪个路径。这是工具网关层的强制约束，不是 prose 层面的"请勿"。 |

### D3. doc-manager（新建，2026-04-28）

| | |
|---|---|
| **来源** | 新建自定义 agent — `.claude/agents/doc-manager.md` |
| **领域** | `docs/` 与 `research/` 树的撰写 / 重构 / 重命名 / 跨引用审计 / 语言合规 |
| **职责** | 文件夹重组、过期文档精简、跨引用修复、语言规则强制（`docs/` 中文 + READMEs 英文）；与 `script-maintainer` 互补：脚本变更引发的多文档改写由其承接 |
| **何时调用** | 文件夹重命名 / 大规模移动；"清理过期文档"；论文资料与执行文档分家；rename / delete 后的批量交叉引用更新；语言合规审计；新文档草稿 |
| **预期输入** | 明确任务（"将 X 移到 Y 并更新所有引用" / "把过期事故记录精简为 runbook"），约束（目标长度、必保留章节、必删章节），是否涉及语言切换 |
| **预期输出** | 结构化交付清单：触动文件 + 跨引用 grep 结果（必须 clean）+ 语言审计 + 范围外项目 |
| **项目关联文件** | `docs/`, `research/`, 任意 `README.md`；现有英文例外（`docs/ops/{scripts_reference,tailscale_runbook}.md`）保留英文 |
| **写权限** | `docs/`, `research/`, root `README.md`。**不可触碰**：`inference/`, `scripts/`, `main.py`, `configs/`, `.claude/`, hooks, `pyproject.toml`, ignore 文件, agent-tool 配置（`CLAUDE.md` 等）|
| **关键约束** | 1) 语言规则零例外（参见上方"项目关联文件"）；2) 任何 rename / move 后必须 grep 项目范围内所有 `*.md` + 源代码注释、确认 0 残留；3) 用 `git mv` 保留历史；4) 范围外任务必须以 `out-of-scope:` 形式回退，不要自行越界 |

### D4. script-maintainer（新建，2026-04-28）

| | |
|---|---|
| **来源** | 新建自定义 agent — `.claude/agents/script-maintainer.md` |
| **领域** | `scripts/` 树（shell + Python）：训练 wrapper、数据集转换、demo sweep、模型导出、验证工具、ops 辅助脚本 |
| **职责** | 重复脚本合并、过期脚本删除、env-var 契约调优、新脚本撰写；保持 `docs/ops/scripts_reference.md` 同步；遵守 `exec` trainer + SEED.txt 项目约定 |
| **何时调用** | 合并 `scripts/foo.sh` 与 `scripts/foo_variant.sh` 这类重复；删除过期脚本；新增训练 / 数据 / demo 脚本；现有脚本契约重构 |
| **预期输入** | 明确任务、env-var 契约必保留 / 可调整、输出目录约定、触发上下文（哪次代码 / 文档变更带出此需求）|
| **预期输出** | 结构化交付清单：触动脚本 + `bash -n` / `py_compile` 验证 + `chmod +x` 验证 + scripts_reference 同步状态 + 跨引用 grep 结果 + codex stop-gate 暴露面（exec 是否守住、SEED 与 `"$@"` 是否一致、是否引入 masking 风险）+ 范围外项目 |
| **项目关联文件** | `scripts/`, `docs/ops/scripts_reference.md`（编辑），`feedback_codex_conflictor_pattern.md`（必读，Pattern B）|
| **写权限** | `scripts/`, `docs/ops/scripts_reference.md`。**不可触碰**：`inference/`, `main.py`, `configs/`, training YAML, hooks, agent files, `pyproject.toml` |
| **关键约束** | 1) 训练 wrapper 必须 `exec` 调用 trainer（无 post-exec 行、无 `|| true`、无 safety net）；2) SEED.txt 在 `exec` 之前预写、且从 `"$@"` 提升 seed override；3) resume 路径不写 SEED；4) 每个 shell 脚本 `bash -n` 通过、Python `py_compile` 通过才算完成；5) 新 shell 脚本一律 `chmod +x`；6) 合并脚本前先列出双方所有 env-var 默认值，证明合并版本无静默 default 变化 |

---

## 队长

- **你（人类 PM）+ 我（主会话）**：项目管理 / 编排。我负责调度专家、综合产出、
  把决策点上报给你。按你的决定，**不再增设助理 PM agent**。

---

## 主动否决的人选

| 候选 | 否决原因 |
|---|---|
| `arm-cortex-expert` | Cortex-M MCU（Teensy/STM32）专长。Jetson Orin 是 Linux ARM SoC + CUDA — 领域不匹配。 |
| `python-pro` | 与主会话的 Python 能力以及 `ml-engineer` 重叠。如有需要按需召唤即可。 |
| `ai-engineer` | 聚焦 LLM/RAG/agent，不是 CV 检测。 |
| `team-lead`（`agent-teams`） | 编排我已在做；引入它会形成回环。 |
| `tdd-orchestrator` | 训练轮次是经验性的，不是 TDD 形态的工作。 |
| 重复的 `code-reviewer`（在 `code-refactoring` / `tdd-workflows` 中） | 与已安装的 `superpowers:code-reviewer` 重复，不引入。 |
| `comprehensive-review:code-reviewer` | 与 `superpowers:code-reviewer` 重复；作为 `comprehensive-review` 插件的副产物会被一同安装，但不进入主流程，按需备用。 |
| `comprehensive-review:security-auditor` | 与 `code-reviewer` 一同被插件捎带安装。本项目当前阶段无显著安全面（无 API / 无服务 / 无 PII），不进入主流程；如未来引入云端组件再启用。 |
| `monorepo-architect` | 本项目不是 monorepo。 |
| `bash-pro` / `posix-shell-pro` | 训练脚本简单，由 `ml-engineer` 兼顾即可。 |
| `codex-pm-deputy`（助理 PM） | 按你的决定。如果 PM 工作量增长，再考虑。 |

---

## 团队工作机制（协议）

### 汇报结构

```
                     ┌──────────────────────┐
                     │   你（人类 PM）      │
                     └──────────┬───────────┘
                                │
                     ┌──────────▼───────────┐
                     │ 我（主会话，队长）   │
                     └─┬───┬───┬───┬───┬────┘
                       │   │   │   │   │
   ┌───────────────────┘   │   │   │   └────────────────────┐
   ▼                       ▼   ▼   ▼                        ▼
┌──────────┐  ┌──────────┐ ┌─────────┐ ┌────────────────────────────────────┐
│ 集群 A   │  │ 集群 B   │ │ Codex C │ │ 集群 D（自定义）                   │
│ (构建)   │  │ (评审)   │ │(对抗)   │ │ demo-reviewer / paper-researcher / │
│          │  │          │ │         │ │ doc-manager / script-maintainer    │
└──────────┘  └──────────┘ └─────────┘ └────────────────────────────────────┘
```

### 对抗回路（codex 反对者）

对高利害交付物，跑 3 步循环：

1. 集群 A 产出工作 → 集群 B 评审 → 我整合
2. 集群 C（角色对应的那位）拿到整合产出做对抗式二轮
3. 冲突上报你裁决；一致则放行

具体而言：
- 计划 → 我起草 → C1 反对 → 你通过 ExitPlanMode 批准
- 阶段报告 → 我起草 → 涉及代码引用时先 B2，再 C2 → 你签字
- 关键代码变更 → A 集群构建 → B2 评审 → C3 二轮 → 我合并

### 不需要走完整回路的情况

- 平凡修复 / 拼写 / 仅文档变更 — 主会话独自处理
- 簿记类（demo-reviewer 在自己的 scope 内自治）
- 纯文献查询（paper-researcher 独立处理）
- 快速重构 — A 集群 + B 集群，跳过 C

### 工具权限策略（初版）

- A 集群：写权限（要构建代码）
- B 集群：只读 + 评论（不自动改；只给建议）
- C 集群：**严格只读** — codex 包装器禁用 `--write`
- D 集群：受限写权限
  - `demo-reviewer`：仅限 `demo/_review/`（由 body scope + 现有 ledger 机制约束）
  - `paper-researcher`：**完全无写权限** — frontmatter 不授予 `Write` / `Edit` / `Bash`
    任一工具，由 Claude Code 工具网关层强制执行（不依赖 prose 自律）。
    草稿通过响应文本返回；保存到 `docs/paper/` 等路径由主会话执行。
  - `doc-manager`：写权限限于 `docs/`、`research/`、根 `README.md`；不可触碰代码 / 配置 / 脚本 / hooks / agent files
  - `script-maintainer`：写权限限于 `scripts/` 与 `docs/ops/scripts_reference.md`；不可触碰 inference / main.py / configs / training YAML / hooks / agent files

---

## 组建该团队需要做的事

### 第 1 步 — 你安装 4 个 wshobson 插件
```
/plugin install machine-learning-ops@claude-code-workflows
/plugin install systems-programming@claude-code-workflows
/plugin install comprehensive-review@claude-code-workflows
/plugin install debugging-toolkit@claude-code-workflows
/reload-plugins
```
这会引入：
- A1 `ml-engineer`、A2 `data-scientist`、A3 `mlops-engineer`（来自 `machine-learning-ops`）
- A4 `cpp-pro`（来自 `systems-programming`，同插件还会捎带 `c-pro` / `golang-pro` / `rust-pro`，本项目不进入主流程）
- B1 `architect-review`（来自 `comprehensive-review`，同插件还会捎带 `code-reviewer` / `security-auditor`，二者均不进入主流程，理由见上方"主动否决"表）
- B3 `debugger`（来自 `debugging-toolkit`，同插件还会捎带 `dx-optimizer`，按需）

**B2 不需要新装** — 直接复用已存在的 `superpowers:code-reviewer`。

### 第 2 步 — 我创建 4 个新的 agent 文件
- `.claude/agents/codex-plan-conflictor.md`（C1）
- `.claude/agents/codex-report-conflictor.md`（C2）
- `.claude/agents/codex-review-conflictor.md`（C3）
- `.claude/agents/paper-researcher.md`（D2）

每个反对者：薄薄的 Bash 包装层调 `codex-companion.mjs`，前置角色化提示模板；
禁用 `--write`；输出结构化 schema。

paper-researcher：完整内容型 agent（不走 codex 包装），明确禁止幻觉，强制
BibTeX 输出规范。

### 第 3 步 — 验证 demo-reviewer（D1）在 `/reload-plugins` 后仍能加载

### 第 4 步 — 把本文档顶部横幅从 "方案" 改为 "生效中"

### 第 5 步 — 提交 `docs/ops/team_roster.md` 让团队归入 git 追踪

---

## 验收 — 如何确认团队已组建

- [ ] 4 个 wshobson 插件在 `/plugin` 列表中显示为已安装
- [ ] Agent picker（`Agent` 工具的 `subagent_type`）接受：`ml-engineer`、`data-scientist`、`mlops-engineer`、`cpp-pro`、`architect-review`、`superpowers:code-reviewer`（已存在）、`debugger`、`codex-plan-conflictor`、`codex-report-conflictor`、`codex-review-conflictor`、`paper-researcher`、`demo-reviewer`、`codex:codex-rescue`（保留作逃生口）
- [ ] 烟测：对每个新建的自定义 agent（4 个）发一条单行探针提示，确认能拿到结构化输出
- [ ] 本文件顶部横幅显示 "状态：生效中"（无 PROPOSAL 横幅）

---

## 不在 "组建团队" 范围内的事项

- 实际派发给团队成员的工作 — 团队就位后按轮次开展
- 自定义编排层 — 我直接通过 Agent 工具编排
- 单 agent 细粒度工具权限 — 留到出现真实冲突再定
- 替换 demo-reviewer — 现状可用
- 在 4 个之外再加插件 — 留到感受到缺口再说
