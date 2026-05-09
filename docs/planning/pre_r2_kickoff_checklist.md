# R2 启动前 Kickoff Checklist

> v1.4 LOCKED（2026-05-09，plan-conflictor 3-iter 收敛完成：iter-1 12 finds + iter-2 8 finds + iter-3 1 find，全部 amend 并 AGREED）。
> **不可变锚点**（每次提交前 §6 一致性 check 必须验证存在）：
> - `~/.claude/plans/elegant-sauteeing-quail.md`，lock_id=`LOCK-iter-3`，证据：plan 文件含 `## Conflictor-loop termination (LOCK after iter 3)` 标题
> - `docs/planning/additional_components_plan.md`，lock_id=`v1.0-AGREED`，证据：frontmatter 或文首版本行匹配
> - `docs/planning/temporal_optimization_plan.md`，lock_id=`v1.1-AGREED`，证据：§1.1 v1.1 amendment 标题存在
> 每个**可执行**主任务遵循 5-stage 生命周期（a. Scaffold → b. Impl → c. Ablation/Test → d. Decision → e. Report）。
> **生命周期适用规则**：
> - stub-only 项（如 §2.4 carry-forward stubs）使用 reduced 生命周期：仅 a（stub） + e（指针登记）；b/c/d 标 `n/a`。
> - blocked 阶段（依赖外部 trigger）标 `blocked` + 注明 unblocker，**不**作为未勾选 actionable 项；R2 close 时计入 carry-forward 而非 incomplete。
> - 决策类 d 阶段（deploy/defer/drop）必须落地为结构化记录（见 §1.2/§1.3/§1.4 / §2.1）；单 checkbox 不构成决策证据。
>
> 本文件不复制已 LOCK 计划内容；仅清单 + 指针。

---

## 0. 外部阻塞（不在清单 actionable 范围）

| # | 等待对象 | 解锁谁 |
|---|---|---|
| 0.1 | R2 数据 freeze | §1 全部 |
| 0.2 | DEIM-D-FINE-L 训练 | §七 KD teacher 选定 |
| 0.3 | 5/12 GPS 决议 | §1.4 §五 实地 vs replay-only 分支 |
| 0.4 | 在车 replay 出 small/far/occluded miss | §3.1 / §3.2 |
| 0.5 | autonomy / planning 团队对接 | §3.3 / §3.6 |

---

## 1. R2 启动前必须（数据 freeze 触发）

### 1.0 共享 schema（在 §1.1+ / §2.1+ b-stage 引用前必须落地；data-independent，可立刻启动）

- [ ] **a. Scaffold**
  - [ ] `scripts/_r2_component_decision_schema.json`：§1.2/§1.3/§1.4 d-stage 写入的 `runs/_r2_component_decisions.json` schema
    - 形状：array，按 `component` 唯一；字段 `{component, outcome ∈ {deploy, defer, drop}, reason, blocking_artifacts: [path|sha256], next_round_action, branch?}`；非-deploy 时 reason + next_round_action 必填；map-prior 必填 `branch ∈ {live, replay_only}`
  - [ ] `scripts/_r2_audit_coverage_schema.json`：§1.1 audit 子集输出 schema
    - 形状：array，per-class `{class_id, class_name, full_val_support, full_val_insufficient, audit_support, audit_coverage_status ∈ {covered, low_power, construction_failed}, audit_low_power, construction_failure_reason?}`
  - [ ] `scripts/_r2_carry_forward_schema.json`：§4.2 carry-forward 登记 schema
    - 形状：array，`{item_id, status ∈ {blocked, scheduled}, blocked_on: [closed_enum], unblock_logic? ∈ {all, any}, unblock_evidence_path?, next_entrypoint}`
    - 规则：`status="blocked"` → `blocked_on` `minItems: 1`；多 token 默认 `unblock_logic="all"`，OR 关系必须显式 `"any"`；`status="scheduled"` → `blocked_on=[]` AND **禁止** `unblock_logic`
    - `blocked_on` closed enum：`{r2_data_freeze, deim_l_training, on_vehicle_replay_failure_modes, hard_neg_manifest_hash, autonomy_team, planning_team, gps_topic_5_12, sahi_b_c_measured, tsm_phase_1b_passed, replay_temporal_flicker_or_state_confusion}`
- [ ] **b. Impl** — 仅 schema 文件，无逻辑代码
- [ ] **c. Test** — 三 schema 通过 JSON Schema draft-07 自检；负向 fixture 各一（缺必填 / 错枚举 / outcome=defer 但 reason 缺失 → 全 fail）
- [ ] **d. Decision** — B2 + C3 loop AGREED，逐 schema 登记：
  - [ ] `_r2_component_decision_schema.json` — B2 ✓ / C3 ✓ / unresolved=0
  - [ ] `_r2_audit_coverage_schema.json` — B2 ✓ / C3 ✓ / unresolved=0
  - [ ] `_r2_carry_forward_schema.json` — B2 ✓ / C3 ✓ / unresolved=0
- [ ] **e. Report** — 三 schema 路径写入 `runs/_r2_verification.json`

### 1.1 R2 验证集 freeze + 分层 audit subset

- [ ] **a. Scaffold** — manifest schema + 采样规则定稿；schema 文件 + 采样 code 也 hash 入 manifest（防 schema drift gaming）
- [ ] **b. Impl**
  - [ ] `runs/_r2_val_manifest.txt` + `.sha256`
  - [ ] `runs/_r2_eval_parity/sample_manifest.json`（每类 ≥ 1，每 scenario tag ≥ 3；class-ID 映射 + scenario taxonomy 冻结于 manifest schema）
  - [ ] **`runs/_r2_audit_coverage.json` + `.sha256`**：per-class records 满足 §1.0 `_r2_audit_coverage_schema.json`（`full_val_support` / `full_val_insufficient` / `audit_support` / `audit_coverage_status` / `audit_low_power` / 可选 `construction_failure_reason`）；audit subset 构造目标：per-class ≥ 5 安全类 instance
  - [ ] **传播规则**（§2.1 b 强制消费）：`audit_coverage_status == "low_power"` → `_r2_decide_precision.py` 写 `audit_low_power=true` + confidence-downgrade；`construction_failed` → escape outcome（不参与 ship_precision 决定，且报告必须显式标该 detector 为 degraded-confidence，不可静默忽略）
- [ ] **c. Test** — image hash 自洽；分层覆盖断言通过；audit_coverage.json 通过 §1.0 schema 校验；负向测试：篡改 schema 文件后 hash 校验必须失败
- [ ] **d. Decision** — manifest + schema + sampling-code 三 hash 一旦定稿即冻结；任意改动 = 重跑全部 parity
- [ ] **e. Report** — manifest 路径 + 三 hash + `runs/_r2_audit_coverage.json` 路径与 sha256 写入 `runs/_r2_verification.json`

### 1.2 §三 Copy-paste + class-balanced loss（参 `additional_components_plan.md` §三）

- [ ] **a. Scaffold** — 数据增强 hook + class-balanced loss 接入点
- [ ] **b. Impl** — rare-class instance copy-paste + FP ceiling 监控（FP 分母与 class 分组在 §三 锁定）
- [ ] **c. Ablation** — 开/关 + 强度 sweep；rare-class FP ceiling vs ≥−1 pp 安全 floor（per-class 而非 aggregate）
- [ ] **d. Decision** — 写入 `runs/_r2_component_decisions.json`，符合 §1.0 `_r2_component_decision_schema.json`；非-deploy 时 reason + next_round_action 必填；`blocking_artifacts` 必须为 path|sha256 引用，禁止自由 prose
- [ ] **e. Report** — `phase_R2.md` §三 子节；defer/drop 时附 evidence 表（消融数据 + 阻塞工件）

### 1.3 §四 Hard-negative mining（参 §四）

- [ ] **a. Scaffold** — mining script + 候选池采样规则
- [ ] **b. Impl**
  - [ ] 基于 baseline 出 hard-neg pool
  - [ ] frozen `runs/_hard_negative_eval_manifest.json` + `.sha256`（与 §七.6 KD acceptance gate #3 共享；§2.2.1 必须 hash-pin 引用此 manifest）
- [ ] **c. Ablation** — 注入比例 sweep；**per-class** 真实灯 recall 下限 ≥ −0.5 pp（aggregate 不足以代理）
- [ ] **d. Decision** — 写入 `runs/_r2_component_decisions.json`，符合 §1.0 schema（同 §1.2 d 规则）
- [ ] **e. Report** — `phase_R2.md` §四 子节；defer/drop 时附消融证据

### 1.4 §五 Map-prior gating（参 §五；by 5/15）

- [ ] **a. Scaffold** — GPS / map service 接口 + replay-only fallback 接口
- [ ] **b. Impl** — gating 决策点 + 三类负控（施工灯 / GPS jitter / missing maps）
- [ ] **c. Ablation** — 开/关 + 三负控；**per-class 最小样本 N 锁定于 `runs/_map_prior_negative_controls.json`**（每负控类 ≥ N instance；N 在 a-stage 定稿）；silent miss 定义：相对 detector-only baseline 仅由 gating 引入的 per-class false negative；要求 zero
- [ ] **d. Decision** — 5/12 GPS 决议未到 → 切 replay-only 分支；写入 `runs/_r2_component_decisions.json` 符合 §1.0 schema；`branch ∈ {live, replay_only}` 必填
- [ ] **e. Report** — `phase_R2.md` §五 子节

---

## 2. 等待期并行任务

> **诚实性修订**：仅 §2.1.1 / §2.2.1 / §2.3 a-stage 真正"数据无关、可立刻启动"。所有 b/c 阶段中"消费 §1.1 audit / val manifest"的步骤均被数据 freeze gate 住，标 `blocked_on=r2_data_freeze`。

### 2.1 R2 Precision Parity 工程（参 `elegant-sauteeing-quail`）

> sensitivity sweep 参数网格、阈值默认值、Case 求值顺序均在 `elegant-sauteeing-quail` LOCK iter-3 中冻结；本节不复制数值，仅做执行清单。

#### 2.1.1 即时 plumbing（数据无关，可立刻启动）

- [ ] **a. Scaffold**
  - [ ] `scripts/_r2_decision_schema.json`（input + output 形状；input 必含 §1.1 audit `audit_coverage_status` 字段 + `$ref` 到 §1.0 `_r2_audit_coverage_schema.json`）
  - [ ] `scripts/_r2_decide_precision.py` 接口框架（无逻辑）
  - [ ] `scripts/_r2_verify.py` 接口框架（scope-clamp 注释强制）
- [ ] **b. Impl（数据无关部分）**
  - [ ] verify：仅 schema / path / hash 校验；**严禁** 阈值分支 / case 分类 / delta 计算
  - [ ] verify 增加 §4.1 soak SHA 比对：当 `runs/_r2_orin_soak_records.json` 含 entry 时，hard-fail 当 `engine_sha256 != selected_artifact_sha256`（仍是纯比对，不是决策逻辑）
  - [ ] `inference/cpp/src/demo.cpp:271-278` 计时拆分（t_detect / t_track；log 行追加 `(detect=Xms, track=Yms)`）
  - [ ] build-determinism 双构建脚本骨架（YOLO + DEIM 各一；实跑落 §2.1.2）
  - [ ] mixed-precision pipeline 加载验证（按路径加载 `best_fp32.engine`）
- [ ] **c. Test（数据无关）** — schema 自检通过；verify 单元负向（path 不存在 / hash mismatch / soak SHA mismatch 均 hard-fail）；demo.cpp 编译 + log 解析 fixture
- [ ] **d. Decision** — B2 + C3 loop AGREED，**逐工件登记**：
  - [ ] `_r2_decide_precision.py` — B2 ✓ / C3 ✓ / unresolved=0
  - [ ] `_r2_decision_schema.json` — B2 ✓ / C3 ✓ / unresolved=0
  - [ ] `_r2_verify.py` — B2 ✓ / C3 ✓ / unresolved=0（含 scope-clamp 强制 + soak SHA 比对纯校验性 review brief）
  - [ ] `demo.cpp` 计时拆分 — B2 ✓ / C3 ✓ / unresolved=0（scoped brief：无行为变更 / 无 timer 双计 / log 可解析）
  - [ ] `export_yolo.sh` + `export_deim.sh` sidecar 字段改动 — B2 ✓ / C3 ✓ / unresolved=0（critical-path；与 §2.2.1 + §2.3 同批次）
- [ ] **e. Report** — 各工件 transcript 路径写入 `runs/_r2_verification.json`

#### 2.1.2 数据 freeze 后执行（blocked_on=r2_data_freeze）

- [ ] **b. Impl（freeze-blocked）**
  - [ ] decide_precision：按 LOCKED 计划实现 Case C → A → D → B；exactly-one outcome；inconclusive_global / audit_disagreement / executor_error 三 escape；sensitivity sweep（参数网格按 LOCKED 计划读取）
  - [ ] decide_precision 必须 hash-pin 消费 `runs/_r2_audit_coverage.json` + `.sha256`：`construction_failed` 触发 escape；`low_power` 写 confidence-downgrade
  - [ ] FP32 engine × 4 候选 + sidecar（atomic 协议已就绪）
  - [ ] build-determinism 双构建实跑；超阈分支显式标 `pre_committed_defer_outcome="r2_fp16_default"` + root-cause classification
- [ ] **c. Test**
  - [ ] eval-parity gate：per-image IoU 0.99 + score ±0.005；mAP@0.5 / @0.5:0.95 ≤ 0.01 pp；class array 顺序一致
  - [ ] eval-parity 失败分类（B2 调用；implementation bug vs expected numeric divergence；**严禁** silent 放宽）
  - [ ] build-variance 阈值（按 LOCKED 计划）；超阈写 `pre_committed_defer_outcome` + root-cause
  - [ ] FP32 demo 完整跑（peak GPU mem / first-frame / median t_detect_ms）
- [ ] **e. Report** — `phase_R2.md` 头表 4 行 + appendix；`runs/_r2_precision_decisions.json` 写入

### 2.2 §七 KD a2 — 拆分为「即时 plumbing」+「blocked 决策执行」

> **诚实性修订**：原 §2 "数据无关、可立刻启动" 仅对 §2.2.1 plumbing 成立；§2.2.2 KD 首 cell 决策同时被三件事 block：(a) R2 student/teacher 选定（依赖 §2.1）、(b) DEIM-D-FINE-L 训练完成（§0.2）、(c) §1.3 hard-neg manifest 冻结（KD acceptance gate #3 共享 hash）。

#### 2.2.1 KD 即时 plumbing（数据无关，可立刻启动）

- [ ] **a. Scaffold**
  - [ ] `scripts/_kd_decision_schema.json`
  - [ ] `scripts/_kd_decide_cell.py` 接口框架
- [ ] **b. Impl**
  - [ ] decide_cell 接口：5 acceptance gate 求值；seed5 强制（ship-decision 路径）；**gate #3 输入必须按路径 + sha256 hash-pin 引用 §1.3 `runs/_hard_negative_eval_manifest.json`**，hash mismatch 或缺失 → 硬失败
  - [ ] `scripts/export_yolo.sh` + `scripts/export_deim.sh` sidecar 加 3 KD 字段（`kd_enabled` / `kd_teacher_path` / `kd_temperature`）
- [ ] **c. Test** — sidecar size-stability 协议保留；`engine_sha256` 流程重测；`bash -n` 通过
- [ ] **d. Decision** — B2 + C3 loop AGREED，**逐工件登记**：
  - [ ] `_kd_decide_cell.py` — B2 ✓ / C3 ✓ / unresolved=0
  - [ ] `_kd_decision_schema.json` — B2 ✓ / C3 ✓ / unresolved=0
  - [ ] `export_yolo.sh` + `export_deim.sh` KD sidecar 字段改动 — B2 ✓ / C3 ✓ / unresolved=0（critical-path；与 §2.1.1 + §2.3 同批次审）
- [ ] **e. Report** — transcript 路径登记于 `runs/_r2_verification.json`

#### 2.2.2 KD 首 cell 决策执行（blocked；R2 close 计入 carry-forward）

- [ ] **blocked**：等待 (a) R2 student/teacher 选定、(b) `runs/_hard_negative_eval_manifest.json` hash 冻结、(c) DEIM-D-FINE-L 训练完成
- [ ] 解锁后入口：`additional_components_plan.md` §七.6 起跳；本 checklist 不追踪后续 a-e

### 2.3 TSM Phase 1-C 解锁（参 `temporal_optimization_plan.md` §1.5.4）

- [ ] **a. Scaffold** — `scripts/_tsm_decide_phase.py` 接口框架（schema 已就绪：`_tsm_activation_schema.json` v1.1）
- [ ] **b. Impl**
  - [ ] decide_phase：7-branch tree（deploy / INT8 defer / 1-C export defer / 4 drop incl. activation-bucket mismatch）
  - [ ] **deploy 路径硬失败条件**（三全否则 fail）：`pretrained_init_mode == "scratch"` AND `deploy_eligible == true` AND `seed == 5`（ship-decision 路径强制 seed5，与 §2.2 KD 同条款）
  - [ ] `runs/_tsm_decisions.json`（plural）写入
  - [ ] export_*.sh sidecar 加 4 TSM 字段（`tsm_enabled` / `tsm_shift_fraction` / `tsm_clip_size_train` / `tsm_feature_cache_stages`），与 §2.2.1 KD 字段同批次
- [ ] **c. Test** — 与 §2.2.1 共用 sidecar 测试套；**deploy-path negative test 矩阵**（每条独立 fail，不可任一 short-circuit）：
  - [ ] fixture-1：`pretrained_init_mode="pretrained_diagnostic"` + 其他两条 valid → deploy 必须 fail
  - [ ] fixture-2：`deploy_eligible=false` + 其他两条 valid → deploy 必须 fail
  - [ ] fixture-3：`seed=0`（任一 != 5） + 其他两条 valid → deploy 必须 fail
  - [ ] fixture-positive：三条全 valid → deploy 通过
- [ ] **d. Decision** — B2 + C3 loop AGREED，**逐工件登记**：
  - [ ] `_tsm_decide_phase.py` — B2 ✓ / C3 ✓ / unresolved=0
  - [ ] `export_yolo.sh` + `export_deim.sh` TSM sidecar 字段改动 — B2 ✓ / C3 ✓ / unresolved=0（critical-path；与 §2.1.1 + §2.2.1 同批次审）
- [ ] **e. Report** — transcript 路径登记；Phase 1-A / 1-B 仍由 §0.4 在车 replay trigger 控制

### 2.4 Carry-forward stub（reduced 生命周期：a + e；b/c/d 标 n/a）

- [ ] **a. Stub** — 文件存在 + 必备字段
  - [ ] `docs/planning/R3_precision_reproducibility.md`：trigger（build-variance defer OR inconclusive-global） + 阻塞决策 + 所需 held-out 数据 + success criteria
  - [ ] `docs/planning/pre_deploy_AGV_integration.md`：接收 Case A FP32 ship；5-min Orin soak；AGV-loop latency budget
- [ ] **e. Report** — stub 路径登记于 `runs/_r2_verification.json`；`phase_R2.md` "Carry-forward stubs created" 节列出

---

## 3. 历史 hang-up（gated；非 actionable）

| # | 任务 | Gate | 解锁入口 |
|---|---|---|---|
| 3.1 | TSM Phase 1-A | R2 主选 detector + 在车 replay 出 small/far/occluded miss | `components/temporal_shift_module/runners/concept_validation.py` |
| 3.2 | HMM scaffold | §0.2 row 2/3 失败模式涌现 | `temporal_optimization_plan.md` §2.2 |
| 3.3 | §八 Multi-camera fusion | autonomy 团队多相机标定 + 时序对齐 | §八 a-stage |
| 3.4 | §九 Adaptive inference / ROI | R2 latency 预算 release | §九 a-stage |
| 3.5 | §十 INT8 QAT | **measured** SAHI latency from §3.7 §六 b/c stage 在 [50, 80) ms OR TSM Phase 1-B 通过且 measured latency 在 [26, 35] ms（latency 来源必须是已运行的实测，不是估计） | §十 a-stage |
| 3.6 | §十一 Planner-prior late fusion | planning 团队接口 + cross_detection_reasoning_plan 扩展 | §十一 a-stage |
| 3.7 | §六 SAHI（inference-only） | 5/15+，无强 gate | §六 a-stage |

---

## 4. R2 close 准入

> **诚实性修订**：原 §4 要求 §2.2 / §2.3 全 a-e 完成是 over-strict——KD 首 cell + TSM Phase 1-A/1-B 由外部 trigger gate 住，不应阻塞 R2 close。拆分为 4.1 / 4.2。

### 4.1 R2 close 必备（`runs/_r2_verification.json` 全 ✓）

- [ ] §1.0 三 schema a-e 完成（B2+C3 逐 schema 登记）
- [ ] §1 全部 a-e 完成（含 manifest 三 hash + `runs/_r2_audit_coverage.json` + 三组件 `runs/_r2_component_decisions.json` 符合 §1.0 schema）
- [ ] §2.1.1 + §2.1.2 全部 a-e 完成（5 工件逐工件 B2+C3；4 候选 FP16+FP32 engine + sidecar；eval / timing / build-variance / audit JSON 完整；`runs/_r2_precision_decisions.json` 4 record + sensitivity sweep + audit/full agreement + pre_committed_defer_outcome 字段；root-cause classification 凡触发 defer 必有）
- [ ] §2.2.1 + §2.3 plumbing a-e 完成（sidecar 字段 carry-forward；schema + executor + B2/C3 transcript；§2.3 c 4 fixture 全过；export_*.sh 改动通过 backward-compat fixture：现有非 KD/非 TSM engine 重新 load 仍合法）
- [ ] §2.4 两 stub a + e 完成
- [ ] **任一 Case A → soak 记录 hard-bound 至选定 engine**：
  - [ ] `runs/_r2_orin_soak_records.json` + `.sha256` 存在；per-detector record `{detector, engine_path, engine_sha256, selected_artifact_sha256, duration_s, peak_gpu_mem, sustained_fps, thermal_mode}`
  - [ ] `_r2_verify.py` 校验 `engine_sha256 == selected_artifact_sha256`，任一 mismatch → R2 close 阻塞
  - [ ] 该 engine 完成完整 demo + 5-min Orin soak
- [ ] `phase_R2.md` 头表 + appendix；`codex-report-conflictor` 对精度子节 AGREED；coverage-gaps 节列出 §3 + §4.2 中仍 hang-up 的项；任一 detector 含 `construction_failed` 类 → 该 detector 在头表显式标 `degraded-confidence`，不可静默忽略

### 4.2 Carry-forward readiness（不阻塞 R2 close，但必须按 §1.0 `_r2_carry_forward_schema.json` 登记）

写入 `runs/_r2_carry_forward.json`，每 record 符合 §1.0 schema（`item_id` / `status` / `blocked_on` ⊆ closed_enum / `unblock_evidence_path` / `next_entrypoint`）。

- [ ] §2.2.2 KD 首 cell：`status="blocked", blocked_on=["deim_l_training", "hard_neg_manifest_hash", "r2_data_freeze"], unblock_logic="all"`（含 student/teacher 选定隐含）
- [ ] §3.1 TSM Phase 1-A：`status="blocked", blocked_on=["on_vehicle_replay_failure_modes"]`
- [ ] §3.2 HMM scaffold：`status="blocked", blocked_on=["replay_temporal_flicker_or_state_confusion"]`
- [ ] §3.3 §八 Multi-cam：`status="blocked", blocked_on=["autonomy_team"]`
- [ ] §3.4 §九 Adaptive ROI：`status="scheduled", blocked_on=[], next_entrypoint="r3_latency_budget_release"`（无外部 trigger，纯 round 调度）
- [ ] §3.5 §十 INT8 QAT：`status="blocked", blocked_on=["sahi_b_c_measured", "tsm_phase_1b_passed"], unblock_logic="any"`（OR 关系显式编码）
- [ ] §3.6 §十一 Planner-prior：`status="blocked", blocked_on=["planning_team"]`
- [ ] §3.7 §六 SAHI：`status="scheduled", blocked_on=[], next_entrypoint="post_5_15"`

---

## 5. 状态总览（v1.1 时点）

| 类别 | 已落地 | 待启动 |
|---|---|---|
| 训练 | YOLO26-s/13-s/L、DEIM-D-FINE-S/M | DEIM-D-FINE-L（训练中） |
| Export | export_yolo.sh / export_deim.sh + atomic sidecar | KD 3 + TSM 4 字段 carry-forward |
| 决策执行器 | — | `_r2_decide_precision.py` / `_r2_verify.py` / `_kd_decide_cell.py` / `_tsm_decide_phase.py` |
| Schema | `_tsm_activation_schema.json` v1.1 | `_r2_decision_schema.json` / `_kd_decision_schema.json` |
| C++ | — | `demo.cpp` 计时拆分 |
| Scaffold | TSM v1.5；KD a1 | — |
| 计划 stub | — | R3_precision_reproducibility.md / pre_deploy_AGV_integration.md |

---

## 6. 维护规则

- **触发**：主任务 batch 完成 OR 已 LOCK 计划修订。
- **diff 范围**：仅 §-引用 + state checkbox；**不复制** LOCKED 计划内容。
- **提交前一致性**（按 header 三个不可变锚点逐一校验）：
  - `elegant-sauteeing-quail.md` 含 `## Conflictor-loop termination (LOCK after iter 3)` 标题 → 否则阻塞
  - `additional_components_plan.md` 文首版本行匹配 `v1.0` AGREED → 否则阻塞
  - `temporal_optimization_plan.md` §1.1 含 `2026-05-09 v1.1 amendment` 标题 → 否则阻塞
  - 任一锚点 missing → 阻塞提交并 surface 给 user 决定（重新 review-conflict / 升级 lock_id / 接受 drift）
