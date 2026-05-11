# R2 启动前 Kickoff Checklist v2.0

v1.1 LOCK 2026-05-10

## 状态

| 项 | 当前值 |
|---|---|
| 用途 | R2 启动、等待期、R2 close gate、carry-forward 登记 |
| 生命周期 | 见 `additional_components_plan.md` §一 |
| R2 must-do | §1.1 R2 val freeze、§1.2 Copy-paste、§1.3 hard-negative、§2.1 precision parity、§2.2.1 KD plumbing、§2.3 TSM plumbing |
| R2 deferred | §1.4 map-prior、§3.4 adaptive ROI、§3.5 INT8 QAT、§3.6 planner-prior |
| Schema authority | `scripts/_r2_carry_forward_schema.json`、`scripts/_r2_component_decision_schema.json` |

| 类别 | 已落地 | 待启动 / blocked |
|---|---|---|
| 训练 | YOLO26-s/13-s/L、DEIM-D-FINE-S/M | DEIM-D-FINE-L |
| Export | `scripts/export_yolo.sh` / `scripts/export_deim.sh` + atomic sidecar | KD 3 字段 + TSM 4 字段 |
| 决策执行器 | `_r2_decide_precision.py` / `_r2_verify.py` scaffold | `_kd_decide_cell.py`、`_tsm_decide_phase.py` |
| Schema | §1.0 三 schema、`_r2_decision_schema.json`、`_r2_schema_utils.py` | `_kd_decision_schema.json` |
| Scaffold | TSM v1.5、KD a1、Copy-paste、Hard-negative、HMM | SAHI a-stage |
| Deferred → R3+ | — | §五 / §九 / §十 / §十一 carry-forward |

## Deferred

| item_id | status | blocked_on | unblock_logic | next_entrypoint |
|---|---|---|---|---|
| `map_prior_gate_§五` | blocked | `gps_topic_5_12`, `r2_raw_video_negative_controls` | all | `additional_components_plan.md §五` |
| `adaptive_roi_§九` | blocked | `map_prior_landed`, `r3_inference_budget_window` | all | `additional_components_plan.md §九` |
| `int8_qat_§十` | blocked | `sahi_b_c_measured`, `tsm_phase_1b_passed` | any | `additional_components_plan.md §十` |
| `planner_prior_§十一` | blocked | `planning_team` | all | `additional_components_plan.md §十一` |

## 外部阻塞

| # | 等待对象 | 解锁 |
|---|---|---|
| 0.1 | R2 数据 freeze | §1.1 / §1.2 / §1.3 / §2.1.2 |
| 0.2 | DEIM-D-FINE-L 训练 | KD teacher 选定 / A7 |
| 0.3 | GPS topic | §五 R3+ 重启 |
| 0.4 | 在车 replay 出 small/far/occluded miss | TSM |
| 0.5 | replay temporal flicker / state confusion | HMM |
| 0.6 | autonomy / planning 团队对接 | 多相机 / planner-prior |

## 1. R2 启动前必须

### 1.0 共享 schema

- [x] a. `scripts/_r2_component_decision_schema.json`
- [x] a. `scripts/_r2_audit_coverage_schema.json`
- [x] a. `scripts/_r2_carry_forward_schema.json`
- [x] b. `scripts/_r2_schema_utils.py` runtime uniqueness / path checks
- [x] c. `scripts/_r2_schemas_test.py` 负向 fixture
- [x] d. B2 + C3 AGREED
- [ ] e. 路径 + transcript 写入 `runs/_r2_verification.json`（blocked on §2.1.1 b-stage）

`runs/_r2_component_decisions.json` row format：

```json
{
  "component": "copy_paste_balance",
  "outcome": "deploy|defer|drop",
  "reason": "...",
  "blocking_artifacts": ["path|sha256"],
  "next_round_action": "...",
  "branch": "live|replay_only"
}
```

`runs/_r2_carry_forward.json` row format：

```json
{
  "item_id": "...",
  "status": "blocked|scheduled",
  "blocked_on": ["closed_enum_token"],
  "unblock_logic": "all|any",
  "unblock_evidence_path": "...",
  "next_entrypoint": "..."
}
```

`blocked_on` closed enum（13 tokens；权威定义见 `scripts/_r2_carry_forward_schema.json`）：

| token | 用途 |
|---|---|
| `r2_data_freeze` | R2 frozen val / train manifest |
| `deim_l_training` | DEIM-D-FINE-L teacher |
| `on_vehicle_replay_failure_modes` | TSM 启动 |
| `hard_neg_manifest_hash` | KD gate #3 |
| `autonomy_team` | 多相机融合 |
| `planning_team` | planner-prior |
| `gps_topic_5_12` | map-prior GPS topic |
| `sahi_b_c_measured` | SAHI measured latency / recall |
| `tsm_phase_1b_passed` | TSM Phase 1-B measured |
| `replay_temporal_flicker_or_state_confusion` | HMM 启动 |
| `r2_raw_video_negative_controls` | map-prior 负面控制 raw video |
| `r3_inference_budget_window` | R3 推理预算窗口 |
| `map_prior_landed` | map-prior a-stage LANDED |

### 1.1 R2 验证集 freeze + 分层 audit subset

- [ ] a. manifest schema + sampling rule 冻结；schema / sampling code hash 入 manifest。
- [ ] b. 生成 `runs/_r2_val_manifest.txt` + `.sha256`。
- [ ] b. 生成 `runs/_r2_eval_parity/sample_manifest.json`。
- [ ] b. 生成 `runs/_r2_audit_coverage.json` + `.sha256`。
- [ ] b. 传播规则：`low_power` → confidence downgrade；`construction_failed` → escape outcome。
- [ ] c. image hash / stratified coverage / schema / tamper negative tests 通过。
- [ ] d. manifest + schema + sampling-code hash freeze；任意改动 = 重跑 parity。
- [ ] e. 路径与 hash 写入 `runs/_r2_verification.json`。

### 1.2 Copy-paste + class-balanced loss

- [ ] a. 数据增强 hook + class-balanced loss 接入点。
- [ ] b. rare-class copy-paste + FP ceiling 监控。
- [ ] c. 开 / 关 + strength sweep；per-class safety floor。
- [ ] d. 写 `runs/_r2_component_decisions.json`，符合 §1.0 schema。
- [ ] e. phase report 子节。

Decision rule：见 `additional_components_plan.md` §三。

### 1.3 Hard-negative mining

- [ ] a. mining script + candidate pool sampling rule。
- [ ] b. baseline 出 hard-neg pool。
- [ ] b. 冻结 `runs/_hard_negative_eval_manifest.json` + `.sha256`。
- [ ] c. 注入比例 sweep；per-class true-light recall delta ≥ -0.5 pp。
- [ ] d. 写 `runs/_r2_component_decisions.json`。
- [ ] e. phase report 子节。

Decision rule：见 `additional_components_plan.md` §四。

### 1.4 Map-prior gating

DEFERRED → R3+。R2 启动前不要求 a-e。登记见 §4.2。

## 2. 等待期并行任务

### 2.1 R2 Precision Parity

#### 2.1.1 即时 plumbing

- [x] a. `scripts/_r2_decision_schema.json`
- [x] a. `scripts/_r2_decide_precision.py` scaffold
- [x] a. `scripts/_r2_verify.py` scaffold
- [ ] b. verify 仅做 schema / path / hash 校验；不得做阈值分支 / case 分类 / delta 计算。
- [ ] b. verify 加 Orin soak SHA 比对：`engine_sha256 == selected_artifact_sha256`。
- [ ] b. `inference/cpp/src/demo.cpp` 增加 `t_detect` / `t_track` 拆分日志。
- [ ] b. build-determinism 双构建脚本骨架。
- [ ] b. mixed-precision pipeline 加载 `best_fp32.engine`。
- [~] c. schema / `$ref` / helper tests 部分通过；verify / demo.cpp fixture 待补。
- [~] d. a-stage 三工件 B2 + C3 AGREED；b-stage 工件 pending。
- [ ] e. transcript 写 `runs/_r2_verification.json`。

#### 2.1.2 数据 freeze 后执行

- [ ] b. `_r2_decide_precision.py` 实现 Case C → A → D → B；exactly-one outcome。
- [ ] b. 支持 `inconclusive_global` / `audit_disagreement` / `executor_error` escape。
- [ ] b. hash-pin `runs/_r2_audit_coverage.json`。
- [ ] b. 构建 FP32 engine × 4 + sidecar。
- [ ] b. build-determinism 实跑；超阈写 `pre_committed_defer_outcome="r2_fp16_default"`。
- [ ] c. eval-parity：per-image IoU 0.99、score ±0.005、mAP ≤ 0.01 pp、class order 一致。
- [ ] c. FP32 demo 完整跑；记录 peak GPU mem / first-frame / median `t_detect_ms`。
- [ ] e. 写 `runs/_r2_precision_decisions.json` + phase report appendix。

### 2.2 KD plumbing

#### 2.2.1 即时 plumbing

- [ ] a. `scripts/_kd_decision_schema.json`。
- [ ] a. `scripts/_kd_decide_cell.py` scaffold。
- [ ] b. 5 acceptance gates；ship-decision 强制 `seed5`。
- [ ] b. gate #3 hash-pin `runs/_hard_negative_eval_manifest.json`。
- [ ] b. `scripts/export_yolo.sh` / `scripts/export_deim.sh` 加 KD sidecar 字段。
- [ ] c. sidecar size-stability；`engine_sha256` 流程；`bash -n`。
- [ ] d. B2 + C3：`_kd_decide_cell.py`、`_kd_decision_schema.json`、export KD fields。
- [ ] e. transcript 写 `runs/_r2_verification.json`。

#### 2.2.2 KD 首 cell 决策执行

blocked：R2 student/teacher 选定 + `hard_neg_manifest_hash` + `deim_l_training` + `r2_data_freeze`。解锁后入口：`additional_components_plan.md` §七。

### 2.3 TSM Phase 1-C plumbing

- [ ] a. `scripts/_tsm_decide_phase.py` scaffold。
- [ ] b. decision tree 实现：deploy / defer INT8 / defer export / 5 drop（覆盖 activation mismatch / Gate-1A / Gate-1B precision / latency / export impossible），见 `temporal_optimization_plan.md` §决策规则。
- [ ] b. deploy hard-fail：`pretrained_init_mode == "scratch"` AND `deploy_eligible == true` AND `seed == 5`。
- [ ] b. 写 `runs/_tsm_decisions.json`。
- [ ] b. export scripts 加 `tsm_enabled` / `tsm_shift_fraction` / `tsm_clip_size_train` / `tsm_feature_cache_stages`。
- [ ] c. deploy-path negative matrix + positive fixture。
- [ ] d. B2 + C3：`_tsm_decide_phase.py`、export TSM fields。
- [ ] e. transcript 写 `runs/_r2_verification.json`。

### 2.4 Carry-forward stubs

Reduced lifecycle：a + e；b/c/d = n/a。

- [ ] a. `docs/planning/R3_precision_reproducibility.md`（trigger / held-out / success criteria）。
- [ ] a. `docs/planning/pre_deploy_AGV_integration.md`（Case A FP32 ship / 5-min Orin soak / AGV latency budget）。
- [ ] e. stub 路径登记于 `runs/_r2_verification.json`。

### 2.5 Pre-R2 ablation rehearsals

Rehearsal outputs 不进 ship-decision。文件名前缀必须为 `rehearsal_`，并含 `rehearsal_kind ∈ {r1_data, synthetic_fixture, demo_only}`。

Executor-side gate：

1. ship-decision JSON 拒绝 `rehearsal_kind`。
2. ship-decision JSON 引用 artifact 拒绝 `rehearsal_` 前缀。
3. ship-decision JSON 必须含 R2 frozen manifest hash。
4. rehearsal outputs 只可挂到 `runs/_r2_verification.json.rehearsal_outputs`。

| rehearsal | action | output |
|---|---|---|
| Hard-negative R1 | demo8/11/13 挖 FP + frozen R1 manifest + shortened A/B | `runs/rehearsal_hard_negative_decision_R1.json` |
| Copy-paste R1 | R1 数据三臂机制验证 | `runs/rehearsal_copy_paste_decision_R1.json` |
| KD A1 wall-clock | R1 单 epoch YOLO / DEIM | `runs/rehearsal_kd_A1_walltime_estimate.json` |
| TSM 1-A | R1 demo + synthetic clip | `runs/rehearsal_tsm_phase_1a_concept.json` |
| HMM | synthetic flicker / transition fixture | `runs/rehearsal_hmm_smoother_synthetic.json` |
| SAHI | R1 demo，需 §六 a-stage 后启动 | `runs/rehearsal_sahi_R1_demo.json` |
| Decision executor | synthetic Case A/B/C/D + escapes | `scripts/_r2_decide_precision_mechanical_test.py` |

## 3. Gated / 非 actionable

| # | 任务 | Gate | 状态 |
|---|---|---|---|
| 3.1 | TSM Phase 1-A | R2 detector + on-vehicle replay miss | gated |
| 3.2 | HMM scaffold / ablation | replay temporal flicker / state confusion | gated |
| 3.3 | Multi-camera fusion | autonomy team | gated |
| 3.4 | Adaptive inference / ROI | map_prior_landed + r3_inference_budget_window | DEFERRED → R3+ |
| 3.5 | INT8 QAT | measured SAHI band OR TSM Phase 1-B band | DEFERRED → R3+ |
| 3.6 | Planner-prior | planning team + cross-detection | DEFERRED → R3+ |
| 3.7 | SAHI | 5/15+ | scheduled |
| 3.8 | Map-prior gating | gps_topic_5_12 + r2_raw_video_negative_controls | DEFERRED → R3+ |

## 4. R2 close 准入

### 4.1 必备

- [ ] §1.0 schema a-e 完成。
- [ ] §1.1 / §1.2 / §1.3 全部 a-e 完成。
- [ ] §2.1.1 / §2.1.2 全部 a-e 完成。
- [ ] §2.2.1 / §2.3 plumbing a-e 完成。
- [ ] §2.4 两 stub a + e 完成。
- [ ] 如启动 §2.5，rehearsal outputs 写入 `runs/_r2_verification.json`。
- [ ] 任一 Case A：`runs/_r2_orin_soak_records.json` + `.sha256` 存在，且 `engine_sha256 == selected_artifact_sha256`。
- [ ] 完整 demo + 5-min Orin soak。
- [ ] phase report 头表 + appendix + coverage-gaps 完整。
- [ ] 任一 detector 含 `construction_failed` 类：头表标 `degraded-confidence`。

Coverage-gaps 行格式：

```text
item_id | status | blocked_on | unblock_logic | unblock_evidence_path | next_entrypoint
```

散文替代 6 字段行 = R2 close 阻塞。

### 4.2 Carry-forward readiness

写入 `runs/_r2_carry_forward.json`；每 record 符合 `scripts/_r2_carry_forward_schema.json`。

| item_id | status | blocked_on | unblock_logic | next_entrypoint |
|---|---|---|---|---|
| `kd_first_cell` | blocked | `deim_l_training`, `hard_neg_manifest_hash`, `r2_data_freeze` | all | `additional_components_plan.md §七` |
| `tsm_phase_1a` | blocked | `on_vehicle_replay_failure_modes` | all | `temporal_optimization_plan.md §1` |
| `hmm_smoother` | blocked | `replay_temporal_flicker_or_state_confusion` | all | `temporal_optimization_plan.md §2.2` |
| `multi_camera_fusion` | blocked | `autonomy_team` | all | `additional_components_plan.md §八` |
| `adaptive_roi_§九` | blocked | `map_prior_landed`, `r3_inference_budget_window` | all | `additional_components_plan.md §九` |
| `int8_qat_§十` | blocked | `sahi_b_c_measured`, `tsm_phase_1b_passed` | any | `additional_components_plan.md §十` |
| `planner_prior_§十一` | blocked | `planning_team` | all | `additional_components_plan.md §十一` |
| `sahi_§六` | scheduled |  |  | `post_5_15` |
| `map_prior_gate_§五` | blocked | `gps_topic_5_12`, `r2_raw_video_negative_controls` | all | `additional_components_plan.md §五` |
| `sahi_int8_retest` | blocked | `sahi_b_c_measured` | all | `additional_components_plan.md §十 探针` |
| `sahi_recall_marginal_retest` | blocked | `r3_inference_budget_window` | all | `additional_components_plan.md §六 决策规则` |
| `tsm_int8_retest` | blocked | `tsm_phase_1b_passed` | all | `additional_components_plan.md §十 探针` |

## 5. 维护规则

- 触发：主任务 batch 完成 OR locked plan 修订。
- diff 范围：§ 引用 + checkbox + state only。
- 提交前一致性：
  - `~/.claude/plans/elegant-sauteeing-quail.md` 含 precision parity LOCK 锚点。
  - `additional_components_plan.md` 文首含 `v1.1 LOCK 2026-05-10`。
  - `temporal_optimization_plan.md` 文首含 `v1.1 LOCK 2026-05-10`。

## 衔接

- `additional_components_plan.md`：生命周期定义与组件决策规则。
- `development_plan.md`：R2 范围、模型、部署 gate。
- `temporal_optimization_plan.md`：TSM / HMM 触发与决策。
- `cross_detection_reasoning_plan.md`：planner-prior / cooccurrence R3 入口。
- `scripts/_r2_carry_forward_schema.json`：13-token closed enum 权威来源。
- `scripts/_r2_component_decision_schema.json`：deploy / defer / drop row 格式权威来源。
