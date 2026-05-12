# 时序优化计划 v2.0

v1.1 LOCK 2026-05-10

## 状态

| 项 | 当前值 |
|---|---|
| 定位 | R2/R3 可选优化轨道；不属于主检测器选型 |
| 启动条件 | R2 baseline + Orin 部署 + 实车 replay 暴露明确失败模式 |
| 首选 | §1 TSM：detector-level 时序；零参数；零额外推理 FLOPs |
| post-detector 首选 | §2.2 HMM：分类跳变 / 非法转移 |
| 现有 baseline | Plan A：ByteTrack + EMA，已在 R1 落地 |
| 生命周期 | 见 `additional_components_plan.md` §一 |

| 组件 | 状态 | 输出 |
|---|---|---|
| TSM scaffold | LANDED v1.5；B2 + C3 AGREED | `components/temporal_shift_module/` |
| TSM activation schema | LANDED v1.1 | `scripts/_tsm_activation_schema.json` |
| TSM runners | stub / `NotImplementedError` | `concept_validation.py`、`full_dataset_train.py`、`streaming_engine_export.py` |
| HMM scaffold | LANDED；消融位面 only | `components/hmm_smoother/` |
| INT8 coupling | DEFERRED → R3+ | `additional_components_plan.md` §十 |

## Deferred

| 项 | 状态 | next_entrypoint |
|---|---|---|
| TSM INT8 retest | blocked on `tsm_phase_1b_passed`; R2 不执行 INT8 | `additional_components_plan.md` §十 探针 |
| Phase 1-C export改造 | blocked until Phase 1-B pass | 本文件 §1.5 |
| HMM deploy integration | gated by HMM `deploy` decision | 本文件 §2.2 决策规则 |

## 行动项

### 0. 启动判定

| 实测问题 | 路径 | 入口 |
|---|---|---|
| 小目标 / 远距 / 遮挡 / 运动模糊漏检 | TSM | §1 |
| EMA 后仍有边缘类别跳变 | HMM | §2.2 |
| 非法状态转移 | HMM | §2.2 |
| 两类问题同时存在 | TSM + HMM | §1 + §2 |

不启动条件：

- 主检测器选型未完成。
- 实车 replay 未到位。
- 问题是 `redRight` / `greenRight` 样本为 0 或严重不足。

### 0.1 共享数据要求

- [ ] R2 raw video 连续 30 fps。
- [ ] 每段 clip ≥ 30 帧。
- [ ] 跨地点 / 跨时段平衡。
- [ ] train / val 按视频文件切分。
- [ ] raw video 保留 ≥6 个月。
- [ ] ByteTrack track ID 可复现。
- [ ] 人工抽检约 200 段；拒收率目标 < 10%。

### 1. TSM

#### 1.1 机制约束

| 项 | 规则 |
|---|---|
| shift fraction | 1/8 channel 前向 shift |
| c2 channel | 训练和推理两侧均置零；causal end-to-end |
| c3 channel | 6/8 直通 |
| pretrained deploy | 默认禁止；deploy 契约为 scratch |
| pretrained diagnostic | 可作为 cost probe；`deploy_eligible=false` |
| clip length | 默认 N=4 |
| clip 增强 | clip 内采样状态复用一次 |

#### 1.2 DEIM-D-FINE 适用边界

| DEIM 层 | TSM |
|---|---|
| HGNetv2 `HG_Block.forward` | 是 |
| `stage2 / stage3 / stage4` | 是 |
| HGStem / `stage1` | 否 |
| `HybridEncoder` / AIFI attention | 否 |
| D-FINE decoder transformer | 否 |

DEIM runner 必须断言 `HGNetv2.pretrained == False` when `--pretrained-init scratch`。

#### 1.3 Phase 1-A 概念验证

- [ ] 写入 `runs/_tsm_activation.json`。
- [ ] 校验 activation schema 6 必填字段：`schema_version`、`selected_detector_artifact_sha256`、`selected_detector_artifact_path`、`replay_evidence_path`、`approved_failure_mode_tags`、`activation_timestamp`。
- [ ] 校验 SHA 三方一致：activation sha == sidecar `engine_sha256` == 文件计算值。
- [ ] 选择一个 detector patch：YOLO26 / YOLOv13 / DEIM。
- [ ] 实现 `TemporalShift`：c1 前向 shift；c2 zero；c3 直通。
- [ ] clip-of-N dataloader；clip 内增强一致。
- [ ] 10% 数据 + 20 epochs；loss 不 NaN。
- [ ] 小 val 对比 TSM-on / TSM-off，按 bbox 高度分桶。

Gate-1A：小目标 recall +2 pp OR 总 mAP +1 pp。若 activation tag 对应 bucket recall delta ≤ 0，则只能 diagnostic continuation，不可进入 deploy / defer。

Activation tags closed enum：`far_distance_miss`、`occluded_miss`、`motion_blur`、`small_object_miss`。

#### 1.4 Phase 1-B 全量训练

- [ ] 全 R2 数据。
- [ ] 与主线同 epoch / patience / augmentation。
- [ ] TSM vs single-frame baseline：总 mAP、bbox height bucket recall、occlusion recall、Orin latency。
- [ ] locked caveat columns：`input_source`、`resolution`、`tracker_mode`、`output_mode`。

Gate-1B：小目标桶 recall ≥ 0.6 AND end-to-end latency < 26 ms。

#### 1.5 Phase 1-C 导出 / Orin

- [ ] Streaming inference 复用上一帧 1/8 channel features。
- [ ] ROS2 节点每相机持有 per-stage cache。
- [ ] ONNX 使用 `Slice` + `Concat`，不得引入不可替代 custom op。
- [ ] trtexec FP16。
- [ ] end-to-end Orin latency < 26 ms。
- [ ] sidecar 含 `tsm_enabled` / `tsm_shift_fraction` / `tsm_clip_size_train` / `tsm_feature_cache_stages`。

#### 1.6 TSM lifecycle

- [x] a. `components/temporal_shift_module/` LANDED v1.5。
- [x] a. `scripts/_tsm_activation_schema.json` LANDED v1.1。
- [ ] b. runners / patches 替换 stubs；activation JSON 通过 7-step validation。
- [ ] c. Phase 1-A → 1-B → 1-C。
- [ ] d. 写 `runs/_tsm_decisions.json`。
- [ ] e. phase report TSM 子节。

### 2. Post-detector smoothers

#### 2.1 顺序

```text
Plan A fixed EMA
  -> HMM
  -> Adaptive EMA
  -> Per-track GRU
  -> Per-track Transformer
```

#### 2.2 HMM

##### 行动项

- [x] a. `components/hmm_smoother/` scaffold LANDED；消融 runner 位面；不改 `inference/`。
- [ ] b. **b-stage 入口测试要求**：在实现 `apply_decision_rule` 行为前，必须先落地以下 executable tests（详见 `components/hmm_smoother/gates/decision_gate.py` `apply_decision_rule` docstring truth-table）：
  - R1a deploy 正路径：`flicker_improvement_pct > 50` + `eligible_illegal_transition_count == 0` + `map_no_regression == True` → deploy；
  - R1b deploy fall-through（mAP 退化）：`flicker_improvement_pct > 50` + eligible == 0 + `map_no_regression == False` → 不 deploy（按 cascade defer 或 drop）；
  - R1c deploy fall-through（残留 eligible，无 exception）：`flicker_improvement_pct > 50` + `eligible_illegal_transition_count > 0` + `tracker_artifact_residual == False` → 不 deploy；
  - R1d deploy via exception：`flicker_improvement_pct > 50` + `eligible_illegal_transition_count > 0` + `tracker_artifact_residual == True` + `tracker_artifact_evidence_ref` 是 dict 五字段全填且 verifiable + `map_no_regression == True` → deploy + `deploy_via_tracker_artifact_exception == True`；
  - R1e deploy fall-through（exception 标志为 True 但 evidence ref 缺失或字段不全）：`flicker_improvement_pct > 50` + `eligible_illegal_transition_count > 0` + `tracker_artifact_residual == True` + `tracker_artifact_evidence_ref` 缺失/非 dict/任一字段为空 → 不 deploy（按 cascade defer 或 drop；evidence 不全视同 exception 不存在，不视为 executor_error）；输出 `decision_reason="tracker_artifact_evidence_missing"` 或 `"tracker_artifact_evidence_malformed"`；同时新增反例测试 R1f：evidence_ref 字段全填但 trace_id 解析不到对应 track / frame_range 越界 → 不 deploy + `decision_reason="tracker_artifact_evidence_unverifiable"`；
  - R2 defer flicker：`flicker_improvement_pct ∈ [20.0, 50.0]` + `baseline_illegal_count == 0` → defer；
  - R3 drop flicker：`flicker_improvement_pct < 20.0` + `baseline_illegal_count == 0` → drop；
  - R4 defer illegal-OR：`flicker_improvement_pct < 20.0` + `baseline_illegal_count > 0` + `illegal_transition_improvement_pct >= 50.0` + `eligible_illegal_transition_count > 0` → defer；
  - R5 drop illegal weak：`flicker_improvement_pct < 20.0` + `baseline_illegal_count > 0` + `illegal_transition_improvement_pct < 50.0` → drop；
  - **non-row cascade consistency note**（candidate 完全消除 illegal）：`baseline_illegal_count > 0 + eligible_illegal_transition_count == 0` **不构成独立行**，**MUST NOT** 在 test ID / schema comment / output artifact 中以 `R6` 名义出现。该输入按 flicker 带通过 R1/R2/R3/R5 cascade：`flicker ∈ [20, 50]` → defer (R2)；`flicker < 20` → drop (R3/R5)；`flicker > 50` → R1a/R1b 的 deploy gate（按 mAP 通过/失败）。测试覆盖 `flicker = 30 + eligible == 0` → defer 与 `flicker = 5 + eligible == 0` → drop 两个 bucket。**precedence 声明（v2.0 lock）**：flicker 闸失败（即 `< 50.0`）支配 illegal-transition 完全消除；不存在 "safety-override" 规则使 illegal-elimination 反向把决策升级到 deploy 或越过 R3/R5 的 drop；如果未来引入 safety-override，必须开新一轮 conflictor 评审，本表不允许默认放行。
  - 反例：`baseline_illegal_count == 0` 不得触发 `executor_error`；
  - 反例：`baseline_flicker_rate == 0` 必须触发 `executor_error`。
  - **safety_class passed 重算测试**：runner 必须先按 `passed_recomputed = (full_val_support < 30) OR (ap_delta_pp >= -map_regression_tolerance_pp)` 重算每个 `safety_class_ap_deltas[i].passed`，断言与 input 一致；不一致 → `outcome="executor_error"`，`decision_reason="executor_error_threshold_oob"`。
  - **map_no_regression 派生测试**：在 passed 重算之后，runner 必须从 `total_map_ap_delta_pp >= -map_regression_tolerance_pp` AND 每个 `safety_class_ap_deltas[i].passed_recomputed == True` 派生 `map_no_regression_recomputed`，断言等于 input；不一致 → 同上。
  - **eval-metrics 三重 hash 完整性测试**：runner 必须 (a) 重算 `eval_metrics_json_path` 的 SHA256 == `eval_metrics_json_sha256`；(b) `eval_split_manifest_sha256` 等于 round frozen manifest hash；(c) `baseline_detector_artifact_sha256` 等于 round selected detector hash；任一 mismatch → `outcome="executor_error"`，`decision_reason="executor_error_eval_metrics_hash_mismatch"`。
  - **candidate provenance pin 测试**：runner 必须断言 (a) `candidate_temporal_config_sha256` 与同 (alpha, mode) cell 跨 sweep 一致；(b) `candidate_output_artifact_sha256` 文件存在 + 实算 hash 匹配；(c) `runner_revision_sha256` 跨整个 sweep 一致；(d) `track_trace_artifact_sha256` 文件存在 + 实算 hash 匹配。
  - **deploy-exception scope 测试**（仅 `tracker_artifact_residual == True` 时）：(d') `candidate_run_submitter != tracker_artifact_evidence_ref.reviewer`（reviewer-independence；身份字符串经过规范化后比较，不区分大小写 / 别名）；(e') `tracker_artifact_evidence_ref.trace_id` 解析到 `track_trace_artifact_path` 中的 track + `frame_start <= frame_end <= max(frame in track)`；任一 mismatch → R1e cascade（不是 executor_error）。
  - **deploy-non-exception scope 测试**（`tracker_artifact_residual == False`）：必须 `tracker_artifact_evidence_ref == null`；非 null → `outcome="executor_error"`，`decision_reason="executor_error_threshold_oob"`（schema 状态不一致）。
- [ ] b. 估计 transition matrix：训练数据 + ByteTrack 多数投票轨迹。
- [ ] b. `configs/temporal_hmm.yaml`：`transition_matrix_path`、`viterbi_window`、`laplace_alpha`、`illegal_transition_set`、`illegal_transition_policy`。
- [ ] b. Python forward-backward / Viterbi。
- [ ] c. A/B：Plan A fixed EMA vs Plan A + HMM。
- [ ] c. α sweep：`0.01` / `0.1` / `1.0`；默认 `0.1`。
- [ ] d. 写 `runs/_hmm_decisions.json`。
- [ ] e. phase report HMM 子节。

##### 评估协议

| 项 | 规则 |
|---|---|
| eligible tracks | track length ≥ 5；evaluation window 内无 ID switch |
| flicker rate | argmax-change count / adjacent valid frame pairs；eligible tracks mean |
| illegal transition | `temporal_hmm.yaml` illegal set 绝对计数 |
| baseline | 同 replay 的 Plan A fixed EMA |
| priority | flicker gate 闸 (`flicker_improvement_pct > 50.0`) 优先：deploy 必须先过 flicker；illegal-transition 完全消除不构成 deploy override（precedence lock，详见 §2.2 决策规则 non-row cascade consistency note） |
| mAP / AP | 总 mAP / 安全类 AP 不退化 |
| latency | HMM < 0.01 ms |

##### 决策规则

公式定义（pre-committed；executor 强制）：

```text
flicker_improvement_pct = 100 * (baseline_flicker_rate - candidate_flicker_rate) / baseline_flicker_rate
illegal_transition_improvement_pct = 100 * (baseline_illegal_count - candidate_illegal_count) / baseline_illegal_count
```

前置条件：
- `baseline_flicker_rate > 0`；否则写 `outcome="executor_error"`。
- 若 defer OR-branch 求值需要 `illegal_transition_improvement_pct`，则要求 `baseline_illegal_count > 0`；当 `baseline_illegal_count == 0` 时，OR-branch 不参与求值（`flicker_improvement_pct` 主分支正常评估）。

| outcome | condition | JSON |
|---|---|---|
| deploy | `flicker_improvement_pct > 50.0` AND (`eligible_illegal_transition_count == 0` OR (`tracker_artifact_residual == True` AND `tracker_artifact_evidence_ref` 是 dict `{trace_id, track_id, frame_start, frame_end, reviewer}` 五字段全部非空 AND `trace_id` 解析到真实 track AND `frame_start <= frame_end` 在 track 帧范围内 AND reviewer 与 candidate run 提交者相互独立)) AND 总 mAP `ap_delta_pp >= -0.2` AND 每个 `full_val_support >= 30` 安全类 `ap_delta_pp >= -0.2`（独立各自满足）| `outcome="deploy"`, set `deploy_via_tracker_artifact_exception` accordingly |
| defer | `20.0 <= flicker_improvement_pct <= 50.0` OR (`baseline_illegal_count > 0` AND `illegal_transition_improvement_pct >= 50.0` AND `eligible_illegal_transition_count > 0`) | `outcome="defer"`, next=`§2.3 AdaEMA` |
| drop | 不满足 deploy / defer 的合法输入 | `outcome="drop"` |
| executor_error | NaN / 缺字段 / 阈值越界 / `total_transitions == 0` / `baseline_flicker_rate == 0` | `outcome="executor_error"` |

求值顺序：deploy → defer → drop。边界：`flicker_improvement_pct = 20.0` → defer；`= 50.0` → defer；`> 50.0` → deploy（strict）。

HMM `deploy` 后才启动 `inference/temporal/`、`inference/cpp/include/temporal.hpp`、`inference/cpp/src/temporal_hmm.cpp`。

#### 2.3 Adaptive EMA

- [ ] 小 MLP 预测 α。
- [ ] 输入：`conf_t`、`entropy(p_{t-1})`、`age_t`、`argmax_change_t`。
- [ ] 延迟目标 < 0.01 ms。
- [ ] 仅在 HMM defer / drop 后按 §2.2 决策说明启动。

#### 2.4 Per-track GRU

- [ ] 自写 PyTorch sequence loop。
- [ ] input dim = `num_classes + 2`。
- [ ] hidden dim = 32 起步。
- [ ] loss = sequence CE + temporal consistency KL。
- [ ] ONNX T=1 单步导出。
- [ ] TRT FP16 engine。
- [ ] 每相机一个 state map；NaN 单轨回退 raw `class_probs`。

#### 2.5 Per-track Transformer

- [ ] 仅在 GRU 落地但效果不足时启动。
- [ ] 1-2 层 causal encoder。
- [ ] TRT reshape / attention 数值 diff 必测。

### 3. StreamYOLO

- [ ] 仅当 TSM Gate-1A 失败或 TSM 跨视频不稳定时评估。
- [ ] 预算 2-3 周。
- [ ] 训练数据要求同 TSM。
- [ ] Orin latency budget：+5-10 ms 预估。

排除：FGFA、ConvLSTM backbone、TransVOD / Selsa / MEGA。

## 决策规则

### TSM d-stage

| branch | condition | result |
|---|---|---|
| deploy | deploy-eligible Phase 1-A pass；Phase 1-B pass；Phase 1-C pass；sidecar match；latency < 26 ms | deploy |
| defer INT8 | deploy-eligible Phase 1-A pass；Phase 1-B precision pass；Phase 1-B or 1-C latency ∈ [26, 35] ms | R2 only writes carry-forward `tsm_int8_retest`; R3+ activates `additional_components_plan.md` §十 探针 |
| defer export | deploy-eligible 1-A + 1-B pass；Phase 1-C export / trtexec / sidecar / eval-parity fails | export改造 |
| drop activation mismatch | Gate-1A OR pass but activation tag bucket delta ≤ 0 | drop immediately; later diagnostics cannot rewrite |
| drop Gate-1A | small recall +<2 pp AND mAP +<1 pp | drop |
| drop Gate-1B precision | Phase 1-B small recall < 0.6 | drop |
| drop latency | latency > 35 ms and INT8 cannot recover OR INT8 hits `additional_components_plan.md` §十 决策规则 drop | drop |
| drop export impossible | custom op has no Slice+Concat equivalent | drop |

Deploy hard-fail: `pretrained_init_mode != "scratch"` OR `deploy_eligible != true` OR `seed != 5`。

### HMM d-stage

见 §2.2 decision table。

### TSM vs SAHI

| 条件 | 选择 |
|---|---|
| 连续 30 fps 数据 + replay small/far/occluded miss | 先 TSM |
| 无时序 SOP 但有小目标痛点 | SAHI 可作为 inference-only 路径 |
| TSM 小目标仍不够 | 再叠加 SAHI，仅路口附近启用 |
| SAHI latency in [50,80) | R3+ INT8 retest |

## 文件清单

### LANDED

| 路径 | 内容 |
|---|---|
| `components/temporal_shift_module/__init__.py` | TSM 顶层契约 |
| `components/temporal_shift_module/modules/` | `TemporalShift` / feature cache stubs |
| `components/temporal_shift_module/data/` | clip collator stubs |
| `components/temporal_shift_module/patches/` | detector patch contracts |
| `components/temporal_shift_module/runners/` | phase runner stubs |
| `components/temporal_shift_module/gates/` | Gate-1A / Gate-1B stubs |
| `scripts/_tsm_activation_schema.json` | activation schema |
| `components/hmm_smoother/` | HMM消融位面 scaffold |

### Phase-gated

| 路径 | 触发 |
|---|---|
| `components/temporal_shift_module/patches/{yolo26_basicblock.py,yolov13_basicblock.py,deim_hg_block.py}` | TSM Phase 1-A |
| `runs/_tsm_activation.json` | TSM Phase 1-A 前 |
| `scripts/export_yolo.sh` / `scripts/export_deim.sh` TSM sidecar fields | TSM Phase 1-C 前 |
| `inference/cpp/include/temporal_cache.hpp` / `inference/cpp/src/temporal_cache.cpp` | TSM Phase 1-C |
| `scripts/fit_temporal_hmm.py` | HMM b-stage |
| `runs/_hmm_transition_matrix.{json,npy}` | HMM b-stage |
| `inference/temporal/` and `inference/cpp/src/temporal_hmm.cpp` | HMM deploy only |
| `scripts/train_gru_head.py` / `scripts/export_gru_head.py` | GRU path |

## 衔接

- `development_plan.md`：Track 1 / Track 2 定位、§Stage 1.A 稳定门（flip-flop rate / burst jitter / dwell time / illegal transition）是本 plan 的 success criterion；§Stage 2 latency 不在本 plan 范围。2026-05-15 deadline retired。
- `additional_components_plan.md`：共享生命周期、SAHI、INT8 QAT。
- `pre_deploy_AGV_integration.md`：TSM / HMM carry-forward 接收（pre_r2_kickoff_checklist.md 2026-05-12 已归档）。
- `cross_detection_reasoning_plan.md`：时序之后的同帧共现 R3 可选路径。
- `scripts/_r2_carry_forward_schema.json`：`tsm_phase_1b_passed`、`on_vehicle_replay_failure_modes`、`replay_temporal_flicker_or_state_confusion` tokens。
