# 附加组件指南 — 用法与当前状态

按 `additional_components_plan.md` §三 ~ §十一 + 时序优化（TSM / HMM）枚举所有附加组件，给出当前阶段、入口 CLI、输入/输出契约、激活前置条件。

**生命周期阶段**（与 §一 一致）：

| 阶段 | 含义 |
|---|---|
| `scaffold STUB` | 模块文件已落地但 main()/CLI 抛 `NotImplementedError` |
| `a-stage` | 数据结构、schema、gate 函数、配置加载已写完；runner orchestration 缺失 |
| `b-stage` | runner CLI 可消费输入并写 decision JSON；尚未在真实数据上验证 |
| `LANDED` | runner CLI 经 B2 + C3 review-conflict AGREED；可在合适硬件上 `--execute` |
| `executed` | 已在 R1 / R2 数据上跑出 sidecar JSON |

---

## 总览表

| 组件 | 入口 | 当前阶段 | 可执行性 |
|---|---|---|---|
| §三 Copy-paste + 类平衡 | `components/copy_paste_balance/runners/ablation.py` | **a-stage scaffold** | ❌ b-stage runner CLI 未实现 |
| §四 硬负样本挖掘 | `components/hard_negative_mining/runners/ablation.py` | **a-stage scaffold** | ❌ b-stage runner CLI 未实现 |
| §五 地图先验门控 | — | **DEFERRED → R3+** | ⏸ 不在 R2 范围 |
| §六 SAHI 切片推理 | — | **gated** | ⏸ 待 a-stage 启动 |
| §七 KD A0 (DEIM baseline GO-LSD off) | `components/knowledge_distillation/runners/deim_baseline_golsd_off.py` | **scaffold STUB** | ❌ |
| §七 KD A1 (scratch + wall-clock anchor) | `components/knowledge_distillation/runners/scratch_baseline.py` | **LANDED** | ✅ ready |
| §七 KD A2a (YOLO logit KD) | `components/knowledge_distillation/runners/yolo_logit_kd.py` | **LANDED** | ✅ ready |
| §七 KD A2b (DEIM LD + logit KD) | `components/knowledge_distillation/runners/deim_logit_localization_kd.py` | **LANDED** | ✅ ready（GPU server + DEIM venv） |
| §七 KD A3 (PKD feature) | `components/knowledge_distillation/runners/pearson_feature_kd.py` | **scaffold STUB** | ❌ |
| §七 KD A4 (logit + feature joint) | `components/knowledge_distillation/runners/logit_plus_feature_kd.py` | **scaffold STUB** | ❌ |
| §七 KD A5 (progressive 2-teacher) | `components/knowledge_distillation/runners/progressive_multi_teacher.py` | **scaffold STUB** | ❌ |
| §七 KD A6 (cross-arch) — runner | `components/knowledge_distillation/runners/cross_arch_feature_kd.py` | **scaffold STUB** | ❌ 1-week PoC 待启动 |
| §七 KD A6 — 设计 spike | `components/knowledge_distillation/spikes/a6_design_spike.py` | **executed**（2026-05-11，path γ） | ✅ 输出 `runs/rehearsal_kd_A6_design_spike.json` |
| §七 KD A7 (DEIM-L teacher TAKD) | `components/knowledge_distillation/runners/takd_large_teacher.py` | **scaffold STUB**（gated on DEIM-L 训练完成） | ❌ |
| §八 多相机融合 | — | **blocked**（autonomy team） | ⏸ R3+ |
| §九 自适应推理 / ROI | — | **DEFERRED → R3+** | ⏸ |
| §十 INT8 QAT | — | **DEFERRED → R3+** | ⏸ |
| §十一 规划器先验融合 | — | **DEFERRED → R3+** | ⏸ |
| TSM (`temporal_optimization_plan.md` §1) | `components/temporal_shift_module/runners/*.py` | **v1.5 AGREED scaffold**；runners STUB | ❌ phase-1a concept 待启动 |
| HMM smoother (`temporal_optimization_plan.md` §2) | `components/hmm_smoother/runners/ablation.py` | **scaffold STUB** | ❌ |

LANDED 总计：4 个入口（A1 / A2a / A2b / A6 spike），全部隶属 §七 KD。其余 13 个组件入口处于 scaffold / a-stage / deferred 状态。

---

## §七 KD A1 — scratch + wall-clock anchor

定位：KD-acceptance §六 #1 / #4 的 `T_scratch_A1` 基准锚点。A1 不被自己对比（它就是 reference）。

**CLI**：

```bash
# Dry-run（不训练，写 pending JSON）
uv run python -m components.knowledge_distillation.runners.scratch_baseline \
    --family {yolo|deim} --size {n,s,m,l} \
    --rehearsal-on-r1 --epochs 1 --seed 0 \
    --dry-run

# Execute（在训练服务器上跑）
uv run python -m components.knowledge_distillation.runners.scratch_baseline \
    --family yolo --size s --rehearsal-on-r1 --epochs 1 --seed 0 --execute
```

**输出**：`runs/rehearsal_kd_A1_walltime_estimate.json`（two families merge per invocation）。

**派发逻辑**：
- `--family yolo` → `uv run python main.py train yolo26<size> --epochs N --seed S`
- `--family deim` → `bash scripts/train_deim.sh <size> -u epoches=N --seed N --output-dir ../runs/rehearsal_kd_A1_deim_<size>_seed<seed>`

**契约**：
- `--epochs >= 1`（argparse 拒绝 0；零 epoch 无 wall-clock 意义）
- `--rehearsal-on-r1` 强制输出文件以 `rehearsal_` 为前缀
- DEIM 调度自带 `--output-dir` 隔离，避免与 R1/R2 已有运行冲突

---

## §七 KD A2a — YOLO26-s ← YOLO26-m cls-logit KL

定位：YOLO 路径同架构 logit KD（YOLO26 `reg_max=1` 无 DFL，LD 不适用，只跑 cls KL）。

**CLI**：

```bash
# Dry-run
uv run python -m components.knowledge_distillation.runners.yolo_logit_kd \
    --teacher-ckpt runs/detect/yolo26m-r1/weights/best.pt \
    --student-init {scratch|coco|r2_baseline|r1_rehearsal} \
    --rehearsal-on-r1 --epochs 1 --seed 0 \
    --kd-lambda 1.0 --kd-temperature 2.0 \
    --dry-run

# Execute（in-process Ultralytics 训练；KDDetectionTrainer subclass）
uv run python -m components.knowledge_distillation.runners.yolo_logit_kd \
    --teacher-ckpt runs/detect/yolo26m-r1/weights/best.pt \
    --rehearsal-on-r1 --epochs 1 --execute
```

**输出**：`runs/rehearsal_kd_A2a_R1.json`（含 `kd_call_count` 字段做 post-hoc audit）。

**契约**：
- `--kd-lambda > 0`（argparse 拒绝 0；零权重 = silent no-op）
- `--epochs >= 1`
- KD 实际生效靠 `KDDetectionTrainer._kd_call_count[0]` 计数；训练结束若计数 = 0 → `RuntimeError` 拒绝记录 `completed`
- 教师在 BN-frozen `.train()` 模式（保证 `Detect.forward` 返回 `{one2many, one2one}` 字典），师 / 生 cls 头形状必须一致（`nc + stride` 一致）

---

## §七 KD A2b — DEIM-D-FINE-S ← DEIM-D-FINE-M LD-on-FDR + cls-logit KL

定位：DEIM 路径同架构 joint KD（cls-logit KL + FDR 分布 LD KL）。spec §5.5 是 GO-LSD vs external-LD overlap ablation。

**CLI**：

```bash
# Dry-run（runner 派发命令，不实际启动 torchrun）
uv run python -m components.knowledge_distillation.runners.deim_logit_localization_kd \
    --teacher-cfg configs/deim_dfine/deim_hgnetv2_m_traffic_light.yml \
    --teacher-ckpt runs/detect/deim_dfine_m-r1/best_stg2.pth \
    --student-cfg configs/deim_dfine/deim_hgnetv2_s_traffic_light.yml \
    --rehearsal-on-r1 --epochs 1 --seed 0 \
    --kd-lambda 1.0 --ld-lambda 1.0 --kd-temperature 2.0 --kd-reg-max 32 \
    --dry-run

# Execute（GPU server，CWD 任意；runner 自身会 `cd DEIM`）
uv run python -m components.knowledge_distillation.runners.deim_logit_localization_kd \
    --teacher-cfg ... --teacher-ckpt ... --student-cfg ... \
    --rehearsal-on-r1 --epochs 1 --execute
```

**输出**：`runs/rehearsal_kd_A2b_R1.json`（含 `kd_call_count` scrape 自 launcher stdout）。

**派发结构**（runner 构造的 shell 命令）：

```
cd DEIM && PYTHONPATH=.. torchrun --master_port=7777 --nproc_per_node=1 \
    ../components/knowledge_distillation/integration/deim_kd_launch.py \
    --teacher-cfg <DEIM-rel> --teacher-ckpt <abs path> \
    --kd-lambda --ld-lambda --kd-temperature --kd-reg-max \
    -c <DEIM-rel student cfg> --use-amp -u epoches=N \
    --output-dir ../runs/rehearsal_kd_A2b_deim_s_seed<seed> --seed=N
```

**契约**：
- 单节点 only（`WORLD_SIZE == LOCAL_WORLD_SIZE`），多节点 launcher 直接拒绝
- `--output-dir` + `--seed` 在非 resume 时必填；`--test-only` 在 launcher 层被拒绝
- 教师 strict state_dict 加载（任何 missing / unexpected key 立即 `RuntimeError`）
- 教师 `decoder.num_denoising = 0`（避免训练态 `targets=None` 触发 denoising 崩溃）
- 教师 BN-frozen `.train()` 模式（保证 `pred_corners` 输出，LD 必需）
- `_kd_terms` 在 `pred_logits` / `pred_corners` 缺失或 shape mismatch 时 `RuntimeError`
- launcher 训练完检查 `kd_call_counter[0] > 0`，否则拒绝 `completed`；同时 emit `KD_CALL_COUNT=N` 给 runner 抓取
- resume 时从 `<output_dir>/SEED.txt` 取种子，剥离用户 `--seed`，注入 `--seed=<recorded>`

---

## §七 KD A6 设计 spike — 跨架构投影层可行性

定位：在 1-week PoC 启动前判别 path β（aux DFL head）/ γ（Integral 坍缩）/ δ（PKD only）。

**CLI**：

```bash
uv run python -m components.knowledge_distillation.spikes.a6_design_spike
```

**输出**：`runs/rehearsal_kd_A6_design_spike.json`。

**当前结果**（2026-05-11）：`spike_pass = true`，`selected_path = "gamma"`，`a6_priority_recommendation = "P1_with_path_gamma"`。

YOLO26-s `reg_max=1` 无原生 DFL → §七 A6 行原 "FDR↔DFL 分布对齐 + projection MLP" 字面方案不可直接实施；γ = DEIM FDR Integral 坍缩 → bbox L1+GIoU KD + cls-logit KL + PKD FPN 投影 conv，是 shape-feasible 且保留 DEIM 长尾信号的最优路径。详见 `docs/planning/kd_a6_design_spike.md`。

---

## §三 Copy-paste 增强 + 类平衡损失

**当前阶段**：a-stage scaffold（commits b6670a1 + d8b3c02，C3 AGREED-CLEAN at iter-11）。

**入口现状**：`components/copy_paste_balance/runners/ablation.py::main()` 抛 `NotImplementedError("b-stage")`。

**已落地（a-stage）**：
- `config.py` / `data/` / `gates/` / `modules/`
- `gates/_copy_paste_decision_schema.json` 决策 schema
- `configs/data_R2_class_weights.yaml` 配置 stub
- `gates/ablation_gate.py` 决策规则
- `main.py train --copy-paste β --cls-weight <yaml>` 训练侧已通

**待实现（b-stage CLI）**：runner orchestration 读取 5 个 per-arm eval JSON、cross-arm invariant 校验、3-arm β-sweep 聚合、决策 JSON 写出。预估 ~250 LOC，独立 review-conflict 周期。

**激活前置**：R2 数据 freeze 后，把 5 个训练 arm（no_aug / cp_only / cp_balanced × {β=0.99, 0.999, 0.9999}）跑出 per-arm eval JSON，再调 aggregator。

---

## §四 硬负样本挖掘

**当前阶段**：a-stage scaffold（commit e802250，C3 AGREED-CLEAN at iter-2）。

**入口现状**：`components/hard_negative_mining/runners/ablation.py::main()` 抛 `NotImplementedError("b-stage")`。

**已落地（a-stage）**：
- `config.py` / `_internals.py` / `data/` / `gates/` / `modules/`
- `gates/_hard_negative_decision_schema.json` 决策 schema
- 2-arm（no_hn / with_hn）§4.7 决策规则代码
- 共享 `runs/_r2_hard_negative_eval_manifest.json` 冻结 FP 评估清单

**待实现（b-stage CLI）**：runner 读取 2 个 per-arm eval JSON、frozen manifest hash 校验、决策 JSON 写出。预估 ~200 LOC。

**激活前置**：R2 数据 freeze + 难场景 FP manifest 冻结后，跑 no_hn / with_hn 两个训练 arm，再调 aggregator。

---

## §五 地图先验门控

**当前阶段**：DEFERRED → R3+（`additional_components_plan.md` §五 + `development_plan.md` Deferred 表）。

**激活前置**：地图 prior 数据源 + GPS topic 接通；R3+ 范围。

---

## §六 SAHI 切片推理

**当前阶段**：gated；a-stage 尚未启动。

**激活前置**：R2 真机回放暴露小目标 long-tail FN 后启动 a-stage。kickoff §2.5 `SAHI` 行处于 `gated §六 a-stage` 状态。

---

## §七 KD 剩余 cell（A0 / A3 / A4 / A5 / A6-runner / A7）

| Cell | runner 文件 | 触发条件 | 状态 |
|---|---|---|---|
| A0 | `deim_baseline_golsd_off.py` | DEIM 路径 baseline（关 GO-LSD 对比） | scaffold STUB |
| A3 | `pearson_feature_kd.py` | always（P0） | scaffold STUB |
| A4 | `logit_plus_feature_kd.py` | `max(A2/A3) lower-CI > A1 point` + 安全类 delta ≥ -0.5 pp | scaffold STUB |
| A5 | `progressive_multi_teacher.py` | A4 通过全部 6 gate | scaffold STUB；drawdown 时优先丢 |
| A6 runner | `cross_arch_feature_kd.py` | A6 spike pass（已 pass）+ A4 通过 | scaffold STUB；1-week PoC 待启动 |
| A7 | `takd_large_teacher.py` | DEIM-L 训练完成（`runs/detect/deim_dfine_l-r1/best_stg2.pth` 存在） | scaffold STUB；gated |

每个 runner 当前 `main()` 抛 `NotImplementedError(...)` 并指向 `docs/planning/additional_components_plan.md` §七 cell 矩阵。

---

## §八 多相机融合

**当前阶段**：blocked on autonomy team（相机配置、外参、ROS2 topic 锁定）。

**激活前置**：见 `additional_components_plan.md` §八。R3+。

---

## §九 / §十 / §十一

| 项 | 状态 | 入口 |
|---|---|---|
| §九 自适应推理 / ROI | DEFERRED → R3+ | n/a |
| §十 INT8 QAT | DEFERRED → R3+ | n/a |
| §十一 规划器先验融合 | DEFERRED → R3+ | n/a |

R2 不消费这三项；carry-forward token 通过 `scripts/_r2_carry_forward_schema.json` 跟踪。

---

## TSM（时序优化 §1，平行轨道）

**当前阶段**：v1.5 plan AGREED；`components/temporal_shift_module/` scaffold 已落地（modules / gates / data / runners / patches 五个子目录齐全）；runner main 抛 `NotImplementedError`。

**入口**：
- `components/temporal_shift_module/runners/concept_validation.py` — phase-1a（R1 demo + synthetic clip 概念验证）
- `components/temporal_shift_module/runners/full_dataset_train.py` — phase-1b（R2 数据全量训练）
- `components/temporal_shift_module/runners/streaming_engine_export.py` — phase-2（causal-end-to-end TRT export，c2 zeroing）

**激活前置**：phase-1a 立即可启动（kickoff §2.5 `TSM 1-A` 行 active；R1 demo + synthetic clip）。

详见 `docs/planning/temporal_optimization_plan.md` §1。

---

## HMM smoother（时序优化 §2，平行轨道）

**当前阶段**：scaffold STUB。

**入口**：`components/hmm_smoother/runners/ablation.py::main()` 抛 `NotImplementedError`。

**激活前置**：kickoff §2.5 `HMM` 行 active；synthetic flicker / transition fixture 上的 phase-1。

---

## 当前可立即执行的入口（4 个）

```bash
# 1) A1 wall-clock anchor（YOLO 或 DEIM 任意一家）
uv run python -m components.knowledge_distillation.runners.scratch_baseline \
    --family yolo --rehearsal-on-r1 --epochs 1 --execute

# 2) A2a YOLO logit KD
uv run python -m components.knowledge_distillation.runners.yolo_logit_kd \
    --teacher-ckpt runs/detect/yolo26m-r1/weights/best.pt \
    --rehearsal-on-r1 --epochs 1 --execute

# 3) A2b DEIM joint KD（GPU server + DEIM venv）
uv run python -m components.knowledge_distillation.runners.deim_logit_localization_kd \
    --teacher-cfg configs/deim_dfine/deim_hgnetv2_m_traffic_light.yml \
    --teacher-ckpt runs/detect/deim_dfine_m-r1/best_stg2.pth \
    --student-cfg configs/deim_dfine/deim_hgnetv2_s_traffic_light.yml \
    --rehearsal-on-r1 --epochs 1 --execute

# 4) A6 cross-arch design spike（CPU only，已 executed 但可复跑）
uv run python -m components.knowledge_distillation.spikes.a6_design_spike
```

所有其他入口当前都是 `scaffold STUB`（`NotImplementedError`）或 a-stage（runner CLI 缺失）— 不能直接 execute。

---

## 衔接

- `docs/planning/additional_components_plan.md` — WHAT spec（cell 矩阵、决策规则、生命周期）
- `../_archive/pre_r2_kickoff_checklist.md (2026-05-12 归档)` §2.5 — pre-R2 rehearsal 启动清单 + 阶段标记
- `docs/planning/kd_a6_design_spike.md` — A6 cross-arch path γ 设计文档
- `docs/planning/temporal_optimization_plan.md` — TSM / HMM 时序优化 plan
- `docs/planning/development_plan.md` Deferred 表 — §五 / §九 / §十 / §十一 R3+ 入口
- `scripts/_r2_carry_forward_schema.json` — carry-forward 13-token 枚举
