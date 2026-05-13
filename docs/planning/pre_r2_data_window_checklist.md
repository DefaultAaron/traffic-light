# R1 关闭 → R2 数据就绪 — GPU 窗口利用 checklist

> **触发**：R1 closed 2026-05-12 + R2 自采数据未到 + GPU 全时可用。
> **退出**：R2 manifest freeze 后切入 `docs/planning/development_plan.md` §R2。
> **本文件性质**：临时检查表，写 WHAT-not-WHY。每项 done/skipped 后划除并补 commit。
> **决策口径**：继承 `additional_components_plan.md` §三 / §四 / §七 v2 cell matrix；本 checklist **新增** "GPU 窗口早期排除" 规则（§排除规则），不修改 plan §七 cell 触发条件。新规则只能给 cell 打 `low-prior` / `negative-on-r1` tag，**不**决定 cell 是否上 R2。

---

## 数据适用性（applicability）

| | R1 数据可执行 | R2 数据就绪后才能执行 |
|---|---|---|
| KD A2a (YOLO26-s ← YOLO26-m, cls KL) | ✅ scratch student, 单帧 | — |
| KD A2b (DEIM-S ← DEIM-M, LD + cls) | ✅ 单帧 | — |
| DEIM-S no-KD scratch baseline refresh | ✅ 条件性，见 §A2 触发 | — |
| Copy-paste β-sweep (3-arm) | ⚠️ runner a-stage stub → Track B 落 b-stage 才跑 | — |
| Hard-neg mining (2-arm) | ⚠️ runner a-stage stub + 依赖 FP-harvest manifest | — |
| KD A6 cross-arch / A7 same-family / A4 progressive | ❌ runner stub；plan §七 trigger 未到（A6: "A4 通过 6 gate"）；本窗口不解锁 | R2 / R3 |
| TSM full-train ablation | ❌ R1 为单帧 stills，无时序标签；伪标签 / raw LISA 序列已退役，均拒绝 | ✅ R2 视频 / 序列 |
| TSM 激活 tripwire (synthetic) | ✅ synthetic fixture，**仅 plumbing diagnostic**，结论字段 `non_decision_diagnostic: true` + `r2_full_train_readiness: false` + `forbidden_inference: "tripwire pass does not unlock TSM full-train or R2 selection"` | — |
| SAHI c0 grid precheck on R1 demos | ⚠️ scaffold 不存在；R2 replay trigger 未现 → 不做 | R2 / R3 |

**R1 数据策略约束**：R1 BSTLD/S2TLD/LISA 已正式退役（memory `feedback_data_replacement_policy.md`），仅作"排除选项"用，**不**作 R2/R3 选择证据。每条 Track A done 项的 `runs/<run>/ablation_record.json` 应带：
- `evidence_scope: "r1_retired_exclusion_only"`
- `r2_selection_eligible: false`
- `rehearsal_on_r1: true`
- `preR2_tag_status`: enum `not-evaluated` / `evaluated-no-tag` / `negative-on-r1` / `negative-on-r1-safety-regression` / `low-prior` / `held:TBD-gated`
- `r2_decision_status: "not-evaluated"`（R2 选型轮自己重写）

`held:TBD-gated` 是行**生命周期 hold 状态**（B-k1 未落地时占位），不是决策 tag — 同一行从 `held:TBD-gated` backfill 到 `negative-on-r1*` / `evaluated-no-tag` / `low-prior` 是合法转移；B-k1 validator 必须接受 hold 作为输入并按此转移。

**Schema 现状**：上述字段当前是**人工约定**，**无 hook 强制**。B-k1（KD b-stage gate 落地）时同步加 JSON schema + 提交期校验；在 B-k1 完成前 ablation row 走人工 review。

---

## Track A — GPU rig，ablation training（顺序）

### A0 — 5-epoch wall-time 探测（gate before queue commit）

启动主队列前，跑 5-epoch smoke on `yolo_logit_kd.py` + `deim_logit_localization_kd.py`：
- 取 **epoch 2–5 中位 walltime + 固定 startup overhead** 作为 per-epoch 投影（不用 epoch-1，warm-up 偏高）。
- Queue blocking 条件：100-epoch 外推 > **1.5×** epoch 2–5 中位投影，**且** epochs 2–5 walltime 不在 ±15% 稳定带内 → codex-rescue，不直接放行。
- 单一 1.5× 超阈 + 稳定带通过 → 主队列可放行，记录在 §wall-time 段。

### A1–A3 — 当前可启动（runner 已实装）

| # | preR2 ID | KD matrix | 配置 | Runner | planning band（A0 之后更新） |
|---|---|---|---|---|---|
| A1 | `preR2-K-A2a` | A2a | YOLO26-s scratch ← YOLO26-m, cls KL only | `components/knowledge_distillation/runners/yolo_logit_kd.py` | 11–14 h |
| A2 | `preR2-D0` | (DEIM no-KD baseline) | DEIM-S scratch, A2b runner-cfg, KD off | `components/knowledge_distillation/runners/scratch_baseline.py` | 35–42 h |
| A3 | `preR2-K-A2b` | A2b | DEIM-S ← DEIM-M, LD on FDR + cls KL | `components/knowledge_distillation/runners/deim_logit_localization_kd.py` | 36–48 h |

**A2 触发规则**（避免 35–42 h 浪费）：A1 完成后，diff `scratch_baseline.py` 当前 cfg vs R1 DEIM-S full-train `runs/detect/deim_dfine_s-r1/args.yaml` 申报字段。
- 完全一致 → 复用 R1 baseline (mAP50=0.848)，跳过 A2，直接进入 A3。
- 任一字段不一致 OR A3 完成后结果与 0.848 落在 ±2 pp 模糊带内 → 触发 A2 全跑。

**主队列 planning band ≈ 47–62 GPU-h（A2 跳过）/ 82–104 GPU-h（A2 全跑）**。

### A4–A5 — Track B 完成后追加

| # | preR2 ID | 配置 | Runner | 触发依赖 |
|---|---|---|---|---|
| A4 | `preR2-CP` | Copy-paste β-sweep 3-arm | `components/copy_paste_balance/runners/ablation.py` | **B-c1 b-stage AGREED** |
| A5 | `preR2-HN` | Hard-neg 2-arm: no_hn / with_hn | `components/hard_negative_mining/runners/ablation.py` | **B-h1 b-stage AGREED + B-h2 FP-harvest manifest 落盘** |

**Idle-GPU 规则**：A1–A3 done 时检查 B-c1 / (B-h1+B-h2) 的 **owner-stamped ETA**（Track B developer 提交期手写 estimate，pinned 到 commit message 或 `MEMORY.md` track-B 行）。
- Owner ETA > **12 h** → 对应 A4 / A5 立即 `deferred-to-r2-plan`，按 §handoff 写 `_r2_carry_forward.json`，**不空转等**。Track A 立刻关闭。
- Owner ETA 缺失 / 不可知 → 默认 `deferred-to-r2-plan`（不允许"未知就等"的 loophole）。

### §排除规则（GPU 窗口早期 tag 规则，pre-committed before Track A 启动）

每条 Track A 任务必须满足以下统计要求才能触发 tag（否则 tag 不生效，单点估计噪音零容忍）：

1. **CI 方法**：种子 ≥ 3 OR 一次跑 + R1 val 上 bootstrap CI 1000× (95%)。
2. **聚类**：bootstrap **按 source video / 站点 cluster** 当 metadata 可得（LISA 派生帧有 video id）；metadata 缺失 → 记 `ci_method: "image_bootstrap_low_confidence"`，**禁止** 在 marginal（threshold ±2 pp 内）触发 `negative-on-r1`。
3. **稀有类**：support < 30 的类 → 标 `insufficient_support: true`，**不参与** safety-class 改进 / 退化检查。
4. **Safety-class 双向规则**：safety set = `additional_components_plan.md` §安全类（含 redLeft / greenLeft 全部 arrow + barrier）。**不是双向同时阻断**，而是**两路独立判**（避免清晰回退反而保留 cell 的反向逻辑）：
   - **清晰改进** = 任一 support ≥ 30 safety 类 `AP_CI_low > baseline_AP_CI_high` → **阻断** `negative-on-r1`（cell 保留，可能在 R2 还是有救）。
   - **清晰退化** = 任一 support ≥ 30 safety 类 `AP_CI_low < baseline_AP_CI_low − 0.5 pp` → **强制** `negative-on-r1-safety-regression`（更强 tag，独立于 aggregate mAP；artifact 必带退化类名 + 退化 AP CI）。
   - 两路都未触发 + aggregate `mAP50_CI_high < baseline − 1.0 pp` → `negative-on-r1`（aggregate-only）。

具体 cell 触发：

| Cell | tag 触发条件（与上述统计前置同时满足） |
|---|---|
| A1 `preR2-K-A2a` `negative-on-r1` | `student_mAP50_CI_high < (R1 YOLO26s-r1 0.849 − 1.0 pp) = 0.839` |
| A3 `preR2-K-A2b` `negative-on-r1` | `student_mAP50_CI_high < (A2 mAP50_CI_low − 1.0 pp)`（A2 跳过时回退到 R1 DEIM-S 0.848，threshold 0.838） |
| A4 `preR2-CP` `negative-on-r1` | 三 arm `mAP50_CI_high` 全部 < `no-copy-paste matched control mAP50_CI_low − 1.0 pp`；matched control = 同种子同 cfg 关 copy-paste flag 跑一次 |
| A5 `preR2-HN` `negative-on-r1` | `with_hn mAP50_CI_high < no_hn mAP50_CI_low` **且** demo FP-harvest 集 FP rate（denominator = 总 frame × 总 class，CI 95% binomial）下降 < 20% |

**Tag 语义（pin）**：写入 commit message + `ablation_results.md` 对应 § + `ablation_record.json:preR2_tag_status`。**Tag 仅影响 R2 选型轮 prior，不影响本窗口剩余 cell 是否启动**。R2 plan 不得仅凭 R1 tag drop / skip cell — drop / skip 需 **R2-frozen evidence** 或独立的资源 drawdown 规则。

**B-k1 未完成时**：A1/A3 done 但 KD gate 未落地 → ablation row 标 `held:TBD-gated`，必填 `raw_metrics_path`, `gate_blocker: "B-k1"`, `backfill_deadline: <date>`；在 B-k1 done + gate 应用前 **禁止** 应用 `negative-on-r1`。

### 重启 / 中断协议

- Trainer wrapper 启动前写 `runs/<run>/SEED.txt`（exec-trainer 约定 CLAUDE.md §reproducibility）；`--resume` 不重写 SEED。
- `deim_logit_localization_kd.py` 当前 `subprocess.run(capture_output=True)` (line 277 TODO) 阻塞实时输出 → 启动 A3 前先 5-epoch live-output smoke 验证 resume 路径不丢 KD state；不通过 → codex-rescue。
- Wall-time > A0 探测 band 1.5× → codex-rescue。

---

## Track B — Local Mac，scaffold development（并行，无 GPU）

`*b-stage*` 项需 **B2 + C3 review-conflict 循环** 至 AGREED；其余项标准 review。

| # | preR2 ID | 任务 | 输出 | 依赖 |
|---|---|---|---|---|
| B-c1 | `preR2-B-CP` | `components/copy_paste_balance/runners/ablation.py` b-stage | runner body + tests + commit | — |
| B-h1 | `preR2-B-HN1` | `components/hard_negative_mining/runners/ablation.py` b-stage | runner body + tests | — |
| B-h2 | `preR2-B-HN2` | FP-harvest pipeline：扩 `components/hard_negative_mining/data/eval_manifest.py`，跑 YOLO26s-r1 over R1 demos，frozen FP manifest | manifest JSON + 文档 | B-h1 schema |
| B-k1 | `preR2-B-KD` | KD `components/knowledge_distillation/gates/`：`_kd_decision_schema.json` + `ablation_gate.py` + `decision_gate.py`；同步加 ablation-record schema 强制 `evidence_scope` 等字段 | gate 套件 + schema + tests | — |
| B-t1 | `preR2-B-TSM` | TSM concept_validation tripwire smoke：synthetic fixture，c2 zeroing + activation tripwire schema v1.1；artifact 必带 `non_decision_diagnostic: true` + `r2_full_train_readiness: false` + `forbidden_inference` 字段 | `runs/tsm_tripwire_<date>.json` + ablation §七 回填 | — |
| B-r1 | `preR2-B-R2` | `scripts/_r2_decide_precision.py` + `_r2_verify.py` **CPU b-stage 单元测试**（synthetic fixture only）；**final precision executor 仍 blocked on R2 eval / timing / audit / build-variance** | b-stage tests pass | `scripts/_r2_decision_schema.json` |

**Deferred / 不在本窗口**：
- `export_yolo.sh` + `export_deim.sh` sidecar offline validation — R2 选型轮一起；收益低。
- SAHI c0 grid precheck — scaffold 不存在，trigger 未现。
- KD A6 / A7 / A4 progressive runner 实装 — plan §七 trigger 未到。
- R1 pseudo-video TSM diagnostic — scope creep。

---

## 退出条件（exit gates）

- [ ] Track A：A1 + A3 **全部 done**；A2 视触发规则 done 或 skipped-by-config-match。
- [ ] Track A：A4 / A5 在 §Idle-GPU 规则下 done 或 `deferred-to-r2-plan`。
- [ ] Track B：B-c1 + B-h1 + B-k1 + B-t1 + B-r1 done；B-h2 仅在 A5 启动时必做。
- [ ] 每条 Track A done 项写入 `docs/reports/ablation_results.md` 对应 §；带 `evidence_scope`；TBD-leftover 仅允许 `held:TBD-gated` 状态（必带 `backfill_deadline`）。
- [ ] **Pre-B-k1 强制 tripwire**：A1/A3 done 但 B-k1 未落地时，`commit` 之前必须人工核对 row 是否含 `gate_blocker="B-k1"` + `raw_metrics_path` + `backfill_deadline` 三字段，且 `preR2_tag_status` ∈ {`not-evaluated`, `held:TBD-gated`}；缺一不可。B-k1 落地时连带一次性 backfill validator 脚本（`scripts/eval/_preR2_ablation_row_validator.py`）补强机械校验，作为 B-k1 的 B2+C3 review 范围之一。
- [ ] R2 manifest freeze → 本文件归档至 `docs/_archive/`，§handoff 落地。

### §handoff（R2 manifest freeze / Idle-GPU 中断 → R2 plan）

未完成 / deferred 项写入 `runs/_r2_carry_forward.json`，**沿用** `scripts/_r2_carry_forward_schema.json` v1.5 closed-enum 不另起新字段：

| preR2 item | `item_id` | `status` | `blocked_on`（closed enum） | `next_entrypoint` |
|---|---|---|---|---|
| A1 / A3 中断 | `preR2_kd_<cell>` | `blocked` | `[r2_data_freeze]`（R2 round 启动时回退） | `additional_components_plan §七 v2 cell A2a/A2b row` |
| A4 deferred | `preR2_copy_paste_sweep` | `blocked` | `[r2_data_freeze]` | `additional_components_plan §三` |
| A5 deferred — B-h2 未完成 | `preR2_hard_neg_sweep` | `blocked` | `[hard_neg_manifest_hash]` | `additional_components_plan §四` |
| A5 deferred — B-h2 已落但 R2 触发 | `preR2_hard_neg_sweep_postharvest` | `blocked` | `[r2_data_freeze]` | `additional_components_plan §四` |
| A5 deferred — 两条 unblock 路径都成立（B-h2 + R2 都可解锁，任一）| `preR2_hard_neg_sweep_dual` | `blocked` | `[hard_neg_manifest_hash, r2_data_freeze]` + `unblock_logic: "any"` + `unblock_evidence_path: <FP manifest 或 R2 manifest doc>` | `additional_components_plan §四` |
| B-c1/B-h1/B-k1/B-t1/B-r1 未完成 | `preR2_b_<scaffold>` | `scheduled` | `[]`（round-internal） | `development_plan §R2 采集 / 标注 / 训练` |

**checklist-本地 状态** （不进 `_r2_carry_forward.json`）：每行附加 `raw_metrics_path`（已起跑时）+ `epochs_completed`，便于 R2 plan ingest 段读 partial progress。

---

## CLI 启动命令（仅已实装 runner）

> 出处验证：rehearsal 一次性命令记录在 `runs/rehearsal_kd_{A1,A2a,A2b,A6}.json:train_command`；本节将 epochs 提到 100 + 改 `--output` / `--output-dir` 路径用于 preR2 full-train 输出。Stub runner 不在此列表，必须先完成 Track B 实装。

### A0 — 5-epoch wall-time 探测（先于主队列）

```bash
# YOLO 路径（A1 + A2-yolo / 共用 yolo_logit_kd 路径，验证 KD callback 不漏触发）
uv run python components/knowledge_distillation/runners/yolo_logit_kd.py \
    --teacher-ckpt runs/detect/yolo26m-r1/weights/best.pt \
    --output runs/preR2_A0_yolo_5ep_probe.json \
    --epochs 5 --seed 0 --execute

# DEIM 路径（A2-deim + A3 共用 deim_kd_launch；同时验证 KD λ/T 启动 + resume 路径不丢 KD state）
cd DEIM && \
PYTHONPATH=.. torchrun --master_port=7778 --nproc_per_node=1 \
    ../components/knowledge_distillation/integration/deim_kd_launch.py \
    --teacher-cfg configs/deim_dfine/deim_hgnetv2_m_traffic_light.yml \
    --teacher-ckpt ../runs/detect/deim_dfine_m-r1/best_stg2.pth \
    --kd-lambda 1.0 --ld-lambda 1.0 --kd-temperature 2.0 --kd-reg-max 32 \
    -c configs/deim_dfine/deim_hgnetv2_s_traffic_light.yml \
    --use-amp -u epoches=5 --output-dir ../runs/preR2_A0_deim_5ep_probe \
    --seed=0
```

退出条件：检查 epochs 2–5 walltime 中位 + 稳定带（±15%），按 §A0 规则放行或 codex-rescue。

### A1 — `preR2-K-A2a` YOLO26-s ← YOLO26-m, cls-logit KL（100 epoch）

```bash
uv run python components/knowledge_distillation/runners/yolo_logit_kd.py \
    --teacher-ckpt runs/detect/yolo26m-r1/weights/best.pt \
    --output runs/preR2_K_A2a_R1.json \
    --epochs 100 --seed 0 --execute
```

种子 ≥ 3 跑法：循环 `--seed 0`, `--seed 1`, `--seed 2`，输出到同一 `--output` 文件（runner `_save_entry` 按 seed key 合并；再用同 file 输入到 B-k1 gate）。

### A2 — `preR2-D0` DEIM-S no-KD scratch baseline（条件触发，100 epoch）

A1 完成后先 diff cfg：

```bash
# 触发判定（与 R1 DEIM-S full-train 现存 args 对比）
diff <(uv run python components/knowledge_distillation/runners/scratch_baseline.py \
        --family deim --size s --epochs 100 --seed 0 \
        --output runs/preR2_D0_deim_s_R1.json --dry-run \
        | grep -E "^\s+--" | sort) \
     <(grep -E "^(epochs|imgsz|batch|seed|optimizer|lr0):" runs/detect/deim_dfine_s-r1/args.yaml | sort)

# 一致 → 跳过 A2，复用 R1 DEIM-S baseline (mAP50=0.848)
# 任一字段不一致 OR A3 mAP50_CI ∩ [0.828, 0.868] ≠ ∅ → 触发 A2 全跑：
uv run python components/knowledge_distillation/runners/scratch_baseline.py \
    --family deim --size s --epochs 100 --seed 0 \
    --output runs/preR2_D0_deim_s_R1.json --execute
```

### A3 — `preR2-K-A2b` DEIM-S ← DEIM-M, LD on FDR + cls KL（100 epoch）

```bash
cd DEIM && \
PYTHONPATH=.. torchrun --master_port=7778 --nproc_per_node=1 \
    ../components/knowledge_distillation/integration/deim_kd_launch.py \
    --teacher-cfg configs/deim_dfine/deim_hgnetv2_m_traffic_light.yml \
    --teacher-ckpt ../runs/detect/deim_dfine_m-r1/best_stg2.pth \
    --kd-lambda 1.0 --ld-lambda 1.0 --kd-temperature 2.0 --kd-reg-max 32 \
    -c configs/deim_dfine/deim_hgnetv2_s_traffic_light.yml \
    --use-amp -u epoches=100 --output-dir ../runs/preR2_K_A2b_R1 \
    --seed=0
```

DEIM 路径 seed ≥ 3：`--seed=0`/`1`/`2` 各跑一次，`--output-dir` 后缀 `_seedN`；B-k1 gate 聚合三 seed。

### A4 / A5 — `preR2-CP` / `preR2-HN`

⚠️ Runner 当前 a-stage stub（`raise NotImplementedError("b-stage")`）。CLI 不可用，等 Track B `preR2-B-CP` / `preR2-B-HN1` b-stage 落地。落地后预期 CLI 形状：

```bash
# A4 — 落地后（占位 spec，B-c1 决定最终 flag 名）
uv run python components/copy_paste_balance/runners/ablation.py \
    --beta low,mid,high --seed-set 0,1,2 \
    --output-root runs/preR2_CP_R1 --execute

# A5 — 落地后（占位 spec，B-h1 决定最终 flag 名；先跑 B-h2 FP-harvest）
uv run python components/hard_negative_mining/data/eval_manifest.py \
    --source-model runs/yolo26s-r1/weights/best.pt \
    --demos demo/**/*.mp4 --output runs/preR2_HN_fp_manifest.json --execute

uv run python components/hard_negative_mining/runners/ablation.py \
    --fp-manifest runs/preR2_HN_fp_manifest.json \
    --arms no_hn,with_hn --seed-set 0,1,2 \
    --output-root runs/preR2_HN_R1 --execute
```

### Track B — 实装工作（无 CLI）

| preR2 ID | 当前状态 | 完成判定 |
|---|---|---|
| B-c1 / B-h1 | runner stub `NotImplementedError("b-stage")` | `uv run python -m pytest components/{copy_paste_balance,hard_negative_mining}/` 通过 + B2+C3 AGREED |
| B-h2 | scaffold `eval_manifest.py` 仅含 schema，无 harvest 实装 | manifest JSON 落盘 + 与 B-h1 schema 一致 |
| B-k1 | `gates/` 空目录 | gate 套件 + `_kd_decision_schema.json` + 一次性 backfill validator + B2+C3 AGREED |
| B-t1 | 全部 runner / gate 均 stub | `concept_validation.py` runner body + tripwire smoke fixture pass |
| B-r1 | scripts 中 `_r2_decide_precision.py` 与 `_r2_verify.py` 主 runner 落盘但无 CPU 单元测 | `uv run python -m pytest scripts/test_*.py` 三测全过（拆 case A/B/C/D 各一） |

---

## 变更记录

| 日期 | 动作 |
|---|---|
| 2026-05-13 | 追加 §CLI 启动命令（A0/A1/A2/A3 实装命令 + A4/A5/Track-B 占位 spec 与状态判定） |
| 2026-05-13 | 文件创建：answering "GPU 窗口怎么用" + TSM full-train R1-blocked + 经 codex-plan-conflictor 三轮重构：pass-1 REJECT（移 A6/A7/copy-paste/hard-neg stub）→ pass-2 APPROVE-WITH-AMENDMENTS（聚类 bootstrap + 稀有类 insufficient_support + B-k1 schema 落地前 `held:TBD-gated` + A2 触发规则 + Idle-GPU 12 h + handoff 沿用 `_r2_carry_forward_schema.json`）→ pass-3 5×ACCEPT-WITH-AMENDMENT（safety-class 改单向逻辑：清晰改进 blocks tag / 清晰退化 trigger `negative-on-r1-safety-regression` 更强 tag；A5 OR semantics 拆成三行 + `unblock_logic:"any"`；§R2 ingest anchor 改指现存 §R2 采集/标注/训练；ETA owner-stamped 默认 deferred；pre-B-k1 tripwire 三字段强制 + B-k1 内一次性 backfill validator） |
