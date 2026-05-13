# R1 Full-Train Ablation Results

> **Re-opened 2026-05-13** — R1 数据上的 full-epoch (100+) 早期 ablation 记录，R2 数据就绪前的 GPU 利用窗口。
> 1-epoch smoke / wall-time rehearsal 结果仍由 runners 写入 `runs/rehearsal_kd_*.json`，本文档只收录 **full-train + eval** 跑通后的数值。
> 决策依据：每 cell 跑完写入对应 §，触发对应 component 的 b-stage gate 自动判定 `pass / fail / inconclusive`。Gate 缺位时（如 KD），结果先入表，等 KD gate 落地再回填 decision case。

## §一 KD A2a — YOLO26-s ← YOLO26-m cls-logit KL (R1, scratch student)

**历史参考行**（2026-05-11 之前的 KD 实现，pre-7-iter-scaffold）：

| 类别       | 图片数 | 实例数 | 准确率 | 召回率 | mAP@50 | mAP@50:95 |
| ---------- | ------ | ------ | ------ | ------ | ------ | --------- |
| all        | 11,091 | 29,081 | 0.927  | 0.710  | 0.827  | 0.595     |
| red        | 5,235  | 11,601 | 0.936  | 0.964  | 0.980  | 0.795     |
| yellow     | 355    | 756    | 0.888  | 0.889  | 0.919  | 0.557     |
| green      | 5,613  | 12,912 | 0.916  | 0.955  | 0.971  | 0.738     |
| redLeft    | 2,533  | 3,216  | 0.911  | 0.954  | 0.972  | 0.824     |
| greenLeft  | 433    | 586    | 0.838  | 0.915  | 0.936  | 0.613     |
| redRight   | 6      | 6      | 1      | 0.298  | 0.361  | 0.170     |
| greenRight | 3      | 4      | 1      | 0      | 0.653  | 0.469     |

**对比基线**：R1 closed YOLO26s-r1 (无 KD) mAP50=0.849 / mAP50-95=0.608 — 历史 A2a 反而 −2.2 pp vs no-KD，说明早期 KD 实现要么超参未调，要么真的是负迁移。**re-run target**：在 7-iter-scaffold 实现 + tuned λ/T 下重测，目标 ≥ no-KD baseline。

**Full-train re-run (2026-05-13+)** — TBD（trial-1 命中表后回填）。

## §二 KD A2b — DEIM-S ← DEIM-M LD on FDR + cls-logit KL (R1)

TBD — 1-epoch smoke `runs/rehearsal_kd_A2b_R1.json` 已跑通（kd_call_count=5548，walltime 1503 s）；full-train 触发 R1→R2 桥接 §A。

## §三 KD A6 — DEIM-M → YOLO26-s cross-arch (R1)

TBD — design spike `runs/rehearsal_kd_A6_design_spike.json` 验证了 head probe + PKD projection；full-train 启用 `components/knowledge_distillation/runners/cross_arch_feature_kd.py`（path γ：L1+GIoU bbox KD + cls KL + PKD projection）。

## §四 KD A7 — DEIM-L → DEIM-M/S same-family (R1)

TBD — R1 closed 后 unlocked（DEIM-L `best_stg1.pth` 已落盘）。触发 R1→R2 桥接 §B。

## §五 Copy-paste + Class-balance β-sweep (3-arm, R1)

TBD — runner `components/copy_paste_balance/runners/ablation.py` AGREED-CLEAN at iter-11；β ∈ {α, mid, high}；触发 R1→R2 桥接 §D。

## §六 Hard-negative mining (2-arm: no_hn / with_hn, R1 + demo)

TBD — runner `components/hard_negative_mining/runners/ablation.py` AGREED at iter-2；blocked on FP-harvest manifest（Track B 任务）；触发 §E。

## §七 TSM 激活 tripwire (R1, **仅 synthetic concept-validation**)

TBD — runner `components/temporal_shift_module/runners/concept_validation.py` + gate `concept_validation_gate.py`，输入 synthetic `(B, T, C, H, W)` fixture，验证 causal-end-to-end c2 zeroing + activation tripwire schema v1.1。**TSM full-train ablation 在 R1 数据上不可行**（R1 merged 为单帧 stills，无时序标签）→ 仅 R2 自采视频/序列数据就绪后启动，本 § 仅记录 tripwire pass/fail。触发 R1→R2 桥接 §F。

## 变更记录

| 日期 | 动作 | 提交 |
|---|---|---|
| 2026-05-12 | 早期 KD A2a stub 归档至 `docs/_archive/` | 7bbf752 |
| 2026-05-13 | 从 archive 还原，重整为 R1 full-train ablation 实时日志 | (本提交) |
