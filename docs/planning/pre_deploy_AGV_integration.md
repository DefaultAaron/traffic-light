# Pre-Deploy AGV 集成规划（R4+ 部署阶段 carry-forward 停车场）

> **状态**：stub — 仅作为 R2 / R3 carry-forward 的接收目录；激活由**证据触发**而非轮次编号（见 §1 范围）。2026-05-12 deadline-bound 规划已 retire；本文档不绑定 R4 / R5 编号。
> **创建**：2026-05-12
> **触发**：R2/R3 round 报告产出的 deploy-tuning 任务、定 Case A 部署候选、以及无法在 R2/R3 范围解决但需在实车 / 半实物日志回放后评估的部署模式选项。

---

## 1. 范围

本文档收容**仅**在以下条件具备后才有意义的工程项：

- 实车 / 半实物日志回放管线就位（不只是 frozen val set；要带速度、ODD tag、GPS、tracker 状态的连续 session）
- 稳定的 ODD 触发源（GPS 速度 / 地图先验 / 场景分类器，三选一已锁定）
- §八 双相机融合在 R2/R3 至少跑过 c1 / c2，Cam-W vs fusion 的 latency / recall 量级已知
- §七 KD round 已闭环或被 deferred，A2a / A2b ship-decision 已 final

R2/R3 round 不在此范围；R2/R3 round 报告把 Case A FP32 ship 候选、Gate #6 deploy-stability trigger、以及本文档下列条目作为 carry-forward 写入。

---

## 2. R4+ carry-forward 条目

### 2.1 运行时相机切换（runtime camera switching；§八 carry-forward）

**来源**：`docs/planning/additional_components_plan.md` §八 c6 + codex-plan-conflictor 2026-05-12 review。

**问题**：Cam-W / Cam-T 在 ODD 不同区段（城市 vs 高速）相对收益不同；R2/R3 首先评估 always-on 双相机 + late fusion + WBF，**若按 §八 c1 outcome 规则未优于 Cam-W-only，则 R2 / Stage 1 采用 Cam-W-only 作为 feasibility baseline**（plan §八 c1 + `runs/_multi_camera_decision.json` `selected_baseline="cam_w_only"` + `stage1_scope="feasibility_baseline"`；该 baseline 仍须通过 development_plan.md §Stage 2.A + 2.B 后方可 ship）。在 Stage 2 通过 + 实车测试后再评估运行时切换：(a) 如果 ship 走 always-on dual fusion，是否切换以摊薄推理成本；(b) 如果 ship 走 Cam-W-only，是否在 ODD 高速段切入 Cam-T。

**激活前置**（必须**全部**满足才能启动）：
- 验证过的 ODD 触发源（GPS 速度 OR 地图先验 OR 实时场景分类器；三选一锁定）
- hysteresis + minimum dwell time 规则（防切换抖动）
- mid-approach handoff failure policy（接近 TL 时切换的 tracker 状态续接策略）
- engine residency policy（双 engine 常驻 vs 冷热切换；后者引入切换延迟，前者抵消成本节省）
- R2/R3 c6 offline ablation 已估出 recall-latency 权衡区间

**Out of scope for R2/R3**：以上前置条件均不在 R2/R3 范围；R2 only allows c6 offline log-replay diagnostic（不进部署）。

**决策路径**：R4 round 启动后，若激活前置全满足 → 进 R4 plan 单独 cell；否则继续 carry-forward 至 R5。

---

## 3. 附录（reserved）

后续 R2/R3 round 报告附加的 carry-forward 条目（如 Case A FP32 soak、Gate #6 deploy-tuning result、KD pre-R2 rehearsal 暴露的部署问题）追加在 §2 下方。
