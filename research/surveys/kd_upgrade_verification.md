# KD Upgrade 提案验证（GPT proposal）

**审计对象**：`docs/kd_upgrade_recommendation.md`（GPT-5 生成的 KD 升级提议）

**审计日期**：2026-05-12

**审计输入**：
- 当前锁定计划：`docs/planning/additional_components_plan.md` §七 (2026-05-11 锁定 v2)
- A6 设计 spike：`docs/planning/kd_a6_design_spike.md` + `runs/rehearsal_kd_A6_design_spike.json`（path γ 选定）
- LANDED 状态：`docs/ops/additional_components_guide.md`（A1 / A2a / A2b / A6 spike LANDED；其余 6 cells STUB）

**审计方式**：双流并行——`paper-researcher`（学术证据）+ `codex-plan-conflictor`（计划形态对抗审查）。两路独立返回；交叉验证后形成最终采纳决定。

---

## 一、约束摘要（决定多项 reject 的硬约束）

| 约束 | 影响 |
|---|---|
| YOLO26 `reg_max = 1`（无原生 DFL 头，4 个标量回归通道）；A6 spike 已实证 | GPT 提议中 "A2a + DFL/box KD" 与 "A6b FDR-to-DFL KD" 架构上不可行 |
| S-only 部署锁（YOLO26-s + DEIM-D-FINE-S 并行候选）；M/L 仅教师 | A8 反向蒸馏（YOLO-M → DEIM-S）方向与 R1 demo 实证矛盾（demo 显示 DEIM 稳定性弱于 YOLO） |
| YOLO26-l 不合格（R1 mAP50=0.850 < YOLO26-m 0.869） | A7 L-tier 教师只能用 DEIM-L |
| 6 项验收门 2026-05-11 锁定（Gate #1-#6） | 新增 Gate #7/#8/#9 直接违反锁定契约；Gate #6 明确"不阻塞 KD ship" |
| WHAT-not-WHY 计划政策（feedback_plan_what_not_why） | GPT 提议大量 rationale 段落会污染锁定计划；citations 应入 research/，不入 plans |
| Drawdown 顺序：先丢 A5，再丢 A4；P0+A6+A7 不可同时丢 | GPT 添加 A8 但未指定 drawdown 位置——歧义注入 |

---

## 二、Paper-researcher 学术证据矩阵

按 GPT 提议的每条核心建议给出 verdict + 引用：

### Q1. MGD / FGD / CWD vs PKD（dense detector 特征 KD）

**VERDICT: WEAKLY-SUPPORTED**——四种方法均有可信论文支撑，但 head-to-head COCO 比较包含 PKD 的较少，且 YOLO-family 小目标 / 长尾基准上的对比稀缺。

- Yang et al., 2022, CVPR — **FGD**（Focal and Global KD for Detectors）：RetinaNet-R50 +3.3 mAP；前景/背景解耦掩码。
- Yang et al., 2022, NeurIPS — **MGD**（Masked Generative Distillation）：将 feature mimicking 改写为 masked reconstruction；多检测器上优于 FGD。
- Shu et al., 2021, ICCV — **CWD**（Channel-wise KD for Dense Prediction）：channel-norm 后 KL；COCO RetinaNet +1-2 mAP。
- Cao et al., 2022, NeurIPS — **PKD**（Pearson 相关 KD）：跨异构 head 设计；与 MGD/FGD 在 COCO 上有竞争力。

**reg_max=1 适配性**：四种方法均作用于 FPN/backbone features，与 YOLO26 头结构正交，A3 / A6c 均可直接套用。

**残余 gap**：未找到 YOLO-family 在小目标 / traffic-light 规模基准上的四方对比；当前仅 RetinaNet / FCOS / Faster R-CNN 论据。

### Q2. DETR-variant 上的 query-level KD（D-FINE / RT-DETR / DETR）

**VERDICT: WEAKLY-SUPPORTED**——method 有论文支撑，但 D-FINE-specific ablation 缺失。

- Chang et al., 2023, ICCV — **DETRDistill**（universal KD for DETR families）：Hungarian-matched query KD + logit + feature KD；在 Deformable-DETR / DAB-DETR 上 ablate 出 query-match 单独贡献。
- Wang et al., 2024, CVPR — **KD-DETR**（consistent distillation points）：处理 teacher-student query 数失配；COCO +2-3 AP。
- Peng et al., 2023, "KD for DETRs with Localization Information Transfer" — `[未验证]`，BibTeX 未给出。

**reg_max=1 适配性**：N/A——只与 A2b（DEIM→DEIM）相关。

**残余 gap**：D-FINE 太新，未见 "query KD vs FDR/LD KD" 在 D-FINE 上的隔离 ablation。

### Q3. Pseudo-label bridge（set→dense 跨架构 KD）

**VERDICT: NO-EVIDENCE**

无法引用 ≥2 篇同时满足 (a) DETR teacher + (b) YOLO/dense student + (c) pseudo-label 优先而非 feature/logit alignment + (d) ablate 对比直接 alignment 的论文。最接近的是通用 SSL 文献（FixMatch, Soft Teacher），但那是 self-training 不是 cross-arch KD。GPT 提议的 conf 阈值 0.3-0.5 在 SSL 范围内但未在 KD-bridge 用途下验证。

**reg_max=1 适配性**：pseudo-label 与头结构无关，绕开 FDR↔DFL 失配——这正是 GPT 提出它的原因。但**缺乏先例**意味着 A6a 是经验探索，不是 best practice 复刻。

**残余 gap**：文献空白；如做则属项目原创贡献。

### Q4. 反向跨架构 KD（dense → DETR student）

**VERDICT: NO-EVIDENCE**

无法引用 ≥2 篇 YOLO-style dense teacher → DETR student 测得 FP 抑制 / hard-negative gain 的论文。已知跨架构 KD 文献（DETRDistill, KD-DETR, FGD 跨头）均是大→小同家族或 transformer teacher。

**reg_max=1 适配性**：N/A（YOLO 是 teacher）。

**残余 gap**：方向完全未探索；A8 不应仅凭文献证据升至 P2 之上。

### Q5. 前景过滤 / teacher-conf-min 类 KD（YOLO）

**VERDICT: STRONGLY-SUPPORTED**（前景过滤原则）；**WEAKLY-SUPPORTED**（0.3-0.5 具体阈值）。

- Zheng et al., 2022, CVPR — **LD**（Localization Distillation）：通过 VLR（Valuable Localization Region）限制 KD 至高质量正样本；COCO +1-2 AP。
- Yang et al., 2022, CVPR — **FGD**（同 Q1）：前景/背景解耦掩码是核心论点。
- Dai et al., 2021, CVPR — **GID**（General Instance Distillation）：selectsbreak teacher+student 高分区域；优于 uniform KD。

**reg_max=1 适配性**：前景过滤作用于 cell selection / mask weights，不依赖 head 结构——可直接套用 YOLO26。

**残余 gap**：0.3-0.5 阈值是启发式；LD/GID 用 score-rank / quality-score 而非原始 conf。安全关键长尾类的具体阈值无文献定量结论。

### Q6. D-FINE GO-LSD 与外部 KD 的交互

**VERDICT: NO-EVIDENCE**

GO-LSD 来自 D-FINE 论文（Peng et al., 2025, CVPR — `[年份需 web 二次验证]`）作为**内置**自蒸馏（decoder 层间）。无法引用 ablate GO-LSD on/off **同时** stack 外部 M→S teacher 的研究。D-FINE 论文本身的 ablation 只隔离 GO-LSD 单独贡献。

**reg_max=1 适配性**：N/A（DEIM 侧专属）。

**残余 gap**：A0（GO-LSD-off baseline）由方法论隔离需求驱动，不由文献干扰证据驱动。GPT 建议"主 DEIM 训练保持 GO-LSD on"合理但无文献支撑。

---

## 三、Codex-plan-conflictor 计划形态裁定

按 GPT 提议每条建议给出 VERDICT 与理由：

| 建议 | VERDICT | 关键否决/采纳依据 |
|---|---|---|
| A2a upgrade（DFL+box+feature+filter） | **REJECT** | reg_max=1 使 DFL 不可行；重开 LANDED P0 cell；高成本（头脚手架 + export/TRT 风险 + 新 B2+C3 周期）；无本地证据 |
| A2b upgrade（query KD + encoder feature KD） | **DEFER-TO-R3** | 当前 A2b LANDED 且实证 KD fires；中-高成本；无本地证据现 A2b 弱；R3 候选 |
| A3 PKD → MGD/FGD/CWD swap | **REJECT** | 把 STUB 的 P0 扩成 method search；与 runner identity `pearson_feature_kd` 冲突；无本地证据 PKD 不足 |
| A6 split A6a/A6b/A6c | **ADOPT-WITH-AMENDMENT** | A6b 因 reg_max=1 失败；A6c 与 spike path γ 重叠；建议改 A6 行措辞至 path γ；可选 pseudo-label 作 fallback 诊断（**非新 cell**） |
| A8 reverse | **REJECT** | R1 demo 实证支持 DEIM→YOLO 召回迁移**而非**反向；无文献；分散 A6/A7 焦点；Gate #6 deploy-tuning trigger 已覆盖 FP/稳定性调优 |
| 新 Gate #7/#8/#9 | **REJECT** | 直接违反锁定 6-gate 契约；与 Gate #2 / #3 / #6 语义重叠；blocking 条件可被 bucket 定义 / 阈值 / NMS 参数游戏化 |
| Suggested execution order | **REJECT** | 前置 rejected work；忽略锁定 drawdown 顺序；未尊重当前 LANDED/STUB 状态 |
| "Minimal Patch" 子节 | **REJECT** | 不 minimal；改动 locked cells / gates / runner names / drawdown 语义——实际是 v3 重写 |

**Cross-cutting findings**：
- 违反 WHAT-not-WHY plan policy（rationale-heavy 入 plan）
- 把 LANDED runner 当 editable draft——制造 review debt + reproducibility churn
- 错过部署现实（S-only / M-teacher / DEIM-L gated / YOLO26-l 不合格）
- A8 添加但未指 drawdown 位置——歧义注入

**Decision-rule loopholes（若强加新 gates）**：
- Gate #7 rare-class recall：可被 bucket 定义 / support 阈值 / conf 阈值 / recall 工作点游戏化；与 Gate #2 重叠
- Gate #8 small-object：缺尺寸定义 / 距离代理 / 分层支持数；与 Gate #3 冲突（提升 recall 可能拉升 FP）
- Gate #9 pseudo-label noise audit：样本敏感、可手动偏置；wrong-label rate 依赖审计员策略 + NMS + top-k + teacher 阈值 + GT 冲突计数

---

## 四、最终采纳决定（cross-checked）

| GPT 建议 | Paper 证据 | Conflictor 裁定 | **采纳** |
|---|---|---|---|
| A2a + DFL/box KD | CONTRADICTED | REJECT | **REJECT**（架构不可行） |
| A2a 前景/conf 过滤 | STRONGLY-SUPPORTED | （并入 A2a REJECT） | **DEFER → R3 additive ablation** |
| A2a / A3 feature KD（MGD/FGD/CWD） | WEAKLY-SUPPORTED | REJECT (A3 swap) | **DEFER → R3 challenger ablation**（若 PKD 在 R2 暴露不足） |
| A2b + query KD + encoder feat | WEAKLY-SUPPORTED | DEFER-TO-R3 | **DEFER → R3** |
| A6 split A6a/b/c | A6b CONTRADICTED, A6a NO-EV | ADOPT-WITH-AMENDMENT | **AMEND A6 措辞为 path γ；pseudo-label 列入 R3 备选诊断** |
| A8 reverse | NO-EVIDENCE | REJECT | **REJECT** |
| 新 Gate #7/#8/#9 | n/a | REJECT | **REJECT**（保留 6-gate 契约） |
| 执行顺序 / Minimal Patch | n/a | REJECT | **REJECT** |

**唯一 ADOPT**：A6 行措辞同步至 A6 spike 选定的 path γ。

**唯一 R3 carry-forward 增项**：A2a 前景过滤 / 类 KD 阈值 + A3 challenger（MGD/FGD/CWD）+ A2b query/encoder KD + A6 pseudo-label fallback——四项进 `docs/planning/R3_precision_reproducibility.md`（或专门的 R3 KD ablation plan）作候选 ablation，**不阻塞 R2**。

---

## 五、计划修订（已应用）

参见 `docs/planning/additional_components_plan.md` §七 A6 行 + line 209 A6 priority note 措辞同步。

详细 patch 范围：
1. Cell 矩阵 A6 行：方法栈措辞从"跨架构 logit + FDR↔DFL 分布对齐 + projection MLP"改为"跨架构 cls-logit KL + DEIM FDR Integral 坍缩 → scalar bbox L1+GIoU KD + PKD FPN 投影 conv（path γ）"
2. A6 优先级上调说明（line 209）："DETR query embed ↔ YOLO 多尺度特征"措辞同步至 path γ（FDR Integral 坍缩 + scalar bbox KD + PKD 投影）

无新 cell、无新 gate、无 runner rename、无 drawdown 改动。

---

## 六、引用 BibTeX

```bibtex
@inproceedings{yang22-fgd,
  title={Focal and Global Knowledge Distillation for Detectors},
  author={Yang, Zhendong and Li, Zhe and Jiang, Xiaohu and Gong, Yuan and Yuan, Zehuan and Zhao, Danpei and Yuan, Chun},
  booktitle={CVPR},
  year={2022}
}
@inproceedings{yang22-mgd,
  title={Masked Generative Distillation},
  author={Yang, Zhendong and Li, Zhe and Shao, Mingqi and Shi, Dachuan and Yuan, Zehuan and Yuan, Chun},
  booktitle={NeurIPS},
  year={2022}
}
@inproceedings{shu21-cwd,
  title={Channel-wise Knowledge Distillation for Dense Prediction},
  author={Shu, Changyong and Liu, Yifan and Gao, Jianfei and Yan, Zheng and Shen, Chunhua},
  booktitle={ICCV},
  year={2021}
}
@inproceedings{cao22-pkd,
  title={PKD: General Distillation Framework for Object Detectors via Pearson Correlation Coefficient},
  author={Cao, Weihan and Zhang, Yifan and Gao, Jianfei and Cheng, Anda and Cheng, Ke and Cheng, Jian},
  booktitle={NeurIPS},
  year={2022}
}
@inproceedings{zheng22-ld,
  title={Localization Distillation for Dense Object Detection},
  author={Zheng, Zhaohui and Ye, Rongguang and Wang, Ping and Ren, Dongwei and Zuo, Wangmeng and Hou, Qibin and Cheng, Ming-Ming},
  booktitle={CVPR},
  year={2022}
}
@inproceedings{dai21-gid,
  title={General Instance Distillation for Object Detection},
  author={Dai, Xing and Jiang, Zeren and Wu, Zhao and Bao, Yiping and Wang, Zhicheng and Liu, Si and Zhou, Erjin},
  booktitle={CVPR},
  year={2021}
}
@inproceedings{chang23-detrdistill,
  title={DETRDistill: A Universal Knowledge Distillation Framework for DETR-families},
  author={Chang, Jiahao and Wang, Shuo and Xu, Hai-Ming and Lin, Zehui and Yang, Chenhongyi and Tian, Yiran},
  booktitle={ICCV},
  year={2023}
}
@inproceedings{wang24-kddetr,
  title={KD-DETR: Knowledge Distillation for Detection Transformers with Consistent Distillation Points},
  author={Wang, Yu and Li, Xin and Zhao, Shuai and others},
  booktitle={CVPR},
  year={2024}
}
@inproceedings{peng25-dfine,
  title={D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement},
  author={Peng, Yansong and Li, Hebei and Wu, Peixi and Zhang, Yueyi and Sun, Xiaoyan and Wu, Feng},
  booktitle={CVPR (year unverified — see open question)},
  year={2025}
}
```

**Open citation questions**:
- D-FINE 实际发表 venue / 年份需 web 二次验证（paper-researcher 标注 CVPR 2025 但需确认）
- Peng et al., 2023 "KD for DETRs with Localization Information Transfer" 未通过验证，已剔除
