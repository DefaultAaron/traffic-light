# 交通信号灯识别 — 开发计划 v2.0

## 状态

| 项 | 当前值 |
|---|---|
| 部署平台 | NVIDIA Tegra Orin 64GB |
| 输入 | 8MP 摄像头；推理分辨率 1280 |
| 输出 | `vision_msgs/Detection2DArray` |
| 当前阶段 | R2：10-14 类联合检测准备 / 执行 |
| 截止 | 2026-05-15 实车测试 |
| v2.0 规则 | WHAT-only；生命周期定义见 `additional_components_plan.md` §一 |

| 里程碑 | 状态 / 交付物 |
|---|---|
| P1：3 类基线 | ✅ 完成；YOLO26s 选定；见 `docs/reports/phase_1_report.md` |
| R1 主力：7 类 YOLO26 n/s/m | ✅ 完成；Orin 1280 FP16 约 25 ms/帧；`xyxy` 后处理修复 |
| R1 备选：YOLOv13-s / DEIM-D-FINE-S/M | 🚧 训练中 |
| R1 跟踪：ByteTrack + EMA | ✅ Python + C++ 落地；`inference/tracker/`、`inference/cpp/src/tracker.cpp` |
| R1 决策 | ⏳ 待三轨训练完成后应用规则 |
| R2 范围 | ✅ 下限 10 类锁定；上限 14 类按 PM / 数据触发 |
| R2 数据 | ⏳ 自采数据 + SOP；R1 数据集退役 |
| R2 训练增强 | ⏳ copy-paste、硬负样本、KD P0 cells |
| R2 部署 | ⏳ 目标 Orin 1280 ≤ 50 ms |
| R2 时序优化 | gated；baseline replay 暴露失败模式后启动 |
| R3+ carry-forward | 地图先验、自适应推理、INT8 QAT、规划器先验 |

## Deferred

| 项 | 状态 | 入口 |
|---|---|---|
| 地图先验门控 | DEFERRED → R3+ | `additional_components_plan.md` §五 |
| 自适应推理 / ROI | DEFERRED → R3+ | `additional_components_plan.md` §九 |
| INT8 QAT | DEFERRED → R3+ | `additional_components_plan.md` §十 |
| 规划器先验融合 | DEFERRED → R3+ | `additional_components_plan.md` §十一 |
| 跨检测共现推理 | R3 可选 | `cross_detection_reasoning_plan.md` |

## 行动项

### 候选模型

| 模型 | 参数量 | 免NMS | 许可证 | 定位 |
|---|---:|---|---|---|
| YOLO26-n | ~2.5M | 是 | AGPL-3.0 | P1 基线；R1 证实容量不足，不部署 |
| YOLO26-s | ~9M | 是 | AGPL-3.0 | R1 部署基线 |
| YOLO26-m | ~20M | 是 | AGPL-3.0 | R1 上限基准 |
| YOLOv13-s | ~9M | 是 | AGPL-3.0 | R1 备选轨道 |
| DEIM-D-FINE-S | ~10M | 是 | Apache-2.0 | R1 / R2 Apache-2.0 备选 |
| DEIM-D-FINE-M | ~19M | 是 | Apache-2.0 | R1 / R2 精度上限 |

### 数据与类别

| 数据集 / 来源 | 状态 | R2 用途 |
|---|---|---|
| S2TLD / BSTLD / LISA | R1 后整体退役 | 不再作为 R2/R3 训练、评估、设计证据 |
| R2 自采 | 唯一 R2 基础 | 训练、验证、replay、栏杆与新增灯型 |
| raw video | 强制保留 ≥6 个月 | SSL、主动学习、时序 / 共现 / 负面控制 |

R1 7 类：

| ID | 类别 | ID | 类别 |
|---:|---|---:|---|
| 0 | `red` | 4 | `greenLeft` |
| 1 | `yellow` | 5 | `redRight` |
| 2 | `green` | 6 | `greenRight` |
| 3 | `redLeft` |  |  |

R2 目标：

| 组 | 下限 | 上限 / 条件 |
|---|---|---|
| 交通灯 | R1 7 类 + `forwardGreen` + `forwardRed` = 9 | 再加 ≤3 类；PM / 自采数据触发 |
| 栏杆 | `barrier` = 1 | `armOn` / `armOff`；每态 ≥500 实例且多样性达标 |
| 总 `nc` | 10 | 14 |

### R2 热路径

| 项 | 任务 | 状态 |
|---|---|---|
| Copy-paste + 类平衡 | R2 freeze 后 b/c/d；写 `runs/_copy_paste_decision.json` | a-stage LANDED |
| 硬负样本挖掘 | demo8/11/13 + R2 难场景；写 `runs/_hard_negative_decision.json` | a-stage LANDED |
| KD P0 cells | A1 + A3 + A2a/A2b；DEIM 路径另跑 A0 | scaffold v1.3 LANDED |
| R2 精度奇偶 | FP16/FP32 sidecar + eval-parity + decision JSON | plumbing scaffold LANDED |

### R2 采集 / 标注 / 训练

- [ ] PM 锁定最终交通灯上限类清单。
- [ ] PM / 标注负责人锁定一页标注 SOP。
- [ ] R2 自采 raw video + 抽帧数据冻结。
- [ ] 冻结 `runs/_r2_val_manifest.txt`、`runs/_r2_audit_coverage.json`。
- [ ] 冻结 `runs/_hard_negative_eval_manifest.json`。
- [ ] 应用 §R2 训练 imgsz 决策规则 → 写 `runs/_r2_train_config.json`（`imgsz` / `multi_scale` / `bbox_width_p50` / `frac_lt_0.03`）。
- [ ] 训练 10-14 类联合检测模型；单一模型输出交通灯 + 栏杆。
- [ ] 导出 Orin TensorRT FP16 / 选定 precision engine。
- [ ] 完整 demo + 5-min Orin soak。
- [ ] 写 `docs/reports/phase_2_round_*.md`，coverage-gaps 列出 deferred / blocked 项。

## 决策规则

### R1 三轨部署

1. YOLO26s/m 在部署域评估集 mAP50 ≥ 0.60 → 备选轨道降为监控，进入 R2。
2. YOLOv13-s 相对 YOLO26s +≥ 3 pp mAP50 → 主力切换为 YOLOv13-s。
3. DEIM-D-FINE 相对 YOLO26 最佳 +≥ 5 pp mAP50 且 Orin FP16 ≤ 50 ms/帧 → 主力切换为 DEIM。
4. 三者差距 < 2 pp → 按许可证成本排序：DEIM > YOLOv13 > YOLO26。

### R2 训练 imgsz 决策规则（2026-05-11 锁定，替代之前的"R2 锁 imgsz=1280"无条件项）

R1 1280-训练实验显示同域 BSTLD/S2TLD/LISA val 上 s/m 在 1280 训练反降 −4.4 / −11.3 pp、仅 n 受益 +6.5 pp；该结论受限于 R1 数据分布（>80% 标签宽度 < 3%），**不能外推到 R2 部署域数据**（竖屏 / 手机 / 国内路况 + 新增 barrier 类预计 bbox 中位数显著更大）。R2 训练分辨率由数据分布决定，在 R2 manifest freeze 后执行：

```
输入：runs/_r2_val_manifest.txt + runs/_r2_train_manifest.txt
计算：
  bbox_width_p50  = R2 train+val 归一化 bbox 宽度中位数
  frac_lt_0.03    = 宽度 < 3% 的标签占比

规则：
  if frac_lt_0.03 >= 0.50:
      imgsz = 1280              # 小目标主导，与 R1 同档但数据已换 → 1280 训练
  elif frac_lt_0.03 <= 0.25 and bbox_width_p50 > 0.04:
      imgsz = 640               # 部署域 bbox 更大，省 3-4× GPU 时长
  else:
      imgsz = 960, multi_scale = True   # Ultralytics 随机 0.5-1.5× 缩放
```

输出：`runs/_r2_train_config.json`，字段 `{imgsz, multi_scale, bbox_width_p50, frac_lt_0.03, rule_branch}`。所有 R2 训练 wrapper 必须 read 该文件，不接受 hardcode 的 imgsz。

**DEIM 注意**：R1 仅有 YOLO26 的 1280-训练数据；DEIM-S/M 在 1280 训练的行为未知。R2 启动时若决策规则选 1280，DEIM 训练前 1 epoch wall-clock 必须 sanity-check（若发散即降回 960 + multi_scale）。

### R2 部署 gate

| Gate | 要求 |
|---|---|
| 类别 | `nc` 在 10-14；类别映射与 frozen manifest 一致 |
| 数据 | R2 自采为唯一训练 / 评估基础 |
| 训练 imgsz | 由 §R2 训练 imgsz 决策规则 派生，写 `runs/_r2_train_config.json` |
| 延迟 | Orin 部署 imgsz 端到端 ≤ 50 ms（部署 imgsz 与训练 imgsz 不必相同，但需在 eval-parity gate 锁定）|
| 精度奇偶 | engine 通过 eval-parity gate；sidecar 完整 |
| 安全类 | 安全类 AP / recall 不越过各组件预承诺下限 |
| 报告 | decision JSON + coverage-gaps + carry-forward JSON 完整 |

## 资源

| 环境 | 硬件 | 用途 |
|---|---|---|
| 开发 | MacBook M4 Pro 24GB | 本地开发、快速验证、CoreML 导出 |
| 训练 | GPU 服务器 / 4090 D | 完整训练、消融、KD |
| 部署 | NVIDIA Tegra Orin 64GB | TensorRT / ROS2 / 实车验证 |

## 衔接

- `timeline.md`：周级排期与 2026-05-15 截止。
- `pre_r2_kickoff_checklist.md`：R2 启动 / close gate、schema、carry-forward 枚举。
- `additional_components_plan.md`：训练 / 推理 / 集成组件；五步生命周期定义。
- `temporal_optimization_plan.md`：TSM / HMM / GRU；仅 replay 暴露失败模式后启动。
- `cross_detection_reasoning_plan.md`：R3 同帧共现 / planner-prior 框架。
- `scripts/_r2_carry_forward_schema.json`：carry-forward 13-token closed enum。
- `scripts/_r2_component_decision_schema.json`：组件 deploy / defer / drop 结构化记录。
