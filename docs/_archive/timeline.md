# 红绿灯识别模型开发时间线 v2.0

## 状态

| 项 | 当前值 |
|---|---|
| R1 | 已完成；7 类红绿灯 + 自定义 TensorRT |
| R2 | 进行中；10-14 类联合检测 |
| 部署平台 | NVIDIA Tegra Orin 64GB |
| 输入 | 8MP 摄像头；推理分辨率 1280 |
| 输出 | `vision_msgs/Detection2DArray` |
| 截止 | 2026-05-15 |
| 生命周期 | 见 `additional_components_plan.md` §一 |

| 轨道 | 状态 |
|---|---|
| R2 主检测器 | data freeze → train → export → Orin → replay |
| R2 训练增强 | Copy-paste / hard-negative / KD P0 |
| R2 精度奇偶 | FP16 / FP32 sidecar + eval-parity |
| R2 时序优化 | gated by replay failure modes |
| R3+ deferred | map-prior、adaptive ROI、INT8、planner-prior |

## Deferred

| 项 | 状态 | 入口 |
|---|---|---|
| 地图先验门控 | DEFERRED → R3+ | `additional_components_plan.md` §五 |
| 自适应推理 / ROI | DEFERRED → R3+ | `additional_components_plan.md` §九 |
| INT8 QAT | DEFERRED → R3+ | `additional_components_plan.md` §十 |
| 规划器先验融合 | DEFERRED → R3+ | `additional_components_plan.md` §十一 |
| 跨检测共现推理 | R3 可选 | `cross_detection_reasoning_plan.md` |

## 行动项

### 已完成：阶段 A / R1

| 日期 | 项 | 输出 |
|---|---|---|
| 4/16-4/19 | 7 类数据流水线 | `scripts/convert_*`、`traffic_light.yaml` |
| 4/17-4/19 | 自定义 TRT 推理流水线 | `inference/trt_pipeline.py`、`inference/demo.py`、`main.py infer` |
| 4/20-4/22 | YOLO26 n/s/m R1 训练 | R1 weights |
| 4/21-4/22 | Orin 部署测试 | 1280 / 1536 FP16 engines；约 25 / 28 ms |
| R1 | ROS2 输出接口 | `Detection2DArray` |
| R1 | Tracker + EMA | Python + C++ Plan A |

### R2：自采数据 + 联合检测

#### 4/23-5/4 数据采集 / 标注

- [ ] PM 最终类别清单。
- [ ] 标注 SOP。
- [ ] R2 raw video 连续 30 fps。
- [ ] 部署评估片段。
- [ ] 栏杆训练集：MVP `barrier` ≥2K 实例。
- [ ] 条件双态：`armOn` / `armOff` 每态 ≥500 实例且多样性达标。
- [ ] 新增灯型：`forwardGreen` / `forwardRed` + PM 上限类。
- [ ] raw video 保留 ≥6 个月。
- [ ] 标注完成并冻结 train / val / audit manifests。

#### 5/5-5/6 数据预处理

- [ ] R2 自采数据作为唯一训练 / 评估基础。
- [ ] R1 数据集退役；不混合 LISA / BSTLD / S2TLD。
- [ ] 类别平衡 / rare-class oversampling。
- [ ] Copy-paste / HSV / Mosaic / MixUp 配置。
- [ ] hard-negative `bg/` / empty-image 接入。
- [ ] 冻结 `runs/_r2_val_manifest.txt`。
- [ ] 冻结 `runs/_r2_audit_coverage.json`。
- [ ] 冻结 `runs/_hard_negative_eval_manifest.json`。

#### 5/6-5/7 第二轮训练

- [ ] 主检测器训练：YOLO26 / YOLOv13 / DEIM 选型胜者。
- [ ] Copy-paste 三臂消融。
- [ ] Hard-negative A/B。
- [ ] KD P0 cells：A1 + A3 + A2a/A2b；DEIM 路径另跑 A0。
- [ ] 写组件 decision JSON。

#### 5/8-5/11 部署测试

- [ ] TensorRT FP16 / selected precision export。
- [ ] sidecar 完整。
- [ ] eval-parity gate。
- [ ] build-determinism check。
- [ ] Orin benchmark：latency / GPU mem / first-frame / median `t_detect_ms`。
- [ ] 规划模块联调：`Detection2DArray` + depth。
- [ ] 5-min Orin soak；`engine_sha256 == selected_artifact_sha256`。

#### 5/12-5/15 实车测试

- [ ] 车载集成。
- [ ] 固定路线测试。
- [ ] 夜间 / 雨天 / 眩光 / 遮挡场景。
- [ ] 红→绿误判零容忍检查。
- [ ] 红↔黄混淆率记录。
- [ ] replay failure mode tags 写入 TSM / HMM trigger 输入。
- [ ] phase report coverage-gaps 按 6 字段格式列出 carry-forward。

## 决策规则

### R2 close gate

| Gate | 要求 |
|---|---|
| 数据 | R2 train / val / audit manifests frozen + sha256 |
| 组件 | Copy-paste、hard-negative、KD P0 完成对应 a-c；deploy 候选完成 d |
| 精度奇偶 | `runs/_r2_precision_decisions.json` 完整 |
| Engine | selected engine sidecar 完整；eval-parity pass |
| Soak | 5-min Orin soak；SHA hard-bound |
| 报告 | phase report + coverage-gaps + carry-forward JSON |

### 2026-05-15 优先级

| 情况 | 处理 |
|---|---|
| 主线未关帧 | 优先主线；时序 / 共现不启动 |
| 主线关帧且 replay 无明确失败 | 时序 / 共现搁置 |
| replay 出 small/far/occluded miss | `temporal_optimization_plan.md` §1 TSM |
| replay 出 flicker / illegal transition | `temporal_optimization_plan.md` §2 HMM |
| SAHI / TSM latency 落入 INT8 band | 只登记 R3+ carry-forward；R2 不做 INT8 |

## 总览

```text
4/16  4/20  4/23       5/5    5/8       5/12  5/15
 |      |      |          |      |          |     |
R1流水线 ██████
R1训练       ██
R1部署        ███
R2采集  █████████████████
R2标注       ████████████
R2预处理                ██
R2训练                   ██
R2部署                     ████
实车测试                         ████
```

## 衔接

- `development_plan.md`：模型、类别、R2 部署 gate。
- `additional_components_plan.md`：Copy-paste、hard-negative、KD、SAHI、R3+ deferred。
- `pre_r2_kickoff_checklist.md`：schema、R2 close gate、coverage-gaps 行格式。
- `temporal_optimization_plan.md`：5/15 后 replay-driven TSM / HMM。
- `cross_detection_reasoning_plan.md`：R3 同帧共现。
- `scripts/_r2_carry_forward_schema.json`：carry-forward 13-token enum。
