# 交通信号灯识别 — 开发计划v2.0

## 项目概述

为自动驾驶系统开发交通信号灯检测与状态分类模块，部署于 **NVIDIA Tegra Orin (64GB)** 平台。该平台同时运行完整的自动驾驶软件栈（感知、规划、控制、传感器融合等），因此信号灯模块必须轻量化，尽量减少 GPU/内存占用。

---

## 一、候选模型

| 模型 | 参数量 | COCO mAP50-95 | 免NMS | 许可证 | 定位 |
|------|--------|--------------|-------|--------|------|
| **YOLO26-n** | ~2.5M | 40.9 | 是 | AGPL-3.0 | P1 基线；R1 证实容量不足，**不部署** |
| **YOLO26-s** | ~9M | 47.5 | 是 | AGPL-3.0 | **R1 部署主力**（Orin 1280 引擎已验证） |
| YOLO26-m | ~20M | 51.5 | 是 | AGPL-3.0 | R1 上限基准（相对 s 仅 +2 pp，不部署） |
| YOLO11-n/s/m | ~2.6–20M | 39.5–50.4 | 否 | AGPL-3.0 | P1 对比基线，R1 起退役 |
| RT-DETR-L | ~32M | ~53.0 | 是 | AGPL-3.0 | P1 精度天花板参考，不部署（参数量超 Orin 预算） |
| **YOLOv13-s** | ~9M | ~52 | 是 | AGPL-3.0 | **R1 备选轨道** — YOLO26 同家族低风险对照 |
| **DEIM-D-FINE-S** | ~10M | 49.0 | 是 | Apache-2.0 | **R1 备选轨道** — 商用许可证备选 + FDR 小目标优势 |
| **DEIM-D-FINE-M** | ~19M | 52.7 | 是 | Apache-2.0 | R1 精度上限（Apache-2.0 线） |

### 选型说明

- **YOLO26**（主力）：最新版本（2026年1月），免NMS端到端推理，延迟更稳定；STAL（小目标感知标签分配）有助于远距离信号灯检测。R1 完整训练 n/s/m 三档。
- **YOLO11**（退役）：P1 对比基线，R1 起不再训练。
- **RT-DETR-L**（参考）：P1 精度天花板对照。R1 阶段以 DEIM-D-FINE-M 取代其角色（同为 Transformer 系、精度相近、参数量低一半、许可证友好）。
- **YOLOv13-s**（R1 备选）：HyperACE / DSC3k2 为 YOLO26 的增量改动，作为"YOLO26 出 bug 时的快速替换"。与 YOLO26 同属 AGPL-3.0，替换不改变合规策略。集成风险最低。
- **DEIM-D-FINE-S / M**（R1 备选）：Apache-2.0，从根本上规避 Ultralytics 企业许可费用；FDR（Fine-grained Distribution Refinement）回归头对交通灯这种 bbox 宽度 < 3% 的目标有 paper 级优势。部署前提是在 Orin JetPack 5.1 / TRT 8.5 上从 OSS 构建 `MultiscaleDeformableAttnPlugin_TRT`（一次性投入，~1 天；见 §五.4）。DEIM-N 按性价比评估不训练。

选型依据详见 [`../proposals/yolo26_alternatives_survey.md`](../proposals/yolo26_alternatives_survey.md)。三条 R1 轨道共用同一合并数据集与 80/20 切分，指标可直接横向对比。

---

## 二、数据集

| 数据集 | 图片数 | 标注数 | 分辨率 | 标注格式 | 许可证 |
|--------|--------|--------|--------|----------|--------|
| **S2TLD** | 1,222 | 2,436 | 1920×1080 | Pascal VOC XML | MIT |
| **BSTLD** | 13,427 | ~24,000 | 1280×720 | YAML | 非商用 |
| **LISA** | 43,075 | ~226,000 | 多种 | CSV | CC BY-NC-SA |
| **合计** | ~57,724 | ~252,000 | — | — | — |

### 原始类别

- **S2TLD（5类）**：`red`, `green`, `yellow`, `wait_on`, `off`
- **BSTLD 训练集（13类）**：Red, RedLeft, RedRight, RedStraight, RedStraightLeft, Green, GreenLeft, GreenRight, GreenStraight, GreenStraightLeft, GreenStraightRight, Yellow, Off
- **BSTLD 测试集（4类）**：Red, Green, Yellow, Off（无方向标注，第二阶段需重标注）
- **LISA（7类）**：`go`(99K), `stop`(90K), `stopLeft`(26K), `warning`(5.5K), `goLeft`(5K), `warningLeft`(701), `goForward`(410)

---

## 三、开发阶段

### 第一阶段：基础颜色检测（已完成 — P1 结束于 2026-04-15；详见 [`../reports/phase_1_report.md`](../reports/phase_1_report.md)）

**目标**：检测信号灯并分类为红/黄/绿三类

| 类别ID | 类别 | 说明 |
|--------|------|------|
| 0 | `red` | 红灯（含所有方向变体） |
| 1 | `yellow` | 黄灯（含所有方向变体） |
| 2 | `green` | 绿灯（含所有方向变体） |

**类别映射规则**：
- 所有方向变体合并为基础颜色（如 `stopLeft` → red，`goLeft` → green）
- 过滤 `off`、`wait_on` 等非可操作状态
- 三个数据集全部参与训练

**工作内容**：
1. 编写数据转换脚本（S2TLD/BSTLD/LISA → 统一 YOLO 格式）
2. 合并数据集，80/20 分层划分训练/验证集
3. 训练 5-7 个模型变体（YOLO26 n/s/m, YOLO11 n/s/m, RT-DETR-L）
4. 对比评估：mAP、逐类精度、推理延迟、GPU 占用、模型大小
5. 选定最优模型架构与尺寸

**训练策略**：

- 使用 COCO 预训练权重进行迁移学习（非从零训练）
- ~100 epochs，基于验证集 mAP 早停
- 禁用垂直翻转（`flipud=0.0`），保留 HSV 增强
- 合并数据集统一训练（非逐数据集顺序训练，避免灾难性遗忘）

### 第二阶段 R1：方向检测（7 类，进行中）

**目标**：在 3 类颜色基础上加入方向语义，打通 Orin 端到端部署链路，并通过三条架构轨道降低选型风险。

| 类别 ID | 类别 |
|--------|------|
| 0 | `red`（圆形） |
| 1 | `yellow`（圆形） |
| 2 | `green`（圆形） |
| 3 | `redLeft` |
| 4 | `greenLeft` |
| 5 | `redRight` |
| 6 | `greenRight` |

**三条并行训练轨道**（均为 7 类、同数据集、同切分）：

| 轨道 | 模型 | 角色 | 状态 |
|------|------|------|------|
| 主力 | YOLO26-n / s / m | 部署路径已打通 | ✅ n/s/m 均已训练；s 为部署基线 |
| 备选 A | YOLOv13-s | YOLO26 同家族低风险对照 | 🚧 训练中 |
| 备选 B | DEIM-D-FINE-S / M | Apache-2.0 商用备选 + 精度上限 | 🚧 训练中 |

**R1 训练策略**：
- `imgsz=640`（与 P1 一致；R2 改 1280）
- `patience=20`（主力轨道）
- `fliplr=0.0`（箭头方向语义反转）
- COCO 预训练权重迁移
- Mosaic 前 90% 轮次启用、最后 10% 关闭

**R1 部署决策规则**（三轨训练完成后按序应用）：
1. 主力 YOLO26s/m 在部署域评估集上 mAP50 ≥ 0.60 → 备选轨道降为监控，直接进入 R2。
2. YOLOv13-s 相对 YOLO26s **+≥ 3 pp mAP50** → 切换主力为 YOLOv13-s（合规策略不变）。
3. DEIM-D-FINE 相对 YOLO26 最佳 **+≥ 5 pp mAP50** 且 Orin FP16 ≤ 50 ms/帧 → 切换为 DEIM（前提是 `MultiscaleDeformableAttnPlugin_TRT` 在 Orin 上 OSS 构建成功）。
4. 三者持平（< 2 pp）→ 按许可证成本排序：**DEIM > YOLOv13 > YOLO26**。

**R1 已完成交付物**：YOLO26{n,s,m}-r1 训练权重；1280 / 1536 TRT 引擎（Orin FP16 25 / 28 ms/帧）；C++ 推理管线；`xyxy` 后处理修复。R1 训练结果与决策详情见 [`../reports/phase_2_round_1_report.md`](../reports/phase_2_round_1_report.md)。

**R1 剩余项**：YOLOv13-s / DEIM-D-FINE-S / DEIM-D-FINE-M 训练完成 → 填入报告 §R1 备选架构 → 应用决策规则 → 锁定 R1 部署基线。

### 第二阶段 R2：方向 + 栏杆联合检测（10–14 类，准备中）

**范围（2026-04-21 PM 例会锁定）**：R1（7 类）基础上扩展为**联合检测模型**，同时输出交通灯与道路栏杆；**下限 10 类锁定**，上限 14 类取决于现场采集结果与 PM 增补清单。

| 组 | 下限（已锁定） | 上限（条件触发） |
|----|---------------|------------------|
| 交通灯 | 9 = R1 7 类 + `forwardGreen` + `forwardRed` | 12 = 再加 ≤3（待 PM 按部署场景观测给出；候选：行人信号灯 / 黄色方向变体 / 闪烁状态）|
| 栏杆 | 1 = `barrier`（仅检测存在，不分状态）| 2 = `armOn` / `armOff`（当采集数据每态 ≥500 实例且多样性达标）|
| **总 nc** | **10**（已锁定）| **14**（条件触发）|

**单一联合模型**（非多模型拼接）。理由：共享 backbone 正则化；Orin 1280 分辨率现 ~25 ms / 帧，拆两模型将逼近 50 ms 预算上限；运维成本减半。

**剩余待 PM 决定项（阻塞数据采集）**：
1. 上限侧的 3 个可选交通灯类别（若不新增则维持 9 类下限）
2. 栏杆是否细分 `armOn` / `armOff`（条件驱动，非 PM 独裁）
3. 一页标注 SOP — 视觉定义与边界样本处理（SOP 缺失是 R1 重标注返工第一原因）

R2 范围细节（类别表、数据可行性评估、采集策略、代码改动点）以 [`../reports/phase_2_round_1_report.md`](../reports/phase_2_round_1_report.md) §"R2 范围扩展（PM 确认事项）" 为准。

---

## 四、环境与基础设施

| 环境 | 硬件 | 用途 |
|------|------|------|
| 开发 | MacBook M4 Pro 24GB | 本地开发、快速验证、CoreML 导出测试 |
| 训练 | GPU 服务器（远程） | 完整训练、超参搜索 |
| 部署 | NVIDIA Tegra Orin 64GB | 最终部署目标 — 与完整自动驾驶栈共享资源 |

**导出格式**：TensorRT FP16（Orin 部署）、CoreML（M4 Pro 开发）、ONNX（通用）

---

## 五、已知问题与风险

### 1. 数据标注问题

| 问题 | 详情 | 影响阶段 |
|------|------|----------|
| **方向标注不一致** | S2TLD 方向标注仅限原始子集；LISA 有左转；BSTLD 训练集有方向但稀缺，测试集（8,334张）无方向标注需重标注 | 第二阶段 |
| **右转箭头数据极度稀缺** | 仅 BSTLD 有右转标注（RedRight 5 条、GreenRight 13 条），需重标注或采集新数据 | 第二阶段 |
| **直行箭头数据待恢复** | R1 曾将直行箭头折叠为圆灯；R2 新增 `forwardRed`/`forwardGreen` 后需从 BSTLD/LISA 已有样本中恢复，并核查是否存在误标 | 第二阶段 |
| **栏杆无公开数据集** | `barrier` / `armOn`/`armOff` 需现场实采；两态模式要求每态 ≥500 实例且多样性足够 | 第二阶段 |
| **黄灯样本偏少** | S2TLD 黄灯仅 75 个标注（红灯 1235，绿灯 816）；LISA `warning` 仅 5.5K（`go` 99K） | 两个阶段 |
| **扩展类 SOP 未定** | PM 最终类别清单 + 标注 SOP 未发布；SOP 缺失是 R1 重标注的主因 | 第二阶段（阻塞） |

### 2. 商用许可问题

| 组件 | 许可证 | 研究可用 | 商用可用 |
|------|--------|----------|----------|
| Ultralytics YOLO（所有版本，含 YOLO11 / YOLO26 / RT-DETR） | AGPL-3.0 | 是 | 需购买企业许可 |
| YOLOv13 | AGPL-3.0 | 是 | 需购买企业许可（同 Ultralytics 线） |
| **DEIM / D-FINE** | **Apache-2.0** | 是 | **是**（R1 备选轨道即为此路径） |
| RT-DETR（百度原版） | Apache-2.0 | 是 | 是 |
| S2TLD 数据集 | MIT | 是 | 是 |
| BSTLD 数据集 | 非商用 | 是 | **否** |
| LISA 数据集 | CC BY-NC-SA | 是 | **否** |

**商用化路径**：

- 研究阶段可使用全部数据集和框架
- 生产部署需要：
  - **架构层**：若 R1 决策规则选定 DEIM-D-FINE，架构许可问题即消解；否则购买 Ultralytics 企业许可
  - **数据层**：仅使用 S2TLD（MIT）数据，或采集/购买商用授权数据集
  - 采集实际驾驶场景数据并标注（同时解决域差异问题）

### 3. 所需资源

| 资源 | 用途 | 优先级 |
|------|------|--------|
| **GPU 训练服务器** | 完整训练 5-7 个模型变体（M4 Pro 仅适合快速验证） | 高 — 第一阶段 |
| **Tegra Orin 开发板** | 部署测试、TensorRT 推理延迟实测、GPU 占用评估 | 高 — 第一阶段 |
| **Ultralytics 企业许可** | 商用部署（如选用 YOLO/RT-DETR Ultralytics 版） | 生产阶段 |
| **标注工具 + 标注人力** | 第二阶段方向标注补充、生产阶段自有数据标注 | 第二阶段 |
| **实际驾驶数据采集** | 商用数据集构建、覆盖中国路况信号灯样式 | 生产阶段 |

### 4. 技术风险

| 风险 | 说明 | 缓解措施 |
|------|------|----------|
| 黄灯检测精度不足 | 黄灯样本远少于红/绿灯，可能影响召回率 | 数据增强（HSV 扰动）、过采样、关注 per-class 指标 |
| 远距离小目标漏检 | 信号灯在远处可能仅占几个像素 | YOLO26 STAL 机制、copy-paste 增强、评估 AP-small |
| Orin 资源超标 | 模型 GPU 占用超出预算（<5% 目标） | 优先选用 nano 变体、TensorRT FP16 优化 |
| 数据集域差异 | 三个数据集来自不同国家/场景，可能与中国路况差异较大 | 评估模型在实际场景的表现，必要时采集本地数据微调 |
| **DEIM Orin 部署门槛** | DEIM 依赖 `MultiscaleDeformableAttnPlugin_TRT`，JetPack 5.1 / TRT 8.5 **stock 不含**此插件；需从 TensorRT-OSS 编译 | R1 决策规则 3 生效前先做可行性验证（~1 天）；失败则 fallback 到 YOLOv13 / YOLO26 |
| **YOLOv13 / DEIM 推理管线改造** | 现有 C++ 管线为 YOLO26 裁头输出定制（letterbox xyxy，无 NMS） | YOLOv13 需 NMS 分支 + 独立 strip 脚本；DEIM 需 DETR-style 后处理分支。仅为 R1 决策选中的那一个模型实施 |

---

## 六、里程碑

周级排期见 [`./timeline.md`](./timeline.md)。

| 阶段 | 里程碑 | 状态 / 交付物 |
|------|--------|--------|
| 第一阶段 | 3 类基线训练 + 选型 | ✅ 完成 — YOLO26s 选定（报告：[`../reports/phase_1_report.md`](../reports/phase_1_report.md)）|
| R1-主力 | YOLO26 n/s/m 7 类训练 + Orin 端到端链路 | ✅ 完成 — Orin 1280 引擎 25 ms/帧；xyxy 后处理修复；YOLO26s 为部署基线 |
| R1-备选 | YOLOv13-s + DEIM-D-FINE-S/M 训练 | 🚧 三模型并行训练中（2026-04-22 起） |
| R1-跟踪 | ByteTrack + EMA 投票（Plan A） | ✅ Python + C++ 两端落地（`inference/tracker/`、`inference/cpp/src/tracker.cpp`）；fixture 单测通过，Orin 集成待触发 |
| R1-决策 | 应用决策规则 → 锁定 R1 部署基线 | ⏳ 待三轨训练完成 |
| R2-规划 | 联合检测方案（10–14 类）锁定 | ✅ 范围锁定（下限 10）；待 PM 给出上限类清单 + 标注 SOP |
| R2-重标注 | 方向 & 直行箭头补标 | ⏳ S2TLD 三子集已完成方向标注（见 `../data/class_distribution.md`）；BSTLD 测试集方向补标 + BSTLD/LISA `forwardGreen`/`forwardRed` 从圆灯恢复待做 |
| R2-数据采集 | 栏杆 + 部署现场数据 | ⏳ 待 SOP — 一次实采同时获得：部署评估片段 + 栏杆训练集（MVP ≥2K，条件达成则双态各 ≥500）+ 新增灯型样本 |
| R2-训练 | 10–14 类联合模型 | ⏳ 待数据 |
| R2-部署 | 联合模型 Orin 端验证 | ⏳ 待训练 — 目标 1280 分辨率 ≤ 50 ms |
| 5/15-实车 | 实车测试通过 | ⏳ 截止日期 2026-05-15 |
| 生产化 | 商用许可 + 数据 + 部署 | 未启动 — 需 Ultralytics 企业许可或 Apache-2.0 替代；商用授权数据集 |
