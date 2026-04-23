# 第二阶段 · 第一轮训练报告（R1）— 方向性检测（7类）

> **文档状态**：**进行中（Living Document）** — 本报告在第二轮（R2）训练启动前会持续更新。  
> **更新策略**：新增结果追加在既有章节下方；重大修订在"变更记录"中登记，不覆盖历史结论。

---

## 概述

第二阶段第一轮训练在第一阶段（3类颜色）基础上扩展到 **7 类方向性检测**（red / yellow / green / redLeft / greenLeft / redRight / greenRight）。R1 的目标：

1. 在现有合并数据集（S2TLD + BSTLD + LISA 的方向重标注子集）上验证 7 类训练可行性，取得基准指标。
2. 打通从训练到 Orin 端 TensorRT C++ 推理的端到端链路。
3. 识别性能瓶颈并为 R2 制定改进方案。

**训练硬件**：RTX 4090 D（远程 GPU 服务器）  
**部署硬件**：NVIDIA Jetson AGX Orin 64GB（JetPack 5.1.2 / TRT 8.5.2）  
**框架**：Ultralytics（AGPL-3.0）

---

## 训练配置

沿用第一阶段超参数，主要变化在数据（7 类）和早停：

| 参数 | 值 | 备注 |
|------|-----|------|
| 轮次 | 100 | 实际触发早停 |
| `patience` | 20 | 早停耐心窗口 |
| 图像尺寸 | **640×640** | 与 P1 一致 |
| 批次大小 | 自动 | |
| 优化器 | Auto（SGD） | |
| `fliplr` | **0.0** | 禁用水平翻转（箭头方向语义反转，保留图像但标签未相应翻转） |
| `flipud` | 0.0 | 同第一阶段 |
| Mosaic | 1.0 | 第10轮后关闭 |
| HSV | H=0.015, S=0.7, V=0.4 | |
| 预训练权重 | COCO（Ultralytics 官方） | |

**数据集路径**：`data/merged/` — 训练 / 验证划分沿用第一阶段策略，按主要类别分层抽样 80/20。

---

## 数据分布（训练集，来自 `runs/detect/yolo26n-r1/labels.jpg`）

| 类别 | 实例数 | 占比 | 评估 |
|------|-------|------|------|
| green | 51,661 | 44.4% | 充足 |
| red | 46,483 | 40.0% | 充足 |
| redLeft | 12,983 | 11.2% | 充足 |
| yellow | 3,141 | 2.7% | 偏少（延续 P1 问题） |
| greenLeft | 2,604 | 2.2% | 偏少 |
| **redRight** | **19** | **0.02%** | **几乎为零** |
| **greenRight** | **13** | **0.01%** | **几乎为零** |

**边界框形态特征**：
- 宽度分布峰值在 0.01–0.03（即 `imgsz=640` 下 6–20 像素宽）— 典型小目标。
- 位置分布集中在 x≈0.55–0.7、y≈0.3–0.4（左舵车驾驶视角的上方右侧）。

---

## R1 训练结果

### 指标汇总

| 模型 | 训练轮次 | 最佳 mAP50 | 最佳 mAP50-95 | Precision | Recall | 状态 |
|------|---------|-----------|--------------|-----------|--------|------|
| YOLO26n-r1 | 39 / 100 | 0.720 | 0.484 | 0.940 | 0.689 | 早停收敛 |
| YOLO26s-r1 | 32 / 100 | 0.849 | 0.608 | 0.930 | 0.670 | 早停收敛 |
| YOLO26m-r1 | 21 / 72 | 0.869 | 0.635 | 0.934 | 0.712 | 训练完成 |

三个模型均在早停窗口内触发 `patience=20` 停止，mAP50-95 曲线自最佳轮次起 20 轮无改善。YOLO26m-r1 相对 s-r1 提升有限（mAP50 +2.0 pp / mAP50-95 +2.7 pp），但推理成本显著更高 — 在 Orin 1280 延迟预算内优先选 s。

### 与第一阶段（3类）对比

| 模型 | P1 mAP50 | R1 mAP50 | P1 mAP50-95 | R1 mAP50-95 | 说明 |
|------|---------|---------|------------|------------|------|
| YOLO26n | 0.866 | 0.720 | 0.601 | 0.484 | **-14.6 / -11.7 pp** |
| YOLO26s | 0.893 | 0.849 | 0.628 | 0.608 | **-4.4 / -2.0 pp** |

**观察**：
- YOLO26s 从 3 类扩到 7 类仅下降 2 pp mAP50-95，可接受。
- YOLO26n 下降显著（>10 pp），在右转 / greenLeft 等小样本类别上严重失衡，印证"容量不足的模型无法补偿类别不均衡"。
- R1 `patience=20` 在多类不均衡任务中可能偏紧（R2 改 `patience=40`）。

### R1 备选架构（并行训练）

为降低单一架构风险 + 探明精度上限，R1 期间并行训练两条备选轨道。选型依据与环境配置见 [`../proposals/yolo26_alternatives_survey.md`](../proposals/yolo26_alternatives_survey.md)。

| 模型 | 许可证 | 角色 | 训练轮次 | 最佳 mAP50 | 最佳 mAP50-95 | Precision | Recall | 状态 |
|------|-------|------|---------|-----------|--------------|-----------|--------|------|
| YOLOv13-s | AGPL-3.0 | YOLO26 同架构家族低风险对照 | — | — | — | — | — | 训练中 |
| DEIM-D-FINE-S | Apache-2.0 | 商用许可证备选 + 小目标精度探底 | — | — | — | — | — | 训练中 |
| DEIM-D-FINE-M | Apache-2.0 | 精度上限基准 | — | — | — | — | — | 训练中 |

**三条轨道的分工**：
- **主力（YOLO26 s/m）**：部署路径最成熟，Orin TRT 管线已打通 — 即便备选轨道胜出，数据处理 / 部署流程可复用。
- **YOLOv13-s**：`HyperACE` / `DSC3k2` 自定义模块相对 YOLO26 为增量改动，作为"YOLO26 出 bug 时的快速替换"。与 YOLO26 同属 AGPL，替换不改变合规策略。
- **DEIM-D-FINE-S/M**：Apache-2.0 从根本上解决 AGPL 商用问题；同时 D-FINE 的 FDR（Fine-grained Distribution Refinement）回归头在小目标定位上有 paper 级优势，对交通灯这种 bbox 宽度 < 3% 的目标理论收益大。N 规格按 `yolo26_alternatives_survey.md §六` 评估不具备数据性价比，R1 不训。

**评估口径统一**：三条轨道共用同一合并数据集 `data/merged/`（YOLO 格式 + COCO JSON 双出）、同一 80/20 分层切分、同一 7 类定义 — 所有指标可直接横向对比。YOLOv13 走独立 venv（`yolov13/.venv`），DEIM 走主 `uv` 的 `--extra deim` 可选组，均不污染主力训练环境。

**增强口径对齐**：三条轨道均禁用水平翻转（箭头方向语义反转），其余光学 / 几何增强保持各框架默认。DEIM 默认数据管线继承 `base/dataloader.yml` 包含 `RandomHorizontalFlip`，在 `deim_hgnetv2_{s,m}_traffic_light.yml` 中显式覆盖 `transforms.ops` 剔除（2026-04-23）；Mosaic / RandomPhotometricDistort / RandomZoomOut / RandomIoUCrop / mixup 保留，由 `stop_epoch` 在尾段关闭以避免稀疏类（`redRight` / `greenRight`）样本被破坏。YOLO26 已在 R1 初版以 `fliplr=0` 锁定，YOLOv13 沿用。

**R1 决策点**（三轨完成后执行）：
1. 若主力 YOLO26s/m 在部署域评估集上已达 R2 验收标准（mAP50 ≥ 0.60），直接进入 R2 数据侧工作，备选轨道降为监控组。
2. 若 YOLOv13-s 相对 YOLO26s **+≥ 3 pp mAP50**，切换主力为 YOLOv13-s（合规策略不变）。
3. 若 DEIM-D-FINE 相对 YOLO26 最佳 **+≥ 5 pp mAP50**，且 Orin 端 FP16 延迟 ≤ 50 ms/帧，启动主力切换 — 需提前验证 Orin JetPack 5.1 / TRT 8.5 上 `MultiscaleDeformableAttnPlugin_TRT` 可从 OSS 构建（风险项，见 survey §三）。
4. 若三者持平（差异 < 2 pp），按许可证成本排序：**DEIM > YOLOv13 > YOLO26**，优先 Apache-2.0。

---

## 部署链路验证

R1 首次完成端到端部署链路：训练 → ONNX 导出 → Head 裁剪（`scripts/strip_yolo26_head.py`）→ Orin 上 `trtexec --fp16` 构建 → C++ 推理。`inference/cpp/build/tl_demo` 在 Orin 上成功运行，1280 / 1536 两种分辨率引擎均已验证。

**部署端 latency（2026-04-21 Orin 实测，yolo26s-r1，FP16）**：
- `imgsz=1280` → ~25 ms/帧（~39 FPS）
- `imgsz=1536` → ~28 ms/帧（~34.5 FPS）

均远低于 50 ms 预算上限。部署和引擎构建的全流程见 [`../integration/trt_pipeline_guide.md`](../integration/trt_pipeline_guide.md)。

---

## Demo 视频诊断（总结）

`demo/demo.mp4`（544×960 竖屏手机视频）上 C++ 推理初版出现"多数帧无检测 + 偶有大框"。通过 Ultralytics `.pt` 原生回放（640 / 1280 / 1536 三分辨率）与 ORT 独立验证，定位到两个并发 bug：

1. **引擎输入固化在 640×640**：导出 ONNX 时未传 `--imgsz`，沿用训练 `imgsz=640`；C++ 管线检测到请求 `imgsz=1280` 与引擎不一致时静默回退到 640 并仅打印 `[TRT] warning`（`trt_pipeline.cpp:296-300`）。→ **修复**：在 1280 / 1536 重新导出 + strip + `trtexec` 构建新引擎。
2. **Postprocess 误把 4 通道当 `cxcywh`**：YOLO26 在 `Concat_3` 前经 `Sub/Add_1 + stride-Mul` 输出的是像素级 `xyxy`。ORT 独立验证 f100：误解码时 bbox 占画面 ~67%（即肉眼可见的"大框"），修复后变为 21×50（0.2% 面积），与 Ultralytics 原生一致。→ **修复**：`inference/cpp/src/trt_pipeline.cpp` 与 `inference/trt_pipeline.py` 均改为 xyxy 解码；引擎无需重建。

两个 bug 修复后，demo 在 1280 / 1536 引擎上输出与 `.pt` 原生推理一致；此前"检测稀疏"实为视频内容（前 150 帧含灯，之后长时间无灯）+ 小目标泛化共同作用，并非模型故障。

### 同域 val 分辨率扫描（YOLO26s-r1）

```
imgsz=640:  mAP50=0.8497  mAP50-95=0.6073
imgsz=1280: mAP50=0.7023  mAP50-95=0.4359
```

同域（横屏 dashcam）val 集上，更高推理分辨率反而降低 mAP50 约 14.7 pp — 模型在 640 训练，其特征图 stride 和 anchor 先验锁定 640；在 1280 推理时小目标先验未相应缩放，FP 增加、TP 减少。

**关键推论**：部署端提高 `imgsz` 对**同域数据降低质量**，但对**竖屏 / 非欧美 / 手机 OOD demo 有收益** — 两者指向相反方向。R1 的 1280 / 1536 引擎是 OOD hotfix；长期方案是 R2 以 `imgsz=1280` 直接训练。

### 模型容量观察

- YOLO26n-r1：仅在 640（训练分辨率）上能检测 demo 目标（conf 0.52），在 1280 / 1536 上 0 detections — **不支持跨尺度推理**。
- YOLO26s-r1：在 640 / 1280 / 1536 上均能检测，1536 上 conf 最高（0.58）— 支持跨尺度泛化。

→ R1 部署选用 **YOLO26s**，不部署 YOLO26n。

### 剩余 R1 数据缺陷（不会被部署修补）

- `redRight` / `greenRight` 样本各 10–20 条，模型实际未学到。
- 验证集全部是 BSTLD 横屏 dashcam，**未覆盖部署域**（竖屏 / 手机 / 国内路况）。
- 训练标签 >80% 宽度 < 3%（imgsz=640 下 <20 px）→ 位置先验集中在上右四分之一（dashcam 倾向）。

以上三项由 R2（`imgsz=1280` 训练 + 部署域采集 + 类别补齐）解决。

---

## 第二轮（R2）计划

R2 目标：**关闭部署域与训练域的差距 + 扩展类别（联合模型）**。

### 部署侧（R1 已完成，供 R2 接力）

- [x] 1280 / 1536 引擎在 Orin 上构建与验证
- [x] xyxy postprocess 修复
- [ ] 部署域评估集：从实际部署摄像头采集 100–300 帧，标注（含 R2 最终类别），作为独立 `data/eval_deployment/`。所有后续对比以该集为准，不再只看同域 val。

### 数据侧（R2 前必须完成）

1. **方向重标注**（延续 P1 行动计划第 5 项）：
   - BSTLD 测试集 8,334 张 → 方向重标注（优先级：高）
   - S2TLD `normal_2` 4,168 张 → 方向重标注
   - S2TLD `normal_1` 796 张 → 方向重标注
2. **右转箭头数据补齐**：
   - 方案 A（短期）：利用 `fliplr` 反向增强将 `redLeft`/`greenLeft` 合成为 `redRight`/`greenRight`，图像与类别映射同时翻转。
   - 方案 B（长期）：国内道路场景实采包含右转箭头的真实数据。
3. **评估"去除极稀疏类"**：若 R2 前无法把 `redRight`/`greenRight` 各自补至 >500 条，考虑合并为 `rightArrow` 单一类或暂时移出训练目标。

### 训练侧

4. **`imgsz=1280` 训练**（非 640）：与部署对齐，减轻小目标信息损失。预估训练时间 3–4×。
5. **`patience=40`**：小样本类别收敛更慢。
6. **类别加权 / 过采样**：对 `yellow`、`greenLeft` 过采样 2–3×。
7. **域适应性增强**：启用 `degrees=5.0`、`perspective=0.0005`；自定义增强管线加入 JPEG 压缩 / motion blur。**不开启** `fliplr`（箭头方向会与标签不一致）。
8. **主力模型**：YOLO26s（R1 下降幅度最小，性价比最优），YOLO26m 作为上限基准并行训练。YOLO26n 在 R2 暂缓。

### 验收标准（R2 退出条件）

- [ ] 部署域评估集上 mAP50 ≥ 0.60（含所有 R2 类别整体）
- [ ] 黄灯召回率 ≥ 0.70（不依赖过采样后的高 precision 陷阱）
- [ ] 方向类（redLeft / greenLeft）mAP50 ≥ 0.55
- [ ] Demo 视频上主观检测率（人工抽查）≥ 80%，无明显过大框误报
- [ ] Orin 端 1280 × 1280 引擎 FP16 延迟 ≤ 50 ms / 帧

---

## R2 范围扩展（PM 确认事项）

2026-04-21 PM 会议新增两项诉求。两者均为"加类别"类型的扩展，与现有 R2 计划可在**同一次训练**中完成 — 不会额外引入轮次，但改变数据准备的优先级和规模。

### 扩展一：栏杆 / 道闸检测（分层需求）

| 层级 | 类别 | 触发条件 |
|------|------|---------|
| **MVP（最低要求）** | **`barrier`**（单类：检测栏杆本体，不分状态）| 必达 |
| **最佳实践（可选）** | **`armOn`**（抬起 / 放行）+ **`armOff`**（放下 / 阻挡）| 两种状态**各自**达到质量阈值（≥500 实例且覆盖足够场景多样性） |

- **架构可行性**：`nc` 从 7 提到 8（MVP）或 9（best-practice），仅 head 输出通道差异。可**先按 MVP 标注**，若采集后两态都达标再于标注阶段批量子标签（CVAT / LabelMe 支持）。
- **数据可行性**：S2TLD / BSTLD / LISA **均不含栏杆**，必须从零采集。
  - 首选：**部署现场拍摄**（与部署域评估集合并采集，一次出差覆盖多件事）。
  - 备选：Roboflow Universe（"boom gate" / "railway barrier" / "parking barrier"）— 商业许可须逐条核对。
- **数据预算**：MVP ≥500 实例（目标 2–5K）；双态模式每类 ≥500 且数量比例差不超 3×。
- **类别定义必须先敲定再标注**：SOP 需明确"损坏 / 折断栏杆"、"部分可见"、"地锁 / 升降柱（非横杆式）" 的归类。模糊定义会在 R3 细分时暴雷。

### 扩展二：交通灯类别扩展（已确认 9 类，最多 12 类）

- **已确认新增 2 类**：`forwardGreen`（直行绿灯）、`forwardRed`（直行红灯）— 国内红绿灯配置中"直行箭头"与"满盘灯"分属独立灯位。
- **交通灯类别最低锁定为 9 类**（原 7 + forwardGreen + forwardRed）。
- **最多扩展到 12 类**（再加 3 类）— 具体清单待 PM 按部署场景观测结果出。候选方向：
  - 行人信号：`pedRed` / `pedGreen` / `pedYellow`
  - 方向类补全：`yellowLeft` / `yellowRight` / `forwardYellow`
  - 黄闪警示、熄灯、故障状态

**`forwardRed` / `forwardGreen` 数据来源**：BSTLD / LISA / S2TLD 的直行箭头样本**可能已存在**但在 R1 被折叠为 `red` / `green`。R2 前用脚本复扫 + 人工复核，不足部分与栏杆同批实采。

### 合并策略：一个 10–14 类模型，而非多模型

| 组 | 最小 | 最大 |
|----|------|------|
| 交通灯 | 9（已确认）| 12 |
| 栏杆 | 1（单类 MVP）| 2（armOn + armOff）|
| **总 `nc`** | **10** | **14** |

理由：
1. **共享主干的正则效应**：联合训练通常比两个独立模型更鲁棒；栏杆与交通灯语义差异大，互相干扰低。
2. **部署成本**：Orin 单引擎 1280 分辨率现 ~25 ms / 帧，延迟预算 50 ms；拆两模型 → ~50 ms，逼近上限。
3. **维护成本**：一份 ONNX → 一份引擎 → 一套后处理 → 一份 class map。

**唯一例外**：若栏杆与交通灯在不同摄像头 / 不同频率下工作，则拆分更合理。此点须 PM 确认部署拓扑。

### 代码与配置变更预清单（待类别锁定后一次性修改）

| 文件 | 变更 |
|------|------|
| `data/traffic_light.yaml` | `nc` 和 `names` |
| `scripts/strip_yolo26_head.py` | 调用处 `--num-classes` |
| `inference/cpp/include/trt_pipeline.hpp` | `kClassNames` 数组及 `std::array<.., N>` 的 N |
| `inference/cpp/src/demo.cpp` | `kColors` BGR 调色板 |
| `inference/trt_pipeline.py` | `CLASS_NAMES` dict |
| `configs/*.yaml` | 硬编码类别引用（如有） |

R2 启动前只需 PM 提供**最终类别清单**，以上 6 处改动 <0.5 天可完成。

### 对 R2 数据计划的优先级重排

在 §数据侧原有项基础上追加两项并与现有项共享出差机会：

1. **现场采集专项**（合并三目标）：部署域评估集 + `armOn` / `armOff` 训练数据 + PM 清单中的新交通灯类别。
2. **PM 清单锁定**（阻塞项）：数据采集开始**之前**出一页 SOP，锁定最终类别清单 + 每类视觉定义 + 边界情况处理。SOP 缺失是 R1 重标注返工的第一原因。

---

## 产物清单（R1）

| 产物 | 路径 | 说明 |
|------|------|------|
| 训练权重 | `runs/detect/yolo26{n,s}-r1/weights/best.pt` | n / s 变体最佳 |
| ONNX（640）| `runs/detect/yolo26{n,s}-r1/weights/stripped.onnx` | 640 导出 + 裁头 |
| ONNX（1280 / 1536）| `runs/detect/yolo26{n,s}-r1/weights/stripped_{1280,1536}.onnx` | 高分辨率导出 + 裁头 |
| TRT 引擎 | `runs/detect/yolo26{n,s}-r1/weights/best*.engine` | Orin 端构建 |
| Demo 回放 | `runs/diagnose/{n,s}-pt-{640,1280,1536}/demo.mp4`、`demo/s-r1-{1280,1536}.mp4` | 诊断用 |
| 训练指标 | `runs/detect/yolo26{n,s}-r1/results.csv` | 逐轮记录 |
| 部署文档 | [`../integration/trt_pipeline_guide.md`](../integration/trt_pipeline_guide.md) | |

---

## 变更记录

| 日期 | 作者 | 变更 |
|------|------|------|
| 2026-04-21 | Zhengri Wu | 初版 — R1 两个模型完成、部署链路打通、demo 性能诊断、R2 计划 |
| 2026-04-21 | Zhengri Wu | 诊断实验：Ultralytics `.pt` 原生在 640 / 1280 / 1536 回放 demo；同域 val 分辨率扫描（s-r1）从 640 mAP50 0.85 降至 1280 的 0.70 — 揭示"部署端提高 imgsz 修 OOD demo"与"同域精度下降"的权衡 |
| 2026-04-21 | Zhengri Wu | 定位"框过大"根因：C++/Python TRT postprocess 把输出 4 通道误解为 `cxcywh`，实际是 `xyxy`。已修复 `inference/cpp/src/trt_pipeline.cpp` 与 `inference/trt_pipeline.py`；ORT 验证 f100 从"67% 面积大框"变为"21×50 紧贴目标"，与 Ultralytics 原生一致 |
| 2026-04-21 | Zhengri Wu | Orin 端延迟实测：s-r1 @ 1280 → ~25 ms/帧，@ 1536 → ~28 ms/帧，均远低于 50 ms 预算。R2 锁定 `imgsz=1280` 训练，1536 作为 OOD 诊断备选 |
| 2026-04-21 | Zhengri Wu | R2 范围扩展：PM 确认交通灯 +2 类（`forwardGreen` / `forwardRed`），总交通灯最少 9 类、最多 12 类；栏杆 MVP 单类 `barrier`，最佳实践 `armOn` / `armOff`。联合模型总 `nc` 范围 **10–14** |
| 2026-04-22 | Zhengri Wu | YOLO26m-r1 训练完成：72 轮早停，best@21 mAP50=0.869 / mAP50-95=0.635 / P=0.934 / R=0.712。相对 s-r1 提升有限（+2.0 / +2.7 pp），Orin 部署仍首选 s |
| 2026-04-22 | Zhengri Wu | R1 范围扩展：并行训练 YOLOv13-s（AGPL 低风险对照）+ DEIM-D-FINE-S/M（Apache-2.0 商用备选 + 精度上限）。选型依据 `yolo26_alternatives_survey.md`；DEIM-N 按性价比评估不训练。三轨同数据、同切分，指标可直接横向对比。R1 决策点写入 §R1 备选架构 |
| 2026-04-23 | Zhengri Wu | DEIM-S/M 训练配置修正：`deim_hgnetv2_{s,m}_traffic_light.yml` 显式覆盖 `train_dataloader.dataset.transforms.ops`，剔除 `base/dataloader.yml` 继承的 `RandomHorizontalFlip`（否则 `redLeft↔redRight` / `greenLeft↔greenRight` 语义反转）。Mosaic / 光学 / IoU-crop / mixup 保留，`stop_epoch` 调度不变。优化器 / LR / 调度器未调整 — 已按单卡 4090 + COCO fine-tune 正确缩放 |
| — | — | **待补充**：部署域评估集建立后的基准指标 |
