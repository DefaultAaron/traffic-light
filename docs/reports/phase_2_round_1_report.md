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
| YOLOv13-s | AGPL-3.0 | YOLO26 同架构家族低风险对照 | 100 / 100 | 0.815 | 0.580 | 0.836 | 0.767 | 完整跑完，未触发早停 |
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

## R1 跨架构综合分析（同域 val）

> 本节基于 `runs/detect/{yolo26n,yolo26s,yolo26m,yolov13s}-r1/` 训练曲线、混淆矩阵与逐类 PR 表（详见 [`phase_2_round_1_results.md`](./phase_2_round_1_results.md)）。**所有结论限同域 BSTLD/S2TLD/LISA 横屏 dashcam val 集**；部署域评估集尚未建立。DEIM-S/M 仍在训练，结果出来后再追加 §DEIM 评估。

### 整体指标对照

| 模型 | mAP50 | mAP50-95 | P | R | F1 | 训练轮次 | 备注 |
|------|-------|---------|----|---|----|---------|------|
| YOLO26n-r1 | 0.720 | 0.484 | 0.940 | 0.689 | 0.795 | 39 / 100 | 容量瓶颈，跨尺度泛化失败 |
| **YOLO26s-r1** | **0.849** | **0.608** | 0.930 | 0.670 | 0.778 | 46 / 100 | **R1 部署主力** |
| YOLO26m-r1 | 0.869 / **0.858** † | **0.646** † | 0.934 | 0.712 | 0.808 | 72 / 100 | 上限基准；†best by mAP50@21 / †best by mAP50-95@52 |
| YOLOv13-s | 0.815 | 0.580 | 0.836 | **0.767** | 0.800 | 100 / 100 | 全程跑完未早停；**召回最高，精度最低** |

> †YOLO26m-r1 的 `best.pt` 取自 Ultralytics fitness 综合最优（mAP50 在 ep21 达到 0.869）；若以 mAP50-95 为单一准绳，ep52 略胜（0.646）。两者部署影响差异 < 1 pp，沿用 ep21 best.pt。

### 关键观察

**1. YOLO26 系列容量与泛化的拐点在 s↔n 之间，不在 s↔m 之间。**
- n→s：mAP50 +12.9 pp，mAP50-95 +12.4 pp（**质变**）。
- s→m：mAP50 +2.0 pp，mAP50-95 +3.8 pp（**收益递减**）。Orin 1280 FP16 上 m 推理时间约为 s 的 1.6×（按 survey §M-vs-S 推算），ROI 不利。
- 结论与 R1 初版一致：部署沿用 **YOLO26s-r1**，m 仅留作精度上限基准与右转箭头召回的对照。

**2. YOLOv13-s vs YOLO26s 是"召回换精度"的非帕累托权衡。**
- YOLOv13-s 整体 mAP50 **低 3.4 pp**、mAP50-95 **低 2.8 pp** → **未触发决策规则 2 的主力切换条件**（要求 +3 pp）。
- 但 YOLOv13-s 的 Recall 反而 **+9.8 pp**（0.767 vs 0.670），代价是 Precision **-9.4 pp**（0.836 vs 0.930）。
- 反映在背景假阳上：YOLOv13-s 的 background→FP 各类率 13–17%（混淆矩阵 background 行），而 YOLO26s 仅 3–7%。
- **R1 决策**：YOLOv13-s 不替换主力，但保留为监控组 — 其高召回属性在 R2 部署域评估上若能保持，可在"宁可误报不漏检"的安全策略下变为有效备选。

**3. 右转箭头类（redRight / greenRight）的真实状况比表面指标更糟。**

| 模型 | redRight R | redRight 混淆矩阵主分布 | greenRight R | greenRight 混淆矩阵主分布 |
|------|-----------|----------------------|-------------|------------------------|
| YOLO26n-r1 | 0 | 67% → 漏检（background）；预测端无 redRight 输出 | 0 | 100% → 误判为 green |
| YOLO26s-r1 | 0.167 | 50% → 漏检；33% → 误判为 red | 0 | 100% → 误判为 green |
| YOLO26m-r1 | 0.333 | 67% → 漏检；33% → 误判为 red | 0 | 75% → 误判为 green；25% → 漏检 |
| YOLOv13-s | 0.333 | 50% → 漏检；33% 正确分类 | **0.655** | 25% 正确分类；25% → 误判为 greenLeft |

- YOLO26 系列的 greenRight `mAP50` 在 s/m 上分别为 **0.875 / 0.823**，**这是统计假象**：仅 4 个 GT 实例，少数高置信预测拉高 AP，但 R=0 表示模型在常用置信阈值下完全不报 greenRight 类。混淆矩阵证实"预测端 greenRight 行整行为空"。
- YOLOv13-s 是**唯一**在 greenRight 类有非零召回（R=0.655）且预测端非空的模型；归因于 HyperACE 的特征聚合更易于在长尾少样本类形成正样本激活。
- **结论**：R1 期间不能基于现有 right-arrow 指标做任何模型决策；样本量（19/13 实例）已低于统计可靠下限。R2 必须靠"反向 fliplr 合成 + 国内场景实采"补齐到每类 ≥ 500 实例，否则该类继续无意义。

**4. 训练效率：YOLOv13-s 显著更慢。**
- YOLO26s-r1：46 epochs × ~210 s/epoch ≈ **2.7 h**（早停于 ep26 best）。
- YOLOv13-s：100 epochs × ~340 s/epoch ≈ **9.4 h**（无早停，全程跑完，测试时间起伏来自 ep47 周围 close_mosaic + resume，参见 results.csv）。
- 同卡 4090，YOLOv13-s 单 epoch 慢 60% 主要源自 HyperACE 注意力路径；R2 升至 1280 训练后差距还会扩大。这把 YOLOv13-s 的"备选轨道"角色定调为"YOLO26 出 bug 时再启用"，平时不主用。

**5. 训练曲线形态。**
- YOLO26 系列（n/s/m）全部呈"早期 P 振荡 → 第 2–10 ep 跳到 0.85+ 平台 → 稳定渐进"。Precision 单调，Recall 随 close_mosaic 关闭后渐进。这是 R2 提前关 mosaic 调度还能挤出收益的信号。
- YOLOv13-s 呈"先 P 主导 → ep47 close_mosaic 触发 P↓R↑ 反转 → ep83 第二次反转再次 R 提升"两次台阶。两次台阶都来自训练策略切换而非模型瓶颈 → 验证了 YOLOv13 的 patience=20 在此任务上确实偏紧（如果按 P↑ 早停就会过早停在 ep45）。

### 类别分布偏置的传导路径（混淆矩阵证据）

所有四个模型一致表现出以下**结构性偏置**，源于训练集 redLeft >>> redRight（12,983 vs 19）的极端不平衡：

- `red` ↔ `redLeft`：YOLOv13-s 上 17% 的真 red 被误判为 redLeft（YOLO26 上仅 1–4%）— 表明 YOLOv13 更容易在低分辨率下"过度方向化"。
- `green` ↔ `greenRight`：所有 YOLO26 模型把 greenRight 的 75–100% 误判为 green。这不是召回率问题，是**类别先验"被吞并"问题** — 模型从未获得足够样本建立 greenRight 决策面。
- `redRight` 主漏检方向是 background（漏检），不是与 redLeft / red 的混淆 — 说明 redRight 真是"小目标 + 类别极少 + 低置信"被 NMS 滤掉。

R2 用于这两组的修复手段不同：
- redLeft↔redRight：靠 fliplr 合成（行动计划 §R2 第 2 项 A 方案）增加 redRight 实例；
- greenRight 漏检：必须实采，光靠 fliplr greenLeft 合成会同步引入数据集分布偏差（部分 greenLeft 是"上下颠倒可读"的非镜像模式）。

---

## Demo 视频实际表现观察

> R1 期间在 `demo/demo{1..15}.mp4`（共 15 段，含横屏 dashcam 与竖屏手机视频）上分别用 yolo26{n,s,m}-r1 的 `best.pt` 做了三种分辨率（640 / 1280 / 1536）的离线推理，结果存放于 `demo/yolo26{n,s,m}-r1/{best,best_1280,best_1536}/`。

### 推理覆盖矩阵

| 模型 | 640 | 1280 | 1536 | 总计 demo 段 |
|------|-----|------|------|------|
| yolo26n-r1 | 15/15 | 15/15 | 15/15 | 45 |
| yolo26s-r1 | 15/15 | 15/15 + 5(/best-1280 增量集) | 15/15 | 50 |
| yolo26m-r1 | 15/15 | 15/15 | 15/15 | 45 |

`yolo26s-r1/best-1280/` 含 demo1–5 的早期增量产物（diagnosis 阶段），与 `best_1280/` 全量集互不冲突。

### 视检方法

2026-04-26 以 ffmpeg 抽帧对 `yolo26s-r1/best_1280/demo{1..15}.mp4` 做了 9 帧均匀采样（3×3 montage）+ 4 段稳定性突击（demo4/10/12/15 各取 8 帧 0.25s 步长，共 2 秒滑窗），对照原始 demo 帧人工辨识 bbox 位置 / 类别。难场景（demo8/10/11/13）追加 yolo26m-r1 同时间戳 montage 做横向对比。所有 montage 缓存在 `/tmp/demo_inspect/`，本节结论均来自该次视检，分三个维度记录。

### 漏检（miss detection）

| Demo | 场景 | 漏检表现 | s vs m |
|------|------|----------|--------|
| demo1 | 远距离路口 | 远端小目标常常未被框出 | 两者均漏 |
| demo3 | 卡车遮挡 | 灯被前车短暂遮挡时立即丢失，不复现到再出现 | 两者均漏 |
| demo9 | 黄昏 + 龙门信号 | 低光下龙门灯几乎完全无检测 | 两者均漏 |
| demo11 | 强逆光（太阳直射） | 中段画面（远 / 中距离）龙门灯丢失，仅近距离能检出 | 两者均漏，但 m 在最后两帧能同时框出红绿 |
| demo13 | 卡车列队 + 龙门 | 大量帧龙门灯未被框出（侧向角度，灯朝向偏） | 两者均严重漏，m 略好但仍不及格 |
| demo14 | 港口排队 | 远端龙门信号几乎不被检出 | 两者均漏 |
| demo10 | 工业区横向龙门 | s 仅检出靠近的少数灯 | **m 显著优于 s**：每帧多检 1–3 个龙门灯 |

漏检的共同规律：**远距离 / 小目标 + 弱光 / 逆光 + 非正面朝向**三类条件叠加时，模型几乎无能为力。这与 §R1 跨架构综合分析中的"640 训练分辨率难以覆盖竖屏 / 远距小灯"诊断一致。

### 误检（wrong detection）

分两种：

**(a) 灯被框对但分类错误**
- demo10：龙门交通灯在多个帧中被一致地标为 `green`，但目测灯形偏暗（疑似不发光或为红，难以从画质判断）。s / m 均做出相同判定 — 这是模型层面的偏置，非偶发抖动。
- demo15：路口闸控指示灯 / 检查站指示灯被识别为交通灯类别（红或绿）。属于"语义边界外但视觉接近"的误判，本数据集未明确将此类样本纳入负样本，模型行为符合预期。

**(b) 非交通灯目标被误判为交通灯**（背景误报，FP）
- demo8（s-r1）：黄色三角警示牌被多帧识别为 `yellow`；卡车驾驶舱的黄灯也被吃成 `yellow`；绿色厂房墙面也曾出现 `green` 框 — **这是 s-r1 在该 demo 上最严重的失稳来源**。
- demo8（m-r1）：上述大部分错误大幅减少，但仍能在中段画面看到龙门桁架顶部出现 `green` 小框（背景结构误激活）。
- demo11：太阳光斑 / 警示牌区域偶发 `yellow` / 红框（视检不能 100% 确定属于警示牌还是真实灯，但与混淆矩阵中 `yellow` 行 background→FP 较高的趋势一致）。

总体上 m 的背景假阳显著少于 s，对应同域 val P 从 0.93 → 0.88，但 R 和 mAP50 都升高 — m 的容量主要花在"减少背景误激活"而非"提升远端召回"。

### 检测稳定性（同一物体的逐帧一致性）

抽样了 4 段 8 帧 / 0.25s 步长的突击：

| Demo | 场景 | 稳定性 |
|------|------|--------|
| demo4 | 近距离单体绿灯，画质干净 | **8/8 帧均稳定检出**，类别一致，无闪烁。理想场景下行为正常。 |
| demo10 | 中距离龙门绿灯（疑似类别误判） | 8/8 帧持续输出同一 `green` 框，**类别错误也是稳定的**（不是抖动） |
| demo12 | 卡车从前方驶离，露出龙门红灯 | 第 1 帧被卡车遮挡 → 无检；第 2–8 帧灯进入视野后立即出现红框并稳定保留。**遮挡转可见的恢复时间 ≤ 1 帧**。 |
| demo15 | 远端 / 小目标闸控灯 | **明显闪烁**：第 1, 3, 4 帧出现绿框，第 2, 5–8 帧无 — 在置信度阈值附近边缘震荡。 |

分类：
- 单体大目标（demo4） → 稳定。
- 单体中目标且有正确特征（demo12 红灯靠近过程） → 稳定。
- 持续误分类（demo10） → "稳定地错误"，反而比抖动更难自动识别为问题。
- 小目标 / 弱信号（demo15） → 抖动严重，部署侧需要 tracker + 多帧投票才能稳态化（与 R2 §temporal encoder 备选方案一致）。

### 视检结论（对部署 / R2 的指导）

1. **部署模型选 s（不变）**：m 在 demo10 类型场景（横向龙门）上有可见提升，但其他场景与 s 接近，且 demo8 的背景误检 m 也未根除。在 +2 pp mAP50 / +1.5× 推理时延的代价下，s 仍是 R1 部署最佳折衷。
2. **R2 需补的训练样本**：远距离龙门信号 / 黄昏 + 弱光 / 逆光强光斑 / 卡车密集队列 + 龙门 — 这些场景在当前训练集中覆盖不足，是漏检的主因。
3. **R2 需要 tracker + 多帧投票**：demo15 类小目标抖动靠提升模型容量解决性价比低，部署侧的时序聚合更高效（与已记录的 temporal encoder 决策一致 — 先靠 tracker，post-5/15 再考虑 LSTM）。
4. **demo8 假阳的根因可能在数据**：警示三角 / 厂房绿墙这类硬负样本若未在训练集出现，模型当然学不到拒识。R2 应主动从 demo 中挖出这些误报帧补回训练集做硬负学习。

### 视检材料路径

- s-r1 montages: `/tmp/demo_inspect/montages/s1280_demo{1..15}.jpg`（3×3 9 帧采样）
- m-r1 难场景对照: `/tmp/demo_inspect/montages/m1280_demo{8,10,11,13}.jpg`
- 稳定性突击: `/tmp/demo_inspect/stability/s_demo{4,10,12,15}_burst.jpg`（8 帧 0.25s 步长）

> 这些产物为临时缓存，R2 启动时应将关键示例（尤其是 demo8 假阳帧 / demo13 漏检帧）提取为 `data/r2_hard_examples/` 永久样本，并标注后回灌训练集。

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
| 训练权重 | `runs/detect/yolo26{n,s,m}-r1/weights/best.pt` | n / s / m 变体最佳 |
| ONNX（640）| `runs/detect/yolo26{n,s}-r1/weights/stripped.onnx` | 640 导出 + 裁头 |
| ONNX（1280 / 1536）| `runs/detect/yolo26{n,s}-r1/weights/stripped_{1280,1536}.onnx` | 高分辨率导出 + 裁头（部署选 s） |
| TRT 引擎 | `runs/detect/yolo26{n,s}-r1/weights/best*.engine` | Orin 端构建 |
| Demo 回放 | `runs/diagnose/{n,s}-pt-{640,1280,1536}/demo.mp4`、`demo/s-r1-{1280,1536}.mp4` | 诊断用；批量扫描脚本 `scripts/run_demos_all_engines.sh` |
| 训练指标 | `runs/detect/yolo26{n,s,m}-r1/results.csv` | 逐轮记录 |
| 跟踪库 | `inference/tracker/*.py`、`inference/cpp/{include,src}/tracker.{hpp,cpp}` | Python + C++ 两端；fixtures 驱动的 parity 单测 |
| 部署文档 | [`../integration/trt_pipeline_guide.md`](../integration/trt_pipeline_guide.md)、[`../integration/tracker_voting_guide.md`](../integration/tracker_voting_guide.md) | |

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
| 2026-04-26 | Zhengri Wu | YOLOv13-s 训练完成：100/100 epoch 全程跑完未触发早停，best mAP50=0.815 / mAP50-95=0.580 / P=0.836 / R=0.767。对比 YOLO26s-r1 **mAP50 低 3.4 pp / mAP50-95 低 2.8 pp / Recall 高 9.8 pp / Precision 低 9.4 pp** — 未达决策规则 2 的 +3 pp 切换阈值，**主力维持 YOLO26s**。但 YOLOv13-s 是唯一在 greenRight 类有非零召回（R=0.655）的模型 — 留作监控组，候选"宁可误报不漏检"安全策略下的备选。新增 §R1 跨架构综合分析 与 §Demo 视频实际表现观察 两节，整合训练曲线 / 混淆矩阵 / 逐类 PR / demo 输出文件层面的所有 R1 证据 |
| 2026-04-26 | Zhengri Wu | Demo 视频人工视检（s-r1 / m-r1）：对 demo{1..15}.mp4 抽 9 帧 montage + demo4/10/12/15 抽 8 帧 0.25s 突击 + demo8/10/11/13 加 m-r1 同时间戳对照。结论：(1) 漏检主因为远距离 + 弱光 / 逆光 + 非正面朝向，s/m 均无能为力；唯一例外是 demo10 横向龙门，m 显著优于 s；(2) 误检分两类——demo10 / 15 是稳定的类别 / 语义偏置（属模型 bias），demo8 警示三角 / 厂房绿墙是背景假阳（s 严重，m 大幅缓解但未根除）；(3) 稳定性：单体大目标稳定，遮挡恢复 ≤ 1 帧；持续误分类是"稳定地错误"；小目标抖动严重，需 tracker + 多帧投票（与 temporal encoder deferred 决策一致）。**部署仍选 s** — m 仅在特定场景有提升，整体不抵 1.5× 时延代价。视检产物缓存于 `/tmp/demo_inspect/`，关键样本应在 R2 启动时提取为 `data/r2_hard_examples/` 永久回灌 |
| — | — | **待补充**：DEIM-D-FINE-S/M 训练完成后的逐类指标 + 决策规则 3 触发判断 |
| — | — | **待补充**：部署域评估集建立后的基准指标 |
