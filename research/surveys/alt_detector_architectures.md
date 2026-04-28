# YOLO26 替代方案调研

> **状态（2026-04-22）**：**并行训练已启动** — YOLOv13-s + DEIM-D-FINE-S/M 已作为 R1 备选轨道投入训练，与 YOLO26s/m 共用 `data/merged/` 和 80/20 切分。选型依据仍为本文；训练完成后结果填入 [`../reports/phase_2_round_1_report.md`](../../docs/reports/phase_2_round_1_report.md) §R1 备选架构，按同报告的决策规则决定 R1 部署基线。
>
> **结论先行**：主力 YOLO26；并行对照 **YOLOv13-s**（最低集成风险）+ **DEIM-D-FINE-S/M**（Apache-2.0，精度上限）。不训练 DEIM-N，不更换到 RT-DETR 家族。

---

## 一、部署约束（先于精度讨论）

| 约束 | 数值 | 对选型的影响 |
|------|------|-------------|
| 部署平台 | NVIDIA Jetson AGX Orin 64GB | 与完整自动驾驶栈共享 GPU |
| 推理栈 | **JetPack 5.1 / TensorRT 8.5** | ⚠️ 不原生支持 `MultiscaleDeformableAttnPlugin_TRT`，所有 DETR 系需自行从 TensorRT-OSS 构建插件 |
| 延迟预算 | 50 ms/帧 @ 1280 输入 | R1 YOLO26s 当前 ~25 ms，剩余 ~25 ms 空间 |
| 商用许可 | 需 Apache-2.0 / MIT 或购买企业许可 | AGPL-3.0 （Ultralytics）仅研究可用 |
| 截止日期 | 2026-05-15 实车测试 | 无全量重写预算；仅接受 ONNX → TRT 8.5 引擎级替换 |

**关键判断**：TRT 8.5 是最硬的约束。纯卷积类 YOLO 可以无痛替换；所有 DETR / RT-DETR 后代都需要一次性的 OSS 插件编译工作（1–2 天工作量，但可复用）。

---

## 二、候选模型对比

### 纯卷积 YOLO 系

| 模型 | 年份 | 参数量 | COCO mAP50-95 | 许可证 | TRT 8.5 | 判定 |
|------|------|--------|---------------|--------|---------|------|
| YOLO26-s（当前基线）| 2026.01 | ~9M | 基线 0.628（本项目） | AGPL-3.0 | 原生 ONNX | — |
| **YOLOv13-s** | 2025 | ~9M | ~52 | AGPL-3.0 | 原生 ONNX，HyperACE 无自定义算子 | **值得尝试** |
| YOLOv12-s | NeurIPS'25 | ~9M | 48.0 | AGPL-3.0 | 原生 ONNX | 边际 — 低于 YOLO26s |
| Hyper-YOLO-S | TPAMI'25 | 小 | +5 AP vs Gold-YOLO-N | GPL-3.0 | 原生 ONNX | 边际 — 许可证比 Ultralytics 更严 |
| YOLO-MS-S | 2023 | 8M | 46.2 | GPL-3.0 | 原生 ONNX | 跳过 — 低于基线 |
| Gold-YOLO-S | 2023 | 22M | 46.4 | GPL-3.0 | 原生 ONNX | 跳过 — 低于基线 |

### DETR / Transformer 系

| 模型 | 年份 | 参数量 | COCO mAP50-95 | 许可证 | TRT 8.5 | 判定 |
|------|------|--------|---------------|--------|---------|------|
| **DEIM-D-FINE-S** | CVPR'25 | 10M | 49.0 | Apache-2.0 | 需 OSS deformable-attn 插件 | **值得尝试** |
| **DEIM-D-FINE-M** | CVPR'25 | 19M | 52.7 | Apache-2.0 | 同上 | 值得尝试（精度上限） |
| D-FINE-S | ICLR'25 | 10M | 48.5 | Apache-2.0 | 同上，仓库针对 TRT 10.4 | 值得尝试（DEIM 备选） |
| DEIMv2-S | arXiv'25 | 9.7M | 50.9 | Apache-2.0 | DINOv3 backbone，TRT 8.5 未知 | 边际 — 论文承认小目标 AP 未提升 |
| DEIMv2-Pico | arXiv'25 | 1.5M | 38.5 | Apache-2.0 | 同上 | 跳过 — 精度低 |
| RF-DETR-N/S/M | ICLR'26 | 30–34M | 48.4–54.7 | Apache-2.0 | DINOv2 backbone，仅测过 TRT 10 | **跳过** — 30M+ 参数量会吃光 Orin 50ms 预算 |
| LW-DETR-tiny/small | arXiv'24 | 12–15M | 42.6–48.0 | Apache-2.0 | 带 `--trt` 导出但仍需插件 | 边际 — 同规模不如 D-FINE |
| RT-DETRv3-R18 | WACV'25 | ~20M | 48.1 | Apache-2.0 | 同插件依赖 | **跳过** — 已在 R1 证实 RT-DETR 家族在本数据集上不收敛 |
| Co-DETR（小）| — | 40M+ | 52+ | MIT | 插件密集 | 跳过 — 参数量超标 |

---

## 三、Top 3 推荐（按集成风险排序）

### 1. YOLOv13-s — 最低风险对照

**定位**：R2 的平行对照实验（hedge）。

- **理由**：与 YOLO26 共用 Ultralytics 训练栈，训练脚本基本零改动；HyperACE（超图注意力）无自定义 ONNX 算子，原生 TRT 8.5 可导出；论文报告 ≥+1.5 AP 相对 YOLOv12。
- **许可证**：AGPL-3.0（与 YOLO26 同问题，无额外负担）
- **集成成本**：~0.5 天（新增 config yaml + 权重下载）
- **训练成本**：与 YOLO26s 相当（RTX 4090 D 约 2–3 小时 @ 640）
- **适用场景**：R2 训练完成后仍想交叉验证，或若 YOLO26 R2 出现不稳定想快速换一个基线
- **链接**：[arXiv 2506.17733](https://arxiv.org/abs/2506.17733)

### 2. DEIM-D-FINE-S — 精度上限候选

**定位**：长期候选，商用许可友好（Apache-2.0）。

- **理由（针对本项目）**：
  - **DEIM 的 Dense-O2O 匹配** 在训练时显著增加每张图的正样本数 → 直接缓解黄灯仅占 2.7% 的类别不平衡问题
  - **D-FINE 的 FDR（Fine-grained Distribution Refinement）回归头** 将边框回归建模为分布而非点估计 → 有助于远距离小目标（1–3% 图像宽度）的亚像素定位
  - Apache-2.0 许可证，商用化时不需要 Ultralytics 企业授权
- **集成成本**：~2–3 天
  - 一次性：在 Orin 上从 TensorRT-OSS 构建 `MultiscaleDeformableAttnPlugin_TRT`（可复用于未来所有 DETR 模型）
  - 训练脚本不在 Ultralytics 框架内，需独立训练流程
- **训练成本**：与 D-FINE 相当（官方报告 DEIM 比 D-FINE 训练快 50%）
- **风险**：
  - 训练稳定性未在我们数据集上验证
  - TRT 8.5 插件构建有一次性坑
- **链接**：[DEIM 仓库](https://github.com/Intellindust-AI-Lab/DEIM) / [D-FINE 仓库](https://github.com/Peterande/D-FINE) / [arXiv 2412.04234](https://arxiv.org/abs/2412.04234)

### 3. D-FINE-S（朴素版）— 安全网

**定位**：若 DEIM 在 55k 合并数据集上训练不稳定时的回退选项。

- **理由**：与 DEIM-D-FINE-S 共享 FDR 架构优势；训练流程更成熟（DEIM 是在 D-FINE 之上的改进）；同 Apache-2.0
- **集成成本**：同 DEIM（共用插件）
- **链接**：[D-FINE 仓库](https://github.com/Peterande/D-FINE) / [arXiv 2410.13842](https://arxiv.org/abs/2410.13842)

---

## 四、明确排除项

| 模型 | 排除理由 |
|------|---------|
| **RF-DETR** | 尽管 mAP 最高（54.7），但参数量下限 30M、依赖 DINOv2 backbone，在 Orin 共享 GPU 场景下 1280 输入会超 50ms 预算；其核心优势是 fine-tuning 收敛性，而这不是本项目的痛点 |
| **RT-DETRv3 全系** | R1 已证实 Ultralytics 版 RT-DETR-L 在本数据集 21 epoch 不收敛（mAP50 仅 0.477）；同家族架构倾向一致，不值得再次投入 |
| **DEIMv2** | 论文明确承认小目标 AP 未改善 vs DEIM-S，语义提升主要在中/大目标；与我们"远距离小信号灯"场景不匹配 |
| **YOLOv10–v12** | COCO 指标均低于或与 YOLO26s 持平，无明显升级理由 |

---

## 五、交通灯领域专项研究

扫描 2024–2026 年 BSTLD / DriveU / LISA 上的专项工作，**无值得切换架构的通用型方案**。结论是：该领域主流做法仍是"通用实时检测器 + 仔细的类别平衡 + 相关性头"。

| 工作 | 备注 | 是否纳入路线图 |
|------|------|---------------|
| [TLD-READY](https://arxiv.org/abs/2409.07284) | 在 BSTLD+LISA+DTLD+Karlsruhe 合并集上 benchmark YOLOv7/8 和 RT-DETR。**关键发现：RT-DETR 在 DTLD 上显著弱于 BSTLD/LISA** — 与我们 R1 RT-DETR-L 不收敛的观察一致 | 作为"避免 RT-DETR"的佐证引用 |
| [CSDETR](https://link.springer.com/article/10.1007/s11554-026-01864-6) | RT-DETR 衍生 TL 检测器，BSTLD 上 +2.4 mAP50 / −13% 参数 / 346 FPS | 单数据集论文，不追 |
| [ATLAS](https://arxiv.org/html/2504.19722) | TL 感知框架，非新 backbone；仅相关性逻辑 | R2 之后的"相关性头"设计可参考 |
| [TLDR](https://arxiv.org/html/2411.07901v1) | 傅里叶域自适应，针对恶劣天气 | 若后续暴露天气泛化问题再看 |

---

## 六、DEIM 尺寸选型（N / S / M）

用户初始计划 N+S+M 三档全开。**实际选型：仅 S 与 M，跳过 N**。

| 尺寸 | 参数量 | COCO AP | 对标 YOLO26 | 决定 |
|------|--------|---------|--------------|------|
| N | 4M | 43.0 | 无（YOLO26n = 2.5M 更小） | **跳过** — 4M 参数 + DETR 架构在 7 类 + 黄灯 2.7% 不平衡场景下很可能欠拟合，且已有 YOLO26n 作为小模型锚点 |
| **S** | 10M | 49.0 | **YOLO26s = 9M**（直接对比） | **主实验** — 参数预算对等，是 DEIM 路线的核心验证 |
| **M** | 19M | 52.7 | YOLO26m = 20M | **精度天花板** — Orin @1280 可能超 50ms 预算但仍值得知道上限 |
| L | 31M | 54.7 | — | 不训练（超预算） |

**训练时长预估**（RTX 4090 D，参考 DEIM 官方 epoch 数）：

| 尺寸 | Epoch | @640 | @1280 |
|------|-------|------|-------|
| S | 132 | ~8–12 hr | ~25–36 hr |
| M | 102 | ~15–20 hr | ~45–60 hr |

**建议顺序**：S 先跑（确认 Apache-2.0 路线可行）→ M 后跑（精度上限）。N 仅在 S 跑完后若有时间再补，非必要。

## 七、环境设置

**推荐方案：复用现有 uv venv**

DEIM 依赖 `faster-coco-eval / tensorboard / scipy / calflops / transformers`；`torch / torchvision` 与 Ultralytics 共用。已将 `deim` 写入 `pyproject.toml` 的 optional-dependencies：

```bash
uv sync --extra deim
```

**备选方案：独立 conda 环境（若 uv 解析器冲突）**

```bash
conda create -n deim python=3.11.9 && conda activate deim
pip install -r DEIM/requirements.txt
```

conda 路线的代价：需要手动激活 / CI 多维护一条环境，但能完全隔离 DEIM 的 torch 版本需求。目前 DEIM 的 `torch>=2.0.1` 与我们的 ultralytics 可共存，预计不需要回退到此路线。

## 八、项目内实现状态（2026-04-22）

| 组件 | 路径 | 说明 |
|------|------|------|
| YOLOv13-s 训练脚本 | `scripts/train_yolov13.sh` | ⚠️ YOLOv13 自带 `DSC3k2` / `HyperACE` 模块，**无法用 stock Ultralytics 加载**。需单独克隆 [iMoonLab/yolov13](https://github.com/iMoonLab/yolov13) 并创建独立 venv。`configs/yolov13s.yaml` 已禁用（重命名为 `.bak`） |
| DEIM 项目配置 | `DEIM/configs/deim_dfine/deim_hgnetv2_{s,m}_traffic_light.yml` | S 与 M 双档训练，N 不训（见 §六） |
| DEIM 数据集配置 | `DEIM/configs/dataset/traffic_light_detection.yml` | `num_classes: 7`，`remap_mscoco_category: False` |
| YOLO→COCO 转换器 | `scripts/yolo_to_coco.py` | 读 `data/merged/`，写 `data/merged/annotations/instances_{train,val}.json`。**不复制图片**，与 Ultralytics 共用同一份图像目录 |
| DEIM 训练 wrapper | `scripts/train_deim.sh` | `./scripts/train_deim.sh {s\|m} -t weights/deim_dfine_{s,m}_coco.pth` |
| uv 扩展 | `pyproject.toml` → `[project.optional-dependencies]` → `deim` | |

**权重下载清单**：

| 权重 | 来源 | 用途 |
|------|------|------|
| `weights/yolov13s.pt` | [iMoonLab/yolov13 releases](https://github.com/iMoonLab/yolov13/releases) | YOLOv13-s 训练 |
| `weights/deim_dfine_s_coco.pth` | [Google Drive](https://drive.google.com/file/d/1tB8gVJNrfb6dhFvoHJECKOF5VpkthhfC/view) | DEIM-S fine-tune |
| `weights/deim_dfine_m_coco.pth` | [Google Drive](https://drive.google.com/file/d/18Lj2a6UN6k_n_UzqnJyiaiLGpDzQQit8/view) | DEIM-M fine-tune |
| HGNetv2 B0/B2 backbones | 由 `engine/backbone/hgnetv2.py` 自动下载 | DEIM 训练启动时联网拉取；缓存在 `DEIM/weight/hgnetv2/` |

## 九、长期视角（5/15 之后）

- DEIM-D-FINE-S 若在 R1/R2 中验证有效，可作为**商用化路径的首选架构**（规避 Ultralytics 企业许可费用）。Orin 侧需一次性完成 `MultiscaleDeformableAttnPlugin_TRT` 的 TensorRT-OSS 构建（~1 天，长期复用）。
- R3 若有 JetPack 升级到 6.x / TRT 10+ 的机会，RF-DETR 与 DEIMv2 可重新评估。

---

## 十、参考资料

1. [D-FINE (ICLR 2025)](https://github.com/Peterande/D-FINE) — [arXiv 2410.13842](https://arxiv.org/abs/2410.13842)
2. [DEIM (CVPR 2025)](https://github.com/Intellindust-AI-Lab/DEIM) — [arXiv 2412.04234](https://arxiv.org/abs/2412.04234)
3. [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2) — [arXiv 2509.20787](https://arxiv.org/abs/2509.20787)
4. [RF-DETR (ICLR 2026)](https://github.com/roboflow/rf-detr) — [arXiv 2511.09554](https://arxiv.org/abs/2511.09554)
5. [LW-DETR](https://github.com/Atten4Vis/LW-DETR) — [arXiv 2406.03459](https://arxiv.org/abs/2406.03459)
6. [YOLOv12 (NeurIPS 2025)](https://github.com/sunsmarterjie/yolov12) — [arXiv 2502.12524](https://arxiv.org/html/2502.12524v1)
7. [YOLOv13](https://arxiv.org/abs/2506.17733)
8. [Hyper-YOLO (TPAMI 2025)](https://github.com/iMoonLab/Hyper-YOLO)
9. [RT-DETRv3 (WACV 2025)](https://github.com/clxia12/RT-DETRv3)
10. [TLD-READY](https://arxiv.org/abs/2409.07284)
11. [ATLAS](https://arxiv.org/html/2504.19722)
12. [TensorRT LayerNorm / opset-17 问题](https://github.com/NVIDIA/TensorRT/issues/3346)
13. [TensorRT deformable-attn 插件（TAO）](https://docs.nvidia.com/tao/tao-toolkit/text/ds_tao/deformable_detr_ds.html)
