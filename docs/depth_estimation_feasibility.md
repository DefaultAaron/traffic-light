# 红绿灯深度估计方案 — 可行性分析

> **阶段**：研究测试阶段，许可证限制暂不作为排除条件，但标注风险。

## 背景

规划模块需要每个检测到的红绿灯的深度（距离）信息，以关联车道并决策。当前 LiDAR 集成方案存在问题，考虑利用车辆上的**两个摄像头**通过深度估计模型获取近似深度。

**硬件平台**：NVIDIA Jetson AGX Orin 64GB（与完整自动驾驶栈共享）

---

## 架构方案：并行推理

检测模型与深度模型可在同一 GPU 上并行运行，总延迟取决于较慢的模型，而非两者之和。

```
Camera Frame ──┬──→ YOLO 检测 (GPU/DLA)  ──→ Detection2DArray ──┐
               │                                                 ├──→ 合并：每个检测结果附带深度
               └──→ 深度估计 (GPU/DLA)   ──→ Dense Depth Map ──┘
                                                   ↓
                                         在 bbox 位置采样深度值
```

**关键设计决策**：

- 深度模型对**全帧**运行，在检测到的 bbox 位置采样深度值
- **不做裁剪推理**——裁剪后的 patch 会丢失视差搜索范围、上下文纹理和矫正几何信息，导致立体匹配失败
- 两个模型并行执行，总延迟 = max(检测延迟, 深度延迟)

---

## Orin 64GB 并行推理能力

### GPU 规格

| 项目 | 值 |
|------|-----|
| 架构 | NVIDIA Ampere |
| CUDA 核心 | 2048（16 个 SM） |
| Tensor 核心 | 64 |
| DLA 加速器 | 2 个独立 DLA 核心 |
| 统一内存 | 64GB LPDDR5 |
| 内存带宽 | 204.8 GB/s |
| AI 算力 | 275 TOPS（峰值） |

### 并行执行策略

| 策略 | 真并行？ | 说明 |
|------|----------|------|
| 同一 GPU 不同 CUDA Stream | ⚠️ 时间片轮转 | 两个模型竞争 SM/L2/带宽，性能下降约 5–15% |
| GPU + DLA | ✅ 真并行 | 检测在 GPU，深度在 DLA（或反之），无资源竞争 |
| 两个 DLA | ✅ 真并行 | 但 DLA 算子支持有限，大型 ViT 模型可能不兼容 |

**推荐方案**：YOLO 检测在 **GPU** 运行，深度模型在 **DLA** 运行（如算子兼容），实现真正的空间并行。如 DLA 不兼容，退回双 CUDA Stream（性能损失约 5–15%）。

### 内存预算

| 模型 | FP16 引擎大小 | 推理峰值内存 | 合计 |
|------|--------------|------------|------|
| YOLO 检测 (1280 输入) | ~25 MB | ~500 MB | ~525 MB |
| 深度模型（典型） | ~50–500 MB | ~650–1200 MB | ~1.2 GB |
| **合计** | | | **~1.7 GB** |

64GB 统一内存下，两个模型合计 ~1.7GB，**内存不是瓶颈**。

### 功耗

| 功耗模式 | GPU 频率 | 适用场景 |
|----------|---------|---------|
| 30W | 1.8 GHz | 车载部署推荐，被动散热可行 |
| 50W | 2.0 GHz | 性能提升约 20%，需主动散热 |
| MAXN (60W) | 2.27 GHz | 研究测试阶段可用，生产环境可能降频 |

---

## 候选模型对比

### 立体深度模型（需要标定的双目摄像头）

| 模型 | 延迟 (RTX 3090) | 延迟 (Orin 估计) | TensorRT | Jetson 验证 | 精度 (Middlebury BP-2) | 许可证 |
|------|-----------------|-----------------|----------|------------|----------------------|--------|
| **FoundationStereo** | ~250ms @ 640x480 (PyTorch) | ~150–200ms (TRT) | ✅ 官方 | ✅ readme_jetson.md | 4.4%（零样本 SOTA） | ⚠️ 联系 NVIDIA |
| **Fast-FoundationStereo** | 14–23ms @ 640x480 (TRT) | ~42–94ms (TRT) | ✅ 官方 | ✅ 已验证 | **2.12%**（超越原版） | ⚠️ 联系 NVIDIA |
| **HITNet** | 实时级 | ~30–50ms | ✅ 社区 | ✅ Orin Nano 实测 | 一般 | ✅ Apache 2.0 |
| CREStereo | 中等 | 未知 | ⚠️ ONNX 可用 | ❌ | 中等 | 研究用途 |
| RAFT-Stereo | 较慢 | 不适合实时 | ⚠️ 需适配 | ❌ | 高 | 研究用途 |

**FoundationStereo vs Fast-FoundationStereo**：

| | FoundationStereo | Fast-FoundationStereo |
|---|---|---|
| 架构 | ViT + CNN 混合骨干 + GRU 迭代细化 | 知识蒸馏 + 神经架构搜索 + 结构化剪枝 |
| RTX 3090 TRT (640x480) | ~50ms | **14–23ms**（快 2–4x） |
| Orin 估计 (TRT) | ~150–200ms | **~42–94ms** |
| Middlebury BP-2 | 4.4% | **2.12%**（更好） |
| ETH3D BP-2 | 1.1% | **0.62%** |
| 实时可行性 | ❌ Orin 上仅 5–7 FPS | ✅ Orin 上 11–24 FPS |

> Fast-FoundationStereo 在提速 2–10x 的同时，精度反而**优于**原版（得益于蒸馏和更大训练集）。

### 单目深度模型（单摄像头即可）

| 模型 | 参数量 | 延迟 (Orin) | TensorRT | 度量深度 | 最大深度 | 许可证 |
|------|--------|------------|----------|----------|----------|--------|
| **Depth Anything v3-S** | 24.7M | ~50–80ms (估) | ⚠️ 待验证 | ✅ Metric 系列 | **无硬性上限** | ✅ Apache 2.0 |
| **Depth Anything v3-Base** | 未公开 | ~80–120ms (估) | ⚠️ 待验证 | ✅ Metric 系列 | 无硬性上限 | ✅ Apache 2.0 |
| **Depth Anything v2-S (metric)** | 24.8M | 23.5ms @ 308 (实测) | ✅ 社区 | ✅ 绝对深度 | **80m** | 待确认 |
| **Metric3D v2 (lightweight)** | 3.83M | **~6ms** (TRT, 官方) | ✅ ONNX→TRT | ✅ 绝对深度 | 无限制 | ⚠️ **非商用** |
| UniDepth v2 | 未公开 | 未知 | ⚠️ 可能 | ✅ 无需标定 | 未知 | 待确认 |

**Depth Anything v3 vs v2 关键改进**：

| | v2 | v3 |
|---|---|---|
| 架构 | 纯单目 | **统一多视角**（单目/双目/多视角/视频） |
| 最大深度 | 80m（metric outdoor） | **无硬性限制** |
| 远距离精度 | 80m 外不可靠 | 显著改进（更好的天空/远景处理） |
| 几何一致性 | 帧间可能漂移 | 空间一致 |
| 多视角融合 | 不支持 | ✅ 可利用双目输入提升精度 |
| TRT 支持 | ✅ 社区验证 | ⚠️ 尚无官方 TRT 流水线 |
| 许可证 | 待确认 | Small/Base: Apache 2.0 ✅ |

> v3 消除了 v2 的 80m 深度上限，且支持多视角融合——如果双目可用，即使不做传统立体匹配，也可利用双目输入提升深度精度。

---

## 延迟分析：并行推理场景

假设两个模型并行运行（GPU + DLA 或双 CUDA Stream），**总延迟 = max(检测, 深度)**。

YOLO 检测 (imgsz=1280) 基线延迟：**~15ms on Orin GPU**

### 立体方案

| 方案 | 深度延迟 (Orin TRT) | 总延迟 | 帧率 | 说明 |
|------|-------------------|--------|------|------|
| Fast-FoundationStereo (快速档) | ~42–56ms | **~42–56ms** | **18–24 FPS** | 精度最优，推荐 |
| Fast-FoundationStereo (精度档) | ~70–94ms | **~70–94ms** | **11–14 FPS** | 更高精度 |
| FoundationStereo (原版) | ~150–200ms | **~150–200ms** | **5–7 FPS** | ⚠️ 偏慢，研究用 |
| HITNet | ~30–50ms | **~30–50ms** | **20–33 FPS** | 最快，许可友好 |

### 单目方案

| 方案 | 深度延迟 (Orin) | 总延迟 | 帧率 | 说明 |
|------|----------------|--------|------|------|
| Metric3D v2 lightweight | ~6ms | **~15ms** | **~67 FPS** | 最快，⚠️ 非商用 |
| DA v2-S metric (308x308) | ~24ms | **~24ms** | **~42 FPS** | 已验证，⚠️ 80m 上限 |
| DA v3-S metric | ~50–80ms (估) | **~50–80ms** | **13–20 FPS** | 无深度上限，TRT 待验证 |
| DA v3-Base metric | ~80–120ms (估) | **~80–120ms** | **8–13 FPS** | 更高精度，TRT 待验证 |
| DA v2-S metric (518x518) | ~98ms | **~98ms** | **~10 FPS** | 高分辨率深度图 |

---

## 关键限制与风险

### 距离精度

红绿灯典型距离：30–150m。

**立体深度误差**（受基线距离影响，假设 ±0.5px 视差误差）：

| 基线 | 50m | 100m | 150m |
|------|-----|------|------|
| 12cm | ±2.1m (4%) | ±8.3m (8%) | ±18.8m (13%) |
| 30cm | ±0.8m (2%) | ±3.3m (3%) | ±7.5m (5%) |
| 50cm | ±0.5m (1%) | ±2.0m (2%) | ±4.5m (3%) |

> 基线 ≥30cm 才能在 100m 处获得可用精度（±5m）。

**单目深度精度**：
- DA v2 metric outdoor 最大深度 **80m**——超过此距离不可靠
- DA v3 无硬性上限，远距离精度显著改善，但绝对精度仍不如立体方案
- Metric3D v2 无深度上限，远距离有尺度漂移风险
- 单目深度整体不如立体深度可靠，但无需硬件前置条件

### 硬件前置条件

| 条件 | 立体方案 | 单目方案 |
|------|---------|---------|
| 双目标定（内外参） | **必须** | 不需要 |
| 双目同步（<5ms） | 推荐（红绿灯静态，可放宽） | 不适用 |
| 重叠视场 | **必须** | 不适用 |
| 基线 ≥30cm | 推荐 | 不适用 |
| 相机内参 | 需要 | Metric3D v2 可选，DA 系列不需要 |

### 许可证风险

> 当前为研究测试阶段，许可证限制不作为排除条件，但**必须在商用部署前解决**。

| 模型 | 许可证 | 研究使用 | 商用部署 | 备注 |
|------|--------|---------|---------|------|
| HITNet | Apache 2.0 | ✅ | ✅ | 无风险 |
| Depth Anything v3 Small/Base | Apache 2.0 | ✅ | ✅ | 无风险 |
| Depth Anything v3 Large/Giant | CC BY-NC 4.0 | ✅ | ❌ | 大模型受限 |
| Depth Anything v2 | 待确认 | ✅ | ⚠️ | 需查看具体条款 |
| FoundationStereo | 需联系 NVIDIA | ✅ | ⚠️ | 联系 bowenw@nvidia.com |
| Fast-FoundationStereo | 需联系 NVIDIA | ✅ | ⚠️ | 同上 |
| Metric3D v2 | BSD 2-Clause 非商用 | ✅ | ❌ | 需联系作者获取商用授权 |

---

## 推荐方案

### 研究测试阶段推荐

#### 首选：Fast-FoundationStereo（如果双目可用）

最佳精度-速度平衡，NVIDIA 官方支持，Jetson 部署文档齐全。

| 项目 | 值 |
|------|-----|
| 输入 | 640x480 矫正立体对 |
| Orin 延迟 (TRT FP16) | ~42–94ms（取决于配置） |
| 并行总延迟 | ~42–94ms（检测 15ms 被掩盖） |
| 帧率 | 11–24 FPS |
| 精度 | Middlebury BP-2: 2.12%（零样本 SOTA） |
| 内存 | ~650 MB 峰值 |
| 部署 | 官方 TRT 脚本 + Jetson 指南 |

#### 备选 1：Depth Anything v3（如果双目不可用或 TRT 待验证期间）

v3 消除了 v2 的 80m 深度限制，且支持多视角融合。

| 项目 | 值 |
|------|-----|
| 输入 | 单目（可选多视角融合） |
| Orin 延迟 | ~50–120ms（估计，视模型大小） |
| 并行总延迟 | ~50–120ms |
| 帧率 | 8–20 FPS |
| 最大深度 | 无硬性上限（v2 的 80m 限制已消除） |
| 许可证 | Small/Base: Apache 2.0 ✅ |
| ⚠️ 风险 | TRT 导出尚未官方验证 |

#### 备选 2：Metric3D v2 lightweight（快速原型验证）

6ms 延迟近乎免费，适合快速验证深度融合流水线。

| 项目 | 值 |
|------|-----|
| 输入 | 单目 |
| Orin 延迟 (TRT) | ~6ms |
| 并行总延迟 | ~15ms（瓶颈在检测端） |
| 帧率 | ~67 FPS |
| 度量深度 | ✅ 绝对深度 |
| ⚠️ 风险 | 非商用许可，生产部署需授权 |

#### 备选 3：HITNet（许可证最安全）

Apache 2.0，Jetson 验证，如果精度要求不高且商用许可是硬约束。

| 项目 | 值 |
|------|-----|
| Orin 延迟 | ~30–50ms |
| 帧率 | 20–33 FPS |
| 许可证 | ✅ Apache 2.0，无任何限制 |

### 建议实验路径

```
第 1 步：Metric3D v2 lightweight 快速原型
         → 验证 "检测+深度并行" 流水线架构
         → 验证 bbox 位置深度采样逻辑
         → 6ms 延迟，不影响检测帧率

第 2 步：确认摄像头硬件条件
         → 双目？基线？标定？
         → 决定走立体还是单目路线

第 3 步A（双目可用）：Fast-FoundationStereo
         → 官方 Jetson 部署指南
         → 精度远超其他方案
         → Benchmark 实际 Orin 延迟

第 3 步B（仅单目）：Depth Anything v3
         → 无 80m 深度限制
         → 验证 TRT 导出
         → Apache 2.0 许可（Small/Base）
```

---

## 综合建议

1. **架构设计应解耦**：深度模块作为独立输入，检测模块不依赖深度。允许后续替换深度源（立体/单目/LiDAR 恢复后）
2. **先用 Metric3D v2 搭建流水线原型**：6ms 延迟近乎免费，可快速验证架构可行性，再替换为生产模型
3. **优先确认摄像头硬件条件**：这决定了立体方案是否可行——基线、标定、同步
4. **Depth Anything v3 值得关注**：消除了 v2 的 80m 限制，多视角融合能力意味着即使不做传统立体匹配也能利用双目
5. **输出格式**：建议在现有 `Detection2DArray` 之外发布独立深度 topic，或与规划组协商扩展消息格式

---

## 对现有时间线的影响

深度估计模块可在 Week 3–4 开发（与数据采集/标注并行），不影响当前 Week 1–2 的检测模型开发：

```
Week 1-2: 检测模型开发 + R1 训练部署（不变）
Week 2:   确认摄像头硬件条件，选择深度方案
Week 3:   快速原型（Metric3D v2）验证流水线架构
Week 3-4: 正式深度模型集成（Fast-FoundationStereo 或 DA v3）
Week 4:   集成测试（检测 + 深度并行 benchmark）
Week 5:   实车验证
```

---

## 待确认事项

- [ ] 两个摄像头的安装方式：双目立体对还是独立摄像头？
- [ ] 基线距离（两摄像头间距）
- [ ] 是否已完成双目标定？
- [ ] 规划模块需要的深度精度（±5m @ 100m 是否足够？）
- [ ] 深度信息的输出方式：扩展现有消息还是单独 topic？
- [ ] Depth Anything v3 TensorRT 导出验证
- [ ] Fast-FoundationStereo 实际 Orin benchmark

---

## 参考资料

- [FoundationStereo (CVPR 2025)](https://github.com/NVlabs/FoundationStereo) — 零样本立体匹配，Middlebury/ETH3D SOTA
- [Fast-FoundationStereo (CVPR 2026)](https://github.com/NVlabs/Fast-FoundationStereo) — 知识蒸馏加速版，2–10x 提速
- [Depth Anything v3 (2025)](https://github.com/ByteDance-Seed/Depth-Anything-3) — 统一多视角深度，无深度上限
- [Depth Anything v2 (2024)](https://github.com/DepthAnything/Depth-Anything-V2) — 单目深度基准
- [Metric3D v2 (CVPR 2024)](https://github.com/YvanYin/Metric3D) — 度量深度，轻量变体 6ms on Orin
- [HITNet (CVPR 2021)](https://github.com/ibaiGorordo/HITNET-Stereo-Depth-estimation) — 实时立体匹配，Apache 2.0
- [Depth Anything v2 Jetson Benchmark](https://github.com/IRCVLab/Depth-Anything-for-Jetson-Orin) — Orin 实测数据
- [Fast-FoundationStereo 论文](https://arxiv.org/abs/2512.11130)
- [SteROI-D (IEEE 2025)](https://arxiv.org/abs/2502.09528) — ROI 立体深度系统设计
