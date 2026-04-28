# TensorRT 推理流水线 — Orin 部署指南

在 **NVIDIA Jetson AGX Orin 64GB**（出厂 JetPack 5.1.x / L4T R35.x）上部署红绿灯检测模型的端到端参考流程。

开发目标机：`nvidia@192.168.30.138`。

---

## 一、推理流水线概览

`inference/` 下提供两个等价的 TensorRT 检测器实现：

| | Python | C++ |
|---|---|---|
| 流水线 | `inference/trt_pipeline.py` | `inference/cpp/src/trt_pipeline.cpp` + `include/trt_pipeline.hpp` |
| Demo | `inference/demo.py` | `inference/cpp/src/demo.cpp` |
| 构建 | `uv`/pip 环境 | CMake ≥ 3.18，见 §三 |
| 使用场景 | 快速迭代、调试、JSON 遥测输出 | Orin 上生产部署 |

两者接受相同参数：`--source`、`--model`、`--conf`、`--imgsz`、`--no-show`、`--save`。类别名与绘制颜色在两种后端中保持同步；R1 输出 7 类，R2 类别清单锁定后将同步更新 `trt_pipeline.py`、`trt_pipeline.hpp`、`demo.cpp` 中的常量。

### Python 部署（开发 / 快速验证）

```bash
# 在 Orin 上（项目根目录执行）
python -m inference.demo --source video.mp4 --model weights/best.engine
python -m inference.demo --source 0 --model weights/best.engine --json  # 摄像头 + JSON
```

### C++ 部署（生产）

见下文 §二–§六。

---

## 二、源码同步到 Orin

```bash
rsync -avz --progress \
    --exclude 'build/' --exclude '*.o' --exclude '.DS_Store' \
    ~/Documents/Projects/mingtai/traffic-light/inference/cpp \
    nvidia@192.168.30.138:traffic-light/inference/
```

---

## 三、Orin 环境

JetPack 5.1.x / L4T R35.x 为出厂预刷版本 — 不使用 Docker，不重刷，不升级 JetPack。依赖矩阵：

| 组件 | 版本 | 位置 |
|---|---|---|
| CUDA | 11.4 | `/usr/local/cuda` |
| TensorRT | 8.5.2 | 头文件：`/usr/include/aarch64-linux-gnu/NvInfer.h`；库：`/usr/lib/aarch64-linux-gnu/` |
| OpenCV | 4.5.4 | 系统预装（`core`、`imgproc`、`highgui`、`videoio`） |
| CMake | 3.16.3（出厂）→ **需 ≥ 3.18** | — |

设备端只升级 CMake（项目 `CMakeLists.txt` 使用 `find_package(CUDAToolkit)`，需 3.18+）。其余保持原版 — **车队版本一致性优于 TensorRT 升级带来的边际收益**，因此通过裁剪 ONNX 绕开 TRT 8.5.2 ONNX parser 的限制（见 §五裁头步骤）。

`nvidia-jetpack` meta-package 无需安装；CUDA、cuDNN、TRT 库以及 OpenCV 均随出厂刷机预装。

### 升级 CMake（3.16 → ≥ 3.18）

任选其一：

```bash
# 方案 A（用户态，最简单）
pip3 install --user cmake
export PATH=$HOME/.local/bin:$PATH

# 方案 B（系统级，使用 Kitware apt 源）
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | \
    gpg --dearmor | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main" | \
    sudo tee /etc/apt/sources.list.d/kitware.list
sudo apt update && sudo apt install -y cmake
```

---

## 四、构建 C++ demo

```bash
cd ~/traffic-light/inference/cpp
export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
# 产物：build/tl_demo
```

---

## 五、在 Orin 上构建 TensorRT 引擎

TensorRT 引擎 **不跨 GPU 架构可迁移** —— `.engine` 文件必须在目标设备上生成。YOLO26 的 ONNX 还必须先裁掉内嵌的 NMS-free head，TRT 8.5.2 才能解析。

### 为什么需要裁头

TRT 8.5.2 的 ONNX parser 在 `ReduceMax → TopK` 上存在形状传播 bug，对合法 axis 也会抛出：

```
ERROR: onnx2trt_utils.cpp:345 In function convertAxis:
[8] Assertion failed: (axis >= 0 && axis <= nbDims)
```

YOLO26 的 head 恰好产生这种模式。图内的 `TopK` 是一个 GPU 侧预筛选优化 —— YOLO26 由训练端 1:1 anchor 匹配保证 NMS-free，推理端不需要 NMS。把 head 裁到最后一个 `Concat([box_xyxy, sigmoid(class_scores)], dim=1)`，暴露出原始 `[1, 4+nc, N]` 输出；C++ 后处理（`inference/cpp/src/trt_pipeline.cpp`）已按此格式走 CPU 侧置信度阈值。

> **Box 格式提示**（2026-04-21 修复）：`Concat_3` 输出的 4 个 box 通道是 letterbox 像素坐标下的 `(x1, y1, x2, y2)`，**不是** `(cx, cy, w, h)`。Python 与 C++ 后处理均按 xyxy 解码。早期误解为 cxcywh 时会产生过大框 —— 调试过程见 [`../reports/phase_2_round_1_report.md`](../reports/phase_2_round_1_report.md)。

### 导出 → 裁头 → 构建

1. 在 Mac 上将 `.pt` 导出为 `.onnx`（YOLO26 ONNX 自动裁头，仅保留 `best_stripped_<imgsz>.onnx`，中间 `best.onnx` 裁完即删）：
   ```bash
   uv run python main.py export runs/detect/yolo26n-r1/weights/best.pt --format onnx --imgsz 1280
   ```
   输出示例：`cut at /model.23/Concat_3: output [1, 11, 33600]`；产物为 `best_stripped_1280.onnx`。`--imgsz` 决定烘焙到引擎的输入分辨率，不同分辨率产出不同文件。如需保留未裁头的 `best.onnx`，加 `--no-strip`；此时可手动调用 `scripts/strip_yolo26_head.py`。
2. 同步到 Orin：
   ```bash
   rsync -avz runs/detect/yolo26n-r1/weights/best_stripped_1280.onnx \
       nvidia@192.168.30.138:traffic-light/runs/yolo26n-r1/
   ```
3. 在 Orin 上构建引擎：
   ```bash
   /usr/src/tensorrt/bin/trtexec \
       --onnx=runs/yolo26n-r1/best_stripped_1280.onnx \
       --saveEngine=runs/yolo26n-r1/best_1280.engine \
       --fp16
   ```

> 高分辨率引擎（1280 / 1536）需先按目标 `imgsz` 重新导出 ONNX（`python main.py export ... --imgsz 1280 --format onnx`）—— 引擎输入尺寸在导出时烘焙。见 §七。

---

## 六、运行

```bash
./build/tl_demo --source /path/to/video.mp4 --model weights/best.engine
./build/tl_demo --source 0 --model weights/best.engine --no-show --save out.mp4

# 启用跟踪（flicker mitigation；细节见 ./tracker.md）
./build/tl_demo --source video.mp4 --model best.engine \
    --track --alpha 0.3 --min-hits 3 --track-json runs/out.jsonl
```

Orin 无显示器时（如纯 SSH 会话）必须加 `--no-show` —— 否则 OpenCV 的 GTK HighGUI 后端会 abort。

### 批量扫描（runs × demos）

`scripts/run_demos.sh` 在所有 `runs/<run>/*.engine × demo/demo*.mp4` 组合上顺序调用 `tl_demo`，输出到 `demo/<run>/<engine_stem>/<demo_name>.mp4`。引擎文件名含 `1280` / `1536` 时自动推断对应 `--imgsz`。**串行设计**：TRT context 不可并发共享，且便于对齐延迟测量。

```bash
./scripts/run_demos.sh                               # 默认：CONF=0.25, SKIP_EXIST=1
CONF=0.3 TRACK=1 ./scripts/run_demos.sh              # 环境变量覆盖
nohup ./scripts/run_demos.sh \
    > logs/run_demos.log 2>&1 &                      # SSH-safe 后台长跑
```

---

## 七、输入分辨率

流水线接受任意分辨率与宽高比的视频。每帧先做 letterbox（保持宽高比的 resize + 灰色填充），再适配到引擎固定的 `imgsz × imgsz`（默认 1280 × 1280）。手机竖拍、4K 素材、竖屏、小于 `imgsz` 的输入均透明处理。

- **引擎输入尺寸在导出时烘焙**。demo 的 `--imgsz` 只作健全性校验 —— 与引擎不一致时以引擎为准，启动时打印 `[TRT] warning`。要换尺寸必须重新导出：`python main.py export ... --imgsz <N> --format onnx`，再重建 `.engine`。
- **输入小于 `imgsz`**（如 720p 进 1280 引擎）：letterbox 双线性上采样，不会失败，但细节无法凭空生成 —— 源里模糊的小 / 远距离信号灯依旧模糊。
- **输入大于 `imgsz`**（如 4K 或竖屏手机视频进 1280 引擎）：letterbox 下采样适配，很小 / 很远的灯可能漏检。要么先把源裁成横屏，要么导出更大 `imgsz` 的引擎（1600、1920），代价是每帧延迟上升。
- **Orin 实测延迟**（yolo26s-r1，FP16）：`imgsz=1280` ≈ 25 ms（~39 FPS）、`imgsz=1536` ≈ 28 ms（~34 FPS）。总预算 50 ms/帧 — R1 部署数据详见 [`../reports/phase_2_round_1_report.md`](../reports/phase_2_round_1_report.md)。

---

## 八、部署状态（R1）

- [x] Python 流水线 + demo 实现并在 M4 Pro 上验证
- [x] C++ 流水线 + demo 实现，语义与 Python 镜像
- [x] 源码已同步至 Orin；CUDA/TRT/OpenCV 随出厂刷机预装
- [x] Orin 端 CMake 升级（3.16 → ≥ 3.18）
- [x] Orin 端首次 `cmake --build` 成功
- [x] YOLO26 ONNX 已裁掉内嵌 NMS head（`scripts/strip_yolo26_head.py`）
- [x] 从裁头 ONNX 用 `trtexec` 在 Orin 本机构建出 `.engine`
- [x] Orin 端 C++ demo 端到端运行
- [x] xyxy postprocess bug 已修复（2026-04-21）；1280 + 1536 引擎已验证
