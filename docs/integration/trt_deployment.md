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
| 使用场景 | 开发机快速迭代 / JSON 遥测；**TensorRT ≥ 10**（或走 ONNX-Runtime fallback） | Orin 上生产部署；**TensorRT 8.5+**（覆盖出厂 JetPack 5.1） |

两者接受相同参数：`--source`、`--model`、`--conf`、`--imgsz`、`--no-show`、`--save`。类别名与绘制颜色在两种后端中保持同步；R1 输出 7 类，R2 类别清单锁定后将同步更新 `trt_pipeline.py`、`trt_pipeline.hpp`、`demo.cpp` 中的常量。

> **Python ↔ C++ 分工（强制）**：Python 侧 `_TRTBackend` 在构造期硬性拒绝 TensorRT < 10 的运行时，并在异常文本里指向 C++ 流水线。Jetson AGX Orin / JetPack 5.1（TRT 8.5）部署**必须**走 `inference/cpp/`；Mac M4 Pro 等开发机若装有 TRT 10+ 可直接用 Python，否则用 ONNX-Runtime（`uv sync --extra onnx`）。两端在 letterbox 比例（Python 显式 `np.float32(...)` 对齐 C++ `static_cast<float>`）与四处契约校验点（动态 shape 选择、image-input 校验、输出 dtype 闸、输出 shape 闸）上逐字段一致，`--track-json` 跨后端比对漂移控制在 ≤1 px。

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

## 五·五、流水线构造契约（构造期硬校验，2026-05 加固）

`inference/trt_pipeline.py` 与 `inference/cpp/src/trt_pipeline.cpp` 的 `_TRTBackend` / `_ONNXBackend` / `tl::TRTDetector` 在加载引擎时执行下列校验，任一失败即抛错并附明文错误信息（不再静默回退、不再让错误格式漏到推理路径）：

### 5.5.1 输入张量契约

1. **NCHW + batch=1 + C=3 + H==W**：图像输入张量必须是 NCHW 4D、batch=1、3 通道、方形（`H == W`），dtype 为 FP32 或 FP16。任何条目不满足直接拒绝。
2. **`imgsz` 自动 snap 到引擎烘焙的 H**：demo CLI 传入的 `--imgsz` 若与引擎烘焙的方形边长不一致，流水线会**覆盖为引擎实际 H** 并向 stderr 打 warning。再不会出现 R1 时期"请求 1280 但引擎是 640，静默回退"的失败模式（参见 [`../reports/phase_2_round_1_report.md`](../reports/phase_2_round_1_report.md) §"Demo 视频诊断"）。
3. **多输入引擎按张量名路由**：blob 喂给的是经过校验的 image-input 张量（C++ 通过 `findImageInput()` 定位，Python 通过 `_image_name`），不再硬编码 `inputs_[0]` 或 `"images"`。DEIM 这类把 `orig_target_sizes` 列在前面的多输入引擎可以正确工作。

### 5.5.2 YOLO 架构输出契约（YOLO26 / YOLO11 / YOLOv13 路径）

恰好 1 个输出张量、dtype FP32 或 FP16、rank ∈ {2, 3}；rank-3 时 batch 必须为 1；shape 中**至少一条轴**等于 `4 + len(CLASS_NAMES)`（项目 7 类即 11）。`(4+nc, 4+nc)` 方阵被显式拒绝（方向歧义 — 无法判定 N 轴在哪一边）；其他情形里两轴中只有一条等于 11、另一条即为 N。

### 5.5.3 DEIM 架构输出契约

- `orig_target_sizes` 输入必须是 int32 或 int64，shape 必须为 `(1, 2)`（C++ `fillOrigTargetSizes()` 会写 `host[0]` / `host[1]`，元素数 < 2 会越界）；`labels` 输出必须是 int32 或 int64；`boxes` / `scores` 输出必须是 FP32 或 FP16。
- 输出形状契约（构造期硬校验，按 axis 逐项检查 — 仅校验 elem_count 比例无法识破 batch=3 或 rank-1 扁平化等异常导出）：
  - `labels`：rank=2，shape `(1, K)`
  - `scores`：rank=2，shape `(1, K)`
  - `boxes`：rank=3，shape `(1, K, 4)`（最后一轴必须为 4）
  - 三者的 K 必须一致
  - ONNX 端遇到动态轴（字符串形式的 K）跳过 K 一致性检查，由运行期索引兜底；其余 rank / 具体轴值仍硬校验
- Python 端 `_ONNXBackend.infer()` 自动把调用方传入的 int64 数组下转为模型期望的 int32（或反之），int32 `orig_target_sizes` 导出无需手动 cast。

### 5.5.4 异常安全 / 析构顺序

- **Python `_TRTBackend.__init__` 是 leak-safe**：构造途中任何步骤抛异常，`_close()` 会扫一遍已分配的 buffer 与 TRT 对象逐个释放，不会留下半状态。
- **C++ `freeAll()` 顺序**：先释放 TRT context / engine / runtime，再销毁 CUDA stream（TRT 8.x 析构里仍可能触碰 stream）。
- **C++ `TensorBuf::is_fp32`**：4-byte 张量进一步区分 `kFLOAT` 与 INT32 等其他 4-byte 类型，后处理不会把 INT32 字节当 float 坐标重解释。

### 5.5.5 常见构造错误对照

| 错误信息（实际抛出的字符串子串） | 原因 | 处理 |
|---|---|---|
| `engine has no NCHW 3-channel image input` | 没有名为 `images` 的 4-D/C=3 张量、也没有任何 4-D/C=3 fallback | 重导 ONNX 时确认输入是 `(1, 3, H, W)` |
| `engine image input has batch=N` | 输入 batch ≠ 1 | 用静态 batch=1 重导 |
| `engine image input is rectangular (WxH)` | H ≠ W | 用方形 `--imgsz` 重导 |
| `engine image input has unsupported dtype` (Python 末尾追加 `; only float32 or float16 are supported`，C++ 追加 `(only FP32 or FP16 are supported)`) | 图像输入是 INT32/INT8/UINT8 等非浮点类型 | 检查导出 / `trtexec` 的 dtype 设定 |
| `YOLO arch dispatched but engine has N outputs (expected 1)` | YOLO 路径下喂了多输出引擎；或 ONNX 没裁 NMS-free head | YOLO26：跑 head-strip（`scripts/strip_yolo26_head.py` 或 `main.py export`）；DEIM：走 DEIM 路径（张量名含 `labels` / `boxes` / `scores`） |
| `YOLO output shape (1, 11, 11) is orientation-ambiguous` | 引擎 `imgsz` 让 N 轴恰好等于 `4+nc`（罕见碰巧）| 改 `--imgsz` 或微调类别数后重建 |
| `YOLO ... has no 11-wide axis` | 类别数 / 头结构与项目 7-class 契约不符 | 确认 `data/traffic_light.yaml` 与训练 nc 一致 |
| `TensorRT X.Y is too old for the Python pipeline (requires >= 10.0). For Jetson / TRT 8.x deployment, use the C++ pipeline at inference/cpp/.` | 在 TRT 8.5 上跑 Python 流水线 | Orin 走 C++ 流水线；开发机 fallback 走 ONNX-Runtime |
| `DEIM 'orig_target_sizes' has unsupported dtype ... (expected int64 or int32)` / `DEIM 'labels' has unsupported dtype ...` / `DEIM 'boxes' has unsupported dtype ...` | DEIM 引擎 / ONNX 与标准导出脚本的 dtype 约定不一致 | 用 `scripts/export_deim.sh` 重新导出，或检查 `model.deploy()` 是否被改 |
| `DEIM 'orig_target_sizes' has shape ...; expected (1, 2)` | DEIM 引擎 / ONNX 的 `orig_target_sizes` 不是 `(1, 2)`（如被改成 `(2,)` 扁平、`(B, 2)` batch≠1 等）| 重新按标准 deploy 脚本导出；C++ 的 `fillOrigTargetSizes()` 假定 elem_count=2，否则会越界 |
| `DEIM 'labels' has shape ...; expected (1, K)` / `DEIM 'boxes' has shape ...; expected (1, K, 4)` / `DEIM K mismatch` | DEIM 输出 rank、batch 或最后一轴异常（如 batch=3、boxes 转置成 `(1, 4, K)`、labels/scores/boxes 的 K 不一致）| 检查 `scripts/_export_deim_onnx.py` deploy 阶段的 reshape 是否被改；按标准导出脚本重建 |

> **参考实现**：`inference/trt_pipeline.py`（Python 三类后端：TRT / ONNX / 自动选择）；`inference/cpp/src/trt_pipeline.cpp` + `inference/cpp/include/trt_pipeline.hpp`（C++ 生产路径）。校验点的具体位置由 `findImageInput()`（C++）与 `_image_name` / Pass-1 dynamic-shape 选择器（Python）承担。

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

- **引擎输入尺寸在导出时烘焙**。demo 的 `--imgsz` 只作健全性校验 —— 流水线在构造期 auto-snap 到引擎烘焙的方形边长 H 并向 stderr warn（详见 §五·五.1）。要换尺寸必须重新导出：`python main.py export ... --imgsz <N> --format onnx`，再重建 `.engine`。
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
- [x] 流水线构造期硬校验（NCHW + 方形 + dtype + 输出形状 + 多输入路由）落地（2026-05），见 §五·五；R1 "imgsz 静默回退" 失败模式已被 imgsz auto-snap + warning 取代
