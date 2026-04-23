# 跟踪 + EMA 投票集成指南（Plan A）

> **状态（2026-04-23）**：**主动开发中**。本组件与 R1 备选架构训练并行推进，在 R2 数据就绪前持续迭代；C++ 端为生产部署目标。背景与权衡见 [`../proposals/temporal_encoder_feasibility.md`](../proposals/temporal_encoder_feasibility.md)，延后策略已更新为"并行而非门控"。

本指南规定在现有逐帧 `TRTDetector` 之上叠加 ByteTrack 关联 + 每轨迹 EMA 类别投票的实现计划。**Python 与 C++ 语义对等**：Python 用于快速验证与离线指标，**C++ 为 Orin 生产部署**。

---

## 一、目标与非目标

- **目标**：
  - 在手机竖拍 replay demo 上将每条轨迹的类别跳变降低 ≥ 60%
  - 在检测输出上新增 `tracking_id` 字段
  - CPU 侧每帧额外开销 < 2 ms（Python）/ < 1 ms（C++ @ Orin）
  - Python 与 C++ 版本在同一视频上产生**对齐的 `tracking_id` 轨迹**（允许 id 编号漂移，但轨迹数量、存续帧区间、平滑类别输出逐帧一致）
- **非目标**：
  - 不重训模型（与 R1 权重/ R1 备选架构并行，不互相阻塞）
  - 不改动 ONNX / engine schema
  - R2 数据就绪前不冻结超参数 —— EMA α、`min_hits`、`track_buffer` 以实车 replay 结果为准持续调优

---

## 二、设计决策（前置锁定）

| 决策 | 选择 | 原因 |
|---|---|---|
| 跟踪器 | **ByteTrack**（Kalman + Hungarian + 低置信度二次关联） | 对齐 Apollo / BSTLD 参考实现；低置信度二次关联恰好对应闪烁场景；CPU 侧 < 1 ms |
| 关联特征 | **仅 IoU，类别无关** | 交通灯位置稳定；平滑类别翻转是方案目的 —— 类别不能参与关联门控 |
| 类别平滑 | **EMA on per-class softmax-like 向量**，`α = 0.3`，`min_hits ≥ 3` | 软融合优于硬投票；单帧误分类可回滚 |
| 框输出 | 逐帧原始框（不使用 Kalman 平滑框） | Kalman 框在起步/停止时滞后；静态 TL 的检测框本身已稳定 |
| 超参数起始值 | `track_thresh=0.25`, `high_thresh=0.5`, `match_thresh=0.8`, `track_buffer=30`, `min_hits=3` | 30 帧缓冲 ≈ 1 s @ 30 fps；`min_hits=3` 抑制一次性 FP；**R2 数据到位前以此为默认，不锁定** |
| R2 nc 迁移 | `num_classes` 由构造器传入（7 → 最大 14） | 记忆 `project_scope_expansion.md` 锁定 nc 范围 10–14 |
| Python / C++ 语义一致性 | `TrackSmoother` 接口同名同形；单测共享合成序列 fixtures（JSON） | 避免语义漂移；Orin 上线时 Python 离线指标即为基线 |

---

## 三、文件布局

```
inference/
├── trt_pipeline.py              (不变)
├── tracker/                     ← Python 侧
│   ├── __init__.py              → 导出 TrackSmoother, TrackedDetection
│   ├── byte_tracker.py          → 引入的 ByteTrack 核心（Kalman + Hungarian + STrack）
│   └── smoother.py              → TrackSmoother: 包装 tracker 与 EMA 类别缓冲
├── demo.py                      (+ --track, + JSON tracking_id, + 按轨迹着色)
└── cpp/                         ← C++ 侧（生产部署）
    ├── CMakeLists.txt           (+ tl_tracker 目标，+ 可选 nlohmann/json fixtures 依赖)
    ├── include/
    │   ├── trt_pipeline.hpp     (不变)
    │   └── tracker.hpp          ← TrackSmoother / TrackedDetection 对等接口
    ├── src/
    │   ├── trt_pipeline.cpp     (不变)
    │   ├── byte_tracker.cpp     ← 引入的 C++ ByteTrack 核心
    │   ├── tracker.cpp          ← TrackSmoother 实现
    │   └── demo.cpp             (+ --track / --alpha / --min-hits / --track-json)
    └── tests/
        └── test_tracker.cpp     ← 消费 Python 端导出的合成序列 fixtures
tests/
└── test_tracker.py              → 合成序列（pytest）
scripts/
└── measure_flicker.py           → 消费 --json 输出的指标脚本
third_party/
└── bytetrack/                   ← 上游 vendored 源码（Python + C++），记录 commit SHA
docs/integration/
└── tracker_voting_guide.md      (本文)
```

---

## 四、公共接口

### 4.1 Python

```python
# inference/tracker/smoother.py
from dataclasses import dataclass
from inference.trt_pipeline import Detection

@dataclass
class TrackedDetection(Detection):
    tracking_id: int
    age: int                     # 自首次出现起经过的帧数
    hits: int                    # 已确认的检测次数
    raw_class_id: int            # 平滑前逐帧 argmax
    class_probs: list[float]     # 平滑后的每类概率

class TrackSmoother:
    def __init__(
        self,
        num_classes: int = 7,
        alpha: float = 0.3,
        track_thresh: float = 0.25,
        high_thresh: float = 0.5,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        min_hits: int = 3,
    ): ...

    def update(self, detections: list[Detection], frame_idx: int) -> list[TrackedDetection]: ...
    def reset(self) -> None: ...  # 切换输入源或帧间隔 > 200 ms 时调用
```

### 4.2 C++（对等）

```cpp
// inference/cpp/include/tracker.hpp
#pragma once
#include "trt_pipeline.hpp"
#include <vector>

namespace tl {

struct TrackedDetection : Detection {
    int tracking_id;
    int age;
    int hits;
    int raw_class_id;
    std::vector<float> class_probs;  // size == num_classes
};

struct TrackerConfig {
    int   num_classes   = 7;
    float alpha         = 0.3f;
    float track_thresh  = 0.25f;
    float high_thresh   = 0.5f;
    float match_thresh  = 0.8f;
    int   track_buffer  = 30;
    int   min_hits      = 3;
};

class TrackSmoother {
public:
    explicit TrackSmoother(const TrackerConfig& cfg = {});
    ~TrackSmoother();

    TrackSmoother(const TrackSmoother&) = delete;
    TrackSmoother& operator=(const TrackSmoother&) = delete;

    std::vector<TrackedDetection> update(const std::vector<Detection>& detections,
                                         int frame_idx);
    void reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tl
```

`TRTDetector` 两端均保持不变。Demo 层组合：

```python
# Python
detector = TRTDetector(model_path=args.model, conf_thresh=args.conf, imgsz=args.imgsz)
tracker = TrackSmoother(num_classes=len(CLASS_NAMES)) if args.track else None
for i, frame in enumerate(frames):
    dets = detector.detect(frame)
    out = tracker.update(dets, i) if tracker else dets
```

```cpp
// C++
tl::TRTDetector detector(model_path, conf, imgsz);
std::optional<tl::TrackSmoother> tracker;
if (opts.track) tracker.emplace(tl::TrackerConfig{.num_classes = 7, .alpha = opts.alpha});
for (int i = 0; ; ++i) {
    auto dets = detector.detect(frame);
    auto out = tracker ? tracker->update(dets, i) : /* wrap as TrackedDetection without ids */;
}
```

---

## 五、实施计划（与训练并行）

两端并行推进。Python 先行 1–2 天，把 fixtures 与指标脚本跑通，C++ 紧随其后，共享同一份合成序列金标。

### 5.1 Python 侧（~3 天）

| 步骤 | 工作量 | 产出 |
|---|---|---|
| P1. 引入 ByteTrack Python 核心 | 1 天 | `third_party/bytetrack/python/`（MIT，记录 commit SHA）；`inference/tracker/byte_tracker.py` 精简版 |
| P2. `TrackSmoother` 实现 | 0.5 天 | `inference/tracker/smoother.py`（EMA 缓冲、`min_hits` 过滤、`reset()`） |
| P3. Demo 集成 | 0.5 天 | `--track` / `--alpha` / `--min-hits` CLI；JSON 增加 `tracking_id` / `raw_class_id` / `class_probs` |
| P4. 合成序列 fixtures + pytest | 0.5 天 | `tests/fixtures/tracker/*.json`（detector 输出 + 期望轨迹）；`tests/test_tracker.py` |
| P5. `measure_flicker.py` + 回放 | 0.5 天 | 在 `runs/diagnose/*/demo.mp4` 上开/关跟踪对比 |

### 5.2 C++ 侧（~4 天，可在 P3 完成后开始）

| 步骤 | 工作量 | 产出 |
|---|---|---|
| C1. 引入 C++ ByteTrack 核心 | 1 天 | `third_party/bytetrack/cpp/`（MIT，见 §六.1）；`inference/cpp/src/byte_tracker.cpp` 精简版；仅依赖 Eigen + OpenCV |
| C2. `TrackSmoother` 实现 | 1 天 | `tracker.hpp` / `tracker.cpp`；pImpl 隐藏 Kalman/Hungarian 细节；签名与 Python 对等 |
| C3. CMake 集成 | 0.5 天 | 新目标 `tl_tracker`（STATIC），`tl_demo` 链接它；Eigen 通过 `find_package(Eigen3 REQUIRED)`（JetPack 预装）；见 §六.3 |
| C4. Demo 集成 | 0.5 天 | `demo.cpp` 新增 `--track` / `--alpha` / `--min-hits` / `--track-json`；按 `tracking_id % N` 着色 |
| C5. Fixture 驱动的单测 | 1 天 | `inference/cpp/tests/test_tracker.cpp` 读取 Python 端导出的 `tests/fixtures/tracker/*.json`（用 nlohmann/json header-only），断言**与 Python 一致的平滑轨迹**；CMake 目标 `tl_tracker_test`，默认 `BUILD_TESTS=OFF` |

### 5.3 与训练的协调

- 不占 Orin GPU：整个跟踪链路 CPU 执行，训练在 4090 上继续跑 YOLOv13 / DEIM
- 不占 Mac GPU：Python 侧用 CPU 回放即可；ONNX Runtime CPU provider 足以驱动合成 / 短视频
- R2 数据就绪前持续迭代：超参数调优在"已标注视频"可用后按新数据重跑 `measure_flicker.py`

---

## 六、C++ 依赖与构建细节

### 6.1 ByteTrack C++ 上游

- 推荐源：`Vertical-Beach/ByteTrack-cpp`（MIT，纯 C++17，无 CUDA / 无 YOLOX 依赖）
- 裁剪：移除可视化、视频 IO、测试二进制；保留 `BYTETracker`、`STrack`、`KalmanFilter`、`lapjv`
- 记录：`third_party/bytetrack/README.md` 写明 upstream commit SHA、裁剪清单、本地 patch 列表

### 6.2 `TrackedDetection` 继承 `Detection` 的 ABI 注意

`Detection` 是 POD；`TrackedDetection` 加入 `std::vector<float>` 后不再可 `memcpy`。仅在 API 层返回；内部跟踪 bookkeeping 用独立的 `Track` 结构，最终 `update()` 退出前物化。

### 6.3 CMake 变更（追加到 `inference/cpp/CMakeLists.txt`）

```cmake
# ---- Eigen (JetPack 预装 /usr/include/eigen3) ----
find_package(Eigen3 REQUIRED NO_MODULE)

add_library(tl_tracker STATIC
    src/byte_tracker.cpp
    src/tracker.cpp
)
target_include_directories(tl_tracker PUBLIC include)
target_link_libraries(tl_tracker PUBLIC Eigen3::Eigen ${OpenCV_LIBS})
target_compile_options(tl_tracker PRIVATE -Wall -Wextra)

target_link_libraries(tl_demo PRIVATE tl_tracker)

option(BUILD_TESTS "Build tracker tests" OFF)
if(BUILD_TESTS)
    # nlohmann/json: 优先 find_package(nlohmann_json)，否则 FetchContent
    find_package(nlohmann_json QUIET)
    if(NOT nlohmann_json_FOUND)
        include(FetchContent)
        FetchContent_Declare(nlohmann_json
            URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
        FetchContent_MakeAvailable(nlohmann_json)
    endif()
    add_executable(tl_tracker_test tests/test_tracker.cpp)
    target_link_libraries(tl_tracker_test PRIVATE tl_tracker nlohmann_json::nlohmann_json)
endif()
```

### 6.4 Orin 验证

- 构建：`cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=ON && cmake --build build -j$(nproc)`
- 运行：`./build/tl_demo --source video.mp4 --model best.engine --track --no-show --save out_tracked.mp4`
- 延迟预算断言：`--track` 开启时每帧 `TrackSmoother::update` < 1 ms（用 `std::chrono::steady_clock` 打点，在 20 条并发轨迹下验证）

---

## 七、测试策略

### 7.1 合成 fixtures（Python 作为金标产生方）

`tests/fixtures/tracker/*.json` 由 Python 端脚本生成，每个 fixture 含：

```jsonc
{
  "name": "static_with_flip",
  "num_classes": 7,
  "frames": [
    { "frame_idx": 0, "detections": [ {"class_id": 0, "confidence": 0.9, "x1": 100, "y1": 100, "x2": 120, "y2": 150} ] },
    // ... 注入 1 帧翻转 class_id=3
  ],
  "expected": {
    "unique_tracks": 1,
    "final_class_id": 0,       // 平滑后仍为 red
    "min_hits_confirmed": 3
  }
}
```

Python 与 C++ 两端均读取同一 JSON，生成相同断言。语义漂移会立刻被看见。

### 7.2 场景覆盖

1. 静态框 + 1 帧类别翻转 → 平滑类别不变
2. 10 帧序列中间 5 帧无检测 → 轨迹存活，`tracking_id` 保持
3. 双框，一静一动 → id 不交换
4. 单次 FP（只出现 1 帧） → 因 `hits < min_hits` 被过滤
5. 真 green → yellow → red 快速过渡 → 平滑延迟 < 10 帧（@ α=0.3）

### 7.3 回放基准

- 输入：`runs/diagnose/{n,s}-pt-{640,1280,1536}/demo.mp4`
- 指标：`scripts/measure_flicker.py` 输出 JSON，两端均跑一次，交叉比对 `class_flips_per_track`

---

## 八、决策门（R2 数据就绪后再锁定超参）

完成 §5.1 + §5.2 后，在 replay 视频上比较开/关跟踪的 `count_class_flips_per_track`。合入主干条件：

- 闪烁降低 ≥ 50%
- 无新增漏检（`hits ≥ min_hits` 过滤未杀死慢增长轨迹）
- Python 每帧额外开销 < 2 ms；C++ < 1 ms
- Python 与 C++ 在共享 fixtures 上逐帧平滑类别一致

**R2 数据就绪后**：用真实连续视频重跑 §5.1 P5 与 §5.2 延迟验证，**按结果重新调优 α / `min_hits` / `track_buffer`**，再定版写入默认值。

不满足则按可行性文档 Plan B（per-track GRU）排入 R3；`TrackedDetection.class_probs` 即为 GRU 的 soft 输入接口。

---

## 九、风险与缓解

| 风险 | 缓解 |
|---|---|
| ByteTrack Kalman 跨批次有状态 | 两端都暴露 `reset()`；切换视频源 / 长间隔时调用 |
| 长间隔后 `tracking_id` 冲突 | `track_buffer=30` 移除僵尸轨迹；id 严格单增 |
| 真实 green→yellow→red 过渡被 EMA 拖慢 | `α=0.3` 对应 ~8 帧达 95%（@ 30 fps ≈ 0.27 s），快于合法黄灯；R2 实车数据上再调 |
| Python ↔ C++ Kalman 实现差异导致轨迹分裂 | fixtures 双端比对；差异 > 0 立刻根因定位（浮点 / lap assignment tie-break） |
| Eigen / nlohmann/json 在 JetPack 缺失 | Eigen 预装；nlohmann/json 用 FetchContent 兜底（仅测试时） |
| `TrackedDetection` 不再 POD，打破旧下游 | Demo 层转换为 `Detection` 再绘制；`to_ros_msg()` 填 `tracking_id` |
| R2 nc 变化打破缓冲区形状 | `num_classes` 构造器参数；形状不匹配时抛错而非静默 |
| ByteTrack 上游许可证 | Python / C++ 均为 MIT；`THIRD_PARTY_LICENSES.md` 各登记一条 |

---

## 十、ROS2 集成（R2 后续）

`vision_msgs/Detection2D` 原生含 `tracking_id: string` 字段。启用跟踪时：

- Python：`TrackedDetection.to_ros_msg()` 填 `det.tracking_id = str(self.tracking_id)`
- C++：在 Orin ROS2 节点的 publisher 层完成同样转换（本仓库暂不落 ROS2 节点）
- Topic 与 QoS 不变（见 [`ros2_integration_guide.md`](./ros2_integration_guide.md)）

---

## 参考

- 可行性分析：[`../proposals/temporal_encoder_feasibility.md`](../proposals/temporal_encoder_feasibility.md)
- TRT 流水线：[`./trt_pipeline_guide.md`](./trt_pipeline_guide.md)
- ROS2 集成：[`./ros2_integration_guide.md`](./ros2_integration_guide.md)
- ByteTrack, ECCV 2022: https://arxiv.org/abs/2110.06864
- ByteTrack-cpp（推荐上游）：https://github.com/Vertical-Beach/ByteTrack-cpp
