# 跟踪 + EMA 投票集成指南（Plan A）

> **状态（2026-04-23）**：**Python 与 C++ 两端已落地并通过终审**。R1 备选架构训练并行推进；两端共享同一份 JSON fixtures 与语义断言，余下为 Orin 构建 + fixture 单测上线。本指南覆盖 Plan A（tracker + EMA）；后续可选时序优化（TSM detector-level / GRU post-detector 等）见 [`../planning/temporal_optimization_plan.md`](../planning/temporal_optimization_plan.md)。

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
│   ├── basetrack.py             → BaseTrack / TrackState（vendored from ByteTrack MIT）
│   ├── byte_tracker.py          → BYTETracker + STrack（vendored，已做本地裁剪）
│   ├── kalman_filter.py         → 8-dim 常速度 Kalman（vendored）
│   ├── matching.py              → IoU 距离 + Hungarian（scipy.linear_sum_assignment）
│   └── smoother.py              → TrackSmoother: 包装 tracker 与 EMA 类别缓冲
├── demo.py                      (+ --track / --alpha / --min-hits / --track-json)
└── cpp/                         ← C++ 侧（生产部署）
    ├── CMakeLists.txt           (+ tl_tracker STATIC 目标，+ TL_BUILD_TRACKER_ONLY 选项)
    ├── include/
    │   ├── detection.hpp        ← 与 TRT 流水线共享的 POD Detection（无 TRT 依赖）
    │   ├── trt_pipeline.hpp     (不变)
    │   └── tracker.hpp          ← TrackSmoother / TrackedDetection / TrackerConfig
    ├── src/
    │   ├── trt_pipeline.cpp     (不变)
    │   ├── tracker.cpp          ← 全部 tracker 逻辑（Kalman + Hungarian + STrack + EMA）
    │   └── demo.cpp             (+ --track / --alpha / --min-hits / --high-thresh /
    │                               --match-thresh / --track-buffer / --track-json)
    └── tests/
        └── test_tracker.cpp     ← 消费 tests/fixtures/tracker/*.json（nlohmann/json FetchContent）
tests/
├── fixtures/tracker/*.json      → 合成序列金标（Python 与 C++ 共享）
└── test_tracker.py              → 合成序列 + 单元测试（pytest）
docs/integration/
└── tracker_voting_guide.md      (本文)
```

> **说明**：C++ 端未单独拆出 `byte_tracker.cpp` —— Kalman / Hungarian / STrack / BYTETracker 全部藏在 `tracker.cpp` 的匿名命名空间内，只通过 `tl::TrackSmoother` 暴露。依赖仅 OpenCV（`cv::Mat` 用作 Kalman 线性代数）+ STL；没有 Eigen，没有 lapjv，没有 cython_bbox。Mac 开发机可用 `-DTL_BUILD_TRACKER_ONLY=ON` 仅构建 tracker 库 + 单测，跳过 CUDA/TensorRT。

---

## 四、公共接口

### 4.1 Python

```python
# inference/tracker/smoother.py（已落地）
from dataclasses import dataclass, field
from inference.trt_pipeline import Detection

@dataclass
class TrackedDetection(Detection):
    tracking_id: int = -1
    age: int = 0                 # 自首次出现起经过的帧数
    hits: int = 0                # 已确认的检测次数
    raw_class_id: int = -1       # 平滑前逐帧 argmax
    raw_confidence: float = 0.0
    class_probs: list[float] = field(default_factory=list)

class TrackSmoother:
    def __init__(
        self,
        num_classes: int = 7,
        alpha: float = 0.3,
        track_thresh: float = 0.25,   # 外层低截断：低于此直接丢弃
        high_thresh: float = 0.5,     # 喂给 BYTETracker 的高/低分界线
        match_thresh: float = 0.8,    # 第一轮 IoU 代价上限
        track_buffer: int = 30,
        min_hits: int = 3,
        frame_rate: int = 30,
    ):
        # 构造器校验：alpha ∈ (0, 1]；num_classes > 0；high_thresh >= track_thresh
        ...

    def update(self, detections: list[Detection], frame_idx: int) -> list[TrackedDetection]: ...
    def reset(self) -> None: ...  # 切换输入源或帧间隔 > track_buffer 时调用
```

### 4.2 C++（对等）

```cpp
// inference/cpp/include/tracker.hpp（已落地，下面为精简展示）
#pragma once
#include "detection.hpp"   // 与 TRT 流水线共享的 POD Detection
#include <memory>
#include <vector>

namespace tl {

struct TrackedDetection : Detection {
    int tracking_id     = -1;
    int age             = 0;    // 自首次出现起经过的帧数
    int hits            = 0;    // 已确认的检测次数
    int raw_class_id    = -1;   // 平滑前逐帧 argmax
    float raw_confidence = 0.f;
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
    int   frame_rate    = 30;   // 仅用于缩放 track_buffer 到时间
};

class TrackSmoother {
public:
    explicit TrackSmoother(const TrackerConfig& cfg = {});
    ~TrackSmoother();

    TrackSmoother(const TrackSmoother&) = delete;
    TrackSmoother& operator=(const TrackSmoother&) = delete;
    TrackSmoother(TrackSmoother&&) noexcept;
    TrackSmoother& operator=(TrackSmoother&&) noexcept;

    std::vector<TrackedDetection> update(const std::vector<Detection>& detections,
                                         int frame_idx);
    void reset();  // 切换输入源或帧间隔 > track_buffer 时调用
    const TrackerConfig& config() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace tl
```

> **线程安全**：`TrackSmoother` 不是线程安全的 —— 它持有所有活跃轨迹的 Kalman 状态。每相机一个实例，同线程内调用 `update()`。
>
> **多相机 ID 命名空间**：track ID 来自进程内全局计数器（与 Python `BaseTrack._count` 语义一致）。多相机部署要么每相机一个进程，要么在 publisher 层为 ID 加前缀。

`TRTDetector` 两端均保持不变。Demo 层组合：

```python
# Python — inference/demo.py
detector = TRTDetector(model_path=args.model, conf_thresh=args.conf, imgsz=args.imgsz)
tracker = TrackSmoother(num_classes=len(CLASS_NAMES)) if args.track else None
for i, frame in enumerate(frames):
    dets = detector.detect(frame)
    out = tracker.update(dets, i) if tracker else dets
```

```cpp
// C++ — inference/cpp/src/demo.cpp
tl::TrackerConfig tcfg;
tcfg.num_classes   = 7;
tcfg.alpha         = opts.alpha;
tcfg.min_hits      = opts.min_hits;
tcfg.high_thresh   = opts.high_thresh;
tcfg.match_thresh  = opts.match_thresh;
tcfg.track_buffer  = opts.track_buffer;
tcfg.frame_rate    = int(cap.get(cv::CAP_PROP_FPS) /*fallback 30*/);
std::optional<tl::TrackSmoother> tracker;
if (opts.track) tracker.emplace(tcfg);
for (int i = 0; /* loop */ ; ++i) {
    auto dets = detector.detect(frame);
    auto out = tracker ? tracker->update(dets, i) : wrapUntracked(dets);
}
```

---

## 五、实施计划（与训练并行）

两端并行推进。Python 先行 1–2 天，把 fixtures 与指标脚本跑通，C++ 紧随其后，共享同一份合成序列金标。

### 5.1 Python 侧 — ✅ 已完成

| 步骤 | 状态 | 产出 |
|---|---|---|
| P1. 引入 ByteTrack Python 核心 | ✅ | `inference/tracker/{basetrack,byte_tracker,matching,kalman_filter}.py`（MIT，upstream `ifzhang/ByteTrack@d1bf019`；本地裁剪见文件 docstring） |
| P2. `TrackSmoother` 实现 | ✅ | `inference/tracker/smoother.py`：EMA 缓冲、`min_hits` 过滤、构造器入参校验、`reset()`、回声腔 fallback |
| P3. Demo 集成 | ✅ | `inference/demo.py` 新增 `--track` / `--alpha` / `--min-hits` / `--track-json` CLI |
| P4. 合成序列 fixtures + pytest | ✅ | `tests/fixtures/tracker/{fast_transition,gap_survival,single_fp_suppression,static_flip,two_box_stability}.json` + `tests/test_tracker.py` |
| P5. `measure_flicker.py` + 回放 | ⏳ | 指标脚本已落地；在实车 replay 到位前指标数字仅为合成值 |

### 5.2 C++ 侧 — ✅ 已完成

| 步骤 | 状态 | 产出 |
|---|---|---|
| C1. 引入 C++ ByteTrack 核心 | ✅ | **自写而非 vendored**：直接在 `tracker.cpp` 匿名命名空间里以 `cv::Mat` 重写 Kalman、手写 O(n²·m) Jonker-Volgenant Hungarian（带 1e9 masking + 方阵填充），避开 Eigen / lapjv 依赖；见 §六.1 |
| C2. `TrackSmoother` 实现 | ✅ | `tracker.hpp` + `tracker.cpp`（约 815 行）；pImpl 隐藏 ByteTracker 内部；签名与 Python 对等，包括 `TrackerConfig` 校验、回声腔 fallback、状态 GC |
| C3. CMake 集成 | ✅ | `TL_BUILD_TRACKER_ONLY` 选项开关（默认 OFF）；开启时**跳过 `project(CUDA)` 与 TRT 流水线**，方便 Mac/Linux 开发机仅构建 tracker + fixture 单测 |
| C4. Demo 集成 | ✅ | `demo.cpp`：`--track`、`--alpha`、`--min-hits`、`--high-thresh`、`--match-thresh`、`--track-buffer`、`--track-json`；按 `tracking_id % 12` 色板着色，绘制 `#<id> <class> <conf>` |
| C5. Fixture 驱动的单测 | ✅ | `inference/cpp/tests/test_tracker.cpp` 通过 FetchContent 拉 nlohmann/json v3.11.3；默认 `TL_BUILD_TESTS=ON`；CTest 目标 `tracker_parity`。覆盖 `per_frame_track_count` / `unique_tracks` / `final_smoothed_class_id` / `smoothed_never_in` / `gap_survival` / `two_box_stability` 六类断言 + `reset()` 烟测 + 构造器校验烟测 |

### 5.2.1 终审修复（2026-04-23）

Python 与 C++ 两端在第二轮终审中发现并修复的**实际 bug**（非风格性差异）：

| 位置 | 问题 | 修复 |
|---|---|---|
| `inference/cpp/src/tracker.cpp::ByteTracker::update` | Lost 轨迹进入 predict 前未归零 `mean[7]`（高度速度），与 `STrack.multi_predict` 在 Python 侧行为不一致 —— gap 期间框高会漂移，再关联时 IoU 退化 | 对 pool 中状态非 `Tracked` 的轨迹显式 `mean.at<double>(7) = 0.0`，对齐 `byte_tracker.py:49–62` |
| `inference/cpp/src/tracker.cpp::linearAssignment` | 空行代价矩阵（pool 为空但有检测）会把 m 也当 0，使 `unmatched_b` 返回空集 —— 第一帧无新轨迹初始化 | 改造签名接受显式 `(n, m)`；三个调用点传入实际维度。根因：`vector<vector<double>>` 无法表示"0 行 × M 列"；scipy 的 `linear_sum_assignment` 通过 numpy `shape[1]` 保留信息，这里通过参数补回 |
| `inference/cpp/src/tracker.cpp` 分桶 | `s == track_thresh` 边界 C++ 落入 second，Python 会直接丢弃（`scores < track_thresh` 严格） | `else if (s > 0.1f && s < trackThresh_)` 两侧严格，匹配 Python 语义 |

现有 fixtures 全部为"静态 / 缓漂移框"，上述三处不会在现有金标里暴露；动态 + gap 场景一定会打爆。动态金标待 R2 实车 replay 就绪后补入。

### 5.3 与训练的协调

- 不占 Orin GPU：整个跟踪链路 CPU 执行，训练在 4090 上继续跑 YOLOv13 / DEIM
- 不占 Mac GPU：Python 侧用 CPU 回放即可；ONNX Runtime CPU provider 足以驱动合成 / 短视频
- R2 数据就绪前持续迭代：超参数调优在"已标注视频"可用后按新数据重跑 `measure_flicker.py`

---

## 六、C++ 依赖与构建细节

### 6.1 为什么没有引入 C++ ByteTrack 上游

原计划引入 `Vertical-Beach/ByteTrack-cpp`，评审后改为**自写精简实现**：

- 上游引入 Eigen + 第三方 lapjv（GPL 区分不一），本仓库已有 OpenCV（`cv::Mat` 可做 8×8 / 4×4 线性代数），不必多一个矩阵库
- Python 侧已用 `scipy.optimize.linear_sum_assignment`（1e9 masking），非 lapjv，上游 C++ 的 lapjv 反而会成为**语义漂移源**
- 跟踪器规模（约 400 行核心逻辑）自写的维护成本远低于长期 vendored 源的许可 / 裁剪 / patch 维护
- 所有数值常量（`track_buffer` 的 `frame_rate/30` 缩放、`det_thresh = track_thresh + 0.1`、Kalman `std_weight_position=1/20`、`std_weight_velocity=1/160`、第二轮 `thresh=0.5`、unconfirmed `thresh=0.7`、`remove_duplicate` 的 `iou<0.15`）在自写版内严格复刻；参考文件的 `inference/tracker/*.py` 均标有 upstream commit `ifzhang/ByteTrack@d1bf019`

第三方许可记录：Python 端的 vendored 源仍记在 `THIRD_PARTY_LICENSES.md`（MIT）；C++ 端为自写，无额外条目。

### 6.2 `TrackedDetection` 继承 `Detection` 的 ABI 注意

`Detection` 是 POD；`TrackedDetection` 加入 `std::vector<float> class_probs` 后不再可 `memcpy`。仅在 API 层返回；内部 bookkeeping（匿名命名空间里的 `TrackState_`）用独立结构持 `std::vector<double>`，`update()` 退出前下降到 `float` 物化到输出。

### 6.3 CMake 实际布局（`inference/cpp/CMakeLists.txt` 已上线）

```cmake
# 默认全量构建（tracker + TRT 流水线 + demo）
# 仅 tracker 开发：cmake -DTL_BUILD_TRACKER_ONLY=ON ..
option(TL_BUILD_TRACKER_ONLY "Build only the tracker library (skip TensorRT/CUDA)" OFF)

if(TL_BUILD_TRACKER_ONLY)
    project(traffic_light_trt CXX)
else()
    project(traffic_light_trt CXX CUDA)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)

add_library(tl_tracker STATIC src/tracker.cpp)
target_include_directories(tl_tracker PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tl_tracker PUBLIC ${OpenCV_LIBS})

option(TL_BUILD_TESTS "Build tracker parity tests" ON)
if(TL_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.3
        GIT_SHALLOW    TRUE)
    FetchContent_MakeAvailable(nlohmann_json)

    add_executable(tl_tracker_tests tests/test_tracker.cpp)
    target_link_libraries(tl_tracker_tests
        PRIVATE tl_tracker nlohmann_json::nlohmann_json)
    target_compile_definitions(tl_tracker_tests PRIVATE
        TL_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/fixtures/tracker")
    enable_testing()
    add_test(NAME tracker_parity COMMAND tl_tracker_tests)
endif()

if(TL_BUILD_TRACKER_ONLY)
    return()  # 跳过 CUDA / TensorRT / tl_demo
endif()

# TRT 流水线 + demo（省略，见实际文件）
```

三点变化相对最初规划：

1. `Eigen3::Eigen` **未使用** —— Kalman 用 `cv::Mat` + `cv::solve(DECOMP_CHOLESKY)`
2. 默认 `TL_BUILD_TESTS=ON`（原先 `BUILD_TESTS=OFF`）—— 单测足够轻量，默认开；Orin 全量构建自动拉 nlohmann/json
3. 新增 `TL_BUILD_TRACKER_ONLY` —— 解决 "Mac 没 CUDA / TensorRT 但想跑 tracker 测试" 的开发机工作流

### 6.4 验证流程

**Mac / Linux 开发机**（无 CUDA / TensorRT）：

```bash
cmake -S inference/cpp -B inference/cpp/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DTL_BUILD_TRACKER_ONLY=ON
cmake --build inference/cpp/build -j$(nproc)
ctest --test-dir inference/cpp/build --output-on-failure
```

**Orin 生产机**（CUDA + TensorRT）：

```bash
cmake -S inference/cpp -B inference/cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build inference/cpp/build -j$(nproc)
ctest --test-dir inference/cpp/build --output-on-failure   # parity 单测先过
./inference/cpp/build/tl_demo --source video.mp4 --model best.engine \
    --track --alpha 0.3 --min-hits 3 --track-json runs/out.jsonl
```

- 延迟预算断言：`--track` 开启时每帧 `TrackSmoother::update` < 1 ms（用 `std::chrono::steady_clock` 在 20 条并发轨迹下打点验证）
- Python↔C++ 逐帧对比：在同一视频上各跑一次 `--track-json`，用 `scripts/measure_flicker.py` 交叉比对 `class_flips_per_track`（允许 `tracking_id` 编号漂移，但数量、区间、平滑类别需一致）

---

## 七、测试策略

### 7.1 合成 fixtures（Python 与 C++ 共享的金标）

`tests/fixtures/tracker/*.json` 是 Python 与 C++ 两端**同时**消费的金标序列。每个文件结构：

```jsonc
{
  "name": "static_flip",
  "num_classes": 7,
  "config": {                      // 可选；未提供则走构造器默认
    "alpha": 0.3,
    "min_hits": 3,
    "high_thresh": 0.5
  },
  "frames": [
    // 每帧是一个 detection 数组
    [ {"class_id": 0, "confidence": 0.9, "x1": 100, "y1": 100, "x2": 120, "y2": 150} ],
    [ {"class_id": 3, "confidence": 0.9, "x1": 100, "y1": 100, "x2": 120, "y2": 150} ],
    // ...
  ],
  "expected": {
    "per_frame_track_count": [0, 0, 1, 1, 1, ...],  // 逐帧轨迹数（可选）
    "unique_tracks": 1,                              // 整段总轨迹数（可选）
    "final_smoothed_class_id": 0,                    // 末帧平滑类别（可选）
    "smoothed_never_in": [3, 5]                      // 平滑输出永不包含的 class_id（可选）
  }
}
```

两端都逐一读文件、运行 `TrackSmoother.update()`、对比 `expected.*`。Python 由 `pytest tests/test_tracker.py` 驱动；C++ 由 `ctest -R tracker_parity` 驱动，底层 `inference/cpp/tests/test_tracker.cpp` 用 nlohmann/json 解析 + 自写的 `TL_CHECK` 宏做断言（有意不引入 GoogleTest / Catch2，降低构建复杂度）。

### 7.2 已覆盖场景（当前 5 个 fixture）

| Fixture | 覆盖 |
|---|---|
| `static_flip.json` | 静态框 + 1 帧类别翻转 → 平滑类别不变 |
| `gap_survival.json` | 序列中段若干帧无检测 → 轨迹存活，`tracking_id` 保持（专项断言 `checkGapSurvival`） |
| `two_box_stability.json` | 双框，一静一动 → `tracking_id` 不交换（专项断言 `checkTwoBoxStability`） |
| `single_fp_suppression.json` | 单帧 FP → `hits < min_hits` 被过滤，不出现在输出 |
| `fast_transition.json` | 真 green → yellow → red 过渡 → 平滑类别无多余中间跳变 |

C++ 端额外烟测（不依赖 fixture）：`reset()` 后 ID 从 1 重开、`num_classes=0` / `alpha=0` / `alpha=1.1` / `high_thresh < track_thresh` 在构造器内全部抛出。

### 7.3 已知金标空白（R2 待补）

现有 fixture 全是**静态或缓漂移**框，不覆盖：

1. 动态（有速度）+ 多帧 gap 场景：会暴露 Lost 轨迹 Kalman 预测时的速度行为（2026-04-23 修复的 Bug A）
2. `score == track_thresh` 精确边界：会暴露分桶边界行为（2026-04-23 修复的 Bug B）

待 R2 实车 replay 到位，补动态 + gap fixture 与边界分值 fixture 各一。

### 7.4 回放基准

- 输入：`runs/diagnose/{n,s}-pt-{640,1280,1536}/demo.mp4`
- 指标：`scripts/measure_flicker.py` 输出 JSON，Python / C++ 两端各跑一次，交叉比对 `class_flips_per_track`（允许 `tracking_id` 编号漂移，但数量、区间、平滑类别需一致）

---

## 八、决策门（R2 数据就绪后再锁定超参）

完成 §5.1 + §5.2 后，在 replay 视频上比较开/关跟踪的 `count_class_flips_per_track`。合入主干条件：

- 闪烁降低 ≥ 50%
- 无新增漏检（`hits ≥ min_hits` 过滤未杀死慢增长轨迹）
- Python 每帧额外开销 < 2 ms；C++ < 1 ms
- Python 与 C++ 在共享 fixtures 上逐帧平滑类别一致

**R2 数据就绪后**：用真实连续视频重跑 §5.1 P5 与 §5.2 延迟验证，**按结果重新调优 α / `min_hits` / `track_buffer`**，再定版写入默认值。

不满足则按 [`../planning/temporal_optimization_plan.md`](../planning/temporal_optimization_plan.md) §0.2 决策门选择后续路径（detector-level TSM 或 post-detector HMM/GRU）；`TrackedDetection.class_probs` 即为下游平滑器的 soft 输入接口。

---

## 九、风险与缓解

| 风险 | 缓解 |
|---|---|
| ByteTrack Kalman 跨批次有状态 | 两端都暴露 `reset()`；切换视频源 / 长间隔时调用 |
| 长间隔后 `tracking_id` 冲突 | `track_buffer=30` 移除僵尸轨迹；id 严格单增 |
| 真实 green→yellow→red 过渡被 EMA 拖慢 | `α=0.3` 对应 ~8 帧达 95%（@ 30 fps ≈ 0.27 s），快于合法黄灯；R2 实车数据上再调 |
| Python ↔ C++ Kalman 实现差异导致轨迹分裂 | fixtures 双端比对；差异 > 0 立刻根因定位（浮点 / lap assignment tie-break） |
| nlohmann/json 在 JetPack 缺失 | 已切到 FetchContent 直接拉 tag `v3.11.3`，不走系统包路径；Eigen 整个需求已移除（Kalman 改用 `cv::Mat`） |
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

- 时序优化集成计划（后续路径）：[`../planning/temporal_optimization_plan.md`](../planning/temporal_optimization_plan.md)
- TRT 流水线：[`./trt_pipeline_guide.md`](./trt_pipeline_guide.md)
- ROS2 集成：[`./ros2_integration_guide.md`](./ros2_integration_guide.md)
- ByteTrack, ECCV 2022: https://arxiv.org/abs/2110.06864
- ByteTrack-cpp（推荐上游）：https://github.com/Vertical-Beach/ByteTrack-cpp
