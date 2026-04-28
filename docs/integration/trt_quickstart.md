# TensorRT × ROS2 集成速查

感知端发布 `vision_msgs/Detection2DArray`，下游节点订阅。完整部署细节见 [`trt_deployment.md`](./trt_deployment.md)；完整消息契约与 API 稳定性承诺见 [`ros2_contract.md`](./ros2_contract.md)。

---

## 一、交付物

| 文件 | 放哪里 |
|---|---|
| `best_<imgsz>.engine` | 感知节点可读路径；默认 `weights/best.engine` |
| `inference/cpp/` | `add_subdirectory()` 接入你的 ROS2 包 |

---

## 二、环境（Jetson AGX Orin，JetPack 5.1.x）

| 组件 | 状态 |
|---|---|
| CUDA 11.4 / TensorRT 8.5.2 / OpenCV 4.5.4 | ✅ 出厂预装 |
| CMake | ❌ 出厂 3.16，需 ≥ 3.18：`pip3 install --user cmake && export PATH=$HOME/.local/bin:$PATH` |
| ROS2 依赖 | `sudo apt install ros-${ROS_DISTRO}-vision-msgs ros-${ROS_DISTRO}-cv-bridge` |

---

## 三、Topic 契约

| 项目 | 值 |
|------|-----|
| Topic | `/traffic_light/detections`（默认） |
| 消息类型 | `vision_msgs/Detection2DArray` |
| 发布频率 | 与相机帧率一致 |
| QoS | `RELIABLE`, depth=10 |
| 坐标系 | bbox 为**原始图像像素**（非归一化） |

```
Detection2DArray
├── header (stamp = 相机原始时间戳, frame_id = 相机 frame_id)
└── detections[]
    └── Detection2D
        ├── bbox: center.position.x/y, size_x, size_y   # 像素
        ├── tracking_id                                  # string，启用 --track 时填充
        └── results[0].hypothesis
            ├── class_id    # string，如 "red"
            └── score       # 0.0–1.0
```

### 检测类别（R1 = 7 类）

`red` / `yellow` / `green` / `redLeft` / `greenLeft` / `redRight` / `greenRight`

R2 追加（消息结构不变，未识别 `class_id` 可忽略）：`forwardRed`、`forwardGreen`、`barrier`，另 PM 待定 ≤3 项。

### 稳定性承诺（5/15 前不变）

- 消息类型：`vision_msgs/Detection2DArray`
- Topic 名：`/traffic_light/detections`
- `class_id` 为**字符串**，非整数
- R1 已有 7 类名不会重命名；R2 仅追加新类
- `results` 数组固定 1 个 `ObjectHypothesisWithPose`

---

## 四、订阅端示例（C++）

```cpp
#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

void on_detections(const vision_msgs::msg::Detection2DArray::SharedPtr msg)
{
    for (const auto & det : msg->detections) {
        if (det.results.empty()) continue;

        const auto & hyp = det.results[0].hypothesis;
        std::string class_name = hyp.class_id;   // "red", "greenLeft"...
        double confidence = hyp.score;

        double cx = det.bbox.center.position.x;
        double cy = det.bbox.center.position.y;
        double w  = det.bbox.size_x;
        double h  = det.bbox.size_y;
        // tracking_id: det.tracking_id（启用跟踪时）
    }
}
```

**CMakeLists.txt**：

```cmake
find_package(rclcpp REQUIRED)
find_package(vision_msgs REQUIRED)
ament_target_dependencies(your_node rclcpp vision_msgs)
```

**package.xml**：

```xml
<depend>rclcpp</depend>
<depend>vision_msgs</depend>
```

### 防御性编码

1. 对未知 `class_id` 按警示处理（R2 会新增类）
2. 基于字符串匹配，不要假设整数编号
3. 访问 `results[0]` 前检查非空
4. 每帧可能 0 个或多个检测

---

## 五、常见问题

| 症状 | 处理 |
|---|---|
| `[TRT] warning: --imgsz mismatch` | 以引擎为准；换尺寸需重建引擎 |
| `libnvinfer.so.8: cannot open shared object file` | `export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH` |
| 引擎在另一台机器报错 | TRT 引擎不跨 GPU 架构 / TRT 版本，须在目标机重建 |
| 框位置明显错（全屏大框 / 偏移） | 预处理不一致；参考 `trt_pipeline.cpp` 的 preprocess；`bbox` 已还原到原始像素 |
| 引擎加载慢（>10 s） | 首次反序列化正常，进程常驻即可 |
| `tracking_id` 为空字符串 | 感知端未启用 `--track`；跟踪细节见 [`tracker.md`](./tracker.md) |
