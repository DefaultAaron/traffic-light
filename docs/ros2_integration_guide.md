# 红绿灯检测 — ROS2 集成指南

## Topic 定义

| 项目 | 值 |
|------|-----|
| Topic | `/traffic_light/detections`（默认） |
| 消息类型 | `vision_msgs/Detection2DArray` |
| 发布频率 | 与相机帧率一致 |
| QoS | `RELIABLE`, depth=10 |

---

## 消息结构

每帧发布一个 `Detection2DArray`，包含该帧所有检测到的红绿灯（0 个或多个）。

```
Detection2DArray
├── header
│   ├── stamp          ← 相机图像原始时间戳
│   └── frame_id       ← 相机 frame_id
└── detections[]
    └── Detection2D
        ├── bbox: BoundingBox2D
        │   ├── center.position.x  ← 边框中心 x（像素）
        │   ├── center.position.y  ← 边框中心 y（像素）
        │   ├── size_x             ← 边框宽度（像素）
        │   └── size_y             ← 边框高度（像素）
        └── results[0]
            └── ObjectHypothesisWithPose
                └── hypothesis
                    ├── class_id   ← 类别名称字符串（见下表）
                    └── score      ← 置信度 0.0–1.0
```

> **注意**：`hypothesis.class_id` 为**字符串**（如 `"red"`），不是整数。

---

## 7 类检测类别

| class_id (string) | 含义 |
|--------------------|------|
| `red` | 红色圆灯 |
| `yellow` | 黄色圆灯 |
| `green` | 绿色圆灯 |
| `redLeft` | 红色左转箭头 |
| `greenLeft` | 绿色左转箭头 |
| `redRight` | 红色右转箭头 |
| `greenRight` | 绿色右转箭头 |

---

## 字段访问参考

### C++

```cpp
for (const auto & det : msg->detections)
{
    if (det.results.empty()) continue;

    const auto & hyp = det.results[0].hypothesis;
    std::string class_name = hyp.class_id;   // "red"、"greenLeft" 等
    double confidence = hyp.score;

    double cx = det.bbox.center.position.x;
    double cy = det.bbox.center.position.y;
    double w  = det.bbox.size_x;
    double h  = det.bbox.size_y;
}
```

### Python

```python
for det in msg.detections:
    if not det.results:
        continue

    class_name = det.results[0].hypothesis.class_id   # "red"、"greenLeft" 等
    confidence = det.results[0].hypothesis.score

    cx = det.bbox.center.position.x
    cy = det.bbox.center.position.y
    w  = det.bbox.size_x
    h  = det.bbox.size_y
```

---

## 依赖安装

**感知模块（Python，Orin 部署）**：

| 包名 | 用途 | 安装方式 |
|------|------|----------|
| `tensorrt` | TRT 推理引擎 | JetPack SDK 预装 |
| `pycuda` | CUDA 缓冲区管理 | `pip install pycuda` |
| `opencv-python` | 图像预处理 | `pip install opencv-python` |
| `rclpy` | ROS2 Python 客户端 | ROS2 安装自带 |
| `vision_msgs` | Detection2D 消息定义 | `sudo apt install ros-${ROS_DISTRO}-vision-msgs` |
| `cv_bridge` | ROS Image ↔ OpenCV 转换 | `sudo apt install ros-${ROS_DISTRO}-cv-bridge` |

**规划模块（C++，订阅端）**：

| 包名 | 用途 | 安装方式 |
|------|------|----------|
| `rclcpp` | ROS2 C++ 客户端 | ROS2 安装自带 |
| `vision_msgs` | Detection2D 消息定义 | `sudo apt install ros-${ROS_DISTRO}-vision-msgs` |

**CMake 依赖**（规划模块 `CMakeLists.txt`）：

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

---

## API 稳定性承诺

以下内容在 5/15 截止日期前**保证不变**：

| 保证稳定 | 说明 |
|----------|------|
| 消息类型 | `vision_msgs/Detection2DArray`，不会更换为自定义消息 |
| Topic 名称 | `/traffic_light/detections`（默认值） |
| `class_id` 为字符串 | 如 `"red"`、`"greenLeft"`，不是整数 |
| 现有 7 类名称不变 | `red`、`yellow`、`green`、`redLeft`、`greenLeft`、`redRight`、`greenRight` |
| `results` 数组 | 每个 `Detection2D` 固定包含 1 个 `ObjectHypothesisWithPose` |
| 坐标系 | bbox 为**原始图像像素坐标**（非归一化） |

### 可能的兼容变更

| 变更类型 | 对订阅端的影响 |
|----------|----------------|
| 新增类别（如 `yellowLeft`） | 不影响已有解析，但未处理的类别会被忽略 |
| 模型更换 | 置信度分布可能变化，class_name 不变 |
| 输入分辨率调整 | bbox 坐标仍为原始图像坐标，对订阅端透明 |

### 防御性编码建议

1. **处理未知类别**：对不在已知集合中的 `class_id`，按警示处理
2. **基于字符串匹配**：使用 `class_id` 字符串，不要假设整数编号
3. **检查 `results` 非空**：访问 `results[0]` 前先检查
4. **不假设检测数量**：每帧可能有 0 个或多个结果
