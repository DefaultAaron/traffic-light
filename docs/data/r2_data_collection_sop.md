# R2 数据采集与标注 SOP（多模态版）

> **状态**：执行 SOP（待硬件锁定后冻结）。
> **日期**：2026-04-28
> **范围**：R2 阶段 10–14 类联合检测器（交通灯 + 栏杆）的数据采集、同步、标注、切分、QA 与发布预备。
> **传感器配置**：双 8MP 相机（normal FOV + wide FOV）+ LiDAR 点云。
> **多模态定位**：本 SOP 一次采集，三路价值——
> （a）2D 检测训练数据（主线 5/15 交付）；
> （b）replay 阶段的距离 / 抖动 / 遮挡客观真值（用于失败模式诊断）；
> （c）R3 候选：跨模态自监督预训练 / 3D 融合检测的种子数据。
>
> 与计划文档的关系：本 SOP 是 [`../planning/development_plan.md`](../planning/development_plan.md) §三 R2 阶段中"R2 数据采集"里程碑的执行规范；指标定义须与 [`../reports/phase_2_round_1_report.md`](../reports/phase_2_round_1_report.md) §"R2 范围扩展"对齐。

---

## 1. 总体原则

1. **一次采集，多路复用**：不为单一任务定制采集；同一份 ROS 2 bag 同时支持检测训练、replay 评测、时序实验、跨模态预训练。
2. **Raw 永远保留**：原始连续视频 + 原始 LiDAR 帧不得丢弃。压缩允许，但不得抽帧或裁切。最少留存 6 个月，建议归档保留 ≥24 个月。
3. **失败先证据后实验**：龙门抖动滤波、Bulb-first 双重标注、NWD/P2 head 等 SOTA 候选项，**不**作为 R2 必备，仅当 replay 暴露对应失败模式时再启动；启动决策依据 §10 + §7.2（抖动）+ [`../../research/surveys/detection_enhancements.md`](../../research/surveys/detection_enhancements.md) §3。
4. **按站点 / 视频切分，不按帧切分**：避免 train / val 时序泄漏（GPT 评审强调）。
5. **稀有类与硬负样本同等优先**：稀有箭头方向类与背景假阳性（警示牌 / 绿色墙面 / 厂房 LED）在采集阶段就有显式覆盖任务，而不是事后补救。
6. **可发布性预留**：采集即按可发布标准记录（隐私脱敏链路、授权条款、设备元数据），即便最终不发布，成本也极低。

---

## 2. 硬件配置与几何布局

### 2.1 相机

| 通道 | 分辨率 | 镜头 FOV | 角色 | 主要任务 |
|---|---|---|---|---|
| **Cam-N**（normal） | 8 MP（≈3840×2160 或 4096×2160） | 水平 ~50–60° | 主感知通道 | 远距 / 远距小灯主标注源；标准 ADAS 视野 |
| **Cam-W**（wide） | 8 MP | 水平 ~100–120° | 周边视野 + 多灯龙门 | 龙门两侧灯 / 路侧栏杆 / 接近路口外延 |

**安装要求**：

- 同一刚性支架，基线 5–15 cm（小基线，**不**作为可靠测距源；测距交给 LiDAR）。
- 与车体振动解耦（防抖橡胶垫 / 隔振柱）。**关键**：抖动若来自相机本身而非龙门，会污染后续抖动诊断。
- 镜头：自动驾驶级（耐温 -20–85 ℃、机械抗振、IR-cut 滤光）。
- 快门：建议 global shutter（行车 60 km/h 时小灯运动模糊会显著影响远距标注）。若仅 rolling shutter，记录扫描方向并在标注 metadata 中带行字段。
- 曝光：双通道独立 AE，避免 wide 通道被路面强反光拉低整帧曝光导致灯过曝。
- 白平衡：建议锁定为日光模式或保存 RAW（DNG）以便离线校正；不允许全自动 WB（导致同一灯前后帧颜色漂移）。
- 帧率：30 fps **必须**，60 fps **建议**（黄→红状态切换 1 s 内，30 fps 最少 30 帧但仍易遗漏过渡帧）。

### 2.2 LiDAR

| 项 | 规格目标 |
|---|---|
| 线数 | 64–128 线（Robosense / Hesai 等；按车辆已选硬件） |
| 频率 | 10 Hz native（不需要插帧） |
| 视场 | 360° 优先；最少前向 180° |
| 安装 | 顶置，水平校准（pitch 误差 < 1°） |
| 输出 | 每点带时间戳 + intensity；保留原始 .pcap 或厂商原生格式 |

**对小信号灯的现实预期**：

- 单点 LiDAR 在 100 m 外难以稳定击中 30 cm 直径灯壳；但击中**龙门钢架与立柱**非常稳定。
- 因此 LiDAR 在本项目中的主作用是：**(1)** 提供龙门 / 立柱的 3D 位置 → 信号灯距离的**代理真值**；**(2)** 抖动诊断；**(3)** 背景几何（区分"墙上的绿色"vs"龙门上的绿灯"）。
- 不假设 LiDAR 能直接给出每盏灯的 3D bbox。

### 2.3 同步与定位

- **硬同步**：PTP（IEEE 1588）服务器 + 相机外触发；LiDAR 跑同一 PTP 域。所有时间戳从同一 epoch 走，漂移目标 **< 1 ms**。
- **GPS / IMU**：必装。GPS 用于站点元数据 + 后续地图先验对齐；IMU 用于车辆自身抖动通道，与龙门抖动诊断做差分。
- **失败响应**：单帧同步漂移 > 5 ms 时，写入 bag 元数据 `sync_warn=1`；连续 > 100 ms 漂移时整段 session 标记 `unusable_for_training`，仅留作 replay raw。

### 2.4 标定

| 类型 | 方法 | 频率 |
|---|---|---|
| 内参（每相机） | OpenCV 棋盘格或 AprilTag 板 | 每月一次 + 任何镜头干预后 |
| 相机-相机外参 | 共视棋盘格 | 月度，与内参同步 |
| 相机-LiDAR 外参 | 反光板 / Apriltag 反光靶 | 月度 |
| 时序对齐验证 | 跑动目标互相关（车辆离开同步触发瞬时） | 每次出车一次（5 分钟标定段） |

**标定档归档**：每次 session 起点写入 `calib/<session_id>.yaml`；后续重处理永远引用该文件，不引用全局"最新标定"。

---

## 3. 采集策略

### 3.1 站点选择（CN 现代龙门为主）

| 站点类型 | 目标 session 数 | 关键特征 |
|---|---|---|
| **现代龙门**（横排多灯 + 直行 / 左 / 右箭头） | ≥ 20 | R2 主战场；公开数据集**严重缺**该构型 |
| **传统垂直立柱** | 5–10 | 兼容场景，避免模型只学龙门 |
| **道闸路口 / 园区出入口** | ≥ 10 | barrier 类训练数据来源 |
| **高架 / 立交龙门** | ≥ 5 | 困难视角（向上仰拍） |
| **背景陷阱站点** | ≥ 5 | 厂房 LED 招牌 / 警示牌 / 绿色幕墙密集区（**无信号灯**），用于硬负挖掘 |

每站点至少要有 **正午 / 黄昏 / 夜晚** 三个时段中的两个。

### 3.2 时段与天气分层

每条 session 在 `session_meta.yaml` 中显式打 tag，至少覆盖：

- `lighting`：`day_clear` / `day_overcast` / `dawn` / `dusk` / `night_lit` / `night_dark`
- `weather`：`clear` / `light_rain` / `heavy_rain` / `fog` / `snow`（按地理可用性）
- `glare`：`none` / `front` / `side`（太阳与车头夹角 < 30° 标 `front`）

R2 截止前**必须**至少各采到一段：dusk-glare、night_lit、light_rain、front-glare-noon。缺一项不视为采集完成。

### 3.3 困难场景主动覆盖

| 失败模式 | 主动采集动作 | 与 GPT 评审条目对应 |
|---|---|---|
| 远距 / 小灯 | 选有 100 m+ 直线接近的路口；贴近接近时不变道 | "distant tiny lights" |
| 逆光 / 黄昏 | 朝向太阳方向接近龙门，时段卡日落前 30 分钟 | "glare / dusk" |
| Halo / blooming | 雨夜 LED 强光下接近 | "halo / blooming" |
| 卡车遮挡 | 货运路段，跟车而非超车 | "occlusion" |
| 龙门抖动验证 | 风力 5 级以上日子定点拍摄高架龙门 ≥ 5 分钟连续 | "wind-induced gantry vibration（先验证再修复）" |
| 背景假阳性 | 厂区警示牌 / 园区绿幕墙 / 户外 LED 屏附近行驶（**非路口**） | "hard-negative seeds" |

### 3.4 不可加速场景（避免假覆盖）

- 夜间不允许通过白天素材人工调暗模拟。
- 雨天不允许通过虚拟雨噪生成"覆盖"（合成数据可作 R3 增强，**不**作 R2 raw 训练替代）。

---

## 4. 录制规范

### 4.1 容器与编码

- 顶层容器：**ROS 2 bag (`.mcap`)**，所有传感器同 bag。便于 replay 工具链一致。
- 相机流：H.264 / H.265 高码率（≥ 25 Mbps for 8MP@30）；**额外**每 60 s 抽一张 PNG / DNG 关键帧入 `raw_keyframes/` 用于 RAW 色彩工作流。
- LiDAR：原生 `.pcap` 或 `sensor_msgs/PointCloud2`，每点保留时间戳与 intensity。
- GPS / IMU：标准 ROS 2 消息类型，10 Hz。

### 4.2 文件布局

```
data/r2/raw/
├── <site_id>/
│   └── <session_id>/                  # session_id = YYYYMMDD-HHmm-driver
│       ├── meta.yaml                  # 见 §4.3
│       ├── calib.yaml                 # session 启用的标定档（拷贝入 session）
│       ├── multimodal.mcap            # 相机 + LiDAR + GPS + IMU 同步 bag
│       ├── raw_keyframes/             # 1Hz PNG/DNG 用于浏览
│       └── sync_log.csv               # 每秒同步漂移记录
data/r2/annotated/
├── <split>/                           # train / val / test （按 site_id 划分）
│   ├── images/                        # 抽帧图像（默认 cam-N）
│   ├── images_wide/                   # 抽帧图像（cam-W；仅子集）
│   ├── labels/                        # YOLO 格式
│   └── annotations/                   # COCO 格式
data/r2/derived/
├── distance_gt/                       # LiDAR 推出来的距离 GT（§7.1）
├── vibration_traces/                  # 龙门抖动频谱（§7.2）
└── hard_negatives/                    # 模型挖出的 FP 切片
```

### 4.3 Session 元数据（`meta.yaml`）

每段 session 必填字段：

```yaml
session_id: 20260503-1730-zwu
site_id: bj-haidian-zhongguancun-N4
driver: 吴正日
weather: light_rain
lighting: dusk
glare: front
gantry_type: modern_horizontal
duration_s: 1820
notes: "5 级风，龙门主动采集；日落前 25 min 接近"
permits: ["驾驶许可 #2026-0503-01"]
calib_ref: calib/2026-04-monthly.yaml
sync_drift_max_ms: 0.7
```

### 4.4 出车前 / 后 checklist

**出车前**：

- [ ] PTP 服务器锁定，LiDAR 时间戳同步通过自检
- [ ] 最新标定档已加载且检验通过（5 min 静态板 + 跑动验证）
- [ ] 存储剩余 ≥ 80 %
- [ ] 镜头 / LiDAR 防尘罩清洁
- [ ] 30 s 测试录制 + 离线校验（同步、码率、对齐）
- [ ] `meta.yaml` 模板填好

**出车后**：

- [ ] 自动 checksum 上传至项目存储
- [ ] 同步漂移报告归档
- [ ] 关键帧 + LiDAR 截图自动入 QA 看板（站点覆盖跟踪）

---

## 5. 隐私 / 合规预处理

### 5.1 自动脱敏

- **人脸 + 车牌**模糊：发布前对所有 RGB 帧执行；内部训练默认**关闭**（避免擦掉小灯邻近像素），但训练数据集需带 `anonymized` flag。
- **GPS**：发布版四舍五入到 100 m 网格；内部 replay 保留全精度。
- **音频**：bag 中默认不录音；如开启车内麦克风（驾驶员事件标注用），发布前彻底剥离。

### 5.2 授权链路

- 车辆驾驶许可（项目层面备案）。
- 拍摄路段公示（园区站点必须）。
- 后续若发布数据集 / 切片：法务复审 + 二次脱敏 + 许可证选择见 §11。

---

## 6. 标注体系

### 6.1 类别清单（R2 基线 10–14 类）

**交通灯下限 9 类**（已锁定，与 R1 7 类直接对接）：

| ID | 类别 | 说明 |
|---|---|---|
| 0 | red | 圆灯红 |
| 1 | yellow | 圆灯黄 |
| 2 | green | 圆灯绿 |
| 3 | redLeft | 左转红 |
| 4 | greenLeft | 左转绿 |
| 5 | redRight | 右转红 |
| 6 | greenRight | 右转绿 |
| 7 | forwardRed | 直行红（**新增**） |
| 8 | forwardGreen | 直行绿（**新增**） |

**交通灯条件扩展**（每类需 ≥ 200 实例 + 多站点出现 ≥ 5 站才晋级）：

- `leftYellow` / `rightYellow` / `forwardYellow`
- 闪烁状态：作为属性而非类别（见 §6.2）

**栏杆 MVP 1 类**：`barrier`（仅检测存在）。
**栏杆条件扩展**：`armOn` / `armOff`，每态 ≥ 500 实例 + ≥ 3 站才晋级（见 [`../planning/development_plan.md`](../planning/development_plan.md) §三 R2 范围）。

### 6.2 每实例属性（不进入类别 ID，但落到 COCO `attributes`）

| 字段 | 取值 | 用途 |
|---|---|---|
| `bbox` | xyxy（贴灯壳） | 主标注 |
| `bulb_bbox`（可选） | xyxy（仅亮 bulb） | R3 bulb-first 候选；R2 仅在远距 / 部分遮挡样本上抽样标注 |
| `occlusion` | none / partial / heavy / invisible | hard-case slice 与训练加权 |
| `truncation` | bool | 边缘裁切 |
| `distance_bucket` | near (<50m) / mid (50–150m) / far (>150m) | 距离桶；标注阶段**人工填**，§7.1 流水线在 `confidence ≥ medium` 时覆盖（保留人工值在 metadata 备查） |
| `state_attr` | steady / flashing / off | 闪烁作为属性，避免类别爆炸 |
| `fixture_type` | gantry / pole / overhead_bridge / barrier_arm | 用于站点结构分析 |
| `lane_relevance` | ego / non_ego / unknown | 关联性标注（**仅**当车道箭头清晰时填） |

### 6.3 负样本标注（背景假阳性）

每条 session 至少抽 100 帧标注 `hard_negative_regions`：警示牌 / LED 招牌 / 绿色幕墙 / 车尾灯阵 / 龙门上的非信号灯（路名牌灯）。这些用于：

- R2 训练阶段的 hard-negative mining 种子；
- 评测阶段的 background-FP 切片专项指标。

### 6.4 借鉴外部资源（不直接派生）

ATLAS 数据集（CC BY-NC-SA 4.0，IEEE IV 2025；arXiv:2504.19722）的 25 类 pictogram-state 标签体系是优秀参考，可对照其 taxonomy 设计字段，但**不得**派生其图像 / 标签（许可证限制非商用 + share-alike）。

---

## 7. 多模态融合（数据层面，超越单纯检测）

> 本节定义 LiDAR + 双相机的"超出 2D bbox"利用方式。R2 不要求模型用上多模态；但**采集阶段就要把这些信号做出来**，否则无法回头补。

### 7.1 LiDAR-辅助距离估计（三层 fallback 流水线）

**目标**：为每个 TL 标注生成 `distance_m`，用于自动化 near / mid / far 桶、replay 阶段的 confidence-vs-distance 校准曲线、训练阶段距离权重实验。

**物理可行性预估**（决定流水线设计）：

- 30 cm 灯壳 @ 100 m，64 线 LiDAR（垂直分辨率 ~0.2°）每帧 **≤ 1 点** 命中。中位数在 0–1 个点上无意义。
- < 50 m：直射返回足够稳定。
- 50–150 m：直射稀疏，需借助龙门结构作锚点。
- \> 150 m：基本无可用直射，距离仅作粗估或交给人工。

不做"用龙门立柱 bbox 作代理"——该 bbox **不在 §6 标注 taxonomy 内**，凭空假设它存在是错误的。下面给出真实可执行的流水线：

#### Tier 1：直接投影（近距，≤ 50 m）

1. 用 session calib 把 LiDAR 点云变换到 cam-N 坐标系并投影到像素平面。
2. 对每个 TL bbox：收集投影落入 bbox 内 **且** 3D 距相机 ∈ [3, 200] m 的点。
3. 若点数 ≥ 3：`distance_m = trimmed_mean(ranges, trim=0.2)`，`confidence=high`，`anchor_kind=direct`。

#### Tier 2：龙门聚类锚定（中远距，自动化，无需新增标注）

Tier 1 点数 < 3 时启用：

1. **地面平面提取**：RANSAC fit ground plane，提供 height-above-ground。
2. **结构候选聚类**：在车前 ±20 m 横向、3–8 m 离地高度范围内做 Euclidean / DBSCAN 聚类。
   - **立柱候选**：垂直跨度 > 2 m、水平横截面 < 1.5 m。
   - **横梁候选**（可选）：两根立柱候选水平距离 5–15 m、高度相近时，连接二者间的水平点云簇。
3. **TL bbox ↔ 候选关联**：把每个候选投影回像素，按下规则找匹配候选：
   - 立柱：bbox 中心到立柱投影竖线的像素距离 ≤ `0.5 × bbox 对角线`。
   - 横梁：bbox 在横梁投影线下方 ≤ `2 × bbox 高度`。
   - **匹配恰好一个候选**才算关联；多义或无关联 → 走 Tier 3。
4. 关联成功：`distance_m = median(cluster_points 距离)`，`confidence=medium`，`anchor_kind=gantry_pole | gantry_beam`。

#### Tier 3：无可用 LiDAR 信号

`distance_m = null`，`confidence=none`，`anchor_kind=none`。**此时**才回退到人工 §6.2 `distance_bucket`（near / mid / far），并以人工标注为唯一真值。

#### 实现入口

`scripts/lidar_distance_gt.py`（待 R2 数据流就绪后实现，**不**在 5/15 关键路径上；先有 Tier 1，Tier 2 可后补）。每条 session 跑一次，离线产出。

#### 输出 schema（`derived/distance_gt/<session>.parquet`）

| 列 | 类型 | 含义 |
|---|---|---|
| `frame_id` | int | 与 COCO `image_id` 对齐 |
| `ann_id` | int | 与 COCO `ann_id` 对齐 |
| `bbox` | [x1,y1,x2,y2] | join 校验用 |
| `tier` | enum | `1` / `2` / `3` |
| `n_points_direct` | int | Tier 1 命中点数 |
| `distance_m` | float \| null | 最终估计 |
| `confidence` | enum | `high` / `medium` / `low` / `none` |
| `anchor_kind` | enum | `direct` / `gantry_pole` / `gantry_beam` / `none` |
| `cluster_id` | int \| null | Tier 2 簇 ID（QA 复盘用） |
| `assoc_pixel_dist_px` | float \| null | Tier 2 关联残差，用于阈值调参 |

#### QA / 验收门

- 抽 5 % 估计交叉核验：可视化 LiDAR 簇 3D 视图 + 像素投影叠加图，由数据负责人逐条确认。
  - **不**用 cam-N + cam-W 作 stereo 验距 —— 基线 5–15 cm，远距 stereo 不可靠（LiDAR 本身就是更可靠的 3D 真值源）。
- **关联率门槛**：mid-range（50–150 m）标注中 Tier 1 + Tier 2 累计 `confidence ≥ medium` 的比例 < 60 % 时，本信号**不**进入评测；先调 Tier 2 启发式。
- LiDAR-相机标定漂移自检：5 % 抽检若发现系统性距离偏差 → 触发 §2.4 重标定。

#### 可选辅助标注（仅当 Tier 2 关联率持续不达标）

若多次调参后 Tier 2 关联率仍 < 60 %，**才**为龙门帧加一个粗略立柱 bbox 标注（标注成本 < 10 %）。**默认不开**，且不阻塞 R2 baseline 训练。

#### 失败模式总结

| 场景 | 表现 | 处理 |
|---|---|---|
| 立柱式（非龙门）路灯 | Tier 2 找不到 5+ m 立柱 | 走 Tier 3 人工估 |
| 双龙门并列（罕见） | 多义关联 | `confidence=none` |
| 雨 / 雾 | LiDAR 稀疏化 | confidence 降档；§7.6 weather_score 加权折扣 |
| 标定漂移 | 系统性偏差 | 5 % 抽检捕获 → 重标定 |

#### 用途

- 距离桶（near / mid / far）自动化，仅当 `confidence ≥ medium` 时覆盖人工标 `distance_bucket`；否则保留人工值。
- replay 阶段的 confidence-vs-distance 校准曲线（用 high/medium 子集）。
- 训练阶段按距离的 sampling weight 实验（远距样本上采样）。

### 7.2 龙门抖动诊断（验证再决定是否做滤波）

**前提**：本节复用 §7.1 Tier 2 的立柱聚类方法识别龙门结构 — **不要求**新增立柱 bbox 标注。

**采集模式**：必须为 §3.3 风天 5 分钟**定点静止**录像（车辆熄火、传感器供电）。行车段不适用，因为车辆自身位移 + 视角变化会主导信号。

**采样率前提（Nyquist 约束）**：

| 通道 | 原始采样率 | Nyquist | 本节决策频带 | 说明 |
|---|---|---|---|---|
| LiDAR 立柱 (x, y, z) 时序 | 10 Hz native | 5 Hz | **1–4 Hz** | 留 1 Hz Nyquist 防混叠保护带 |
| cam-N 像素 (u, v) 时序 | 30 fps（**60 fps 优先**） | 15 / 30 Hz | **1–5 Hz** | 像素侧仍可探至 5 Hz |
| IMU 车体加速度 | ≥ 100 Hz（车端常规） | ≥ 50 Hz | 用作扣除参考，不做带内决策 | — |

**采样率规则（按通道，互不通用）**：

- **LiDAR 通道**：保持 10 Hz native，**不**降采样；决策带因 Nyquist 受限于 4 Hz 上限。
- **像素通道**：保持相机 native fps（30 / 60 fps）逐帧处理，**不**降采样；决策带可用 1–5 Hz。
- **绝对禁止**：把任何通道降采到 < 2 × 决策带上限的速率（前一版"按 1 fps 抽 300 帧用于 FFT"即此错误，已删除）。

**流程**：

1. **车辆静止性核验**：IMU 加速度 RMS 在 5 分钟窗口内 < 阈值（车体未怠速振动）。否则该 session 抖动数据弃用。
2. **立柱跟踪**（LiDAR 通道，10 Hz）：每帧用 §7.1 Tier 2 提取立柱聚类；按聚类质心连续帧关联（最近邻 + 距离阈值），形成 `(t, x, y, z)` 时序。要求至少一根立柱**在整个 5 分钟内连续可见**。300 s × 10 Hz = 3000 样本。
3. **像素轨迹**（cam-N 通道，**全帧率不抽帧**）：在 5 分钟段内人工选**一个高对比度灯泡 ROI**（亮 LED bulb，跨 ≥ 5 px），用归一化互相关（NCC）模板匹配跨 9000 / 18 000 帧（30 / 60 fps）逐帧定位中心，得到亚像素精度 (u, v) 时序。**不**依赖逐帧人工 bbox 标注。
4. **车体扣除**：IMU 加速度二次积分（带 high-pass 抑制零飘）得到车体位移 → 同时从 LiDAR 立柱时序与像素时序中扣除（坐标变换后），剔除车体怠速 / 阵风导致的整车晃动。
5. **PSD 估计（Welch）**：两路时序分别做 Welch's method（汉宁窗、30 s 段、50 % 重叠）。频率分辨率 ≈ 0.03 Hz，决策带内有 ~30 个 bin 足够稳定。**不**用一次性 FFT —— 单次 FFT 方差过大，3 σ 阈值不可信。
6. **逐通道噪声地板**：
   - 像素 PSD：取 `8–13 Hz` 段中位数为 noise floor（30 fps 时 Nyquist 15 Hz，留 ~13 % 防 Nyquist-边缘伪影；60 fps 时上限可放宽至 20 Hz，但 8–13 Hz 已足够稳健）。
   - LiDAR PSD：取 `0.05–0.5 Hz` 段中位数为 noise floor（受限于 Nyquist 5 Hz，无法用上带外参考）。
7. **逐通道带内峰检**：
   - 像素：1–5 Hz 内有 PSD bin > **3 × noise floor**。
   - LiDAR：1–4 Hz 内有 PSD bin > **3 × noise floor**。
8. **跨通道相干性**：两通道带内峰中心频率差 ≤ **0.2 Hz** 才视为一致；否则不算抖动证据（仅像素峰可能是 rolling-shutter / AE 振荡 / 树枝阴影抖动等假象）。

**产出**：`derived/vibration_traces/<session>.parquet` —— 列：`session_id, gantry_cluster_id, lidar_peak_hz, lidar_amp_m, lidar_snr, pixel_peak_hz, pixel_amp_px, pixel_snr, freq_diff_hz, coherent_peak (bool)`。

**决策门**（启动 [`../planning/temporal_optimization_plan.md`](../planning/temporal_optimization_plan.md) 中 tracker 高通滤波的条件）：高架 / 龙门 session 中 `coherent_peak=True` 的 session 比例 **> 30 %**。

- 仅像素一路有峰：先排查相机原因（rolling-shutter / AE 抖动 / 阴影），**不**作为龙门抖动证据。
- 仅 LiDAR 有峰但像素无：可能是 LiDAR 自身机械振动；查 IMU 一致性。
- 双通道都无峰：抖动假说在本数据上**不成立**，滤波器**不**落地。

**已知限制**：

- 静止录像段对司机操作有要求（找安全停车点对着龙门 5 分钟，可能受限于路况）。如难以获得，本诊断推迟到 R3，并把 [`../planning/temporal_optimization_plan.md`](../planning/temporal_optimization_plan.md) 的抖动滤波默认设为"未验证 / 不启动"。
- 立柱被树木 / 标志牌遮挡时聚类不稳，需切换到另一根立柱重做。

### 7.3 跨模态硬负挖掘

- **方法**：对部署 R1 模型的 false-positive 候选区域，查 LiDAR 是否存在与"信号灯位于 3–6 m 高度"几何一致的返回点。如果该区域 LiDAR 显示是平面墙体（无垂直龙门结构），则强假阳性。
- **产出**：`derived/hard_negatives/<session>/`，按规则排序的切片。
- **用途**：R2 训练前的 hard-neg pool 优先级排序。

### 7.4 跨模态自监督预训练（R3 候选）

- **资产**：6 个月以上的 raw bag = camera-LiDAR 配对帧池。
- **方法候选**：
  - DINOv2 风格视觉自监督，单纯用 raw RGB（不需要 LiDAR）；
  - MoCo / ImageBind 风格 camera-LiDAR 对比学习（R3 探索）；
  - 像素级 depth-aware MAE（mask 像素时 condition on LiDAR depth）。
- **决策**：R3 启动；R2 阶段只**保证数据保留**，不做训练。

### 7.5 双相机 HDR / 视场互补

- **HDR**：cam-N 与 cam-W 可设为不同曝光基线（cam-N 暗优先保灯亮区；cam-W 亮优先保路面 / 行人）。重叠视场内可像素级融合（R3 增强），不进入 R2 主线。
- **视场互补**：cam-W 主要用于：
  - 龙门两侧 / 视场外缘灯（cam-N 视场被裁掉的）；
  - 道闸（栏杆通常贴近车头，cam-N 容易切到）；
  - 大转弯路口的非主观察方向灯。

### 7.6 天气自动打分

- **方法**：LiDAR 每帧返回点密度统计（`points_in_range / expected_points`）做 z-score；密度显著下降 → 雨 / 雾。
- **产出**：每帧自动打 `weather_auto_score`，与 §4.3 人工填的 `weather` tag 对照，差异样本进 QA 看板。

### 7.7 远期：3D 检测 / BEV 融合（R3 之外）

- 单纯记一笔：保留 raw 多模态 bag = 未来如果做 BEV / point-pillar + camera 融合，数据已就位。本 SOP 不为该路线做特殊设计，但也不挡路。

---

## 8. 标注流程

### 8.1 工具链

- **CVAT**（默认）— 支持 video 模式 + 多通道（双相机），可直接导出 YOLO + COCO。
- 如 CVAT 无法满足 LiDAR 投影叠加，备选 SUSTechPOINTS / Xtreme1。

### 8.2 三阶段流程

| 阶段 | 动作 | 产出 |
|---|---|---|
| 1. 模型预标 | R1 best 模型在抽帧上跑推理 → 转 CVAT pre-label | 80% bbox 已成形，标注员只改 |
| 2. 人工修正 | 标注员逐帧调框 + 填属性 | 完整 YOLO + COCO 标签 |
| 3. 高级抽检 | 资深标注员抽 5 % 样本，IAA（kappa）目标 ≥ 0.85 | QA 报表，不达标的 session 整体回炉 |

### 8.3 关键边界规则

- **重度遮挡但 bulb 可见** → 仍标，`occlusion=heavy`。
- **关灯 / off** → 标 `state_attr=off`，类别按 fixture 默认（如 `red` 配色但灯灭也走 fixture 颜色，不允许"猜"应为何色）。
- **同一龙门多灯** → 每盏单独标，**不**合并。
- **行人 / 自行车 pictogram** → R2 不标，统一打 `out_of_scope=true`，留 R3 决定。
- **闪烁** → 单帧只反映当前可见态；闪烁属性看 ≥ 1 s 内的状态序列。

### 8.4 双相机标注分工

- **cam-N**：**全标**（主训练源）。
- **cam-W**：仅在以下情况标：
  - cam-N 视场外有灯（FOV 互补帧）；
  - cam-N / cam-W 同一灯出现在重叠区（一致性 QA 用，按 1 % 抽样）。
- 标注成本：粗算 cam-W 标注量 ≤ cam-N 的 20 %。

---

## 9. 切分与评测

### 9.1 切分原则

- **按 site_id 切分**：train / val / test 不共享站点。
  - 推荐：12 站 train / 4 站 val / 4 站 test（含至少 1 个夜晚站点 + 1 个龙门站点 + 1 个 barrier 站点纳入 test-only）。
- **不**做随机帧切分。
- **不**用 R1 异域数据（LISA / BSTLD / S2TLD）做 R2 评测；R1 数据彻底退役（见 user memory：R1 数据替换政策）。

### 9.2 必报指标（不仅是 overall mAP）

| 指标族 | 内容 |
|---|---|
| **每类 AP / Recall** | 9 个 TL 类 + 1 / 2 个 barrier 类全列；不做平均掩盖 |
| **距离桶 mAP** | near / mid / far 三档，每档分类报；远距档单独评是否过线 |
| **困难场景切片 mAP** | glare / dusk / halo / rain / occlusion / overhead_gantry 各切片 |
| **背景 FP 率** | 在 §6.3 背景陷阱站点上的 FP / 帧 |
| **状态稳定性**（如启用时序轨道） | flicker rate / 跟踪 ID 寿命 |

### 9.3 困难切片协议

每条 test 帧打可叠加的多个标签：`{glare, dusk, halo, rain, distant, occlusion, overhead_gantry, background_trap}`。切片标签独立于训练标签，存于 `splits/hard_case_tags.csv`，便于发布。

---

## 10. 决策门 / 时间轴

| 节点 | 日期 | 必达条件 |
|---|---|---|
| 硬件锁定 | 2026-04-30 | 双 8MP 相机 + LiDAR 型号定 + PTP 同步通过自测 |
| 标定档 V1 | 2026-05-02 | 内 / 外参 + 时序对齐验证 |
| 首日采集 | 2026-05-03 | ≥ 1 个龙门站点 + 1 个 barrier 站点 |
| 1k 帧标注完成 | 2026-05-08 | 模型预标 → 人工修正 → 5 % 抽检 |
| R2 baseline 第一次训练 | 2026-05-11 | 任一备选检测器 + 当前已标数据 |
| Replay + 失败桶分析 | 2026-05-13 | 自动产出 §9.2 指标全套 |
| **5/15 主线交付** | 2026-05-15 | 训练版 R2 + 部署评测 + 困难切片报告 |

5/15 之后所有"启动条件依赖 replay"的实验（NWD、P2 head、SPD-Conv、Bulb-first、抖动滤波、SAHI 全开等）才进入排期。

---

## 11. 发布预备（5/15 后启动）

### 11.1 可发布资产候选

按 GPT 评审 §5.1 的优先级：

1. CN 现代龙门交通灯数据集（自采，可发布）；
2. 困难切片协议 + 切片清单；
3. TL × Barrier 联合检测 benchmark；
4. Orin 部署报告（YOLO-family vs DEIM-D-FINE）；
5. tracker / smoother 跨语言 fixture 测试集；
6. **多模态延伸**（项目专属优势）：
   - 距离 GT（LiDAR 推出来的）— 学界很少有；
   - 龙门抖动频谱数据集 — 文献几乎为零；
   - camera-LiDAR 配对帧（小子集，做跨模态预训练 benchmark 种子）。

### 11.2 许可证基线

- 自采代码：**Apache-2.0**。
- 自采数据集：候选 **CC BY 4.0**（最大使用）或 **CC BY-NC-SA 4.0**（保守）；最终由法务定。
- 不得发布派生自 LISA / BSTLD / ATLAS 的内容。

### 11.3 多模态发布的额外门槛

- LiDAR 点云包含车辆周边 360° 几何，可能涉及拍摄区域机密 / 园区版图；发布前**必须**裁切到信号灯相关 ROI（前向 100 m × 30 m）。
- GPS 必须打 100 m 网格化。
- 任何 IMU / 车速序列发布前评估能否反推路线。

---

## 12. 风险与缓解

| 风险 | 触发场景 | 缓解 |
|---|---|---|
| 同步漂移 > 1 ms | PTP 失锁 | 实时监控 + sync_warn flag；连段失败整 session 弃训练 |
| 8MP × 2 + LiDAR 存储爆 | ≈ 25–40 GB / 小时 | 提前规划 ≥ 20 TB 项目存储 + 滚动归档策略 |
| LiDAR 远距打不到灯 | 物理限制 | 用龙门立柱代理（§7.1 已设计） |
| cam-W 标注成本失控 | 默认全标 | §8.4 限制为子集标注 |
| 站点偏移（少数站点过采） | 单一驾驶员习惯路线 | 按 §3.1 配额硬性约束 + 周度站点覆盖看板 |
| 法务发布卡点 | 5/15 后才启动评估 | 采集阶段就按可发布标准记录；不影响主线 |
| Bulb-bbox 子集太小不足 R3 用 | §6.2 仅抽样标 | 抽样比例 ≥ 5 %；远距 / 遮挡样本提升至 ≥ 20 % |
| LiDAR 元数据泄漏 | 直接发 raw bag | 发布通道**禁止**直接发 raw；只发派生数据（§11.1） |

---

## 13. 与计划文档的对接

| 计划文档 | 本 SOP 对接段 |
|---|---|
| [`../planning/development_plan.md`](../planning/development_plan.md) §三 R2 | §1, §6（类别）, §10（时间轴） |
| [`../planning/temporal_optimization_plan.md`](../planning/temporal_optimization_plan.md) | §3.3（连续视频要求）, §7.2（抖动诊断 → 滤波启动条件） |
| [`../planning/cross_detection_reasoning_plan.md`](../planning/cross_detection_reasoning_plan.md) | §6.2 `lane_relevance` + planner-prior 数据基础 |
| [`../../research/surveys/detection_enhancements.md`](../../research/surveys/detection_enhancements.md) | §3.3（hard-negative 主动采集）, §6.3（负样本标注） |
| [`../../research/contributions/field_gaps_and_contributions.md`](../../research/contributions/field_gaps_and_contributions.md) | §11（发布预备） |

---

## 14. 待 PM / 团队确认事项

1. 双相机型号 + LiDAR 型号最终选定（影响内 / 外参标定模板）。
2. 站点清单（§3.1）需地面团队补充实地可达站点。
3. 标注外包还是自建（影响 §8.2 流程时长）。
4. 数据集发布范围与许可证（§11.2）— **5/15 主线交付前不需要决定**，但建议 5/30 前定。
5. cam-W 是否进入主训练流（默认否，等 cam-N baseline 建立后再做 wide-FOV ablation）。

---

*本 SOP 在硬件锁定 + 首日采集复盘后冻结 V1；后续修订须经数据负责人 + PM 联签。*
