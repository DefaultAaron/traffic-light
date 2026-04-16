# 7 类标签分布分析

**日期**: 2026-04-16
**数据源**: S2TLD（全部子集已重标注）、BSTLD（训练集 YAML 含方向标签）、LISA

---

## 预期 7 类分布（转换脚本重跑后）

| 类别 | ID | S2TLD | BSTLD | LISA | 总计 | 占比 |
|------|-----|------:|------:|-----:|-----:|-----:|
| red | 0 | 5,380 | 8,387 | 44,318 | **58,085** | 39.8% |
| yellow | 1 | 280 | 598 | 3,019 | **3,897** | 2.7% |
| green | 2 | 4,881 | 12,796 | 46,928 | **64,605** | 44.2% |
| redLeft | 3 | 2,375 | 1,092 | 12,734 | **16,201** | 11.1% |
| greenLeft | 4 | 536 | 178 | 2,476 | **3,190** | 2.2% |
| redRight | 5 | 20 | 5 | 0 | **25** | 0.0% |
| greenRight | 6 | 4 | 13 | 0 | **17** | 0.0% |
| **总计** | | **13,476** | **23,069** | **109,475** | **146,020** | |

---

## 各数据集原始标签详情

### S2TLD

所有子集均使用 `Annotations-fix/`（重标注，含方向标签）。

| 子集 | XML 数 | red | yellow | green | redLeft | greenLeft | redRight | greenRight | off |
|------|-------:|----:|-------:|------:|--------:|----------:|---------:|-----------:|----:|
| original (1920×1080) | 1,222 | 945 | 93 | 919 | 738 | 77 | 0 | 0 | 1 |
| normal_1 (1280×720) | 779 | 488 | 35 | 984 | 221 | 87 | 0 | 0 | 110 |
| normal_2 (1280×720) | 3,785 | 3,947 | 152 | 2,978 | 1,416 | 372 | 20 | 4 | 388 |
| **合计** | **5,786** | **5,380** | **280** | **4,881** | **2,375** | **536** | **20** | **4** | **499** |

### BSTLD

| 子集 | 来源 | 原始标签 |
|------|------|----------|
| train (1280×720) | YAML（原生方向标签）| Green(5207), Red(3057), RedLeft(1092), off(726), Yellow(444), GreenLeft(178), GreenStraight(20), GreenRight(13), RedStraight(9), RedRight(5) |
| test (1280×720) | `annotations_fix/` XML | green(7569), red(5321), off(442), yellow(154) |

> BSTLD 训练集 YAML 自带方向标签，无需重标注。测试集重标注仅含 3 类圆灯。

### LISA

| 来源 | 原始标签 |
|------|----------|
| 48 个 CSV 文件 | go(46,723), stop(44,318), stopLeft(12,734), warning(2,669), goLeft(2,476), warningLeft(350), goForward(205) |

> LISA 映射: stop→red, stopLeft→redLeft, go→green, goLeft→greenLeft, goForward→green, warning/warningLeft→yellow

---

## 标签折叠规则

以下标签折叠为基础类别（在转换脚本中实现）：

| 原始标签 | 目标类别 | 原因 |
|----------|----------|------|
| redForward, RedStraight | red (0) | 直行箭头功能等同于圆灯 |
| greenForward, GreenStraight, goForward | green (2) | 直行箭头功能等同于圆灯 |
| yellowLeft, yellowForward, yellowRight, warningLeft | yellow (1) | 黄灯方向数据极少，全部归入黄灯圆灯 |

跳过的标签：

| 标签 | 原因 |
|------|------|
| off | 灭灯状态，非检测目标 |
| wait_on / Wait_on | S2TLD 特有倒计时状态 |
| RedStraightLeft, GreenStraightLeft, GreenStraightRight | BSTLD 复合方向灯，仅 5 个样本 |

---

## 关键问题

### 🔴 严重：右转箭头数据极度匮乏

| 类别 | 样本数 | 来源 |
|------|--------|------|
| redRight (5) | 25 | S2TLD normal_2 (20) + BSTLD train (5) |
| greenRight (6) | 17 | BSTLD train (13) + S2TLD normal_2 (4) |

**42 个样本无法支撑训练。**

**建议**:
- R1 训练保留类别定义，但不依赖其精度
- 自采数据（第 3 周）必须针对右转箭头路口定向采集
- 如自采数据仍不足，考虑 Copy-Paste 合成增强

### 🟡 注意：greenLeft 数据偏少

greenLeft 共 3,190 个样本（2.2%），分布在三个数据集。需关注 R1 模型在此类别的表现。

### 🟡 注意：yellow 持续不平衡

yellow 占比 2.7%（3,897 个样本），各数据集一致偏少。这是所有交通灯数据集的共性问题（黄灯持续时间短）。

### ℹ️ LISA 主导数据集

LISA 贡献了 75% 的标注（109,475 / 146,020）。LISA 的场景分布偏美国道路，与中国交通灯存在视觉差异。R2 训练（自采数据为主）会降低 LISA 的占比。

---

## 当前转换状态

**现有 `yolo_labels/` 目录中的标签是旧版转换结果**（normal_1/normal_2 使用了旧 Annotations 而非 Annotations-fix）。需重新运行转换脚本：

```bash
uv run python scripts/convert_s2tld.py
uv run python scripts/convert_bstld.py
uv run python scripts/convert_lisa.py
uv run python scripts/merge_datasets.py
```
