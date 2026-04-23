# 标签分布分析（R1 7 类 / R2 9 类基线）

**日期**: 2026-04-16（R1 7 类初版）、2026-04-23（追加 R2 9 类）
**数据源**: S2TLD（全部子集已重标注）、BSTLD（训练集 YAML 含方向标签）、LISA

> **R2 范围更新（2026-04-21）**：PM 确认 R2 将扩展为 **10–14 类联合模型**（9–12 类交通灯 + 1–2 类道路栏杆）。
> - 交通灯确认新增 `forwardGreen`、`forwardRed`（总计至少 9 类），另可再新增 ≤3 类由 PM 最终敲定
> - 栏杆 MVP 为单类 `barrier`，数据充分则升级为 `armOn` / `armOff`
>
> 本文保留 **R1 7 类** 表作为历史基线，追加 **R2 9 类** 表反映直行箭头从圆灯折叠中回收后的分布。栏杆与 PM 待定类别的样本规划见 [`../reports/phase_2_round_1_report.md`](../reports/phase_2_round_1_report.md) §"R2 范围扩展（PM 确认事项）"。

---

## R1 7 类分布

R1 转换脚本将直行箭头折叠进圆灯（`RedStraight` / `redForward` → `red`；`GreenStraight` / `greenForward` / `goForward` → `green`）。

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

## R2 9 类分布（R1 + `forwardRed` / `forwardGreen`）

R2 从 R1 圆灯中回收直行箭头为独立类 — `RedStraight` / `redForward` → `forwardRed`，`GreenStraight` / `greenForward` / `goForward` → `forwardGreen`。其余 7 类数量 = R1 表减去回收的直行样本，总标注数不变。

| 类别 | ID | S2TLD | BSTLD | LISA | 总计 | 占比 |
|------|-----|------:|------:|-----:|-----:|-----:|
| red | 0 | 5,380 | 8,378 | 44,318 | **58,076** | 39.8% |
| yellow | 1 | 280 | 598 | 3,019 | **3,897** | 2.7% |
| green | 2 | 4,881 | 12,776 | 46,723 | **64,380** | 44.1% |
| redLeft | 3 | 2,375 | 1,092 | 12,734 | **16,201** | 11.1% |
| greenLeft | 4 | 536 | 178 | 2,476 | **3,190** | 2.2% |
| redRight | 5 | 20 | 5 | 0 | **25** | 0.0% |
| greenRight | 6 | 4 | 13 | 0 | **17** | 0.0% |
| **forwardRed** | 7 | 0 | 9 | 0 | **9** | 0.006% |
| **forwardGreen** | 8 | 0 | 20 | 205 | **225** | 0.15% |
| **总计** | | **13,476** | **23,069** | **109,475** | **146,020** | |

**回收来源**：
- `forwardRed` = BSTLD train `RedStraight` (9)；S2TLD annotations-fix 与 LISA 均无对应直行红灯标签
- `forwardGreen` = BSTLD train `GreenStraight` (20) + LISA `goForward` (205)；S2TLD annotations-fix 无 `greenForward` 标签

**数据缺口**：
- `forwardRed` 仅 9 条 — 低于 `redRight` / `greenRight` 之外所有类别，**无法独立训练**。
- `forwardGreen` 225 条 — 仍严重不足（< `yellow` 3,897 的 6%），模型大概率在此类收敛失败。
- 两者均需 R2 现场采集 / BSTLD-LISA 人工复扫补齐。优先级高于 `redRight` / `greenRight`（后者在 R1 已证实无法训练但暂保留类别定义）。

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

## 标签折叠规则（R1）

以下标签折叠为基础类别（在 R1 转换脚本中实现）：

| 原始标签 | 目标类别 | 原因 |
|----------|----------|------|
| redForward, RedStraight | red (0) | R1 决策：直行箭头功能等同于圆灯 |
| greenForward, GreenStraight, goForward | green (2) | R1 决策：直行箭头功能等同于圆灯 |
| yellowLeft, yellowForward, yellowRight, warningLeft | yellow (1) | 黄灯方向数据极少，全部归入黄灯圆灯 |

跳过的标签：

| 标签 | 原因 |
|------|------|
| off | 灭灯状态，非检测目标 |
| wait_on / Wait_on | S2TLD 特有倒计时状态 |
| RedStraightLeft, GreenStraightLeft, GreenStraightRight | BSTLD 复合方向灯，仅 5 个样本 |

> **R2 变化**：`redForward / RedStraight / goForward / greenForward / GreenStraight` 将从圆灯折叠中恢复，分别映射为新的 `forwardRed` / `forwardGreen` 类。该折叠在 R2 转换脚本切版时修改。若 PM 将 `yellowForward` / `warningLeft` 纳入 R2 新增 3 类额度，则对应折叠一并取消。

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
