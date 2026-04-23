# 数据转换计划

> **文档版本**：本文记录 P1 3 类转换流程。R1 已切至 7 类（见 [`class_distribution.md`](./class_distribution.md)），R2 将切至 10–14 类；类别映射在各 `convert_*.py` 中更新，本文的映射表以 P1 3 类为基准，**当前 R1 脚本映射以源码为准**。

## 概述

将三个原始数据集（S2TLD、BSTLD、LISA）转换为统一的 YOLO 格式，合并后进行 80/20 分层划分；DEIM 额外用 `scripts/yolo_to_coco.py` 从合并目录生成 COCO JSON（图片不重复）。

共 5 个脚本：3 个转换脚本（S2TLD / BSTLD / LISA）+ 1 个合并脚本 + 1 个 COCO 导出脚本。

---

## 一、`scripts/convert_s2tld.py` — Pascal VOC XML → YOLO txt

**数据源**：`data/raw/S2TLD/`
- 图片：`JPEGImages/*.jpg`（1,222 张，1920×1080）
- 标注：`Annotations/*.xml`（Pascal VOC 格式）
- 文件名包含空格（如 `2020-04-04 11_10_21.323753353`）

**类别映射**（第一阶段）：

| 原始类别 | → 类别 ID |
|----------|-----------|
| `red` | 0 |
| `yellow` | 1 |
| `green` | 2 |
| `off`, `wait_on` | 跳过 |

**转换逻辑**：
1. 解析 XML → 提取 `<name>`、`<bndbox>`
2. `xmin/ymin/xmax/ymax`（绝对像素）→ `cx cy w h`（归一化，除以图片宽高）
3. 每张图片输出一个 `.txt` 文件，文件名与图片对应

**输出**：`data/raw/S2TLD/yolo_labels/*.txt`

---

## 二、`scripts/convert_bstld.py` — YAML → YOLO txt

**数据源**：`data/raw/BSTLD/`
- 训练集：`train/train.yaml`（5,093 张，10,756 个框）+ `train/rgb/train/*.png`
- 测试集：`test/test.yaml`（8,334 张，13,486 个框）+ `test/rgb/test/*.png`
- 分辨率：1280×720

**类别映射**（第一阶段）：

| 原始类别 | → 类别 ID |
|----------|-----------|
| `Red`, `RedLeft`, `RedRight`, `RedStraight` | 0 |
| `Yellow` | 1 |
| `Green`, `GreenLeft`, `GreenRight`, `GreenStraight` | 2 |
| `off` | 跳过 |
| `RedStraightLeft`, `GreenStraightLeft`, `GreenStraightRight` | 跳过（组合多方向标签，合计仅5条，中国路况罕见） |

**注意事项**：
- 训练集 YAML 中的路径为相对路径：`./rgb/train/.../*.png`
- 测试集 YAML 中的路径为 Bosch 内部绝对路径（`/net/pal-soc1.us.bosch.com/...`），需提取文件名后映射到 `test/rgb/test/` 目录下的实际文件

**输出**：`data/raw/BSTLD/yolo_labels/*.txt`

---

## 三、`scripts/convert_lisa.py` — CSV → YOLO txt

**数据源**：`data/raw/LISA/`
- 22 个 CSV 文件，分布在 6 个标注文件夹中（~109K 条标注）
- 图片分散在嵌套目录 `<folder>/<folder>/frames/` 或 `<folder>/<folder>/<clip>/frames/`

**CSV 格式**（分号分隔）：
```
Filename;Annotation tag;Upper left corner X;Upper left corner Y;Lower right corner X;Lower right corner Y;...
```

**路径映射** — CSV 中的路径 vs 实际路径：

| CSV 路径前缀 | 实际路径 |
|-------------|---------|
| `dayTraining/dayClip1--00000.jpg` | `dayTrain/dayTrain/dayClip1/frames/dayClip1--00000.jpg` |
| `nightTraining/nightClip1--00000.jpg` | `nightTrain/nightTrain/nightClip1/frames/nightClip1--00000.jpg` |
| `dayTest/daySequence1--00000.jpg` | `daySequence1/daySequence1/frames/daySequence1--00000.jpg` |
| `dayTest/daySequence2--00000.jpg` | `daySequence2/daySequence2/frames/daySequence2--00000.jpg` |
| `nightTest/nightSequence1--00000.jpg` | `nightSequence1/nightSequence1/frames/nightSequence1--00000.jpg` |
| `nightTest/nightSequence2--00000.jpg` | `nightSequence2/nightSequence2/frames/nightSequence2--00000.jpg` |

**类别映射**（第一阶段）：

| 原始类别 | → 类别 ID |
|----------|-----------|
| `stop`, `stopLeft` | 0（红灯） |
| `warning`, `warningLeft` | 1（黄灯） |
| `go`, `goLeft`, `goForward` | 2（绿灯） |

**注意事项**：
- 图片分辨率不统一，需逐图读取尺寸进行归一化
- 跳过 `sample-dayClip6`、`sample-nightClip1` 文件夹（无对应标注）

**输出**：`data/raw/LISA/yolo_labels/*.txt`

---

## 四、`scripts/merge_datasets.py` — 合并 + 划分

1. 收集三个数据集转换后的所有（图片, 标注）对
2. 复制图片至 `data/merged/images/{train,val}/`
3. 复制标注至 `data/merged/labels/{train,val}/`
4. **80/20 分层划分**：按类别分布进行分层随机采样
5. 文件名添加数据集前缀避免冲突：`s2tld_`、`bstld_`、`lisa_`
6. 输出统计摘要：总图片数、逐类标注数、训练/验证集划分

---

## 五、多阶段数据管理

### 目录结构

```
data/merged/
├── images/{train,val}/          # 图片（两阶段共用，只存一份）
├── labels_phase1/{train,val}/   # 第一阶段标注（3 类）
├── labels_phase2/{train,val}/   # 第二阶段标注（7 类）
└── labels → labels_phase1       # 软链接，切换阶段时重定向
```

### 设计原则

- **图片只存一份**：两阶段使用完全相同的图片，仅标注不同，避免磁盘空间浪费
- **软链接切换阶段**：Ultralytics 自动将图片路径中的 `images/` 替换为 `labels/` 来查找标注文件，因此通过软链接 `labels/` 指向对应阶段即可
  - 切换至第一阶段：`ln -sfn labels_phase1 data/merged/labels`
  - 切换至第二阶段：`ln -sfn labels_phase2 data/merged/labels`
- **数据集配置文件无需修改**：`traffic_light.yaml` 始终引用 `images/` 和 `labels/`，类别名称列表按阶段维护两份

### 分层划分策略

- **按 7 类（第二阶段）进行分层划分**：7 类是 3 类的细分子集，按 7 类分层后 3 类自动平衡（反之不成立）
- **当前阶段**：第二阶段标注尚未就绪，暂按 3 类分层。第二阶段标注完成后，重新执行一次合并脚本，按 7 类分层，同时输出两套标注目录
- **训练/验证集划分在两阶段间保持一致**：同一张图片在两阶段中始终属于同一集合（训练或验证），确保评估公平性

---

## 六、标注数量汇总

R1 7 类分布见 [`class_distribution.md`](./class_distribution.md)；R2 10–14 类的新增类别样本统计见 [`../reports/phase_2_round_1_report.md`](../reports/phase_2_round_1_report.md) §R2 范围扩展。

**P1 3 类基线汇总**（已归档，供历史对照）：

| 数据集 | 红灯 | 黄灯 | 绿灯 | 合计 | 图片数 |
|--------|------|------|------|------|--------|
| S2TLD | 1,235 | 75 | 816 | 2,126 | 1,222 |
| BSTLD 训练集 | 4,164 | 444 | 5,422 | 10,030 | 5,093 |
| BSTLD 测试集 | 5,321 | 154 | 7,569 | 13,044 | 8,334 |
| LISA | 57,052 | 3,019 | 49,404 | 109,475 | ~43,000 |
| **合计** | **~67,772** | **~3,692** | **~63,211** | **~134,675** | **~57,000** |

**持续问题**：黄灯样本严重不足（P1 ~2.7%，R1 仍为 2.7%），R2 过采样 2–3× 已列入训练计划。
