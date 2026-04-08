# Traffic Light Recognition

Traffic light detection and state classification (Red/Yellow/Green) for autonomous driving.

## Project Structure

```
traffic-light/
├── pyproject.toml
├── .python-version
├── main.py                     # Entry point - train, eval, export commands
├── configs/                    # Training configs per model
│   ├── yolo26n.yaml
│   ├── yolo26s.yaml
│   ├── yolo11n.yaml
│   ├── yolo11s.yaml
│   └── rtdetr-r18.yaml
├── weights/                    # Pretrained weights (gitignored)
│   ├── yolo26n.pt
│   ├── yolo26s.pt
│   ├── yolo11n.pt
│   ├── yolo11s.pt
│   └── rtdetr-r18.pt
├── data/
│   ├── traffic_light.yaml      # Ultralytics dataset config
│   ├── raw/                    # Original downloaded datasets (gitignored)
│   │   ├── s2tld/
│   │   ├── bstld/
│   │   └── lisa/
│   └── merged/                 # Unified YOLO format after conversion (gitignored)
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
├── scripts/                    # Dataset conversion and utilities
│   ├── convert_s2tld.py
│   ├── convert_bstld.py
│   ├── convert_lisa.py
│   └── merge_datasets.py
├── runs/                       # Training outputs (gitignored)
└── notebooks/                  # Experiment analysis
    └── compare_results.ipynb
```

## Candidate Models

| Model | Params | COCO mAP | NMS-Free |
|-------|--------|----------|----------|
| YOLO26-n/s | ~2.5M / ~9M | 40.9% / 47.5% | Yes |
| YOLO11-n/s | ~2.6M / ~9M | 39.5% / 43.5% | No |
| RT-DETR-R18 | ~18M | 46.5% | Yes |

## Datasets

| Dataset | Images | Annotations | License |
|---------|--------|-------------|---------|
| S2TLD | 5.8K | 14K | MIT |
| BSTLD | 13K | 24K | Non-commercial |
| LISA | 43K | 113K | CC BY-NC-SA |

Classes: `red`, `yellow`, `green`
