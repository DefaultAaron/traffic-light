# Traffic Light Recognition

Traffic light detection and state classification (Red/Yellow/Green) for autonomous driving.

## Project Structure

```
traffic-light/
├── pyproject.toml
├── .python-version
├── main.py                     # Entry point - train, eval, export commands
├── configs/                    # Training configs per model
├── weights/                    # Pretrained weights (gitignored)
├── data/
│   ├── traffic_light.yaml      # Ultralytics dataset config
│   ├── raw/                    # Original downloaded datasets (gitignored)
│   └── merged/                 # Unified YOLO format after conversion (gitignored)
│       ├── images/
│       └── labels/
├── scripts/                    # Dataset conversion and utilities
└── runs/                       # Training outputs (gitignored)
```

## Candidate Models

| Model | Params | COCO mAP | NMS-Free |
|-------|--------|----------|----------|
| YOLO26-n/s | ~2.5M / ~9M | 40.9% / 47.5% | Yes |
| YOLO11-n/s | ~2.6M / ~9M | 39.5% / 43.5% | No |
| RT-DETR-L | ~32M | ~53.0% | Yes |

## Datasets

| Dataset | Images | Annotations | License |
|---------|--------|-------------|---------|
| S2TLD | 5.8K | 14K | MIT |
| BSTLD | 13K | 24K | Non-commercial |
| LISA | 43K | 113K | CC BY-NC-SA |

Classes: `red`, `yellow`, `green`

## Usage

### Train a single model

```bash
python main.py train yolo26n
python main.py train yolo11s --epochs 50 --device 0
python main.py train yolo26n --epochs 3 --device mps  # quick sanity check on M4 Pro
```

### Train all model variants

```bash
python main.py train-all                          # all 5 variants sequentially
python main.py train-all --models yolo26n yolo11n  # selected models only
python main.py train-all --device 0 --batch 16
```

### Validate a trained model

```bash
python main.py val runs/yolo26n/weights/best.pt
python main.py val runs/yolo26n/weights/best.pt --split test
```

### Export to deployment format

```bash
python main.py export runs/yolo26n/weights/best.pt --format engine --half  # TensorRT FP16 (Orin)
python main.py export runs/yolo26n/weights/best.pt --format coreml         # CoreML (M4 Pro)
python main.py export runs/yolo26n/weights/best.pt --format onnx           # ONNX (portable)
```
