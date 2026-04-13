"""Traffic light detection — train, evaluate, and export models."""

import argparse
from pathlib import Path

import yaml
from ultralytics import RTDETR, YOLO

ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = ROOT / "configs"
VALID_MODELS = [p.stem for p in sorted(CONFIGS_DIR.glob("*.yaml"))]

RTDETR_MODELS = {"rtdetr-l"}


def load_config(model_name: str) -> dict:
    path = CONFIGS_DIR / f"{model_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def build_model(config: dict):
    """Instantiate the correct Ultralytics model class."""
    model_path = config["model"]
    if any(model_path.startswith(f"weights/{name}") for name in RTDETR_MODELS):
        return RTDETR(model_path)
    else:
        return YOLO(model_path)


def train(args):
    config = load_config(args.model)
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch:
        config["batch"] = args.batch
    if args.device:
        config["device"] = args.device

    model = build_model(config)
    model_key = config.pop("model")  # already loaded, don't pass twice
    model.train(**config)


def val(args):
    """Validate a trained model."""
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    if "rtdetr" in weights.name:
        model = RTDETR(str(weights))
    else:
        model = YOLO(str(weights))

    model.val(
        data="data/traffic_light.yaml",
        imgsz=640,
        split=args.split,
    )


def export(args):
    """Export a trained model to deployment format."""
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    if "rtdetr" in weights.name:
        model = RTDETR(str(weights))
    else:
        model = YOLO(str(weights))

    model.export(format=args.format, imgsz=640, half=args.half)


def train_all(args):
    """Train all model variants sequentially."""
    models = args.models if args.models else VALID_MODELS
    for name in models:
        print(f"\n{'='*60}")
        print(f"Training {name}")
        print(f"{'='*60}\n")
        config = load_config(name)
        if args.epochs:
            config["epochs"] = args.epochs
        if args.batch:
            config["batch"] = args.batch
        if args.device:
            config["device"] = args.device

        model = build_model(config)
        config.pop("model")
        model.train(**config)


def main():
    parser = argparse.ArgumentParser(description="Traffic Light Detection")
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train a single model")
    p_train.add_argument("model", choices=VALID_MODELS, help="Model config name")
    p_train.add_argument("--epochs", type=int)
    p_train.add_argument("--batch", type=int)
    p_train.add_argument("--device", type=str, help="e.g. 0, cpu, mps")
    p_train.set_defaults(func=train)

    # train-all
    p_all = sub.add_parser("train-all", help="Train all (or selected) model variants")
    p_all.add_argument("--models", nargs="+", choices=VALID_MODELS)
    p_all.add_argument("--epochs", type=int)
    p_all.add_argument("--batch", type=int)
    p_all.add_argument("--device", type=str)
    p_all.set_defaults(func=train_all)

    # val
    p_val = sub.add_parser("val", help="Validate a trained model")
    p_val.add_argument(
        "weights", help="Path to trained weights (e.g. runs/yolo26n/weights/best.pt)"
    )
    p_val.add_argument("--split", default="val", choices=["val", "test"])
    p_val.set_defaults(func=val)

    # export
    p_export = sub.add_parser("export", help="Export model to deployment format")
    p_export.add_argument("weights", help="Path to trained weights")
    p_export.add_argument(
        "--format",
        default="engine",
        choices=["engine", "coreml", "onnx"],
        help="Export format (default: engine/TensorRT)",
    )
    p_export.add_argument("--half", action="store_true", help="FP16 quantization")
    p_export.set_defaults(func=export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
