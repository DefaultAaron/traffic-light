"""Traffic light detection — train, evaluate, and export models."""

import argparse
import math
from pathlib import Path

import yaml
from ultralytics import RTDETR, YOLO

ROOT = Path(__file__).resolve().parent
CONFIGS_DIR = ROOT / "configs"


def _is_model_config(path: Path) -> bool:
    """True iff `path` is a real Ultralytics model config (carries a ``model:`` key).

    B2 review I3 2026-05-09: ablation YAMLs (``copy_paste_balance.yaml``,
    ``data_R2_class_weights.yaml``, ``temporal_hmm.yaml``) have no
    ``model:`` key and would crash ``build_model()`` with a confusing
    ``KeyError`` if a contributor tab-completes them in the train choice
    list. Filter the choice list to the YAMLs ``build_model`` can
    actually consume. Malformed YAMLs are silently excluded (a strictly
    broken config can't be a valid ``train`` argument anyway).
    """
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError:
        return False
    return isinstance(data, dict) and "model" in data


VALID_MODELS = [
    p.stem for p in sorted(CONFIGS_DIR.glob("*.yaml")) if _is_model_config(p)
]

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


def _resolve_trainer_seed(trainer, fallback: int) -> int:
    """Return the seed Ultralytics actually resolved for this trainer.

    Reading from trainer.args.seed is more robust than the closure value: if
    Ultralytics ever overrides config["seed"] during arg-merging (yaml ↔ kwargs
    ↔ defaults), the trainer's view is authoritative. Falls back to the value
    we passed in if the attr is somehow missing.
    """
    args = getattr(trainer, "args", None)
    if args is None:
        return fallback
    return getattr(args, "seed", fallback)


def _write_seed_marker(model, seed: int) -> None:
    """Write SEED.txt next to args.yaml in the run directory.

    Ultralytics already writes args.yaml; SEED.txt makes the seed prominent and
    decouples reproducibility-checking from YAML parsing.
    """
    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir is None:
        return
    actual = _resolve_trainer_seed(trainer, seed)
    Path(save_dir, "SEED.txt").write_text(f"{actual}\n")


def _register_seed_marker(model, seed: int) -> None:
    """Register a callback so SEED.txt is written at training start, not after.

    A run dir created at start survives interrupted training; a marker written
    only at end vanishes whenever training crashes. The seed written reflects
    what Ultralytics actually resolved (trainer.args.seed), not just what we
    passed in.
    """
    def _cb(trainer):
        actual = _resolve_trainer_seed(trainer, seed)
        Path(trainer.save_dir).mkdir(parents=True, exist_ok=True)
        Path(trainer.save_dir, "SEED.txt").write_text(f"{actual}\n")
    model.add_callback("on_pretrain_routine_start", _cb)


def _apply_ablation_overrides(config: dict, args) -> None:
    """Apply --copy-paste / --cls-weight ablation overrides to the train config.

    Both flags default to None at the argparse layer; when supplied they
    override the active model config in-place. The runner-side ablation
    contract (components/copy_paste_balance/runners/ablation.py) consumes
    the resulting trained checkpoints; this function only wires the
    trainer-side plumbing.

    Default-OFF preservation (B2 review I1 2026-05-09): use ``getattr`` so
    callers / subparsers that lack these attrs entirely behave as if the
    flag was None (no-op). Any future subcommand that imports this helper
    without re-declaring the args is therefore safe.

    --copy-paste FLOAT
        Forwarded directly into Ultralytics' built-in `copy_paste=` flag
        (per-image probability of paste op). Range [0, 1]. Plan §3.5
        notes the y-center mask + fliplr=0 lock; those are b-stage
        responsibilities owned by the dataloader callback (the a-stage
        scaffold lives under components/copy_paste_balance/modules/).

        B2 review C1 2026-05-09: argparse's ``type=float`` rejects bools
        at parse time, but this helper is also directly callable by
        tests / programmatic harnesses where ``True`` would silently
        coerce via ``float(True) == 1.0``. Validate bool/int/float +
        finite + range here so the system-boundary discipline matches
        every dataclass __post_init__ in this scaffold tree.

    --cls-weight PATH
        Path to configs/data_R2_class_weights.yaml (or a fitter-emitted
        equivalent). a-stage scaffold validates existence and stashes the
        path on args.cls_weight for the b-stage callback that patches
        the loss module's class-weight buffer to consume. We deliberately
        do NOT inject the weights into Ultralytics' `cls=` scalar gain
        because that flag is a single scalar loss multiplier, not a
        per-class vector — silently routing per-class weights through it
        would corrupt training without any indication.

        B2 review I2 2026-05-09: a-stage refuses the flag with SystemExit
        rather than printing a warning. A stderr warning is silently
        swallowed by the conflictor / CI wrappers and would let a
        c-stage ablation kicked off before b-stage lands silently train
        an unweighted baseline. b-stage replaces the SystemExit with the
        actual callback registration.
    """
    copy_paste = getattr(args, "copy_paste", None)
    cls_weight = getattr(args, "cls_weight", None)
    if copy_paste is not None:
        # bool is an int subclass in Python; argparse type=float rejects it
        # for CLI input but a direct caller can still pass it. Reject before
        # the range check so the diagnostic names the type error clearly.
        if isinstance(copy_paste, bool) or not isinstance(copy_paste, (int, float)):
            raise SystemExit(
                f"--copy-paste must be float; got "
                f"{type(copy_paste).__name__}={copy_paste!r}"
            )
        if not math.isfinite(copy_paste):
            raise SystemExit(f"--copy-paste must be finite; got {copy_paste!r}")
        if not (0.0 <= copy_paste <= 1.0):
            raise SystemExit(f"--copy-paste must be in [0, 1]; got {copy_paste}")
        config["copy_paste"] = float(copy_paste)
    if cls_weight is not None:
        weights_path = Path(cls_weight)
        if not weights_path.exists():
            raise SystemExit(f"--cls-weight file not found: {weights_path}")
        raise SystemExit(
            f"--cls-weight {weights_path} recorded but a-stage cannot apply it. "
            f"The b-stage callback under components/copy_paste_balance/ is "
            f"required to wire per-class weights into Ultralytics' loss module; "
            f"running c-stage training without that callback would silently "
            f"produce an unweighted baseline. Wait for b-stage to land before "
            f"using this flag."
        )


def train(args):
    if args.resume:
        ckpt = Path(args.resume)
        if not ckpt.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {ckpt}")
        model_cls = RTDETR if "rtdetr" in str(ckpt).lower() else YOLO
        model = model_cls(str(ckpt))
        # Resume preserves the original run dir + its existing SEED.txt /
        # args.yaml. Do NOT register or write a SEED marker — args.seed here
        # is the CLI default (0), not the seed the original run used, so
        # overwriting would corrupt the reproducibility metadata.
        model.train(resume=True)
        return

    if not args.model:
        raise SystemExit("train: 'model' is required unless --resume is given")

    config = load_config(args.model)
    if args.epochs:
        config["epochs"] = args.epochs
    if args.batch:
        config["batch"] = args.batch
    if args.device:
        config["device"] = args.device
    if args.imgsz:
        config["imgsz"] = args.imgsz
    config["seed"] = args.seed
    _apply_ablation_overrides(config, args)

    model = build_model(config)
    config.pop("model")  # already loaded, don't pass twice
    _register_seed_marker(model, args.seed)
    model.train(**config)
    _write_seed_marker(model, args.seed)


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
        imgsz=args.imgsz,
        split=args.split,
    )


def export(args):
    """Export a trained model to deployment format.

    When ``--format onnx`` is used on a YOLO26 model, the NMS-free head is
    auto-stripped into ``*_stripped.onnx`` so the result parses on TRT 8.5.2
    (JetPack 5.1). Pass ``--no-strip`` to disable.
    """
    weights = Path(args.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    if "rtdetr" in weights.name:
        model = RTDETR(str(weights))
    else:
        model = YOLO(str(weights))

    output_path = model.export(format=args.format, imgsz=args.imgsz, half=args.half)

    if args.format == "onnx" and "yolo26" in str(weights).lower() and not args.no_strip:
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "strip_yolo26_head", ROOT / "scripts" / "strip_yolo26_head.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        src = Path(output_path)
        dst = src.with_name(f"{src.stem}_stripped_{args.imgsz}.onnx")
        mod.strip_head(str(src), str(dst), num_classes=len(model.names))
        src.unlink()
        print(f"removed intermediate {src}")


def infer(args):
    """Run inference using TRT pipeline (TensorRT or ONNX Runtime)."""
    from inference.demo import run_video
    from inference.tracker import TrackSmoother
    from inference.trt_pipeline import CLASS_NAMES, TRTDetector

    detector = TRTDetector(
        model_path=args.model,
        conf_thresh=args.conf,
        imgsz=args.imgsz,
    )

    tracker = None
    if getattr(args, "track", False):
        tracker = TrackSmoother(
            num_classes=len(CLASS_NAMES),
            alpha=args.alpha,
            track_thresh=args.conf,
            high_thresh=args.high_thresh,
            min_hits=args.min_hits,
            track_buffer=args.track_buffer,
        )

    # Parse source: integer for camera, string for file
    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    run_video(
        source, detector,
        show=not args.no_show,
        save=args.save,
        output_json=args.json,
        tracker=tracker,
    )


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
        if args.imgsz:
            config["imgsz"] = args.imgsz
        config["seed"] = args.seed
        _apply_ablation_overrides(config, args)

        model = build_model(config)
        config.pop("model")
        _register_seed_marker(model, args.seed)
        model.train(**config)
        _write_seed_marker(model, args.seed)


def main():
    parser = argparse.ArgumentParser(description="Traffic Light Detection")
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train a single model")
    p_train.add_argument("model", nargs="?", choices=VALID_MODELS,
                         help="Model config name (omit when using --resume)")
    p_train.add_argument("--epochs", type=int)
    p_train.add_argument("--batch", type=int)
    p_train.add_argument("--imgsz", type=int, help="Override training image size (default: config value)")
    p_train.add_argument("--device", type=str, help="e.g. 0, cpu, mps")
    p_train.add_argument("--resume", type=str, metavar="CKPT",
                         help="Resume from checkpoint (e.g. runs/yolo26n/weights/last.pt)")
    p_train.add_argument("--seed", type=int, default=0,
                         help="Random seed for reproducibility (default: 0)")
    # Ablation knobs (default OFF — preserve current pipeline behavior).
    # Authoritative spec: docs/planning/additional_components_plan.md §三 +
    # components/copy_paste_balance/. Supplying these flags routes the run
    # into the c-stage ablation arm tree (no_aug / cp_only / cp_balanced).
    p_train.add_argument("--copy-paste", type=float, default=None,
                         dest="copy_paste",
                         metavar="FLOAT",
                         help="Override Ultralytics copy_paste= flag (range [0, 1]). "
                              "Default None preserves the model config value.")
    p_train.add_argument("--cls-weight", type=str, default=None,
                         dest="cls_weight",
                         metavar="PATH",
                         help="Path to per-class weights YAML "
                              "(configs/data_R2_class_weights.yaml). a-stage "
                              "scaffold; full integration requires the b-stage "
                              "callback under components/copy_paste_balance/.")
    p_train.set_defaults(func=train)

    # train-all
    p_all = sub.add_parser("train-all", help="Train all (or selected) model variants")
    p_all.add_argument("--models", nargs="+", choices=VALID_MODELS)
    p_all.add_argument("--epochs", type=int)
    p_all.add_argument("--batch", type=int)
    p_all.add_argument("--imgsz", type=int, help="Override training image size (default: config value)")
    p_all.add_argument("--device", type=str)
    p_all.add_argument("--seed", type=int, default=0,
                       help="Random seed for reproducibility (default: 0)")
    p_all.add_argument("--copy-paste", type=float, default=None,
                       dest="copy_paste",
                       metavar="FLOAT",
                       help="Override Ultralytics copy_paste= flag for ALL "
                            "selected variants (range [0, 1]).")
    p_all.add_argument("--cls-weight", type=str, default=None,
                       dest="cls_weight",
                       metavar="PATH",
                       help="Path to per-class weights YAML applied to ALL "
                            "selected variants (a-stage scaffold).")
    p_all.set_defaults(func=train_all)

    # val
    p_val = sub.add_parser("val", help="Validate a trained model")
    p_val.add_argument(
        "weights", help="Path to trained weights (e.g. runs/yolo26n/weights/best.pt)"
    )
    p_val.add_argument("--split", default="val", choices=["val", "test"])
    p_val.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    p_val.set_defaults(func=val)

    # infer
    p_infer = sub.add_parser("infer", help="Run inference with TRT/ONNX pipeline")
    p_infer.add_argument("--source", required=True, help="Video file or camera index (0, 1)")
    p_infer.add_argument("--model", required=True, help="Model path (.engine or .onnx)")
    p_infer.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p_infer.add_argument("--imgsz", type=int, default=1280, help="Input image size (default: 1280 for 8MP cameras)")
    p_infer.add_argument("--no-show", action="store_true", help="Disable display window")
    p_infer.add_argument("--save", type=str, default=None, help="Save output video to path")
    p_infer.add_argument("--json", action="store_true", help="Output per-frame JSON to stdout")
    p_infer.add_argument("--track", action="store_true",
                         help="Enable ByteTrack + EMA class voting")
    p_infer.add_argument("--alpha", type=float, default=0.3,
                         help="EMA smoothing coefficient (default: 0.3)")
    p_infer.add_argument("--min-hits", type=int, default=3,
                         help="Minimum observations before emitting a track (default: 3)")
    p_infer.add_argument("--high-thresh", type=float, default=0.5,
                         help="High/low detection split for two-pass association (default: 0.5)")
    p_infer.add_argument("--track-buffer", type=int, default=30,
                         help="Frames to keep lost tracks alive (default: 30)")
    p_infer.set_defaults(func=infer)

    # export
    p_export = sub.add_parser("export", help="Export model to deployment format")
    p_export.add_argument("weights", help="Path to trained weights")
    p_export.add_argument(
        "--format",
        default="engine",
        choices=["engine", "coreml", "onnx"],
        help="Export format (default: engine/TensorRT)",
    )
    p_export.add_argument("--imgsz", type=int, default=640, help="Input image size (default: 640)")
    p_export.add_argument("--half", action="store_true", help="FP16 quantization")
    p_export.add_argument("--no-strip", action="store_true",
                          help="For YOLO26 ONNX export, skip the NMS-free head strip")
    p_export.set_defaults(func=export)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
