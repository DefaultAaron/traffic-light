"""DEIM-D-FINE ONNX export wrapper. Mirrors upstream DEIM/tools/deployment/
export_onnx.py but bakes the dummy input at the *config-declared*
`eval_spatial_size` instead of the upstream's hardcoded 640.

Why not expose --imgsz?  The DEIM traffic_light configs inherit
`eval_spatial_size: [640, 640]` from base/dfine_hgnetv2.yml; switching to
1280 requires changes to the *training* pipeline (resize ops, base_size,
batch reduction, LR halving) — the deploy graph alone cannot be moved up
without retraining. So the correct contract is "export size = config size";
no override, single source of truth.

Side-effect: writes a sidecar `<ckpt>.imgsz` containing the integer used,
so the surrounding bash wrapper can construct trtexec --shapes without
re-reading the YAML.

Invoked by scripts/export_deim.sh; not intended for direct user use.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True, help="DEIM YAML config path (relative to DEIM/)")
    parser.add_argument("-r", "--resume", required=True, help="checkpoint .pth path (absolute)")
    parser.add_argument("--check", action="store_true", default=True)
    parser.add_argument("--simplify", action="store_true", default=True)
    args = parser.parse_args()

    deim_root = Path(__file__).resolve().parent.parent / "DEIM"
    if not deim_root.is_dir():
        raise SystemExit(f"DEIM submodule not found at {deim_root}")
    sys.path.insert(0, str(deim_root))

    # cwd matters: DEIM YAMLConfig resolves config-relative paths from cwd.
    os.chdir(deim_root)

    import torch
    import torch.nn as nn
    from engine.core import YAMLConfig  # type: ignore[import-not-found]

    cfg = YAMLConfig(args.config, resume=args.resume)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    # Read the spatial size declared in the resolved (post-__include__) YAML.
    # Default 640 mirrors the value baked into base/dfine_hgnetv2.yml so a
    # config that omits the field still exports at the model's training size.
    eval_size = cfg.yaml_cfg.get("eval_spatial_size", [640, 640])
    if not (isinstance(eval_size, (list, tuple)) and len(eval_size) == 2):
        raise SystemExit(f"unexpected eval_spatial_size in {args.config}: {eval_size!r}")
    H, W = int(eval_size[0]), int(eval_size[1])
    if H != W:
        # Pipeline letterboxes to a square — non-square eval would silently
        # mis-feed orig_target_sizes downstream.
        raise SystemExit(f"non-square eval_spatial_size {H}x{W} not supported by trt_pipeline")
    imgsz = H

    checkpoint = torch.load(args.resume, map_location="cpu")
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model()

    # Dummy inputs sized at the config-declared eval size. Batch >1 so
    # torch.onnx.export traces shape-dependent ops without baking batch=1;
    # dynamic_axes below marks dim 0 dynamic.
    data = torch.rand(2, 3, imgsz, imgsz)
    size = torch.tensor([[imgsz, imgsz], [imgsz, imgsz]])
    _ = model(data, size)

    output_file = args.resume.replace(".pth", ".onnx") if args.resume.endswith(".pth") else f"{args.resume}.onnx"
    dynamic_axes = {"images": {0: "N"}, "orig_target_sizes": {0: "N"}}

    torch.onnx.export(
        model,
        (data, size),
        output_file,
        input_names=["images", "orig_target_sizes"],
        output_names=["labels", "boxes", "scores"],
        dynamic_axes=dynamic_axes,
        opset_version=16,
        verbose=False,
        do_constant_folding=True,
    )

    if args.check:
        import onnx
        onnx_model = onnx.load(output_file)
        onnx.checker.check_model(onnx_model)
        print(f"check ok: {output_file}")

    if args.simplify:
        import onnx
        import onnxsim
        input_shapes = {"images": list(data.shape), "orig_target_sizes": list(size.shape)}
        simp, ok = onnxsim.simplify(output_file, test_input_shapes=input_shapes)
        onnx.save(simp, output_file)
        print(f"simplify ok: {ok}")

    # Sidecar file: bash wrapper reads this to build trtexec --shapes
    # without re-parsing YAML. Avoids drift between the two paths.
    sidecar = Path(output_file).with_suffix(Path(output_file).suffix + ".imgsz")
    sidecar.write_text(f"{imgsz}\n")
    print(f"ONNX written: {output_file}  (imgsz={imgsz}, sidecar={sidecar.name})")


if __name__ == "__main__":
    main()
