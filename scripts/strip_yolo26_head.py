"""Strip YOLO26's in-graph NMS-free head, exposing raw [1, 4+nc, N] output.

TRT 8.5.2 (JetPack 5.1.x) cannot parse YOLO26's head due to a
shape-propagation bug in its ONNX parser around `ReduceMax -> TopK`.
Even after rewriting the negative axis to its positive index, the
assertion `axis <= nbDims` fails — TRT sees an inconsistent rank.

The C++ pipeline (`inference/cpp/src/trt_pipeline.cpp`) already decodes
raw anchor output with confidence thresholding. YOLO26 is NMS-free by
training, so no NMS is needed at inference; the in-graph TopK is only
a GPU-side pre-selection optimization.

This script rewrites the exported ONNX so its final output is the head
`Concat` (decoded box xywh || sigmoid class scores), cutting out
ReduceMax, TopK, Gather, and everything else downstream.
"""

import argparse
import sys

import onnx
import onnx_graphsurgeon as gs


def _dim_as_int(d):
    if isinstance(d, int):
        return d
    try:
        return int(d)
    except (TypeError, ValueError):
        return None


def find_head_concat(graph, num_classes):
    expected = 4 + num_classes
    order = {n.name: i for i, n in enumerate(graph.nodes)}
    candidates = []
    for node in graph.nodes:
        if node.op != "Concat":
            continue
        out = node.outputs[0]
        if out.shape is None or len(out.shape) != 3:
            continue
        dims = [_dim_as_int(d) for d in out.shape]
        if expected in dims:
            candidates.append((node, out))
    candidates.sort(key=lambda c: order[c[0].name])
    return candidates[-1] if candidates else None


def strip_head(src: str, dst: str, num_classes: int) -> None:
    """Strip YOLO26's NMS-free head, writing raw [1, 4+nc, N] output to ``dst``."""
    model = onnx.load(src)
    model = onnx.shape_inference.infer_shapes(model)
    graph = gs.import_onnx(model)

    match = find_head_concat(graph, num_classes)
    if match is None:
        raise RuntimeError(
            f"no Concat node with output shape [*, {4+num_classes}, *] found "
            "(expected channel dim 4+num_classes = box xywh + class scores)"
        )

    node, out = match
    print(f"cut at {node.name}: output {out.name} shape {out.shape}")

    before = len(graph.nodes)
    graph.outputs = [out]
    graph.cleanup().toposort()
    onnx.save(gs.export_onnx(graph), dst)
    print(f"{before} -> {len(graph.nodes)} nodes, wrote {dst}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", help="input .onnx (from Ultralytics export)")
    ap.add_argument("dst", help="output .onnx")
    ap.add_argument("--num-classes", type=int, default=7,
                    help="number of detection classes (default: 7)")
    args = ap.parse_args()

    try:
        strip_head(args.src, args.dst, args.num_classes)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
