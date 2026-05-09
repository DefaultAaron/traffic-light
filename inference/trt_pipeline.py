"""TensorRT inference pipeline for traffic light detection.

Supports two backends:
  - TensorRT (production, Orin): requires `tensorrt` and `pycuda`
  - ONNX Runtime (development): requires `onnxruntime` or `onnxruntime-gpu`

Supports two detector architectures, auto-detected at construction time:
  - YOLO26 (1 input "images", 1 output (1, N, 4+nc) or (1, 4+nc, N), NMS-free)
  - DEIM-D-FINE (2 inputs "images"+"orig_target_sizes", 3 outputs
    "labels"+"boxes"+"scores"; deploy postprocessor bakes top-K + sigmoid
    inside the graph)

The public surface is identical for both — `TRTDetector(model_path).detect(frame)`
returns `list[Detection]` regardless of architecture, so demo.py / tracker /
run_demos.sh do not need to know which model they are running.

Usage:
    detector = TRTDetector("best.engine")  # YOLO or DEIM, auto-detected
    detections = detector.detect(frame)

Thread safety: TRTDetector is NOT thread-safe. It holds a single CUDA
stream and pre-allocated pinned host buffers shared across calls. Construct
and call `detect()` on the same thread. Use one detector per worker thread.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = {
    0: "red",
    1: "yellow",
    2: "green",
    3: "redLeft",
    4: "greenLeft",
    5: "redRight",
    6: "greenRight",
}


class Arch(Enum):
    YOLO = "yolo"
    DEIM = "deim"


@dataclass
class Detection:
    """Single detection result."""

    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.class_id, f"cls_{self.class_id}")

    def to_dict(self) -> dict:
        """Serialize to dict (JSON-friendly)."""
        d = asdict(self)
        d["class_name"] = self.class_name
        return d

    def to_ros_msg(self):
        """Convert to vision_msgs/Detection2D message.

        Requires: ros2 vision_msgs package (optional dependency).
        """
        from vision_msgs.msg import (
            BoundingBox2D,
            Detection2D,
            ObjectHypothesisWithPose,
        )

        det = Detection2D()

        # Bounding box (center + size format)
        det.bbox = BoundingBox2D()
        det.bbox.center.position.x = (self.x1 + self.x2) / 2.0
        det.bbox.center.position.y = (self.y1 + self.y2) / 2.0
        det.bbox.size_x = self.x2 - self.x1
        det.bbox.size_y = self.y2 - self.y1

        # Classification result
        hyp = ObjectHypothesisWithPose()
        hyp.hypothesis.class_id = self.class_name
        hyp.hypothesis.score = self.confidence
        det.results.append(hyp)

        return det


def _letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Resize image with letterboxing (maintain aspect ratio, pad with gray).

    Returns:
        (letterboxed_image, scale_ratio, (pad_w, pad_h))
    """
    h, w = img.shape[:2]
    # float32 cast + half-away-from-zero rounding match the C++ pipeline's
    # `static_cast<int>(std::round(...))`. Python's built-in round() does
    # banker's rounding (half-to-even), which diverges on .5 cases.
    def _round_haz(x: float) -> int:
        return int(np.floor(x + 0.5)) if x >= 0 else -int(np.floor(-x + 0.5))

    r = float(np.float32(min(new_shape[0] / h, new_shape[1] / w)))
    new_unpad = (_round_haz(w * r), _round_haz(h * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = _round_haz(dh - 0.1), _round_haz(dh + 0.1)
    left, right = _round_haz(dw - 0.1), _round_haz(dw + 0.1)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (dw, dh)


def _preprocess(img: np.ndarray, imgsz: int = 640) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Preprocess frame for inference.

    BGR (OpenCV) → RGB → letterbox → normalize → CHW → float32 → batch dim.
    Returns (input_tensor, scale, (pad_w, pad_h)).

    Both YOLO26 and DEIM-D-FINE consume normalized float32 NCHW in [0, 1].
    Padding is 114-gray (Ultralytics + DEIM training-transform convention;
    the zero-pad in DEIM/tools/inference/onnx_inf.py is demo-only).
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lb, scale, pad = _letterbox(img_rgb, (imgsz, imgsz))
    blob = img_lb.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)  # HWC → CHW
    blob = np.expand_dims(blob, 0)  # add batch dim
    blob = np.ascontiguousarray(blob)
    return blob, scale, pad


def _validate_yolo_output_shape(shape, source: str) -> None:
    """Construction-time YOLO output shape gate.

    Rank must be 2 or 3, batch (rank-3) must be 1, one axis must equal
    4 + nc, and the (4+nc, 4+nc) square is rejected as orientation-ambiguous.
    Non-int dims (ORT dynamic axes) defer to the runtime check.
    """
    expected_row = 4 + len(CLASS_NAMES)
    shape = list(shape)
    nd = len(shape)
    if nd != 2 and nd != 3:
        raise ValueError(
            f"YOLO {source} rank is {nd} (expected 2 or 3); shape={tuple(shape)}"
        )

    def is_concrete(d) -> bool:
        return isinstance(d, int) and d > 0

    if nd == 3 and is_concrete(shape[0]) and shape[0] != 1:
        raise ValueError(
            f"YOLO {source} batch dim is {shape[0]} (expected 1); "
            f"shape={tuple(shape)}"
        )

    if all(is_concrete(d) for d in shape):
        if expected_row not in shape:
            raise ValueError(
                f"YOLO {source} shape {tuple(shape)} has no "
                f"{expected_row}-wide axis (nc={len(CLASS_NAMES)})"
            )
        if nd == 3 and shape[1] == expected_row and shape[2] == expected_row:
            raise ValueError(
                f"YOLO {source} shape (1, {expected_row}, {expected_row}) "
                "is orientation-ambiguous"
            )
        if nd == 2 and shape[0] == expected_row and shape[1] == expected_row:
            raise ValueError(
                f"YOLO {source} shape ({expected_row}, {expected_row}) "
                "is orientation-ambiguous"
            )


def _detect_arch(input_names: list[str], output_names: list[str]) -> Arch:
    """Pick architecture by tensor-name signature.

    DEIM-D-FINE deploy graph: inputs {"images", "orig_target_sizes"},
    outputs {"labels", "boxes", "scores"}. Anything else falls through to
    YOLO (single image input, single concat output).
    """
    in_set = {n.lower() for n in input_names}
    out_set = {n.lower() for n in output_names}
    if "orig_target_sizes" in in_set and {"labels", "boxes", "scores"}.issubset(out_set):
        return Arch.DEIM
    return Arch.YOLO


class _TRTBackend:
    """TensorRT engine backend using pycuda.

    Returns the full output dict keyed by tensor name so DEIM (3 outputs) and
    YOLO (1 output) share a single backend implementation.
    """

    def __init__(self, engine_path: str, imgsz: int | None = None):
        import pycuda.autoinit  # noqa: F401 — initializes CUDA context
        import pycuda.driver as cuda
        import tensorrt as trt

        # The Pass-1 / Pass-2 loops below use TRT 10 APIs (num_io_tensors,
        # set_tensor_address, execute_async_v3); pre-TRT-10 would AttributeError
        # mid-construction. Jetson / JetPack 5.1 (TRT 8.5) deployments use the
        # C++ pipeline.
        trt_version = tuple(int(p) for p in trt.__version__.split(".")[:2])
        if trt_version < (10, 0):
            raise RuntimeError(
                f"TensorRT {trt.__version__} is too old for the Python pipeline "
                "(requires >= 10.0). For Jetson / TRT 8.x deployment, use the "
                "C++ pipeline at inference/cpp/."
            )

        # Sentinels for _close() on partial-init failure.
        self._cuda = cuda
        self.engine = None
        self.context = None
        self.stream: object | None = None
        self.inputs: list[dict] = []
        self.outputs: list[dict] = []

        try:
            logger = trt.Logger(trt.Logger.WARNING)
            with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            if self.engine is None:
                raise RuntimeError(f"failed to deserialize engine: {engine_path}")
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError("failed to create execution context")
            self.stream = cuda.Stream()

            # Pass 1: bind dynamic input shapes — dynamic spatial dims of the
            # image input go to imgsz, any other dynamic dim goes to 1.
            # Without this, a spatially-dynamic engine allocates 1×3×1×1 and
            # crashes on the first feed.
            image_name = self._identify_image_input_name()
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                    continue
                engine_shape = list(self.engine.get_tensor_shape(name))
                shape = list(engine_shape)
                is_image = (name == image_name)
                for k, d in enumerate(shape):
                    if d < 0:
                        if is_image and imgsz is not None and len(shape) == 4 and k >= 2:
                            shape[k] = int(imgsz)
                        else:
                            shape[k] = 1
                # set_input_shape returns False on shape outside the engine's
                # optimization profile.
                if not self.context.set_input_shape(name, shape):
                    raise RuntimeError(
                        f"set_input_shape({name}, {shape}) failed — likely "
                        "outside the engine's optimization profile"
                    )

            # Pass 2: allocate per-tensor buffers. Append incrementally so
            # _close() sees the correct list on mid-loop failure.
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                engine_shape = tuple(self.engine.get_tensor_shape(name))  # keeps -1 for dynamic
                shape = list(self.context.get_tensor_shape(name))
                for k, d in enumerate(shape):
                    if d < 0:
                        shape[k] = 1
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                size = max(1, int(np.prod(shape)))
                host_mem = cuda.pagelocked_empty(size, dtype)
                try:
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                except Exception:
                    del host_mem
                    raise
                buf = {
                    "host": host_mem,
                    "device": device_mem,
                    "shape": tuple(shape),
                    "engine_shape": engine_shape,
                    "dtype": dtype,
                    "name": name,
                }
                if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    self.inputs.append(buf)
                else:
                    self.outputs.append(buf)

            # Image input must be NCHW, batch=1, square — `[imgsz, imgsz]`
            # for orig_target_sizes assumes a square letterbox. The "images"
            # name is preferred but the rank-4/C=3 predicate is what binds.
            def _looks_like_image(b):
                return len(b["shape"]) == 4 and b["shape"][1] == 3

            image_buf = next(
                (b for b in self.inputs if b["name"] == "images" and _looks_like_image(b)),
                None,
            )
            if image_buf is None:
                image_buf = next(
                    (b for b in self.inputs if _looks_like_image(b)),
                    None,
                )
            if image_buf is None:
                raise ValueError(
                    "engine has no NCHW 3-channel image input "
                    "(expected a tensor named 'images' or a 4-D input with C=3)"
                )
            ish = image_buf["shape"]
            if ish[0] != 1:
                raise ValueError(
                    f"engine image input has batch={ish[0]}; only batch=1 is supported"
                )
            # Reject the post-binding concrete shape directly. Using only
            # engine_shape misses mixed dynamic/static cases like (1, 3, -1, 640)
            # where one spatial dim was dynamic at export and bound to imgsz
            # later — the concrete bound shape is still rectangular. C++ rejects
            # at this same point.
            if len(ish) == 4 and ish[2] != ish[3]:
                raise ValueError(
                    f"engine image input is rectangular ({ish[3]}x{ish[2]}); "
                    "only square inputs are supported (orig_target_sizes assumes square)"
                )
            img_dtype = np.dtype(image_buf["dtype"])
            if img_dtype.itemsize not in (2, 4) or img_dtype.kind != "f":
                raise ValueError(
                    f"engine image input has unsupported dtype {img_dtype}; "
                    "only float32 or float16 are supported"
                )
            self._image_name: str = image_buf["name"]

            # DEIM dtype gates at construction. Detect by tensor-name
            # signature — the arch_ flag itself is set later in TRTDetector.
            in_set = {b["name"] for b in self.inputs}
            out_set = {b["name"] for b in self.outputs}
            is_deim = (
                "orig_target_sizes" in in_set
                and {"labels", "boxes", "scores"}.issubset(out_set)
            )
            # Signed int only; unsigned would mis-decode DEIM "no object"
            # padding (negative sentinel).
            def _is_int_cls(b):
                d = np.dtype(b["dtype"])
                return d.kind == "i" and d.itemsize in (4, 8)

            def _is_float_hp(b):
                d = np.dtype(b["dtype"])
                return d.kind == "f" and d.itemsize in (2, 4)

            if is_deim:
                labels = next(b for b in self.outputs if b["name"] == "labels")
                boxes = next(b for b in self.outputs if b["name"] == "boxes")
                scores = next(b for b in self.outputs if b["name"] == "scores")
                ots = next(b for b in self.inputs if b["name"] == "orig_target_sizes")
                if not _is_int_cls(labels):
                    raise ValueError(
                        f"DEIM 'labels' has unsupported dtype "
                        f"{np.dtype(labels['dtype'])} (expected int64 or int32)"
                    )
                if not _is_float_hp(boxes):
                    raise ValueError(
                        f"DEIM 'boxes' has unsupported dtype "
                        f"{np.dtype(boxes['dtype'])} (expected float32 or float16)"
                    )
                if not _is_float_hp(scores):
                    raise ValueError(
                        f"DEIM 'scores' has unsupported dtype "
                        f"{np.dtype(scores['dtype'])} (expected float32 or float16)"
                    )
                if not _is_int_cls(ots):
                    raise ValueError(
                        f"DEIM 'orig_target_sizes' has unsupported dtype "
                        f"{np.dtype(ots['dtype'])} (expected int64 or int32)"
                    )
                # Shape gate: detect() writes np.array([[imgsz, imgsz]]) so the
                # buffer must be exactly (1, 2). Catches the overflow at
                # construction instead of at np.copyto.
                ots_shape = tuple(ots["shape"])
                if ots_shape != (1, 2):
                    raise ValueError(
                        f"DEIM 'orig_target_sizes' has shape {ots_shape}; expected (1, 2)"
                    )
                # Top-K shape contract: labels=(1, K), boxes=(1, K, 4),
                # scores=(1, K), with K consistent across all three.
                l_shape = labels["shape"]
                b_shape = boxes["shape"]
                s_shape = scores["shape"]
                if len(l_shape) != 2 or l_shape[0] != 1:
                    raise ValueError(
                        f"DEIM 'labels' has shape {l_shape}; expected (1, K)"
                    )
                if len(s_shape) != 2 or s_shape[0] != 1:
                    raise ValueError(
                        f"DEIM 'scores' has shape {s_shape}; expected (1, K)"
                    )
                if len(b_shape) != 3 or b_shape[0] != 1 or b_shape[2] != 4:
                    raise ValueError(
                        f"DEIM 'boxes' has shape {b_shape}; expected (1, K, 4)"
                    )
                if l_shape[1] != s_shape[1] or l_shape[1] != b_shape[1]:
                    raise ValueError(
                        f"DEIM K mismatch: labels K={l_shape[1]} "
                        f"scores K={s_shape[1]} boxes K={b_shape[1]}"
                    )
            else:
                # YOLO: exactly one output, FP32/FP16, valid shape contract.
                if len(self.outputs) != 1:
                    raise ValueError(
                        f"YOLO arch dispatched but engine has "
                        f"{len(self.outputs)} outputs (expected 1)"
                    )
                if not _is_float_hp(self.outputs[0]):
                    raise ValueError(
                        f"YOLO output '{self.outputs[0]['name']}' has "
                        f"unsupported dtype {np.dtype(self.outputs[0]['dtype'])}; "
                        "only float32 or float16 are supported"
                    )
                _validate_yolo_output_shape(
                    self.outputs[0]["shape"], "engine output"
                )
        except BaseException:
            self._close()
            raise

    def _close(self) -> None:
        """Release device + pinned-host buffers and TRT objects. Idempotent."""
        try:
            for buf in list(self.inputs) + list(self.outputs):
                dev = buf.get("device")
                if dev is not None:
                    try:
                        dev.free()
                    except Exception:
                        pass
                # Pinned host allocation is freed via refcount drop.
                buf["host"] = None
                buf["device"] = None
        finally:
            self.inputs = []
            self.outputs = []
            self.context = None
            self.engine = None
            self.stream = None

    def _identify_image_input_name(self) -> str | None:
        """Return the engine-tensor name of the NCHW image input.

        Used during Pass-1 dynamic-shape binding. Same rank-4/C=3 predicate
        as the final validator — a malformed tensor named "images" must not
        be selected over a legitimate 4-D/C=3 fallback.
        """
        import tensorrt as trt
        named = None
        fallback = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                continue
            shape = self.engine.get_tensor_shape(name)
            looks_like_image = (len(shape) == 4 and shape[1] == 3)
            if name == "images" and looks_like_image:
                named = name
                break
            if looks_like_image and fallback is None:
                fallback = name
        return named if named is not None else fallback

    @property
    def input_names(self) -> list[str]:
        return [b["name"] for b in self.inputs]

    @property
    def output_names(self) -> list[str]:
        return [b["name"] for b in self.outputs]

    @property
    def image_name(self) -> str:
        """Validated image-input tensor name (set during __init__)."""
        return self._image_name

    def image_hw(self) -> tuple[int, int] | None:
        """Return (H, W) of the validated image input if statically baked.

        Reads the engine-level shape (-1 for dynamic dims preserved). Returns
        None when either spatial dim is dynamic so the caller can keep imgsz
        under user control.
        """
        buf = next((b for b in self.inputs if b["name"] == self._image_name), None)
        if buf is None:
            return None
        shape = buf.get("engine_shape", buf["shape"])
        if len(shape) != 4:
            return None
        h, w = int(shape[2]), int(shape[3])
        if h <= 0 or w <= 0:
            return None
        return h, w

    def infer(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Run inference. `feeds` maps input tensor name → ndarray.

        Returns dict mapping output name → ndarray reshaped to engine shape.
        """
        cuda = self._cuda

        # H2D for every input.
        for buf in self.inputs:
            arr = feeds.get(buf["name"])
            if arr is None:
                raise KeyError(f"missing feed for input '{buf['name']}'")
            arr = np.ascontiguousarray(arr).astype(buf["dtype"], copy=False)
            np.copyto(buf["host"], arr.ravel())
            cuda.memcpy_htod_async(buf["device"], buf["host"], self.stream)

        # Bind tensor addresses every call (cheap; survives shape-change paths).
        for buf in self.inputs + self.outputs:
            self.context.set_tensor_address(buf["name"], int(buf["device"]))
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # D2H every output.
        for buf in self.outputs:
            cuda.memcpy_dtoh_async(buf["host"], buf["device"], self.stream)
        self.stream.synchronize()

        return {b["name"]: b["host"].reshape(b["shape"]) for b in self.outputs}


class _ONNXBackend:
    """ONNX Runtime backend for development/testing.

    Mirrors `_TRTBackend.infer` interface: name-keyed feeds → name-keyed
    outputs, so the dispatcher in TRTDetector is backend-agnostic.
    """

    # ORT type-string → numpy dtype, used by infer() to coerce feeds so an
    # int64 caller-array survives an int32-typed model input.
    _ORT_TO_NUMPY: "dict[str, np.dtype]" = {}

    def __init__(self, model_path: str):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self._inputs = list(self.session.get_inputs())
        self._input_names = [i.name for i in self._inputs]
        self._output_names = [o.name for o in self.session.get_outputs()]

        if not _ONNXBackend._ORT_TO_NUMPY:
            _ONNXBackend._ORT_TO_NUMPY.update({
                "tensor(float)":   np.dtype(np.float32),
                "tensor(float16)": np.dtype(np.float16),
                "tensor(int64)":   np.dtype(np.int64),
                "tensor(int32)":   np.dtype(np.int32),
            })
        self._input_dtypes = {
            i.name: _ONNXBackend._ORT_TO_NUMPY.get(i.type) for i in self._inputs
        }

        def _looks_like_image(inp):
            shape = inp.shape
            return len(shape) == 4 and isinstance(shape[1], int) and shape[1] == 3

        named = next(
            (i for i in self._inputs if i.name == "images" and _looks_like_image(i)),
            None,
        )
        if named is None:
            named = next((i for i in self._inputs if _looks_like_image(i)), None)
        if named is None:
            raise ValueError(
                "ONNX model has no NCHW 3-channel image input "
                "(expected a tensor named 'images' or a 4-D input with C=3)"
            )
        self._image_name: str = named.name

        FLOAT_TYPES = {"tensor(float)", "tensor(float16)"}
        INT_CLS_TYPES = {"tensor(int32)", "tensor(int64)"}

        if named.type not in FLOAT_TYPES:
            raise ValueError(
                f"ONNX image input '{named.name}' has unsupported type "
                f"{named.type}; only tensor(float) or tensor(float16) are supported"
            )

        # Batch + rectangular gates (parity with _TRTBackend / C++). ONNX dims
        # may be strings (dynamic); only enforce when concrete.
        img_shape = named.shape
        if isinstance(img_shape[0], int) and img_shape[0] != 1:
            raise ValueError(
                f"ONNX image input has batch={img_shape[0]}; only batch=1 is supported"
            )
        if (
            isinstance(img_shape[2], int)
            and isinstance(img_shape[3], int)
            and img_shape[2] != img_shape[3]
        ):
            raise ValueError(
                f"ONNX image input is rectangular ({img_shape[3]}x{img_shape[2]}); "
                "only square inputs are supported (orig_target_sizes assumes square)"
            )

        outputs = self.session.get_outputs()
        out_map = {o.name: o for o in outputs}
        in_set = {i.name for i in self._inputs}
        is_deim = (
            "orig_target_sizes" in in_set
            and {"labels", "boxes", "scores"}.issubset(out_map.keys())
        )
        if is_deim:
            if out_map["labels"].type not in INT_CLS_TYPES:
                raise ValueError(
                    f"DEIM 'labels' has unsupported type "
                    f"{out_map['labels'].type} (expected int32 or int64)"
                )
            if out_map["boxes"].type not in FLOAT_TYPES:
                raise ValueError(
                    f"DEIM 'boxes' has unsupported type "
                    f"{out_map['boxes'].type} (expected float32 or float16)"
                )
            if out_map["scores"].type not in FLOAT_TYPES:
                raise ValueError(
                    f"DEIM 'scores' has unsupported type "
                    f"{out_map['scores'].type} (expected float32 or float16)"
                )
            ots = next(i for i in self._inputs if i.name == "orig_target_sizes")
            if ots.type not in INT_CLS_TYPES:
                raise ValueError(
                    f"DEIM 'orig_target_sizes' has unsupported type "
                    f"{ots.type} (expected int32 or int64)"
                )
            # Shape gate (concrete dims only — ONNX may have dynamic batch).
            ots_shape = ots.shape
            if (
                len(ots_shape) != 2
                or (isinstance(ots_shape[0], int) and ots_shape[0] != 1)
                or (isinstance(ots_shape[1], int) and ots_shape[1] != 2)
            ):
                raise ValueError(
                    f"DEIM 'orig_target_sizes' has shape {ots_shape}; expected (1, 2)"
                )
            # Top-K shape contract: labels=(1, K), boxes=(1, K, 4),
            # scores=(1, K). ONNX shape entries are int|str (dynamic axes);
            # check rank + concrete dims, defer dynamic ones.
            l_shape = out_map["labels"].shape
            b_shape = out_map["boxes"].shape
            s_shape = out_map["scores"].shape

            def _check_axis(name_, shape_, axis, expected):
                d = shape_[axis] if axis < len(shape_) else None
                if isinstance(d, int) and d != expected:
                    raise ValueError(
                        f"DEIM '{name_}' axis {axis} is {d}; expected {expected} "
                        f"(shape={shape_})"
                    )

            if len(l_shape) != 2:
                raise ValueError(f"DEIM 'labels' rank is {len(l_shape)}; expected 2")
            if len(s_shape) != 2:
                raise ValueError(f"DEIM 'scores' rank is {len(s_shape)}; expected 2")
            if len(b_shape) != 3:
                raise ValueError(f"DEIM 'boxes' rank is {len(b_shape)}; expected 3")
            _check_axis("labels", l_shape, 0, 1)
            _check_axis("scores", s_shape, 0, 1)
            _check_axis("boxes",  b_shape, 0, 1)
            _check_axis("boxes",  b_shape, 2, 4)
            # K consistency check only when all three K dims are concrete.
            ks = [d for d in (l_shape[1], s_shape[1], b_shape[1]) if isinstance(d, int)]
            if len(ks) == 3 and len(set(ks)) > 1:
                raise ValueError(
                    f"DEIM K mismatch: labels K={l_shape[1]} "
                    f"scores K={s_shape[1]} boxes K={b_shape[1]}"
                )
        else:
            # YOLO: exactly one output, FP32/FP16, valid shape. Dynamic ONNX
            # dims defer to the runtime check.
            if len(outputs) != 1:
                raise ValueError(
                    f"YOLO arch dispatched but ONNX model has "
                    f"{len(outputs)} outputs (expected 1)"
                )
            if outputs[0].type not in FLOAT_TYPES:
                raise ValueError(
                    f"YOLO output '{outputs[0].name}' has unsupported type "
                    f"{outputs[0].type}; only tensor(float) or tensor(float16) are supported"
                )
            _validate_yolo_output_shape(outputs[0].shape, "ONNX output")

    @property
    def input_names(self) -> list[str]:
        return list(self._input_names)

    @property
    def output_names(self) -> list[str]:
        return list(self._output_names)

    @property
    def image_name(self) -> str:
        """Validated image-input tensor name (set during __init__)."""
        return self._image_name

    def image_hw(self) -> tuple[int, int] | None:
        """Return (H, W) of the validated image input, or None when dynamic.

        ONNX dynamic axes appear as strings; concrete (H, W) only when both
        are int.
        """
        buf = next((i for i in self._inputs if i.name == self._image_name), None)
        if buf is None:
            return None
        shape = buf.shape
        if len(shape) != 4:
            return None
        h, w = shape[2], shape[3]
        if not isinstance(h, int) or not isinstance(w, int) or h <= 0 or w <= 0:
            return None
        return h, w

    def infer(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        # ORT validates feed types strictly; coerce to per-input dtype so an
        # int64 feed survives an int32 model input.
        coerced: dict[str, np.ndarray] = {}
        for name, arr in feeds.items():
            target = self._input_dtypes.get(name)
            if target is not None and arr.dtype != target:
                arr = arr.astype(target, copy=False)
            coerced[name] = np.ascontiguousarray(arr)
        outs = self.session.run(self._output_names, coerced)
        return dict(zip(self._output_names, outs))


class TRTDetector:
    """Traffic light detector using TensorRT or ONNX Runtime.

    Backend is picked by file extension:
      - .engine → TensorRT
      - .onnx   → ONNX Runtime

    Architecture (YOLO26 vs DEIM-D-FINE) is auto-detected from tensor names.
    YOLO26 is NMS-free by training. DEIM-D-FINE's deploy top-K is followed
    by a runtime per-class IoU NMS @ 0.5 (see _postprocess_deim) — Dense-
    O2O training emits multiple queries per target, producing near-but-
    not-identical box outputs that flatten-then-topk does not collapse.
    Output type and unscale math are shared.
    """

    def __init__(
        self,
        model_path: str,
        conf_thresh: float = 0.25,
        imgsz: int = 1280,
    ):
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz

        path = Path(model_path)
        if path.suffix == ".engine":
            # imgsz lets the backend size buffers for spatially-dynamic
            # engines; ignored for static engines.
            self.backend = _TRTBackend(model_path, imgsz=imgsz)
        elif path.suffix == ".onnx":
            self.backend = _ONNXBackend(model_path)
        else:
            raise ValueError(f"Unsupported model format: {path.suffix} (expected .engine or .onnx)")

        self.arch = _detect_arch(self.backend.input_names, self.backend.output_names)

        # Snap self.imgsz to the engine's baked input size (static only).
        # Rectangular engines are rejected upstream so eh == ew here.
        engine_hw = self.backend.image_hw()
        if engine_hw is not None:
            eh, ew = engine_hw
            if eh != ew:
                raise ValueError(
                    f"engine image input is rectangular ({ew}x{eh}); "
                    "only square inputs are supported"
                )
            if eh != self.imgsz:
                print(
                    f"[TRTDetector] warning: engine input {ew}x{eh} differs "
                    f"from imgsz={self.imgsz}; using engine size {eh}.",
                    file=sys.stderr,
                )
                self.imgsz = eh

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a BGR frame. Returns list of Detection objects."""
        blob, scale, pad = _preprocess(frame, self.imgsz)
        image_name = self.backend.image_name

        if self.arch is Arch.DEIM:
            # DEIM deploy graph: `bbox_pred *= orig_target_sizes.repeat(1,2)`
            # broadcasts (s0, s1) onto (x1, y1, x2, y2). For a square
            # letterbox the order is symmetric and `[imgsz, imgsz]` lands
            # boxes in letterbox-pixel coords (shared with YOLO).
            feeds = {
                image_name: blob,
                "orig_target_sizes": np.array(
                    [[self.imgsz, self.imgsz]], dtype=np.int64
                ),
            }
        else:
            feeds = {image_name: blob}

        outputs = self.backend.infer(feeds)

        if self.arch is Arch.DEIM:
            return self._postprocess_deim(outputs, frame.shape[:2], scale, pad)
        return self._postprocess_yolo(outputs, frame.shape[:2], scale, pad)

    def detect_json(self, frame: np.ndarray) -> str:
        """Run detection and return JSON string."""
        detections = self.detect(frame)
        return json.dumps([d.to_dict() for d in detections])

    def detect_ros(self, frame: np.ndarray, header=None):
        """Run detection and return vision_msgs/Detection2DArray.

        Args:
            frame: BGR numpy array.
            header: std_msgs/Header to attach (preserves camera timestamp).
                    If None, creates an empty header.

        Requires: ros2 vision_msgs package.
        """
        from std_msgs.msg import Header
        from vision_msgs.msg import Detection2DArray

        detections = self.detect(frame)
        msg = Detection2DArray()
        msg.header = header if header is not None else Header()
        msg.detections = [d.to_ros_msg() for d in detections]
        return msg

    def _unscale_clip(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        scale: float,
        pad: tuple[float, float],
        orig_shape: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        """Letterbox → original-frame xyxy with image-bound clipping."""
        pad_w, pad_h = pad
        h_orig, w_orig = orig_shape
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale
        x1 = max(0.0, min(x1, float(w_orig)))
        y1 = max(0.0, min(y1, float(h_orig)))
        x2 = max(0.0, min(x2, float(w_orig)))
        y2 = max(0.0, min(y2, float(h_orig)))
        return x1, y1, x2, y2

    def _postprocess_yolo(
        self,
        outputs: dict[str, np.ndarray],
        orig_shape: tuple[int, int],
        scale: float,
        pad: tuple[float, float],
    ) -> list[Detection]:
        """Decode YOLO26 output into Detection objects.

        Expected output (batch dropped): (N, 4 + nc) — the stripped head
        emits xyxy directly (NOT cxcywh). Some exports transpose to (4+nc, N).
        """
        if len(outputs) != 1:
            raise RuntimeError(
                f"YOLO postprocess: expected 1 output, got {len(outputs)} "
                f"({list(outputs.keys())}) — arch dispatch fall-through?"
            )
        raw = next(iter(outputs.values()))
        # Strip the batch axis explicitly — np.squeeze() strips ALL singletons
        # and would collapse a valid (1, 1, 11) shape (single-detection
        # rank-3 export) to rank 1. C++ accepts (1, 1, 11) at construction.
        if raw.ndim == 3 and raw.shape[0] == 1:
            output = raw[0]
        elif raw.ndim == 2:
            output = raw
        else:
            raise RuntimeError(
                f"YOLO postprocess: output has rank {raw.ndim}, shape {raw.shape} "
                "(expected rank 2 or 3-with-batch=1)"
            )

        expected_row_len = 4 + len(CLASS_NAMES)
        if output.ndim != 2 or expected_row_len not in output.shape:
            raise RuntimeError(
                f"YOLO postprocess: output shape {raw.shape} has no "
                f"{expected_row_len}-wide axis (nc={len(CLASS_NAMES)})"
            )
        if output.shape[0] == expected_row_len and output.shape[1] == expected_row_len:
            raise RuntimeError(
                f"YOLO postprocess: output shape {output.shape} is "
                "orientation-ambiguous (both axes equal 4+nc)"
            )
        if output.shape[0] == expected_row_len and output.shape[1] != expected_row_len:
            output = output.T

        detections: list[Detection] = []

        for row in output:
            class_scores = row[4:]
            cls_id = int(np.argmax(class_scores))
            conf = float(class_scores[cls_id])

            if conf < self.conf_thresh:
                continue

            x1, y1, x2, y2 = self._unscale_clip(
                float(row[0]), float(row[1]), float(row[2]), float(row[3]),
                scale, pad, orig_shape,
            )
            detections.append(Detection(
                class_id=cls_id, confidence=conf,
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))

        return detections

    def _postprocess_deim(
        self,
        outputs: dict[str, np.ndarray],
        orig_shape: tuple[int, int],
        scale: float,
        pad: tuple[float, float],
    ) -> list[Detection]:
        """Decode DEIM-D-FINE deploy output into Detection objects.

        Deploy postprocessor (DEIM/engine/deim/postprocessor.py:50) emits:
          - labels: (N, K)  — class id per query, already top-K sorted
          - boxes : (N, K, 4) — xyxy already scaled by `orig_target_sizes`
          - scores: (N, K)  — sigmoid (focal-loss path), descending

        Two-phase postprocess. Phase 1 collects per-slot survivors in
        letterbox space after conf threshold + class validation +
        same-query bit-identical dedup. Phase 2 runs per-class greedy IoU
        NMS over those survivors, then unscales + clips the keepers to
        image space.

        PHASE 1 — same-query dedup. postprocessor.py:59 does
        `topk(scores.flatten(1), K)` over (query × class) score pairs, so
        a single high-confidence query can occupy multiple top-K slots
        under different class labels. The bbox is fetched via
        `bbox_pred.gather(query_idx)` (postprocessor.py:64), so same-query
        slots produce the bit-identical letterbox box.

        Strict equality is correct: same source slot read twice yields
        the same bytes (FP16→FP32 widening is deterministic). Dedup is
        done BEFORE the affine unscale to avoid wasted work and to
        justify `==` instead of an epsilon that could collapse two
        genuinely distinct objects.

        PHASE 2 — per-class IoU NMS @ 0.5. DEIM Dense-O2O training
        intentionally assigns multiple queries per target, so distinct
        queries can emit near-but-not-identical boxes for the same
        physical object; phase 1's strict-equality dedup never catches
        these. Without NMS, DEIM-M emits ~30 detections per frame at
        conf=0.25 on busy traffic-light scenes (most are cross-query
        duplicates).

        Per-class (NOT class-agnostic): adjacent traffic lights of
        different states (e.g. red next to yellow) must not suppress
        each other — that would corrupt the per-class AP signal the
        locked plan's safety-class guardrail keys on.

        Threshold 0.5 — RT-DETR / D-FINE postprocessors use 0.45–0.5.
        DEIM upstream's deploy mode runs NO NMS at all
        (DEIM/engine/deim/postprocessor.py:75 returns labels/boxes/scores
        raw from topk), which is precisely why this hotfix is needed.
        Hardcoded; if a future round wants per-detector tuning, lift to
        a TRTDetector ctor arg alongside conf_thresh. Re-evaluate the
        0.5 choice if a future round adds dense rows of small (<8px)
        targets where adjacent objects routinely sit at IoU > 0.5.

        IoU computed in letterbox space (pre-clip). Doing it post-clip
        would change box areas whenever a box extends past the image
        edge, biasing NMS toward suppressing edge detections.

        Float32 arithmetic for NMS — bit-parity with the C++ runtime
        (which uses `float` end-to-end). A naive Python implementation
        would let intermediates widen to float64 and disagree with C++
        on borderline IoU pairs near the 0.5 boundary. We materialize
        survivor boxes + areas as np.float32 arrays before phase 2 and
        keep the IoU math in float32 throughout.

        C++ parity: equivalent two-phase pipeline in
        inference/cpp/src/trt_pipeline.cpp postprocessDeim() —
        DeimSurvivor struct + same per-class IoU loop.

        Residual of phase 1: two DISTINCT queries that emit bit-identical
        boxes (e.g. FP16 quantization) collapse onto the highest-scored
        class. Degrades like NMS at IoU=1.0; the strictly-correct fix is
        to export `query_idx` as a 4th DEIM deploy output (out of scope).
        """
        for k in ("labels", "boxes", "scores"):
            if outputs[k].shape[0] != 1:
                raise RuntimeError(
                    f"DEIM postprocess: '{k}' has batch={outputs[k].shape[0]}; "
                    "only batch=1 is supported"
                )
        labels = outputs["labels"][0]
        boxes = outputs["boxes"][0]
        scores = outputs["scores"][0]

        keep_score = scores >= self.conf_thresh
        if not np.any(keep_score):
            return []

        labels = labels[keep_score]
        boxes = boxes[keep_score]
        scores = scores[keep_score]

        # PHASE 0 — score-desc sort BEFORE phase 1 dedup. Sorting before
        # dedup ensures the dedup naturally keeps the HIGHEST-CONFIDENCE
        # slot for any bit-identical letterbox-box tuple (the first one
        # encountered in sorted order). Doing this AFTER dedup would
        # only fix NMS ordering, but leave the dedup itself trusting
        # DEIM's `torch.topk(sorted=True)` contract for class-label
        # selection — which we explicitly want to stop depending on,
        # per the production-correctness amendment below.
        #
        # np.argsort with kind='stable' preserves the original topk
        # tie-break order on exact-conf ties (rare but possible at FP16
        # quantization boundaries).
        order = np.argsort(-scores, kind="stable")
        labels = labels[order]
        boxes = boxes[order]
        scores = scores[order]

        nc = len(CLASS_NAMES)

        # PHASE 1 — collect survivors after same-query bit-identical
        # letterbox-box dedup. Parallel lists materialize the float32
        # NMS arrays below.
        #
        # INVARIANT (enforced, not trusted): the resulting cls_list /
        # conf_list / box_list are in score-descending order, because
        # we iterate the phase-0-sorted arrays and only DROP entries
        # (never reorder). Phase 2 greedy NMS depends on this. DEIM's
        # `torch.topk(sorted=True)` already provides the order, but
        # the explicit phase-0 sort means (a) we don't depend on that
        # upstream contract continuing to hold across DEIM versions,
        # and (b) we don't need a `__debug__`-only assert that would
        # compile out under `python -O`.
        seen_lbox: set[tuple[float, float, float, float]] = set()
        cls_list: list[int] = []
        conf_list: list[float] = []
        box_list: list[tuple[float, float, float, float]] = []
        for cls_id, (lx1, ly1, lx2, ly2), conf in zip(labels.tolist(), boxes, scores.tolist()):
            cls_id = int(cls_id)
            # DEIM may emit cls_id == nc for "no object"; drop instead of
            # clamp. Done before dedup so a "no object" slot does not
            # consume a valid box's first-seen slot.
            if cls_id < 0 or cls_id >= nc:
                continue
            lbox = (float(lx1), float(ly1), float(lx2), float(ly2))
            if lbox in seen_lbox:
                continue
            seen_lbox.add(lbox)
            cls_list.append(cls_id)
            conf_list.append(float(conf))
            box_list.append(lbox)

        n = len(cls_list)
        if n == 0:
            return []

        # Phase 2: per-class greedy IoU NMS @ 0.5 in float32.
        # O(N^2) — N is bounded by num_top_queries=300 and is ~30 in
        # practice on busy DEIM-M frames after phase 1. Worst-case
        # upper-triangular pair count is N(N-1)/2 ≈ 45k at 300;
        # realistic ~900/frame at 30 FPS is ~27k/s; trivially sub-ms.
        classes = np.asarray(cls_list, dtype=np.int32)
        boxes_lb = np.asarray(box_list, dtype=np.float32)
        # Per-survivor area precomputed once (O(N) instead of O(N^2)).
        # Zero-area degenerate boxes get area=0; they pass through
        # emission because the IoU numerator stays 0 against any
        # partner, but the `uni > 0` guard below specifically protects
        # against the both-degenerate (a_area == b_area == 0)
        # divide-by-zero case.
        widths = np.maximum(np.float32(0), boxes_lb[:, 2] - boxes_lb[:, 0])
        heights = np.maximum(np.float32(0), boxes_lb[:, 3] - boxes_lb[:, 1])
        areas = widths * heights

        IOU_THRESH = np.float32(0.5)
        zero_f32 = np.float32(0)
        keep_nms = np.ones(n, dtype=bool)
        for i in range(n):
            if not keep_nms[i]:
                continue
            ci = classes[i]
            ax1, ay1, ax2, ay2 = boxes_lb[i]
            a_area = areas[i]
            for j in range(i + 1, n):
                if not keep_nms[j]:
                    continue
                if classes[j] != ci:
                    continue
                bx1, by1, bx2, by2 = boxes_lb[j]
                ix1 = max(ax1, bx1)
                iy1 = max(ay1, by1)
                ix2 = min(ax2, bx2)
                iy2 = min(ay2, by2)
                iw = ix2 - ix1
                ih = iy2 - iy1
                if iw <= zero_f32 or ih <= zero_f32:
                    continue
                inter = iw * ih
                uni = a_area + areas[j] - inter
                # `uni > 0` guard handles the both-degenerate
                # (a_area == b_area == 0) case to avoid div-by-zero.
                if uni > zero_f32 and (inter / uni) > IOU_THRESH:
                    keep_nms[j] = False

        detections: list[Detection] = []
        for i in range(n):
            if not keep_nms[i]:
                continue
            lx1, ly1, lx2, ly2 = boxes_lb[i]
            x1, y1, x2, y2 = self._unscale_clip(
                float(lx1), float(ly1), float(lx2), float(ly2),
                scale, pad, orig_shape,
            )
            detections.append(Detection(
                class_id=int(classes[i]),
                confidence=conf_list[i],
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))

        return detections
