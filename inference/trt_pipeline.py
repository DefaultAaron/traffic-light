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
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, r, (dw, dh)


def _preprocess(img: np.ndarray, imgsz: int = 640) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Preprocess frame for inference.

    BGR (OpenCV) → RGB → letterbox → normalize → CHW → float32 → batch dim.
    Returns (input_tensor, scale, (pad_w, pad_h)).

    Both YOLO26 and DEIM-D-FINE consume normalized float32 NCHW in [0, 1] —
    DEIM's reference deploy code uses torchvision.transforms.ToTensor() which
    is also `/255`, so the preprocessor is shared. Padding color is 114-gray
    (matches YOLO Ultralytics + DEIM's training transform `letterbox` aug;
    the `Image.new("RGB")` zero-pad in DEIM/tools/inference/onnx_inf.py is
    only the inference *demo*, not the training pipeline — verified against
    DEIM/engine/data/transforms/).
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lb, scale, pad = _letterbox(img_rgb, (imgsz, imgsz))
    blob = img_lb.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)  # HWC → CHW
    blob = np.expand_dims(blob, 0)  # add batch dim
    blob = np.ascontiguousarray(blob)
    return blob, scale, pad


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

        self._cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.inputs: list[dict] = []
        self.outputs: list[dict] = []
        self.stream = cuda.Stream()

        # Pass 1: bind dynamic input shapes so output shapes resolve and
        # buffers are sized for the actual workload. Without this, an
        # engine with dynamic spatial dims allocates 1×3×1×1 input buffer
        # and crashes on the first H×W feed. Substitution rules:
        #   - dynamic spatial dims of the image input  → imgsz (when given)
        #   - any other dynamic dim                    → 1
        # The image input is identified by name ("images") if present, else
        # the first 4-D NCHW input with 3 channels.
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
            self.context.set_input_shape(name, shape)

        # Pass 2: allocate per-tensor buffers using context-resolved shapes.
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
            device_mem = cuda.mem_alloc(host_mem.nbytes)
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

    def _identify_image_input_name(self) -> str | None:
        """Return the engine-tensor name of the NCHW image input.

        Used during buffer alloc to decide which input's dynamic spatial
        dims should be bound to imgsz (vs. left at 1). Prefers the name
        "images" (export convention for both YOLO26 stripped head and
        DEIM-D-FINE deploy graph); falls back to the first 4-D input with
        3 channels.
        """
        import tensorrt as trt
        candidates_4d = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
                continue
            shape = self.engine.get_tensor_shape(name)
            if name == "images":
                return name
            if len(shape) == 4 and shape[1] == 3:
                candidates_4d.append(name)
        return candidates_4d[0] if candidates_4d else None

    @property
    def input_names(self) -> list[str]:
        return [b["name"] for b in self.inputs]

    @property
    def output_names(self) -> list[str]:
        return [b["name"] for b in self.outputs]

    def image_hw(self) -> tuple[int, int] | None:
        """Return (H, W) of the NCHW image input *if statically baked*.

        Reads the engine-level shape (which keeps -1 for truly dynamic dims)
        rather than the context-resolved shape (which has dynamic dims
        clobbered to 1 in __init__ for buffer allocation). Returns None
        when either spatial dim is dynamic — caller keeps imgsz under user
        control in that case rather than crashing the auto-correct on a
        bogus `(1, 1)`.
        """
        candidates = [b for b in self.inputs if b["name"] == "images"]
        if not candidates:
            candidates = [b for b in self.inputs if len(b.get("engine_shape", ())) == 4]
        if not candidates:
            return None
        shape = candidates[0].get("engine_shape", candidates[0]["shape"])
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

    def __init__(self, model_path: str):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self._inputs = list(self.session.get_inputs())
        self._input_names = [i.name for i in self._inputs]
        self._output_names = [o.name for o in self.session.get_outputs()]

    @property
    def input_names(self) -> list[str]:
        return list(self._input_names)

    @property
    def output_names(self) -> list[str]:
        return list(self._output_names)

    def image_hw(self) -> tuple[int, int] | None:
        """Return (H, W) of the static NCHW image input, or None when dynamic.

        ONNX dynamic axes appear as strings (e.g. 'N', 'H', 'W'); only return
        a concrete (H, W) if both spatial dims are integers. Otherwise the
        upstream `imgsz` knob keeps full control.
        """
        candidates = [i for i in self._inputs if i.name == "images"]
        if not candidates:
            candidates = [i for i in self._inputs if len(i.shape) == 4]
        if not candidates:
            return None
        shape = candidates[0].shape
        if len(shape) != 4:
            return None
        h, w = shape[2], shape[3]
        if not isinstance(h, int) or not isinstance(w, int) or h <= 0 or w <= 0:
            return None
        return h, w

    def infer(self, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        outs = self.session.run(self._output_names, feeds)
        return dict(zip(self._output_names, outs))


class TRTDetector:
    """Traffic light detector using TensorRT or ONNX Runtime.

    Backend is picked by file extension:
      - .engine → TensorRT
      - .onnx   → ONNX Runtime

    Architecture (YOLO26 vs DEIM-D-FINE) is auto-detected from tensor names.
    Both arches are NMS-free as exposed: YOLO26 by training, DEIM by the
    deploy postprocessor's top-K. Output type and unscale math are shared.
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
            # Pass imgsz so the backend can size buffers correctly for
            # spatially-dynamic engines (would otherwise allocate at 1×3×1×1
            # and overflow on the first feed). For statically-baked engines
            # imgsz is ignored at this stage and the auto-correct below
            # snaps self.imgsz to the engine's actual size.
            self.backend = _TRTBackend(model_path, imgsz=imgsz)
        elif path.suffix == ".onnx":
            self.backend = _ONNXBackend(model_path)
        else:
            raise ValueError(f"Unsupported model format: {path.suffix} (expected .engine or .onnx)")

        self.arch = _detect_arch(self.backend.input_names, self.backend.output_names)

        # Auto-correct imgsz against the engine's baked input. Without this,
        # a TRT engine exported at 640 (current default for DEIM) crashes
        # when the demo runs with --imgsz 1280: preprocess builds a 1280x1280
        # blob that doesn't fit the 640x640 input buffer. The C++ pipeline
        # has the same safeguard in allocateBuffers(); we mirror it here so
        # the two paths stay in lockstep. Static shapes only — dynamic
        # engines/ONNX (image_hw == None) leave self.imgsz under user control.
        engine_hw = self.backend.image_hw()
        if engine_hw is not None:
            eh, ew = engine_hw
            es = min(eh, ew)
            if es != self.imgsz:
                import sys
                print(
                    f"[TRTDetector] warning: engine input {ew}x{eh} differs "
                    f"from imgsz={self.imgsz}; using engine size {es}.",
                    file=sys.stderr,
                )
                self.imgsz = es

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a BGR frame. Returns list of Detection objects."""
        blob, scale, pad = _preprocess(frame, self.imgsz)

        if self.arch is Arch.DEIM:
            # DEIM deploy graph emits boxes scaled to `orig_target_sizes`.
            # Feeding the letterbox size (H_lb, W_lb) makes boxes land in
            # letterbox-pixel coords — same frame as YOLO output — so the
            # unscale path `(bb - pad) / scale` is shared. Note the order:
            # postprocessor.py multiplies by `orig_target_sizes.repeat(1,2)`
            # which after the `.unsqueeze(1)` broadcasts to (x_scale, y_scale,
            # x_scale, y_scale). With the canonical (H, W) order the x-axis
            # would scale by H — wrong. The training/eval code feeds
            # `(width, height)` here (RT-DETR convention), so we match that.
            feeds = {
                "images": blob,
                "orig_target_sizes": np.array(
                    [[self.imgsz, self.imgsz]], dtype=np.int64
                ),
            }
        else:
            feeds = {self.backend.input_names[0]: blob}

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
        """Letterbox → original-frame xyxy with image-bound clipping.

        Shared between YOLO and DEIM postprocess paths so any future
        rounding/clipping change propagates uniformly.
        """
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

        Expected output (batch dim dropped): (N, 4 + num_classes) where each
        row is (x1, y1, x2, y2, cls0, cls1, ...) in letterbox pixels. The
        stripped YOLO26 head emits xyxy directly via (anchor ± DFL-decoded
        distance) × stride — NOT cxcywh. Some exports transpose to
        (4 + num_classes, N); we detect this by matching the expected class
        count (len(CLASS_NAMES)) against one of the two dimensions.
        """
        # YOLO has a single output; pick the first one regardless of name.
        raw = next(iter(outputs.values()))
        output = np.squeeze(raw)  # remove batch dim

        expected_row_len = 4 + len(CLASS_NAMES)
        if output.ndim == 2 and output.shape[0] == expected_row_len and output.shape[1] != expected_row_len:
            output = output.T  # (4+nc, N) → (N, 4+nc)

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
        We drop the batch dim, threshold scores, and unscale through the
        shared letterbox math. No NMS is needed: the top-K with focal
        scores already serves the role.
        """
        labels = np.squeeze(outputs["labels"], axis=0)
        boxes = np.squeeze(outputs["boxes"], axis=0)
        scores = np.squeeze(outputs["scores"], axis=0)

        keep = scores >= self.conf_thresh
        if not np.any(keep):
            return []

        labels = labels[keep]
        boxes = boxes[keep]
        scores = scores[keep]

        nc = len(CLASS_NAMES)
        detections: list[Detection] = []
        for cls_id, (lx1, ly1, lx2, ly2), conf in zip(labels.tolist(), boxes, scores.tolist()):
            cls_id = int(cls_id)
            # Drop padded/out-of-range labels — DEIM occasionally emits class
            # indices == num_classes for "no object" depending on the head
            # config; clamping silently would mislabel detections so we
            # discard instead.
            if cls_id < 0 or cls_id >= nc:
                continue
            x1, y1, x2, y2 = self._unscale_clip(
                float(lx1), float(ly1), float(lx2), float(ly2),
                scale, pad, orig_shape,
            )
            detections.append(Detection(
                class_id=cls_id, confidence=float(conf),
                x1=x1, y1=y1, x2=x2, y2=y2,
            ))

        return detections
