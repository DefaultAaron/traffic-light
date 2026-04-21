"""TensorRT inference pipeline for traffic light detection.

Supports two backends:
  - TensorRT (production, Orin): requires `tensorrt` and `pycuda`
  - ONNX Runtime (development): requires `onnxruntime` or `onnxruntime-gpu`

Usage:
    detector = TRTDetector("best.engine")  # or "best.onnx"
    detections = detector.detect(frame)

Thread safety: TRTDetector is NOT thread-safe. It holds a single CUDA
stream and pre-allocated pinned host buffers shared across calls. Construct
and call `detect()` on the same thread. Use one detector per worker thread.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
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
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_lb, scale, pad = _letterbox(img_rgb, (imgsz, imgsz))
    blob = img_lb.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)  # HWC → CHW
    blob = np.expand_dims(blob, 0)  # add batch dim
    blob = np.ascontiguousarray(blob)
    return blob, scale, pad


class _TRTBackend:
    """TensorRT engine backend using pycuda."""

    def __init__(self, engine_path: str):
        import pycuda.autoinit  # noqa: F401 — initializes CUDA context
        import pycuda.driver as cuda
        import tensorrt as trt

        self._cuda = cuda
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate host/device buffers
        self.inputs: list[dict] = []
        self.outputs: list[dict] = []
        self.bindings: list[int] = []
        self.stream = cuda.Stream()

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = abs(int(np.prod(shape)))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))
            buf = {"host": host_mem, "device": device_mem, "shape": shape, "name": name}

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append(buf)
            else:
                self.outputs.append(buf)

    def infer(self, input_blob: np.ndarray) -> np.ndarray:
        """Run inference. Returns raw output array."""
        cuda = self._cuda
        # Copy input to device
        np.copyto(self.inputs[0]["host"], input_blob.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)

        # Set tensor addresses and execute
        for buf in self.inputs + self.outputs:
            self.context.set_tensor_address(buf["name"], int(buf["device"]))
        self.context.execute_async_v3(stream_handle=self.stream.handle)

        # Copy output to host
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        self.stream.synchronize()

        return self.outputs[0]["host"].reshape(self.outputs[0]["shape"])


class _ONNXBackend:
    """ONNX Runtime backend for development/testing."""

    def __init__(self, model_path: str):
        import onnxruntime as ort

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, input_blob: np.ndarray) -> np.ndarray:
        """Run inference. Returns raw output array."""
        outputs = self.session.run(None, {self.input_name: input_blob})
        return outputs[0]


class TRTDetector:
    """Traffic light detector using TensorRT or ONNX Runtime.

    Backend is picked by file extension:
      - .engine → TensorRT
      - .onnx   → ONNX Runtime

    YOLO26 is NMS-free by architecture, so no external NMS is applied.
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
            self.backend = _TRTBackend(model_path)
        elif path.suffix == ".onnx":
            self.backend = _ONNXBackend(model_path)
        else:
            raise ValueError(f"Unsupported model format: {path.suffix} (expected .engine or .onnx)")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run detection on a BGR frame. Returns list of Detection objects."""
        blob, scale, pad = _preprocess(frame, self.imgsz)
        raw_output = self.backend.infer(blob)
        return self._postprocess(raw_output, frame.shape[:2], scale, pad)

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

    def _postprocess(
        self,
        raw_output: np.ndarray,
        orig_shape: tuple[int, int],
        scale: float,
        pad: tuple[float, float],
    ) -> list[Detection]:
        """Decode YOLO26 output into Detection objects.

        Expected output (batch dim dropped): (N, 4 + num_classes) where each
        row is (cx, cy, w, h, cls0, cls1, ...). Some exports transpose to
        (4 + num_classes, N); we detect this by matching the expected class
        count (len(CLASS_NAMES)) against one of the two dimensions.
        """
        output = np.squeeze(raw_output)  # remove batch dim

        expected_row_len = 4 + len(CLASS_NAMES)
        if output.ndim == 2 and output.shape[0] == expected_row_len and output.shape[1] != expected_row_len:
            output = output.T  # (4+nc, N) → (N, 4+nc)

        h_orig, w_orig = orig_shape
        pad_w, pad_h = pad
        detections: list[Detection] = []

        for row in output:
            # row: [x1, y1, x2, y2, cls0_conf, ..., clsN_conf] in letterbox frame.
            # YOLO26 head (post-strip at Concat_3) emits xyxy directly via
            # (anchor ± DFL-decoded distance) × stride — not cxcywh.
            class_scores = row[4:]
            cls_id = int(np.argmax(class_scores))
            conf = float(class_scores[cls_id])

            if conf < self.conf_thresh:
                continue

            lx1, ly1, lx2, ly2 = row[:4]
            x1 = (lx1 - pad_w) / scale
            y1 = (ly1 - pad_h) / scale
            x2 = (lx2 - pad_w) / scale
            y2 = (ly2 - pad_h) / scale

            # Clip to image bounds
            x1 = max(0, min(x1, w_orig))
            y1 = max(0, min(y1, h_orig))
            x2 = max(0, min(x2, w_orig))
            y2 = max(0, min(y2, h_orig))

            detections.append(Detection(
                class_id=cls_id,
                confidence=conf,
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
            ))

        return detections
