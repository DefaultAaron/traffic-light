# Traffic Light Recognition

Traffic light detection and state classification for autonomous driving on NVIDIA Jetson AGX Orin.

- **R1 (current)**: 7-class traffic-light detector — `red`, `yellow`, `green`, `redLeft`, `greenLeft`, `redRight`, `greenRight`
- **R2 (planned)**: joint detector expanded to **10–14 classes** — 9–12 traffic-light classes (adds `forwardRed`/`forwardGreen` + PM-pending extensions) and 1–2 barrier classes (`barrier` MVP, optional `armOn`/`armOff`). See `docs/phase_2_round_1_report.md` §"R2 范围扩展".

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

Classes: R1 uses 7 classes (3 colors + 4 arrow variants). R2 will extend to 10–14 classes (see top of README).

## Usage

### Train a single model

```bash
python main.py train yolo26n
python main.py train yolo11s --epochs 50 --device 0
python main.py train yolo26n --epochs 3 --device mps  # quick sanity check on M4 Pro
```

### Resume an interrupted training run

```bash
python main.py train --resume runs/yolo26n/weights/last.pt
```

Ultralytics restores epochs, batch, device, and all other hyperparameters from the run's `args.yaml`, so `--epochs`, `--batch`, and `--device` are ignored when `--resume` is set. The run must have at least one completed epoch (`last.pt` present).

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

## Deployment

Target device: **NVIDIA Jetson AGX Orin 64GB** at `nvidia@192.168.30.138`.

### Inference pipelines

Two equivalent implementations of the TensorRT detector live under `inference/`:

| | Python | C++ |
|---|---|---|
| Pipeline | `inference/trt_pipeline.py` | `inference/cpp/src/trt_pipeline.cpp` + `include/trt_pipeline.hpp` |
| Demo | `inference/demo.py` | `inference/cpp/src/demo.cpp` |
| Build | `uv`/pip env | CMake ≥ 3.18, see below |
| Use case | Quick iteration, debugging, JSON telemetry | Production deployment on Orin |

Both accept the same flags: `--source`, `--model`, `--conf`, `--imgsz`, `--no-show`, `--save`. Class names and draw colors are kept in sync between the two backends; R1 ships 7 classes, and when R2's class list is locked the constants in `trt_pipeline.py`, `trt_pipeline.hpp`, and `demo.cpp` will be updated in lockstep.

### Python deployment (dev / quick test)

```bash
# On Orin (from project root)
python -m inference.demo --source video.mp4 --model weights/best.engine
python -m inference.demo --source 0 --model weights/best.engine --json  # webcam + JSON
```

### C++ deployment (production)

#### 1. Sync source to the Orin

```bash
rsync -avz --progress \
    --exclude 'build/' --exclude '*.o' --exclude '.DS_Store' \
    ~/Documents/Projects/mingtai/traffic-light/inference/cpp \
    nvidia@192.168.30.138:traffic-light/inference/
```

#### 2. Orin environment

JetPack 5.1.x / L4T R35.x ships from the factory flash — no Docker, no reflash, no JetPack upgrade. Dependency matrix:

| Component | Version | Location |
|---|---|---|
| CUDA | 11.4 | `/usr/local/cuda` |
| TensorRT | 8.5.2 | headers: `/usr/include/aarch64-linux-gnu/NvInfer.h`, libs: `/usr/lib/aarch64-linux-gnu/` |
| OpenCV | 4.5.4 | system (`core`, `imgproc`, `highgui`, `videoio`) |
| CMake | 3.16.3 (factory) → **≥ 3.18 required** | — |

Only CMake is upgraded on-device (the project's `CMakeLists.txt` uses `find_package(CUDAToolkit)`, which needs 3.18+). The rest is used as-is — **fleet consistency outweighs any benefit from upgrading TensorRT**, so we work around TRT 8.5.2's ONNX parser limitations instead (see §5, head-stripping step).

The `nvidia-jetpack` meta-package is not required; CUDA, cuDNN, TRT libraries, and OpenCV are pre-installed from the factory flash.

#### 3. Upgrade CMake (3.16 → ≥ 3.18)

The project requires CMake 3.18+ for `find_package(CUDAToolkit)`. Pick one:

```bash
# Option A (user-space, simplest)
pip3 install --user cmake
export PATH=$HOME/.local/bin:$PATH

# Option B (system, Kitware apt repo)
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | \
    gpg --dearmor | sudo tee /usr/share/keyrings/kitware-archive-keyring.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ focal main" | \
    sudo tee /etc/apt/sources.list.d/kitware.list
sudo apt update && sudo apt install -y cmake
```

#### 4. Build

```bash
cd ~/traffic-light/inference/cpp
export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
# produces: build/tl_demo
```

#### 5. Build the TensorRT engine *on the Orin*

TensorRT engines are **not portable** across GPU architectures — the `.engine` file must be produced on the target device. The YOLO26 ONNX must also be stripped of its in-graph NMS-free head before TRT 8.5.2 will parse it.

##### Why the strip step exists

TRT 8.5.2's ONNX parser has a shape-propagation bug around `ReduceMax → TopK`: it fails with

```
ERROR: onnx2trt_utils.cpp:345 In function convertAxis:
[8] Assertion failed: (axis >= 0 && axis <= nbDims)
```

even on a valid axis. YOLO26's head emits exactly this pattern. The in-graph `TopK` is a GPU-side pre-selection optimization — YOLO26 is NMS-free by training (1:1 anchor matching enforced in the loss), so no NMS is needed at inference. Cutting the head at the final `Concat([box_xywh, sigmoid(class_scores)], dim=1)` exposes raw `[1, 4+nc, N]` output; the C++ postprocess (`inference/cpp/src/trt_pipeline.cpp`) already handles this format with CPU-side confidence thresholding.

##### Export → strip → build

1. On the Mac, export `.pt` → `.onnx`:
   ```bash
   uv run python main.py export runs/detect/yolo26n-r1/weights/best.pt --format onnx
   ```
2. Rewrite the ONNX to drop the NMS-free head:
   ```bash
   uv run python scripts/strip_yolo26_head.py \
       runs/detect/yolo26n-r1/weights/best.onnx \
       runs/detect/yolo26n-r1/weights/best_stripped.onnx \
       --num-classes 7
   ```
   Output shows the cut node, e.g. `cut at /model.23/Concat: output /model.23/Concat_output_0 shape [1, 11, 21504]`.
3. Ship to the Orin:
   ```bash
   rsync -avz runs/detect/yolo26n-r1/weights/best_stripped.onnx \
       nvidia@192.168.30.138:traffic-light/runs/yolo26n-r1/
   ```
4. On the Orin, build the engine:
   ```bash
   /usr/src/tensorrt/bin/trtexec \
       --onnx=runs/yolo26n-r1/best_stripped.onnx \
       --saveEngine=runs/yolo26n-r1/best.engine \
       --fp16
   ```

#### 6. Run

```bash
./build/tl_demo --source /path/to/video.mp4 --model weights/best.engine
./build/tl_demo --source 0 --model weights/best.engine --no-show --save out.mp4
```

### Input resolution

The pipeline accepts video of any resolution or aspect ratio. Every frame is letterboxed (aspect-preserving resize + gray padding) to the engine's fixed `imgsz × imgsz` (default 1280 × 1280). Phone clips, 4K footage, portrait orientation, and sub-`imgsz` inputs all work transparently.

- **Engine input size is baked in at export time.** The demo's `--imgsz` flag is a sanity-check override only — if it disagrees with the engine, the engine wins (you'll see a `[TRT] warning` on startup). To run at a different size, re-export: `python main.py export ... --imgsz <N> --format onnx` and rebuild the `.engine`.
- **Input smaller than `imgsz`** (e.g., 720p into a 1280 engine): letterbox upsamples with bilinear interpolation. No failure, but detail can't be invented — small/distant traffic lights that were blurry in the source stay blurry.
- **Input larger than `imgsz`** (e.g., 4K or portrait phone footage into a 1280 engine): letterbox downsamples to fit, so very small or distant lights may be missed. Either crop the source to landscape before inference, or export a larger-`imgsz` engine (1600, 1920) — at the cost of higher per-frame latency.

### Deployment status

- [x] Python pipeline + demo implemented and tested on M4 Pro
- [x] C++ pipeline + demo implemented, mirrors Python semantics
- [x] Source synced to Orin; CUDA/TRT/OpenCV present from factory flash
- [x] CMake upgrade on Orin (3.16 → ≥ 3.18)
- [x] First successful `cmake --build` on Orin
- [x] YOLO26 ONNX stripped of in-graph NMS head (`scripts/strip_yolo26_head.py`)
- [x] `.engine` built on-device from stripped ONNX via `trtexec`
- [x] End-to-end C++ demo run on Orin

