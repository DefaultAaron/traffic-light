# Scripts Reference

Single index for everything under `scripts/`. Each script's own header
docstring is authoritative — this file is the lookup table.

## Layout

```
scripts/
├── training/         train_deim.sh, train_yolov13.sh
├── dataset/          yolo_to_coco.py
├── export/           _export_deim_onnx.py, strip_yolo26_head.py
├── eval/             _remote_deim_eval.sh, _parse_deim_per_class.py
├── tracker/          measure_flicker.py, validate_flicker_reduction.py
├── ops/              setup_reverse_tunnel.sh
└── (root)            run_demos.sh, export_yolo.sh, export_deim.sh,
                      _r2_*.py / _r2_*.json, _r2_schema_utils.py,
                      _r2_schemas_test.py, _tsm_activation_schema.json
```

> **Why some scripts stay at root**: `run_demos.sh`, `export_yolo.sh`,
> `export_deim.sh`, and all `_r2_*` artifacts are pinned by the locked
> R2 precision-parity plan (`~/.claude/plans/elegant-sauteeing-quail.md`).
> `_tsm_activation_schema.json` is pinned by the TSM scaffold v1.1.
> Moving any of those requires plan amendment.

Quick map:

| Area | Scripts |
|---|---|
| [Demo sweep (Orin TRT)](#demo-sweep-orin-trt) | `run_demos.sh` |
| [Training](#training) | `training/train_deim.sh`, `training/train_yolov13.sh` |
| [DEIM per-class evaluation (post-training)](#deim-per-class-evaluation-post-training) | `eval/_remote_deim_eval.sh`, `eval/_parse_deim_per_class.py` |
| [Dataset (R2 self-collected)](#dataset-r2-self-collected) | `dataset/yolo_to_coco.py` |
| [Model export](#model-export) | `export/strip_yolo26_head.py`, `export/_export_deim_onnx.py`, `export_yolo.sh`, `export_deim.sh` |
| [Flicker / tracker validation](#flicker--tracker-validation) | `tracker/measure_flicker.py`, `tracker/validate_flicker_reduction.py` |
| [Network / ops](#network--ops) | `ops/setup_reverse_tunnel.sh` |

> **R1 dataset scripts retired** (2026-05-08): `annotate_bstld.py`, `annotate_s2tld.py`, `convert_bstld.py`, `convert_lisa.py`, `convert_s2tld.py`, `merge_datasets.py` were removed alongside R1 dataset abandonment per R2 data-replacement policy. R2 uses self-collected data only; the active workflow lives in `docs/data/r2_data_collection_sop.md`. Original R1 docs preserved under `docs/_archive/`.

> **One-shot R1-closure scripts retired** (2026-05-12): `_debug_a2b_kd.sh` (KD A2b distributed-init hang diagnostic; underlying bug fixed in commit `1ad1c2b fix(kd): disable HGNetv2 stage1 preload on teacher build`) and `_diag_deim_dets.sh` (one-time DEIM postprocess residual-det diagnostic for commit `8a3ece8` per-query letterbox dedup) were `git rm`'d as one-shot diagnostics with no live callers. `_deim_eval_diff_audit.py` was relocated to `docs/reports/r1_evidence/_deim_eval_diff_audit.py` to colocate the one-shot R1-closure equivalence reproducer with its output JSON `deim_eval_old_vs_new_diff.json`.

> **Functional-subfolder reorg** (2026-05-12): scripts were grouped into `training/`, `dataset/`, `export/`, `eval/`, `tracker/`, `ops/`. Pinned scripts (`run_demos.sh`, `export_yolo.sh`, `export_deim.sh`, all `_r2_*`, `_tsm_activation_schema.json`) stayed at root per locked plan / scaffold contracts. Inter-script callers and prose docs were updated in the same commit.

---

## Demo sweep (Orin TRT)

`run_demos.sh` runs every `(run × engine × demo)` triple through `tl_demo`
and writes annotated `.mp4` outputs. Detector-only and tracker modes
coexist under the same `OUT_DIR` because tracker mode auto-suffixes the
run dir with `_tracker`. Sequential by design — each call holds the
GPU/TRT context.

```bash
./scripts/run_demos.sh                    # detector-only → demo/<run>/<engine>/<demo>.mp4
TRACK=1 ./scripts/run_demos.sh            # tracker     → demo/<run>_tracker/<engine>/<demo>.mp4
```

### Env overrides (always available)

| Env var | Default | Effect |
|---|---|---|
| `TL_DEMO` | `inference/cpp/build/tl_demo` | Path to the `tl_demo` binary |
| `RUNS_DIR` | `runs` | Root containing `<run>/*.engine` |
| `DEMOS_DIR` | `demo` | Root containing `demo*.mp4` clips |
| `OUT_DIR` | `$DEMOS_DIR` | Output root |
| `CONF` | `0.25` | Detector confidence threshold |
| `TRACK` | `0` | `1` enables `--track` and the `_tracker` suffix |
| `SKIP_EXIST` | `1` | `1` skips outputs already on disk (resume-friendly) |
| `OVERWRITE` | `0` | `1` deletes stale output (and any companion `.tracks.jsonl`) before re-running; forces `SKIP_EXIST=0` |
| `ENGINE_FILTER` | empty | engine stem 子串匹配；仅扫匹配的 engine（例：`_fp32` 只扫 FP32 对照引擎） |
| `ENGINE_EXCLUDE` | empty | engine stem 子串匹配；命中的 engine 被跳过（例：`_fp32` 在生产扫描时排除 FP32 对照引擎） |

### Tracker tuning (`TRACK=1` only)

Unset env vars fall through to the `tl_demo` built-in defaults.

| Env var | Effect |
|---|---|
| `ALPHA` | Tracker EMA alpha |
| `MIN_HITS` | Min hits before a track is confirmed |
| `HIGH_THRESH` | First-pass IoU/score threshold |
| `MATCH_THRESH` | Match cost cutoff |
| `TRACK_BUFFER` | Frames a lost track survives |
| `SAVE_TRACK_JSON` | `1` writes `<demo>.tracks.jsonl` next to the mp4 |

Resolution is inferred from engine filename: `*1536* → 1536`, `*1280* →
1280`, otherwise `640`.

### Common combinations

```bash
# Resume — only fills in missing outputs.
./scripts/run_demos.sh

# Force regenerate everything (e.g. after retraining).
OVERWRITE=1 ./scripts/run_demos.sh

# Tracker sweep, dump parity JSONL alongside each mp4, fresh output dir.
TRACK=1 SAVE_TRACK_JSON=1 \
    RUNS_DIR=runs/yolo26m-r1 OUT_DIR=demo_v2 \
    ./scripts/run_demos.sh

# Higher confidence tracker sweep, regenerate.
TRACK=1 CONF=0.35 OVERWRITE=1 \
    ./scripts/run_demos.sh

# 生产 FP16 单独扫一遍（跳过同目录下的 _fp32 对照引擎）。
ENGINE_EXCLUDE=_fp32 ./scripts/run_demos.sh

# 仅扫 FP32 对照引擎（导出后做 FP32↔FP16 demo 视觉差异校核）。
ENGINE_FILTER=_fp32 ./scripts/run_demos.sh
```

### Output formats

`tl_demo --save` writes the annotated `.mp4`. With `TRACK=1
SAVE_TRACK_JSON=1`, a parity `<demo>.tracks.jsonl` is also written. The
JSONL schema (`{"frame": N, "tracks": [...]}`) matches `inference/demo.py
--track-json`, so Python and C++ outputs are diffable for parity.

For flicker analysis, see [`measure_flicker.py`](#flicker--tracker-validation)
below — it reads a different schema (`--json`, not `--track-json`), so
generate it with a one-off `inference/demo.py` invocation rather than
the sweep.

---

## Training

### `train_yolov13.sh`

YOLOv13 ships custom modules (DSC3k2, HyperACE) that need its own
Ultralytics fork — keep it in a separate venv:

```bash
git clone https://github.com/iMoonLab/yolov13.git
uv venv yolov13/.venv --python 3.11
source yolov13/.venv/bin/activate
uv pip install -r yolov13/requirements.txt
uv pip install -e yolov13/

# Verify
python -c "from ultralytics.nn.modules.block import DSC3k2; print('DSC3k2 OK')"
```

Then (extra args are forwarded to Ultralytics `yolo train`, which uses
`key=value` syntax — not `--flag`):

```bash
./scripts/training/train_yolov13.sh s
./scripts/training/train_yolov13.sh s imgsz=1280 epochs=100
```

Weights expected at `weights/yolov13{n,s,m,l}.pt`. **Python 3.11 only**
(the upstream `requirements.txt` pins cp311-only wheels).

### `train_deim.sh`

DEIM-D-FINE training. Needs the COCO-format dataset first
(`uv run python scripts/dataset/yolo_to_coco.py`). Single-GPU is the default;
use `NPROC=N` for multi-GPU. Extra args after the size are forwarded to
DEIM's `train.py`, **not** to `torchrun` — so don't pass torchrun flags
like `--nproc_per_node` here.

```bash
# Single GPU (default)
./scripts/training/train_deim.sh s

# Multi-GPU
NPROC=4 ./scripts/training/train_deim.sh s

# Fine-tune from COCO checkpoint (-t is a train.py flag)
./scripts/training/train_deim.sh m -t weights/deim_dfine_m_coco.pth
```

DEIM uses its own standalone venv on the training server. The main
project venv (`uv` workspace at the repo root) is YOLO26-only — do not
add DEIM torch/torchvision pins to project `pyproject.toml`.

The script auto-activates `DEIM/.venv` if present (after `cd DEIM`), so
it works whether invoked from an already-activated shell or routed
through `uv run` (which would otherwise force the project venv into the
subprocess env and shadow any prior `source DEIM/.venv/bin/activate`).
Override the venv path with `DEIM_VENV=...` (relative to `DEIM/`); set
it to a non-existent path to skip auto-activation.

---

## DEIM per-class evaluation (post-training)

After DEIM-D-FINE training completes, `eval/latest.pth` only captures the
**last** epoch's COCOeval pickle — not the deployment-eligible best
checkpoint (`best_stg2.pth` for S/M; `best_stg1.pth` for L when stage-2
did not surpass stage-1). For unified-methodology per-class AP / P / R
across the DEIM family, re-eval the deployment ckpt directly:

### `_remote_deim_eval.sh`

Runs `DEIM/train.py --test-only` on the deployment-eligible best
checkpoint for DEIM-S, DEIM-M, DEIM-L and parses the resulting
`eval.pth` into a paste-ready per-class table.

```bash
bash scripts/eval/_remote_deim_eval.sh
```

No CLI args. Outputs land under `logs/deim_eval_<size>/`:

| File | Contents |
|------|----------|
| `test_only.log` | DEIM stdout/stderr (COCO 12-tuple at the bottom) |
| `eval.pth` | pickled `coco_evaluator.coco_eval['bbox'].eval` — precision/recall arrays |
| `per_class.json` | machine-readable per-class table |
| `per_class.txt` | markdown table ready to paste into `docs/reports/phase_*_results.md` |

Auto-activates `DEIM/.venv` and runs `scripts/dataset/yolo_to_coco.py --splits val`
if `data/merged/annotations/instances_val.json` is missing. Runs on the
training rig (4090 D ≈ 10 min total).

### `_parse_deim_per_class.py`

Standalone parser, invoked by `_remote_deim_eval.sh`. Reads
`eval.pth` + COCO val JSON + `data/traffic_light.yaml` and emits the
per-class table (P/R at best-F1 point on the IoU=0.5 PR curve; overall
row = uniform mean across the 7 classes — matches the existing
results-doc convention).

```bash
python scripts/eval/_parse_deim_per_class.py \
    --eval-pth logs/deim_eval_s/eval.pth \
    --ann-json data/merged/annotations/instances_val.json \
    --data-yaml data/traffic_light.yaml \
    --out-json logs/deim_eval_s/per_class.json \
    --out-txt  logs/deim_eval_s/per_class.txt
```

Use directly if you need to re-parse without re-running `--test-only`.

---

## Dataset (R2 self-collected)

R2 uses self-collected data exclusively (R1 datasets retired — see banner
at top of this file). The dataset prep workflow now consists of just one
step: convert YOLO labels to COCO format for DEIM training.

```bash
uv run python scripts/dataset/yolo_to_coco.py
uv run python scripts/dataset/yolo_to_coco.py --splits train val   # explicit
```

Reads `data/merged/{images,labels}/{train,val}/` and writes
`data/merged/annotations/instances_{train,val}.json`. Categories are
0-indexed to match `traffic_light.yaml` (so set DEIM's
`remap_mscoco_category: False`, `num_classes: 7`).

For the upstream collection + labeling SOP, see
`docs/data/r2_data_collection_sop.md`.

---

## Model export

### `strip_yolo26_head.py`

YOLO26's in-graph NMS-free head can't be parsed by TRT 8.5.2 (JetPack
5.1). This script rewrites the exported ONNX so the final output is the
head `Concat` (`[1, 4+nc, N]` of decoded `xyxy || sigmoid class scores`).
The C++ pipeline already decodes that directly.

```bash
uv run python scripts/export/strip_yolo26_head.py best.onnx best_stripped.onnx
uv run python scripts/export/strip_yolo26_head.py best.onnx best_stripped.onnx --num-classes 7
```

Then on the Orin: `trtexec --onnx=best_stripped.onnx --saveEngine=...
--fp16`.

### `export_yolo.sh`

Export an Ultralytics YOLO (R2: YOLO26) checkpoint to ONNX, then
optionally to a TensorRT engine on the Orin. Mirrors the
`export_deim.sh` contract for size selection, FP16/FP32 precision,
`_fp32` suffix, and the `<engine>.meta.json` sidecar. The pipeline is
**three-stage**: `yolo export format=onnx` → `strip_yolo26_head.py`
(removes the in-graph `ReduceMax → TopK` that TRT 8.5.2 cannot parse) →
`trtexec` directly on the stripped ONNX. We do **not** use
`yolo export format=engine` because its unified pipeline skips the strip
and produces engines that either fail to build on JetPack 5.1 or yield
graphs the C++ pipeline cannot decode. The shared
`inference/trt_pipeline.{py,cpp}` auto-detects the YOLO arch from tensor
shapes — no flag needed downstream.

```bash
# ONNX only (host CPU is fine; Ultralytics venv must be active)
scripts/export_yolo.sh s runs/yolo26_s-r1/weights/best.pt

# ONNX + engine in one shot (run on Orin)
scripts/export_yolo.sh s runs/yolo26_s-r1/weights/best.pt --build-engine
```

Outputs (next to the input `.pt`):

| File | When emitted | Purpose |
|---|---|---|
| `best.onnx`              | Always | Full-head ONNX (provenance / pre-strip source artifact). Not the file the C++/Python decoders read — both expect the stripped raw `[1, 4+nc, N]` shape. |
| `best_stripped.onnx`     | Always | Head-stripped ONNX (raw `[1, 4+nc, N]`); the artifact `trtexec` consumes AND the artifact to use for Python ONNXRuntime parity testing against the TRT engine. **Note**: written even without `--build-engine` — its presence does NOT imply an engine has been built; check for `<engine>.meta.json` to confirm a trusted engine exists. |
| `best.engine`            | `--build-engine` and `FP16=1` (default) | Orin TRT production engine. **Existing engines are not overwritten** — delete the prior engine before re-building the same precision. |
| `best_fp32.engine`       | `--build-engine` and `FP16=0` | FP32↔FP16 parity comparison only; coexists with the FP16 engine. |
| `<engine>.meta.json`     | After every successful engine build | Atomic provenance sidecar (see [Engine sidecar contract](#engine-sidecar-contract) below). Records `num_classes` and `imgsz` as first-class fields; `imgsz` is consumed by `run_demos.sh`. |

Env overrides:

| Var | Default | Effect |
|---|---|---|
| `YOLO_BIN`              | `yolo`     | Path to the `yolo` CLI (looks up on `PATH`) |
| `PYTHON`                | auto-detect (`python` → `python3`) | Interpreter used for `onnx.checker`, `strip_yolo26_head.py`, and the JSON sidecar writer. Preflight asserts both `import onnx` and `import onnx_graphsurgeon` succeed before any expensive export. |
| `TRTEXEC`               | `trtexec`  | Path to the `trtexec` binary. Required when `--build-engine` is passed. |
| `FP16`                  | `1`        | `0` builds an `_fp32.engine` instead of overwriting the FP16 production engine |
| `SKIP_EXPORT`           | `0`        | `1` reuses an existing `.onnx` if it validates. The stripped ONNX is reused only if it is **also at least as new as the `.onnx`**; if the `.onnx` was just refreshed (or the stripped artifact is stale / corrupt), the strip step re-runs automatically (sub-second cost). The engine itself is always rebuilt when `--build-engine` is set. |
| `WORKSPACE_GB`          | `4`        | Converted to MB for `trtexec --memPoolSize=workspace:N`. **Note the unit difference: YOLO surface uses GB, DEIM surface uses MB.** |
| `ALLOW_LARGE_WORKSPACE` | `0`        | `1` bypasses the 32 GB sanity cap on `WORKSPACE_GB` (defends against accidental MB→GB unit confusion). Recorded in the sidecar's `allow_large_workspace` field. |
| `ALLOW_NON_YOLO26`      | `0`        | The script gates on a `/yolo26<…>` path segment in the checkpoint path. R2 only validated YOLO26. |
| `IMGSZ`                 | unset (auto-detect from `.pt`) | Override input size. The script auto-detects from the `.pt`'s baked-in `args.imgsz`; failure to detect is a hard error (no silent default), so set both `IMGSZ` and `NUM_CLASSES` explicitly when working with a checkpoint whose metadata is unusual. |
| `NUM_CLASSES`           | unset (auto-detect from `model.names`) | Override class count passed to `strip_yolo26_head.py --num-classes`. **Important**: `NUM_CLASSES` at export MUST match the runtime `CLASS_NAMES` size in `inference/trt_pipeline.py` and the C++ `kClassNames` size in `inference/cpp/src/trt_pipeline.cpp`. The sidecar records the value for traceability but inference does not yet read it. |

FP16 vs FP32 parity workflow:

```bash
# Build 1: FP16 production engine
scripts/export_yolo.sh s runs/yolo26_s-r1/weights/best.pt --build-engine

# Build 2: FP32 reference engine. SKIP_EXPORT=1 reuses the .onnx and
#          _stripped.onnx from build 1 (only the engine is rebuilt at FP32).
FP16=0 SKIP_EXPORT=1 scripts/export_yolo.sh s runs/yolo26_s-r1/weights/best.pt --build-engine
```

> **YOLO26 head-strip is integrated into both `main.py export` and
> `export_yolo.sh`** — R1 required a manual
> `scripts/export/strip_yolo26_head.py` pass on the ONNX before `trtexec`; both
> wrappers now run that pass automatically and point `trtexec` at
> `best_stripped.onnx`. The strip script remains on disk for legacy /
> debug use (working with hand-built ONNX outside the wrappers).

### `export_deim.sh`

Export a trained DEIM-D-FINE checkpoint to ONNX, then optionally invoke
`trtexec` on the Orin to build the `.engine`. The exported graph
includes `model.deploy() + postprocessor.deploy()`, so inference emits
`labels / boxes / scores` directly; the shared
`inference/trt_pipeline.{py,cpp}` auto-detects DEIM by tensor names, no
flag needed.

```bash
# Step 1 (DEIM venv, CPU is fine): export ONNX
scripts/export_deim.sh s runs/detect/deim_dfine_s-r1/best_stg2.pth

# Step 2 (Orin): build engine in the same call
scripts/export_deim.sh s runs/detect/deim_dfine_s-r1/best_stg2.pth --build-engine
```

Outputs (next to the input `.pth`):

| File | When emitted | Purpose |
|---|---|---|
| `best_stg2.onnx`         | Always; `SKIP_EXPORT=1` reuses if `.onnx`+`.imgsz` both exist and pass `onnx.checker` + shape cross-check | Python ONNXRuntime / cross-host transfer |
| `best_stg2.onnx.imgsz`   | Paired with the `.onnx` | Sidecar carrying the spatial size baked into the graph |
| `best_stg2.engine`       | `--build-engine` and `FP16=1` (default) | Orin TRT production engine |
| `best_stg2_fp32.engine`  | `--build-engine` and `FP16=0` | FP32↔FP16 parity comparison only; coexists with FP16 engine |
| `<engine>.meta.json`     | After every successful engine build | Atomic provenance sidecar (see [Engine sidecar contract](#engine-sidecar-contract) below) |

Env overrides:

| Var | Default | Effect |
|---|---|---|
| `PYTHON`                 | `python`   | DEIM venv interpreter (on the GPU server, `source DEIM/.venv/bin/activate` first or pass `PYTHON=...`) |
| `FP16`                   | `1`        | `--build-engine` adds `--fp16` to `trtexec`; `0` outputs `<ckpt>_fp32.engine` |
| `SKIP_EXPORT`            | `0`        | `1` skips `pth → onnx` when `.onnx`+`.imgsz` exist AND the cached pair passes `onnx.checker` + a shape cross-check (`.imgsz` value vs the ONNX `images` input dim). A stale `.imgsz` next to a fresh `.onnx` aborts with a refresh hint instead of silently building a mismatched engine. |
| `WORKSPACE_MB`           | `4096`     | `trtexec --memPoolSize=workspace:N`. **Note the unit difference: DEIM uses MB, YOLO uses GB.** |
| `ALLOW_SMALL_WORKSPACE`  | `0`        | `1` bypasses the 256 MB sanity floor on `WORKSPACE_MB` (defends against accidental GB→MB unit confusion when copying flags from `export_yolo.sh` — `WORKSPACE_GB=4` interpreted as MB would silently change builder tactics). |
| `TRTEXEC`                | `trtexec` (PATH), fallback `/usr/src/tensorrt/bin/trtexec` | Custom `trtexec` path. **An explicitly set but missing `TRTEXEC` is a hard error** — the script refuses to silently fall back, because engine provenance is correctness-critical for Python↔C++ TRT parity. |

FP16 vs FP32 parity workflow (build FP16 first, reuse the same `.onnx` for FP32):

```bash
# Build 1: FP16 production engine
scripts/export_deim.sh s runs/.../best_stg2.pth --build-engine

# Build 2: FP32 reference engine — reuses the .onnx from Build 1 (cheap)
SKIP_EXPORT=1 FP16=0 scripts/export_deim.sh s runs/.../best_stg2.pth --build-engine
```

> **No `IMGSZ` override.** The DEIM `traffic_light` config chain
> (`base/dfine_hgnetv2.yml`) declares `eval_spatial_size`; the wrapper
> `scripts/export/_export_deim_onnx.py` reads that field and writes
> `<ckpt>.onnx.imgsz`, then the bash side feeds it into `trtexec
> --shapes`. To switch to 1280 etc., follow the config comments in
> `deim_hgnetv2_s_traffic_light.yml` (lines 13–18) — retrain end-to-end.
> Changing only the export size produces a model with collapsed
> accuracy.
>
> Also: DEIM marks the batch axis as dynamic by default. The wrapper
> pins `min=opt=max=1` to let TRT specialize kernels for batch-1
> deployment. Multi-batch deployment would need a different opt/max.

#### Engine sidecar contract

Both `export_yolo.sh` and `export_deim.sh` write a JSON sidecar at
`<engine>.meta.json` after a successful build. The contract is:

- **Atomic** — written to `<engine>.meta.json.tmp`, fsync'd, then
  `os.replace`'d into place. Readers never see a partial file.
- **Tied to engine integrity** — on any failure or interrupt before the
  sidecar lands, the script's cleanup trap removes BOTH the engine and
  any partial sidecar. The contract: **engine without a valid sidecar
  is treated as untrusted and never appears on disk**.
- **Size-stable** — engine SHA256 is computed only after the file size
  is identical across two reads 0.5 s apart (handles slow flushes from
  the underlying `trtexec` child).

Schema (shared fields between YOLO and DEIM sidecars):

| Field | YOLO | DEIM | Notes |
|---|---|---|---|
| `precision`              | `fp16` / `fp32` | `fp16` / `fp32` | |
| `exporter`               | `ultralytics yolo + strip_yolo26_head + trtexec` | `trtexec` | Which tools produced the engine. YOLO is now a three-stage pipeline (export ONNX → strip head → trtexec) so the field reflects that; DEIM is single-call trtexec. |
| `exporter_cmdline`       | full `yolo export … && python strip_yolo26_head.py …` (shlex-joined) | `null` | YOLO records the ONNX-export + strip-script composite |
| `trtexec_cmdline`        | full `trtexec …` (shlex-joined) | `shlex.join` of the actual trtexec call | Both wrappers now record paste-runnable `trtexec` invocations |
| `trt_version`            | best-effort | best-effort | `unknown` if the probe fails |
| `cuda_version`           | best-effort | best-effort | parsed from `nvcc --version` |
| `jetpack_version`        | best-effort | best-effort | parsed from `/etc/nv_tegra_release` |
| `build_host`             | `hostname` | `hostname` | |
| `build_timestamp`        | UTC ISO-8601 | UTC ISO-8601 | |
| `source_pt[h]`           | `source_pt`, `source_pt_sha256` | `source_pth`, `source_pth_sha256` | absolute path + content hash |
| `source_onnx_sha256`     | required (full-head ONNX validated) | required (post-export ONNX validated) | provenance artifact |
| `source_onnx_stripped_sha256` | required (head-stripped ONNX validated) | n/a | YOLO-only — the ONNX `trtexec` actually consumed |
| `num_classes`            | required (read from `.pt` or `NUM_CLASSES` env) | n/a | YOLO-only first-class field; carries the value passed to `strip_yolo26_head.py --num-classes`. **Must match the runtime `CLASS_NAMES` size in `inference/trt_pipeline.{py,cpp}`.** |
| `imgsz`                  | required (read from `.pt` or `IMGSZ` env) | required (from `.imgsz` sidecar) | Both now carry spatial size explicitly. `run_demos.sh` reads this field; legacy filename heuristic only applies when the sidecar is absent. |
| `engine_sha256`          | required | required | computed after size-stability protocol passes |
| `engine_size_bytes`      | required | required | |
| `workspace_gb` / `workspace_mb` | `workspace_gb`, `allow_large_workspace` | `workspace_mb` | YOLO carries the `ALLOW_LARGE_WORKSPACE` decision so audits can distinguish intentional 64 GB from unit-confusion |

---

## Flicker / tracker validation

### `measure_flicker.py`

Computes per-track raw-vs-smoothed class-flip counts. **Reads
`inference/demo.py --json` output**, where each record is
`{"frame": N, "detections": [...]}` and tracked detections carry
`tracking_id` + `raw_class_id`. The sweep scripts' `--track-json`
output uses a different schema (`tracks[]`) and is **not** compatible —
feeding `.tracks.jsonl` here would silently report 0 detections.

```bash
# Generate the right schema with a one-off demo run
python -m inference.demo --source demo/demo.mp4 --model weights/best.engine \
    --track --no-show --json > /tmp/tracked.jsonl

# Then analyze
uv run python scripts/tracker/measure_flicker.py /tmp/tracked.jsonl
uv run python scripts/tracker/measure_flicker.py /tmp/tracked.jsonl --dump-per-track
```

Reads from stdin if no path given (`-`).

### `validate_flicker_reduction.py`

Synthetic stress test for `TrackSmoother`. Drives 300 frames of a
deterministically noisy detection stream and reports raw-vs-smoothed flip
reduction against the ≥50% gate.

```bash
uv run python scripts/tracker/validate_flicker_reduction.py
uv run python scripts/tracker/validate_flicker_reduction.py --seed 1 --flip-rate 0.4
```

Defaults: `--frames 300 --flip-rate 0.3 --seed 0`. Useful when the real
demo footage is too clean to exercise the reduction path.

---

## Network / ops

Workaround for the unstable campus network — see
[`tailscale_runbook.md`](tailscale_runbook.md) for the underlying root
cause and triage.

### `setup_reverse_tunnel.sh`

Installs an `autossh` reverse tunnel (systemd-managed) from this host to
a public VPS, so SSH still works when Tailscale's control plane is being
throttled. This is the production workaround.

```bash
sudo VPS_HOST=vps.example.com VPS_USER=ubuntu VPS_PORT=22 \
     REMOTE_PORT=2222 LOCAL_SSH_PORT=22 \
     bash scripts/ops/setup_reverse_tunnel.sh

sudo bash scripts/ops/setup_reverse_tunnel.sh --status
sudo bash scripts/ops/setup_reverse_tunnel.sh --disable
```

After setup: `ssh -p 2222 jun-user@vps.example.com`. VPS sshd needs
`GatewayPorts yes`.

---

## When in doubt

Each script's top-of-file docstring is the authoritative source of truth
— this index just points you to the right one. If something here drifts
from a script's actual behavior, the script wins.
