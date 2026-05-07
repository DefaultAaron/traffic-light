#pragma once

// TensorRT inference pipeline for traffic-light detection.
//
// Two architectures are auto-detected at engine load time:
//   - YOLO26   : 1 input "images", 1 output (1, N, 4+nc) or (1, 4+nc, N)
//   - DEIM-D-FINE : 2 inputs "images" + "orig_target_sizes",
//                   3 outputs "labels" + "boxes" + "scores"
//                   (deploy postprocessor bakes top-K + sigmoid in-graph)
//
// Public API (Detection POD, detect()) is identical for both arches so
// demo.cpp / tracker / run_demos.sh do not need to know which model is
// loaded.
//
// Thread safety: TRTDetector is NOT thread-safe. It owns a single CUDA
// stream and pre-allocated pinned host buffers that are reused across
// `detect()` calls. Create one detector per worker thread, or serialize
// access externally.

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "detection.hpp"

// Forward-declare TensorRT / CUDA types so callers don't need the headers.
namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
class ILogger;
}
struct CUstream_st;
typedef struct CUstream_st* cudaStream_t;

namespace tl {

enum class DetectorArch {
    kYOLO,   // YOLO26 stripped head: 1×{4+nc, N} concat output
    kDEIM,   // DEIM-D-FINE deploy graph: labels + boxes + scores
};

class TRTDetector {
public:
    TRTDetector(const std::string& model_path,
                float conf_thresh = 0.25f,
                int imgsz = 1280);
    ~TRTDetector();

    TRTDetector(const TRTDetector&) = delete;
    TRTDetector& operator=(const TRTDetector&) = delete;

    // Run detection on a BGR frame. Both supported arches are NMS-free at
    // the public surface (YOLO26 by training; DEIM by the deploy top-K) —
    // no post-hoc NMS is applied here.
    std::vector<Detection> detect(const cv::Mat& frame);

    float conf_thresh() const { return conf_thresh_; }
    int imgsz() const { return imgsz_; }
    DetectorArch arch() const { return arch_; }

private:
    struct TensorBuf {
        std::string name;
        std::vector<int64_t> shape;
        size_t elem_count = 0;
        size_t elem_size = 0;  // bytes per element
        // Explicit dtype flags — postprocess uses these (not elem_size) so
        // INT32 (4 bytes) doesn't get reinterpreted as FP32.
        bool is_fp32 = false;
        bool is_fp16 = false;
        bool is_int64 = false;
        bool is_int32 = false;
        void* device = nullptr;
        void* host = nullptr;
        int binding_index = -1;  // TRT 8.x only; -1 on TRT 10+
    };

    void loadEngine(const std::string& path);
    void allocateBuffers();
    void detectArch();
    void freeAll() noexcept;
    void preprocess(const cv::Mat& frame, float& scale, float& pad_w, float& pad_h);
    void fillOrigTargetSizes();  // DEIM only — writes [imgsz, imgsz] into the int64 input.
    std::vector<Detection> postprocessYolo(const cv::Mat& orig,
                                           float scale, float pad_w, float pad_h);
    std::vector<Detection> postprocessDeim(const cv::Mat& orig,
                                           float scale, float pad_w, float pad_h);

    const TensorBuf* findOutput(const char* name) const;
    // Prefers tensor named "images"; falls back to first 4-D input with C=3.
    // Returns nullptr if no NCHW/C=3 candidate exists.
    TensorBuf* findImageInput() noexcept;

    float conf_thresh_;
    int imgsz_;
    DetectorArch arch_ = DetectorArch::kYOLO;

    std::unique_ptr<nvinfer1::ILogger> logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_ = nullptr;

    std::vector<TensorBuf> inputs_;
    std::vector<TensorBuf> outputs_;

    // TRT 8.x enqueueV2 binding-indexed pointer array; unused on TRT 10+.
    std::vector<void*> binding_ptrs_;

    // FP16 → float32 expansion scratch (reused per frame).
    std::vector<float> fp16_scratch_;
    std::vector<float> deim_boxes_scratch_;
    std::vector<float> deim_scores_scratch_;
};

}  // namespace tl
