#pragma once

// TensorRT inference pipeline for YOLO26 traffic-light detection.
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

class TRTDetector {
public:
    TRTDetector(const std::string& model_path,
                float conf_thresh = 0.25f,
                int imgsz = 1280);
    ~TRTDetector();

    TRTDetector(const TRTDetector&) = delete;
    TRTDetector& operator=(const TRTDetector&) = delete;

    // Run detection on a BGR frame. YOLO26 is NMS-free — no post-hoc NMS.
    std::vector<Detection> detect(const cv::Mat& frame);

    float conf_thresh() const { return conf_thresh_; }
    int imgsz() const { return imgsz_; }

private:
    struct TensorBuf {
        std::string name;
        std::vector<int64_t> shape;
        size_t elem_count = 0;
        size_t elem_size = 0;  // bytes per element (matches engine dtype)
        bool is_fp16 = false;
        void* device = nullptr;  // cudaMalloc'd
        void* host = nullptr;    // pinned host buffer
        int binding_index = -1;  // TRT 8.x enqueueV2 binding slot; unused on TRT 10+
    };

    void loadEngine(const std::string& path);
    void allocateBuffers();
    void freeAll() noexcept;
    void preprocess(const cv::Mat& frame, float& scale, float& pad_w, float& pad_h);
    std::vector<Detection> postprocess(const cv::Mat& orig, float scale, float pad_w, float pad_h);

    float conf_thresh_;
    int imgsz_;

    std::unique_ptr<nvinfer1::ILogger> logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;
    cudaStream_t stream_ = nullptr;

    std::vector<TensorBuf> inputs_;
    std::vector<TensorBuf> outputs_;

    // TRT 8.x enqueueV2 needs a binding-indexed pointer array. Populated in
    // allocateBuffers() and reused for every detect() call. Unused on TRT 10+.
    std::vector<void*> binding_ptrs_;

    // Reused per-frame buffer for FP16 → float32 expansion during postprocess.
    std::vector<float> fp16_scratch_;
};

}  // namespace tl
