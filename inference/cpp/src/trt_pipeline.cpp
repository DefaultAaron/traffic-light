#include "trt_pipeline.hpp"

#include <cuda_runtime_api.h>
#include <NvInfer.h>

#if !defined(NV_TENSORRT_MAJOR)
#error "NV_TENSORRT_MAJOR not defined — is NvInfer.h included?"
#elif NV_TENSORRT_MAJOR < 8 || (NV_TENSORRT_MAJOR == 8 && NV_TENSORRT_MINOR < 5)
#error "This pipeline requires TensorRT 8.5 or newer."
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace tl {

namespace {

#define CUDA_CHECK(call)                                                                          \
    do {                                                                                          \
        cudaError_t err = (call);                                                                 \
        if (err != cudaSuccess) {                                                                 \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err));      \
        }                                                                                         \
    } while (0)

class TRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << std::endl;
        }
    }
};

size_t dtypeSize(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kBOOL:  return 1;
        case nvinfer1::DataType::kUINT8: return 1;
        default: return 4;
    }
}

// IEEE-754 float32 → float16 with round-to-nearest-even. Handles normal,
// inf, and NaN; flushes subnormals (both directions) to zero — adequate
// for image-range inputs in [0, 1].
uint16_t float32ToHalf(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint16_t sign = static_cast<uint16_t>((x >> 16) & 0x8000);
    int32_t e32 = static_cast<int32_t>((x >> 23) & 0xFF);
    uint32_t m32 = x & 0x7FFFFF;

    if (e32 == 0xFF) {
        // inf or NaN. Preserve NaN-ness by keeping non-zero mantissa.
        uint16_t m16 = static_cast<uint16_t>(m32 >> 13);
        if (m32 != 0 && m16 == 0) m16 = 1;
        return static_cast<uint16_t>(sign | 0x7C00 | m16);
    }
    int e = e32 - 127 + 15;
    if (e >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00);  // overflow → inf
    if (e <= 0)   return sign;                                    // underflow / subnormal → ±0

    uint32_t m_hi = m32 >> 13;
    uint32_t m_lo = m32 & 0x1FFF;
    if (m_lo > 0x1000 || (m_lo == 0x1000 && (m_hi & 1))) {
        m_hi += 1;
        if (m_hi & 0x400) {  // mantissa overflow → bump exponent
            m_hi = 0;
            e += 1;
            if (e >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00);
        }
    }
    return static_cast<uint16_t>(sign | (e << 10) | m_hi);
}

float halfToFloat32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign << 31;
        } else {
            exp = 1;
            while ((mant & 0x400) == 0) { mant <<= 1; --exp; }
            mant &= 0x3FF;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = (sign << 31) | (0xFF << 23) | (mant << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

// Letterbox resize: preserve aspect, pad with 114 gray.
cv::Mat letterbox(const cv::Mat& img, int new_w, int new_h,
                  float& scale, float& pad_w, float& pad_h) {
    int h = img.rows, w = img.cols;
    scale = std::min(static_cast<float>(new_h) / h, static_cast<float>(new_w) / w);
    int unpad_w = static_cast<int>(std::round(w * scale));
    int unpad_h = static_cast<int>(std::round(h * scale));
    pad_w = (new_w - unpad_w) / 2.0f;
    pad_h = (new_h - unpad_h) / 2.0f;

    cv::Mat resized;
    if (unpad_w != w || unpad_h != h) {
        cv::resize(img, resized, cv::Size(unpad_w, unpad_h), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = img;
    }

    int top    = static_cast<int>(std::round(pad_h - 0.1f));
    int bottom = static_cast<int>(std::round(pad_h + 0.1f));
    int left   = static_cast<int>(std::round(pad_w - 0.1f));
    int right  = static_cast<int>(std::round(pad_w + 0.1f));

    cv::Mat out;
    cv::copyMakeBorder(resized, out, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    return out;
}

}  // namespace

TRTDetector::TRTDetector(const std::string& model_path, float conf_thresh, int imgsz)
    : conf_thresh_(conf_thresh), imgsz_(imgsz) {
    if (model_path.size() < 7 || model_path.substr(model_path.size() - 7) != ".engine") {
        throw std::runtime_error(
            "TRTDetector requires a .engine file. For ONNX, use the Python pipeline.");
    }

    try {
        logger_.reset(new TRTLogger());
        CUDA_CHECK(cudaStreamCreate(&stream_));
        loadEngine(model_path);
        allocateBuffers();
    } catch (...) {
        freeAll();
        throw;
    }
}

TRTDetector::~TRTDetector() {
    freeAll();
}

void TRTDetector::freeAll() noexcept {
    auto free_buf = [](TensorBuf& b) {
        if (b.device) { cudaFree(b.device); b.device = nullptr; }
        if (b.host)   { cudaFreeHost(b.host); b.host = nullptr; }
    };
    for (auto& b : inputs_)  free_buf(b);
    for (auto& b : outputs_) free_buf(b);
    inputs_.clear();
    outputs_.clear();
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    // context_, engine_, runtime_ released via unique_ptr (TRT 10 uses standard delete).
}

void TRTDetector::loadEngine(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) throw std::runtime_error("Cannot open engine file: " + path);
    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> blob(size);
    f.read(blob.data(), size);

    runtime_.reset(nvinfer1::createInferRuntime(*logger_));
    if (!runtime_) throw std::runtime_error("Failed to create TRT runtime");

    engine_.reset(runtime_->deserializeCudaEngine(blob.data(), size));
    if (!engine_) throw std::runtime_error("Failed to deserialize engine: " + path);

    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("Failed to create execution context");
}

void TRTDetector::allocateBuffers() {
#if NV_TENSORRT_MAJOR >= 10
    const int n = engine_->getNbIOTensors();

    // Pass 1: set all input shapes (replacing any dynamic dim with 1) so
    // output shapes resolve correctly via the context.
    for (int i = 0; i < n; ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT) continue;
        nvinfer1::Dims dims = engine_->getTensorShape(name);
        for (int k = 0; k < dims.nbDims; ++k) {
            if (dims.d[k] < 0) dims.d[k] = 1;
        }
        if (!context_->setInputShape(name, dims)) {
            throw std::runtime_error(std::string("setInputShape failed for ") + name);
        }
    }

    // Pass 2: allocate device + pinned host for every I/O tensor using the
    // context-resolved shapes.
    for (int i = 0; i < n; ++i) {
        const char* name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        auto dtype = engine_->getTensorDataType(name);
        nvinfer1::Dims dims = context_->getTensorShape(name);

        TensorBuf buf;
        buf.name = name;
        buf.shape.assign(dims.d, dims.d + dims.nbDims);
        for (auto& d : buf.shape) if (d < 0) d = 1;  // belt-and-suspenders
        buf.elem_count = std::accumulate(buf.shape.begin(), buf.shape.end(), size_t{1},
                                         std::multiplies<size_t>());
        buf.elem_size = dtypeSize(dtype);
        buf.is_fp16 = (dtype == nvinfer1::DataType::kHALF);
        const size_t dev_bytes = buf.elem_count * buf.elem_size;

        CUDA_CHECK(cudaMalloc(&buf.device, dev_bytes));
        CUDA_CHECK(cudaMallocHost(&buf.host, dev_bytes));

        context_->setTensorAddress(name, buf.device);

        if (mode == nvinfer1::TensorIOMode::kINPUT) inputs_.push_back(std::move(buf));
        else                                        outputs_.push_back(std::move(buf));
    }
#else
    // TRT 8.x: index-based bindings API + enqueueV2.
    const int n = engine_->getNbBindings();

    // Pass 1: resolve dynamic input dims to 1 via the context.
    for (int i = 0; i < n; ++i) {
        if (!engine_->bindingIsInput(i)) continue;
        nvinfer1::Dims dims = engine_->getBindingDimensions(i);
        for (int k = 0; k < dims.nbDims; ++k) {
            if (dims.d[k] < 0) dims.d[k] = 1;
        }
        if (!context_->setBindingDimensions(i, dims)) {
            throw std::runtime_error("setBindingDimensions failed for binding " +
                                     std::to_string(i));
        }
    }

    // Pass 2: allocate every binding; cache device pointers by binding index
    // for enqueueV2.
    binding_ptrs_.assign(n, nullptr);
    for (int i = 0; i < n; ++i) {
        const char* name = engine_->getBindingName(i);
        bool is_input = engine_->bindingIsInput(i);
        auto dtype = engine_->getBindingDataType(i);
        nvinfer1::Dims dims = context_->getBindingDimensions(i);

        TensorBuf buf;
        buf.name = name ? name : "";
        buf.binding_index = i;
        buf.shape.assign(dims.d, dims.d + dims.nbDims);
        for (auto& d : buf.shape) if (d < 0) d = 1;
        buf.elem_count = std::accumulate(buf.shape.begin(), buf.shape.end(), size_t{1},
                                         std::multiplies<size_t>());
        buf.elem_size = dtypeSize(dtype);
        buf.is_fp16 = (dtype == nvinfer1::DataType::kHALF);
        const size_t dev_bytes = buf.elem_count * buf.elem_size;

        CUDA_CHECK(cudaMalloc(&buf.device, dev_bytes));
        CUDA_CHECK(cudaMallocHost(&buf.host, dev_bytes));
        binding_ptrs_[i] = buf.device;

        if (is_input) inputs_.push_back(std::move(buf));
        else          outputs_.push_back(std::move(buf));
    }
#endif

    if (inputs_.empty() || outputs_.empty()) {
        throw std::runtime_error("Engine has no input or output tensors");
    }

    // Sanity check: input must be NCHW with 3 channels matching imgsz.
    const auto& in_shape = inputs_[0].shape;
    if (in_shape.size() == 4) {
        int h = static_cast<int>(in_shape[2]);
        int w = static_cast<int>(in_shape[3]);
        if (h != imgsz_ || w != imgsz_) {
            std::cerr << "[TRT] warning: engine input " << w << "x" << h
                      << " differs from imgsz=" << imgsz_ << "; using engine size." << std::endl;
            imgsz_ = std::min(h, w);
        }
    }

    // Sanity check: primary output row length must be 4 + num_classes.
    const auto& out_shape = outputs_[0].shape;
    if (out_shape.size() >= 2) {
        int64_t d0 = out_shape.size() >= 3 ? out_shape[1] : out_shape[0];
        int64_t d1 = out_shape.size() >= 3 ? out_shape[2] : out_shape[1];
        const int64_t expected = 4 + static_cast<int64_t>(kClassNames.size());
        if (d0 != expected && d1 != expected) {
            std::cerr << "[TRT] warning: output shape has no dim equal to "
                      << expected << " (got " << d0 << "x" << d1
                      << "); decoder may misinterpret rows." << std::endl;
        }
    }
}

void TRTDetector::preprocess(const cv::Mat& frame, float& scale, float& pad_w, float& pad_h) {
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    cv::Mat lb = letterbox(rgb, imgsz_, imgsz_, scale, pad_w, pad_h);

    cv::Mat lb_f32;
    lb.convertTo(lb_f32, CV_32FC3, 1.0 / 255.0);

    auto& in = inputs_[0];
    const int chw_plane = imgsz_ * imgsz_;

    if (in.is_fp16) {
        std::vector<cv::Mat> ch(3);
        cv::split(lb_f32, ch);
        uint16_t* host = static_cast<uint16_t*>(in.host);
        for (int c = 0; c < 3; ++c) {
            const float* src = reinterpret_cast<const float*>(ch[c].data);
            uint16_t* dst = host + c * chw_plane;
            for (int i = 0; i < chw_plane; ++i) dst[i] = float32ToHalf(src[i]);
        }
    } else {
        float* host = static_cast<float*>(in.host);
        std::vector<cv::Mat> ch(3);
        for (int c = 0; c < 3; ++c) {
            ch[c] = cv::Mat(imgsz_, imgsz_, CV_32FC1, host + c * chw_plane);
        }
        cv::split(lb_f32, ch);
    }

    CUDA_CHECK(cudaMemcpyAsync(in.device, in.host,
                               in.elem_count * in.elem_size,
                               cudaMemcpyHostToDevice, stream_));
}

std::vector<Detection> TRTDetector::detect(const cv::Mat& frame) {
    float scale = 1.f, pad_w = 0.f, pad_h = 0.f;
    preprocess(frame, scale, pad_w, pad_h);

#if NV_TENSORRT_MAJOR >= 10
    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("enqueueV3 failed");
    }
#else
    if (!context_->enqueueV2(binding_ptrs_.data(), stream_, nullptr)) {
        throw std::runtime_error("enqueueV2 failed");
    }
#endif

    for (auto& out : outputs_) {
        CUDA_CHECK(cudaMemcpyAsync(out.host, out.device, out.elem_count * out.elem_size,
                                   cudaMemcpyDeviceToHost, stream_));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));

    return postprocess(frame, scale, pad_w, pad_h);
}

std::vector<Detection> TRTDetector::postprocess(const cv::Mat& orig,
                                                float scale, float pad_w, float pad_h) {
    const auto& out = outputs_[0];
    const auto& shape = out.shape;

    const float* data;
    if (out.is_fp16) {
        fp16_scratch_.resize(out.elem_count);
        const uint16_t* src = static_cast<const uint16_t*>(out.host);
        for (size_t i = 0; i < out.elem_count; ++i) fp16_scratch_[i] = halfToFloat32(src[i]);
        data = fp16_scratch_.data();
    } else {
        data = static_cast<const float*>(out.host);
    }

    // YOLO26 emits (1, N, 4+nc); some exports transpose to (1, 4+nc, N).
    // Pick orientation by matching against the known class count.
    int64_t d0 = shape.size() >= 3 ? shape[1] : shape[0];
    int64_t d1 = shape.size() >= 3 ? shape[2] : shape[1];
    const int64_t expected_row = 4 + static_cast<int64_t>(kClassNames.size());

    bool transposed;
    if (d1 == expected_row) {
        transposed = false;
    } else if (d0 == expected_row) {
        transposed = true;
    } else {
        transposed = d0 < d1;  // fallback heuristic
    }
    int64_t num_rows = transposed ? d1 : d0;
    int64_t row_len  = transposed ? d0 : d1;
    if (row_len < 5) return {};

    const int num_classes = static_cast<int>(row_len) - 4;
    const int orig_w = orig.cols;
    const int orig_h = orig.rows;

    std::vector<Detection> dets;
    dets.reserve(64);

    for (int64_t i = 0; i < num_rows; ++i) {
        float cx, cy, bw, bh;
        int cls_id = 0;
        float best = 0.f;

        if (transposed) {
            cx = data[0 * num_rows + i];
            cy = data[1 * num_rows + i];
            bw = data[2 * num_rows + i];
            bh = data[3 * num_rows + i];
            for (int c = 0; c < num_classes; ++c) {
                float s = data[(4 + c) * num_rows + i];
                if (s > best) { best = s; cls_id = c; }
            }
        } else {
            const float* row = data + i * row_len;
            cx = row[0]; cy = row[1]; bw = row[2]; bh = row[3];
            for (int c = 0; c < num_classes; ++c) {
                if (row[4 + c] > best) { best = row[4 + c]; cls_id = c; }
            }
        }

        if (best < conf_thresh_) continue;

        float x1 = (cx - bw / 2.f - pad_w) / scale;
        float y1 = (cy - bh / 2.f - pad_h) / scale;
        float x2 = (cx + bw / 2.f - pad_w) / scale;
        float y2 = (cy + bh / 2.f - pad_h) / scale;

        x1 = std::max(0.f, std::min(x1, static_cast<float>(orig_w)));
        y1 = std::max(0.f, std::min(y1, static_cast<float>(orig_h)));
        x2 = std::max(0.f, std::min(x2, static_cast<float>(orig_w)));
        y2 = std::max(0.f, std::min(y2, static_cast<float>(orig_h)));

        dets.push_back({cls_id, best, x1, y1, x2, y2});
    }

    return dets;
}

}  // namespace tl
