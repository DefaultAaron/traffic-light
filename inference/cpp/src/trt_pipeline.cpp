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
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 9
        case nvinfer1::DataType::kINT64: return 8;
#endif
        default: return 4;
    }
}

bool isInt64Dtype(nvinfer1::DataType dt) {
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 9
    return dt == nvinfer1::DataType::kINT64;
#else
    (void)dt;
    return false;
#endif
}

bool isInt32Dtype(nvinfer1::DataType dt) {
    return dt == nvinfer1::DataType::kINT32;
}

// float32 → float16, IEEE-754 round-to-nearest-even. Subnormals flush to
// zero (adequate for [0, 1] image input).
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
        detectArch();
    } catch (...) {
        freeAll();
        throw;
    }
}

void TRTDetector::detectArch() {
    // DEIM-D-FINE deploy graph signature: input "orig_target_sizes" + outputs
    // labels/boxes/scores. Anything else falls through to YOLO.
    bool has_orig_size_input = false;
    for (const auto& in : inputs_) {
        if (in.name == "orig_target_sizes") {
            has_orig_size_input = true;
            break;
        }
    }
    bool has_deim_outputs =
        findOutput("labels") && findOutput("boxes") && findOutput("scores");

    // Reject INT32 (also 4 bytes) and any other non-float output dtype, which
    // postprocess would reinterpret as float and silently corrupt.
    auto ensure_float_output = [](const TensorBuf& o, const char* role) {
        if (!o.is_fp32 && !o.is_fp16) {
            throw std::runtime_error(
                std::string(role) + " output '" + o.name +
                "' has unsupported dtype "
                "(only FP32 or FP16 are supported)");
        }
    };

    if (has_orig_size_input && has_deim_outputs) {
        const TensorBuf* labels = findOutput("labels");
        const TensorBuf* boxes  = findOutput("boxes");
        const TensorBuf* scores = findOutput("scores");
        if (!labels->is_int64 && !labels->is_int32) {
            throw std::runtime_error(
                "DEIM 'labels' has unsupported dtype "
                "(expected int64 or int32)");
        }
        ensure_float_output(*boxes,  "DEIM 'boxes'");
        ensure_float_output(*scores, "DEIM 'scores'");

        // Top-K shape contract: labels=(1, K), boxes=(1, K, 4), scores=(1, K)
        // with K consistent. Plain element-count parity is not enough — e.g.
        // labels=(3,100)+boxes=(3,100,4) has the right ratio but batch=3.
        auto shape_str = [](const TensorBuf& t) {
            std::string s;
            for (size_t i = 0; i < t.shape.size(); ++i) {
                if (i) s += "x";
                s += std::to_string(t.shape[i]);
            }
            return s;
        };
        if (labels->shape.size() != 2 || labels->shape[0] != 1) {
            throw std::runtime_error(
                "DEIM 'labels' has shape " + shape_str(*labels) +
                "; expected (1, K)");
        }
        if (scores->shape.size() != 2 || scores->shape[0] != 1) {
            throw std::runtime_error(
                "DEIM 'scores' has shape " + shape_str(*scores) +
                "; expected (1, K)");
        }
        if (boxes->shape.size() != 3 || boxes->shape[0] != 1 || boxes->shape[2] != 4) {
            throw std::runtime_error(
                "DEIM 'boxes' has shape " + shape_str(*boxes) +
                "; expected (1, K, 4)");
        }
        const int64_t K = labels->shape[1];
        if (scores->shape[1] != K || boxes->shape[1] != K) {
            throw std::runtime_error(
                "DEIM K mismatch: labels K=" + std::to_string(K) +
                " scores K=" + std::to_string(scores->shape[1]) +
                " boxes K=" + std::to_string(boxes->shape[1]));
        }

        const TensorBuf* size_in = nullptr;
        for (const auto& in : inputs_) {
            if (in.name == "orig_target_sizes") { size_in = &in; break; }
        }
        if (size_in && !size_in->is_int64 && !size_in->is_int32) {
            throw std::runtime_error(
                "DEIM 'orig_target_sizes' has unsupported dtype "
                "(expected int64 or int32)");
        }

        arch_ = DetectorArch::kDEIM;
        std::cerr << "[TRTDetector] arch detected: DEIM-D-FINE (3 outputs: labels/boxes/scores)"
                  << std::endl;
        return;
    }

    // YOLO contract: exactly 1 output, rank 2 or 3 (batch=1 if rank 3),
    // shape has a (4 + nc) axis, not orientation-ambiguous.
    if (outputs_.size() != 1) {
        throw std::runtime_error(
            "YOLO arch dispatched but engine has " +
            std::to_string(outputs_.size()) + " outputs (expected 1)");
    }
    const auto& shape = outputs_[0].shape;
    const int64_t expected_row = 4 + static_cast<int64_t>(kClassNames.size());
    auto shape_str = [&shape]() {
        std::string s;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i) s += "x";
            s += std::to_string(shape[i]);
        }
        return s;
    };
    if (shape.size() != 2 && shape.size() != 3) {
        throw std::runtime_error(
            "YOLO arch dispatched but output rank is " +
            std::to_string(shape.size()) + " (expected 2 or 3); shape=" +
            shape_str());
    }
    if (shape.size() == 3 && shape[0] != 1) {
        throw std::runtime_error(
            "YOLO output batch dim is " + std::to_string(shape[0]) +
            " (expected 1); shape=" + shape_str());
    }
    bool found = false;
    for (auto d : shape) if (d == expected_row) { found = true; break; }
    if (!found) {
        throw std::runtime_error(
            "YOLO arch dispatched but output shape " + shape_str() +
            " has no " + std::to_string(expected_row) + "-wide axis");
    }
    if (shape.size() == 3 && shape[1] == expected_row && shape[2] == expected_row) {
        throw std::runtime_error(
            "YOLO output shape (1, " + std::to_string(expected_row) + ", " +
            std::to_string(expected_row) + ") is orientation-ambiguous");
    }
    if (shape.size() == 2 && shape[0] == expected_row && shape[1] == expected_row) {
        throw std::runtime_error(
            "YOLO output shape (" + std::to_string(expected_row) + ", " +
            std::to_string(expected_row) + ") is orientation-ambiguous");
    }
    ensure_float_output(outputs_[0], "YOLO");

    arch_ = DetectorArch::kYOLO;
    std::cerr << "[TRTDetector] arch detected: YOLO26 (single concat output)" << std::endl;
}

const TRTDetector::TensorBuf* TRTDetector::findOutput(const char* name) const {
    for (const auto& o : outputs_) {
        if (o.name == name) return &o;
    }
    return nullptr;
}

TRTDetector::TensorBuf* TRTDetector::findImageInput() noexcept {
    // Both branches require NCHW with C=3 — the name preference is only a
    // hint, not a structural override.
    auto looks_like_image = [](const TensorBuf& in) {
        return in.shape.size() == 4 && in.shape[1] == 3;
    };
    for (auto& in : inputs_) {
        if (in.name == "images" && looks_like_image(in)) return &in;
    }
    for (auto& in : inputs_) {
        if (looks_like_image(in)) return &in;
    }
    return nullptr;
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
    binding_ptrs_.clear();
    // TRT objects must release before the CUDA stream — TRT 8.x destructors
    // can touch the stream during teardown.
    context_.reset();
    engine_.reset();
    runtime_.reset();
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
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

    // Image input: dynamic spatial dims bind to imgsz_; other dynamic dims
    // bind to 1. Same rank-4/C=3 predicate as findImageInput().
    std::string image_input_name;
    std::string fallback_image_name;
    for (int i = 0; i < n; ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT) continue;
        nvinfer1::Dims d = engine_->getTensorShape(name);
        const bool looks_like_image = (d.nbDims == 4 && d.d[1] == 3);
        if (std::string(name) == "images" && looks_like_image) {
            image_input_name = name;
            break;
        }
        if (looks_like_image && fallback_image_name.empty()) {
            fallback_image_name = name;
        }
    }
    if (image_input_name.empty()) image_input_name = fallback_image_name;

    for (int i = 0; i < n; ++i) {
        const char* name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) != nvinfer1::TensorIOMode::kINPUT) continue;
        nvinfer1::Dims dims = engine_->getTensorShape(name);
        const bool is_image = (std::string(name) == image_input_name);
        for (int k = 0; k < dims.nbDims; ++k) {
            if (dims.d[k] < 0) {
                dims.d[k] = (is_image && dims.nbDims == 4 && k >= 2) ? imgsz_ : 1;
            }
        }
        if (!context_->setInputShape(name, dims)) {
            throw std::runtime_error(std::string("setInputShape failed for ") + name);
        }
    }

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
        buf.is_fp32 = (dtype == nvinfer1::DataType::kFLOAT);
        buf.is_fp16 = (dtype == nvinfer1::DataType::kHALF);
        buf.is_int64 = isInt64Dtype(dtype);
        buf.is_int32 = isInt32Dtype(dtype);
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

    int image_binding = -1;
    int fallback_binding = -1;
    for (int i = 0; i < n; ++i) {
        if (!engine_->bindingIsInput(i)) continue;
        const char* name = engine_->getBindingName(i);
        nvinfer1::Dims d = engine_->getBindingDimensions(i);
        const bool looks_like_image = (d.nbDims == 4 && d.d[1] == 3);
        if (name && std::string(name) == "images" && looks_like_image) {
            image_binding = i;
            break;
        }
        if (looks_like_image && fallback_binding < 0) {
            fallback_binding = i;
        }
    }
    if (image_binding < 0) image_binding = fallback_binding;

    for (int i = 0; i < n; ++i) {
        if (!engine_->bindingIsInput(i)) continue;
        nvinfer1::Dims dims = engine_->getBindingDimensions(i);
        const bool is_image = (i == image_binding);
        for (int k = 0; k < dims.nbDims; ++k) {
            if (dims.d[k] < 0) {
                dims.d[k] = (is_image && dims.nbDims == 4 && k >= 2) ? imgsz_ : 1;
            }
        }
        if (!context_->setBindingDimensions(i, dims)) {
            throw std::runtime_error("setBindingDimensions failed for binding " +
                                     std::to_string(i));
        }
    }

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
        buf.is_fp32 = (dtype == nvinfer1::DataType::kFLOAT);
        buf.is_fp16 = (dtype == nvinfer1::DataType::kHALF);
        buf.is_int64 = isInt64Dtype(dtype);
        buf.is_int32 = isInt32Dtype(dtype);
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

    // Image input must be NCHW, batch=1, square. orig_target_sizes=
    // [imgsz, imgsz] assumes a square letterbox.
    const TensorBuf* image_in = findImageInput();
    if (image_in == nullptr) {
        throw std::runtime_error(
            "engine has no NCHW 3-channel image input "
            "(expected a tensor named 'images' or a 4-D input with C=3)");
    }

    const auto& in_shape = image_in->shape;
    int n = static_cast<int>(in_shape[0]);
    int h = static_cast<int>(in_shape[2]);
    int w = static_cast<int>(in_shape[3]);
    if (n != 1) {
        throw std::runtime_error(
            "engine image input has batch=" + std::to_string(n) +
            "; only batch=1 is supported");
    }
    if (h != w) {
        throw std::runtime_error(
            "engine image input is rectangular (" + std::to_string(w) + "x" +
            std::to_string(h) + "); only square inputs are supported "
            "(orig_target_sizes assumes square)");
    }
    if (h != imgsz_) {
        std::cerr << "[TRT] warning: engine input " << w << "x" << h
                  << " differs from imgsz=" << imgsz_
                  << "; using engine size " << h << "." << std::endl;
        imgsz_ = h;
    }

    // Reject INT32/INT8/UINT8 image inputs — preprocess writes float bytes,
    // an integer dtype would corrupt or overrun.
    if (!image_in->is_fp32 && !image_in->is_fp16) {
        throw std::runtime_error(
            "engine image input has unsupported dtype "
            "(only FP32 or FP16 are supported)");
    }
}

void TRTDetector::preprocess(const cv::Mat& frame, float& scale, float& pad_w, float& pad_h) {
    cv::Mat rgb;
    cv::cvtColor(frame, rgb, cv::COLOR_BGR2RGB);
    cv::Mat lb = letterbox(rgb, imgsz_, imgsz_, scale, pad_w, pad_h);

    cv::Mat lb_f32;
    lb.convertTo(lb_f32, CV_32FC3, 1.0 / 255.0);

    TensorBuf* image_in = findImageInput();
    if (image_in == nullptr) {
        throw std::runtime_error("preprocess: image input lookup failed");
    }
    auto& in = *image_in;
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

void TRTDetector::fillOrigTargetSizes() {
    // DEIM postprocessor: `bbox_pred *= orig_target_sizes.repeat(1, 2)`
    // broadcasts (s0, s1) onto (x1, y1, x2, y2). Square letterbox makes the
    // order symmetric — `[imgsz, imgsz]` lands boxes in letterbox-pixel coords.
    TensorBuf* size_in = nullptr;
    for (auto& in : inputs_) {
        if (in.name == "orig_target_sizes") { size_in = &in; break; }
    }
    if (size_in == nullptr) {
        throw std::runtime_error(
            "DEIM detector missing 'orig_target_sizes' input — engine misnamed?");
    }

    if (size_in->is_int64) {
        auto* host = static_cast<int64_t*>(size_in->host);
        host[0] = static_cast<int64_t>(imgsz_);
        host[1] = static_cast<int64_t>(imgsz_);
    } else if (size_in->is_int32) {
        auto* host = static_cast<int32_t*>(size_in->host);
        host[0] = static_cast<int32_t>(imgsz_);
        host[1] = static_cast<int32_t>(imgsz_);
    } else {
        throw std::runtime_error(
            "DEIM 'orig_target_sizes' has unsupported dtype "
            "(expected int64 or int32); engine likely exported with a "
            "non-standard postprocessor");
    }

    CUDA_CHECK(cudaMemcpyAsync(size_in->device, size_in->host,
                               size_in->elem_count * size_in->elem_size,
                               cudaMemcpyHostToDevice, stream_));
}

std::vector<Detection> TRTDetector::detect(const cv::Mat& frame) {
    float scale = 1.f, pad_w = 0.f, pad_h = 0.f;
    preprocess(frame, scale, pad_w, pad_h);

    if (arch_ == DetectorArch::kDEIM) {
        fillOrigTargetSizes();
    }

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

    if (arch_ == DetectorArch::kDEIM) {
        return postprocessDeim(frame, scale, pad_w, pad_h);
    }
    return postprocessYolo(frame, scale, pad_w, pad_h);
}

std::vector<Detection> TRTDetector::postprocessYolo(const cv::Mat& orig,
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

    // YOLO26 emits (N, 4+nc) / (4+nc, N); detectArch() guarantees one axis
    // equals 4+nc, so the orientation pick below is exact.
    int64_t d0 = shape.size() >= 3 ? shape[1] : shape[0];
    int64_t d1 = shape.size() >= 3 ? shape[2] : shape[1];
    const int64_t expected_row = 4 + static_cast<int64_t>(kClassNames.size());

    const bool transposed = (d0 == expected_row);
    int64_t num_rows = transposed ? d1 : d0;
    int64_t row_len  = transposed ? d0 : d1;

    const int num_classes = static_cast<int>(row_len) - 4;
    const int orig_w = orig.cols;
    const int orig_h = orig.rows;

    std::vector<Detection> dets;
    dets.reserve(64);

    // Stripped YOLO26 head emits xyxy directly (NOT cxcywh).
    for (int64_t i = 0; i < num_rows; ++i) {
        float lx1, ly1, lx2, ly2;
        int cls_id = 0;
        float best = 0.f;

        if (transposed) {
            lx1 = data[0 * num_rows + i];
            ly1 = data[1 * num_rows + i];
            lx2 = data[2 * num_rows + i];
            ly2 = data[3 * num_rows + i];
            for (int c = 0; c < num_classes; ++c) {
                float s = data[(4 + c) * num_rows + i];
                if (s > best) { best = s; cls_id = c; }
            }
        } else {
            const float* row = data + i * row_len;
            lx1 = row[0]; ly1 = row[1]; lx2 = row[2]; ly2 = row[3];
            for (int c = 0; c < num_classes; ++c) {
                if (row[4 + c] > best) { best = row[4 + c]; cls_id = c; }
            }
        }

        if (best < conf_thresh_) continue;

        float x1 = (lx1 - pad_w) / scale;
        float y1 = (ly1 - pad_h) / scale;
        float x2 = (lx2 - pad_w) / scale;
        float y2 = (ly2 - pad_h) / scale;

        x1 = std::max(0.f, std::min(x1, static_cast<float>(orig_w)));
        y1 = std::max(0.f, std::min(y1, static_cast<float>(orig_h)));
        x2 = std::max(0.f, std::min(x2, static_cast<float>(orig_w)));
        y2 = std::max(0.f, std::min(y2, static_cast<float>(orig_h)));

        dets.push_back({cls_id, best, x1, y1, x2, y2});
    }

    return dets;
}

// Read FP32/FP16 output into a contiguous float pointer (FP32 zero-copy,
// FP16 expanded into scratch).
static const float* readFloatOutput(const void* host, size_t elem_count,
                                    bool is_fp16, std::vector<float>& scratch) {
    if (is_fp16) {
        scratch.resize(elem_count);
        const uint16_t* src = static_cast<const uint16_t*>(host);
        for (size_t i = 0; i < elem_count; ++i) scratch[i] = halfToFloat32(src[i]);
        return scratch.data();
    }
    return static_cast<const float*>(host);
}

std::vector<Detection> TRTDetector::postprocessDeim(const cv::Mat& orig,
                                                    float scale,
                                                    float pad_w,
                                                    float pad_h) {
    const TensorBuf* labels_buf = findOutput("labels");
    const TensorBuf* boxes_buf  = findOutput("boxes");
    const TensorBuf* scores_buf = findOutput("scores");
    if (!labels_buf || !boxes_buf || !scores_buf) {
        throw std::runtime_error("DEIM postprocess: missing labels/boxes/scores outputs");
    }

    // Shapes (1, K) / (1, K, 4) / (1, K) verified at construction
    // (detectArch DEIM gate).
    const size_t K = static_cast<size_t>(labels_buf->shape[1]);

    const float* boxes  = readFloatOutput(boxes_buf->host,  boxes_buf->elem_count,
                                          boxes_buf->is_fp16, deim_boxes_scratch_);
    const float* scores = readFloatOutput(scores_buf->host, scores_buf->elem_count,
                                          scores_buf->is_fp16, deim_scores_scratch_);

    const int orig_w = orig.cols;
    const int orig_h = orig.rows;
    const int nc = static_cast<int>(kClassNames.size());

    std::vector<Detection> dets;
    dets.reserve(32);

    for (size_t i = 0; i < K; ++i) {
        float conf = scores[i];
        if (conf < conf_thresh_) continue;

        int cls_id = -1;
        if (labels_buf->is_int64) {
            const int64_t* lab = static_cast<const int64_t*>(labels_buf->host);
            cls_id = static_cast<int>(lab[i]);
        } else if (labels_buf->is_int32) {
            const int32_t* lab = static_cast<const int32_t*>(labels_buf->host);
            cls_id = static_cast<int>(lab[i]);
        } else {
            throw std::runtime_error(
                "DEIM 'labels' tensor has unsupported dtype "
                "(expected int64 or int32)");
        }
        // DEIM may emit cls_id == nc for "no object"; drop instead of clamp.
        if (cls_id < 0 || cls_id >= nc) continue;

        const float* bb = boxes + i * 4;
        float lx1 = bb[0], ly1 = bb[1], lx2 = bb[2], ly2 = bb[3];

        float x1 = (lx1 - pad_w) / scale;
        float y1 = (ly1 - pad_h) / scale;
        float x2 = (lx2 - pad_w) / scale;
        float y2 = (ly2 - pad_h) / scale;

        x1 = std::max(0.f, std::min(x1, static_cast<float>(orig_w)));
        y1 = std::max(0.f, std::min(y1, static_cast<float>(orig_h)));
        x2 = std::max(0.f, std::min(x2, static_cast<float>(orig_w)));
        y2 = std::max(0.f, std::min(y2, static_cast<float>(orig_h)));

        dets.push_back({cls_id, conf, x1, y1, x2, y2});
    }

    return dets;
}

}  // namespace tl
