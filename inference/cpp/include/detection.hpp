#pragma once

// Shared POD detection type used by both the TRT pipeline and the tracker.
// Kept in its own header so the tracker library does not need to pull in
// TensorRT / CUDA headers — letting it compile in isolation on dev machines
// (Mac/Linux without CUDA) and in tests.

#include <array>
#include <cstddef>

namespace tl {

inline const std::array<const char*, 7> kClassNames = {
    "red", "yellow", "green", "redLeft", "greenLeft", "redRight", "greenRight",
};

struct Detection {
    int class_id;
    float confidence;
    float x1, y1, x2, y2;

    const char* class_name() const {
        return (class_id >= 0 && static_cast<size_t>(class_id) < kClassNames.size())
                   ? kClassNames[class_id]
                   : "unknown";
    }
};

}  // namespace tl
