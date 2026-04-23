cmake_minimum_required(VERSION 3.18)

# The tracker library depends only on OpenCV + STL — it can build on dev
# machines (Mac/Linux without CUDA). The TRT pipeline + demo need CUDA and
# TensorRT, available only on the Orin (and on x86 boxes with matching
# TensorRT installs).
#
# Build modes:
#   -DTL_BUILD_TRACKER_ONLY=ON   tracker library + tracker tests only
#                                (no CUDA, no TensorRT — use on Mac)
#   -DTL_BUILD_TRACKER_ONLY=OFF  full build: tracker + TRT pipeline + demo
#                                (default — use on Orin)
option(TL_BUILD_TRACKER_ONLY "Build only the tracker library (skip TensorRT/CUDA)" OFF)

if(TL_BUILD_TRACKER_ONLY)
    project(traffic_light_trt CXX)
else()
    project(traffic_light_trt CXX CUDA)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- OpenCV (needed by both tracker and pipeline) ----
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)

# ---- tl_tracker (pure C++ / OpenCV, no CUDA or TensorRT) ----
add_library(tl_tracker STATIC
    src/tracker.cpp
)
target_include_directories(tl_tracker PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tl_tracker PUBLIC ${OpenCV_LIBS})
target_compile_options(tl_tracker PRIVATE -Wall -Wextra -Wno-unused-parameter)

# ---- Tracker parity tests ----
# Shares the JSON fixtures under tests/fixtures/tracker/ with the Python
# suite so semantic drift between the two ports fails here immediately.
option(TL_BUILD_TESTS "Build tracker parity tests" ON)
if(TL_BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.3
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)

    add_executable(tl_tracker_tests tests/test_tracker.cpp)
    target_link_libraries(tl_tracker_tests PRIVATE tl_tracker nlohmann_json::nlohmann_json)
    target_compile_options(tl_tracker_tests PRIVATE -Wall -Wextra)
    # Resolve the fixtures path at configure time so the test binary can
    # find it without an argv flag (CI / IDE debugging both work).
    target_compile_definitions(tl_tracker_tests PRIVATE
        TL_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../../tests/fixtures/tracker"
    )

    enable_testing()
    add_test(NAME tracker_parity COMMAND tl_tracker_tests)
endif()

if(TL_BUILD_TRACKER_ONLY)
    message(STATUS "TL_BUILD_TRACKER_ONLY=ON — skipping TensorRT pipeline and demo")
    return()
endif()

# ---- CUDA ----
find_package(CUDAToolkit REQUIRED)

# ---- TensorRT (JetPack default layout; override -DTENSORRT_ROOT=... for custom installs) ----
set(TENSORRT_ROOT "" CACHE PATH "TensorRT install root")
find_path(TRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_ROOT}/include /usr/include/aarch64-linux-gnu /usr/include/x86_64-linux-gnu
    REQUIRED)
find_library(TRT_NVINFER nvinfer
    HINTS ${TENSORRT_ROOT}/lib /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu
    REQUIRED)
find_library(TRT_NVINFER_PLUGIN nvinfer_plugin
    HINTS ${TENSORRT_ROOT}/lib /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)

message(STATUS "TensorRT include: ${TRT_INCLUDE_DIR}")
message(STATUS "TensorRT libs:    ${TRT_NVINFER} ${TRT_NVINFER_PLUGIN}")

add_library(tl_trt_pipeline STATIC
    src/trt_pipeline.cpp
)
target_include_directories(tl_trt_pipeline PUBLIC include ${TRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(tl_trt_pipeline
    PUBLIC
        ${OpenCV_LIBS}
        ${TRT_NVINFER}
        CUDA::cudart
)
if(TRT_NVINFER_PLUGIN)
    target_link_libraries(tl_trt_pipeline PUBLIC ${TRT_NVINFER_PLUGIN})
endif()
target_compile_options(tl_trt_pipeline PRIVATE -Wall -Wextra -Wno-unused-parameter)

add_executable(tl_demo src/demo.cpp)
target_link_libraries(tl_demo PRIVATE tl_trt_pipeline tl_tracker)
target_compile_options(tl_demo PRIVATE -Wall -Wextra)
