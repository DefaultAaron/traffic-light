cmake_minimum_required(VERSION 3.18)
project(traffic_light_trt CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ---- CUDA ----
find_package(CUDAToolkit REQUIRED)

# ---- OpenCV ----
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio)

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
target_link_libraries(tl_demo PRIVATE tl_trt_pipeline)
target_compile_options(tl_demo PRIVATE -Wall -Wextra)
