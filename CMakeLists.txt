cmake_minimum_required(VERSION 3.6)
project(yolov11_trt LANGUAGES CXX CUDA)

cmake_policy(SET CMP0057 NEW)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.4)
set(TensorRT_DIR /usr/share/doc/tensorrt-8.5.2.2)

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_library(NVINFER nvinfer)
find_library(NVINFER_PLUGIN nvinfer_plugin)

set(dependencies
  OpenCV
  CUDA
  TensorRT
)

if (NOT CUDA_INCLUDE_DIRS)
    set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/include)
endif()
if (NOT CUDA_LIBRARIES)
    set(CUDA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib)
endif()

include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
)

set(SOURCES
    src/main.cpp
    src/yolov11.cpp
    src/preprocess.cu
)

link_directories(${TENSORRT_DIR}/lib)
set(TENSORRT_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser)

cuda_add_executable(yolov11_trt ${SOURCES})

target_link_libraries(yolov11_trt
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
)