cmake_minimum_required(VERSION 3.18)

# Project declaration with C++ and CUDA support
project(yolov11-trt LANGUAGES CXX CUDA)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.4)
set(TENSORRT_PATH /usr/share/doc/tensorrt-8.5.2.2)
set(Torch_PATH $ENV{TORCH_PATH})
set(Torch_DIR "${Torch_PATH}/share/cmake/Torch")

# Allow overriding TensorRT and OpenCV paths via command line
# e.g., cmake -DTENSORRT_PATH="path/to/TensorRT" -DOpenCV_DIR="path/to/OpenCV" ..
option(TENSORRT_PATH_OPTION "Path to TensorRT installation" ${TENSORRT_PATH})
set(TENSORRT_PATH ${TENSORRT_PATH_OPTION} CACHE PATH "Path to TensorRT installation")

# Find TensorRT
find_package(TensorRT REQUIRED)
if(NOT TensorRT_FOUND)
    message(FATAL_ERROR "TensorRT not found. Please install TensorRT or set TensorRT_DIR.")
endif()

find_library(NVINFER nvinfer)
find_library(NVINFER_PLUGIN nvinfer_plugin)

# Find OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV or set OpenCV_DIR.")
endif()

# Find CUDA
find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found. Please install the CUDA Toolkit.")
endif()

if (NOT CUDA_INCLUDE_DIRS)
  set(CUDA_INCLUDE_DIRS ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/include)
endif()
if (NOT CUDA_LIBRARIES)
  set(CUDA_LIBRARIES ${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib)
endif()

# Include directories for dependencies
include_directories(${TensorRT_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

# Include directory for this project
include_directories(${CMAKE_SOURCE_DIR}/include)

# Define source files (including CUDA sources)
set(SOURCES
src/main.cpp
    src/yolov11.cpp
    src/preprocess.cu
)

# Create executable (CMake handles CUDA sources automatically)
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Define API_EXPORTS macro
target_compile_definitions(${PROJECT_NAME} PRIVATE API_EXPORTS)

# Specify include directories (modern CMake approach)
target_include_directories(${PROJECT_NAME} PRIVATE
    src/
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_PATH}/include
)

# Link TensorRT libraries
# Specify full paths to TensorRT libraries to avoid relying on link_directories
set(TENSORRT_LIBS
    "${TENSORRT_PATH}/lib/nvinfer.lib"
    "${TENSORRT_PATH}/lib/nvonnxparser.lib"
    "${TENSORRT_PATH}/lib/nvparsers.lib"
    "${TENSORRT_PATH}/lib/nvinfer_plugin.lib"
)

# Link libraries to the target
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${TensorRT_LIBRARIES}
    ${NVINFER}
    ${NVINFER_PLUGIN}
)

# Enable separable compilation for CUDA (optional but recommended)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# (Optional) Specify CUDA architectures based on your GPU hardware
# set(CMAKE_CUDA_ARCHITECTURES 75)  # Example for Turing architecture

# (Optional) Set output directories for binaries
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)