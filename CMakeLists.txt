cmake_minimum_required(VERSION 3.18)

# Project declaration with C++ and CUDA support
project(yolov11_trt LANGUAGES CXX CUDA)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CUDA_TOOLKIT_ROOT_DIR /usr/local/cuda-11.4)
set(TENSORRT_PATH /usr/share/doc/tensorrt-8.5.2.2)
set(Torch_PATH $ENV{TORCH_PATH})
set(Torch_DIR "${Torch_PATH}/share/cmake/Torch")

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)
find_package(TensorRT REQUIRED)
find_library(NVINFER nvinfer)
find_library(NVINFER_PLUGIN nvinfer_plugin)

set(dependencies
  rclcpp
  rclcpp_components
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

# Include directories
include_directories(
    include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
)

add_library(yolov11_node SHARED src/yolov11.cpp src/preprocess.cu)
set_target_properties(yolov11_node PROPERTIES COMPILE_DEFINITIONS "COMPOSITION_BUILDING_DLL" CUDA_SEPARABLE_COMPILATION ON)
ament_target_dependencies(yolov11_node ${dependencies})
target_link_libraries(yolov11_node
    ${rclcpp_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${TensorRT_LIBRARIES}
    ${NVINFER}
    ${NVINFER_PLUGIN}
)

add_executable(yolov11_exec src/main.cpp)
target_link_libraries(yolov11_exec PRIVATE yolov11_node ${OpenCV_LIBRARIES})

install(TARGETS 
    yolov11_node
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION include)
ament_export_include_directories(include)

ament_export_dependencies(${dependencies})
ament_export_libraries(${PROJECT_NAME})
ament_package()