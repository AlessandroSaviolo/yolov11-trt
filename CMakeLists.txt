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
ament_export_dependencies(${dependencies})

add_executable(yolov11_exec src/main.cpp)
target_link_libraries(yolov11_exec PRIVATE yolov11_node ${OpenCV_LIBRARIES})

install(TARGETS 
    yolov11_node
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/ DESTINATION share/${PROJECT_NAME}/include/)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_package()