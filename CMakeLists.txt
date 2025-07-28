cmake_minimum_required(VERSION 3.18)
project(yolov11-tensorrt LANGUAGES CXX CUDA)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA 编译选项
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 添加头文件路径
include_directories(
    src/
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# 设置 TensorRT 路径
set(TENSORRT_DIR "/home/marten/TensorRT-10.10.0.31")
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)

# 查找 OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 源文件
set(SOURCES
    main.cpp
    src/yolov11.cpp
    src/preprocess.cu
)

# 添加可执行文件
add_executable(${PROJECT_NAME} ${SOURCES})

# 链接依赖库
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    nvinfer
    nvinfer_plugin
    cudart
)