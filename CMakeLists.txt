cmake_minimum_required(VERSION 3.5.0)
project(inspurocr VERSION 0.1.0 LANGUAGES C CXX)


# Detect platform type
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ONNX_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/linux-x86/include")
    set(ONNX_LIB_DIR "${CMAKE_SOURCE_DIR}/third_party/linux-x86/lib")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ONNX_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/linux-arm64/include")
    set(ONNX_LIB_DIR "${CMAKE_SOURCE_DIR}/third_party/linux-arm64/lib")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Include ONNX Runtime
include_directories(${ONNX_INCLUDE_DIR})
link_directories(${ONNX_LIB_DIR})

# Set OpenCV directory
set(OpenCV_DIR /home/xuming/.local/opencv-4.9/share/OpenCV)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})



## 包含 src 目录中的头文件
include_directories(${PROJECT_SOURCE_DIR}/src)

# # 添加可执行文件
# add_executable(inspurocr 
#     # src/det_main.cpp 
#     # src/cls_main.cpp
#     # src/rec_main.cpp
#     src/main.cpp
#     src/PredictClassifier.cpp
#     src/PredictClassifier.h
#     src/ExportSystem.h
#     src/ExportSystem.cpp
#     # src/PredictSystem.h
#     # src/PredictSystem.cpp
#     src/PredictBase.h 
#     src/PredictBase.cpp
#     src/PredictDetector.h
#     src/PredictDetector.cpp
#     src/PredictRecognizer.h
#     src/PredictRecognizer.cpp
#     src/json.hpp
#     )
# 添加可执行文件
add_library(inspurocr  SHARED
    # src/det_main.cpp 
    # src/cls_main.cpp
    # src/rec_main.cpp
    # src/main.cpp
    src/PredictClassifier.cpp
    src/PredictClassifier.h
    src/ExportSystem.h
    src/ExportSystem.cpp
    # src/PredictSystem.h
    # src/PredictSystem.cpp
    src/PredictBase.h 
    src/PredictBase.cpp
    src/PredictDetector.h
    src/PredictDetector.cpp
    src/PredictRecognizer.h
    src/PredictRecognizer.cpp
    src/json.hpp
    )

# 链接 ONNX Runtime 和 OpenCV 库
target_link_libraries(inspurocr onnxruntime ${OpenCV_LIBS})