cmake_minimum_required(VERSION 3.15)
project(LearningVisualOdometry)

set(CMAKE_CXX_STANDARD 14)

FIND_PACKAGE(OpenCV REQUIRED)

add_executable(FastFeature FastFeature.cpp)
TARGET_LINK_LIBRARIES (FastFeature  ${OpenCV_LIBS} )
