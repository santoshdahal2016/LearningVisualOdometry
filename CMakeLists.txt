cmake_minimum_required(VERSION 3.15)
project(LearningVisualOdometry)

set(CMAKE_CXX_STANDARD 14)


add_subdirectory("Feature Extraction")
add_executable(LearningVisualOdometry main.cpp)