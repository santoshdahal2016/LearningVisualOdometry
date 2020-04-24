cmake_minimum_required(VERSION 3.15)
project(LearningVisualOdometry)

set(CMAKE_CXX_STANDARD 14)


find_package(Eigen3 3.3 REQUIRED)

include_directories("/usr/include/eigen3" )

FIND_PACKAGE(OpenCV REQUIRED)

find_package(Pangolin REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

add_subdirectory("Feature Extraction")
add_executable(LearningVisualOdometry main.cpp)


add_executable(FASTVO FASTVO.cpp)
TARGET_LINK_LIBRARIES (FASTVO  ${OpenCV_LIBS} ${Pangolin_LIBRARIES})
