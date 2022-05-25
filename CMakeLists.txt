# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(main_project)

find_package(OpenCV REQUIRED)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  # Add OpenCV headers location to your include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Declare the executable target built from your sources
add_executable(opencv_test src/opencv_test.cpp)
add_executable(feature_extraction src/feature_extraction.cpp)
add_executable(pose_estimation_2d2d src/pose_estimation_2d2d.cpp)

# Link your application with OpenCV libraries
target_link_libraries(opencv_test ${OpenCV_LIBS})
target_link_libraries(feature_extraction ${OpenCV_LIBS})
target_link_libraries(pose_estimation_2d2d ${OpenCV_LIBS})