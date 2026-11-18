cmake_minimum_required(VERSION 3.10)
project(transformation_perspective)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(transform_perspective main.cpp)
target_link_libraries(transform_perspective ${OpenCV_LIBS})

