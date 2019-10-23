cmake_minimum_required(VERSION 3.7)
project(cuda_load_generator CUDA CXX)

find_package(Boost REQUIRED)

#Find all source files
file(GLOB_RECURSE SRCS "${CMAKE_SOURCE_DIR}/src/*[.cpp|.cu]")

include_directories(${Boost_INCLUDE_DIRS})

add_executable(cuda_load_generator ${SRCS})
set_target_properties(cuda_load_generator PROPERTIES CXX_STANDARD 17)