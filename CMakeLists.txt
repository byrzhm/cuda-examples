cmake_minimum_required(VERSION 3.28.3)

project(AdvancedCUDAExamples CUDA)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(${PROJECT_SOURCE_DIR}/3rdparty/nvbench)
add_subdirectory(${PROJECT_SOURCE_DIR}/advanced/reduce)
