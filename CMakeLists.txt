# NOTE:
# Ref: https://github.com/NVIDIA/cuda-samples

cmake_minimum_required(VERSION 3.20)

project(cuda-gpu-programming-study LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 90) # (75 80 86 87 89 90 100 110 120)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
if(ENABLE_CUDA_DEBUG)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
  # enable cuda-gdb (may significantly affect performance on some targets)
else()
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
  # add line information to all builds for debug tools (exclusive to -G option)
endif()

# WARN:
# error: unknown argument: '--extended-lambda' [clang-diagnostic-error]
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

include_directories(src/common)
file(GLOB_RECURSE ALL_CU_SOURCES "src/*.cu")
foreach(cu_source ${ALL_CU_SOURCES})
  get_filename_component(executable_name ${cu_source} NAME_WE)
  add_executable(${executable_name} ${cu_source})
endforeach()
