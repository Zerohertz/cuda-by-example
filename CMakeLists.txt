# NOTE:
# Ref: https://github.com/NVIDIA/cuda-samples

cmake_minimum_required(VERSION 3.20)

option(USE_NVCC "Use NVCC instead of Clang++ for CUDA" OFF)
option(ENABLE_CUDA_DEBUG "Enable CUDA debug mode" OFF)

set(CMAKE_CXX_COMPILER "clang++")
if(USE_NVCC)
  set(CMAKE_CUDA_COMPILER "nvcc")
  message(STATUS "Using NVCC compiler")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
  if(ENABLE_CUDA_DEBUG)
    set(CMAKE_BUILD_TYPE "Debug")
    message(STATUS "Clang++ debug mode enabled")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")
    message(STATUS "NVCC debug mode enabled")
  else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
  endif()
else()
  set(CMAKE_CUDA_COMPILER "clang++")
  message(STATUS "Using Clang++ compiler")
  if(ENABLE_CUDA_DEBUG)
    set(CMAKE_BUILD_TYPE "Debug")
    message(STATUS "Clang++ debug mode enabled")
  endif()
endif()

project(cuda-gpu-programming-study LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 90) # (75 80 86 87 89 90 100 110 120)

include_directories(src/common)
file(GLOB_RECURSE ALL_CPP_SOURCES "src/*.cpp")
foreach(cpp_source ${ALL_CPP_SOURCES})
  get_filename_component(executable_name ${cpp_source} NAME_WE)
  add_executable(${executable_name} ${cpp_source})
endforeach()
file(GLOB_RECURSE ALL_CU_SOURCES "src/*.cu")
foreach(cu_source ${ALL_CU_SOURCES})
  get_filename_component(executable_name ${cu_source} NAME_WE)
  add_executable(${executable_name} ${cu_source})
endforeach()
