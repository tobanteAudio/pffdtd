# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch
cmake_minimum_required(VERSION 3.24)

option(PFFDTD_ENABLE_CUDA "Build with CUDA" OFF)
option(PFFDTD_ENABLE_METAL "Build with Metal compute shaders" ON)
option(PFFDTD_ENABLE_SYCL_ACPP "Build with AdaptiveCpp SYCL" OFF)
option(PFFDTD_ENABLE_SYCL_ONEAPI "Build with Intel SYCL" OFF)

find_program(CCACHE ccache)
if (CCACHE)
    set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE})
    set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE})
endif ()

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(pffdtd VERSION 0.1.0 LANGUAGES C CXX)

if(PFFDTD_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
endif()

if(PFFDTD_ENABLE_METAL AND APPLE)
    enable_language(OBJC)
    enable_language(OBJCXX)
    set(PFFDTD_HAS_METAL ON)
endif()

include(FetchContent)
FetchContent_Declare(mdspan GIT_REPOSITORY "https://github.com/kokkos/mdspan" GIT_TAG "stable" GIT_SHALLOW TRUE)
FetchContent_MakeAvailable(mdspan)

find_package(CLI11 REQUIRED)
find_package(fmt REQUIRED)
find_package(HDF5 REQUIRED)
find_package(OpenMP REQUIRED)

if(PFFDTD_ENABLE_SYCL_ACPP)
    find_package(AdaptiveCpp REQUIRED)
    set(PFFDTD_HAS_SYCL ON)
endif()

if(PFFDTD_ENABLE_SYCL_ONEAPI)
    find_package(IntelSYCL REQUIRED)
    set(PFFDTD_HAS_SYCL ON)
endif()

add_subdirectory(src/cpp)
