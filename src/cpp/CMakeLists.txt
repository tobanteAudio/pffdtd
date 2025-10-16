# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2024 Tobias Hienzsch

project(pffdtd)

add_library(pffdtd)
add_library(pffdtd::pffdtd ALIAS pffdtd)
target_include_directories(pffdtd PUBLIC ${PROJECT_SOURCE_DIR})
target_compile_definitions(pffdtd PUBLIC _CRT_SECURE_NO_WARNINGS=1 MDSPAN_USE_BRACKET_OPERATOR=0 MDSPAN_USE_PAREN_OPERATOR=1)

target_link_libraries(pffdtd
    PUBLIC
        fmt::fmt
        HDF5::HDF5
        OpenMP::OpenMP_CXX
        mdspan::mdspan
        pffdtd::compiler_warnings
)

target_sources(pffdtd
    PRIVATE
        pffdtd/assert.hpp
        pffdtd/double.hpp
        pffdtd/engine_cpu_2d.cpp
        pffdtd/engine_cpu_2d.hpp
        pffdtd/engine_cpu_3d.cpp
        pffdtd/engine_cpu_3d.hpp
        pffdtd/exception.hpp
        pffdtd/float.hpp
        pffdtd/hdf.hpp
        pffdtd/mdspan.hpp
        pffdtd/precision.cpp
        pffdtd/precision.hpp
        pffdtd/print.hpp
        pffdtd/progress.cpp
        pffdtd/progress.hpp
        pffdtd/simulation_2d.cpp
        pffdtd/simulation_2d.hpp
        pffdtd/simulation_3d.cpp
        pffdtd/simulation_3d.hpp
        pffdtd/time.hpp
        pffdtd/utility.hpp
)

if(PFFDTD_ENABLE_CUDA)
    target_sources(pffdtd PRIVATE pffdtd/engine_cuda_3d.cu pffdtd/engine_cuda_3d.hpp)
    target_compile_definitions(pffdtd PUBLIC PFFDTD_HAS_CUDA=1)
endif()

if(PFFDTD_HAS_SYCL)
    target_sources(pffdtd PRIVATE pffdtd/engine_sycl_2d.cpp pffdtd/engine_sycl_2d.hpp)
    target_sources(pffdtd PRIVATE pffdtd/engine_sycl_3d.cpp pffdtd/engine_sycl_3d.hpp)
    target_sources(pffdtd PRIVATE pffdtd/sycl.cpp pffdtd/sycl.hpp)
    add_sycl_to_target(TARGET pffdtd SOURCES pffdtd/engine_sycl_2d.cpp pffdtd/engine_sycl_3d.cpp pffdtd/sycl.cpp)
    target_compile_definitions(pffdtd PUBLIC PFFDTD_HAS_SYCL=1)

    if(PFFDTD_ENABLE_SYCL_ONEAPI)
        set(PFFDTD_SYCL_FLAGS "-fsycl-targets=spir64,nvptx64-nvidia-cuda -Xsycl-target-backend=nvptx64-nvidia-cuda --offload-arch=sm_89")
        set_source_files_properties(pffdtd/engine_sycl_2d.cpp PROPERTIES COMPILE_FLAGS ${PFFDTD_SYCL_FLAGS})
        set_source_files_properties(pffdtd/engine_sycl_3d.cpp PROPERTIES COMPILE_FLAGS ${PFFDTD_SYCL_FLAGS})
        target_link_libraries(pffdtd PRIVATE ${PFFDTD_SYCL_FLAGS})
    endif()
endif()

if(PFFDTD_HAS_METAL)
    target_compile_definitions(pffdtd PUBLIC PFFDTD_HAS_METAL=1)
    target_sources(pffdtd PRIVATE pffdtd/engine_metal_2d.hpp pffdtd/engine_metal_2d.mm)
    target_sources(pffdtd PRIVATE pffdtd/engine_metal_3d.hpp pffdtd/engine_metal_3d.mm)

    set(SHADER_FILES "${CMAKE_CURRENT_SOURCE_DIR}/pffdtd/engine_metal.metal")
    set(SHADER_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/engine_metal.air")
    set(SHADER_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/default.metallib")

    # Add a custom command to compile the Metal shader
    add_custom_command(
        OUTPUT ${SHADER_OUTPUT}
        COMMAND xcrun -sdk macosx metal -std=metal3.0 -c ${SHADER_FILES} -o ${SHADER_OUTPUT}
        DEPENDS ${SHADER_FILES}
        COMMENT "Compiling Metal shader: ${SHADER_FILES}"
        VERBATIM
    )

    # Add a custom command to link the compiled shader into a metallib file
    add_custom_command(
        OUTPUT ${SHADER_LIBRARY}
        COMMAND xcrun -sdk macosx metallib ${SHADER_OUTPUT} -o ${SHADER_LIBRARY}
        DEPENDS ${SHADER_OUTPUT}
        COMMENT "Linking Metal shader into metallib: ${SHADER_LIBRARY}"
        VERBATIM
    )

    # Add a custom target for the Metal library
    add_custom_target(pffdtd-engine-metal-shaders ALL DEPENDS ${SHADER_LIBRARY})

    find_library(Foundation_Framework Foundation)
    find_library(Metal_Framework Metal)
    find_library(MetalKit_Framework MetalKit)

    target_link_libraries(pffdtd PUBLIC ${Foundation_Framework} ${Metal_Framework} ${MetalKit_Framework})
    set_target_properties(pffdtd PROPERTIES LINK_FLAGS "-Wl,-sectcreate,addseg,addsect,${SHADER_LIBRARY}")
    add_dependencies(pffdtd pffdtd-engine-metal-shaders)
endif()

add_executable(pffdtd-engine main.cpp)
target_link_libraries(pffdtd-engine PRIVATE CLI11::CLI11 pffdtd::pffdtd)
