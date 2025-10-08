# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (C) 2025 Tobias Hienzsch

add_library(pffdtd.compiler_warnings INTERFACE)
add_library(pffdtd::compiler_warnings ALIAS pffdtd.compiler_warnings)

if(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    target_compile_options(pffdtd.compiler_warnings INTERFACE
        "/W3"
    )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(pffdtd.compiler_warnings INTERFACE
        "-Wall"
        "-Wextra"
        "-Wpedantic"
        "-Wsign-compare"
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang|Clang|IntelLLVM")
    target_compile_options(pffdtd.compiler_warnings INTERFACE
        "-Wall"
        "-Wextra"
        "-Wpedantic"
        "-Wsign-compare"
        "-Wno-deprecated-declarations" # SYCL 2020 deprecations
    )
endif ()
