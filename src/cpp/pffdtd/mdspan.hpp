// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#pragma once

#if defined(SYCL_LANGUAGE_VERSION)
  #define SYCL_WAS_DEFINED SYCL_LANGUAGE_VERSION
  #undef SYCL_LANGUAGE_VERSION
#endif

#if defined(__clang__)
  #pragma clang diagnostic push
  #pragma clang diagnostic ignored "-Wextra-semi"
  #pragma clang diagnostic ignored "-Wshadow"
  #pragma clang diagnostic ignored "-Wsign-compare"
  #pragma clang diagnostic ignored "-Wunused-parameter"
#elif defined(__GNUC__)
  #pragma GCC diagnostic push
  #pragma GCC diagnostic ignored "-Wextra-semi"
  #pragma GCC diagnostic ignored "-Wshadow"
  #pragma GCC diagnostic ignored "-Wsign-compare"
#endif

#include <mdspan/mdarray.hpp>
#include <mdspan/mdspan.hpp>

#if defined(__clang__)
  #pragma clang diagnostic pop
#elif defined(__GNUC__)
  #pragma GCC diagnostic pop
#endif

#if defined(SYCL_WAS_DEFINED)
  #define SYCL_LANGUAGE_VERSION SYCL_WAS_DEFINED
  #undef SYCL_WAS_DEFINED
#endif

namespace stdex {
using Kokkos::default_accessor;
using Kokkos::dextents;
using Kokkos::extents;
using Kokkos::full_extent;
using Kokkos::layout_left;
using Kokkos::layout_right;
using Kokkos::layout_stride;
using Kokkos::mdspan;
using Kokkos::submdspan;
using Kokkos::Experimental::mdarray;
} // namespace stdex
