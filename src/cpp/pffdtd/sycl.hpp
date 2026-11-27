// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#pragma once

#include "pffdtd/time.hpp"

#if not defined(PFFDTD_HAS_SYCL)
  #error "SYCL must be enabled in CMake via PFFDTD_ENABLE_SYCL_ACPP or PFFDTD_ENABLE_SYCL_ONEAPI"
#endif

#include <sycl/sycl.hpp>

#include <string>
#include <utility>

namespace pffdtd {

template<typename Accessor>
[[nodiscard]] auto getPtr(Accessor&& a) -> auto* {
  return std::forward<Accessor>(a).template get_multi_ptr<sycl::access::decorated::no>().get();
}

[[nodiscard]] auto elapsedTime(sycl::event const& event) -> std::chrono::nanoseconds;
[[nodiscard]] auto elapsedTime(sycl::event const& startEvent, sycl::event const& endEvent) -> std::chrono::nanoseconds;

auto toString(sycl::info::device_type type) -> std::string;
auto summary(sycl::device const& dev) -> void;

} // namespace pffdtd
