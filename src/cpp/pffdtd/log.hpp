// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#pragma once

#include <fmt/format.h>

namespace pffdtd {

struct Log {
  template<typename... Args>
  static auto info(fmt::format_string<Args...> str, Args&&... args) -> void {
    fmt::println("--SIM-ENGINE: {}", fmt::format(str, std::forward<Args>(args)...));
  }
};

} // namespace pffdtd
