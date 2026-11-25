// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#pragma once

#include <fmt/core.h>
#include <fmt/format.h>

#include <stdexcept>

namespace pffdtd {

template<typename E, typename... Args>
[[noreturn]] auto raise(Args&&... args) -> void {
  throw E{std::forward<Args>(args)...};
}

template<typename E, typename... Args>
[[noreturn]] auto raisef(fmt::format_string<Args...> str, Args&&... args) -> void {
  raise<E>(fmt::format(str, std::forward<Args>(args)...));
}

} // namespace pffdtd
