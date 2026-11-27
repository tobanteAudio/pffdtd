// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#pragma once

#include <utility>

namespace pffdtd {

enum struct Precision : unsigned char {
  Float,
  Double,
};

[[nodiscard]] auto getEpsilon(Precision precision) -> double;
[[nodiscard]] auto getMinMaxExponent(Precision precision) -> std::pair<int, int>;

} // namespace pffdtd
