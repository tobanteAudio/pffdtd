// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#pragma once

#include <limits>

namespace pffdtd {

template<typename T>
struct FloatTraits;

template<>
struct FloatTraits<float> {
  static constexpr auto digits      = std::numeric_limits<float>::digits;
  static constexpr auto minExponent = std::numeric_limits<float>::min_exponent;
  static constexpr auto maxExponent = std::numeric_limits<float>::max_exponent;
};

template<>
struct FloatTraits<double> {
  static constexpr auto digits      = std::numeric_limits<double>::digits;
  static constexpr auto minExponent = std::numeric_limits<double>::min_exponent;
  static constexpr auto maxExponent = std::numeric_limits<double>::max_exponent;
};

} // namespace pffdtd
