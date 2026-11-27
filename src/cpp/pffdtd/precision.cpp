// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#include "precision.hpp"

#include "pffdtd/exception.hpp"

namespace pffdtd {

template<typename T>
constexpr auto EPS = 0.0;

template<>
constexpr auto EPS<float> = 1.19209289e-07;

auto getEpsilon(Precision precision) -> double {
  switch (precision) {
    case Precision::Float: return EPS<float>;
    case Precision::Double: return EPS<double>;
    default: break;
  }

  raisef<std::invalid_argument>("invalid precision = {}", int(precision));
}

auto getMinMaxExponent(Precision precision) -> std::pair<int, int> {
  switch (precision) {
    case Precision::Float:
      return std::pair{
          std::numeric_limits<float>::min_exponent,
          std::numeric_limits<float>::max_exponent,
      };
    case Precision::Double:
      return std::pair{
          std::numeric_limits<double>::min_exponent,
          std::numeric_limits<double>::max_exponent,
      };
    default: break;
  }

  raisef<std::invalid_argument>("invalid precision = {}", int(precision));
}

} // namespace pffdtd
