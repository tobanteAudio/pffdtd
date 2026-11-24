// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton
// Misc function definitions not specific to FDTD simulation
#pragma once

#include "pffdtd/mat_quad.hpp"

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <type_traits>
#include <vector>

#ifndef DIV_CEIL
  #define DIV_CEIL(x, y) (((x) + (y) - 1) / (y)) // this works for x≥0 and y>0
#endif
#define GET_BIT(var, pos)          (((var) >> (pos)) & 1)
#define SET_BIT(var, pos)          ((var) |= (1ULL << (pos)))
#define SET_BIT_VAL(var, pos, val) ((var) = ((var) & ~(1ULL << (pos))) | ((val) << (pos)))

namespace pffdtd {

template<typename T>
[[nodiscard]] auto allocate_zeros(std::integral auto count) -> std::unique_ptr<T[]> {
  auto ptr = std::make_unique<T[]>(static_cast<size_t>(count));
  std::memset(static_cast<void*>(ptr.get()), 0, static_cast<size_t>(count) * sizeof(T));
  return ptr;
}

template<typename T>
[[nodiscard]] constexpr auto get_bit_as(std::integral auto word, std::integral auto pos) -> T {
  return static_cast<T>(GET_BIT(word, pos));
}

template<typename T>
auto convertTo(std::vector<double> const& in) {
  auto out = std::vector<T>(in.size());
  std::ranges::transform(in, out.begin(), [](auto v) { return static_cast<T>(v); });
  return out;
}

template<typename T>
auto convertTo(std::vector<MatQuad<double>> const& in) {
  auto out = std::vector<MatQuad<T>>(in.size());
  std::ranges::transform(in, out.begin(), [](MatQuad<double> mq) {
    return MatQuad<T>{
        .b   = static_cast<T>(mq.b),
        .bd  = static_cast<T>(mq.bd),
        .bDh = static_cast<T>(mq.bDh),
        .bFh = static_cast<T>(mq.bFh),
    };
  });
  return out;
}

template<std::integral T>
[[nodiscard]] constexpr auto idiv(T x, T y) noexcept -> T {
  return (x + y - 1) / y;
}

template<std::integral T>
[[nodiscard]] constexpr auto ipow(T base, T exponent) -> T {
  T result = 1;
  for (T i = 0; i < exponent; i++) {
    result *= base;
  }
  return result;
}

template<auto Base>
  requires std::integral<decltype(Base)>
[[nodiscard]] constexpr auto ipow(decltype(Base) exponent) -> decltype(Base) {
  using Int  = decltype(Base);
  using UInt = std::make_unsigned_t<Int>;

  if constexpr (Base == Int(2)) {
    return static_cast<Int>(UInt(1) << UInt(exponent));
  } else if constexpr (Base == Int(4)) {
    return static_cast<Int>(UInt(1) << UInt(exponent * Int(2)));
  } else if constexpr (Base == Int(8)) {
    return static_cast<Int>(UInt(1) << UInt(exponent * Int(3)));
  } else if constexpr (Base == Int(16)) {
    return static_cast<Int>(UInt(1) << UInt(exponent * Int(4)));
  } else if constexpr (Base == Int(32)) {
    return static_cast<Int>(UInt(1) << UInt(exponent * Int(5)));
  } else {
    return ipow(Base, exponent);
  }
}

} // namespace pffdtd
