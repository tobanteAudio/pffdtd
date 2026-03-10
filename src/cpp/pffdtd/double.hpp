// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch
#pragma once

#include "pffdtd/float.hpp"
#include "pffdtd/utility.hpp"

#if defined(SYCL_LANGUAGE_VERSION)
  #include "pffdtd/sycl.hpp"
#endif

#include <cmath>
#include <concepts>
#include <limits>

namespace pffdtd {

/// DoubleDouble extented precision
///
/// - https://hal.science/hal-01351529v3/document
/// - https://inria.hal.science/inria-00070314/document
/// - https://people.eecs.berkeley.edu/~jrs/papers/robustr.pdf
/// - https://github.com/sukop/doubledouble
/// - https://github.com/JuliaMath/DoubleFloats.jl
/// - https://github.com/FlorisSteenkamp/double-double
template<typename Real>
struct Double {
  constexpr Double() = default;

  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  constexpr Double(Real x) noexcept {
    auto const [h, l] = split(x);
    _high             = h;
    _low              = l;
  }

  constexpr Double(Real x, Real y) noexcept : _high{x}, _low{y} { }

  [[nodiscard]] constexpr auto high() const noexcept -> Real { return _high; }

  [[nodiscard]] constexpr auto low() const noexcept -> Real { return _low; }

  template<typename OtherReal>
  explicit constexpr operator OtherReal() const noexcept {
    return static_cast<OtherReal>(high());
  }

  template<typename OtherReal>
  explicit constexpr operator Double<OtherReal>() const noexcept {
    return Double<OtherReal>{
        static_cast<OtherReal>(high()),
        static_cast<OtherReal>(low()),
    };
  }

  friend constexpr auto operator+(Double x) noexcept -> Double { return x; }

  friend constexpr auto operator-(Double x) noexcept -> Double { return {-x.high(), -x.low()}; }

  friend constexpr auto operator+(Double lhs, Real rhs) noexcept -> Double {
    auto [sl, sh] = twoSum(lhs.high(), rhs);
    return twoSumFast(sh, lhs.low() + sl);
  }

  friend constexpr auto operator+(Double lhs, Double rhs) noexcept -> Double {
    auto [r, e] = twoSum(lhs.high(), rhs.high());
    e += lhs.low() + rhs.low();
    return twoSumFast(r, e);
  }

  friend constexpr auto operator-(Double lhs, Double rhs) noexcept -> Double {
    auto [r, e] = twoDifference(lhs.high(), rhs.high());
    e += lhs.low() - rhs.low();
    return twoSumFast(r, e);
  }

  friend constexpr auto operator*(Double lhs, Double rhs) noexcept -> Double {
    auto [r, e] = twoProduct(lhs.high(), rhs.high());
    e += lhs.high() * rhs.low() + lhs.low() * rhs.high();
    return twoSumFast(r, e);
  }

  friend constexpr auto operator/(Double lhs, Double rhs) noexcept -> Double {
    auto r      = lhs.high() / rhs.high();
    auto [s, f] = twoProduct(r, rhs.high());
    auto e      = (lhs.high() - s - f + lhs.low() - r * rhs.low()) / rhs.high();
    return twoSumFast(r, e);
  }

  friend constexpr auto operator+=(Double& lhs, Double rhs) noexcept -> Double& {
    lhs = lhs + rhs;
    return lhs;
  }

  friend constexpr auto operator-=(Double& lhs, Double rhs) noexcept -> Double& {
    lhs = lhs - rhs;
    return lhs;
  }

  friend constexpr auto operator*=(Double& lhs, Double rhs) noexcept -> Double& {
    lhs = lhs * rhs;
    return lhs;
  }

  friend constexpr auto operator/=(Double& lhs, Double rhs) noexcept -> Double& {
    lhs = lhs / rhs;
    return lhs;
  }

  friend constexpr auto operator==(Double lhs, Double rhs) noexcept -> bool = default;

  private:
  [[nodiscard]] static constexpr auto split(Real a) noexcept -> Double {
    constexpr auto const digits = FloatTraits<Real>::digits;
    constexpr auto const f      = ipow<2>(idiv(digits, 2)) + 1;

    if constexpr (std::same_as<Real, double> and sizeof(double) == 8) {
      static_assert(f == 134217729); // 2**27 + 1
    }

    auto const c   = static_cast<Real>(f) * a;
    auto const a_h = c - (c - a);
    auto const a_l = a - a_h;

    return {a_h, a_l};
  }

  [[nodiscard]] static constexpr auto twoSum(Real x, Real y) noexcept -> Double {
    auto r = x + y;
    auto t = r - x;
    auto e = (x - (r - t)) + (y - t);
    return Double{r, e};
  }

  [[nodiscard]] static constexpr auto twoSumFast(Real x, Real y) noexcept -> Double {
    auto r = x + y;
    auto e = y - (r - x);
    return Double{r, e};
  }

  [[nodiscard]] static constexpr auto twoDifference(Real x, Real y) noexcept -> Double {
    auto r = x - y;
    auto t = r - x;
    auto e = (x - (r - t)) - (y + t);
    return Double{r, e};
  }

  [[nodiscard]] static constexpr auto twoProduct(Real x, Real y) noexcept -> Double {
    auto r = x * y;

#if defined(PFFDTD_HAS_FLOAT16)
    if constexpr (std::same_as<Real, _Float16>) {
      auto e = static_cast<_Float16>(std::fma(float{x}, float{y}, float{-r}));
      return Double{r, e};
    } else
#endif
    {
      auto e = std::fma(x, y, -r);
      return Double{r, e};
    }
  }

  Real _high{static_cast<Real>(0)};
  Real _low{static_cast<Real>(0)};
};

template<typename Real>
struct FloatTraits<Double<Real>> : FloatTraits<Real> { };

} // namespace pffdtd
