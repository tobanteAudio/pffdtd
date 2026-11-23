// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#pragma once

#include "mat_quad.hpp"

static_assert(sizeof(long) == 8, "sizeof(long) should equal sizeof(int64_t)");

namespace pffdtd {

template<typename Real>
struct Constants2D {
  long Nx;
  long Ny;
  long Nt;
  Real l2;
  Real lossFactor;
};

template<typename Real>
struct Constants3D {
  Real l;
  Real lo2;
  Real sl2;
  Real a1;
  Real a2;
  long Nx;
  long Ny;
  long Nz;
  long NzNy;
  long Nb;
  long Nbl;
  long Nba;
  long Ns;
  long Nr;
  long Nt;
};

template<typename T>
auto get_bit(long word, long pos) -> T {
  return static_cast<T>((word >> pos) & 1);
}

} // namespace pffdtd
