// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton
// CPU-based implementation of FDTD engine, with OpenMP

#include "engine_cpu_3d.hpp"

#include "pffdtd/double.hpp"
#include "pffdtd/exception.hpp"
#include "pffdtd/progress.hpp"
#include "pffdtd/time.hpp"
#include "pffdtd/utility.hpp"

#include <fmt/format.h>

#include <omp.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <vector>

namespace pffdtd {

namespace {

// function that does freq-dep RLC boundaries.  See 2016 ISMRA paper and
// accompanying webpage (slightly improved here)
template<typename Real>
auto process_bnl_fd(
    Real* u0b,
    Real const* u2b,
    Real const* ssaf_bnl,
    int8_t const* mat_bnl,
    int64_t Nbl,
    int8_t const* Mb,
    Real lo2,
    Real* vh1,
    Real* gh1,
    MatQuad<Real> const* mat_quads,
    Real const* mat_beta
) -> std::chrono::nanoseconds {
  auto const start = getTime();
#pragma omp parallel for schedule(static)
  for (int64_t nb = 0; nb < Nbl; nb++) {
    Real _1         = 1.0;
    Real _2         = 2.0;
    int32_t const k = mat_bnl[nb];

    Real lo2Kbg = lo2 * ssaf_bnl[nb] * mat_beta[k];
    Real fac    = _2 * lo2 * ssaf_bnl[nb] / (_1 + lo2Kbg);

    Real u0bint = u0b[nb];
    Real u2bint = u2b[nb];

    u0bint = (u0bint + lo2Kbg * u2bint) / (_1 + lo2Kbg);

    Real vh1nb[MMb];
    for (int8_t m = 0; m < Mb[k]; m++) {
      int64_t const nbm       = nb * MMb + m;
      int32_t const mbk       = k * MMb + m;
      MatQuad<Real> const& tm = mat_quads[mbk];
      vh1nb[m]                = vh1[nbm];
      u0bint -= fac * (_2 * tm.bDh * vh1nb[m] - tm.bFh * gh1[nbm]);
    }

    Real du = u0bint - u2bint;

    for (int8_t m = 0; m < Mb[k]; m++) {
      int64_t const nbm       = nb * MMb + m;
      int32_t const mbk       = k * MMb + m;
      MatQuad<Real> const& tm = mat_quads[mbk];
      Real vh0nbm             = tm.b * du + tm.bd * vh1nb[m] - _2 * tm.bFh * gh1[nbm];
      gh1[nbm] += (vh0nbm + vh1nb[m]) / _2;
      vh1[nbm] = vh0nbm;
    }

    u0b[nb] = u0bint;
  }
  return getTime() - start;
}

template<typename Real>
auto run(Simulation3D const& sim) -> void {
  auto const ssaf_bnl_real  = convertTo<Real>(sim.ssaf_bnl);
  auto const mat_beta_real  = convertTo<Real>(sim.mat_beta);
  auto const mat_quads_real = convertTo<Real>(sim.mat_quads);

  // keep local ints, scalars
  int64_t const Ns   = sim.Ns;
  int64_t const Nr   = sim.Nr;
  int64_t const Nt   = sim.Nt;
  int64_t const Npts = sim.Npts;
  int64_t const Nx   = sim.Nx;
  int64_t const Ny   = sim.Ny;
  int64_t const Nz   = sim.Nz;
  int64_t const Nb   = sim.Nb;
  int64_t const Nbl  = sim.Nbl;
  int64_t const Nba  = sim.Nba;
  auto const grid    = sim.grid;

  // keep local copies of pointers (style choice)
  int8_t const* Mb               = sim.Mb.data();
  int64_t const* bn_ixyz         = sim.bn_ixyz.data();
  int64_t const* bnl_ixyz        = sim.bnl_ixyz.data();
  int64_t const* bna_ixyz        = sim.bna_ixyz.data();
  int64_t const* in_ixyz         = sim.in_ixyz.data();
  int64_t const* out_ixyz        = sim.out_ixyz.data();
  uint16_t const* adj_bn         = sim.adj_bn.data();
  uint8_t const* bn_mask         = sim.bn_mask.data();
  int8_t const* mat_bnl          = sim.mat_bnl.data();
  int8_t const* Q_bna            = sim.Q_bna.data();
  double const* in_sigs          = sim.in_sigs.data();
  Real const* ssaf_bnl           = ssaf_bnl_real.data();
  Real const* mat_beta           = mat_beta_real.data();
  MatQuad<Real> const* mat_quads = mat_quads_real.data();
  double* u_out                  = sim.u_out.get();

  // allocate memory
  auto u0_buf   = std::vector<Real>(static_cast<size_t>(Npts));
  auto u1_buf   = std::vector<Real>(static_cast<size_t>(Npts));
  auto u0b_buf  = std::vector<Real>(static_cast<size_t>(Nbl));
  auto u1b_buf  = std::vector<Real>(static_cast<size_t>(Nbl));
  auto u2b_buf  = std::vector<Real>(static_cast<size_t>(Nbl));
  auto u2ba_buf = std::vector<Real>(static_cast<size_t>(Nba));
  auto vh1_buf  = std::vector<Real>(static_cast<size_t>(Nbl * MMb));
  auto gh1_buf  = std::vector<Real>(static_cast<size_t>(Nbl * MMb));

  auto* u0   = u0_buf.data();
  auto* u1   = u1_buf.data();
  auto* u0b  = u0b_buf.data();
  auto* u1b  = u1b_buf.data();
  auto* u2b  = u2b_buf.data();
  auto* u2ba = u2ba_buf.data();
  auto* vh1  = vh1_buf.data();
  auto* gh1  = gh1_buf.data();

  // sim coefficients
  auto const lo2 = static_cast<Real>(sim.lo2);
  auto const sl2 = static_cast<Real>(sim.sl2);
  auto const l   = static_cast<Real>(sim.l);
  auto const a1  = static_cast<Real>(sim.a1);
  auto const a2  = static_cast<Real>(sim.a2);

  // can control outside with OMP_NUM_THREADS env variable
  int const numWorkers = omp_get_max_threads();

  fmt::println("ENGINE: fcc_flag={}", static_cast<int8_t>(grid));
  fmt::println("fcc={}", isFCC(grid) ? "true" : "false");

  // for timing
  auto elapsedAir       = std::chrono::nanoseconds{0};
  auto elapsedBn        = std::chrono::nanoseconds{0};
  auto elapsedSampleAir = std::chrono::nanoseconds{0};
  auto elapsedSampleBn  = std::chrono::nanoseconds{0};
  auto const startTime  = getTime();

  int64_t const NzNy = Nz * Ny;
  for (int64_t n = 0; n < Nt; n++) {
    auto const sampleStartTime = getTime();

// copy last state ABCs
#pragma omp parallel for
    for (int64_t nb = 0; nb < Nba; nb++) {
      u2ba[nb] = u0[bna_ixyz[nb]];
    }
    if (grid == Grid::FCC_FOLDED) { // copy y-z face for FCC folded grid
#pragma omp parallel for
      for (int64_t ix = 0; ix < Nx; ix++) {
        for (int64_t iz = 0; iz < Nz; iz++) {
          u1[ix * NzNy + (Ny - 1) * Nz + iz] = u1[ix * NzNy + (Ny - 2) * Nz + iz];
        }
      }
    }

// halo flips for ABCs
#pragma omp parallel for
    for (int64_t ix = 0; ix < Nx; ix++) {
      for (int64_t iy = 0; iy < Ny; iy++) {
        u1[ix * NzNy + iy * Nz + 0]      = u1[ix * NzNy + iy * Nz + 2];
        u1[ix * NzNy + iy * Nz + Nz - 1] = u1[ix * NzNy + iy * Nz + Nz - 3];
      }
    }
#pragma omp parallel for
    for (int64_t ix = 0; ix < Nx; ix++) {
      for (int64_t iz = 0; iz < Nz; iz++) {
        u1[ix * NzNy + 0 * Nz + iz] = u1[ix * NzNy + 2 * Nz + iz];
      }
    }
    if (grid != Grid::FCC_FOLDED) { // only this y-face if not FCC folded grid
#pragma omp parallel for
      for (int64_t ix = 0; ix < Nx; ix++) {
        for (int64_t iz = 0; iz < Nz; iz++) {
          u1[ix * NzNy + (Ny - 1) * Nz + iz] = u1[ix * NzNy + (Ny - 3) * Nz + iz];
        }
      }
    }
#pragma omp parallel for
    for (int64_t iy = 0; iy < Ny; iy++) {
      for (int64_t iz = 0; iz < Nz; iz++) {
        u1[0 * NzNy + iy * Nz + iz]        = u1[2 * NzNy + iy * Nz + iz];
        u1[(Nx - 1) * NzNy + iy * Nz + iz] = u1[(Nx - 3) * NzNy + iy * Nz + iz];
      }
    }

    // air update for schemes
    if (grid == Grid::CART) { // cartesian scheme
#pragma omp parallel for
      for (int64_t ix = 1; ix < Nx - 1; ix++) {
        for (int64_t iy = 1; iy < Ny - 1; iy++) {
          for (int64_t iz = 1; iz < Nz - 1; iz++) { // contiguous
            int64_t const ii = ix * NzNy + iy * Nz + iz;
            if ((GET_BIT(bn_mask[ii >> 3], ii % 8)) == 0) {
              auto partial = a1 * u1[ii] - u0[ii];
              partial += a2 * u1[ii + NzNy];
              partial += a2 * u1[ii - NzNy];
              partial += a2 * u1[ii + Nz];
              partial += a2 * u1[ii - Nz];
              partial += a2 * u1[ii + 1];
              partial += a2 * u1[ii - 1];
              u0[ii] = partial;
            }
          }
        }
      }
    } else if (isFCC(grid)) {
#pragma omp parallel for
      for (int64_t ix = 1; ix < Nx - 1; ix++) {
        for (int64_t iy = 1; iy < Ny - 1; iy++) {
          // while loop iterates iterates over both types of FCC grids
          int64_t iz = grid == Grid::FCC ? 2 - (ix + iy) % 2 : 1;
          while (iz < Nz - 1) {
            int64_t const ii = ix * NzNy + iy * Nz + iz;
            if ((GET_BIT(bn_mask[ii >> 3], ii % 8)) == 0) {
              Real partial = a1 * u1[ii] - u0[ii];
              partial += a2 * u1[ii + NzNy + Nz];
              partial += a2 * u1[ii - NzNy - Nz];
              partial += a2 * u1[ii + Nz + 1];
              partial += a2 * u1[ii - Nz - 1];
              partial += a2 * u1[ii + NzNy + 1];
              partial += a2 * u1[ii - NzNy - 1];
              partial += a2 * u1[ii + NzNy - Nz];
              partial += a2 * u1[ii - NzNy + Nz];
              partial += a2 * u1[ii + Nz - 1];
              partial += a2 * u1[ii - Nz + 1];
              partial += a2 * u1[ii + NzNy - 1];
              partial += a2 * u1[ii - NzNy + 1];
              u0[ii] = partial;
            }
            iz += (grid == Grid::FCC ? 2 : 1);
          }
        }
      }
    }
    // ABC loss (2nd-order accurate first-order Engquist-Majda)
    for (int64_t nb = 0; nb < Nba; nb++) {
      Real const lQ    = l * Q_bna[nb];
      int64_t const ib = bna_ixyz[nb];
      u0[ib]           = (u0[ib] + lQ * u2ba[nb]) / (1.0 + lQ);
    }

    // rigid boundary nodes, using adj data
    elapsedSampleAir = getTime() - sampleStartTime;
    elapsedAir += elapsedSampleAir;
    if (grid == Grid::CART) {
#pragma omp parallel for
      for (int64_t nb = 0; nb < Nb; nb++) {
        auto const ii   = bn_ixyz[nb];
        auto const adj  = adj_bn[nb];
        auto const Kint = std::popcount(adj);

        auto const _2 = static_cast<Real>(2.0);
        auto const K  = static_cast<Real>(Kint);
        auto const b2 = static_cast<Real>(a2);
        auto const b1 = (_2 - sl2 * K);

        auto partial = b1 * u1[ii] - u0[ii];
        partial += b2 * get_bit_as<Real>(adj, 0) * u1[ii + NzNy];
        partial += b2 * get_bit_as<Real>(adj, 1) * u1[ii - NzNy];
        partial += b2 * get_bit_as<Real>(adj, 2) * u1[ii + Nz];
        partial += b2 * get_bit_as<Real>(adj, 3) * u1[ii - Nz];
        partial += b2 * get_bit_as<Real>(adj, 4) * u1[ii + 1];
        partial += b2 * get_bit_as<Real>(adj, 5) * u1[ii - 1];
        u0[ii] = partial;
      }
    } else if (isFCC(grid)) {
#pragma omp parallel for
      for (int64_t nb = 0; nb < Nb; nb++) {
        auto const ii   = bn_ixyz[nb];
        auto const adj  = adj_bn[nb];
        auto const Kint = std::popcount(adj);

        auto const _2 = static_cast<Real>(2.0);
        auto const K  = static_cast<Real>(Kint);
        auto const b2 = static_cast<Real>(a2);
        auto const b1 = (_2 - sl2 * K);

        auto partial = b1 * u1[ii] - u0[ii];
        partial += b2 * get_bit_as<Real>(adj, 0) * u1[ii + NzNy + Nz];
        partial += b2 * get_bit_as<Real>(adj, 1) * u1[ii - NzNy - Nz];
        partial += b2 * get_bit_as<Real>(adj, 2) * u1[ii + Nz + 1];
        partial += b2 * get_bit_as<Real>(adj, 3) * u1[ii - Nz - 1];
        partial += b2 * get_bit_as<Real>(adj, 4) * u1[ii + NzNy + 1];
        partial += b2 * get_bit_as<Real>(adj, 5) * u1[ii - NzNy - 1];
        partial += b2 * get_bit_as<Real>(adj, 6) * u1[ii + NzNy - Nz];
        partial += b2 * get_bit_as<Real>(adj, 7) * u1[ii - NzNy + Nz];
        partial += b2 * get_bit_as<Real>(adj, 8) * u1[ii + Nz - 1];
        partial += b2 * get_bit_as<Real>(adj, 9) * u1[ii - Nz + 1];
        partial += b2 * get_bit_as<Real>(adj, 10) * u1[ii + NzNy - 1];
        partial += b2 * get_bit_as<Real>(adj, 11) * u1[ii - NzNy + 1];
        u0[ii] = partial;
      }
    }

// read bn points (not strictly necessary, just mirrors CUDA implementation)
#pragma omp parallel for
    for (int64_t nb = 0; nb < Nbl; nb++) {
      u0b[nb] = u0[bnl_ixyz[nb]];
    }
    // process FD boundary nodes
    elapsedSampleBn = process_bnl_fd(u0b, u2b, ssaf_bnl, mat_bnl, Nbl, Mb, lo2, vh1, gh1, mat_quads, mat_beta);
    elapsedBn += elapsedSampleBn;
// write back
#pragma omp parallel for
    for (int64_t nb = 0; nb < Nbl; nb++) {
      u0[bnl_ixyz[nb]] = u0b[nb];
    }

    // read output at current sample
    for (int64_t nr = 0; nr < Nr; nr++) {
      int64_t const ii   = out_ixyz[nr];
      u_out[nr * Nt + n] = static_cast<double>(u1[ii]);
    }

    // add current sample to next (as per update)
    for (int64_t ns = 0; ns < Ns; ns++) {
      int64_t const ii = in_ixyz[ns];
      u0[ii] += static_cast<Real>(in_sigs[ns * Nt + n]);
    }

    // swap pointers
    std::swap(u0, u1);

    // using extra state here for simplicity
    auto* tmp = u2b;
    u2b       = u1b;
    u1b       = u0b;
    u0b       = tmp;

    auto const now           = getTime();
    auto const elapsed       = now - startTime;
    auto const elapsedSample = now - sampleStartTime;

    print(
        ProgressReport{
            .n                     = n,
            .Nt                    = Nt,
            .Npts                  = Npts,
            .Nb                    = Nb,
            .elapsed               = elapsed,
            .elapsedSample         = elapsedSample,
            .elapsedAir            = elapsedAir,
            .elapsedSampleAir      = elapsedSampleAir,
            .elapsedBoundary       = elapsedBn,
            .elapsedSampleBoundary = elapsedSampleBn,
            .numWorkers            = numWorkers,
        }
    );
  }
  fmt::println("");

  auto const endTime       = getTime();
  auto const elapsed       = Seconds(endTime - startTime).count();
  auto const elapsedAirSec = Seconds(elapsedAir).count();
  auto const elapsedBnSec  = Seconds(elapsedBn).count();

  fmt::println("Air update: {:.6}s, {:.2} Mvox/s", elapsedAirSec, static_cast<double>(Npts * Nt) / 1e6 / elapsedAirSec);
  fmt::println("Boundary loop: {:.6}s, {:.2} Mvox/s", elapsedBnSec, static_cast<double>(Nb * Nt) / 1e6 / elapsedBnSec);
  fmt::println("Combined (total): {:.6}s, {:.2} Mvox/s", elapsed, static_cast<double>(Npts * Nt) / 1e6 / elapsed);
}

} // namespace

auto EngineCPU3D::operator()(Simulation3D const& sim) const -> void {
  switch (sim.precision) {
    case Precision::Float: run<float>(sim); return;
    case Precision::Double: run<double>(sim); return;
    case Precision::DoubleFloat: run<Double<float>>(sim); return;
    case Precision::DoubleDouble: run<Double<double>>(sim); return;
    default: raisef<std::invalid_argument>("invalid precision {}", static_cast<int>(sim.precision));
  }
}

} // namespace pffdtd
