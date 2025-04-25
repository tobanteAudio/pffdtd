// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#include "engine_cpu_2d.hpp"

#include "pffdtd/double.hpp"
#include "pffdtd/exception.hpp"
#include "pffdtd/progress.hpp"
#include "pffdtd/time.hpp"

#include <fmt/format.h>

#include <omp.h>

#include <algorithm>
#include <concepts>
#include <ranges>

namespace pffdtd {

namespace {

template<typename Real>
auto run(Simulation2D const& sim) {
  summary(sim);

  auto const Nx         = sim.Nx;
  auto const Ny         = sim.Ny;
  auto const Npts       = sim.Nx * sim.Ny;
  auto const Nt         = sim.Nt;
  auto const Nb         = static_cast<int64_t>(sim.adj_bn.size());
  auto const Ns         = static_cast<int64_t>(sim.in_ixy.size());
  auto const Nr         = static_cast<int64_t>(sim.out_ixy.size());
  auto const lossFactor = static_cast<Real>(sim.loss_factor);

  auto u0_buf  = stdex::mdarray<Real, stdex::dextents<size_t, 1>>(Npts);
  auto u1_buf  = stdex::mdarray<Real, stdex::dextents<size_t, 1>>(Npts);
  auto u2_buf  = stdex::mdarray<Real, stdex::dextents<size_t, 1>>(Npts);
  auto out_buf = stdex::mdarray<double, stdex::dextents<size_t, 2>>(Nr, Nt);

  auto u0  = u0_buf.to_mdspan();
  auto u1  = u1_buf.to_mdspan();
  auto u2  = u2_buf.to_mdspan();
  auto out = out_buf.to_mdspan();

  auto elapsedAir      = std::chrono::nanoseconds{0};
  auto elapsedBoundary = std::chrono::nanoseconds{0};

  auto const numWorkers = omp_get_max_threads();
  auto const start      = getTime();

  for (auto n{0LL}; n < Nt; ++n) {
    auto const sampleStart = getTime();

    auto const elapsedAirSample = timeit([&] {
// Air Update
#pragma omp parallel for
      for (int64_t x = 1; x < Nx - 1; ++x) {
        for (int64_t y = 1; y < Ny - 1; ++y) {
          auto const idx    = x * Ny + y;
          auto const left   = u1[idx - 1];
          auto const right  = u1[idx + 1];
          auto const bottom = u1[idx - Ny];
          auto const top    = u1[idx + Ny];
          auto const last   = u2[idx];
          auto const tmp    = static_cast<Real>(0.5) * (left + right + bottom + top) - last;

          u0[idx] = sim.in_mask[idx] * tmp;
        }
      }
    });

    auto const elapsedBoundarySample = timeit([&] {
// Boundary Rigid
#pragma omp parallel for
      for (int64_t i = 0; i < Nb; ++i) {
        auto const ib = sim.bn_ixy[i];
        auto const K  = static_cast<Real>(sim.adj_bn[i]);

        auto const last1 = u1[ib];
        auto const last2 = u2[ib];

        auto const left      = u1[ib - 1];
        auto const right     = u1[ib + 1];
        auto const bottom    = u1[ib - Ny];
        auto const top       = u1[ib + Ny];
        auto const neighbors = left + right + top + bottom;

        u0[ib] = (Real(2) - Real(0.5) * K) * last1 + Real(0.5) * neighbors - last2;
      }

// Boundary Loss
#pragma omp parallel for
      for (int64_t i = 0; i < Nb; ++i) {
        auto const ib = sim.bn_ixy[i];
        auto const K  = sim.adj_bn[i];
        auto const K4 = static_cast<Real>(4 - K);
        auto const lf = lossFactor;

        auto const current = u0[ib];
        auto const prev    = u2[ib];

        u0[ib] = (current + lf * K4 * prev) / (Real(1) + lf * K4);
      }
    });

    // Add sources
    for (int64_t s = 0; s < Ns; ++s) {
      u0[sim.in_ixy[s]] += sim.in_sigs[n];
    }

    // Read outputs
    for (int64_t r = 0; r < Nr; ++r) {
      out[r, n] = static_cast<double>(u0[sim.out_ixy[r]]);
    }

    // Rotate buffers
    auto tmp = u2;
    u2       = u1;
    u1       = u0;
    u0       = tmp;

    auto const now           = getTime();
    auto const elapsed       = now - start;
    auto const elapsedSample = now - sampleStart;

    elapsedAir += elapsedAirSample;
    elapsedBoundary += elapsedBoundarySample;

    print(
        ProgressReport{
            .n                     = n,
            .Nt                    = Nt,
            .Npts                  = Nx * Ny,
            .Nb                    = Nb,
            .elapsed               = elapsed,
            .elapsedSample         = elapsedSample,
            .elapsedAir            = elapsedAir,
            .elapsedSampleAir      = elapsedAirSample,
            .elapsedBoundary       = elapsedBoundary,
            .elapsedSampleBoundary = elapsedBoundarySample,
            .numWorkers            = numWorkers,
        }
    );
  }

  fmt::println("");

  return out_buf;
}

} // namespace

auto EngineCPU2D::operator()(Simulation2D const& sim, Precision precision) const
    -> stdex::mdarray<double, stdex::dextents<size_t, 2>> {
  switch (precision) {
    case Precision::Float: return run<float>(sim);
    case Precision::Double: return run<double>(sim);
    case Precision::DoubleFloat: return run<Double<float>>(sim);
    case Precision::DoubleDouble: return run<Double<double>>(sim);
#if defined(PFFDTD_HAS_FLOAT16)
    case Precision::Half: return run<_Float16>(sim);
    case Precision::DoubleHalf: return run<Double<_Float16>>(sim);
#endif
    default: raisef<std::invalid_argument>("invalid precision {}", static_cast<int>(precision));
  }
}

} // namespace pffdtd
