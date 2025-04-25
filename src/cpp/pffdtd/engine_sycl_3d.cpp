// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch
#include "engine_sycl_3d.hpp"

#include "pffdtd/assert.hpp"
#include "pffdtd/double.hpp"
#include "pffdtd/exception.hpp"
#include "pffdtd/precision.hpp"
#include "pffdtd/progress.hpp"
#include "pffdtd/sycl.hpp"
#include "pffdtd/time.hpp"

#include <fmt/format.h>

#include <utility>

namespace pffdtd {

namespace {

template<typename T>
struct RigidUpdateCart;
template<typename T>
struct CopyToBoundary;
template<typename T>
struct ApplyBoundaryLoss;
template<typename T>
struct CopyFromBoundary;
template<typename T>
struct CopyABCs;
template<typename T>
struct FlipHaloXY;
template<typename T>
struct FlipHaloXZ;
template<typename T>
struct FlipHaloYZ;
template<typename T>
struct AddSources;
template<typename T>
struct AirUpdateCart;
template<typename T>
struct LossABC;
template<typename T>
struct ReadOutput;

template<typename Real>
auto run(Simulation3D const& sim) -> void {
  PFFDTD_ASSERT(sim.grid == Grid::CART);

  auto queue  = sycl::queue{sycl::property::queue::enable_profiling{}};
  auto device = queue.get_device();
  summary(device);

  auto const Nx   = sim.Nx;
  auto const Ny   = sim.Ny;
  auto const Nz   = sim.Nz;
  auto const NzNy = Nz * Ny;
  auto const Npts = sim.Npts;
  auto const Nb   = sim.Nb;
  auto const Nbl  = sim.Nbl;
  auto const Nba  = sim.Nba;
  auto const Nr   = sim.Nr;
  auto const Ns   = sim.Ns;
  auto const Nt   = sim.Nt;

  auto const lo2 = static_cast<Real>(sim.lo2);
  auto const sl2 = static_cast<Real>(sim.sl2);
  auto const l   = static_cast<Real>(sim.l);
  auto const a1  = static_cast<Real>(sim.a1);
  auto const a2  = static_cast<Real>(sim.a2);

  auto const ssaf_bnl_real  = convertTo<Real>(sim.ssaf_bnl);
  auto const mat_beta_real  = convertTo<Real>(sim.mat_beta);
  auto const mat_quads_real = convertTo<Real>(sim.mat_quads);

  auto Q_bna_buf     = sycl::buffer{sim.Q_bna};
  auto bn_mask_buf   = sycl::buffer{sim.bn_mask};
  auto adj_bn_buf    = sycl::buffer{sim.adj_bn};
  auto bn_ixyz_buf   = sycl::buffer{sim.bn_ixyz};
  auto bnl_ixyz_buf  = sycl::buffer{sim.bnl_ixyz};
  auto bna_ixyz_buf  = sycl::buffer{sim.bna_ixyz};
  auto in_ixyz_buf   = sycl::buffer{sim.in_ixyz};
  auto out_ixyz_buf  = sycl::buffer{sim.out_ixyz};
  auto in_sigs_buf   = sycl::buffer{sim.in_sigs};
  auto mat_beta_buf  = sycl::buffer{mat_beta_real};
  auto mat_bnl_buf   = sycl::buffer{sim.mat_bnl};
  auto mat_quads_buf = sycl::buffer{mat_quads_real};
  auto Mb_buf        = sycl::buffer{sim.Mb};
  auto ssaf_bnl_buf  = sycl::buffer{ssaf_bnl_real};

  auto u0_buf    = sycl::buffer<Real>(static_cast<size_t>(Npts));
  auto u1_buf    = sycl::buffer<Real>(static_cast<size_t>(Npts));
  auto u0b_buf   = sycl::buffer<Real>(static_cast<size_t>(Nbl));
  auto u1b_buf   = sycl::buffer<Real>(static_cast<size_t>(Nbl));
  auto u2b_buf   = sycl::buffer<Real>(static_cast<size_t>(Nbl));
  auto u2ba_buf  = sycl::buffer<Real>(static_cast<size_t>(Nba));
  auto vh1_buf   = sycl::buffer<Real>(static_cast<size_t>(Nbl * MMb));
  auto gh1_buf   = sycl::buffer<Real>(static_cast<size_t>(Nbl * MMb));
  auto u_out_buf = sycl::buffer<Real>(static_cast<size_t>(Nr * Nt));

  auto elapsedAir      = std::chrono::nanoseconds{0};
  auto elapsedBoundary = std::chrono::nanoseconds{0};
  auto const start     = getTime();

  for (int64_t n = 0; n < Nt; n++) {
    auto const sampleStart = getTime();

    // Rigid update
    auto boundaryStartEvent = queue.submit([&](sycl::handler& cgh) {
      auto u0      = sycl::accessor{u0_buf, cgh, sycl::write_only};
      auto u1      = sycl::accessor{u1_buf, cgh, sycl::read_only};
      auto bn_ixyz = sycl::accessor{bn_ixyz_buf, cgh, sycl::read_only};
      auto adj_bn  = sycl::accessor{adj_bn_buf, cgh, sycl::read_only};
      cgh.parallel_for<RigidUpdateCart<Real>>(Nb, [=](sycl::id<1> id) {
        auto const nb   = id[0];
        auto const ii   = bn_ixyz[nb];
        auto const adj  = adj_bn[nb];
        auto const Kint = sycl::popcount(adj);

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
      });
    });

    // Copy to boundary buffer
    queue.submit([&](sycl::handler& cgh) {
      auto u0       = sycl::accessor{u0_buf, cgh, sycl::read_only};
      auto u0b      = sycl::accessor{u0b_buf, cgh, sycl::write_only};
      auto bnl_ixyz = sycl::accessor{bnl_ixyz_buf, cgh, sycl::read_only};
      cgh.parallel_for<CopyToBoundary<Real>>(Nbl, [=](sycl::id<1> id) {
        auto nb = id[0];
        u0b[nb] = u0[bnl_ixyz[nb]];
      });
    });

    // Apply boundary loss
    queue.submit([&](sycl::handler& cgh) {
      auto mat_beta  = sycl::accessor{mat_beta_buf, cgh, sycl::read_only};
      auto mat_bnl   = sycl::accessor{mat_bnl_buf, cgh, sycl::read_only};
      auto ssaf_bnl  = sycl::accessor{ssaf_bnl_buf, cgh, sycl::read_only};
      auto u2b       = sycl::accessor{u2b_buf, cgh, sycl::read_only};
      auto Mb        = sycl::accessor{Mb_buf, cgh, sycl::read_only};
      auto mat_quads = sycl::accessor{mat_quads_buf, cgh, sycl::read_only};
      auto u0b       = sycl::accessor{u0b_buf, cgh, sycl::read_write};
      auto gh1       = sycl::accessor{gh1_buf, cgh, sycl::read_write};
      auto vh1       = sycl::accessor{vh1_buf, cgh, sycl::read_write};
      cgh.parallel_for<ApplyBoundaryLoss<Real>>(Nbl, [=](sycl::id<1> id) {
        auto nb         = static_cast<int64_t>(id[0]);
        Real _1         = 1.0;
        Real _2         = 2.0;
        int32_t const k = mat_bnl[nb];

        Real lo2Kbg = lo2 * ssaf_bnl[nb] * mat_beta[k];
        Real fac    = _2 * lo2 * ssaf_bnl[nb] / (_1 + lo2Kbg);

        Real u0bint = u0b[nb];
        Real u2bint = u2b[nb];

        u0bint = (u0bint + lo2Kbg * u2bint) / (_1 + lo2Kbg);

        Real vh1nb[MMb]{};
        for (int8_t m = 0; m < Mb[k]; m++) {
          int64_t const nbm = nb * MMb + m;
          int32_t const mbk = k * MMb + m;
          auto const& tm    = mat_quads[mbk];
          vh1nb[m]          = vh1[nbm];
          u0bint -= fac * (_2 * tm.bDh * vh1nb[m] - tm.bFh * gh1[nbm]);
        }

        Real du = u0bint - u2bint;

        for (int8_t m = 0; m < Mb[k]; m++) {
          int64_t const nbm = nb * MMb + m;
          int32_t const mbk = k * MMb + m;
          auto const& tm    = mat_quads[mbk];
          Real vh0nbm       = tm.b * du + tm.bd * vh1nb[m] - _2 * tm.bFh * gh1[nbm];
          gh1[nbm] += (vh0nbm + vh1nb[m]) / _2;
          vh1[nbm] = vh0nbm;
        }

        u0b[nb] = u0bint;
      });
    });

    // Copy from boundary buffer
    auto boundaryEndEvent = queue.submit([&](sycl::handler& cgh) {
      auto u0       = sycl::accessor{u0_buf, cgh, sycl::write_only};
      auto u0b      = sycl::accessor{u0b_buf, cgh, sycl::read_only};
      auto bnl_ixyz = sycl::accessor{bnl_ixyz_buf, cgh, sycl::read_only};
      cgh.parallel_for<CopyFromBoundary<Real>>(Nbl, [=](sycl::id<1> id) {
        auto nb          = id[0];
        u0[bnl_ixyz[nb]] = u0b[nb];
      });
    });

    // Copy last state ABCs
    auto airStartEvent = queue.submit([&](sycl::handler& cgh) {
      auto u2ba     = sycl::accessor{u2ba_buf, cgh, sycl::write_only};
      auto u0       = sycl::accessor{u0_buf, cgh, sycl::read_only};
      auto bna_ixyz = sycl::accessor{bna_ixyz_buf, cgh, sycl::read_only};

      cgh.parallel_for<CopyABCs<Real>>(Nba, [=](sycl::id<1> id) {
        auto nb  = id[0];
        u2ba[nb] = u0[bna_ixyz[nb]];
      });
    });

    // Flip halo XY
    queue.submit([&](sycl::handler& cgh) {
      auto u1 = sycl::accessor{u1_buf, cgh, sycl::read_write};
      cgh.parallel_for<FlipHaloXY<Real>>(sycl::range<2>(Nx, Ny), [=](sycl::id<2> id) {
        auto const i   = id[0] * NzNy + id[1] * Nz;
        u1[i + 0]      = u1[i + 2];
        u1[i + Nz - 1] = u1[i + Nz - 3];
      });
    });

    // Flip halo XZ
    queue.submit([&](sycl::handler& cgh) {
      auto u1 = sycl::accessor{u1_buf, cgh, sycl::read_write};
      cgh.parallel_for<FlipHaloXZ<Real>>(sycl::range<2>(Nx, Nz), [=](sycl::id<2> id) {
        auto const ix = id[0];
        auto const iz = id[1];

        u1[ix * NzNy + 0 * Nz + iz]        = u1[ix * NzNy + 2 * Nz + iz];
        u1[ix * NzNy + (Ny - 1) * Nz + iz] = u1[ix * NzNy + (Ny - 3) * Nz + iz];
      });
    });

    // Flip halo YZ
    queue.submit([&](sycl::handler& cgh) {
      auto u1 = sycl::accessor{u1_buf, cgh, sycl::read_write};
      cgh.parallel_for<FlipHaloYZ<Real>>(sycl::range<2>(Ny, Nz), [=](sycl::id<2> id) {
        auto const iy = id[0];
        auto const iz = id[1];

        u1[0 * NzNy + iy * Nz + iz]        = u1[2 * NzNy + iy * Nz + iz];
        u1[(Nx - 1) * NzNy + iy * Nz + iz] = u1[(Nx - 3) * NzNy + iy * Nz + iz];
      });
    });

    // Add sources
    queue.submit([&](sycl::handler& cgh) {
      auto u0      = sycl::accessor{u0_buf, cgh, sycl::read_write};
      auto in_sigs = sycl::accessor{in_sigs_buf, cgh, sycl::read_only};
      auto in_ixyz = sycl::accessor{in_ixyz_buf, cgh, sycl::read_only};
      cgh.parallel_for<AddSources<Real>>(Ns, [=](sycl::id<1> id) {
        auto const ns = id[0];
        auto const ii = in_ixyz[ns];
        u0[ii] += in_sigs[ns * Nt + n];
      });
    });

    // Air update
    queue.submit([&](sycl::handler& cgh) {
      auto u0      = sycl::accessor{u0_buf, cgh, sycl::write_only};
      auto u1      = sycl::accessor{u1_buf, cgh, sycl::read_only};
      auto bn_mask = sycl::accessor{bn_mask_buf, cgh, sycl::read_only};
      cgh.parallel_for<AirUpdateCart<Real>>(sycl::range<3>(Nx - 2, Ny - 2, Nz - 2), [=](sycl::id<3> id) {
        auto const ix = id[0] + 1;
        auto const iy = id[1] + 1;
        auto const iz = id[2] + 1;
        auto const ii = ix * NzNy + iy * Nz + iz;

        if ((GET_BIT(bn_mask[ii >> 3], ii % 8)) != 0) {
          return;
        }

        auto partial = a1 * u1[ii] - u0[ii];
        partial += a2 * u1[ii + NzNy];
        partial += a2 * u1[ii - NzNy];
        partial += a2 * u1[ii + Nz];
        partial += a2 * u1[ii - Nz];
        partial += a2 * u1[ii + 1];
        partial += a2 * u1[ii - 1];
        u0[ii] = partial;
      });
    });

    // ABC Loss
    auto airEndEvent = queue.submit([&](sycl::handler& cgh) {
      auto u0       = sycl::accessor{u0_buf, cgh, sycl::read_write};
      auto u2ba     = sycl::accessor{u2ba_buf, cgh, sycl::read_only};
      auto Q_bna    = sycl::accessor{Q_bna_buf, cgh, sycl::read_only};
      auto bna_ixyz = sycl::accessor{bna_ixyz_buf, cgh, sycl::read_only};
      cgh.parallel_for<LossABC<Real>>(Nba, [=](sycl::id<1> id) {
        auto const nb = id[0];
        auto const lQ = l * Q_bna[nb];
        auto const ib = bna_ixyz[nb];
        u0[ib]        = (u0[ib] + lQ * u2ba[nb]) / (Real(1) + lQ);
      });
    });

    // Read output
    queue.submit([&](sycl::handler& cgh) {
      auto u1       = sycl::accessor{u1_buf, cgh, sycl::read_only};
      auto out_ixyz = sycl::accessor{out_ixyz_buf, cgh, sycl::read_only};
      auto u_out    = sycl::accessor{u_out_buf, cgh, sycl::write_only};
      cgh.parallel_for<ReadOutput<Real>>(Nr, [=](sycl::id<1> id) {
        auto const nr      = id[0];
        auto const ii      = out_ixyz[nr];
        u_out[nr * Nt + n] = static_cast<double>(u1[ii]);
      });
    });

    queue.wait_and_throw();

    std::swap(u0_buf, u1_buf);

    auto tmp = u2b_buf;
    u2b_buf  = u1b_buf;
    u1b_buf  = u0b_buf;
    u0b_buf  = tmp;

    auto const now = getTime();

    auto const elapsed       = now - start;
    auto const elapsedSample = now - sampleStart;

    auto const elapsedAirSample = elapsedTime(airStartEvent, airEndEvent);
    elapsedAir += elapsedAirSample;

    auto const elapsedBoundarySample = elapsedTime(boundaryStartEvent, boundaryEndEvent);
    elapsedBoundary += elapsedBoundarySample;

    print(
        ProgressReport{
            .n                     = n,
            .Nt                    = Nt,
            .Npts                  = Npts,
            .Nb                    = Nb,
            .elapsed               = elapsed,
            .elapsedSample         = elapsedSample,
            .elapsedAir            = elapsedAir,
            .elapsedSampleAir      = elapsedAirSample,
            .elapsedBoundary       = elapsedBoundary,
            .elapsedSampleBoundary = elapsedBoundarySample,
            .numWorkers            = 1,
        }
    );
  }

  // Copy output to host
  auto host = sycl::host_accessor{u_out_buf, sycl::read_only};
  for (auto i{0UL}; std::cmp_less(i, Nr * Nt); ++i) {
    sim.u_out[i] = static_cast<double>(host[i]);
  }
}

} // namespace

auto EngineSYCL3D::operator()(Simulation3D const& sim) const -> void {
  switch (sim.precision) {
#if defined(PFFDTD_HAS_FLOAT16)
    case Precision::Half: run<_Float16>(sim); return;
    case Precision::DoubleHalf: run<Double<_Float16>>(sim); return;
#endif

    case Precision::Float: run<float>(sim); return;
    case Precision::Double: run<double>(sim); return;
    case Precision::DoubleFloat: run<Double<float>>(sim); return;
    case Precision::DoubleDouble: run<Double<double>>(sim); return;

    default: raisef<std::invalid_argument>("invalid precision {}", static_cast<int>(sim.precision));
  }
}

} // namespace pffdtd
