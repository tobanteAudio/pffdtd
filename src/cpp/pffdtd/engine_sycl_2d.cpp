// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#include "engine_sycl_2d.hpp"

#include "pffdtd/double.hpp"
#include "pffdtd/exception.hpp"
#include "pffdtd/print.hpp"
#include "pffdtd/progress.hpp"
#include "pffdtd/sycl.hpp"
#include "pffdtd/time.hpp"
#include "pffdtd/utility.hpp"

#include <fmt/format.h>

#include <concepts>
#include <utility>

namespace pffdtd {

namespace {

template<typename Real>
struct AirUpdate;
template<typename Real>
struct BoundaryRigid;
template<typename Real>
struct BoundaryLoss;
template<typename Real>
struct CopyInput;
template<typename Real>
struct CopyOutput;

[[nodiscard]] constexpr auto to_ixy(auto x, auto y, auto /*Nx*/, auto Ny) { return x * Ny + y; }

template<typename Real>
auto scaleInput(std::vector<double> const& buf) {
  static constexpr auto min_exp = static_cast<double>(FloatTraits<Real>::minExponent);
  static constexpr auto max_exp = static_cast<double>(FloatTraits<Real>::maxExponent);
  println("min_exp = {}, min_exp = {}", min_exp, max_exp);

  auto const aexp  = 0.5;
  auto const pow2  = static_cast<int>(std::round(aexp * max_exp + (1 - aexp) * min_exp));
  auto const norm1 = static_cast<double>(ipow(2, pow2));

  auto const max_in    = *std::ranges::max_element(buf, [](auto l, auto r) { return std::fabs(l) < std::fabs(r); });
  auto const inv_infac = norm1 / max_in;
  auto const infac     = 1.0 / inv_infac;

  println(
      "max_in = {:.16e}, pow2 = {}, norm1 = {:.16e}, inv_infac = {:.16e}, infac = {:.16e}\n",
      max_in,
      pow2,
      norm1,
      inv_infac,
      infac
  );

  auto buf_out = buf;
  std::ranges::transform(buf_out, buf_out.begin(), [=](auto v) { return v * inv_infac; });
  return std::pair{buf_out, infac};
}

template<typename Real>
auto run(Simulation2D const& sim) {
  summary(sim);

  auto const Nx          = sim.Nx;
  auto const Ny          = sim.Ny;
  auto const Npts        = Nx * Ny;
  auto const Nb          = sim.adj_bn.size();
  auto const Nt          = sim.Nt;
  auto const Ns          = sim.in_ixy.size();
  auto const Nr          = sim.out_ixy.size();
  auto const loss_factor = static_cast<Real>(sim.loss_factor);

  auto queue  = sycl::queue{sycl::property::queue::enable_profiling{}};
  auto device = queue.get_device();
  summary(device);

  auto const [in_sigs_scaled, infac] = scaleInput<Real>(sim.in_sigs);

  auto u0_buf  = sycl::buffer<Real, 1>(sycl::range<1>(Npts));
  auto u1_buf  = sycl::buffer<Real, 1>(sycl::range<1>(Npts));
  auto u2_buf  = sycl::buffer<Real, 1>(sycl::range<1>(Npts));
  auto out_buf = sycl::buffer<Real, 2>(sycl::range<2>(Nr, Nt));

  auto in_mask_buf = sycl::buffer{sim.in_mask};
  auto bn_ixy_buf  = sycl::buffer{sim.bn_ixy};
  auto adj_bn_buf  = sycl::buffer{sim.adj_bn};
  auto in_sigs_buf = sycl::buffer{in_sigs_scaled};
  auto in_ixy_buf  = sycl::buffer{sim.in_ixy};
  auto out_ixy_buf = sycl::buffer{sim.out_ixy};

  auto elapsedAir      = std::chrono::nanoseconds{0};
  auto elapsedBoundary = std::chrono::nanoseconds{0};
  auto const start     = getTime();

  for (int64_t n{0}; n < Nt; ++n) {
    auto const sampleStart = getTime();

    auto const airEvent = queue.submit([&](sycl::handler& cgh) {
      auto u0      = sycl::accessor{u0_buf, cgh, sycl::write_only};
      auto u1      = sycl::accessor{u1_buf, cgh, sycl::read_only};
      auto u2      = sycl::accessor{u2_buf, cgh, sycl::read_only};
      auto in_mask = sycl::accessor{in_mask_buf, cgh, sycl::read_only};

      cgh.parallel_for<AirUpdate<Real>>(sycl::range<2>(Nx - 2, Ny - 2), [=](sycl::id<2> id) {
        auto const x   = static_cast<int64_t>(id[0] + 1);
        auto const y   = static_cast<int64_t>(id[1] + 1);
        auto const idx = to_ixy(x, y, 0, Ny);

        if (in_mask[idx] == 0) {
          return;
        }

        auto const left   = u1[idx - 1];
        auto const right  = u1[idx + 1];
        auto const bottom = u1[idx - Ny];
        auto const top    = u1[idx + Ny];
        auto const last   = u2[idx];

        u0[idx] = Real(0.5) * (left + right + bottom + top) - last;
      });
    });

    auto const boundaryStartEvent = queue.submit([&](sycl::handler& cgh) {
      auto u0     = sycl::accessor{u0_buf, cgh, sycl::write_only};
      auto u1     = sycl::accessor{u1_buf, cgh, sycl::read_only};
      auto u2     = sycl::accessor{u2_buf, cgh, sycl::read_only};
      auto bn_ixy = sycl::accessor{bn_ixy_buf, cgh, sycl::read_only};
      auto adj_bn = sycl::accessor{adj_bn_buf, cgh, sycl::read_only};

      cgh.parallel_for<BoundaryRigid<Real>>(Nb, [=](sycl::id<1> id) {
        auto const ib = bn_ixy[id];
        auto const K  = static_cast<Real>(adj_bn[id]);

        auto const last1 = u1[ib];
        auto const last2 = u2[ib];

        auto const left      = u1[ib - 1];
        auto const right     = u1[ib + 1];
        auto const bottom    = u1[ib - Ny];
        auto const top       = u1[ib + Ny];
        auto const neighbors = left + right + top + bottom;

        u0[ib] = (Real(2) - Real(0.5) * K) * last1 + Real(0.5) * neighbors - last2;
      });
    });

    auto const boundaryEndEvent = queue.submit([&](sycl::handler& cgh) {
      auto u0     = sycl::accessor{u0_buf, cgh, sycl::write_only};
      auto u2     = sycl::accessor{u2_buf, cgh, sycl::read_only};
      auto bn_ixy = sycl::accessor{bn_ixy_buf, cgh, sycl::read_only};
      auto adj_bn = sycl::accessor{adj_bn_buf, cgh, sycl::read_only};

      cgh.parallel_for<BoundaryLoss<Real>>(Nb, [=](sycl::id<1> id) {
        auto const ib      = bn_ixy[id];
        auto const K       = adj_bn[id];
        auto const current = u0[ib];
        auto const prev    = u2[ib];
        auto const K4      = static_cast<Real>(4 - K);

        u0[ib] = (current + loss_factor * K4 * prev) / (Real(1) + loss_factor * K4);
      });
    });

    queue.submit([&](sycl::handler& cgh) {
      auto u0      = sycl::accessor{u0_buf, cgh, sycl::read_write};
      auto in_ixy  = sycl::accessor{in_ixy_buf, cgh, sycl::read_only};
      auto in_sigs = sycl::accessor{in_sigs_buf, cgh, sycl::read_only};

      cgh.parallel_for<CopyInput<Real>>(Ns, [=](sycl::id<1> id) {
        auto src_ixy = to_ixy(id[0], n, Ns, Nt);
        u0[in_ixy[id[0]]] += static_cast<Real>(in_sigs[src_ixy]);
      });
    });

    queue.submit([&](sycl::handler& cgh) {
      auto u0      = sycl::accessor{u0_buf, cgh, sycl::read_only};
      auto out     = sycl::accessor{out_buf, cgh, sycl::write_only};
      auto out_ixy = sycl::accessor{out_ixy_buf, cgh, sycl::read_only};

      cgh.parallel_for<CopyOutput<Real>>(Nr, [=](sycl::id<1> id) {
        auto const r = id[0];
        out[r][n]    = u0[out_ixy[r]];
      });
    });

    queue.wait_and_throw();

    auto tmp = u2_buf;
    u2_buf   = u1_buf;
    u1_buf   = u0_buf;
    u0_buf   = tmp;

    auto const now = getTime();

    auto const elapsed       = now - start;
    auto const elapsedSample = now - sampleStart;

    auto const elapsedAirSample = elapsedTime(airEvent);
    elapsedAir += elapsedAirSample;

    auto const elapsedBoundarySample = elapsedTime(boundaryStartEvent, boundaryEndEvent);
    elapsedBoundary += elapsedBoundarySample;

    print(
        ProgressReport{
            .n                     = n,
            .Nt                    = Nt,
            .Npts                  = Npts,
            .Nb                    = static_cast<int64_t>(Nb),
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

  auto outputs = stdex::mdarray<double, stdex::dextents<size_t, 2>>(Nr, Nt);
  auto host    = sycl::host_accessor{out_buf, sycl::read_only};
  for (auto it{0UL}; std::cmp_less(it, Nt); ++it) {
    for (auto ir{0UL}; ir < Nr; ++ir) {
      outputs(ir, it) = static_cast<double>(host[ir][it]) * infac;
    }
  }

  fmt::println("");

  return outputs;
}

} // namespace

auto EngineSYCL2D::operator()(Simulation2D const& sim, Precision precision) const
    -> stdex::mdarray<double, stdex::dextents<size_t, 2>> {
  switch (precision) {
#if defined(PFFDTD_HAS_FLOAT16)
    case Precision::Half: return run<_Float16>(sim);
    case Precision::DoubleHalf: return run<Double<_Float16>>(sim);
#endif

    case Precision::Float: return run<float>(sim);
    case Precision::Double: return run<double>(sim);
    case Precision::DoubleFloat: return run<Double<float>>(sim);
    case Precision::DoubleDouble: return run<Double<double>>(sim);
    default: raisef<std::invalid_argument>("invalid precision {}", static_cast<int>(precision));
  }
}
} // namespace pffdtd
