// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton

#include "engine_cuda_3d.hpp"

#include "pffdtd/assert.hpp"
#include "pffdtd/progress.hpp"
#include "pffdtd/time.hpp"
#include "pffdtd/utility.hpp"

#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>

namespace pffdtd {

namespace {

template<typename Real>
__device__ auto add(Real x, Real y) -> Real {
  if constexpr (std::same_as<Real, double>) {
    return __dadd_rn(x, y);
  } else if constexpr (std::same_as<Real, float>) {
    return __fadd_rz(x, y);
  } else {
    static_assert(always_false<Real>);
  }
}

template<typename Real>
__device__ auto fma(Real x, Real y, Real z) -> Real {
  if constexpr (std::same_as<Real, double>) {
    return __fma_rn(x, y, z);
  } else if constexpr (std::same_as<Real, float>) {
    return __fmaf_rn(x, y, z);
  } else {
    static_assert(always_false<Real>);
  }
}

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define gpuErrchk(ans)                                                                                                 \
  do {                                                                                                                 \
    gpuAssert((ans), __FILE__, __LINE__);                                                                              \
  } while (false)

auto gpuAssert(cudaError_t code, char const* file, int line) -> void {
  if (code != cudaSuccess) {
    throw std::runtime_error{
        cudaGetErrorString(code) + std::string(":") + std::string(file) + ":" + std::to_string(line)
    };
  }
}

[[nodiscard]] auto elapsedTime(cudaEvent_t start, cudaEvent_t end) -> std::chrono::nanoseconds {
  auto millis = 0.0F;
  gpuErrchk(cudaEventElapsedTime(&millis, start, end));
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<float, std::milli>(millis));
}

// want 0 to map to 1, otherwise kernel errors
constexpr auto cu_divceil(auto x, auto y) { return ((DIV_CEIL(x, y) == 0) ? (1) : (DIV_CEIL(x, y))); }

// thread-block dims for 3d kernels
constexpr auto cuBx = 32;
constexpr auto cuBy = 2;
constexpr auto cuBz = 2;

// thread-block dims for 2d kernels (fcc fold, ABCs)
constexpr auto cuBx2 = 16;
constexpr auto cuBy2 = 8;

// thread-block dims for 1d kernels (bn, ABC loss)
constexpr auto cuBrw = 128;
constexpr auto cuBb  = 128;

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables)
// constant memory (all per device)
__constant__ float c1_f32;
__constant__ float c2_f32;
__constant__ float cl_f32;
__constant__ float csl2_f32;
__constant__ float clo2_f32;
__constant__ double c1_f64;
__constant__ double c2_f64;
__constant__ double cl_f64;
__constant__ double csl2_f64;
__constant__ double clo2_f64;
__constant__ int64_t cuNx;
__constant__ int64_t cuNy;
__constant__ int64_t cuNz;
__constant__ int64_t cuNb;
__constant__ int64_t cuNbl;
__constant__ int64_t cuNba;
__constant__ int64_t cuNxNy;
__constant__ int8_t cuMb[MNm]; // to store Mb per mat

// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

// this is data on host, sometimes copied and recomputed for copy to GPU devices
// (indices), sometimes just aliased pointers (scalar arrays)
template<typename Real>
struct HostData {                      // arrays on host (for copy), mirrors gpu local data
  std::unique_ptr<int64_t[]> in_ixyz;  // recomputed
  std::unique_ptr<int64_t[]> out_ixyz; // recomputed
  std::unique_ptr<int64_t[]> bn_ixyz;  // recomputed
  std::unique_ptr<int64_t[]> bnl_ixyz; // recomputed
  std::unique_ptr<int64_t[]> bna_ixyz; // recomputed
  std::unique_ptr<uint8_t[]> bn_mask;  // recomputed
  double const* in_sigs{};             // aliased
  Real* u_out_buf{};                   // aliased
  double* u_out{};                     // aliased
  Real const* ssaf_bnl{};              // aliased
  int8_t const* Q_bna{};               // aliased
  uint16_t const* adj_bn{};            // aliased
  int8_t const* mat_bnl{};             // aliased
  int8_t const* K_bn{};                // aliased
  int64_t Ns{};
  int64_t Nr{};
  int64_t Npts{};
  int64_t Nx{};
  int64_t Nxh{};
  int64_t Nb{};
  int64_t Nbl{};
  int64_t Nba{};
  int64_t Nbm{}; // bytes for bn_mask
};

// these are arrays pointing to GPU device memory, or CUDA stuff (dim3, events)
template<typename Real>
struct DeviceData { // for or on gpu (arrays all on GPU)
  int64_t* bn_ixyz{};
  int64_t* bnl_ixyz{};
  int64_t* bna_ixyz{};
  int8_t* Q_bna{};
  int64_t* out_ixyz{};
  uint16_t* adj_bn{};
  Real* ssaf_bnl{};
  uint8_t* bn_mask{};
  int8_t* mat_bnl{};
  int8_t* K_bn{};
  Real* mat_beta{};
  MatQuad<Real>* mat_quads{};
  Real* u0{};
  Real* u1{};
  Real* u0b{};
  Real* u1b{};
  Real* u2b{};
  Real* u2ba{};
  Real* vh1{};
  Real* gh1{};
  Real* u_out_buf{};
  dim3 block_dim_air;
  dim3 grid_dim_air;
  dim3 block_dim_fold;
  dim3 grid_dim_fold;
  dim3 block_dim_readout;
  dim3 grid_dim_readout;
  dim3 block_dim_bn;
  dim3 block_dim_halo_xy;
  dim3 block_dim_halo_yz;
  dim3 block_dim_halo_xz;
  dim3 grid_dim_bn;
  dim3 grid_dim_bnl;
  dim3 grid_dim_bna;
  dim3 grid_dim_halo_xy;
  dim3 grid_dim_halo_yz;
  dim3 grid_dim_halo_xz;
  cudaStream_t cuStream_air{};
  cudaStream_t cuStream_bn{};
  cudaEvent_t cuEv_air_start{};
  cudaEvent_t cuEv_air_end{};
  cudaEvent_t cuEv_bn_roundtrip_start{};
  cudaEvent_t cuEv_bn_roundtrip_end{};
  cudaEvent_t cuEv_readout_end{};
  int64_t totalmembytes{};
};

template<typename Real>
__device__ auto constant(float f32, double f64) -> Real {
  if constexpr (std::is_same_v<Real, float>) {
    return f32;
  } else if constexpr (std::is_same_v<Real, double>) {
    return f64;
  } else {
    static_assert(always_false<Real>);
  }
}

// NB. 'x' is contiguous dim in CUDA domain

// vanilla scheme, unrolled, intrinsics to control rounding errors
template<typename Real>
__global__ void KernelAirCart(Real* __restrict__ u0, Real const* __restrict__ u1, uint8_t const* __restrict__ bn_mask) {
  int64_t const cx = blockIdx.x * cuBx + threadIdx.x + 1;
  int64_t const cy = blockIdx.y * cuBy + threadIdx.y + 1;
  int64_t const cz = blockIdx.z * cuBz + threadIdx.z + 1;

  auto const c1 = constant<Real>(c1_f32, c1_f64);
  auto const c2 = constant<Real>(c2_f32, c2_f64);

  if ((cx < cuNx - 1) && (cy < cuNy - 1) && (cz < cuNz - 1)) {
    int64_t const ii = cz * cuNxNy + cy * cuNx + cx;
    // divide-conquer add for better accuracy
    Real tmp1 = NAN;
    Real tmp2 = NAN;
    tmp1      = add(u1[ii + cuNxNy], u1[ii - cuNxNy]);
    tmp2      = add(u1[ii + cuNx], u1[ii - cuNx]);
    tmp1      = add(tmp1, tmp2);
    tmp2      = add(u1[ii + 1], u1[ii - 1]);
    tmp1      = add(tmp1, tmp2);
    tmp1      = fma(c1, u1[ii], fma(c2, tmp1, -u0[ii]));

    // write final value back to global memory
    if ((GET_BIT(bn_mask[ii >> 3], ii % 8)) == 0) {
      u0[ii] = tmp1;
    }
  }
}

// air update for FCC, on folded grid (improvement to 2013 DAFx paper)
template<typename Real>
__global__ void KernelAirFCC(Real* __restrict__ u0, Real const* __restrict__ u1, uint8_t const* __restrict__ bn_mask) {
  // get ix,iy,iz from thread and block Id's
  int64_t const cx = blockIdx.x * cuBx + threadIdx.x + 1;
  int64_t const cy = blockIdx.y * cuBy + threadIdx.y + 1;
  int64_t const cz = blockIdx.z * cuBz + threadIdx.z + 1;

  auto const c1 = constant<Real>(c1_f32, c1_f64);
  auto const c2 = constant<Real>(c2_f32, c2_f64);

  if ((cx < cuNx - 1) && (cy < cuNy - 1) && (cz < cuNz - 1)) {
    // x is contiguous
    int64_t const ii = cz * cuNxNy + cy * cuNx + cx;
    Real tmp1        = NAN;
    Real tmp2        = NAN;
    Real tmp3        = NAN;
    Real tmp4        = NAN;
    // divide-conquer add as much as possible
    tmp1 = add(u1[ii + cuNxNy + cuNx], u1[ii - cuNxNy - cuNx]);
    tmp2 = add(u1[ii + cuNx + 1], u1[ii - cuNx - 1]);
    tmp1 = add(tmp1, tmp2);
    tmp3 = add(u1[ii + cuNxNy + 1], u1[ii - cuNxNy - 1]);
    tmp4 = add(u1[ii + cuNxNy - cuNx], u1[ii - cuNxNy + cuNx]);
    tmp3 = add(tmp3, tmp4);
    tmp2 = add(u1[ii + cuNx - 1], u1[ii - cuNx + 1]);
    tmp1 = add(tmp1, tmp2);
    tmp4 = add(u1[ii + cuNxNy - 1], u1[ii - cuNxNy + 1]);
    tmp3 = add(tmp3, tmp4);
    tmp1 = add(tmp1, tmp3);
    tmp1 = fma(c1, u1[ii], fma(c2, tmp1, -u0[ii]));
    // write final value back to global memory
    if ((GET_BIT(bn_mask[ii >> 3], ii % 8)) == 0) {
      u0[ii] = tmp1;
    }
  }
}

// this folds in half of FCC subgrid so everything is nicely homogenous (no
// braching for stencil)
template<typename Real>
__global__ void KernelFoldFCC(Real* __restrict__ u1) {
  int64_t const cx = blockIdx.x * cuBx2 + threadIdx.x;
  int64_t const cz = blockIdx.y * cuBy2 + threadIdx.y;
  // fold is along middle dimension
  if ((cx < cuNx) && (cz < cuNz)) {
    u1[cz * cuNxNy + (cuNy - 1) * cuNx + cx] = u1[cz * cuNxNy + (cuNy - 2) * cuNx + cx];
  }
}

// rigid boundaries, cartesian, using adj info
template<typename Real>
__global__ void KernelBoundaryRigidCart(
    Real* __restrict__ u0,
    Real const* __restrict__ u1,
    uint16_t const* __restrict__ adj_bn,
    int64_t const* __restrict__ bn_ixyz,
    int8_t const* __restrict__ K_bn
) {
  int64_t const nb = blockIdx.x * cuBb + threadIdx.x;

  auto const c2   = constant<Real>(c2_f32, c2_f64);
  auto const csl2 = constant<Real>(csl2_f32, csl2_f64);

  if (nb < cuNb) {
    int64_t const ii   = bn_ixyz[nb];
    uint16_t const adj = adj_bn[nb];
    Real const K       = K_bn[nb];

    Real const _2 = 2.0;
    Real const b1 = (_2 - csl2 * K);
    Real const b2 = c2;

    Real tmp1 = NAN;
    Real tmp2 = NAN;
    tmp1      = add((Real)GET_BIT(adj, 0) * u1[ii + cuNxNy], (Real)GET_BIT(adj, 1) * u1[ii - cuNxNy]);
    tmp2      = add((Real)GET_BIT(adj, 2) * u1[ii + cuNx], (Real)GET_BIT(adj, 3) * u1[ii - cuNx]);
    tmp1      = add(tmp1, tmp2);
    tmp2      = add((Real)GET_BIT(adj, 4) * u1[ii + 1], (Real)GET_BIT(adj, 5) * u1[ii - 1]);
    tmp1      = add(tmp1, tmp2);
    tmp1      = fma(b1, u1[ii], fma(b2, tmp1, -u0[ii]));

    // u0[ii] = partial; //write back to global memory
    u0[ii] = tmp1; // write back to global memory
  }
}

// rigid boundaries, FCC, using adj info
template<typename Real>
__global__ void KernelBoundaryRigidFCC(
    Real* __restrict__ u0,
    Real const* __restrict__ u1,
    uint16_t const* __restrict__ adj_bn,
    int64_t const* __restrict__ bn_ixyz,
    int8_t const* __restrict__ K_bn
) {
  int64_t const nb = blockIdx.x * cuBb + threadIdx.x;

  auto const c2   = constant<Real>(c2_f32, c2_f64);
  auto const csl2 = constant<Real>(csl2_f32, csl2_f64);

  if (nb < cuNb) {
    int64_t const ii   = bn_ixyz[nb];
    uint16_t const adj = adj_bn[nb];
    Real const K       = K_bn[nb];

    Real const _2 = 2.0;
    Real const b1 = (_2 - csl2 * K);
    Real const b2 = c2;

    Real tmp1 = NAN;
    Real tmp2 = NAN;
    Real tmp3 = NAN;
    Real tmp4 = NAN;
    tmp1      = add((Real)GET_BIT(adj, 0) * u1[ii + cuNxNy + cuNx], (Real)GET_BIT(adj, 1) * u1[ii - cuNxNy - cuNx]);
    tmp2      = add((Real)GET_BIT(adj, 2) * u1[ii + cuNx + 1], (Real)GET_BIT(adj, 3) * u1[ii - cuNx - 1]);
    tmp1      = add(tmp1, tmp2);
    tmp3      = add((Real)GET_BIT(adj, 4) * u1[ii + cuNxNy + 1], (Real)GET_BIT(adj, 5) * u1[ii - cuNxNy - 1]);
    tmp4      = add((Real)GET_BIT(adj, 6) * u1[ii + cuNxNy - cuNx], (Real)GET_BIT(adj, 7) * u1[ii - cuNxNy + cuNx]);
    tmp3      = add(tmp3, tmp4);
    tmp2      = add((Real)GET_BIT(adj, 8) * u1[ii + cuNx - 1], (Real)GET_BIT(adj, 9) * u1[ii - cuNx + 1]);
    tmp1      = add(tmp1, tmp2);
    tmp4      = add((Real)GET_BIT(adj, 10) * u1[ii + cuNxNy - 1], (Real)GET_BIT(adj, 11) * u1[ii - cuNxNy + 1]);
    tmp3      = add(tmp3, tmp4);
    tmp1      = add(tmp1, tmp3);
    tmp1      = fma(b1, u1[ii], fma(b2, tmp1, -u0[ii]));

    u0[ii] = tmp1; // write back to global memory
  }
}

// ABC loss at boundaries of simulation grid
template<typename Real>
__global__ void KernelBoundaryABC(
    Real* __restrict__ u0,
    Real const* __restrict__ u2ba,
    int8_t const* __restrict__ Q_bna,
    int64_t const* __restrict__ bna_ixyz
) {
  int64_t const nb = blockIdx.x * cuBb + threadIdx.x;
  auto const cl    = constant<Real>(cl_f32, cl_f64);

  if (nb < cuNba) {
    Real const _1    = 1.0;
    Real const lQ    = cl * Q_bna[nb];
    int64_t const ib = bna_ixyz[nb];
    Real partial     = u0[ib];
    partial          = (partial + lQ * u2ba[nb]) / (_1 + lQ);
    u0[ib]           = partial;
  }
}

// Part of freq-dep boundary update
template<typename Real>
__global__ void KernelBoundaryFD(
    Real* __restrict__ u0b,
    Real const* u2b,
    Real* __restrict__ vh1,
    Real* __restrict__ gh1,
    Real const* ssaf_bnl,
    int8_t const* mat_bnl,
    Real const* __restrict__ mat_beta,
    MatQuad<Real> const* __restrict__ mat_quads
) {
  int64_t const nb = blockIdx.x * cuBb + threadIdx.x;

  auto const clo2 = constant<Real>(clo2_f32, clo2_f64);

  if (nb < cuNbl) {
    Real const _1     = 1.0;
    Real const _2     = 2.0;
    int32_t const k   = mat_bnl[nb];
    Real const ssaf   = ssaf_bnl[nb];
    Real const lo2Kbg = clo2 * ssaf * mat_beta[k];
    Real const fac    = _2 * clo2 * ssaf / (_1 + lo2Kbg);

    Real u0bint       = u0b[nb];
    Real const u2bint = u2b[nb];

    u0bint = (u0bint + lo2Kbg * u2bint) / (_1 + lo2Kbg);

    Real vh1int[MMb]; // size has to be constant at compile time
    Real gh1int[MMb];
    for (int8_t m = 0; m < cuMb[k]; m++) { // faster on average than MMb
      int64_t const nbm       = m * cuNbl + nb;
      int32_t const mbk       = k * MMb + m;
      MatQuad<Real> const& tm = mat_quads[mbk];
      vh1int[m]               = vh1[nbm];
      gh1int[m]               = gh1[nbm];
      u0bint -= fac * (_2 * tm.bDh * vh1int[m] - tm.bFh * gh1int[m]);
    }

    Real const du = u0bint - u2bint;

    // NOLINTBEGIN(clang-analyzer-core.UndefinedBinaryOperatorResult)
    for (int8_t m = 0; m < cuMb[k]; m++) { // faster on average than MMb
      int64_t const nbm       = m * cuNbl + nb;
      int32_t const mbk       = k * MMb + m;
      MatQuad<Real> const& tm = mat_quads[mbk];
      Real const vh0m         = tm.b * du + tm.bd * vh1int[m] - _2 * tm.bFh * gh1int[m];
      gh1[nbm]                = gh1int[m] + (vh0m + vh1int[m]) / _2;
      vh1[nbm]                = vh0m;
    }
    // NOLINTEND(clang-analyzer-core.UndefinedBinaryOperatorResult)
    u0b[nb] = u0bint;
  }
}

// add source input (one at a time for simplicity)
template<typename Real>
__global__ void AddIn(Real* u0, Real sample) {
  u0[0] += sample;
}

// dst-src copy from buffer to grid
template<typename Real>
__global__ void CopyToGridKernel(Real* u, Real const* buffer, int64_t const* locs, int64_t N) {
  int64_t const i = blockIdx.x * cuBrw + threadIdx.x;
  if (i < N) {
    u[locs[i]] = buffer[i];
  }
}

// dst-src copy to buffer from  grid (not needed, but to make more explicit)
template<typename Real>
__global__ void CopyFromGridKernel(Real* buffer, Real const* u, int64_t const* locs, int64_t N) {
  int64_t const i = blockIdx.x * cuBrw + threadIdx.x;
  if (i < N) {
    buffer[i] = u[locs[i]];
  }
}

// flip halos for ABCs
template<typename Real>
__global__ void FlipHaloXY_Zbeg(Real* __restrict__ u1) {
  int64_t const cx = blockIdx.x * cuBx2 + threadIdx.x;
  int64_t const cy = blockIdx.y * cuBy2 + threadIdx.y;
  if ((cx < cuNx) && (cy < cuNy)) {
    int64_t ii = 0;
    ii         = 0 * cuNxNy + cy * cuNx + cx;
    u1[ii]     = u1[ii + 2 * cuNxNy];
  }
}

template<typename Real>
__global__ void FlipHaloXY_Zend(Real* __restrict__ u1) {
  int64_t const cx = blockIdx.x * cuBx2 + threadIdx.x;
  int64_t const cy = blockIdx.y * cuBy2 + threadIdx.y;
  if ((cx < cuNx) && (cy < cuNy)) {
    int64_t ii = 0;
    ii         = (cuNz - 1) * cuNxNy + cy * cuNx + cx;
    u1[ii]     = u1[ii - 2 * cuNxNy];
  }
}

template<typename Real>
__global__ void FlipHaloXZ_Ybeg(Real* __restrict__ u1) {
  int64_t const cx = blockIdx.x * cuBx2 + threadIdx.x;
  int64_t const cz = blockIdx.y * cuBy2 + threadIdx.y;
  if ((cx < cuNx) && (cz < cuNz)) {
    int64_t ii = 0;
    ii         = cz * cuNxNy + 0 * cuNx + cx;
    u1[ii]     = u1[ii + 2 * cuNx];
  }
}

template<typename Real>
__global__ void FlipHaloXZ_Yend(Real* __restrict__ u1) {
  int64_t const cx = blockIdx.x * cuBx2 + threadIdx.x;
  int64_t const cz = blockIdx.y * cuBy2 + threadIdx.y;
  if ((cx < cuNx) && (cz < cuNz)) {
    int64_t ii = 0;
    ii         = cz * cuNxNy + (cuNy - 1) * cuNx + cx;
    u1[ii]     = u1[ii - 2 * cuNx];
  }
}

template<typename Real>
__global__ void FlipHaloYZ_Xbeg(Real* __restrict__ u1) {
  int64_t const cy = blockIdx.x * cuBx2 + threadIdx.x;
  int64_t const cz = blockIdx.y * cuBy2 + threadIdx.y;
  if ((cy < cuNy) && (cz < cuNz)) {
    int64_t ii = 0;
    ii         = cz * cuNxNy + cy * cuNx + 0;
    u1[ii]     = u1[ii + 2];
  }
}

template<typename Real>
__global__ void FlipHaloYZ_Xend(Real* __restrict__ u1) {
  int64_t const cy = blockIdx.x * cuBx2 + threadIdx.x;
  int64_t const cz = blockIdx.y * cuBy2 + threadIdx.y;
  if ((cy < cuNy) && (cz < cuNz)) {
    int64_t ii = 0;
    ii         = cz * cuNxNy + cy * cuNx + (cuNx - 1);
    u1[ii]     = u1[ii - 2];
  }
}

// print some gpu details
auto print_gpu_details(int i) -> uint64_t {
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, i);
  std::printf("\nDevice Number: %d [%s]\n", i, prop.name);
  std::printf("  Compute: %d.%d\n", prop.major, prop.minor);
  std::printf("  Peak Memory Bandwidth: %.3f GB/s\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8.0) / 1.0e6);
  std::printf(
      "  Total global memory: [ %.3f GB | %.3f GiB | %lu MiB ]\n",
      (double)prop.totalGlobalMem / 1e9,
      (double)prop.totalGlobalMem / (1024 * 1024 * 1024),
      prop.totalGlobalMem >> 20
  );
  std::printf(
      "  L2 cache memory: [ %.3f MB | %.3f MiB ]\n",
      static_cast<double>(prop.l2CacheSize) / 1e6,
      static_cast<double>(prop.l2CacheSize) / (1024 * 1024)
  );
  std::printf("  Max grid size:  [%d, %d, %d]\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  std::printf("  Max block size: [%d, %d, %d]\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
  std::printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
  std::printf("  Shared memory per block: %zu\n", prop.sharedMemPerBlock);
  std::printf("  Registers per block: %d\n", prop.regsPerBlock);
  std::printf("  Concurrent Kernels: %d\n", prop.concurrentKernels);
  std::printf("  Async Engine: %d\n", prop.asyncEngineCount);
  std::printf("  Multi processors: %d\n", prop.multiProcessorCount);
  std::printf("  Warp size: %d\n", prop.warpSize);
  std::printf("\n");
  return prop.totalGlobalMem;
}

// input indices need to be sorted for multi-device allocation
void checkSorted(Simulation3D const& sim) {
  int64_t const* bn_ixyz  = sim.bn_ixyz.data();
  int64_t const* bnl_ixyz = sim.bnl_ixyz.data();
  int64_t const* bna_ixyz = sim.bna_ixyz.data();
  int64_t const* in_ixyz  = sim.in_ixyz.data();
  int64_t const* out_ixyz = sim.out_ixyz.data();
  int64_t const Nb        = sim.Nb;
  int64_t const Nbl       = sim.Nbl;
  int64_t const Nba       = sim.Nba;
  int64_t const Ns        = sim.Ns;
  int64_t const Nr        = sim.Nr;
  for (int64_t i = 1; i < Nb; i++) {
    PFFDTD_ASSERT(bn_ixyz[i] > bn_ixyz[i - 1]); // check save_gpu_folder
  }
  for (int64_t i = 1; i < Nbl; i++) {
    PFFDTD_ASSERT(bnl_ixyz[i] > bnl_ixyz[i - 1]);
  }
  for (int64_t i = 1; i < Nba; i++) {
    PFFDTD_ASSERT(bna_ixyz[i] > bna_ixyz[i - 1]);
  }
  for (int64_t i = 1; i < Ns; i++) {
    PFFDTD_ASSERT(in_ixyz[i] > in_ixyz[i - 1]);
  }
  for (int64_t i = 1; i < Nr; i++) {
    PFFDTD_ASSERT(out_ixyz[i] >= out_ixyz[i - 1]); // possible to have duplicates
  }
}

// counts for splitting data across GPUs
template<typename Real>
void splitData(Simulation3D const& sim, std::span<HostData<Real>> ghds) {
  auto const Nx    = sim.Nx;
  auto const Ny    = sim.Ny;
  auto const Nz    = sim.Nz;
  auto const ngpus = static_cast<int>(ghds.size());

  // initialise
  for (int gid = 0; gid < ngpus; gid++) {
    auto& hd = ghds[gid];
    hd.Nx    = 0;
    hd.Nb    = 0;
    hd.Nbl   = 0;
    hd.Nba   = 0;
    hd.Ns    = 0;
    hd.Nr    = 0;
  }

  // split Nx layers (Nz contiguous)
  PFFDTD_ASSERT(ngpus > 0);
  int64_t const Nxm = Nx / ngpus;
  int64_t const Nxl = Nx % ngpus;

  for (int gid = 0; gid < ngpus; gid++) {
    auto& hd = ghds[gid];
    hd.Nx    = Nxm;
  }
  for (int gid = 0; gid < Nxl; gid++) {
    auto& hd = ghds[gid];
    hd.Nx += 1;
  }
  int64_t Nx_check = 0;
  for (int gid = 0; gid < ngpus; gid++) {
    auto& hd = ghds[gid];
    std::printf("gid=%d, Nx[%d]=%ld, Nx=%ld\n", gid, gid, hd.Nx, Nx);
    Nx_check += hd.Nx;
  }
  PFFDTD_ASSERT(Nx_check == Nx);

  // now count Nr,Ns,Nb for each gpu
  auto Nxcc = std::vector<int64_t>(static_cast<size_t>(ngpus));
  Nxcc[0]   = ghds[0].Nx;
  std::printf("Nxcc[%d]=%ld\n", 0, Nxcc[0]);
  for (int gid = 1; gid < ngpus; gid++) {
    auto& hd  = ghds[gid];
    Nxcc[gid] = hd.Nx + Nxcc[gid - 1];
    std::printf("Nxcc[%d]=%ld\n", gid, Nxcc[gid]);
  }

  // bn_ixyz - Nb
  int64_t const* bn_ixyz = sim.bn_ixyz.data();
  int64_t const Nb       = sim.Nb;
  {
    int gid = 0;
    for (int64_t i = 0; i < Nb; i++) {
      while (bn_ixyz[i] >= Nxcc[gid] * Ny * Nz) {
        gid++;
      }
      (ghds[gid].Nb)++;
    }
  }
  int64_t Nb_check = 0;
  for (int gid = 0; gid < ngpus; gid++) {
    auto& hd = ghds[gid];
    std::printf("gid=%d, Nb[%d]=%ld, Nb=%ld\n", gid, gid, hd.Nb, Nb);
    Nb_check += hd.Nb;
  }
  PFFDTD_ASSERT(Nb_check == Nb);

  // bnl_ixyz - Nbl
  int64_t const* bnl_ixyz = sim.bnl_ixyz.data();
  int64_t const Nbl       = sim.Nbl;
  {
    int gid = 0;
    for (int64_t i = 0; i < Nbl; i++) {
      while (bnl_ixyz[i] >= Nxcc[gid] * Ny * Nz) {
        gid++;
      }
      (ghds[gid].Nbl)++;
    }
  }
  int64_t Nbl_check = 0;
  for (int gid = 0; gid < ngpus; gid++) {
    auto& hd = ghds[gid];
    std::printf("gid=%d, Nbl[%d]=%ld, Nbl=%ld\n", gid, gid, hd.Nbl, Nbl);
    Nbl_check += hd.Nbl;
  }
  PFFDTD_ASSERT(Nbl_check == Nbl);

  // bna_ixyz - Nba
  int64_t const* bna_ixyz = sim.bna_ixyz.data();
  int64_t const Nba       = sim.Nba;
  {
    int gid = 0;
    for (int64_t i = 0; i < Nba; i++) {
      while (bna_ixyz[i] >= Nxcc[gid] * Ny * Nz) {
        gid++;
      }
      (ghds[gid].Nba)++;
    }
  }
  int64_t Nba_check = 0;
  for (int gid = 0; gid < ngpus; gid++) {
    auto& hd = ghds[gid];
    std::printf("gid=%d, Nba[%d]=%ld, Nbl=%ld\n", gid, gid, hd.Nba, Nba);
    Nba_check += hd.Nba;
  }
  PFFDTD_ASSERT(Nba_check == Nba);

  // in_ixyz - Ns
  int64_t const* in_ixyz = sim.in_ixyz.data();
  int64_t const Ns       = sim.Ns;
  {
    int gid = 0;
    for (int64_t i = 0; i < Ns; i++) {
      while (in_ixyz[i] >= Nxcc[gid] * Ny * Nz) {
        gid++;
      }
      (ghds[gid].Ns)++;
    }
  }
  int64_t Ns_check = 0;
  for (int gid = 0; gid < ngpus; gid++) {
    auto& hd = ghds[gid];
    std::printf("gid=%d, Ns[%d]=%ld, Ns=%ld\n", gid, gid, hd.Ns, Ns);
    Ns_check += hd.Ns;
  }
  PFFDTD_ASSERT(Ns_check == Ns);

  // out_ixyz - Nr
  int64_t const* out_ixyz = sim.out_ixyz.data();
  int64_t const Nr        = sim.Nr;
  {
    int gid = 0;
    for (int64_t i = 0; i < Nr; i++) {
      while (out_ixyz[i] >= Nxcc[gid] * Ny * Nz) {
        gid++;
      }
      (ghds[gid].Nr)++;
    }
  }
  int64_t Nr_check = 0;
  for (int gid = 0; gid < ngpus; gid++) {
    auto& hd = ghds[gid];
    std::printf("gid=%d, Nr[%d]=%ld, Nr=%ld\n", gid, gid, hd.Nr, Nr);
    Nr_check += hd.Nr;
  }
  PFFDTD_ASSERT(Nr_check == Nr);
}

template<typename Real>
auto run(Simulation3D const& sim) -> void { // NOLINT(readability-function-cognitive-complexity)
  // if you want to test synchronous, env variable for that
  auto const* s = std::getenv("CUDA_LAUNCH_BLOCKING"); // NOLINT(concurrency-mt-unsafe)
  if (s != nullptr) {
    if (s[0] == '1') {
      std::printf("******************SYNCHRONOUS (DEBUG  ONLY!!!)*********************\n");
      std::printf("...continue?\n");
      [[maybe_unused]] auto ch = std::getchar();
    }
  }

  PFFDTD_ASSERT(sim.grid != Grid::FCC); // uses either cartesian or FCC folded grid

  int max_ngpus = 0;
  cudaGetDeviceCount(&max_ngpus); // control outside with CUDA_VISIBLE_DEVICES
  auto const ngpus = max_ngpus;
  PFFDTD_ASSERT(ngpus < (sim.Nx));

  auto ghds = std::vector<HostData<Real>>(static_cast<size_t>(ngpus));
  auto gds  = std::vector<DeviceData<Real>>(static_cast<size_t>(ngpus));

  auto const ssaf_bnl_real  = convertTo<Real>(sim.ssaf_bnl);
  auto const mat_beta_real  = convertTo<Real>(sim.mat_beta);
  auto const mat_quads_real = convertTo<Real>(sim.mat_quads);

  if (ngpus > 1) {
    checkSorted(sim); // needs to be sorted for multi-GPU
  }

  // get local counts for Nx,Nb,Nr,Ns
  splitData<Real>(sim, ghds);

  for (int gid = 0; gid < ngpus; gid++) {
    gds[gid].totalmembytes = print_gpu_details(gid);
  }

  auto lo2 = static_cast<Real>(sim.lo2);
  auto a1  = static_cast<Real>(sim.a1);
  auto a2  = static_cast<Real>(sim.a2);
  auto l   = static_cast<Real>(sim.l);
  auto sl2 = static_cast<Real>(sim.sl2);

  // timing stuff
  auto elapsed               = std::chrono::nanoseconds{0};
  auto elapsedBoundary       = std::chrono::nanoseconds{0};
  auto elapsedSample         = std::chrono::nanoseconds{0};
  auto elapsedSampleBoundary = std::chrono::nanoseconds{0};
  auto elapsedAir            = std::chrono::nanoseconds{0};
  auto elapsedSampleAir      = std::chrono::nanoseconds{0};

  std::printf("a1 = %.16g\n", a1);
  std::printf("a2 = %.16g\n", a2);

  // start moving data to GPUs
  for (int gid = 0; gid < ngpus; gid++) {
    auto& h = ghds[gid];
    std::printf("GPU %d -- ", gid);
    std::printf("Nx=%ld Ns=%ld Nr=%ld Nb=%ld Nbl=%ld Nba=%ld\n", h.Nx, h.Ns, h.Nr, h.Nb, h.Nbl, h.Nba);
  }

  int64_t Ns_read  = 0;
  int64_t Nr_read  = 0;
  int64_t Nb_read  = 0;
  int64_t Nbl_read = 0;
  int64_t Nba_read = 0;
  int64_t Nx_read  = 0;
  int64_t Nx_pos   = 0;
  // uint64_t Nx_pos2=0;

  Real* u_out_buf = nullptr;
  gpuErrchk(cudaMallocHost(&u_out_buf, (size_t)(sim.Nr * sizeof(Real))));
  memset(u_out_buf, 0, (size_t)(sim.Nr * sizeof(Real))); // set floats to zero

  int64_t Nzy = (sim.Nz) * (sim.Ny); // area-slice

  // here we recalculate indices to move to devices
  for (int gid = 0; gid < ngpus; gid++) {
    gpuErrchk(cudaSetDevice(gid));

    DeviceData<Real>& gpu = gds[gid];
    HostData<Real>& host  = ghds[gid];
    std::printf("---------\n");
    std::printf("GPU %d\n", gid);
    std::printf("---------\n");

    std::printf("Nx to read = %ld\n", host.Nx);
    std::printf("Nb to read = %ld\n", host.Nb);
    std::printf("Nbl to read = %ld\n", host.Nbl);
    std::printf("Nba to read = %ld\n", host.Nba);
    std::printf("Ns to read = %ld\n", host.Ns);
    std::printf("Nr to read = %ld\n", host.Nr);

    // Nxh (effective Nx with extra halos)
    host.Nxh = host.Nx;
    if (gid > 0) {
      (host.Nxh)++; // add bottom halo
    }
    if (gid < ngpus - 1) {
      (host.Nxh)++; // add top halo
    }
    // calculate Npts for this gpu
    host.Npts = Nzy * (host.Nxh);
    // boundary mask
    host.Nbm = cu_divceil(host.Npts, 8);

    std::printf("Nx=%ld Ns=%ld Nr=%ld Nb=%ld, Npts=%ld\n", host.Nx, host.Ns, host.Nr, host.Nb, host.Npts);

    // aliased pointers (to memory already allocated)
    host.in_sigs   = sim.in_sigs.data() + Ns_read * sim.Nt;
    host.ssaf_bnl  = ssaf_bnl_real.data() + Nbl_read;
    host.adj_bn    = sim.adj_bn.data() + Nb_read;
    host.mat_bnl   = sim.mat_bnl.data() + Nbl_read;
    host.K_bn      = sim.K_bn.data() + Nb_read;
    host.Q_bna     = sim.Q_bna.data() + Nba_read;
    host.u_out     = sim.u_out.get() + Nr_read * sim.Nt;
    host.u_out_buf = u_out_buf + Nr_read;

    // recalculate indices, these are associated host versions to copy over to devices
    host.bn_ixyz  = allocate_zeros<int64_t>(host.Nb);
    host.bnl_ixyz = allocate_zeros<int64_t>(host.Nbl);
    host.bna_ixyz = allocate_zeros<int64_t>(host.Nba);
    host.bn_mask  = allocate_zeros<uint8_t>(host.Nbm);
    host.in_ixyz  = allocate_zeros<int64_t>(host.Ns);
    host.out_ixyz = allocate_zeros<int64_t>(host.Nr);

    int64_t const offset = Nzy * Nx_pos;
    for (int64_t nb = 0; nb < (host.Nb); nb++) {
      int64_t const ii = sim.bn_ixyz[nb + Nb_read]; // global index
      int64_t const jj = ii - offset;               // local index
      PFFDTD_ASSERT(jj >= 0);
      PFFDTD_ASSERT(jj < host.Npts);
      host.bn_ixyz[nb] = jj;
      SET_BIT_VAL(host.bn_mask[jj >> 3], jj % 8, GET_BIT(sim.bn_mask[ii >> 3], ii % 8)); // set bit
    }
    for (int64_t nb = 0; nb < (host.Nbl); nb++) {
      int64_t const ii = sim.bnl_ixyz[nb + Nbl_read]; // global index
      int64_t const jj = ii - offset;                 // local index
      PFFDTD_ASSERT(jj >= 0);
      PFFDTD_ASSERT(jj < host.Npts);
      host.bnl_ixyz[nb] = jj;
    }

    for (int64_t nb = 0; nb < (host.Nba); nb++) {
      int64_t const ii = sim.bna_ixyz[nb + Nba_read]; // global index
      int64_t const jj = ii - offset;                 // local index
      PFFDTD_ASSERT(jj >= 0);
      PFFDTD_ASSERT(jj < host.Npts);
      host.bna_ixyz[nb] = jj;
    }

    for (int64_t ns = 0; ns < (host.Ns); ns++) {
      int64_t const ii = sim.in_ixyz[ns + Ns_read];
      int64_t const jj = ii - offset;
      PFFDTD_ASSERT(jj >= 0);
      PFFDTD_ASSERT(jj < host.Npts);
      host.in_ixyz[ns] = jj;
    }
    for (int64_t nr = 0; nr < (host.Nr); nr++) {
      int64_t const ii = sim.out_ixyz[nr + Nr_read];
      int64_t const jj = ii - offset;
      PFFDTD_ASSERT(jj >= 0);
      PFFDTD_ASSERT(jj < host.Npts);
      host.out_ixyz[nr] = jj;
    }

    gpuErrchk(cudaMalloc(&(gpu.u0), (size_t)((host.Npts) * sizeof(Real))));
    gpuErrchk(cudaMemset(gpu.u0, 0, (size_t)((host.Npts) * sizeof(Real))));

    gpuErrchk(cudaMalloc(&(gpu.u1), (size_t)((host.Npts) * sizeof(Real))));
    gpuErrchk(cudaMemset(gpu.u1, 0, (size_t)((host.Npts) * sizeof(Real))));

    gpuErrchk(cudaMalloc(&(gpu.K_bn), (size_t)(host.Nb * sizeof(int8_t))));
    gpuErrchk(cudaMemcpy(gpu.K_bn, host.K_bn, host.Nb * sizeof(int8_t), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&(gpu.ssaf_bnl), (size_t)(host.Nbl * sizeof(Real))));
    gpuErrchk(cudaMemcpy(gpu.ssaf_bnl, host.ssaf_bnl, host.Nbl * sizeof(Real), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&(gpu.u0b), (size_t)(host.Nbl * sizeof(Real))));
    gpuErrchk(cudaMemset(gpu.u0b, 0, (size_t)(host.Nbl * sizeof(Real))));

    gpuErrchk(cudaMalloc(&(gpu.u1b), (size_t)(host.Nbl * sizeof(Real))));
    gpuErrchk(cudaMemset(gpu.u1b, 0, (size_t)(host.Nbl * sizeof(Real))));

    gpuErrchk(cudaMalloc(&(gpu.u2b), (size_t)(host.Nbl * sizeof(Real))));
    gpuErrchk(cudaMemset(gpu.u2b, 0, (size_t)(host.Nbl * sizeof(Real))));

    gpuErrchk(cudaMalloc(&(gpu.u2ba), (size_t)(host.Nba * sizeof(Real))));
    gpuErrchk(cudaMemset(gpu.u2ba, 0, (size_t)(host.Nba * sizeof(Real))));

    gpuErrchk(cudaMalloc(&(gpu.vh1), (size_t)(host.Nbl * MMb * sizeof(Real))));
    gpuErrchk(cudaMemset(gpu.vh1, 0, (size_t)(host.Nbl * MMb * sizeof(Real))));

    gpuErrchk(cudaMalloc(&(gpu.gh1), (size_t)(host.Nbl * MMb * sizeof(Real))));
    gpuErrchk(cudaMemset(gpu.gh1, 0, (size_t)(host.Nbl * MMb * sizeof(Real))));

    gpuErrchk(cudaMalloc(&(gpu.u_out_buf), (size_t)(host.Nr * sizeof(Real))));
    gpuErrchk(cudaMemset(gpu.u_out_buf, 0, (size_t)(host.Nr * sizeof(Real))));

    gpuErrchk(cudaMalloc(&(gpu.bn_ixyz), (size_t)(host.Nb * sizeof(int64_t))));
    gpuErrchk(cudaMemcpy(gpu.bn_ixyz, host.bn_ixyz.get(), (size_t)host.Nb * sizeof(int64_t), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&(gpu.bnl_ixyz), (size_t)(host.Nbl * sizeof(int64_t))));
    gpuErrchk(
        cudaMemcpy(gpu.bnl_ixyz, host.bnl_ixyz.get(), (size_t)host.Nbl * sizeof(int64_t), cudaMemcpyHostToDevice)
    );

    gpuErrchk(cudaMalloc(&(gpu.bna_ixyz), (size_t)(host.Nba * sizeof(int64_t))));
    gpuErrchk(
        cudaMemcpy(gpu.bna_ixyz, host.bna_ixyz.get(), (size_t)host.Nba * sizeof(int64_t), cudaMemcpyHostToDevice)
    );

    gpuErrchk(cudaMalloc(&(gpu.Q_bna), (size_t)(host.Nba * sizeof(int8_t))));
    gpuErrchk(cudaMemcpy(gpu.Q_bna, host.Q_bna, host.Nba * sizeof(int8_t), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&(gpu.out_ixyz), (size_t)(host.Nr * sizeof(int64_t))));
    gpuErrchk(cudaMemcpy(gpu.out_ixyz, host.out_ixyz.get(), (size_t)host.Nr * sizeof(int64_t), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&(gpu.adj_bn), (size_t)(host.Nb * sizeof(uint16_t))));
    gpuErrchk(cudaMemcpy(gpu.adj_bn, host.adj_bn, (size_t)host.Nb * sizeof(uint16_t), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&(gpu.mat_bnl), (size_t)(host.Nbl * sizeof(int8_t))));
    gpuErrchk(cudaMemcpy(gpu.mat_bnl, host.mat_bnl, (size_t)host.Nbl * sizeof(int8_t), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&(gpu.mat_beta), (size_t)sim.Nm * sizeof(Real)));
    gpuErrchk(cudaMemcpy(gpu.mat_beta, mat_beta_real.data(), (size_t)sim.Nm * sizeof(Real), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMalloc(&(gpu.mat_quads), (size_t)sim.Nm * MMb * sizeof(MatQuad<Real>)));
    gpuErrchk(cudaMemcpy(
        gpu.mat_quads,
        mat_quads_real.data(),
        (size_t)sim.Nm * MMb * sizeof(MatQuad<Real>),
        cudaMemcpyHostToDevice
    ));

    gpuErrchk(cudaMalloc(&(gpu.bn_mask), (size_t)(host.Nbm * sizeof(uint8_t))));
    gpuErrchk(cudaMemcpy(gpu.bn_mask, host.bn_mask.get(), (size_t)host.Nbm * sizeof(uint8_t), cudaMemcpyHostToDevice));

    Ns_read += host.Ns;
    Nr_read += host.Nr;
    Nb_read += host.Nb;
    Nbl_read += host.Nbl;
    Nba_read += host.Nba;
    Nx_read += host.Nx;
    Nx_pos = Nx_read - 1; // back up one at subsequent stage

    std::printf("Nx_read = %ld\n", Nx_read);
    std::printf("Nb_read = %ld\n", Nb_read);
    std::printf("Nbl_read = %ld\n", Nbl_read);
    std::printf("Ns_read = %ld\n", Ns_read);
    std::printf("Nr_read = %ld\n", Nr_read);

    std::printf("Global memory allocation done\n");
    std::printf("\n");

    // swapping x and z here (CUDA has first dim contiguous)
    // same for all devices
    gpuErrchk(cudaMemcpyToSymbol(cuNx, &(sim.Nz), sizeof(int64_t)));
    gpuErrchk(cudaMemcpyToSymbol(cuNy, &(sim.Ny), sizeof(int64_t)));
    gpuErrchk(cudaMemcpyToSymbol(cuNz, &(host.Nxh), sizeof(int64_t)));
    gpuErrchk(cudaMemcpyToSymbol(cuNb, &(host.Nb), sizeof(int64_t)));
    gpuErrchk(cudaMemcpyToSymbol(cuNbl, &(host.Nbl), sizeof(int64_t)));
    gpuErrchk(cudaMemcpyToSymbol(cuNba, &(host.Nba), sizeof(int64_t)));
    gpuErrchk(cudaMemcpyToSymbol(cuMb, sim.Mb.data(), sim.Nm * sizeof(int8_t)));
    gpuErrchk(cudaMemcpyToSymbol(cuNxNy, &Nzy, sizeof(int64_t)));

    if constexpr (std::is_same_v<Real, float>) {
      gpuErrchk(cudaMemcpyToSymbol(c1_f32, &a1, sizeof(float)));
      gpuErrchk(cudaMemcpyToSymbol(c2_f32, &a2, sizeof(float)));
      gpuErrchk(cudaMemcpyToSymbol(cl_f32, &l, sizeof(float)));
      gpuErrchk(cudaMemcpyToSymbol(csl2_f32, &sl2, sizeof(float)));
      gpuErrchk(cudaMemcpyToSymbol(clo2_f32, &lo2, sizeof(float)));
    } else if constexpr (std::is_same_v<Real, double>) {
      gpuErrchk(cudaMemcpyToSymbol(c1_f64, &a1, sizeof(double)));
      gpuErrchk(cudaMemcpyToSymbol(c2_f64, &a2, sizeof(double)));
      gpuErrchk(cudaMemcpyToSymbol(cl_f64, &l, sizeof(double)));
      gpuErrchk(cudaMemcpyToSymbol(csl2_f64, &sl2, sizeof(double)));
      gpuErrchk(cudaMemcpyToSymbol(clo2_f64, &lo2, sizeof(double)));
    } else {
      static_assert(always_false<Real>);
    }

    std::printf("Constant memory loaded\n");
    std::printf("\n");

    // threads grids and blocks (swap x and z)
    int64_t const cuGx  = cu_divceil(sim.Nz - 2, cuBx);
    int64_t const cuGy  = cu_divceil(sim.Ny - 2, cuBy);
    int64_t const cuGz  = cu_divceil(host.Nxh - 2, cuBz);
    int64_t const cuGr  = cu_divceil(host.Nr, cuBrw);
    int64_t const cuGb  = cu_divceil(host.Nb, cuBb);
    int64_t const cuGbl = cu_divceil(host.Nbl, cuBb);
    int64_t const cuGba = cu_divceil(host.Nba, cuBb);

    int64_t const cuGx2 = cu_divceil(sim.Nz, cuBx2);   // full face
    int64_t const cuGz2 = cu_divceil(host.Nxh, cuBy2); // full face

    PFFDTD_ASSERT(cuGx >= 1);
    PFFDTD_ASSERT(cuGy >= 1);
    PFFDTD_ASSERT(cuGz >= 1);
    PFFDTD_ASSERT(cuGr >= 1);
    PFFDTD_ASSERT(cuGb >= 1);
    PFFDTD_ASSERT(cuGbl >= 1);
    PFFDTD_ASSERT(cuGba >= 1);

    gpu.block_dim_air     = dim3(cuBx, cuBy, cuBz);
    gpu.block_dim_readout = dim3(cuBrw, 1, 1);
    gpu.block_dim_bn      = dim3(cuBb, 1, 1);

    gpu.grid_dim_air     = dim3(cuGx, cuGy, cuGz);
    gpu.grid_dim_readout = dim3(cuGr, 1, 1);
    gpu.grid_dim_bn      = dim3(cuGb, 1, 1);
    gpu.grid_dim_bnl     = dim3(cuGbl, 1, 1);
    gpu.grid_dim_bna     = dim3(cuGba, 1, 1);

    gpu.block_dim_halo_xy = dim3(cuBx2, cuBy2, 1);
    gpu.block_dim_halo_yz = dim3(cuBx2, cuBy2, 1);
    gpu.block_dim_halo_xz = dim3(cuBx2, cuBy2, 1);
    gpu.grid_dim_halo_xy  = dim3(cu_divceil(sim.Nz, cuBx2), cu_divceil(sim.Ny, cuBy2), 1);
    gpu.grid_dim_halo_yz  = dim3(cu_divceil(sim.Ny, cuBx2), cu_divceil(host.Nxh, cuBy2), 1);
    gpu.grid_dim_halo_xz  = dim3(cu_divceil(sim.Nz, cuBx2), cu_divceil(host.Nxh, cuBy2), 1);

    gpu.block_dim_fold = dim3(cuBx2, cuBy2, 1);
    gpu.grid_dim_fold  = dim3(cuGx2, cuGz2, 1);

    // create streams
    gpuErrchk(cudaStreamCreate(&(gpu.cuStream_air)));
    gpuErrchk(cudaStreamCreate(&(gpu.cuStream_bn))); // no priority

    // cuda events
    gpuErrchk(cudaEventCreate(&(gpu.cuEv_air_start)));
    gpuErrchk(cudaEventCreate(&(gpu.cuEv_air_end)));
    gpuErrchk(cudaEventCreate(&(gpu.cuEv_bn_roundtrip_start)));
    gpuErrchk(cudaEventCreate(&(gpu.cuEv_bn_roundtrip_end)));
    gpuErrchk(cudaEventCreate(&(gpu.cuEv_readout_end)));
  }
  PFFDTD_ASSERT(Nb_read == sim.Nb);
  PFFDTD_ASSERT(Nbl_read == sim.Nbl);
  PFFDTD_ASSERT(Nba_read == sim.Nba);
  PFFDTD_ASSERT(Nr_read == sim.Nr);
  PFFDTD_ASSERT(Ns_read == sim.Ns);
  PFFDTD_ASSERT(Nx_read == sim.Nx);

  // these will be on first GPU only
  cudaEvent_t cuEv_main_start        = nullptr;
  cudaEvent_t cuEv_main_end          = nullptr;
  cudaEvent_t cuEv_main_sample_start = nullptr;
  cudaEvent_t cuEv_main_sample_end   = nullptr;
  gpuErrchk(cudaSetDevice(0));
  gpuErrchk(cudaEventCreate(&cuEv_main_start));
  gpuErrchk(cudaEventCreate(&cuEv_main_end));
  gpuErrchk(cudaEventCreate(&cuEv_main_sample_start));
  gpuErrchk(cudaEventCreate(&cuEv_main_sample_end));

  for (int64_t n = 0; n < sim.Nt; n++) {    // loop over time-steps
    for (int gid = 0; gid < ngpus; gid++) { // loop over GPUs (one thread launches all kernels)
      gpuErrchk(cudaSetDevice(gid));
      DeviceData<Real> const& gpu = gds[gid];  // get struct of gpu pointers
      HostData<Real> const& host  = ghds[gid]; // get struct of host points (corresponding to gpu)

      // start first timer
      if (gid == 0) {
        if (n == 0) {
          // not sure if to put on stream, check slides again
          gpuErrchk(cudaEventRecord(cuEv_main_start, nullptr));
        }
        gpuErrchk(cudaEventRecord(cuEv_main_sample_start, nullptr));
      }
      // boundary updates (using intermediate buffer)
      gpuErrchk(cudaEventRecord(gpu.cuEv_bn_roundtrip_start, gpu.cuStream_bn));

      // boundary updates
      if (sim.grid == Grid::CART) {
        KernelBoundaryRigidCart<<<gpu.grid_dim_bn, gpu.block_dim_bn, 0, gpu.cuStream_bn>>>(
            gpu.u0,
            gpu.u1,
            gpu.adj_bn,
            gpu.bn_ixyz,
            gpu.K_bn
        );
      } else {
        KernelFoldFCC<<<gpu.grid_dim_fold, gpu.block_dim_fold, 0, gpu.cuStream_bn>>>(gpu.u1);
        KernelBoundaryRigidFCC<<<gpu.grid_dim_bn, gpu.block_dim_bn, 0, gpu.cuStream_bn>>>(
            gpu.u0,
            gpu.u1,
            gpu.adj_bn,
            gpu.bn_ixyz,
            gpu.K_bn
        );
      }
      // using buffer to then update FD boundaries
      CopyFromGridKernel<<<gpu.grid_dim_bnl, gpu.block_dim_bn, 0, gpu.cuStream_bn>>>(
          gpu.u0b,
          gpu.u0,
          gpu.bnl_ixyz,
          host.Nbl
      );
      // possible this could be moved to host
      KernelBoundaryFD<<<gpu.grid_dim_bnl, gpu.block_dim_bn, 0, gpu.cuStream_bn>>>(
          gpu.u0b,
          gpu.u2b,
          gpu.vh1,
          gpu.gh1,
          gpu.ssaf_bnl,
          gpu.mat_bnl,
          gpu.mat_beta,
          gpu.mat_quads
      );
      // copy to back to grid
      CopyToGridKernel<<<gpu.grid_dim_bnl, gpu.block_dim_bn, 0, gpu.cuStream_bn>>>(
          gpu.u0,
          gpu.u0b,
          gpu.bnl_ixyz,
          host.Nbl
      );
      gpuErrchk(cudaEventRecord(gpu.cuEv_bn_roundtrip_end, gpu.cuStream_bn));

      // air updates (including source
      gpuErrchk(cudaStreamWaitEvent(gpu.cuStream_air, gpu.cuEv_bn_roundtrip_end,
                                    0)); // might as well wait
      // run air kernel (with mask)
      gpuErrchk(cudaEventRecord(gpu.cuEv_air_start, gpu.cuStream_air));

      // for absorbing boundaries at boundaries of grid
      CopyFromGridKernel<<<gpu.grid_dim_bna, gpu.block_dim_bn, 0, gpu.cuStream_air>>>(
          gpu.u2ba,
          gpu.u0,
          gpu.bna_ixyz,
          host.Nba
      );
      if (gid == 0) {
        FlipHaloXY_Zbeg<<<gpu.grid_dim_halo_xy, gpu.block_dim_halo_xy, 0, gpu.cuStream_air>>>(gpu.u1);
      }
      if (gid == ngpus - 1) {
        FlipHaloXY_Zend<<<gpu.grid_dim_halo_xy, gpu.block_dim_halo_xy, 0, gpu.cuStream_air>>>(gpu.u1);
      }
      FlipHaloXZ_Ybeg<<<gpu.grid_dim_halo_xz, gpu.block_dim_halo_xz, 0, gpu.cuStream_air>>>(gpu.u1);
      if (sim.grid == Grid::CART) {
        FlipHaloXZ_Yend<<<gpu.grid_dim_halo_xz, gpu.block_dim_halo_xz, 0, gpu.cuStream_air>>>(gpu.u1);
      }
      FlipHaloYZ_Xbeg<<<gpu.grid_dim_halo_yz, gpu.block_dim_halo_yz, 0, gpu.cuStream_air>>>(gpu.u1);
      FlipHaloYZ_Xend<<<gpu.grid_dim_halo_yz, gpu.block_dim_halo_yz, 0, gpu.cuStream_air>>>(gpu.u1);

      // injecting source first, negating sample to add it in first (NB source
      // on different stream than bn)
      for (int64_t ns = 0; ns < host.Ns; ns++) {
        AddIn<<<1, 1, 0, gpu.cuStream_air>>>(gpu.u0 + host.in_ixyz[ns], (Real)(-(host.in_sigs[ns * sim.Nt + n])));
      }
      // now air updates (not conflicting with bn updates because of bn_mask)
      if (sim.grid == Grid::CART) {
        KernelAirCart<<<gpu.grid_dim_air, gpu.block_dim_air, 0, gpu.cuStream_air>>>(gpu.u0, gpu.u1, gpu.bn_mask);
      } else {
        KernelAirFCC<<<gpu.grid_dim_air, gpu.block_dim_air, 0, gpu.cuStream_air>>>(gpu.u0, gpu.u1, gpu.bn_mask);
      }
      // boundary ABC loss
      KernelBoundaryABC<<<gpu.grid_dim_bna, gpu.block_dim_bn, 0, gpu.cuStream_air>>>(
          gpu.u0,
          gpu.u2ba,
          gpu.Q_bna,
          gpu.bna_ixyz
      );
      gpuErrchk(cudaEventRecord(gpu.cuEv_air_end, gpu.cuStream_air)); // for timing

      // readouts
      CopyFromGridKernel<<<gpu.grid_dim_readout, gpu.block_dim_readout, 0, gpu.cuStream_bn>>>(
          gpu.u_out_buf,
          gpu.u1,
          gpu.out_ixyz,
          host.Nr
      );
      // then async memory copy of outputs (not really async because on same
      // stream as CopyFromGridKernel)
      gpuErrchk(cudaMemcpyAsync(
          host.u_out_buf,
          gpu.u_out_buf,
          host.Nr * sizeof(Real),
          cudaMemcpyDeviceToHost,
          gpu.cuStream_bn
      ));
      gpuErrchk(cudaEventRecord(gpu.cuEv_readout_end, gpu.cuStream_bn));
    }

    // readouts
    for (int gid = 0; gid < ngpus; gid++) {
      gpuErrchk(cudaSetDevice(gid));
      DeviceData<Real> const& gpu = gds[gid];
      HostData<Real> const& host  = ghds[gid];
      gpuErrchk(cudaEventSynchronize(gpu.cuEv_readout_end));
      // copy grid points off output buffer
      for (int64_t nr = 0; nr < host.Nr; nr++) {
        host.u_out[nr * sim.Nt + n] = (double)(host.u_out_buf[nr]);
      }
    }
    // synchronise streams
    for (int gid = 0; gid < ngpus; gid++) {
      gpuErrchk(cudaSetDevice(gid));
      DeviceData<Real> const& gpu = gds[gid];             // don't really need to set gpu gpu to sync
      gpuErrchk(cudaStreamSynchronize(gpu.cuStream_air)); // interior complete
      gpuErrchk(cudaStreamSynchronize(gpu.cuStream_bn));  // transfer complete
    }
    // dst then src, stream with src gives best performance (CUDA thing)

    // now asynchronous halo swaps, even/odd pairs concurrent
    // these are not async to rest of scheme, just async to other swaps

    // copy forward (even)
    for (int gid = 0; gid < ngpus - 1; gid += 2) {
      gpuErrchk(cudaSetDevice(gid));
      gpuErrchk(cudaMemcpyPeerAsync(
          gds[gid + 1].u0,
          gid + 1,
          gds[gid].u0 + Nzy * (ghds[gid].Nxh - 2),
          gid,
          (size_t)(Nzy * sizeof(Real)),
          gds[gid].cuStream_bn
      ));
    }
    // copy back (odd)
    for (int gid = 1; gid < ngpus; gid += 2) {
      gpuErrchk(cudaSetDevice(gid));
      gpuErrchk(cudaMemcpyPeerAsync(
          gds[gid - 1].u0 + Nzy * (ghds[gid - 1].Nxh - 1),
          gid - 1,
          gds[gid].u0 + Nzy,
          gid,
          (size_t)(Nzy * sizeof(Real)),
          gds[gid].cuStream_bn
      ));
    }
    // copy forward (odd)
    for (int gid = 1; gid < ngpus - 1; gid += 2) {
      gpuErrchk(cudaSetDevice(gid));
      gpuErrchk(cudaMemcpyPeerAsync(
          gds[gid + 1].u0,
          gid + 1,
          gds[gid].u0 + Nzy * (ghds[gid].Nxh - 2),
          gid,
          (size_t)(Nzy * sizeof(Real)),
          gds[gid].cuStream_bn
      ));
    }
    // copy back (even) -- skip zero
    for (int gid = 2; gid < ngpus; gid += 2) {
      gpuErrchk(cudaSetDevice(gid));
      gpuErrchk(cudaMemcpyPeerAsync(
          gds[gid - 1].u0 + Nzy * (ghds[gid - 1].Nxh - 1),
          gid - 1,
          gds[gid].u0 + Nzy,
          gid,
          (size_t)(Nzy * sizeof(Real)),
          gds[gid].cuStream_bn
      ));
    }

    for (int gid = 0; gid < ngpus; gid++) {
      gpuErrchk(cudaSetDevice(gid));
      DeviceData<Real> const& gpu = gds[gid];
      gpuErrchk(cudaStreamSynchronize(gpu.cuStream_bn)); // transfer complete
    }
    for (int gid = 0; gid < ngpus; gid++) {
      DeviceData<Real>& gpu = gds[gid];
      // update pointers
      Real* tmp_ptr = nullptr;
      tmp_ptr       = gpu.u1;
      gpu.u1        = gpu.u0;
      gpu.u0        = tmp_ptr;

      // will use extra vector for this (simpler than extra copy kernel)
      tmp_ptr = gpu.u2b;
      gpu.u2b = gpu.u1b;
      gpu.u1b = gpu.u0b;
      gpu.u0b = tmp_ptr;

      if (gid == 0) {
        gpuErrchk(cudaSetDevice(gid));
        gpuErrchk(cudaEventRecord(cuEv_main_sample_end, nullptr));
      }
    }

    {
      // timing only on gpu0
      gpuErrchk(cudaSetDevice(0));
      gpuErrchk(cudaEventSynchronize(cuEv_main_sample_end)); // not sure this is correct

      auto const& gpu = gds[0];

      elapsed               = elapsedTime(cuEv_main_start, cuEv_main_sample_end);
      elapsedSample         = elapsedTime(cuEv_main_sample_start, cuEv_main_sample_end);
      elapsedSampleAir      = elapsedTime(gpu.cuEv_air_start, gpu.cuEv_air_end);
      elapsedSampleBoundary = elapsedTime(gpu.cuEv_bn_roundtrip_start, gpu.cuEv_bn_roundtrip_end);

      elapsedAir += elapsedSampleAir;
      elapsedBoundary += elapsedSampleBoundary;

      print(
          ProgressReport{
              .n                     = n,
              .Nt                    = sim.Nt,
              .Npts                  = sim.Npts,
              .Nb                    = sim.Nb,
              .elapsed               = elapsed,
              .elapsedSample         = elapsedSample,
              .elapsedAir            = elapsedAir,
              .elapsedSampleAir      = elapsedSampleAir,
              .elapsedBoundary       = elapsedBoundary,
              .elapsedSampleBoundary = elapsedSampleBoundary,
              .numWorkers            = ngpus,
          }
      );
    }
  }
  std::printf("\n");

  for (int gid = 0; gid < ngpus; gid++) {
    gpuErrchk(cudaSetDevice(gid));
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
  }
  {
    // timing (on gpu 0)
    gpuErrchk(cudaSetDevice(0));
    gpuErrchk(cudaEventRecord(cuEv_main_end));
    gpuErrchk(cudaEventSynchronize(cuEv_main_end));

    elapsed = elapsedTime(cuEv_main_start, cuEv_main_end);
  }

  /*------------------------
   * FREE WILLY
  ------------------------*/
  gpuErrchk(cudaSetDevice(0));
  gpuErrchk(cudaEventDestroy(cuEv_main_start));
  gpuErrchk(cudaEventDestroy(cuEv_main_end));
  gpuErrchk(cudaEventDestroy(cuEv_main_sample_start));
  gpuErrchk(cudaEventDestroy(cuEv_main_sample_end));
  for (int gid = 0; gid < ngpus; gid++) {
    gpuErrchk(cudaSetDevice(gid));
    DeviceData<Real> const& gpu = gds[gid];

    // cleanup streams
    gpuErrchk(cudaStreamDestroy(gpu.cuStream_air));
    gpuErrchk(cudaStreamDestroy(gpu.cuStream_bn));

    // cleanup events
    gpuErrchk(cudaEventDestroy(gpu.cuEv_air_start));
    gpuErrchk(cudaEventDestroy(gpu.cuEv_air_end));
    gpuErrchk(cudaEventDestroy(gpu.cuEv_bn_roundtrip_start));
    gpuErrchk(cudaEventDestroy(gpu.cuEv_bn_roundtrip_end));
    gpuErrchk(cudaEventDestroy(gpu.cuEv_readout_end));

    // free memory
    gpuErrchk(cudaFree(gpu.u0));
    gpuErrchk(cudaFree(gpu.u1));
    gpuErrchk(cudaFree(gpu.out_ixyz));
    gpuErrchk(cudaFree(gpu.bn_ixyz));
    gpuErrchk(cudaFree(gpu.bnl_ixyz));
    gpuErrchk(cudaFree(gpu.bna_ixyz));
    gpuErrchk(cudaFree(gpu.Q_bna));
    gpuErrchk(cudaFree(gpu.adj_bn));
    gpuErrchk(cudaFree(gpu.mat_bnl));
    gpuErrchk(cudaFree(gpu.K_bn));
    gpuErrchk(cudaFree(gpu.ssaf_bnl));
    gpuErrchk(cudaFree(gpu.mat_beta));
    gpuErrchk(cudaFree(gpu.mat_quads));
    gpuErrchk(cudaFree(gpu.bn_mask));
    gpuErrchk(cudaFree(gpu.u0b));
    gpuErrchk(cudaFree(gpu.u1b));
    gpuErrchk(cudaFree(gpu.u2b));
    gpuErrchk(cudaFree(gpu.u2ba));
    gpuErrchk(cudaFree(gpu.vh1));
    gpuErrchk(cudaFree(gpu.gh1));
    gpuErrchk(cudaFree(gpu.u_out_buf));
  }
  gpuErrchk(cudaFreeHost(u_out_buf));

  // reset after frees (for some reason it conflicts with cudaFreeHost)
  for (int gid = 0; gid < ngpus; gid++) {
    gpuErrchk(cudaSetDevice(gid));
    gpuErrchk(cudaDeviceReset());
  }

  auto const elapsedSec         = Seconds(elapsed).count();
  auto const elapsedAirSec      = Seconds(elapsedAir).count();
  auto const elapsedBoundarySec = Seconds(elapsedBoundary).count();

  auto const airMvox      = static_cast<double>(sim.Npts * sim.Nt) / 1e6 / elapsedAirSec;
  auto const boundaryMvox = static_cast<double>(sim.Nb * sim.Nt) / 1e6 / elapsedBoundarySec;
  auto const totalMvox    = static_cast<double>(sim.Npts * sim.Nt) / 1e6 / elapsedSec;

  std::printf("Air update: %.6fs, %.2f Mvox/s\n", elapsedAirSec, airMvox);
  std::printf("Boundary loop: %.6fs, %.2f Mvox/s\n", elapsedBoundarySec, boundaryMvox);
  std::printf("Combined (total): %.6fs, %.2f Mvox/s\n", elapsedSec, totalMvox);
}

} // namespace

auto EngineCUDA3D::operator()(Simulation3D const& sim) const -> void {
  switch (sim.precision) {
    case Precision::Float: run<float>(sim); return;
    case Precision::Double: run<double>(sim); return;
    default: throw std::invalid_argument("invalid precision " + std::to_string(static_cast<int>(sim.precision)));
  }
}

} // namespace pffdtd
