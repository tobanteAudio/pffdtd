// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#include "engine_metal.hpp"

#include <metal_stdlib>

namespace pffdtd {
namespace sim2d {

[[kernel]] void airUpdate(
    device float* u0 [[buffer(0)]],
    device float const* u1 [[buffer(1)]],
    device float const* u2 [[buffer(2)]],
    device uint8_t const* in_mask [[buffer(3)]],
    constant Constants2D<float>& constants [[buffer(4)]],
    uint2 id [[thread_position_in_grid]]
) {
  // Metal linearizes threads with x varying fastest, so map it to the contiguous y grid dimension.
  int64_t const x   = id.y + 1;
  int64_t const y   = id.x + 1;
  int64_t const idx = x * constants.Ny + y;

  if (in_mask[idx] == 0) {
    return;
  }

  float const left   = u1[idx - 1];
  float const right  = u1[idx + 1];
  float const bottom = u1[idx - constants.Ny];
  float const top    = u1[idx + constants.Ny];
  float const last   = u2[idx];

  u0[idx] = 0.5 * (left + right + bottom + top) - last;
}

[[kernel]] void boundaryRigid(
    device float* u0 [[buffer(0)]],
    device float const* u1 [[buffer(1)]],
    device float const* u2 [[buffer(2)]],
    device int64_t const* bn_ixy [[buffer(3)]],
    device int64_t const* adj_bn [[buffer(4)]],
    constant Constants2D<float>& constants [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
  int64_t const ib = bn_ixy[id];
  int64_t const K  = adj_bn[id];

  float const last1 = u1[ib];
  float const last2 = u2[ib];

  float const left      = u1[ib - 1];
  float const right     = u1[ib + 1];
  float const bottom    = u1[ib - constants.Ny];
  float const top       = u1[ib + constants.Ny];
  float const neighbors = left + right + top + bottom;

  u0[ib] = (2.0 - 0.5 * K) * last1 + 0.5 * neighbors - last2;
}

[[kernel]] void boundaryLoss(
    device float* u0 [[buffer(0)]],
    device float const* u2 [[buffer(1)]],
    device int64_t const* bn_ixy [[buffer(2)]],
    device int64_t const* adj_bn [[buffer(3)]],
    constant Constants2D<float>& constants [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  int64_t const ib = bn_ixy[id];
  int64_t const K  = adj_bn[id];

  float const current    = u0[ib];
  float const prev       = u2[ib];
  float const K4         = 4 - K;
  float const lossFactor = constants.lossFactor;

  u0[ib] = (current + lossFactor * K4 * prev) / (1.0 + lossFactor * K4);
}

[[kernel]] void addSource(
    device float* u0 [[buffer(0)]],
    device float const* in_sigs [[buffer(1)]],
    device int64_t const* in_ixy [[buffer(2)]],
    constant int64_t& timestep [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
  u0[in_ixy[id]] += in_sigs[timestep];
}

[[kernel]] void readOutput(
    device float* out [[buffer(0)]],
    device float const* u0 [[buffer(1)]],
    device int64_t const* out_ixy [[buffer(2)]],
    constant Constants2D<float>& constants [[buffer(3)]],
    constant int64_t& timestep [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  out[id * constants.Nt + timestep] = u0[out_ixy[id]];
}

} // namespace sim2d

namespace sim3d {

[[kernel]] void airUpdateCart(
    device float* u0 [[buffer(0)]],
    device float const* u1 [[buffer(1)]],
    device uint8_t const* bn_mask [[buffer(2)]],
    constant Constants3D<float>& constants [[buffer(3)]],
    uint3 id [[thread_position_in_grid]]
) {
  float const a1 = constants.a1;
  float const a2 = constants.a2;

  int64_t const Nz   = constants.Nz;
  int64_t const NzNy = constants.NzNy;

  int64_t const x = id.x + 1;
  int64_t const y = id.y + 1;
  int64_t const z = id.z + 1;
  int64_t const i = x * NzNy + y * Nz + z;

  if (get_bit<int>(bn_mask[i / 8], i % 8) != 0) {
    return;
  }

  float partial = a1 * u1[i] - u0[i];
  partial += a2 * u1[i + NzNy];
  partial += a2 * u1[i - NzNy];
  partial += a2 * u1[i + Nz];
  partial += a2 * u1[i - Nz];
  partial += a2 * u1[i + 1];
  partial += a2 * u1[i - 1];
  u0[i] = partial;
}

[[kernel]] void rigidUpdateCart(
    device float* u0 [[buffer(0)]],
    device float const* u1 [[buffer(1)]],
    device int64_t const* bn_ixyz [[buffer(2)]],
    device uint16_t const* adj_bn [[buffer(3)]],
    constant Constants3D<float>& constants [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  int64_t const ii = bn_ixyz[id];
  auto const adj   = adj_bn[id];
  auto const Kint  = metal::popcount(adj);

  float const _2 = 2.0;
  float const K  = Kint;
  float const b2 = constants.a2;
  float const b1 = (_2 - constants.sl2 * K);

  float partial = b1 * u1[ii] - u0[ii];
  partial += b2 * get_bit<float>(adj, 0) * u1[ii + constants.NzNy];
  partial += b2 * get_bit<float>(adj, 1) * u1[ii - constants.NzNy];
  partial += b2 * get_bit<float>(adj, 2) * u1[ii + constants.Nz];
  partial += b2 * get_bit<float>(adj, 3) * u1[ii - constants.Nz];
  partial += b2 * get_bit<float>(adj, 4) * u1[ii + 1];
  partial += b2 * get_bit<float>(adj, 5) * u1[ii - 1];
  u0[ii] = partial;
}

[[kernel]] void boundaryLossFD(
    device float* u0b [[buffer(0)]],
    device float const* u2b [[buffer(1)]],
    device float* vh1 [[buffer(2)]],
    device float* gh1 [[buffer(3)]],
    device float const* ssaf_bnl [[buffer(4)]],
    device int8_t const* mat_bnl [[buffer(5)]],
    device float const* mat_beta [[buffer(6)]],
    device MatQuad<float> const* mat_quads [[buffer(7)]],
    device uint8_t const* Mb [[buffer(8)]],
    constant Constants3D<float>& constants [[buffer(9)]],
    uint id [[thread_position_in_grid]]
) {
  auto nb         = static_cast<int64_t>(id);
  float _1        = 1.0;
  float _2        = 2.0;
  int32_t const k = mat_bnl[nb];

  float lo2Kbg = constants.lo2 * ssaf_bnl[nb] * mat_beta[k];
  float fac    = _2 * constants.lo2 * ssaf_bnl[nb] / (_1 + lo2Kbg);

  float u0bint = u0b[nb];
  float u2bint = u2b[nb];

  u0bint = (u0bint + lo2Kbg * u2bint) / (_1 + lo2Kbg);

  float vh1nb[MMb]{};
  for (int8_t m = 0; m < Mb[k]; m++) {
    int64_t const nbm = nb * MMb + m;
    int32_t const mbk = k * MMb + m;
    auto const tm     = mat_quads[mbk];
    vh1nb[m]          = vh1[nbm];
    u0bint -= fac * (_2 * tm.bDh * vh1nb[m] - tm.bFh * gh1[nbm]);
  }

  float du = u0bint - u2bint;

  for (int8_t m = 0; m < Mb[k]; m++) {
    int64_t const nbm = nb * MMb + m;
    int32_t const mbk = k * MMb + m;
    auto const tm     = mat_quads[mbk];
    float vh0nbm      = tm.b * du + tm.bd * vh1nb[m] - _2 * tm.bFh * gh1[nbm];
    gh1[nbm] += (vh0nbm + vh1nb[m]) / _2;
    vh1[nbm] = vh0nbm;
  }

  u0b[nb] = u0bint;
}

[[kernel]] void boundaryLossABC(
    device float* u0 [[buffer(0)]],
    device float const* u2ba [[buffer(1)]],
    device float const* Q_bna [[buffer(2)]],
    device int64_t const* bna_ixyz [[buffer(3)]],
    constant Constants3D<float>& constants [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  auto const lQ = constants.l * Q_bna[id];
  auto const ib = bna_ixyz[id];
  u0[ib]        = (u0[ib] + lQ * u2ba[id]) / (1.0F + lQ);
}

[[kernel]] void flipHaloXY(
    device float* u1 [[buffer(0)]],
    constant Constants3D<float>& constants [[buffer(1)]],
    uint2 id [[thread_position_in_grid]]
) {
  auto const Nz   = constants.Nz;
  auto const NzNy = constants.NzNy;

  auto const x = static_cast<int64_t>(id.x);
  auto const y = static_cast<int64_t>(id.y);
  auto const i = x * NzNy + y * Nz;

  u1[i + 0]      = u1[i + 2];
  u1[i + Nz - 1] = u1[i + Nz - 3];
}

[[kernel]] void flipHaloXZ(
    device float* u1 [[buffer(0)]],
    constant Constants3D<float>& constants [[buffer(1)]],
    uint2 id [[thread_position_in_grid]]
) {
  auto const Ny   = constants.Ny;
  auto const Nz   = constants.Nz;
  auto const NzNy = constants.NzNy;

  auto const x = static_cast<int64_t>(id.x);
  auto const z = static_cast<int64_t>(id.y);

  u1[x * NzNy + 0 * Nz + z]        = u1[x * NzNy + 2 * Nz + z];
  u1[x * NzNy + (Ny - 1) * Nz + z] = u1[x * NzNy + (Ny - 3) * Nz + z];
}

[[kernel]] void flipHaloYZ(
    device float* u1 [[buffer(0)]],
    constant Constants3D<float>& constants [[buffer(1)]],
    uint2 id [[thread_position_in_grid]]
) {
  auto const Nx   = constants.Nx;
  auto const Nz   = constants.Nz;
  auto const NzNy = constants.NzNy;

  auto const y = static_cast<int64_t>(id.x);
  auto const z = static_cast<int64_t>(id.y);

  u1[0 * NzNy + y * Nz + z]        = u1[2 * NzNy + y * Nz + z];
  u1[(Nx - 1) * NzNy + y * Nz + z] = u1[(Nx - 3) * NzNy + y * Nz + z];
}

[[kernel]] void copyFromGrid(
    device float* out [[buffer(0)]],
    device float const* in [[buffer(1)]],
    device int64_t const* locs [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  out[id] = in[locs[id]];
}

[[kernel]] void copyToGrid(
    device float* out [[buffer(0)]],
    device float const* in [[buffer(1)]],
    device int64_t const* locs [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
  out[locs[id]] = in[id];
}

[[kernel]] void addSource(
    device float* u0 [[buffer(0)]],
    device float const* in_sigs [[buffer(1)]],
    device int64_t const* in_ixyz [[buffer(2)]],
    constant Constants3D<float>& constants [[buffer(3)]],
    constant int64_t& timestep [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  u0[in_ixyz[id]] += in_sigs[id * constants.Nt + timestep];
}

[[kernel]] void readOutput(
    device float* u_out [[buffer(0)]],
    device float const* u1 [[buffer(1)]],
    device int64_t const* out_ixyz [[buffer(2)]],
    constant Constants3D<float>& constants [[buffer(3)]],
    constant int64_t& timestep [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
  u_out[id * constants.Nt + timestep] = u1[out_ixyz[id]];
}

} // namespace sim3d

} // namespace pffdtd
