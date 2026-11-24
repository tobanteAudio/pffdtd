// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton
// Function definitions for handling loading of simulation data from HDF5 files,
// preparing for simulation, and writing outputs
#pragma once

#include "pffdtd/mat_quad.hpp"
#include "pffdtd/precision.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <vector>

namespace pffdtd {

enum struct Grid : int8_t {
  CART       = 0,
  FCC        = 1,
  FCC_FOLDED = 2,
};

[[nodiscard]] constexpr auto isFCC(Grid grid) noexcept -> bool { return grid != Grid::CART; }

// main sim data, on host
struct Simulation3D {
  std::vector<int64_t> bn_ixyz;           // boundary node indices
  std::vector<int64_t> bnl_ixyz;          // lossy boundary node indices
  std::vector<int64_t> bna_ixyz;          // absorbing boundary node indices
  std::vector<int8_t> Q_bna;              // integer for ABCs (wall 1,edge 2,corner 3)
  std::vector<int64_t> in_ixyz;           // input points
  std::vector<int64_t> out_ixyz;          // output points
  std::vector<int64_t> out_reorder;       // ordering for outputs point for final print/save
  std::vector<uint16_t> adj_bn;           // nearest-neighbour adjancencies for all boundary nodes
  std::vector<double> ssaf_bnl;           // surface area corrections (with extra volume scaling)
  std::vector<uint8_t> bn_mask;           // bit mask for bounday nodes
  std::vector<int8_t> mat_bnl;            // material indices for lossy boundary nodes
  std::vector<int8_t> K_bn;               // number of adjacent neighbours, boundary nodesa
  std::vector<double> in_sigs;            // input signals
  std::unique_ptr<double[]> u_out;        // for output signals
  int64_t Ns;                             // number of input grid points
  int64_t Nr;                             // number of output grid points
  int64_t Nt;                             // number of samples simulation
  int64_t Npts;                           // number of Cartesian grid points
  int64_t Nx;                             // x-dim (non-continguous)
  int64_t Ny;                             // y-dim
  int64_t Nz;                             // z-dim (continguous)
  int64_t Nb;                             // number of boundary nodes
  int64_t Nbl;                            // number of lossy boundary nodes
  int64_t Nba;                            // number of ABC nodes
  double l;                               // Courant number (CFL)
  double l2;                              // CFL number squared
  Grid grid;                              // Grid type
  int8_t NN;                              // integer, neareast neighbours
  int8_t Nm;                              // number of materials used
  std::vector<int8_t> Mb;                 // number of branches per material
  std::vector<MatQuad<double>> mat_quads; // RLC coefficients (essentially)
  std::vector<double> mat_beta;           // part of FD-boundaries one per material
  double infac;                           // rescaling of input (for numerical reason)
  double sl2;                             // scaled l2 (for single precision)
  double lo2;                             // 0.5*l
  double a2;                              // update stencil coefficient
  double a1;                              // update stencil coefficient
  Precision precision;                    // Runtime floating-point precision
};

[[nodiscard]] auto loadSimulation3D(std::filesystem::path const& simDir, Precision precision) -> Simulation3D;
auto scaleInput(Simulation3D& sim) -> void;
auto printLastSample(Simulation3D const& sim) -> void;
auto writeOutputs(Simulation3D const& sim, std::filesystem::path const& simDir) -> void;

inline void rescaleOutput(Simulation3D& sim) {
  int64_t const Nt = sim.Nt;
  int64_t const Nr = sim.Nr;
  double infac     = sim.infac;
  double* u_out    = sim.u_out.get();

  std::transform(u_out, u_out + Nr * Nt, u_out, [infac](auto sample) { return sample * infac; });
}

} // namespace pffdtd
