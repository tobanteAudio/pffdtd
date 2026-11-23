// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#pragma once

#include <cstdint>
#include <filesystem>
#include <vector>

namespace pffdtd {

struct Simulation2D {
  std::filesystem::path dir;

  int64_t Nx; // Number of x sample nodes
  int64_t Ny; // Number of y sample nodes
  int64_t Nt; // Number of time steps
  double l2;  // CFL number squared

  std::vector<uint8_t> in_mask; // Mask for interior nodes
  std::vector<int64_t> adj_bn;  // Adjacency nodes
  std::vector<int64_t> bn_ixy;  // Boundary nodes
  double loss_factor;           // Boundary loss

  std::vector<double> in_sigs;  // Source signal
  std::vector<int64_t> in_ixy;  // Source nodes
  std::vector<int64_t> out_ixy; // Receiver nodes
};

auto loadSimulation2D(std::filesystem::path const& dir) -> Simulation2D;
auto summary(Simulation2D const& sim) -> void;

} // namespace pffdtd
