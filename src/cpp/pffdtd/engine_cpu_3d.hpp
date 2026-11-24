// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton
// CPU-based implementation of FDTD engine, with OpenMP

#pragma once

#include "pffdtd/simulation_3d.hpp"

namespace pffdtd {

struct EngineCPU3D {
  auto operator()(Simulation3D const& sim) const -> void;
};

} // namespace pffdtd
