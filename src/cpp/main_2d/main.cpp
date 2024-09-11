// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#include "engine_native.hpp"

#if defined(PFFDTD_HAS_SYCL)
  #include "engine_sycl.hpp"
#endif

#include "pffdtd/exception.hpp"
#include "pffdtd/hdf.hpp"
#include "pffdtd/simulation_2d.hpp"

#include <CLI/CLI.hpp>

#include <fmt/format.h>

#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>

struct Arguments {
  std::string engine{"native"};
  std::string simDir{};
  std::string out{"out.h5"};
};

int main(int argc, char** argv) {
  auto app  = CLI::App{"pffdtd-2d"};
  auto args = Arguments{};
  app.add_option("-e,--engine", args.engine);
  app.add_option("-s,--sim_dir", args.simDir)->check(CLI::ExistingDirectory);
  app.add_option("-o,--out", args.out);
  CLI11_PARSE(app, argc, argv);

  auto const start  = std::chrono::steady_clock::now();
  auto const simDir = std::filesystem::path{args.simDir};
  auto const sim    = pffdtd::loadSimulation2D(simDir);
  auto const out    = [&] {
    if (args.engine == "native") {
      fmt::println("Using engine: NATIVE");
      auto const engine = pffdtd::EngineNative{};
      return engine(sim);
    } else if (args.engine == "sycl") {
#if defined(PFFDTD_HAS_SYCL)
      fmt::println("Using engine: SYCL");
      auto const engine = pffdtd::EngineSYCL{};
      return engine(sim);
#else
      pffdtd::raisef<std::runtime_error>("pffdtd built without SYCL support");
#endif
    } else {
      pffdtd::raisef<std::runtime_error>("invalid engine '{}'", args.engine);
    }
  }();

  auto results = pffdtd::H5FWriter{simDir / args.out};
  results.write("out", out);

  auto const stop = std::chrono::steady_clock::now();
  auto const sec  = std::chrono::duration<double>(stop - start);
  fmt::println("Simulation time: {} s", sec.count());

  return EXIT_SUCCESS;
}
