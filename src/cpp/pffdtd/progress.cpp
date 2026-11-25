// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2021 Brian Hamilton

#include "progress.hpp"

#include "pffdtd/assert.hpp"
#include "pffdtd/time.hpp"

#include <fmt/core.h>
#include <fmt/format.h>

#include <cstdio>
#include <ctime>

#if defined(_WIN32)
  #include <windows.h>
#else
  #include <sys/ioctl.h>
#endif

namespace pffdtd {

namespace {

[[nodiscard]] auto getConsoleWidth() -> int {
#if defined(_WIN32)
  auto info = CONSOLE_SCREEN_BUFFER_INFO{};
  if (GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &info)) {
    return info.srWindow.Right - info.srWindow.Left + 1;
  } else {
    return 80;
  }
#else
  auto w = winsize{};
  ioctl(0, TIOCGWINSZ, &w);
  return w.ws_col;
#endif
}

} // namespace

// hacky print progress (like tqdm)..
// N.B. this conflicts with tmux scrolling (stdout needs to flush)
// and not great for piping output to log (better to disable or change for those
// cases)
auto print(ProgressReport const& progress) -> void {
  auto const& p                    = progress;
  auto const elapsed               = Seconds(p.elapsed).count();
  auto const elapsedAir            = Seconds(p.elapsedAir).count();
  auto const elapsedBoundary       = Seconds(p.elapsedBoundary).count();
  auto const elapsedSample         = Seconds(p.elapsedSample).count();
  auto const elapsedSampleAir      = Seconds(p.elapsedSampleAir).count();
  auto const elapsedSampleBoundary = Seconds(p.elapsedSampleBoundary).count();

  auto const n          = static_cast<double>(p.n);
  auto const Npts       = static_cast<double>(p.Npts);
  auto const Nt         = static_cast<double>(p.Nt);
  auto const Nb         = static_cast<double>(p.Nb);
  auto const numWorkers = static_cast<double>(p.numWorkers);

  // progress bar (doesn't impact performance unless simulation is really tiny)
  auto const ncols  = getConsoleWidth();
  auto const ncolsl = 80;

  double const pcnt = (100.0 * n) / Nt;
  int const nlines  = 6;
  if (p.n > 0) {
    // back up
    for (int nl = 0; nl < nlines; nl++) {
      fmt::print("\033[1A");
    }
    // clear lines
    for (int nl = 0; nl < nlines; nl++) {
      for (int cc = 0; cc < ncols; cc++) {
        fmt::print(" ");
      }
      fmt::print("\n");
    }
    // back up
    for (int nl = 0; nl < nlines; nl++) {
      fmt::print("\033[1A");
    }
  }
  fmt::print("\n");
  // progress bar
  fmt::print("Running [{:.1f}%]", pcnt);
  if (ncols >= ncolsl) {
    for (int cc = 0; cc < (0.01 * pcnt * ncolsl); cc++) {
      fmt::print("=");
    }
    fmt::print(">");
    for (auto cc = static_cast<int>(0.01 * pcnt * ncolsl); cc < ncolsl; cc++) {
      fmt::print(".");
    }
  }

  auto const est_total = elapsed * Nt / n;

  auto const sec = static_cast<int>(elapsed);
  auto const h_e = (sec / 3600);
  auto const m_e = (sec - (3600 * h_e)) / 60;
  auto const s_e = (sec - (3600 * h_e) - (m_e * 60));

  auto const sec_e = static_cast<int>(est_total);
  auto const h_t   = (sec_e / 3600);
  auto const m_t   = (sec_e - (3600 * h_t)) / 60;
  auto const s_t   = (sec_e - (3600 * h_t) - (m_t * 60));

  fmt::print("[");
  fmt::print("{:02d}:{:02d}:{:02d}<{:02d}:{:02d}:{:02d}]", h_e, m_e, s_e, h_t, m_t, s_t);
  fmt::println("");
  fmt::print("T: {:06.1f}", 1e-6 * Npts * n / elapsed); //"total" Mvox/s (averaged up to current time)
  fmt::print(" - ");
  fmt::print("I: {:06.1f}", 1e-6 * Npts / elapsedSample); // instantaneous Mvox/s (per time-step)
  fmt::print(" | ");
  fmt::print("TPW: {:06.1f}", 1e-6 * Npts * n / elapsed / numWorkers); // total per worker
  fmt::print(" - ");
  fmt::print("IPW: {:06.1f}", 1e-6 * Npts / elapsedSample / numWorkers); // inst per worker
  fmt::println("");

  fmt::print("TA: {:06.1f}", 1e-6 * Npts * n / elapsedAir); // total for air bit
  fmt::print(" - ");
  fmt::print("IA: {:06.1f}", 1e-6 * Npts / elapsedSampleAir); // inst for air bit

  fmt::println("");
  fmt::print("TB: {:06.1f}", 1e-6 * Nb * n / elapsedBoundary); // total for bn
  fmt::print(" - ");
  fmt::print("IB: {:06.1f}", 1e-6 * Nb / elapsedSampleBoundary); // inst for bn

  fmt::println("");

  fmt::print("T: {:02.1f}%", 100.0 * elapsedAir / elapsed); //% for air (total)
  fmt::print(" - ");
  fmt::print("I: {:02.1f}%", 100.0 * elapsedSampleAir / elapsedSample); //% for air (inst)
  fmt::println("");

  auto const status = std::fflush(stdout);
  PFFDTD_ASSERT(status == 0);
}

} // namespace pffdtd
