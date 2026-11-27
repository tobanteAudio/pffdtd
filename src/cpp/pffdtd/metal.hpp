// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#pragma once

#if not defined(PFFDTD_HAS_METAL)
  #error "Metal must be enabled in CMake via PFFDTD_ENABLE_METAL"
#endif

#include "pffdtd/assert.hpp"

#include <fmt/core.h>
#include <fmt/format.h>

#include <Foundation/Foundation.h>
#include <Metal/Metal.h>

#include <cstdint>
#include <cstring>

namespace pffdtd {

inline void summary(id<MTLDevice> device) {
  fmt::println("- Device {}", [device.name UTF8String]);
  fmt::println("  - Unified memory: {}", device.hasUnifiedMemory != 0 ? "true" : "false");
  fmt::println("  - Max buffer length {} MB", device.maxBufferLength / 1'000'000);
  fmt::println("  - Recommended max working set size: {} MB", device.recommendedMaxWorkingSetSize / 1'000'000);
  fmt::println("  - Max transfer rate: {}", device.maxTransferRate);
  fmt::println("  - Max threads per threadgroup: {}", device.maxThreadsPerThreadgroup.width);
  fmt::println("  - Max threadgroup memory size: {} bytes\n", device.maxThreadgroupMemoryLength);
}

inline id<MTLComputePipelineState> makeComputePipeline(id<MTLLibrary> library, char const* function) {
  id<MTLFunction> kernel = [library newFunctionWithName:[NSString stringWithUTF8String:function]];
  PFFDTD_ASSERT(kernel != nil);

  NSError* error                       = nullptr;
  id<MTLComputePipelineState> pipeline = [library.device newComputePipelineStateWithFunction:kernel error:&error];
  PFFDTD_ASSERT(error == nullptr);

  fmt::println("- {}:", function);
  fmt::println("  - threadExecutionWidth = {}", pipeline.threadExecutionWidth);
  fmt::println("  - maxTotalThreadsPerThreadgroup = {}", pipeline.maxTotalThreadsPerThreadgroup);
  fmt::println("  - staticThreadgroupMemoryLength = {}\n", pipeline.staticThreadgroupMemoryLength);

  return pipeline;
}

template<typename T, typename ModeT>
id<MTLBuffer> makeBuffer(id<MTLDevice> device, std::vector<T> const& data, ModeT mode) {
  auto const size = data.size() * sizeof(T);
  id<MTLBuffer> buf;
  if (data.empty()) {
    buf = [device newBufferWithLength:sizeof(T) options:mode];
    PFFDTD_ASSERT(buf != nil);
    PFFDTD_ASSERT([buf contents] != nullptr);
    std::memset([buf contents], 0, sizeof(T));
  } else {
    buf = [device newBufferWithBytes:data.data() length:size options:mode];
  }
  PFFDTD_ASSERT(buf != nil);
  return buf;
}

template<typename T, typename ModeT>
id<MTLBuffer> makeEmptyBuffer(id<MTLDevice> device, int64_t count, ModeT mode) {
  auto const size           = static_cast<size_t>(count) * sizeof(T);
  auto const allocationSize = std::max(size, sizeof(T));
  id<MTLBuffer> buf         = [device newBufferWithLength:allocationSize options:mode];
  PFFDTD_ASSERT(buf != nil);
  PFFDTD_ASSERT([buf contents] != nullptr);
  std::memset([buf contents], 0, allocationSize);
  return buf;
}

} // namespace pffdtd
