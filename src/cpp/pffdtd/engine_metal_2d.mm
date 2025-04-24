// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch

#include "engine_metal_2d.hpp"

#include "pffdtd/engine_metal.hpp"
#include "pffdtd/metal.hpp"
#include "pffdtd/progress.hpp"
#include "pffdtd/time.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <concepts>

namespace pffdtd {

namespace {

auto toFloat(std::vector<double> const& buf) {
  auto buf32 = std::vector<float>(buf.size());
  std::ranges::transform(buf, buf32.begin(), [](auto v) { return static_cast<float>(v); });
  return buf32;
}

auto run(Simulation2D const& sim) {
  @autoreleasepool {
    summary(sim);

    auto const Nx          = sim.Nx;
    auto const Ny          = sim.Ny;
    auto const Npts        = Nx * Ny;
    auto const Nt          = sim.Nt;
    auto const Nb          = static_cast<int64_t>(sim.adj_bn.size());
    auto const Nr          = static_cast<int64_t>(sim.out_ixy.size());
    auto const loss_factor = sim.loss_factor;
    auto const in_sigs_f32 = toFloat(sim.in_sigs);

    auto const c = Constants2D<float>{
        .Nx         = Nx,
        .Ny         = Ny,
        .Nt         = Nt,
        .in_ixy     = sim.in_ixy.at(0),
        .lossFactor = static_cast<float>(loss_factor),
    };

    // Device
    NSArray* devices     = MTLCopyAllDevices();
    id<MTLDevice> device = devices[0];
    PFFDTD_ASSERT(device != nil);
    summary(device);

    // Library
    id<MTLLibrary> library = [device newDefaultLibrary];
    PFFDTD_ASSERT(library != nil);

    id<MTLComputePipelineState> airUpdateKernel   = makeComputePipeline(library, "pffdtd::sim2d::airUpdate");
    id<MTLComputePipelineState> rigidUpdateKernel = makeComputePipeline(library, "pffdtd::sim2d::boundaryRigid");
    id<MTLComputePipelineState> rigidLossKernel   = makeComputePipeline(library, "pffdtd::sim2d::boundaryLoss");
    id<MTLComputePipelineState> addSourceKernel   = makeComputePipeline(library, "pffdtd::sim2d::addSource");
    id<MTLComputePipelineState> readOutputKernel  = makeComputePipeline(library, "pffdtd::sim2d::readOutput");

    // Buffer
    id<MTLBuffer> in_mask   = makeBuffer(device, sim.in_mask, MTLResourceStorageModeShared);
    id<MTLBuffer> bn_ixy    = makeBuffer(device, sim.bn_ixy, MTLResourceStorageModeShared);
    id<MTLBuffer> adj_bn    = makeBuffer(device, sim.adj_bn, MTLResourceStorageModeShared);
    id<MTLBuffer> out_ixy   = makeBuffer(device, sim.out_ixy, MTLResourceStorageModeShared);
    id<MTLBuffer> in_sigs   = makeBuffer(device, in_sigs_f32, MTLResourceStorageModeShared);
    id<MTLBuffer> constants = [device newBufferWithBytes:&c
                                                  length:sizeof(Constants2D<float>)
                                                 options:MTLResourceStorageModeShared];
    PFFDTD_ASSERT(constants != nil);

    id<MTLBuffer> u0  = makeEmptyBuffer<float>(device, Npts, MTLResourceStorageModeShared);
    id<MTLBuffer> u1  = makeEmptyBuffer<float>(device, Npts, MTLResourceStorageModeShared);
    id<MTLBuffer> u2  = makeEmptyBuffer<float>(device, Npts, MTLResourceStorageModeShared);
    id<MTLBuffer> out = makeEmptyBuffer<float>(device, Npts, MTLResourceStorageModeShared);

    // Queue
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    PFFDTD_ASSERT(commandQueue != nil);

    auto const start = getTime();

    for (int64_t n{0}; n < Nt; ++n) {
      auto const sampleStart = getTime();

      id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];

      id<MTLBuffer> timestep = [device newBufferWithBytes:&n
                                                   length:sizeof(int64_t)
                                                  options:MTLResourceStorageModeShared];
      PFFDTD_ASSERT(timestep != nil);

      // Air Update
      id<MTLComputeCommandEncoder> airUpdate = [commandBuffer computeCommandEncoder];
      [airUpdate setComputePipelineState:airUpdateKernel];
      [airUpdate setBuffer:u0 offset:0 atIndex:0];
      [airUpdate setBuffer:u1 offset:0 atIndex:1];
      [airUpdate setBuffer:u2 offset:0 atIndex:2];
      [airUpdate setBuffer:in_mask offset:0 atIndex:3];
      [airUpdate setBuffer:constants offset:0 atIndex:4];
      [airUpdate dispatchThreads:MTLSizeMake(Nx - 2, Ny - 2, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [airUpdate endEncoding];

      // Rigid Update
      id<MTLComputeCommandEncoder> rigidUpdate = [commandBuffer computeCommandEncoder];
      [rigidUpdate setComputePipelineState:rigidUpdateKernel];
      [rigidUpdate setBuffer:u0 offset:0 atIndex:0];
      [rigidUpdate setBuffer:u1 offset:0 atIndex:1];
      [rigidUpdate setBuffer:u2 offset:0 atIndex:2];
      [rigidUpdate setBuffer:bn_ixy offset:0 atIndex:3];
      [rigidUpdate setBuffer:adj_bn offset:0 atIndex:4];
      [rigidUpdate setBuffer:constants offset:0 atIndex:5];
      [rigidUpdate dispatchThreads:MTLSizeMake(Nb, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [rigidUpdate endEncoding];

      // Rigid Loss
      id<MTLComputeCommandEncoder> rigidLoss = [commandBuffer computeCommandEncoder];
      [rigidLoss setComputePipelineState:rigidLossKernel];
      [rigidLoss setBuffer:u0 offset:0 atIndex:0];
      [rigidLoss setBuffer:u2 offset:0 atIndex:1];
      [rigidLoss setBuffer:bn_ixy offset:0 atIndex:2];
      [rigidLoss setBuffer:adj_bn offset:0 atIndex:3];
      [rigidLoss setBuffer:constants offset:0 atIndex:4];
      [rigidLoss dispatchThreads:MTLSizeMake(Nb, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [rigidLoss endEncoding];

      // Add Source
      id<MTLComputeCommandEncoder> addSource = [commandBuffer computeCommandEncoder];
      [addSource setComputePipelineState:addSourceKernel];
      [addSource setBuffer:u0 offset:0 atIndex:0];
      [addSource setBuffer:in_sigs offset:0 atIndex:1];
      [addSource setBuffer:constants offset:0 atIndex:2];
      [addSource setBuffer:timestep offset:0 atIndex:3];
      [addSource dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [addSource endEncoding];

      // Read Outputs
      id<MTLComputeCommandEncoder> readOutput = [commandBuffer computeCommandEncoder];
      [readOutput setComputePipelineState:readOutputKernel];
      [readOutput setBuffer:out offset:0 atIndex:0];
      [readOutput setBuffer:u0 offset:0 atIndex:1];
      [readOutput setBuffer:out_ixy offset:0 atIndex:2];
      [readOutput setBuffer:constants offset:0 atIndex:3];
      [readOutput setBuffer:timestep offset:0 atIndex:4];
      [readOutput dispatchThreads:MTLSizeMake(Nr, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [readOutput endEncoding];

      // Wait
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
      PFFDTD_ASSERT(commandBuffer.status == MTLCommandBufferStatusCompleted);

      // Rotate buffers
      auto tmp = u2;
      u2       = u1;
      u1       = u0;
      u0       = tmp;

      auto const now           = getTime();
      auto const elapsed       = now - start;
      auto const elapsedSample = now - sampleStart;

      print(
          ProgressReport{
              .n                     = n,
              .Nt                    = Nt,
              .Npts                  = Npts,
              .Nb                    = Nb,
              .elapsed               = elapsed,
              .elapsedSample         = elapsedSample,
              .elapsedAir            = {},
              .elapsedSampleAir      = {},
              .elapsedBoundary       = {},
              .elapsedSampleBoundary = {},
              .numWorkers            = 1,
          }
      );
    }

    auto outputs_f64 = stdex::mdarray<double, stdex::dextents<size_t, 2>>(Nr, Nt);

    float* outputs_f32 = (float*)[out contents];
    for (auto i{0UL}; i < static_cast<size_t>(Nr * Nt); ++i) {
      outputs_f64.data()[i] = outputs_f32[i];
    }

    return outputs_f64;
  }
}

} // namespace

auto EngineMETAL2D::operator()(Simulation2D const& sim, Precision precision) const
    -> stdex::mdarray<double, stdex::dextents<size_t, 2>> {
  PFFDTD_ASSERT(precision == Precision::Float);
  return run(sim);
}

} // namespace pffdtd
