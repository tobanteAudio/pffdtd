// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2024 Tobias Hienzsch
#include "engine_metal_3d.hpp"

#include "pffdtd/assert.hpp"
#include "pffdtd/engine_metal.hpp"
#include "pffdtd/metal.hpp"
#include "pffdtd/progress.hpp"
#include "pffdtd/time.hpp"
#include "pffdtd/utility.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <vector>

namespace pffdtd {

namespace {

constexpr auto simdGroupsPerThreadgroup = NSUInteger{8};

auto threadsPerThreadgroup1D(id<MTLComputePipelineState> pipeline) -> MTLSize {
  auto const width      = pipeline.threadExecutionWidth;
  auto const maxThreads = pipeline.maxTotalThreadsPerThreadgroup;
  PFFDTD_ASSERT(width > 0);
  PFFDTD_ASSERT(maxThreads >= width);

  auto const simdGroups = std::min(simdGroupsPerThreadgroup, maxThreads / width);
  return MTLSizeMake(width * simdGroups, 1, 1);
}

auto threadsPerThreadgroup2D(id<MTLComputePipelineState> pipeline) -> MTLSize {
  auto const width      = pipeline.threadExecutionWidth;
  auto const maxThreads = pipeline.maxTotalThreadsPerThreadgroup;
  PFFDTD_ASSERT(width > 0);
  PFFDTD_ASSERT(maxThreads >= width);

  auto const rows = std::min(simdGroupsPerThreadgroup, maxThreads / width);
  return MTLSizeMake(width, rows, 1);
}

auto threadsPerThreadgroup3D(id<MTLComputePipelineState> pipeline) -> MTLSize {
  auto const width      = pipeline.threadExecutionWidth;
  auto const maxThreads = pipeline.maxTotalThreadsPerThreadgroup;
  PFFDTD_ASSERT(width > 0);
  PFFDTD_ASSERT(maxThreads >= width);

  auto const rows  = std::min(NSUInteger{2}, maxThreads / width);
  auto const depth = std::min(NSUInteger{2}, maxThreads / (width * rows));
  return MTLSizeMake(width, rows, depth);
}

template<typename Real>
auto run(Simulation3D const& sim) {
  @autoreleasepool {

    PFFDTD_ASSERT(sim.grid == Grid::CART);

    // Device
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    PFFDTD_ASSERT(device != nil);
    summary(device);

    // Library
    id<MTLLibrary> library = [device newDefaultLibrary];
    PFFDTD_ASSERT(library != nil);

    id<MTLComputePipelineState> kernelAirCart         = makeComputePipeline(library, "pffdtd::sim3d::airUpdateCart");
    id<MTLComputePipelineState> kernelRigidCart       = makeComputePipeline(library, "pffdtd::sim3d::rigidUpdateCart");
    id<MTLComputePipelineState> kernelCopyFromGrid    = makeComputePipeline(library, "pffdtd::sim3d::copyFromGrid");
    id<MTLComputePipelineState> kernelCopyToGrid      = makeComputePipeline(library, "pffdtd::sim3d::copyToGrid");
    id<MTLComputePipelineState> kernelBoundaryLossFD  = makeComputePipeline(library, "pffdtd::sim3d::boundaryLossFD");
    id<MTLComputePipelineState> kernelBoundaryLossABC = makeComputePipeline(library, "pffdtd::sim3d::boundaryLossABC");
    id<MTLComputePipelineState> kernelFlipHaloXY      = makeComputePipeline(library, "pffdtd::sim3d::flipHaloXY");
    id<MTLComputePipelineState> kernelFlipHaloXZ      = makeComputePipeline(library, "pffdtd::sim3d::flipHaloXZ");
    id<MTLComputePipelineState> kernelFlipHaloYZ      = makeComputePipeline(library, "pffdtd::sim3d::flipHaloYZ");
    id<MTLComputePipelineState> kernelAddSource       = makeComputePipeline(library, "pffdtd::sim3d::addSource");
    id<MTLComputePipelineState> kernelReadOutput      = makeComputePipeline(library, "pffdtd::sim3d::readOutput");

    auto const airThreads      = threadsPerThreadgroup3D(kernelAirCart);
    auto const rigidThreads    = threadsPerThreadgroup1D(kernelRigidCart);
    auto const copyFromThreads = threadsPerThreadgroup1D(kernelCopyFromGrid);
    auto const copyToThreads   = threadsPerThreadgroup1D(kernelCopyToGrid);
    auto const lossFDThreads   = threadsPerThreadgroup1D(kernelBoundaryLossFD);
    auto const lossABCThreads  = threadsPerThreadgroup1D(kernelBoundaryLossABC);
    auto const haloXYThreads   = threadsPerThreadgroup2D(kernelFlipHaloXY);
    auto const haloXZThreads   = threadsPerThreadgroup2D(kernelFlipHaloXZ);
    auto const haloYZThreads   = threadsPerThreadgroup2D(kernelFlipHaloYZ);
    auto const sourceThreads   = threadsPerThreadgroup1D(kernelAddSource);
    auto const outputThreads   = threadsPerThreadgroup1D(kernelReadOutput);

    auto const Nx   = sim.Nx;
    auto const Ny   = sim.Ny;
    auto const Nz   = sim.Nz;
    auto const NzNy = Nz * Ny;
    auto const Npts = sim.Npts;
    auto const Nb   = sim.Nb;
    auto const Nbl  = sim.Nbl;
    auto const Nba  = sim.Nba;
    auto const Nr   = sim.Nr;
    auto const Ns   = sim.Ns;
    auto const Nt   = sim.Nt;

    auto const lo2 = static_cast<Real>(sim.lo2);
    auto const sl2 = static_cast<Real>(sim.sl2);
    auto const l   = static_cast<Real>(sim.l);
    auto const a1  = static_cast<Real>(sim.a1);
    auto const a2  = static_cast<Real>(sim.a2);

    auto const in_sigs_real   = convertTo<Real>(sim.in_sigs);
    auto const ssaf_bnl_real  = convertTo<Real>(sim.ssaf_bnl);
    auto const mat_beta_real  = convertTo<Real>(sim.mat_beta);
    auto const mat_quads_real = convertTo<Real>(sim.mat_quads);

    fmt::println("COPY-BUFFERS");
    id<MTLBuffer> out_ixyz  = makeBuffer(device, sim.out_ixyz, MTLResourceStorageModeShared);
    id<MTLBuffer> in_ixyz   = makeBuffer(device, sim.in_ixyz, MTLResourceStorageModeShared);
    id<MTLBuffer> in_sigs   = makeBuffer(device, in_sigs_real, MTLResourceStorageModeShared);
    id<MTLBuffer> adj_bn    = makeBuffer(device, sim.adj_bn, MTLResourceStorageModeShared);
    id<MTLBuffer> bn_mask   = makeBuffer(device, sim.bn_mask, MTLResourceStorageModeShared);
    id<MTLBuffer> bn_ixyz   = makeBuffer(device, sim.bn_ixyz, MTLResourceStorageModeShared);
    id<MTLBuffer> bnl_ixyz  = makeBuffer(device, sim.bnl_ixyz, MTLResourceStorageModeShared);
    id<MTLBuffer> bna_ixyz  = makeBuffer(device, sim.bna_ixyz, MTLResourceStorageModeShared);
    id<MTLBuffer> ssaf_bnl  = makeBuffer(device, ssaf_bnl_real, MTLResourceStorageModeShared);
    id<MTLBuffer> mat_beta  = makeBuffer(device, mat_beta_real, MTLResourceStorageModeShared);
    id<MTLBuffer> mat_bnl   = makeBuffer(device, sim.mat_bnl, MTLResourceStorageModeShared);
    id<MTLBuffer> Mb        = makeBuffer(device, sim.Mb, MTLResourceStorageModeShared);
    id<MTLBuffer> mat_quads = makeBuffer(device, mat_quads_real, MTLResourceStorageModeShared);
    id<MTLBuffer> Q_bna     = makeBuffer(device, sim.Q_bna, MTLResourceStorageModeShared);

    fmt::println("EMPTY-BUFFERS");
    id<MTLBuffer> u0    = makeEmptyBuffer<Real>(device, Npts, MTLResourceStorageModeShared);
    id<MTLBuffer> u1    = makeEmptyBuffer<Real>(device, Npts, MTLResourceStorageModeShared);
    id<MTLBuffer> u0b   = makeEmptyBuffer<Real>(device, Nbl, MTLResourceStorageModeShared);
    id<MTLBuffer> u1b   = makeEmptyBuffer<Real>(device, Nbl, MTLResourceStorageModeShared);
    id<MTLBuffer> u2b   = makeEmptyBuffer<Real>(device, Nbl, MTLResourceStorageModeShared);
    id<MTLBuffer> u2ba  = makeEmptyBuffer<Real>(device, Nba, MTLResourceStorageModeShared);
    id<MTLBuffer> gh1   = makeEmptyBuffer<Real>(device, Nbl * MMb, MTLResourceStorageModeShared);
    id<MTLBuffer> vh1   = makeEmptyBuffer<Real>(device, Nbl * MMb, MTLResourceStorageModeShared);
    id<MTLBuffer> u_out = makeEmptyBuffer<Real>(device, Nr * Nt, MTLResourceStorageModeShared);

    auto const c = Constants3D<Real>{
        .l    = l,
        .lo2  = lo2,
        .sl2  = sl2,
        .a1   = a1,
        .a2   = a2,
        .Nx   = Nx,
        .Ny   = Ny,
        .Nz   = Nz,
        .NzNy = NzNy,
        .Nb   = Nb,
        .Nbl  = Nbl,
        .Nba  = Nba,
        .Ns   = Ns,
        .Nr   = Nr,
        .Nt   = Nt,
    };
    id<MTLBuffer> constants = [device newBufferWithBytes:&c length:sizeof(c) options:MTLResourceStorageModeShared];

    // Queue
    id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    PFFDTD_ASSERT(commandQueue != nil);

    fmt::println("START");
    auto const start = getTime();
    for (int64_t n = 0; n < Nt; n++) {
      auto const sampleStart = getTime();

      id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
      id<MTLBuffer> timestep = [device newBufferWithBytes:&n length:sizeof(n) options:MTLResourceStorageModeShared];
      PFFDTD_ASSERT(timestep != nil);

      // Copy the previous state at the absorbing boundary.
      id<MTLComputeCommandEncoder> copyToABC = [commandBuffer computeCommandEncoder];
      [copyToABC setComputePipelineState:kernelCopyFromGrid];
      [copyToABC setBuffer:u2ba offset:0 atIndex:0];
      [copyToABC setBuffer:u0 offset:0 atIndex:1];
      [copyToABC setBuffer:bna_ixyz offset:0 atIndex:2];
      [copyToABC dispatchThreads:MTLSizeMake(Nba, 1, 1) threadsPerThreadgroup:copyFromThreads];
      [copyToABC endEncoding];

      // Flip halo XY
      id<MTLComputeCommandEncoder> flipHaloXY = [commandBuffer computeCommandEncoder];
      [flipHaloXY setComputePipelineState:kernelFlipHaloXY];
      [flipHaloXY setBuffer:u1 offset:0 atIndex:0];
      [flipHaloXY setBuffer:constants offset:0 atIndex:1];
      [flipHaloXY dispatchThreads:MTLSizeMake(Ny, Nx, 1) threadsPerThreadgroup:haloXYThreads];
      [flipHaloXY endEncoding];

      // Flip halo XZ
      id<MTLComputeCommandEncoder> flipHaloXZ = [commandBuffer computeCommandEncoder];
      [flipHaloXZ setComputePipelineState:kernelFlipHaloXZ];
      [flipHaloXZ setBuffer:u1 offset:0 atIndex:0];
      [flipHaloXZ setBuffer:constants offset:0 atIndex:1];
      [flipHaloXZ dispatchThreads:MTLSizeMake(Nz, Nx, 1) threadsPerThreadgroup:haloXZThreads];
      [flipHaloXZ endEncoding];

      // Flip halo YZ
      id<MTLComputeCommandEncoder> flipHaloYZ = [commandBuffer computeCommandEncoder];
      [flipHaloYZ setComputePipelineState:kernelFlipHaloYZ];
      [flipHaloYZ setBuffer:u1 offset:0 atIndex:0];
      [flipHaloYZ setBuffer:constants offset:0 atIndex:1];
      [flipHaloYZ dispatchThreads:MTLSizeMake(Nz, Ny, 1) threadsPerThreadgroup:haloYZThreads];
      [flipHaloYZ endEncoding];

      // Air update
      id<MTLComputeCommandEncoder> airUpdate = [commandBuffer computeCommandEncoder];
      [airUpdate setComputePipelineState:kernelAirCart];
      [airUpdate setBuffer:u0 offset:0 atIndex:0];
      [airUpdate setBuffer:u1 offset:0 atIndex:1];
      [airUpdate setBuffer:bn_mask offset:0 atIndex:2];
      [airUpdate setBuffer:constants offset:0 atIndex:3];
      [airUpdate dispatchThreads:MTLSizeMake(Nz - 2, Ny - 2, Nx - 2) threadsPerThreadgroup:airThreads];
      [airUpdate endEncoding];

      // ABC loss
      id<MTLComputeCommandEncoder> abcLoss = [commandBuffer computeCommandEncoder];
      [abcLoss setComputePipelineState:kernelBoundaryLossABC];
      [abcLoss setBuffer:u0 offset:0 atIndex:0];
      [abcLoss setBuffer:u2ba offset:0 atIndex:1];
      [abcLoss setBuffer:Q_bna offset:0 atIndex:2];
      [abcLoss setBuffer:bna_ixyz offset:0 atIndex:3];
      [abcLoss setBuffer:constants offset:0 atIndex:4];
      [abcLoss dispatchThreads:MTLSizeMake(Nba, 1, 1) threadsPerThreadgroup:lossABCThreads];
      [abcLoss endEncoding];

      // Rigid boundary update
      id<MTLComputeCommandEncoder> rigidUpdate = [commandBuffer computeCommandEncoder];
      [rigidUpdate setComputePipelineState:kernelRigidCart];
      [rigidUpdate setBuffer:u0 offset:0 atIndex:0];
      [rigidUpdate setBuffer:u1 offset:0 atIndex:1];
      [rigidUpdate setBuffer:bn_ixyz offset:0 atIndex:2];
      [rigidUpdate setBuffer:adj_bn offset:0 atIndex:3];
      [rigidUpdate setBuffer:constants offset:0 atIndex:4];
      [rigidUpdate dispatchThreads:MTLSizeMake(Nb, 1, 1) threadsPerThreadgroup:rigidThreads];
      [rigidUpdate endEncoding];

      if (Nbl > 0) {
        // Copy lossy boundary nodes to their state buffer.
        id<MTLComputeCommandEncoder> copyToBoundary = [commandBuffer computeCommandEncoder];
        [copyToBoundary setComputePipelineState:kernelCopyFromGrid];
        [copyToBoundary setBuffer:u0b offset:0 atIndex:0];
        [copyToBoundary setBuffer:u0 offset:0 atIndex:1];
        [copyToBoundary setBuffer:bnl_ixyz offset:0 atIndex:2];
        [copyToBoundary dispatchThreads:MTLSizeMake(Nbl, 1, 1) threadsPerThreadgroup:copyFromThreads];
        [copyToBoundary endEncoding];

        // Apply frequency-dependent boundary loss.
        id<MTLComputeCommandEncoder> boundaryLoss = [commandBuffer computeCommandEncoder];
        [boundaryLoss setComputePipelineState:kernelBoundaryLossFD];
        [boundaryLoss setBuffer:u0b offset:0 atIndex:0];
        [boundaryLoss setBuffer:u2b offset:0 atIndex:1];
        [boundaryLoss setBuffer:vh1 offset:0 atIndex:2];
        [boundaryLoss setBuffer:gh1 offset:0 atIndex:3];
        [boundaryLoss setBuffer:ssaf_bnl offset:0 atIndex:4];
        [boundaryLoss setBuffer:mat_bnl offset:0 atIndex:5];
        [boundaryLoss setBuffer:mat_beta offset:0 atIndex:6];
        [boundaryLoss setBuffer:mat_quads offset:0 atIndex:7];
        [boundaryLoss setBuffer:Mb offset:0 atIndex:8];
        [boundaryLoss setBuffer:constants offset:0 atIndex:9];
        [boundaryLoss dispatchThreads:MTLSizeMake(Nbl, 1, 1) threadsPerThreadgroup:lossFDThreads];
        [boundaryLoss endEncoding];

        // Copy the updated lossy boundary nodes back to the grid.
        id<MTLComputeCommandEncoder> copyFromBoundary = [commandBuffer computeCommandEncoder];
        [copyFromBoundary setComputePipelineState:kernelCopyToGrid];
        [copyFromBoundary setBuffer:u0 offset:0 atIndex:0];
        [copyFromBoundary setBuffer:u0b offset:0 atIndex:1];
        [copyFromBoundary setBuffer:bnl_ixyz offset:0 atIndex:2];
        [copyFromBoundary dispatchThreads:MTLSizeMake(Nbl, 1, 1) threadsPerThreadgroup:copyToThreads];
        [copyFromBoundary endEncoding];
      }

      // Read receiver
      id<MTLComputeCommandEncoder> readOutput = [commandBuffer computeCommandEncoder];
      [readOutput setComputePipelineState:kernelReadOutput];
      [readOutput setBuffer:u_out offset:0 atIndex:0];
      [readOutput setBuffer:u1 offset:0 atIndex:1];
      [readOutput setBuffer:out_ixyz offset:0 atIndex:2];
      [readOutput setBuffer:constants offset:0 atIndex:3];
      [readOutput setBuffer:timestep offset:0 atIndex:4];
      [readOutput dispatchThreads:MTLSizeMake(Nr, 1, 1) threadsPerThreadgroup:outputThreads];
      [readOutput endEncoding];

      // Add the source to the next state after the stencil and boundary updates.
      id<MTLComputeCommandEncoder> addSource = [commandBuffer computeCommandEncoder];
      [addSource setComputePipelineState:kernelAddSource];
      [addSource setBuffer:u0 offset:0 atIndex:0];
      [addSource setBuffer:in_sigs offset:0 atIndex:1];
      [addSource setBuffer:in_ixyz offset:0 atIndex:2];
      [addSource setBuffer:constants offset:0 atIndex:3];
      [addSource setBuffer:timestep offset:0 atIndex:4];
      [addSource dispatchThreads:MTLSizeMake(Ns, 1, 1) threadsPerThreadgroup:sourceThreads];
      [addSource endEncoding];

      // Wait
      [commandBuffer commit];
      [commandBuffer waitUntilCompleted];
      PFFDTD_ASSERT(commandBuffer.status == MTLCommandBufferStatusCompleted);

      // Swap buffers
      {
        auto tmp = u1;
        u1       = u0;
        u0       = tmp;
      }
      {
        auto tmp = u2b;
        u2b      = u1b;
        u1b      = u0b;
        u0b      = tmp;
      }

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

    float* output = (float*)[u_out contents];
    for (auto i{0UL}; i < static_cast<size_t>(Nr * Nt); ++i) {
      sim.u_out[i] = output[i];
    }
  }
}

} // namespace

auto EngineMETAL3D::operator()(Simulation3D const& sim) const -> void {
  PFFDTD_ASSERT(sim.precision == Precision::Float);
  run<float>(sim);
}

} // namespace pffdtd
