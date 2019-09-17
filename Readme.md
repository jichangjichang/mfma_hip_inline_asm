## inline asm  ###

This tutorial is about how to use inline mfma GCN asm in kernel.

## Introduction:

MI-100 support MFMA (Matrix Fused Multiply Add) instructions set.
This example just introduces how to call and compile
1. mfma fp32
2. mfma fp16 
in HIP source kernel.

For more insight Please read the following blogs by Ben Sander
[The Art of AMDGCN Assembly: How to Bend the Machine to Your Will](gpuopen.com/amdgcn-assembly)
[AMD GCN Assembly: Cross-Lane Operations](http://gpuopen.com/amd-gcn-assembly-cross-lane-operations/)

For more information:
[AMD GCN3 ISA Architecture Manual](http://gpuopen.com/compute-product/amd-gcn3-isa-architecture-manual/)
[User Guide for AMDGPU Back-end](llvm.org/docs/AMDGPUUsage.html)

## Requirement:

