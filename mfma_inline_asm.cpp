/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>

// hip header file
#include "hip/hip_runtime.h"
typedef float float32 __attribute__((ext_vector_type(32)));
typedef float float16 __attribute__((ext_vector_type(16))); 
typedef float mfma_float4 __attribute__((ext_vector_type(4)));
typedef __fp16 half4 __attribute__((ext_vector_type(4)));

extern "C" __device__ float32   __llvm_amdgcn_mfma_f32_32x32x1f32(float, float, float32, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x1f32");
extern "C" __device__ float16 __llvm_amdgcn_mfma_f32_16x16x1f32(float, float, float16, int, int, int) __asm("llvm.amdgcn.mfma.f32.16x16x1f32");
extern "C" __device__ float4  __llvm_amdgcn_mfma_f32_4x4x1f32(float, float, float4, int, int, int) __asm("llvm.amdgcn.mfma.f32.4x4x1f32");
extern "C" __device__ float16 __llvm_amdgcn_mfma_f32_32x32x2f32(float, float, float16, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x2f32");
extern "C" __device__ mfma_float4  __llvm_amdgcn_mfma_f32_16x16x4f32(float, float, mfma_float4, int, int, int) __asm("llvm.amdgcn.mfma.f32.16x16x4f32");

extern "C" __device__ float32   __llvm_amdgcn_mfma_f32_32x32x4f16(half4, half4, float32, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x4f16");
extern "C" __device__ float16 __llvm_amdgcn_mfma_f32_16x16x4f16(half4, half4, float16, int, int, int) __asm("llvm.amdgcn.mfma.f32.16x16x4f16");
extern "C" __device__ mfma_float4  __llvm_amdgcn_mfma_f32_4x4x4f16(half4, half4, mfma_float4, int, int, int) __asm("llvm.amdgcn.mfma.f32.4x4x4f16");
extern "C" __device__ float16 __llvm_amdgcn_mfma_f32_32x32x8f16(half4, half4, float16, int, int, int) __asm("llvm.amdgcn.mfma.f32.32x32x8f16");
extern "C" __device__ mfma_float4  __llvm_amdgcn_mfma_f32_16x16x16f16(half4, half4, mfma_float4, int, int, int) __asm("llvm.amdgcn.mfma.f32.16x16x16f16");

#define WIDTH 1024

#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 8
#define THREADS_PER_BLOCK_Y 8
#define THREADS_PER_BLOCK_Z 1

// Device (Kernel) function, it must be void
__global__ void matrix_fp32(float* out, float* in, const int width) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    mfma_float4 in_C,out_D;
    float in_A, in_B;

    out_D = __llvm_amdgcn_mfma_f32_16x16x4f32(in_A, in_B,in_C,0,0,0 );
    out[x*width+y] = *(float*)&out_D;
}

__global__ void matrix_fp16(float* out, float* in, const int width) {
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
    int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

    float16 in_C,out_D;
    half4 in_A, in_B;

    out_D = __llvm_amdgcn_mfma_f32_32x32x8f16(in_A, in_B,in_C,0,0,0 );
    out[x*width+y] = *(float*)&out_D;
}

int main() {
    float* Matrix;
    float* TransposeMatrix;

    float* gpuMatrix;
    float* gpuTransposeMatrix;

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);

    std::cout << "Device name " << devProp.name << std::endl;

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    float eventMs = 1.0f;

    int i;
    int errors;

    Matrix = (float*)malloc(NUM * sizeof(float));
    TransposeMatrix = (float*)malloc(NUM * sizeof(float));

    // initialize the input data
    for (i = 0; i < NUM; i++) {
        Matrix[i] = (float)i * 10.0f;
    }

    // allocate the memory on the device side
    hipMalloc((void**)&gpuMatrix, NUM * sizeof(float));
    hipMalloc((void**)&gpuTransposeMatrix, NUM * sizeof(float));

    // Record the start event
    hipEventRecord(start, NULL);

    // Memory transfer from host to device
    hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice);

    // Record the stop event
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf("hipMemcpyHostToDevice time taken  = %6.3fms\n", eventMs);
    // pre-run
    hipLaunchKernelGGL(matrix_fp16, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, gpuTransposeMatrix,
                    gpuMatrix, WIDTH);

    // Record the start event
    hipEventRecord(start, NULL);

    // Lauching kernel from host
    hipLaunchKernelGGL(matrix_fp16, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, gpuTransposeMatrix,
                    gpuMatrix, WIDTH);

    // Record the stop event
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf("fp16 kernel Execution time             = %6.3fms\n", eventMs);

    // Record the start event
    hipEventRecord(start, NULL);

    // Lauching kernel from host
    hipLaunchKernelGGL(matrix_fp32, dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0, 0, gpuTransposeMatrix,
                    gpuMatrix, WIDTH);

    // Record the stop event
    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);

    printf("fp32 kernel Execution time             = %6.3fms\n", eventMs);


    printf("PASSED!\n");
    // free the resources on device side
    hipFree(gpuMatrix);
    hipFree(gpuTransposeMatrix);

    // free the resources on host side
    free(Matrix);
    free(TransposeMatrix);

    return 0;
}
