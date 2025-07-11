#include "cuda/cuda_matrixes.h"

#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

__global__ void matrix_multiply_kernel(const __half *A, const __half *B,
                                       __half *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        __half sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CUDA matrix multiplication implementation
void cuda_matrix_gemm(const Matrix &A, const Matrix &B, Matrix &C) {
    const int M = A.dimH();
    const int K = A.dimW();
    const int N = B.dimW();

    C.allocate_memory_gpu();

    // Configure and launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiply_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        A.gpu_data(), B.gpu_data(), C.gpu_data(), M, N, K);

    //  Check for launch‐invalid errors:
    cudaError_t launchErr = cudaPeekAtLastError();
    if (launchErr != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(launchErr)
                  << "\n";
        throw std::runtime_error("Kernel launch failed");
    }

    //  Synchronize to catch any runtime errors inside the kernel:
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        std::cerr << "Kernel execution failed (during sync): "
                  << cudaGetErrorString(syncErr) << "\n";
        throw std::runtime_error("Kernel execution failed");
    }
}