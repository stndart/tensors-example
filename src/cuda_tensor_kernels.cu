#include "tensor.h"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

__global__ void matrix_multiply_kernel(const float *A, const float *B, float *C,
                                       int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// CUDA tensor operations

void cuda_tensor_multiply(Tensor4D &A, Tensor4D &B, Tensor4D &C) {
    A.allocate_memory_gpu();
    A.H2D();

    B.allocate_memory_gpu();
    B.H2D();

    C.allocate_memory_gpu();

    // Configure and launch kernel
    const size_t M = A.dimX();
    const size_t N = A.dimX();
    const size_t K = A.cols();

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y,

    );

    matrix_multiply_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M,
                                                               N, K);

    // Check for kernel errors
    if ((err = cudaGetLastError())) {
        throw std::runtime_error("Kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    // Copy result back to host
    if ((err = cudaMemcpy(C.data(), d_C, M * N * sizeof(float),
                          cudaMemcpyDeviceToHost))) {
        goto cleanup;
    }
}
void cuda_tensor_add(const Tensor4D &A, const Tensor4D &B, Tensor4D &C) {}
void cuda_tensor_add(const Tensor4D &A, const __half B, Tensor4D &C) {}
void cuda_tensor_scale(const Tensor4D &A, const __half B, Tensor4D &C) {}

void cuda_tensor_sum(const Tensor4D &A, const size_t index, Tensor4D &C) {}
void cuda_tensor_max(const Tensor4D &A, const size_t index, Tensor4D &C) {}
void cuda_tensor_mean(const Tensor4D &A, const size_t index, Tensor4D &C) {}

// CUDA matrix multiplication implementation
void cuda_matrix_multiply(const Tensor &A, const Tensor &B, Tensor &C) {
    const int M = A.rows();
    const int N = B.cols();
    const int K = A.cols();

    // Allocate device memory
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    cudaError_t err;

    if (err = cudaMalloc(&d_A, M * K * sizeof(float))) {
        throw std::runtime_error("cudaMalloc d_A failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    if (err = cudaMalloc(&d_B, K * N * sizeof(float))) {
        cudaFree(d_A);
        throw std::runtime_error("cudaMalloc d_B failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
    if (err = cudaMalloc(&d_C, M * N * sizeof(float))) {
        cudaFree(d_A);
        cudaFree(d_B);
        throw std::runtime_error("cudaMalloc d_C failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    // Copy data to device
    if ((err = cudaMemcpy(d_A, A.data(), M * K * sizeof(float),
                          cudaMemcpyHostToDevice))) {
        goto cleanup;
    }
    if ((err = cudaMemcpy(d_B, B.data(), K * N * sizeof(float),
                          cudaMemcpyHostToDevice))) {
        goto cleanup;
    }

    // Configure and launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiply_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M,
                                                               N, K);

    // Check for kernel errors
    if ((err = cudaGetLastError())) {
        throw std::runtime_error("Kernel launch failed: " +
                                 std::string(cudaGetErrorString(err)));
    }

    // Copy result back to host
    if ((err = cudaMemcpy(C.data(), d_C, M * N * sizeof(float),
                          cudaMemcpyDeviceToHost))) {
        goto cleanup;
    }

cleanup:
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    if (err != cudaSuccess) {
        throw std::runtime_error("CUDA operation failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}