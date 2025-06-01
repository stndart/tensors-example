#include "cuda_memory.h"
#include <cuda_runtime.h>
#include <string>

template <typename T> void cuda_allocate(T **ptr, size_t count) {
    if (err = cudaMalloc(reinterpret_cast<void **>(ptr), count * sizeof(T))) {
        throw std::runtime_error("cudaMalloc failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

template <typename T> void cuda_free(T **ptr) {
    cudaFree(reinterpret_cast<void **>(ptr));
}

template <typename T> void cuda_h2d(T *host, T *device, size_t count) {
    if (err = cudaMemcpy(reinterpret_cast<void *>(device),
                         reinterpret_cast<void *>(host), count * sizeof(T),
                         cudaMemcpyHostToDevice)) {
        throw std::runtime_error("cudaMemcpy H2D failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}

template <typename T> void cuda_d2h(T *host, T *device, size_t count) {
    if (err = cudaMemcpy(reinterpret_cast<void *>(host),
                         reinterpret_cast<void *>(device), count * sizeof(T),
                         cudaMemcpyDeviceToHost)) {
        throw std::runtime_error("cudaMemcpy H2D failed: " +
                                 std::string(cudaGetErrorString(err)));
    }
}
