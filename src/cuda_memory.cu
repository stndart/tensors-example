#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " code=" << err << " \"" << cudaGetErrorString(err)   \
                      << "\"" << std::endl;                                    \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

template <typename T> void cuda_allocate(T **ptr, size_t count) {
    cudaError_t err;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(ptr), count * sizeof(T)));
}

template <typename T> void cuda_free(T **ptr) {
    CUDA_CHECK(cudaFree(reinterpret_cast<void *>(*ptr)));
}

template <typename T> void cuda_h2d(T *host, T *device, size_t count) {
    cudaError_t err;
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(device),
                          reinterpret_cast<void *>(host), count * sizeof(T),
                          cudaMemcpyHostToDevice));
}

template <typename T> void cuda_d2h(T *host, T *device, size_t count) {
    cudaError_t err;
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void *>(host),
                          reinterpret_cast<void *>(device), count * sizeof(T),
                          cudaMemcpyDeviceToHost));
}
#endif