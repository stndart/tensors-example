#pragma once

#ifdef USE_CUDA
template <typename T> void cuda_allocate(T **ptr, size_t count);
template <typename T> void cuda_free(T **ptr);
template <typename T> void cuda_h2d(T *host, T *device, size_t count);
template <typename T> void cuda_d2h(T *host, T *device, size_t count);
#endif