#pragma once

#include "tensor.h"

#ifdef USE_CUDA
void cuda_max_pooling(Tensor4D input, Tensor4D output, Tensor4D argmax_cache,
                      size_t H_pad, size_t W_pad, size_t H_stride,
                      size_t W_stride, size_t H_size, size_t W_size);
#endif