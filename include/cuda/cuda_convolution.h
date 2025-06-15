#pragma once

#include "tensor.h"

#ifdef USE_CUDA
void cuda_convolution(const Tensor4D &input, const Tensor4D &kernel,
                      Tensor4D &output, const size_t H_pad, const size_t W_pad,
                      const size_t H_stride, const size_t W_stride,
                      const Matrix *flatten_kernel = nullptr);
#endif