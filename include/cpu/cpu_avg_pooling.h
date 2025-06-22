#pragma once

#include "cpu/padding.h"
#include "tensor.h"

void cpu_avg_pooling(const Tensor4D &input, Tensor4D &output,
                     PaddingMode padding_mode, size_t H_pad, size_t W_pad,
                     size_t H_stride, size_t W_stride, size_t H_size,
                     size_t W_size);

void cpu_avg_pooling_backward(const Tensor4D &output_gradient,
                              Tensor4D &input_gradient,
                              PaddingMode padding_mode, size_t H_pad,
                              size_t W_pad, size_t H_stride, size_t W_stride,
                              size_t H_size, size_t W_size);