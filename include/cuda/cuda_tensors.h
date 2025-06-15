#pragma once

#include "tensor.h"

#ifdef USE_CUDA
void cuda_tensor_im2col(const Tensor4D &TA, Matrix &TB, const size_t kH,
                        const size_t kW, const size_t H_pad, const size_t W_pad,
                        const size_t H_stride, const size_t W_stride);
void cuda_tensor_col2im(const Matrix &A, Tensor4D &B, const size_t kH,
                        const size_t kW, const size_t H_pad, const size_t W_pad,
                        const size_t H_stride, const size_t W_stride);

void cuda_tensor_add(const Tensor4D &A, const Tensor4D &B, Tensor4D &C);
void cuda_tensor_add(const Tensor4D &A, const float B, Tensor4D &C);
void cuda_tensor_scale(const Tensor4D &A, const float B, Tensor4D &C);

void cuda_tensor_sum(const Tensor4D &A, const size_t index, Tensor4D &C);
void cuda_tensor_max(const Tensor4D &A, const size_t index, Tensor4D &C);
void cuda_tensor_mean(const Tensor4D &A, const size_t index, Tensor4D &C);
#endif