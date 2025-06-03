#pragma once

#include "cuda_precision.h"
#include "matrix.h"
#include <iostream>
#include <stdexcept>
#include <vector>

class Tensor4D {
  private:
    size_t dimW_;
    size_t dimX_;
    size_t dimY_;
    size_t dimZ_;

    __half *data_;
    __half *gpu_data_;

  public:
    Tensor4D(size_t dimW, size_t dimX, size_t dimY, size_t dimZ);
    ~Tensor4D();

    void allocate_memory();
    void allocate_memory_gpu();

    void H2D();
    void D2H();

    void initialize(const std::vector<__half> &data);
    size_t size() const { return dimW_ * dimX_ * dimY_ * dimZ_; }
    void print() const;

    static void im2col(const Tensor4D &TA, Matrix &TB, const size_t kH,
                       const size_t kW, const size_t H_pad, const size_t W_pad,
                       const size_t H_stride, const size_t W_stride);
    static void col2im(const Matrix &A, Tensor4D &B, const size_t kH,
                       const size_t kW, const size_t H_pad, const size_t W_pad,
                       const size_t H_stride, const size_t W_stride);

    // Tensor element-wise operations
    // static void multiply(const Tensor4D &A, const Tensor4D &B, Tensor4D &C);
    static void add(const Tensor4D &A, const Tensor4D &B, Tensor4D &C);
    static void add(const Tensor4D &A, const __half B, Tensor4D &C);
    static void scale(const Tensor4D &A, const __half B, Tensor4D &C);

    // Reductions
    static void sum(const Tensor4D &A, const size_t index, Tensor4D &C);
    static void max(const Tensor4D &A, const size_t index, Tensor4D &C);
    static void mean(const Tensor4D &A, const size_t index, Tensor4D &C);

    // Getters
    size_t dimW() const { return dimW_; }
    size_t dimX() const { return dimX_; }
    size_t dimY() const { return dimY_; }
    size_t dimZ() const { return dimZ_; }

    __half *data() { return data_; }
    const __half *data() const { return data_; }

    __half *gpu_data() { return gpu_data_; }
    const __half *gpu_data() const { return gpu_data_; }
};

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