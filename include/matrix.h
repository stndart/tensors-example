#pragma once

#include "cuda_precision.h"
#include <iostream>
#include <vector>

class Matrix {
  private:
    size_t dimH_;
    size_t dimW_;

    __half *data_;
    __half *gpu_data_;

  public:
    Matrix(size_t dimH, size_t dimW);
    ~Matrix();

    void allocate_memory();
    void allocate_memory_gpu();

    void H2D();
    void D2H();

    void fill(const __half value);
    void initialize(const std::vector<__half> &data);
    size_t size() const { return dimH_ * dimW_; }
    void print() const;

    static void gemm(const Matrix &A, const Matrix &B, Matrix &C);

    size_t dimH() const { return dimH_; }
    size_t dimW() const { return dimW_; }

    __half *data() { return data_; }
    const __half *data() const { return data_; }

    __half *gpu_data() { return gpu_data_; }
    const __half *gpu_data() const { return gpu_data_; }
};

#ifdef USE_CUDA
void cuda_matrix_gemm(const Matrix &A, const Matrix &B, Matrix &C);
#endif