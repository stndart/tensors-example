#pragma once

#include <iostream>
#include <stdexcept>
#include <vector>

#include "cuda_precision.h"

struct Index2 {
    size_t x, y;
};

class Matrix;
class Tensor4D;

void matrix_to_tensor_reshape(Matrix &TA, Tensor4D &TB, bool copy = true);

class Matrix {
  private:
    size_t dimH_;
    size_t dimW_;

    __half *data_;
    __half *gpu_data_;

    template <typename T> T &access(const Index2 &idx) const;

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
    void print(std::string name = "") const;

    static void gemm(const Matrix &A, const Matrix &B, Matrix &C);

    size_t dimH() const { return dimH_; }
    size_t dimW() const { return dimW_; }

    __half *data() { return data_; }
    const __half *data() const { return data_; }
    __half &operator[](const Index2 &idx);
    const __half &operator[](const Index2 &idx) const;

    __half *gpu_data() { return gpu_data_; }
    const __half *gpu_data() const { return gpu_data_; }

    friend void matrix_to_tensor_reshape(Matrix &TA, Tensor4D &TB, bool copy);
};

#ifdef USE_CUDA
void cuda_matrix_gemm(const Matrix &A, const Matrix &B, Matrix &C);
#endif