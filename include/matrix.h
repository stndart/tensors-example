#pragma once

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <tuple>

#include "cuda/cuda_precision.h"

struct Index2 {
    union {
        struct {
            int32_t x, y;
        };
        int32_t dims[2];
    };

    int32_t &operator[](size_t i) {
        assert(i < 2);
        return dims[i];
    }
    const int32_t &operator[](size_t i) const {
        assert(i < 2);
        return dims[i];
    }

    // for things like `auto [a, b] = idx.as_tuple();`
    auto as_tuple() const { return std::make_tuple(x, y); }

    bool operator==(const Index2 &other) const {
        return x == other.x && y == other.y;
    }
    bool operator!=(const Index2 &other) const { return !operator==(other); }

    // work as all()
    bool operator<(const size_t other) const { return x < other && y < other; }
    bool operator<=(const size_t other) const {
        return x <= other && y <= other;
    }
    bool operator>(const size_t other) const { return x > other && y > other; }
    bool operator>=(const size_t other) const {
        return x >= other && y >= other;
    }
    bool operator<(const Index2 other) const {
        return x < other.x && y < other.y;
    }
    bool operator<=(const Index2 other) const {
        return x <= other.x && y <= other.y;
    }
    bool operator>(const Index2 other) const {
        return x > other.x && y > other.y;
    }
    bool operator>=(const Index2 other) const {
        return x >= other.x && y >= other.y;
    }

    friend std::ostream &operator<<(std::ostream &os, const Index2 &idx) {
        os << "[" << idx.x << ", " << idx.y << "]";
        return os;
    }
    friend std::istream &operator>>(std::istream &is, Index2 &idx) {
        is >> idx.x >> idx.y;
        return is;
    }
};

class Matrix;
class Tensor4D;

void tensor_to_matrix_reshape(Tensor4D &TA, Matrix &TB);
void tensor_to_matrix_reshape_const(const Tensor4D &TA, Matrix &TB);
void matrix_to_tensor_reshape(Matrix &TA, Tensor4D &TB);
void matrix_to_tensor_reshape_const(const Matrix &TA, Tensor4D &TB);

class Matrix {
  public:
    const size_t NDIMS = 2;
    using Index = Index2;

  private:
    size_t dimH_;
    size_t dimW_;

    __half *data_;
    __half *gpu_data_;

    template <typename T> T &access(const Index2 &idx) const;

  public:
    Matrix(size_t dimH, size_t dimW);
    ~Matrix();
    void clear();

    void allocate_memory();
    void allocate_memory_gpu();

    void H2D();
    void D2H();

    void fill(const __half value);
    void initialize(const std::vector<__half> &data);
    size_t size() const { return dimH_ * dimW_; }
    Index2 vsize() const {
        return {static_cast<int32_t>(dimH_), static_cast<int32_t>(dimW_)};
    }
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

    friend void tensor_to_matrix_reshape(Tensor4D &TA, Matrix &TB);
    friend void tensor_to_matrix_reshape_const(const Tensor4D &TA, Matrix &TB);
    friend void matrix_to_tensor_reshape(Matrix &TA, Tensor4D &TB);
    friend void matrix_to_tensor_reshape_const(const Matrix &TA, Tensor4D &TB);
};
