#pragma once

#include <cassert>
#include <iostream>
#include <tuple>
#include <vector>

#include "cuda/cuda_precision.h"
#include "matrix.h"

struct Index4 {
    union {
        struct {
            int32_t w, x, y, z;
        };
        int32_t dims[4];
    };

    int32_t &operator[](size_t i) {
        assert(i < 4);
        return dims[i];
    }
    const int32_t &operator[](size_t i) const {
        assert(i < 4);
        return dims[i];
    }

    // for things like `auto [a, b, c, d] = idx.as_tuple();`
    auto as_tuple() const { return std::make_tuple(w, x, y, z); }

    bool operator==(const Index4 &other) const {
        return w == other.w && x == other.x && y == other.y && z == other.z;
    }
    bool operator!=(const Index4 &other) const { return !operator==(other); }

    // work as all()
    bool operator<(const int32_t other) const {
        return w < other && x < other && y < other && z < other;
    }
    bool operator<=(const int32_t other) const {
        return w <= other && x <= other && y <= other && z <= other;
    }
    bool operator>(const int32_t other) const {
        return w > other && x > other && y > other && z > other;
    }
    bool operator>=(const int32_t other) const {
        return w >= other && x >= other && y >= other && z >= other;
    }
    bool operator<(const Index4 other) const {
        return w < other.w && x < other.x && y < other.y && z < other.z;
    }
    bool operator<=(const Index4 other) const {
        return w <= other.w && x <= other.x && y <= other.y && z <= other.z;
    }
    bool operator>(const Index4 other) const {
        return w > other.w && x > other.x && y > other.y && z > other.z;
    }
    bool operator>=(const Index4 other) const {
        return w >= other.w && x >= other.x && y >= other.y && z >= other.z;
    }

    friend std::ostream &operator<<(std::ostream &os, const Index4 &idx) {
        os << "[" << idx.w << ", " << idx.x << ", " << idx.y << ", " << idx.z
           << "]";
        return os;
    }
    friend std::istream &operator>>(std::istream &is, Index4 &idx) {
        is >> idx.w >> idx.x >> idx.y >> idx.z;
        return is;
    }
};

class Tensor4D {
  public:
    const static size_t NDIMS = 4;
    using Index = Index4;

  private:
    size_t dimW_;
    size_t dimX_;
    size_t dimY_;
    size_t dimZ_;

    __half *data_;
    __half *gpu_data_;

    template <typename T> T &access(const Index4 &idx) const;

  public:
    Tensor4D(size_t dimW, size_t dimX, size_t dimY, size_t dimZ);
    Tensor4D(Index4 dims);
    ~Tensor4D();
    void clear();

    void allocate_memory();
    void allocate_memory_gpu();

    void H2D();
    void D2H();

    void fill(const __half value);
    void initialize(const std::vector<__half> &data);
    size_t size() const { return dimW_ * dimX_ * dimY_ * dimZ_; }
    Index4 vsize() const {
        return {static_cast<int32_t>(dimW_), static_cast<int32_t>(dimX_),
                static_cast<int32_t>(dimY_), static_cast<int32_t>(dimZ_)};
    }
    void print(std::string name = "") const;

    static void im2col(const Tensor4D &TA, Matrix &TB, const size_t kH,
                       const size_t kW, const size_t H_pad, const size_t W_pad,
                       const size_t H_stride, const size_t W_stride);
    static void col2im(const Matrix &A, Tensor4D &B, const size_t kH,
                       const size_t kW, const size_t H_pad, const size_t W_pad,
                       const size_t H_stride, const size_t W_stride);

    friend void tensor_to_matrix_reshape(Tensor4D &TA, Matrix &TB);
    friend void tensor_to_matrix_reshape_const(const Tensor4D &TA, Matrix &TB);
    friend void matrix_to_tensor_reshape(Matrix &TA, Tensor4D &TB);
    friend void matrix_to_tensor_reshape_const(const Matrix &TA, Tensor4D &TB);

    // Tensor element-wise operations
    // static void multiply(const Tensor4D &A, const Tensor4D &B, Tensor4D
    // &C);
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
    __half &operator[](const Index4 &idx);
    const __half &operator[](const Index4 &idx) const;

    __half *gpu_data() { return gpu_data_; }
    const __half *gpu_data() const { return gpu_data_; }
};

void calculate_HW_out(const size_t H_in, const size_t W_in, const size_t kH,
                      const size_t kW, const size_t H_pad, const size_t W_pad,
                      const size_t H_stride, const size_t W_stride,
                      size_t &H_out, size_t &W_out);
