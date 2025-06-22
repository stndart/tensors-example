
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "cuda/cuda_memory.cu"
#include "tensor.h"

Tensor4D::Tensor4D(size_t dimW, size_t dimX, size_t dimY, size_t dimZ)
    : dimW_(dimW), dimX_(dimX), dimY_(dimY), dimZ_(dimZ),
      axes_order({0, 1, 2, 3}), data_(nullptr), gpu_data_(nullptr) {}

Tensor4D::Tensor4D(Index4 dims)
    : dimW_(dims.w), dimX_(dims.x), dimY_(dims.y), dimZ_(dims.z),
      axes_order({0, 1, 2, 3}), data_(nullptr), gpu_data_(nullptr) {
    assert(dims >= 0);
}

Tensor4D::~Tensor4D() { clear(); }

void Tensor4D::clear() {
    if (data_ != nullptr)
        delete[] data_;

#ifdef USE_CUDA
    if (gpu_data_ != nullptr) {
        cuda_free(&gpu_data_);
    }
#endif
}

void Tensor4D::allocate_memory() {
    if (data_ == nullptr) {
        data_ = new __half[size()];
    }
}

void Tensor4D::allocate_memory_gpu() {
#ifdef USE_CUDA
    if (gpu_data_ == nullptr) {
        cuda_allocate(&gpu_data_, size());
    }
#endif
}

void Tensor4D::H2D() {
#ifdef USE_CUDA
    if (data_ != nullptr) {
        allocate_memory_gpu();
        cuda_h2d(data_, gpu_data_, size());
    }
#endif
}

void Tensor4D::D2H() {
#ifdef USE_CUDA
    if (gpu_data_ != nullptr) {
        allocate_memory();
        cuda_h2d(data_, gpu_data_, size());
    }
#endif
}

void Tensor4D::fill(const __half value) {
    for (size_t i = 0; i < size(); ++i)
        data_[i] = value;
}

void Tensor4D::initialize(const std::vector<__half> &data) {
    if (data.size() != size()) {
        throw std::runtime_error("Invalid data size");
    }
    allocate_memory();
    std::memcpy(data_, data.data(), size() * sizeof(__half));
}

void Tensor4D::print(std::string name) const {
    if (data_ == nullptr) {
        throw std::runtime_error("Print: memory is not allocated");
    }
    std::cout << "Tensor " << name << " dims are " << dimW() << "x" << dimX()
              << "x" << dimY() << "x" << dimZ() << "\n";
    std::cout << "Element size is " << sizeof(__half) << " bytes\n";

    for (int32_t d1 = 0; d1 < dimW(); ++d1) {
        for (int32_t d3 = 0; d3 < dimY(); ++d3) {
            for (int32_t d2 = 0; d2 < dimX(); ++d2) {
                for (int32_t d4 = 0; d4 < dimZ(); ++d4) {
                    float elem = (*this)[{d1, d2, d3, d4}];
                    std::cout << elem << " ";
                }
                if (d2 != dimX() - 1)
                    std::cout << "| ";
            }
            std::cout << "\n";
        }
        if (d1 != dimW() - 1) {
            int total_cols = dimY() * dimZ() + (dimY() - 1) * 1;
            for (int i = 0; i < total_cols * 2; ++i)
                std::cout << "-";
            std::cout << "\n";
        }
    }
}

void Tensor4D::set_axes_order(Index4 order) { axes_order = order; }
Index4 &Tensor4D::get_axes_order() { return axes_order; }

// changes order of last two axes, as if it was a matrix
void Tensor4D::transpose() {
    // swaps axes_order[2] and [3]
    __half C = axes_order[2];
    axes_order[2] = axes_order[3];
    axes_order[3] = C;
}

Index4 Tensor4D::real_index(const Index4 index) const {
    Index4 res;
    for (size_t i = 0; i < 4; ++i) {
        res[axes_order[i]] = index[i];
    }
    return res;
}

template <typename T> T &Tensor4D::access(const Index4 &idx) const {
    if (data_ == nullptr)
        throw std::runtime_error("Tensor is not allocated");
    if (idx.w < 0 || idx.x < 0 || idx.y < 0 || idx.z < 0)
        throw std::range_error("Tensor index error");
    if (idx.w >= dimW() || idx.x >= dimX() || idx.y >= dimY() ||
        idx.z >= dimZ())
        throw std::range_error("Tensor index error");

    Index4 ridx = real_index(idx);
    const size_t flat_index =
        ((ridx.w * dimX() + ridx.x) * dimY() + ridx.y) * dimZ() + ridx.z;
    return const_cast<T &>(data_[flat_index]);
}

__half &Tensor4D::operator[](const Index4 &idx) { return access<__half>(idx); }

const __half &Tensor4D::operator[](const Index4 &idx) const {
    return access<const __half>(idx);
}

// CPU implementations

void calculate_HW_out(const size_t H_in, const size_t W_in, const size_t kH,
                      const size_t kW, const size_t H_pad, const size_t W_pad,
                      const size_t H_stride, const size_t W_stride,
                      size_t &H_out, size_t &W_out) {
    H_out = std::floor((H_in + 2 * H_pad - kH) / H_stride) + 1;
    W_out = std::floor((W_in + 2 * W_pad - kW) / W_stride) + 1;
}

void cpu_tensor_im2col(const Tensor4D &TA, Matrix &TB, const size_t kH,
                       const size_t kW, const size_t H_pad, const size_t W_pad,
                       const size_t H_stride, const size_t W_stride) {
    const size_t B = TA.dimW();
    const size_t C = TA.dimX();
    const size_t H = TA.dimY();
    const size_t W = TA.dimZ();

    size_t H_out, W_out;
    calculate_HW_out(H, W, kH, kW, H_pad, W_pad, H_stride, W_stride, H_out,
                     W_out);
    const size_t N_patches = B * H_out * W_out;

    if (TB.dimH() != C * kH * kW || TB.dimW() != N_patches) {
        throw std::runtime_error("Matrix dimensions mismatch");
    }
    TB.allocate_memory();

    for (int32_t bi = 0; bi < B; ++bi) {
        for (size_t i = 0; i < H_out; ++i) {
            for (size_t j = 0; j < W_out; ++j) {
                int32_t ow_idx = bi * H_out * W_out + i * W_out + j;
                for (int32_t ci = 0; ci < C; ++ci) {
                    for (size_t ih = 0; ih < kH; ++ih) {
                        size_t oh_idx_base = ci * kW * kH + ih * kW;
                        int ih_idx = -H_pad + H_stride * i + ih;
                        for (size_t iw = 0; iw < kW; ++iw) {
                            int32_t oh_idx = oh_idx_base + iw;
                            size_t output_idx = oh_idx * N_patches + ow_idx;

                            int iw_idx = -W_pad + W_stride * j + iw;
                            if (ih_idx < 0 || iw_idx < 0 || ih_idx >= H ||
                                iw_idx >= W) {
                                TB[{oh_idx, ow_idx}] = 0;
                                continue;
                            }

                            size_t input_idx = bi * C * W * H + ci * W * H +
                                               ih_idx * W + iw_idx;
                            TB[{oh_idx, ow_idx}] = TA[{bi, ci, ih_idx, iw_idx}];
                        }
                    }
                }
            }
        }
    }
}

void cpu_tensor_col2im(const Matrix &TA, Tensor4D &TB, const size_t kH,
                       const size_t kW, const size_t H_pad, const size_t W_pad,
                       const size_t H_stride, const size_t W_stride) {
    const size_t B = TB.dimW();
    const size_t C = TB.dimX();
    const size_t H = TB.dimY();
    const size_t W = TB.dimZ();

    if (TA.dimH() != C * kH * kW) {
        throw std::runtime_error("Input matrix dimensions mismatch");
    }

    size_t H_out, W_out;
    calculate_HW_out(H, W, kH, kW, H_pad, W_pad, H_stride, W_stride, H_out,
                     W_out);
    const size_t N_patches = TA.dimW();

    TB.allocate_memory();
    TB.fill(0);

    for (int32_t bi = 0; bi < B; ++bi) {
        for (size_t i = 0; i < H_out; ++i) {
            for (size_t j = 0; j < W_out; ++j) {
                for (int32_t ci = 0; ci < C; ++ci) {
                    for (size_t ih = 0; ih < kH; ++ih) {
                        int ih_idx = -H_pad + H_stride * i + ih;
                        for (size_t iw = 0; iw < kW; ++iw) {
                            int iw_idx = -W_pad + W_stride * j + iw;
                            if (ih_idx < 0 || iw_idx < 0 || ih_idx >= H ||
                                iw_idx >= W) {
                                continue;
                            }

                            const int32_t col_idx =
                                bi * H_out * W_out + i * W_out + j;
                            const int32_t row_idx = ci * kH * kW + ih * kW + iw;

                            // std::cout << col_idx << "x" << row_idx << "
                            // >> "
                            //           << bi << "x" << ci << "x" << ih_idx
                            //           << "x"
                            //           << iw_idx << "\n";

                            TB[{bi, ci, ih_idx, iw_idx}] +=
                                TA[{row_idx, col_idx}];
                        }
                    }
                }
            }
        }
    }
}

void cpu_tensor_add(const Tensor4D &A, const Tensor4D &B, Tensor4D &C) {
    for (size_t i = 0; i < A.size(); ++i) {
        C.data()[i] = A.data()[i] + B.data()[i];
    }
}
void cpu_tensor_add(const Tensor4D &A, const __half B, Tensor4D &C) {
    for (size_t i = 0; i < A.size(); ++i) {
        C.data()[i] = A.data()[i] + B;
    }
}
void cpu_tensor_scale(const Tensor4D &A, const __half B, Tensor4D &C) {
    for (size_t i = 0; i < A.size(); ++i) {
        C.data()[i] = A.data()[i] * B;
    }
}

void cpu_tensor_sum(const Tensor4D &A, const size_t index, Tensor4D &C) {
    C.fill(0.0f);
    for (int32_t i = 0; i < A.dimW(); ++i)
        for (int32_t j = 0; j < A.dimX(); ++j)
            for (int32_t k = 0; k < A.dimY(); ++k)
                for (int32_t m = 0; m < A.dimZ(); ++m) {
                    Index4 idx{i, j, k, m};
                    idx[index] = 0;
                    C[idx] += A[{i, j, k, m}];
                }
}
void cpu_tensor_max(const Tensor4D &A, const size_t index, Tensor4D &C) {
    C.fill(0.0f);
    for (int32_t i = 0; i < A.dimW(); ++i)
        for (int32_t j = 0; j < A.dimX(); ++j)
            for (int32_t k = 0; k < A.dimY(); ++k)
                for (int32_t m = 0; m < A.dimZ(); ++m) {
                    Index4 idx{i, j, k, m};
                    idx[index] = 0;
                    C[idx] = std::max(C[idx], A[{i, j, k, m}]);
                }
}
void cpu_tensor_mean(const Tensor4D &A, const size_t index, Tensor4D &C) {
    cpu_tensor_sum(A, index, C);
    __half scale = 1.0f / A.vsize()[index];
    cpu_tensor_scale(C, scale, C); // inplace scale
}

// Unified tensor operations with CPU fallback

void Tensor4D::im2col(const Tensor4D &A, Matrix &B, const size_t kH,
                      const size_t kW, const size_t H_pad, const size_t W_pad,
                      const size_t H_stride, const size_t W_stride) {
#ifdef USE_CUDA
    try {
        cuda_tensor_im2col(A, B, kH, kW, H_pad, W_pad, H_stride, W_stride);
        return;
    } catch (const std::exception &e) {
        std::cerr << "CUDA error: " << e.what();
        std::cerr << "Falling back to CPU\n";
    }
#endif

    // CPU fallback
    cpu_tensor_im2col(A, B, kH, kW, H_pad, W_pad, H_stride, W_stride);
}

void Tensor4D::col2im(const Matrix &A, Tensor4D &B, const size_t kH,
                      const size_t kW, const size_t H_pad, const size_t W_pad,
                      const size_t H_stride, const size_t W_stride) {
#ifdef USE_CUDA
    try {
        cuda_tensor_col2im(A, B, kH, kW, H_pad, W_pad, H_stride, W_stride);
        return;
    } catch (const std::exception &e) {
        std::cerr << "CUDA error: " << e.what();
        std::cerr << "Falling back to CPU\n";
    }
#endif

    // CPU fallback
    B.allocate_memory();
    cpu_tensor_col2im(A, B, kH, kW, H_pad, W_pad, H_stride, W_stride);
}

#ifdef USE_CUDA
#define operation_macro(name, second_arg)                                      \
    try {                                                                      \
        cuda_tensor_##name(A, second_arg, C);                                  \
        return;                                                                \
    } catch (const std::exception &e) {                                        \
        std::cerr << "CUDA error: " << e.what();                               \
        std::cerr << "Falling back to CPU\n";                                  \
    }                                                                          \
    C.allocate_memory();                                                       \
    cpu_tensor_##name(A, second_arg, C);
#else
#define operation_macro(name, second_arg)                                      \
    C.allocate_memory();                                                       \
    cpu_tensor_##name(A, second_arg, C);
#endif

// Tensor element-wise operations
void Tensor4D::add(const Tensor4D &A, const Tensor4D &B, Tensor4D &C) {
    if (A.vsize() != B.vsize())
        throw std::runtime_error("Tensor dimensions A & B mismatch");
    if (A.vsize() != C.vsize())
        throw std::runtime_error("Tensor dimensions A & C mismatch");
    if (C.vsize() != B.vsize())
        throw std::runtime_error("Tensor dimensions B & C mismatch");
    operation_macro(add, B)
}
void Tensor4D::add(const Tensor4D &A, const __half B, Tensor4D &C) {
    if (A.vsize() != C.vsize())
        throw std::runtime_error("Tensor dimensions A & C mismatch");
    operation_macro(add, B)
}
void Tensor4D::scale(const Tensor4D &A, const __half B, Tensor4D &C) {
    if (A.vsize() != C.vsize())
        throw std::runtime_error("Tensor dimensions A & C mismatch");
    operation_macro(scale, B)
}

// Reductions
void Tensor4D::sum(const Tensor4D &A, const size_t index, Tensor4D &C) {
    Index4 Asize = A.vsize();
    Asize[index] = 1;
    if (Asize != C.vsize())
        throw std::runtime_error("Tensor dimensions A & C mismatch");

    operation_macro(sum, index)
}
void Tensor4D::max(const Tensor4D &A, const size_t index, Tensor4D &C) {
    Index4 Asize = A.vsize();
    Asize[index] = 1;
    if (Asize != C.vsize())
        throw std::runtime_error("Tensor dimensions A & C mismatch");

    operation_macro(max, index)
}
void Tensor4D::mean(const Tensor4D &A, const size_t index, Tensor4D &C) {
    Index4 Asize = A.vsize();
    Asize[index] = 1;
    if (Asize != C.vsize())
        throw std::runtime_error("Tensor dimensions A & C mismatch");

    operation_macro(mean, index)
}