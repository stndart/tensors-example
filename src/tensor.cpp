#include "tensor.h"
#include "cuda_memory.h"
#include <cstring>

Tensor4D::Tensor4D(size_t dimW, size_t dimX, size_t dimY, size_t dimZ)
    : dimW_(dimW), dimX_(dimX), dimY_(dimY), dimZ_(dimZ), data_(nullptr),
      gpu_data_(nullptr) {}

Tensor4D::~Tensor4D() {
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

void Tensor4D::initialize(const std::vector<__half> &data) {
    if (data.size() != size()) {
        throw std::runtime_error("Invalid data size");
    }
    allocate_memory();
    std::memcpy(data_, data.data(), size() * sizeof(__half));
}

void Tensor4D::print() const {
    if (data_ == nullptr) {
        throw std::runtime_error("Print: memory is not allocated");
    }
    std::cout << "Element size is " << sizeof(__half) << "\n";

    for (int d1 = 0; d1 < dimW_; ++d1) {
        for (int d3 = 0; d3 < dimX_; ++d3) {
            for (int d2 = 0; d2 < dimY_; ++d2) {
                for (int d4 = 0; d4 < dimZ_; ++d4) {
                    float elem = data_[d1 * dimY_ * dimX_ * dimZ_ +
                                       d2 * dimX_ * dimZ_ + d3 * dimZ_ + d4];
                    std::cout << elem << " ";
                }
                if (d2 != dimY_ - 1)
                    std::cout << "| ";
            }
            std::cout << "\n";
        }
        if (d1 != dimW_ - 1) {
            int total_cols = dimY_ * dimZ_ + (dimY_ - 1) * 1;
            for (int i = 0; i < total_cols * 2; ++i)
                std::cout << "-";
            std::cout << "\n";
        }
    }
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
    const size_t N_patches = B * C * H_out * W_out;

    if (TB.dimW() != C * kH * kW || TB.dimH() != N_patches) {
        throw std::runtime_error("Matrix dimensions mismatch");
    }

    for (size_t bi = 0; bi < B; ++bi) {
        for (size_t ci = 0; ci < C; ++ci) {
            for (size_t i = 0; i < H_out; ++i) {
                for (size_t j = 0; j < W_out; ++j) {
                    size_t H_shift = -H_pad + H_stride * i;
                    size_t W_shift = -W_pad + W_stride * j;
                    size_t input_shift =
                        bi * C * W * H + ci * W * H + H_shift * W + W_shift;
                    size_t patch = bi * C * H_out * W_out + ci * H_out * W_out +
                                   i * W_out + j;
                    for (size_t iH = 0; iH < kH; ++iH) {
                        for (size_t iW = 0; iW < kW; ++iW) {
                            // zero padding from top and bottom
                            size_t W_index = W_shift + iW;
                            size_t H_index = H_shift + iH;
                            if (H_index < 0 || H_index >= H) {
                                TB.data()[patch + iH * kW + iW] = 0;
                                continue;
                            }
                            // zero padding from sides
                            if (W_index < 0 || W_index >= W) {
                                TB.data()[patch + iH * kW + iW] = 0;
                                continue;
                            }
                            TB.data()[patch + iH * kW + iW] =
                                TA.data()[input_shift + H_index * W + W_index];
                        }
                    }
                }
            }
        }
    }
}

void cpu_tensor_col2im(const Matrix &A, Tensor4D &B) {}

void cpu_tensor_multiply(const Tensor4D &A, const Tensor4D &B, Tensor4D &C) {}
void cpu_tensor_add(const Tensor4D &A, const Tensor4D &B, Tensor4D &C) {}
void cpu_tensor_add(const Tensor4D &A, const __half B, Tensor4D &C) {}
void cpu_tensor_scale(const Tensor4D &A, const __half B, Tensor4D &C) {}

void cpu_tensor_sum(const Tensor4D &A, const size_t index, Tensor4D &C) {}
void cpu_tensor_max(const Tensor4D &A, const size_t index, Tensor4D &C) {}
void cpu_tensor_mean(const Tensor4D &A, const size_t index, Tensor4D &C) {}

// Unified tensor operations with CPU fallback

void Tensor4D::im2col(const Tensor4D &A, Matrix &B) {
#ifdef USE_CUDA
    try {
        cuda_tensor_im2col(A, B);
        return;
    } catch (const std::exception &e) {
        std::cerr << "CUDA error: " << e.what();
        std::cerr << "Falling back to CPU\n";
    }
#endif

    // CPU fallback
    B.allocate_memory();
    cpu_tensor_im2col(A, B);
}

void Tensor4D::col2im(const Matrix &A, Tensor4D &B) {
#ifdef USE_CUDA
    try {
        cuda_tensor_col2im(A, B);
        return;
    } catch (const std::exception &e) {
        std::cerr << "CUDA error: " << e.what();
        std::cerr << "Falling back to CPU\n";
    }
#endif

    // CPU fallback
    B.allocate_memory();
    cpu_tensor_col2im(A, B);
}

#ifdef USE_CUDA
#define operation_macro(name, second_arg)                                      \
    try {                                                                      \
        cuda_tensor_multiply(A, second_arg, C);                                \
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
void Tensor4D::multiply(const Tensor4D &A, const Tensor4D &B, Tensor4D &C) {
    // TODO: some checks
    operation_macro(multiply, B)
}
void Tensor4D::add(const Tensor4D &A, const Tensor4D &B, Tensor4D &C) {
    if (A.dimW() != B.dimW() || A.dimX() != B.dimX() || A.dimY() != B.dimY() ||
        A.dimZ() != B.dimZ()) {
        throw std::runtime_error("Tensor dimensions A & B mismatch");
    }
    if (A.dimW() != C.dimW() || A.dimX() != C.dimX() || A.dimY() != C.dimY() ||
        A.dimZ() != C.dimZ()) {
        throw std::runtime_error("Tensor dimensions A & C mismatch");
    }
    operation_macro(add, B)
}
void Tensor4D::add(const Tensor4D &A, const __half B, Tensor4D &C) {
    operation_macro(add, B)
}
void Tensor4D::scale(const Tensor4D &A, const __half B, Tensor4D &C) {
    operation_macro(scale, B)
}

// Reductions
void Tensor4D::sum(const Tensor4D &A, const size_t index, Tensor4D &C) {
    operation_macro(sum, index)
}
void Tensor4D::max(const Tensor4D &A, const size_t index, Tensor4D &C) {
    operation_macro(max, index)
}
void Tensor4D::mean(const Tensor4D &A, const size_t index, Tensor4D &C) {
    operation_macro(mean, index)
}