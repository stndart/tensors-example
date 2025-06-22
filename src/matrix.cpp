#include <cstring>
#include <stdexcept>
#include <string>

#include "cpu/cpu_matrixes.h"
#include "cuda/cuda_memory.cu"
#include "matrix.h"

Matrix::Matrix(size_t dimH, size_t dimW)
    : dimH_(dimH), dimW_(dimW), axes_order({0, 1}), data_(nullptr),
      gpu_data_(nullptr) {}

Matrix::~Matrix() { clear(); }

void Matrix::clear() {
    if (data_ != nullptr)
        delete[] data_;

#ifdef USE_CUDA
    if (gpu_data_ != nullptr) {
        cuda_free(&gpu_data_);
    }
#endif
}

void Matrix::allocate_memory() {
    if (data_ == nullptr) {
        data_ = new __half[size()];
    }
}

void Matrix::allocate_memory_gpu() {
#ifdef USE_CUDA
    if (gpu_data_ == nullptr) {
        cuda_allocate(&gpu_data_, size());
    }
#endif
}

void Matrix::H2D() {
#ifdef USE_CUDA
    if (data_ != nullptr) {
        allocate_memory_gpu();
        cuda_h2d(data_, gpu_data_, size());
    }
#endif
}

void Matrix::D2H() {
#ifdef USE_CUDA
    if (gpu_data_ != nullptr) {
        allocate_memory();
        cuda_d2h(data_, gpu_data_, size());
    }
#endif
}

void Matrix::fill(const __half value) {
    for (size_t i = 0; i < size(); ++i)
        data_[i] = value;
}

void Matrix::initialize(const std::vector<__half> &data) {
    if (data.size() != size()) {
        throw std::runtime_error("Invalid data size");
    }
    allocate_memory();
    std::memcpy(data_, data.data(), size() * sizeof(__half));
}

void Matrix::print(std::string name) const {
    if (data_ == nullptr) {
        throw std::runtime_error("Print: memory is not allocated");
    }

    std::cout << "Matrix " << name << " dims are " << dimH() << "x" << dimW()
              << "\n";
    std::cout << "Element size is " << sizeof(__half) << " bytes\n";

    for (int32_t i = 0; i < dimH(); ++i) {
        for (int32_t j = 0; j < dimW(); ++j) {
            float elem = (*this)[{i, j}];
            std::cout << elem << " ";
        }
        std::cout << "\n";
    }
}

void Matrix::set_axes_order(Index2 order) { axes_order = order; }
Index2 &Matrix::get_axes_order() { return axes_order; }

void Matrix::transpose() {
    __half C = axes_order[0];
    axes_order[0] = axes_order[1];
    axes_order[1] = C;
}

Index2 Matrix::real_index(const Index2 index) const {
    Index2 res;
    for (size_t i = 0; i < 2; ++i) {
        res[axes_order[i]] = index[i];
    }
    return res;
}

template <typename T> T &Matrix::access(const Index2 &idx) const {
    if (data_ == nullptr)
        throw std::runtime_error("Matrix data_ is not allocated");
    if (idx.x < 0 || idx.y < 0)
        throw std::range_error("Matrix index error");
    if (idx.x >= dimH() || idx.y >= dimW())
        throw std::range_error("Matrix index error");

    Index2 ridx = real_index(idx);
    const size_t flat_index = ridx.x * dimW() + ridx.y;
    return const_cast<T &>(data_[flat_index]);
}

__half &Matrix::operator[](const Index2 &idx) { return access<__half>(idx); }

const __half &Matrix::operator[](const Index2 &idx) const {
    return access<const __half>(idx);
}

// Unified matrix multiplication with CUDA fallback
void Matrix::gemm(const Matrix &A, const Matrix &B, Matrix &C) {
    if (A.dimW() != B.dimH()) {
        throw std::runtime_error("gemm: Matrix A to Bdimension mismatch");
    }
    if (A.dimH() != C.dimH()) {
        throw std::runtime_error("gemm: Matrix C to A dimension mismatch");
    }
    if (B.dimW() != C.dimW()) {
        throw std::runtime_error("gemm: Matrix C to B dimension mismatch");
    }

#ifdef USE_CUDA
    try {
        cuda_matrix_gemm(A, B, C);
        return;
    } catch (const std::exception &e) {
        std::cerr << "CUDA error: " << e.what() << std::endl << std::flush;
        std::cerr << "Falling back to CPU\n";
    }
#endif

    // CPU fallback
    C.allocate_memory();
    cpu_matrix_gemm(A, B, C);
}