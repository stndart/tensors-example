#include <cstring>
#include <stdexcept>
#include <string>

#include "cuda_memory.cu"
#include "matrix.h"

Matrix::Matrix(size_t dimH, size_t dimW)
    : dimH_(dimH), dimW_(dimW), data_(nullptr), gpu_data_(nullptr) {}

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

    std::cout << "Matrix " << name << " dims are " << dimH_ << "x" << dimW_
              << "\n";
    std::cout << "Element size is " << sizeof(__half) << " bytes\n";

    for (int i = 0; i < dimH_; ++i) {
        for (int j = 0; j < dimW_; ++j) {
            float elem = data_[i * dimW_ + j];
            std::cout << elem << " ";
        }
        std::cout << "\n";
    }
}

template <typename T> T &Matrix::access(const Index2 &idx) const {
    if (data_ == nullptr)
        throw std::runtime_error("Matrix data_ is not allocated");
    if (idx.x >= dimH_ || idx.y >= dimW_)
        throw std::range_error("Matrix index error");

    const size_t flat_index = idx.x * dimW_ + idx.y;
    return const_cast<T &>(data_[flat_index]);
}

__half &Matrix::operator[](const Index2 &idx) { return access<__half>(idx); }

const __half &Matrix::operator[](const Index2 &idx) const {
    return access<const __half>(idx);
}

// CPU matrix multiplication implementation
void cpu_matrix_multiply(const Matrix &A, const Matrix &B, Matrix &C) {
    const size_t M = A.dimH();
    const size_t K = A.dimW();
    const size_t N = B.dimW();

    // std::cout << "A = [" << A.dimH() << "x" << A.dimW() << "]\n";
    // std::cout << "B = [" << B.dimH() << "x" << B.dimW() << "]\n";
    // std::cout << "C = [" << C.dimH() << "x" << C.dimW() << "]\n";
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            __half sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[{i, k}] * B[{k, j}];
            }
            // std::cout << "ixj = " << i << "x" << j << "\n";
            C[{i, j}] = sum;
        }
    }
    // std::cout << "gemm\n";
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
    cpu_matrix_multiply(A, B, C);
}