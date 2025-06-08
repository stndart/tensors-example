#include "matrix.h"
#include "cuda_memory.cu"
#include <stdexcept>
#include <string>

Matrix::Matrix(size_t dimH, size_t dimW)
    : dimH_(dimH), dimW_(dimW), data_(nullptr), gpu_data_(nullptr) {}

Matrix::~Matrix() {
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

void Matrix::initialize(const std::vector<__half> &data) {
    if (data.size() != size()) {
        throw std::runtime_error("Invalid data size");
    }
    allocate_memory();
    std::memcpy(data_, data.data(), size() * sizeof(__half));
}

void Matrix::print() const {
    if (data_ == nullptr) {
        throw std::runtime_error("Print: memory is not allocated");
    }
    std::cout << "Element size is " << sizeof(__half) << "\n";

    for (int i = 0; i < dimH_; ++i) {
        for (int j = 0; j < dimW_; ++j) {
            float elem = data_[i * dimW_ + j];
            std::cout << elem << " ";
        }
    }
}

// CPU matrix multiplication implementation
void cpu_matrix_multiply(const Matrix &A, const Matrix &B, Matrix &C) {
    const size_t M = A.dimH();
    const size_t K = A.dimW();
    const size_t N = B.dimW();

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            __half sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A.data()[i * K + k] * B.data()[k * N + j];
            }
            C.data()[i * N + j] = sum;
        }
    }
}

// Unified matrix multiplication with CUDA fallback
void Matrix::gemm(const Matrix &A, const Matrix &B, Matrix &C) {
    if (A.dimW() != B.dimH()) {
        throw std::runtime_error("Matrix dimension mismatch");
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