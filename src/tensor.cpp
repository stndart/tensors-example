#include "tensor.h"
#include <cstring>

Tensor::Tensor(int rows, int cols) 
    : rows_(rows), cols_(cols), data_(nullptr) {}

Tensor::~Tensor() {
    delete[] data_;
}

void Tensor::allocate_memory() {
    if (data_ == nullptr) {
        data_ = new float[rows_ * cols_];
    }
}

void Tensor::initialize(const std::vector<float>& data) {
    if (data.size() != static_cast<size_t>(rows_ * cols_)) {
        throw std::runtime_error("Invalid data size");
    }
    allocate_memory();
    std::memcpy(data_, data.data(), rows_ * cols_ * sizeof(float));
}

void Tensor::print() const {
    for (int i = 0; i < rows_; ++i) {
        for (int j = 0; j < cols_; ++j) {
            std::cout << data_[i * cols_ + j] << " ";
        }
        std::cout << "\n";
    }
}

// CPU matrix multiplication implementation
void cpu_matrix_multiply(const Tensor& A, const Tensor& B, Tensor& C) {
    const int M = A.rows();
    const int N = B.cols();
    const int K = A.cols();
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A.data()[i * K + k] * B.data()[k * N + j];
            }
            C.data()[i * N + j] = sum;
        }
    }
}

// Unified matrix multiplication with CUDA fallback
void Tensor::multiply(const Tensor& A, const Tensor& B, Tensor& C) {
    if (A.cols() != B.rows()) {
        throw std::runtime_error("Matrix dimension mismatch");
    }
    
    C.allocate_memory();
    
    #ifdef USE_CUDA
    try {
        cuda_matrix_multiply(A, B, C);
        return;
    } catch (const std::exception& e) {
        std::cerr << "CUDA error: " << e.what() << "\n";
        std::cerr << "Falling back to CPU implementation\n";
    }
    #endif
    
    // CPU fallback
    cpu_matrix_multiply(A, B, C);
}