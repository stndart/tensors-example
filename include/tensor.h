#pragma once

#include <vector>
#include <iostream>
#include <stdexcept>

class Tensor {
public:
    Tensor(int rows, int cols);
    ~Tensor();

    void allocate_memory();
    void initialize(const std::vector<float>& data);
    void print() const;

    // Matrix multiplication
    static void multiply(const Tensor& A, const Tensor& B, Tensor& C);

    // Getters
    int rows() const { return rows_; }
    int cols() const { return cols_; }
    float* data() { return data_; }
    const float* data() const { return data_; }

private:
    int rows_;
    int cols_;
    float* data_;
};

// CUDA matrix multiplication declaration
#ifdef USE_CUDA
void cuda_matrix_multiply(const Tensor& A, const Tensor& B, Tensor& C);
#endif