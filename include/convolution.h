#pragma once

#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

#include "matrix.h"
#include "tensor.h"

class Convolution {
  private:
    Tensor4D *kernel;
    Matrix *flatten_kernel;
    size_t H_pad, W_pad, H_stride, W_stride;

  public:
    Convolution(Tensor4D &Kernel, std::optional<size_t> H_pad = std::nullopt,
                std::optional<size_t> W_pad = std::nullopt,
                std::optional<size_t> H_stride = std::nullopt,
                std::optional<size_t> W_stride = std::nullopt);
    ~Convolution();

    void get_flatten_kernel();

    void forward(const Tensor4D &input, Tensor4D &output);
    void forward_simple(const Tensor4D &input, Tensor4D &output);
    // void backward(const )

    // Getters
    int in_channels() const { return kernel->dimW(); }
    int out_channels() const { return kernel->dimX(); }
    int rows() const { return kernel->dimY(); }
    int cols() const { return kernel->dimZ(); }
};

#ifdef USE_CUDA
void cuda_convolution_forward(const Tensor4D &A, const Tensor4D &B,
                              Tensor4D &C);
#endif