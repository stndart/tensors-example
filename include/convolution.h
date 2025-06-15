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

    static Matrix *get_flatten_kernel(const Tensor4D &kernel);

    void forward(const Tensor4D &input, Tensor4D &output) const;
    void forward_simple(const Tensor4D &input, Tensor4D &output) const;
    void backward(const Tensor4D &input, const Tensor4D &output_gradient,
                  Tensor4D &kernel_gradient, Tensor4D &input_gradient) const;
    void apply_gradient_step(const Tensor4D &kernel_gradient_step);

    // Getters
    int in_channels() const { return kernel->dimW(); }
    int out_channels() const { return kernel->dimX(); }
    int rows() const { return kernel->dimY(); }
    int cols() const { return kernel->dimZ(); }
};

#ifdef USE_CUDA
void cuda_convolution(const Tensor4D &input, const Tensor4D &kernel,
                      Tensor4D &output, const size_t H_pad, const size_t W_pad,
                      const size_t H_stride, const size_t W_stride,
                      const Matrix *flatten_kernel = nullptr);
#endif