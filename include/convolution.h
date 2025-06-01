#pragma once

#include <iostream>
#include <stdexcept>
#include <vector>

#include "tensor.h"

class Convolution {
  private:
    Tensor4D *kernel;

  public:
    Convolution(Tensor4D Kernel);
    ~Convolution();

    void forward(const Tensor4D &input, Tensor4D &output);
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