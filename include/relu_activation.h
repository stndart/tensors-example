#pragma once

#include <optional>

#include "tensor.h"

class Relu {
  private:
    __half threshold;

  public:
    Relu(__half threshold = 0);
    ~Relu() {}

    void forward(const Tensor4D &input, Tensor4D &output) const;
    void backward(const Tensor4D &output_grad, const Tensor4D &output, Tensor4D &input_grad) const;
};

void cpu_relu(const Tensor4D &input, Tensor4D &output, __half threshold);
void cpu_relu_backward(const Tensor4D &output_grad, const Tensor4D &output, Tensor4D &input_grad, __half threshold);