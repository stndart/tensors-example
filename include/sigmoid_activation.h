#pragma once

#include <optional>

#include "tensor.h"

class Sigmoid {
  private:
    __half threshold;

  public:
    Sigmoid(__half threshold = 0);
    ~Sigmoid() {}

    void forward(const Tensor4D &input, Tensor4D &output) const;
    void backward(const Tensor4D &output_grad, const Tensor4D &output, Tensor4D &input_grad) const;
};

void cpu_sigmoid(const Tensor4D &input, Tensor4D &output, __half threshold);
void cpu_sigmoid_backward(const Tensor4D &output_grad, const Tensor4D &output, Tensor4D &input_grad, __half threshold);