#pragma once

#include <optional>

#include "tensor.h"

class Softmax {
  public:
    Softmax() {}
    ~Softmax() {}

    void forward(const Tensor4D &input, Tensor4D &output) const;
    void backward(const Tensor4D &output_grad, const Tensor4D &output, Tensor4D &input_grad) const;
};

void cpu_softmax(const Tensor4D &input, Tensor4D &output);
void cpu_softmax_backward(const Tensor4D &output_grad, const Tensor4D &output, Tensor4D &input_grad);