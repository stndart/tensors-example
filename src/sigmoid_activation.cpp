#include "sigmoid_activation.h"

Sigmoid::Sigmoid(__half threshold) : threshold(threshold) {}

void Sigmoid::forward(const Tensor4D &input, Tensor4D &output) const {
    if (output.vsize() != input.vsize())
        throw std::runtime_error("Forward: output dimensions mismatch");

#ifdef USE_CUDA
    try {
        cuda_sigmoid(input, output, threshold);
        return;
    } catch (const std::exception &e) {
        std::cerr << "CUDA error: " << e.what();
        std::cerr << "Falling back to CPU\n";
    }
#endif

    // CPU fallback
    output.allocate_memory();
    cpu_sigmoid(input, output, threshold);
}

void Sigmoid::backward(const Tensor4D &output_grad, const Tensor4D &output, Tensor4D &input_grad) const {

    if (output_grad.vsize() != input_grad.vsize())
        throw std::runtime_error(
            "Backward: input gradient dimensions mismatch");
    if (output_grad.vsize() != output.vsize())
        throw std::runtime_error(
            "Backward: output gradient dimensions mismatch");

#ifdef USE_CUDA
    try {
        cuda_sigmoid_backward(output_grad, output, input_grad, threshold);
        return;
    } catch (const std::exception &e) {
        std::cerr << "CUDA error: " << e.what();
        std::cerr << "Falling back to CPU\n";
    }
#endif

    // CPU fallback
    input_grad.allocate_memory();
    cpu_sigmoid_backward(output_grad, output, input_grad, threshold);
}

void cpu_sigmoid(const Tensor4D &input, Tensor4D &output, __half threshold) {
    for (size_t i = 0; i < input.size(); ++i)
        output.data()[i] = 1.0f / (1.0f + exp(-input.data()[i]));
}

void cpu_sigmoid_backward(const Tensor4D &output_grad, const Tensor4D &output, Tensor4D &input_grad, __half threshold) {
    for (size_t i = 0; i < output_grad.size(); ++i)
        input_grad.data()[i] = output.data()[i] * (1 - output.data()[i]) * output_grad.data()[i];
}