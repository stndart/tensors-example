#include "relu_activation.h"

Relu::Relu(__half threshold) : threshold(threshold) {}

void Relu::forward(const Tensor4D &input, Tensor4D &output) const {
    if (output.vsize() != input.vsize())
        throw std::runtime_error("Forward: output dimensions mismatch");

#ifdef USE_CUDA
    try {
        cuda_relu(input, output, threshold);
        return;
    } catch (const std::exception &e) {
        std::cerr << "CUDA error: " << e.what();
        std::cerr << "Falling back to CPU\n";
    }
#endif

    // CPU fallback
    output.allocate_memory();
    cpu_relu(input, output, threshold);
}

void Relu::backward(const Tensor4D &output_grad, const Tensor4D &output, Tensor4D &input_grad) const {

    if (output_grad.vsize() != input_grad.vsize())
        throw std::runtime_error(
            "Backward: input gradient dimensions mismatch");

#ifdef USE_CUDA
    try {
        cuda_relu_backward(output_grad, output, input_grad, threshold);
        return;
    } catch (const std::exception &e) {
        std::cerr << "CUDA error: " << e.what();
        std::cerr << "Falling back to CPU\n";
    }
#endif

    // CPU fallback
    input_grad.allocate_memory();
    cpu_relu_backward(output_grad, output, input_grad, threshold);
}

void cpu_relu(const Tensor4D &input, Tensor4D &output, __half threshold) {
    for (size_t i = 0; i < input.size(); ++i)
        output.data()[i] = max(input.data()[i], threshold);
}

void cpu_relu_backward(const Tensor4D &output_grad, const Tensor4D &output, Tensor4D &input_grad, __half threshold) {
    for (size_t i = 0; i < output_grad.size(); ++i)
        input_grad.data()[i] = (output.data()[i] > threshold) ? output_grad.data()[i] : 0;
}