#include <iostream>

#include "convolution.h"
#include "matrix.h"
#include "tensor.h"

int main() {
    std::cout << "Running TensorMultiply application..." << std::endl;
#ifdef USE_CUDA
    std::cout << "Using CUDA acceleration" << std::endl;
#else
    std::cout << "Using CPU implementation" << std::endl;
#endif

    // try {
    // Input: 1x2x2x2 (B=1, C=2, H=2, W=2)
    Tensor4D input(1, 2, 2, 2);
    input.initialize({1.0f, 2.0f, 3.0f, 4.0f, // Channel 1

                      5.0f, 6.0f, 7.0f, 8.0f}); // Channel 2

    // Output gradient: 1x1x3x3 (B=1, O=1, H_out=3, W_out=3)
    Tensor4D output_grad(1, 1, 3, 3);
    output_grad.initialize(
        {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f});

    // Convolution: 1 output channel, 2 input channels, 2x2 kernel

    Tensor4D kernel(1, 2, 2, 2);
    kernel.initialize({0.5f, 1.0f, 1.5f, 2.0f,   // Channel 1 weights
                       0.5f, 1.0f, 1.5f, 2.0f}); // Channel 2 weights
    Convolution conv(kernel, 1, 1, 1, 1);

    // Gradient buffers
    Tensor4D kernel_grad(1, 2, 2, 2);
    Tensor4D input_grad(1, 2, 2, 2);

    // Compute gradients
    conv.backward_simple(input, output_grad, kernel_grad, input_grad);

    kernel_grad.print("kernel grad");
    input_grad.print("input grad");

    // } catch (const std::exception &e) {
    //     std::cout << "Exception caught: " << e.what() << std::endl;
    // }

    return 0;
}