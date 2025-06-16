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
    // Input: 1x1x3x3 (B=1, C=1, H=3, W=3)
    Tensor4D input(1, 1, 3, 3);
    input.initialize({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});

    // Output gradient: 1x1x2x2 (B=1, O=1, H_out=2, W_out=2)
    Tensor4D output_grad(1, 1, 2, 2);
    output_grad.initialize({1.0f, 1.0f, 1.0f, 1.0f});

    // Initialize convolution: 1 output channel, 2x2 kernel
    Tensor4D kernel(1, 1, 2, 2);
    kernel.initialize({1.0f, 0.0f, 0.0f, 1.0f}); // Identity-like kernel
    Convolution conv(kernel, 0, 0, 1, 1);

    // Gradient buffers
    Tensor4D kernel_grad(1, 1, 2, 2);
    Tensor4D input_grad(1, 1, 3, 3);

    // Compute gradients
    conv.backward_simple(input, output_grad, kernel_grad, input_grad);

    // Verify kernel gradient
    // Expected:
    //   top-left: 1*1 + 2*1 + 4*1 + 5*1 = 12
    //   top-right: 2*1 + 3*1 + 5*1 + 6*1 = 16
    //   bottom-left: 4*1 + 5*1 + 7*1 + 8*1 = 24
    //   bottom-right: 5*1 + 6*1 + 8*1 + 9*1 = 28
    kernel_grad.print("kernel grad");

    // Verify input gradient
    // Expected:
    //   [1, 1+1, 1,
    //    1+1, 1+1+1+1, 1+1,
    //    1, 1+1, 1]
    input_grad.print("input grad");
    // } catch (const std::exception &e) {
    //     std::cout << "Exception caught: " << e.what() << std::endl;
    // }

    return 0;
}