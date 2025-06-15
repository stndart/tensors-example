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

    try {
        // Input: 1x1x2x2 tensor
        Tensor4D input(1, 1, 2, 2);
        input.initialize({1.0f, 2.0f, 3.0f, 4.0f});

        // Identity kernel (2x2): [[1,0],[0,1]]
        Tensor4D kernel(1, 1, 2, 2);
        kernel.initialize({1.0f, 0.0f, 0.0f, 1.0f});
        Convolution conv(kernel, 0, 0, 1, 1);

        Tensor4D output_im2col(1, 1, 1, 1), output_direct(1, 1, 1, 1);
        output_im2col.allocate_memory();
        output_direct.allocate_memory();

        conv.forward(input, output_im2col);
        conv.forward_simple(input, output_direct);

        // Should compute: 1*1 + 2*0 + 3*0 + 4*1 = 5
        std::cout << "Compare " << output_im2col.data()[0] << " to " << 5.0f
                  << "\n";
        std::cout << "Compare " << output_direct.data()[0] << " to " << 5.0f
                  << "\n";
    } catch (const std::exception &e) {
        std::cout << "Exception caught: " << e.what() << std::endl;
    }

    return 0;
}