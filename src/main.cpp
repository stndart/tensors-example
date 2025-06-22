#include <iostream>

#include "convolution.h"
#include "matrix.h"
#include "tensor.h"
#include "max_pooling.h"
#include "avg_pooling.h"

int main() {
    std::cout << "Running TensorMultiply application..." << std::endl;
#ifdef USE_CUDA
    std::cout << "Using CUDA acceleration" << std::endl;
#else
    std::cout << "Using CPU implementation" << std::endl;
#endif

    // try {
    
    // Input: 1x1x3x3
    Tensor4D input(1, 1, 3, 3);
    input.initialize({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});

    // 2x2 pooling, stride=2, padding=1 (circular)
    MaxPooling pool(input.vsize(), PaddingMode::CIRCULAR_PADDING, 1, 1, 2, 2);
    Tensor4D output(1, 1, 2, 2);
    pool.forward(input, output);

    // Padded input (circular):
    /* 9,7,8,9,7
       3,1,2,3,1
       6,4,5,6,4
       9,7,8,9,7
       3,1,2,3,1 */
    // Output blocks:
    // TL: max(9,7,3,1) = 9
    // TR: max(8,9,2,3) = 9
    // BL: max(6,4,9,7) = 9
    // BR: max(5,6,8,9) = 9
    std::cout << (output[{0, 0, 0, 0}]) << "\n";
    std::cout << (output[{0, 0, 0, 1}]) << "\n";
    std::cout << (output[{0, 0, 1, 0}]) << "\n";
    std::cout << (output[{0, 0, 1, 1}]) << "\n";


    // } catch (const std::exception &e) {
    //     std::cout << "Exception caught: " << e.what() << std::endl;
    // }

    return 0;
}