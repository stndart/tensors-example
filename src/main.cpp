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
    
    Tensor4D input(1, 1, 3, 3);
    input.initialize({1.0f, 2.0f, 3.0f,
                     4.0f, 5.0f, 6.0f,
                     7.0f, 8.0f, 9.0f});
    
    // Expected: Reflected values
    /* 9,8,7,8,9
       6,5,4,5,6
       3,2,1,2,3
       6,5,4,5,6
       9,8,7,8,9 */
    std::cout << padded(input, {0,0,-1,-1}, PaddingMode::REFLECTION_PADDING) << "\n";
    std::cout << padded(input, {0,0,-1,0}, PaddingMode::REFLECTION_PADDING) << "\n";
    std::cout << padded(input, {0,0,0,-1}, PaddingMode::REFLECTION_PADDING) << "\n";
    std::cout << padded(input, {0,0,3,3}, PaddingMode::REFLECTION_PADDING) << "\n";


    // } catch (const std::exception &e) {
    //     std::cout << "Exception caught: " << e.what() << std::endl;
    // }

    return 0;
}