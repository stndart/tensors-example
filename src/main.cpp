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

    Tensor4D input(2, 3, 1, 4);
    // input.initialize({ 1.0f, 2.0f, 3.0f, 4.0f });
    input.allocate_memory();
    input.fill(0);
    // swap H and W axes
    input.print();
    input.set_axes_order({2, 3, 1, 0});
    input.print();


    // } catch (const std::exception &e) {
    //     std::cout << "Exception caught: " << e.what() << std::endl;
    // }

    return 0;
}