#include "matrix.h"
#include <iostream>

int main() {
    std::cout << "Running TensorMultiply application..." << std::endl;

    // Your application logic here
    Matrix A(2, 2);
    A.initialize({1.0f, 2.0f, 3.0f, 4.0f});

    Matrix B(2, 2);
    B.initialize({5.0f, 6.0f, 7.0f, 8.0f});

    Matrix C(2, 2);

#ifdef USE_CUDA
    std::cout << "Using CUDA acceleration" << std::endl;
    A.H2D();
    B.H2D();
    Matrix::gemm(A, B, C);
    C.D2H();
#else
    std::cout << "Using CPU implementation" << std::endl;
    Matrix::gemm(A, B, C);
#endif

    std::cout << "Result: ";
    C.print();

    return 0;
}