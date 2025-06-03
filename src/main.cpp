#include "matrix_tests.h"
#include <iostream>

int main() {
#ifdef USE_CUDA
    std::cout << "CUDA support enabled\n" << std::flush;
#endif

    test_matrix_creation();
    test_initialize_and_print();
    test_gpu_transfer();
    clear_cuda_error();
    test_gemm_basic();
    test_invalid_init();

    if (MATRIX_TESTS_PASSED)
        std::cout << "All tests passed!\n" << std::flush;

    return 0;
}
