#include "matrix.h"
#include <cassert>
#include <iostream>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

bool MATRIX_TESTS_PASSED = true;

void clear_cuda_error() {
#ifdef USE_CUDA
    // Clear any prior error
    cudaError_t err0 = cudaGetLastError();
    if (err0 != cudaSuccess) {
        std::cerr << "pre-test CUDA error (ignored): "
                  << cudaGetErrorString(err0) << "\n";
    }
#endif
}

void test_matrix_creation() {
    bool pre_tests_passed = MATRIX_TESTS_PASSED;
    MATRIX_TESTS_PASSED = false;

    Matrix mat(2, 3);
    assert(mat.dimH() == 2);
    assert(mat.dimW() == 3);
    mat.allocate_memory();
    assert(mat.data() != nullptr);

    std::cout << "test_matrix_creation passed.\n";
    if (pre_tests_passed)
        MATRIX_TESTS_PASSED = true;
}

void test_initialize_and_print() {
    bool pre_tests_passed = MATRIX_TESTS_PASSED;
    MATRIX_TESTS_PASSED = false;

    Matrix mat(2, 2);
    std::vector<__half> values = {1.0f, 2.0f, 3.0f, 4.0f};
    mat.initialize(values);
    mat.print();

    std::cout << "\ntest_initialize_and_print passed.\n" << std::flush;
    if (pre_tests_passed)
        MATRIX_TESTS_PASSED = true;
}

void test_gpu_transfer() {
    bool pre_tests_passed = MATRIX_TESTS_PASSED;
    MATRIX_TESTS_PASSED = false;

    clear_cuda_error();

    Matrix mat(2, 2);
    std::vector<__half> values = {1.0f, 2.0f, 3.0f, 4.0f};
    mat.initialize(values);
    mat.H2D();

#ifdef USE_CUDA
    // This test specifically tests GPU
    cudaError_t err0 = cudaGetLastError();
    if (err0 != cudaSuccess) {
        std::cerr << "CUDA error during Matrix::gemm: "
                  << cudaGetErrorString(err0) << "\n";
        std::cout << "test_gemm_basic failed.\n" << std::flush;
        return;
    }
#endif

    std::cout << "test_gpu_transfer passed.\n" << std::flush;
    if (pre_tests_passed)
        MATRIX_TESTS_PASSED = true;
}

void test_gemm_basic() {
    bool pre_tests_passed = MATRIX_TESTS_PASSED;
    MATRIX_TESTS_PASSED = false;

    clear_cuda_error();

    Matrix A(2, 2);
    Matrix B(2, 2);
    Matrix C(2, 2);

    A.initialize({1.0f, 2.0f, 3.0f, 4.0f}); // 2x2
    B.initialize({5.0f, 6.0f, 7.0f, 8.0f}); // 2x2

    A.H2D();
    B.H2D();
    Matrix::gemm(A, B, C); // C = A x B
    C.D2H();

    const __half *result = C.data();

#ifdef USE_CUDA
    // This test specifically tests GPU
    cudaError_t err0 = cudaGetLastError();
    if (err0 != cudaSuccess) {
        std::cerr << "CUDA error during Matrix::gemm: "
                  << cudaGetErrorString(err0) << "\n";
        std::cout << "test_gemm_basic failed.\n" << std::flush;
        return;
    }
#endif

    // Expected: [1*5+2*7=19, 1*6+2*8=22, 3*5+4*7=43, 3*6+4*8=50]
    assert(result[0] == 19.0f);
    assert(result[1] == 22.0f);
    assert(result[2] == 43.0f);
    assert(result[3] == 50.0f);

    std::cout << "test_gemm_basic passed.\n" << std::flush;
    if (pre_tests_passed)
        MATRIX_TESTS_PASSED = true;
}

void test_invalid_init() {
    Matrix mat(2, 3);
    try {
        mat.initialize({1.0f, 2.0f}); // wrong size
        assert(false);                // Should not reach here
    } catch (const std::runtime_error &e) {
        std::cout << "test_invalid_init passed.\n" << std::flush;
    }
}