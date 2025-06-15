#include "cpu/cpu_matrixes.h"
#include "matrix.h"

// CPU matrix multiplication implementation
void cpu_matrix_gemm(const Matrix &A, const Matrix &B, Matrix &C) {
    const size_t M = A.dimH();
    const size_t K = A.dimW();
    const size_t N = B.dimW();

    // std::cout << "A = [" << A.dimH() << "x" << A.dimW() << "]\n";
    // std::cout << "B = [" << B.dimH() << "x" << B.dimW() << "]\n";
    // std::cout << "C = [" << C.dimH() << "x" << C.dimW() << "]\n";
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            __half sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                sum += A[{i, k}] * B[{k, j}];
            }
            // std::cout << "ixj = " << i << "x" << j << "\n";
            C[{i, j}] = sum;
        }
    }
    // std::cout << "gemm\n";
}