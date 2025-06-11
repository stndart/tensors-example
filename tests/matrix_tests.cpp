#include "matrix.h"
#include <gtest/gtest.h>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

class MatrixTest : public ::testing::Test {
  protected:
    void SetUp() override {
#ifdef USE_CUDA
        // Clear any prior CUDA errors before each test
        cudaGetLastError();
#endif
    }
};

TEST_F(MatrixTest, Creation) {
    Matrix mat(2, 3);
    EXPECT_EQ(mat.dimH(), 2);
    EXPECT_EQ(mat.dimW(), 3);
    mat.allocate_memory();
    EXPECT_NE(mat.data(), nullptr);
    EXPECT_EQ(mat.size(), 2 * 3);
}

TEST_F(MatrixTest, InitializeAndPrint) {
    Matrix mat(2, 2);
    std::vector<__half> values = {1.0f, 2.0f, 3.0f, 4.0f};
    mat.initialize(values);
    // mat.print(); // Optionally suppress or redirect stdout
    SUCCEED();
}

#ifdef USE_CUDA
TEST_F(MatrixTest, GPUTransfer) {
    Matrix mat(2, 2);
    mat.initialize({1.0f, 2.0f, 3.0f, 4.0f});
    mat.H2D();

    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);
}
#endif

TEST_F(MatrixTest, GemmBasic) {
    Matrix A(2, 2), B(2, 2), C(2, 2);
    A.initialize({1.0f, 2.0f, 3.0f, 4.0f});
    B.initialize({5.0f, 6.0f, 7.0f, 8.0f});
    A.H2D();
    B.H2D();
    Matrix::gemm(A, B, C);
    C.D2H();

    const __half *result = C.data();
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(result[0], (__half)19.0f);
    EXPECT_EQ(result[1], (__half)22.0f);
    EXPECT_EQ(result[2], (__half)43.0f);
    EXPECT_EQ(result[3], (__half)50.0f);

#ifdef USE_CUDA
    cudaError_t err = cudaGetLastError();
    EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
#endif
}

TEST_F(MatrixTest, InvalidInitializeThrows) {
    Matrix mat(2, 3);
    EXPECT_THROW(mat.initialize({1.0f, 2.0f}), std::runtime_error);
}
