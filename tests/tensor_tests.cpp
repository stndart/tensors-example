#include "tensor.h"
#include <gtest/gtest.h>
#include <vector>

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

class TensorTest : public ::testing::Test {
  protected:
    void SetUp() override {
#ifdef USE_CUDA
        // Clear any prior CUDA errors before each test
        cudaGetLastError();
#endif
    }
};

TEST_F(TensorTest, Creation) {
}

TEST_F(TensorTest, InitializeAndPrint) {
}

#ifdef USE_CUDA
TEST_F(TensorTest, GPUTransfer) {
}
#endif

TEST_F(TensorTest, GemmBasic) {
    
#ifdef USE_CUDA
    cudaError_t err = cudaGetLastError();
    EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
#endif
}

TEST_F(TensorTest, InvalidInitializeThrows) {
}
