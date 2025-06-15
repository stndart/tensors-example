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
    Tensor4D tensor(1, 2, 3, 4);
    EXPECT_EQ(tensor.dimW(), 1);
    EXPECT_EQ(tensor.dimX(), 2);
    EXPECT_EQ(tensor.dimY(), 3);
    EXPECT_EQ(tensor.dimZ(), 4);
    tensor.allocate_memory();
    EXPECT_NE(tensor.data(), nullptr);
    EXPECT_EQ(tensor.size(), 1 * 2 * 3 * 4);
}

TEST_F(TensorTest, InitializeAndPrint) {
    Tensor4D tensor(1, 2, 3, 1);
    std::vector<__half> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    tensor.initialize(values);
    // tensor.print(); // Optionally suppress or redirect stdout
    SUCCEED();
}

#ifdef USE_CUDA
TEST_F(TensorTest, GPUTransfer) {
    Tensor4D tensor(1, 2, 3, 1);
    tensor.initialize({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    tensor.H2D();

    cudaError_t err = cudaGetLastError();
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);
}
#endif

TEST_F(TensorTest, InvalidInitializeThrows) {
    Tensor4D tensor(2, 3, 1, 1);
    EXPECT_THROW(tensor.initialize({1.0f, 2.0f}), std::runtime_error);
}

TEST_F(TensorTest, SumTensors) {
    Tensor4D tensor(1, 2, 3, 2);
    tensor.initialize({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    Tensor4D tensor2(1, 2, 3, 2);
    tensor2.initialize({11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0});

    Tensor4D output(1, 2, 3, 2);
    output.allocate_memory();
    Tensor4D::add(tensor, tensor2, output);

    EXPECT_FLOAT_EQ((output[{0, 0, 0, 0}]), 11.0f); // (0,0)
    EXPECT_FLOAT_EQ((output[{0, 0, 0, 1}]), 11.0f); // (0,1)
    EXPECT_FLOAT_EQ((output[{0, 1, 0, 0}]), 11.0f); // (1,0)
    EXPECT_FLOAT_EQ((output[{0, 1, 0, 1}]), 11.0f); // (1,1)

#ifdef USE_CUDA
    cudaError_t err = cudaGetLastError();
    EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
#endif
}

TEST_F(TensorTest, mean) {
    Tensor4D tensor(1, 2, 3, 2);
    tensor.initialize({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

    Tensor4D output(1, 2, 1, 2);
    output.allocate_memory();
    Tensor4D::mean(tensor, 2, output);

    EXPECT_FLOAT_EQ((output[{0, 0, 0, 0}]), 2.0f); // (0,0)
    EXPECT_FLOAT_EQ((output[{0, 0, 0, 1}]), 3.0f); // (0,1)
    EXPECT_FLOAT_EQ((output[{0, 1, 0, 0}]), 8.0f); // (1,0)
    EXPECT_FLOAT_EQ((output[{0, 1, 0, 1}]), 9.0f); // (1,1)

#ifdef USE_CUDA
    cudaError_t err = cudaGetLastError();
    EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
#endif
}

TEST_F(TensorTest, im2col_basic_2x2) {
    const size_t B = 1, C = 1, H = 2, W = 2;
    Tensor4D tensor(B, C, H, W);
    tensor.initialize({1.0f, 2.0f, 3.0f, 4.0f});

    const size_t kH = 2, kW = 2;
    size_t H_out, W_out;
    calculate_HW_out(H, W, kH, kW, 0, 0, 1, 1, H_out,
                     W_out); // No padding, stride=1

    Matrix mat(C * kH * kW, B * H_out * W_out);
    Tensor4D::im2col(tensor, mat, kH, kW, 0, 0, 1, 1);

    // Should have 1 patch: [1,2,3,4]
    EXPECT_FLOAT_EQ(mat.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(mat.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(mat.data()[2], 3.0f);
    EXPECT_FLOAT_EQ(mat.data()[3], 4.0f);

#ifdef USE_CUDA
    cudaError_t err = cudaGetLastError();
    EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
#endif
}

TEST_F(TensorTest, im2col_multi_channel_padding) {
    const size_t B = 1, C = 2, H = 2, W = 2;
    // Channel 1: [1,2], Channel 2: [-1,-2]
    //            [3,4]            [-3,-4]
    Tensor4D tensor(B, C, H, W);
    tensor.initialize({1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f});

    const size_t kH = 3, kW = 3; // 3x3 kernel requires padding for 2x2 input
    size_t H_out, W_out;
    calculate_HW_out(H, W, kH, kW, 1, 1, 1, 1, H_out,
                     W_out); // Padding=1, stride=1

    Matrix mat(C * kH * kW, B * H_out * W_out);
    Tensor4D::im2col(tensor, mat, kH, kW, 1, 1, 1, 1);

    /* Expected patch (9 elements per channel):
       Channel 1: [0,0,0, 0,1,2, 0,3,4] -> top-left 3x3 patch
       Channel 2: [0,0,0, 0,-1,-2, 0,-3,-4] */
    // Channel 1
    EXPECT_FLOAT_EQ((mat[{0, 0}]), 0.0f); // (0,0)
    EXPECT_FLOAT_EQ((mat[{1, 0}]), 0.0f); // (0,1)
    EXPECT_FLOAT_EQ((mat[{2, 0}]), 0.0f); // (0,2)
    EXPECT_FLOAT_EQ((mat[{3, 0}]), 0.0f); // (1,0)
    EXPECT_FLOAT_EQ((mat[{4, 0}]), 1.0f); // (1,1)
    EXPECT_FLOAT_EQ((mat[{5, 0}]), 2.0f); // (1,2)
    EXPECT_FLOAT_EQ((mat[{6, 0}]), 0.0f); // (2,0)
    EXPECT_FLOAT_EQ((mat[{7, 0}]), 3.0f); // (2,1)
    EXPECT_FLOAT_EQ((mat[{8, 0}]), 4.0f); // (2,2)

    // Channel 2 (offset 9)
    EXPECT_FLOAT_EQ((mat[{9, 0}]), 0.0f);   // (0,0)
    EXPECT_FLOAT_EQ((mat[{10, 0}]), 0.0f);  // (0,1)
    EXPECT_FLOAT_EQ((mat[{11, 0}]), 0.0f);  // (0,2)
    EXPECT_FLOAT_EQ((mat[{12, 0}]), 0.0f);  // (1,0)
    EXPECT_FLOAT_EQ((mat[{13, 0}]), -1.0f); // (1,1)
    EXPECT_FLOAT_EQ((mat[{14, 0}]), -2.0f); // (1,2)
    EXPECT_FLOAT_EQ((mat[{15, 0}]), 0.0f);  // (2,0)
    EXPECT_FLOAT_EQ((mat[{16, 0}]), -3.0f); // (2,1)
    EXPECT_FLOAT_EQ((mat[{17, 0}]), -4.0f); // (2,2)

#ifdef USE_CUDA
    cudaError_t err = cudaGetLastError();
    EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
#endif
}

TEST_F(TensorTest, col2im_single_patch) {
    const size_t C = 1, kH = 2, kW = 2;
    const size_t B = 1, H_out = 1, W_out = 1;
    const size_t N_patches = B * H_out * W_out;

    Matrix mat(C * kH * kW, N_patches);
    mat.initialize({1.0f, 2.0f, 3.0f, 4.0f}); // Single column

    Tensor4D output(B, C, 2, 2);                       // Original 2x2 input
    Tensor4D::col2im(mat, output, kH, kW, 0, 0, 1, 1); // No padding, stride=1

    /* Should reconstruct to:
       [[1,2],
        [3,4]] */
    EXPECT_FLOAT_EQ(output.data()[0], 1.0f);
    EXPECT_FLOAT_EQ(output.data()[1], 2.0f);
    EXPECT_FLOAT_EQ(output.data()[2], 3.0f);
    EXPECT_FLOAT_EQ(output.data()[3], 4.0f);

#ifdef USE_CUDA
    cudaError_t err = cudaGetLastError();
    EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
#endif
}

TEST_F(TensorTest, col2im_overlapping_patches) {
    /* Input: 1x1x3x3 tensor
       Patches:
         TL: [1,2,4,5] -> position (0,0)
         TR: [2,3,5,6] -> position (0,1)
         BL: [4,5,7,8] -> position (1,0)
         BR: [5,6,8,9] -> position (1,1) */
    const size_t C = 1, kH = 2, kW = 2;
    const size_t B = 1, H_out = 2, W_out = 2;
    const size_t N_patches = B * H_out * W_out;

    Matrix mat(C * kH * kW, N_patches);
    mat.initialize({1.0f, 2.0f, 4.0f, 5.0f, 2.0f, 3.0f, 5.0f, 6.0f, 4.0f, 5.0f,
                    7.0f, 8.0f, 5.0f, 6.0f, 8.0f, 9.0f});

    Tensor4D output(B, C, 3, 3);                       // Original 3x3 input
    Tensor4D::col2im(mat, output, kH, kW, 0, 0, 1, 1); // No padding, stride=1

    /* Reconstructed tensor should be:
       [1, 2+2, 3,
        4+4, 5+5+5+5, 6+6,
        7, 8+8, 9] */
    EXPECT_FLOAT_EQ(output.data()[0], 1.0f);  // Position (0,0)
    EXPECT_FLOAT_EQ(output.data()[1], 4.0f);  // 2+2
    EXPECT_FLOAT_EQ(output.data()[2], 3.0f);  // Position (0,2)
    EXPECT_FLOAT_EQ(output.data()[3], 8.0f);  // 4+4
    EXPECT_FLOAT_EQ(output.data()[4], 20.0f); // 5*4
    EXPECT_FLOAT_EQ(output.data()[5], 12.0f); // 6+6
    EXPECT_FLOAT_EQ(output.data()[6], 7.0f);  // Position (2,0)
    EXPECT_FLOAT_EQ(output.data()[7], 16.0f); // 8+8
    EXPECT_FLOAT_EQ(output.data()[8], 9.0f);  // Position (2,2)

#ifdef USE_CUDA
    cudaError_t err = cudaGetLastError();
    EXPECT_EQ(err, cudaSuccess) << cudaGetErrorString(err);
#endif
}