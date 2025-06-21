#include <gtest/gtest.h>

#include "avg_pooling.h"
#include "max_pooling.h"

class PoolingTest : public ::testing::Test {
  protected:
    void SetUp() override {
#ifdef USE_CUDA
        // Clear any prior CUDA errors before each test
        cudaGetLastError();
#endif
    }
};

TEST_F(PoolingTest, MaxPoolForwardSimple) {
    // Input: 1x1x4x4
    Tensor4D input(1, 1, 4, 4);
    input.initialize({
        1.0f, 2.0f, 3.0f, 4.0f,    // 0th row
        5.0f, 6.0f, 7.0f, 8.0f,    // 1st row
        9.0f, 10.0f, 11.0f, 12.0f, // 2nd row
        13.0f, 14.0f, 15.0f, 16.0f // 3rd row
    });

    MaxPooling pool(input.vsize(), PaddingMode::ZERO_PADDING, 0, 0, 2, 2);
    Tensor4D output(1, 1, 2, 2);
    pool.forward(input, output);

    // Expected: max of each 2x2 block
    EXPECT_FLOAT_EQ((output[{0, 0, 0, 0}]), 6.0f);  // Block 1: max(1,2,5,6)
    EXPECT_FLOAT_EQ((output[{0, 0, 0, 1}]), 8.0f);  // Block 2: max(3,4,7,8)
    EXPECT_FLOAT_EQ((output[{0, 0, 1, 0}]), 14.0f); // Block 3: max(9,10,13,14)
    EXPECT_FLOAT_EQ((output[{0, 0, 1, 1}]), 16.0f); // Block 4: max(11,12,15,16)
}

TEST_F(PoolingTest, MaxPoolForwardWithPadding) {
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
    EXPECT_FLOAT_EQ((output[{0, 0, 0, 0}]), 9.0f);
    EXPECT_FLOAT_EQ((output[{0, 0, 0, 1}]), 9.0f);
    EXPECT_FLOAT_EQ((output[{0, 0, 1, 0}]), 9.0f);
    EXPECT_FLOAT_EQ((output[{0, 0, 1, 1}]), 9.0f);
}