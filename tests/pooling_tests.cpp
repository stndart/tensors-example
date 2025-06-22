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

TEST_F(PoolingTest, MaxPoolBackwardSimple) {
    // Forward pass
    Tensor4D input(1, 1, 2, 2);
    input.initialize({1.0f, 3.0f, 2.0f, 4.0f});

    MaxPooling pool(input.vsize()); // 2x2 window, pad=0, stride=2
    Tensor4D output(1, 1, 1, 1);
    pool.forward(input, output); // output = max(1,3,2,4) = 4

    // Backward pass
    Tensor4D grad_output(1, 1, 1, 1);
    grad_output.initialize({1.0f}); // dL/doutput = 1
    Tensor4D grad_input(1, 1, 2, 2);
    pool.backward(grad_output, grad_input);

    // Expected: gradient only at max position (4)
    EXPECT_FLOAT_EQ((grad_input[{0, 0, 0, 0}]), 0.0f);
    EXPECT_FLOAT_EQ((grad_input[{0, 0, 0, 1}]), 0.0f);
    EXPECT_FLOAT_EQ((grad_input[{0, 0, 1, 0}]), 0.0f);
    EXPECT_FLOAT_EQ((grad_input[{0, 0, 1, 1}]), 1.0f); // Position of max
}

TEST_F(PoolingTest, MaxPoolBackwardWithTie) {
    GTEST_SKIP(); // tie is not implemented

    // Input with duplicate max values
    Tensor4D input(1, 1, 2, 2);
    input.initialize({5.0f, 5.0f, 5.0f, 5.0f});

    MaxPooling pool(input.vsize()); // 2x2 window, pad=0, stride=2
    Tensor4D output(1, 1, 1, 1);
    pool.forward(input, output); // output = max(1,3,2,4) = 4

    // Backward
    Tensor4D grad_output(1, 1, 1, 1);
    grad_output.initialize({1.0f});
    Tensor4D grad_input(1, 1, 2, 2);
    pool.backward(grad_output, grad_input);

    // Expected: Gradient divided equally among max positions
    EXPECT_FLOAT_EQ((grad_input[{0, 0, 0, 0}]), 0.25f);
    EXPECT_FLOAT_EQ((grad_input[{0, 0, 0, 1}]), 0.25f);
    EXPECT_FLOAT_EQ((grad_input[{0, 0, 1, 0}]), 0.25f);
    EXPECT_FLOAT_EQ((grad_input[{0, 0, 1, 1}]), 0.25f);
}

TEST_F(PoolingTest, AvgPoolForwardSimple) {
    // Input: 1x1x4x4
    Tensor4D input(1, 1, 4, 4);
    input.initialize({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                      10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f});

    AvgPooling pool(input.vsize(), PaddingMode::ZERO_PADDING, 0, 0, 2,
                    2); // 2x2 kernel, stride=2
    Tensor4D output(1, 1, 2, 2);
    pool.forward(input, output);

    // Expected: average of each 2x2 block
    EXPECT_FLOAT_EQ((output[{0, 0, 0, 0}]), (1 + 2 + 5 + 6) / 4.0f);
    EXPECT_FLOAT_EQ((output[{0, 0, 0, 1}]), (3 + 4 + 7 + 8) / 4.0f);
    EXPECT_FLOAT_EQ((output[{0, 0, 1, 0}]), (9 + 10 + 13 + 14) / 4.0f);
    EXPECT_FLOAT_EQ((output[{0, 0, 1, 1}]), (11 + 12 + 15 + 16) / 4.0f);
}

TEST_F(PoolingTest, AvgPoolForwardWithReflectionPadding) {
    // Input: 1x1x2x2
    Tensor4D input(1, 1, 2, 2);
    input.initialize({1.0f, 2.0f, 3.0f, 4.0f});

    // 3x3 pooling, stride=1, reflection padding=1
    AvgPooling pool(input.vsize(), PaddingMode::REFLECTION_PADDING, 1, 1, 1, 1,
                    3, 3);
    Tensor4D output(1, 1, 2, 2);
    pool.forward(input, output);

    // Padded input (reflection):
    /* 4, 3, 4, 3
       2, 1, 2, 1
       4, 3, 4, 3
       2, 1, 2, 1 */
    EXPECT_FLOAT_EQ((output[{0, 0, 0, 0}]), 3.0f);
    EXPECT_FLOAT_EQ((output[{0, 0, 0, 1}]), 8.0f / 3.0f);
    EXPECT_FLOAT_EQ((output[{0, 0, 1, 0}]), 7.0f / 3.0f);
    EXPECT_FLOAT_EQ((output[{0, 0, 1, 1}]), 2.0f);
}

TEST_F(PoolingTest, AvgPoolBackwardSimple) {
    // Forward pass
    Tensor4D input(1, 1, 2, 2);
    input.initialize({1.0f, 2.0f, 3.0f, 4.0f});

    AvgPooling pool(input.vsize()); // 2x2 kernel, pad=0
    Tensor4D output(1, 1, 1, 1);
    pool.forward(input, output); // output = (1+2+3+4)/4 = 2.5

    // Backward pass
    Tensor4D grad_output(1, 1, 1, 1);
    grad_output.initialize({1.0f}); // dL/doutput = 1
    Tensor4D grad_input(1, 1, 2, 2);
    pool.backward(grad_output, grad_input);

    // Expected: gradient distributed equally
    const float expected_grad = 1.0f / 4.0f;
    EXPECT_FLOAT_EQ((grad_input[{0, 0, 0, 0}]), expected_grad);
    EXPECT_FLOAT_EQ((grad_input[{0, 0, 0, 1}]), expected_grad);
    EXPECT_FLOAT_EQ((grad_input[{0, 0, 1, 0}]), expected_grad);
    EXPECT_FLOAT_EQ((grad_input[{0, 0, 1, 1}]), expected_grad);
}

TEST_F(PoolingTest, AvgPoolBackwardWithPadding) {
    // Forward pass with padding
    Tensor4D input(1, 1, 3, 3);
    input.initialize({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f});

    // 2x2 pooling, stride=1, zero padding=0
    AvgPooling pool(input.vsize(), PaddingMode::ZERO_PADDING, 0, 0, 1, 1, 2, 2);
    Tensor4D output(1, 1, 2, 2);
    pool.forward(input, output);

    // Backward pass
    Tensor4D grad_output(1, 1, 2, 2);
    grad_output.initialize({1.0f, 1.0f, 1.0f, 1.0f}); // Uniform gradient
    Tensor4D grad_input(1, 1, 3, 3);
    pool.backward(grad_output, grad_input);

    /* Each input element contributes to multiple output positions:
       Input gradient = sum(grad_output * (1/area))
       Area for each input position depends on how many pools cover it:
        TL: covered by 1 pool (1/4)
        TR: covered by 2 pools (2/4)
        BL: covered by 2 pools (2/4)
        BR: covered by 4 pools (4/4) */
    EXPECT_FLOAT_EQ((grad_input[{0, 0, 0, 0}]), 1.0f * (1.0f / 4.0f)); // TL
    EXPECT_FLOAT_EQ((grad_input[{0, 0, 0, 1}]), 1.0f * (2.0f / 4.0f)); // TR
    EXPECT_FLOAT_EQ((grad_input[{0, 0, 1, 0}]), 1.0f * (2.0f / 4.0f)); // BL
    EXPECT_FLOAT_EQ((grad_input[{0, 0, 1, 1}]), 1.0f * (4.0f / 4.0f)); // BR
}