#include <gtest/gtest.h>

#include "relu_activation.h"
// #include "sigmoid_activation.h"

class ActivationTest : public ::testing::Test {
  protected:
    void SetUp() override {
#ifdef USE_CUDA
        // Clear any prior CUDA errors before each test
        cudaGetLastError();
#endif
    }
};

TEST_F(ActivationTest, ReluForwardSimple) {
    Tensor4D input(1, 1, 1, 4);
    input.initialize({-1.0f, 0.0f, 2.0f, -3.0f});

    Relu relu;
    Tensor4D output(1, 1, 1, 4);
    relu.forward(input, output);

    EXPECT_FLOAT_EQ((output[{0, 0, 0, 0}]), 0.0f);
    EXPECT_FLOAT_EQ((output[{0, 0, 0, 1}]), 0.0f);
    EXPECT_FLOAT_EQ((output[{0, 0, 0, 2}]), 2.0f);
    EXPECT_FLOAT_EQ((output[{0, 0, 0, 3}]), 0.0f);
}

TEST_F(ActivationTest, ReluBackwardSimple) {
    Tensor4D input(1, 1, 1, 4);
    input.initialize({-1.0f, 0.0f, 2.0f, -3.0f});

    Relu relu;
    Tensor4D output(1, 1, 1, 4);
    relu.forward(input, output); // Needed to store output for backward

    Tensor4D grad_output(1, 1, 1, 4);
    grad_output.initialize({1.0f, 1.0f, 1.0f, 1.0f});

    Tensor4D grad_input(1, 1, 1, 4);
    relu.backward(grad_output, output, grad_input);

    EXPECT_FLOAT_EQ((grad_input[{0, 0, 0, 0}]), 0.0f);
    EXPECT_FLOAT_EQ((grad_input[{0, 0, 0, 1}]), 0.0f);
    EXPECT_FLOAT_EQ((grad_input[{0, 0, 0, 2}]), 1.0f);
    EXPECT_FLOAT_EQ((grad_input[{0, 0, 0, 3}]), 0.0f);
}

// TEST_F(ActivationTest, SigmoidForwardSimple) {
//     Tensor4D input(1, 1, 1, 2);
//     input.initialize({0.0f, 2.0f});

//     Sigmoid sigmoid;
//     Tensor4D output(1, 1, 1, 2);
//     sigmoid.forward(input, output);

//     EXPECT_NEAR((output[{0, 0, 0, 0}]), 0.5f, 1e-5);
//     EXPECT_NEAR((output[{0, 0, 0, 1}]), 1.0f / (1.0f + std::exp(-2.0f)), 1e-5);
// }

// TEST_F(ActivationTest, SigmoidBackwardSimple) {
//     Tensor4D input(1, 1, 1, 2);
//     input.initialize({0.0f, 2.0f});

//     Sigmoid sigmoid;
//     Tensor4D output(1, 1, 1, 2);
//     sigmoid.forward(input, output); // needed for backward

//     Tensor4D grad_output(1, 1, 1, 2);
//     grad_output.initialize({1.0f, 1.0f});

//     Tensor4D grad_input(1, 1, 1, 2);
//     sigmoid.backward(grad_output, grad_input);

//     float y0 = output[{0, 0, 0, 0}]; // sigmoid(0.0) = 0.5
//     float y1 = output[{0, 0, 0, 1}]; // sigmoid(2.0)
//     EXPECT_NEAR((grad_input[{0, 0, 0, 0}]), y0 * (1 - y0), 1e-5);
//     EXPECT_NEAR((grad_input[{0, 0, 0, 1}]), y1 * (1 - y1), 1e-5);
// }
