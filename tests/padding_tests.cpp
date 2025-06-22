#include <gtest/gtest.h>

#include "avg_pooling.h"
#include "max_pooling.h"

class PaddingTest : public ::testing::Test {
  protected:
    void SetUp() override {
#ifdef USE_CUDA
        // Clear any prior CUDA errors before each test
        cudaGetLastError();
#endif
    }
};

TEST_F(PaddingTest, ZeroPadding) {
    // Input: 1x1x2x2
    Tensor4D input(1, 1, 2, 2);
    input.initialize({1.0f, 2.0f, 
                     3.0f, 4.0f});
    
    // Expected: 4x4 with zeros
    EXPECT_FLOAT_EQ(padded(input, {0,0,-1,-1}, PaddingMode::ZERO_PADDING), 0.0f);  // TL
    EXPECT_FLOAT_EQ(padded(input, {0,0,0,0}, PaddingMode::ZERO_PADDING), 1.0f);  // Center
    EXPECT_FLOAT_EQ(padded(input, {0,0,1,1}, PaddingMode::ZERO_PADDING), 4.0f);  // BR
    EXPECT_FLOAT_EQ(padded(input, {0,0,2,2}, PaddingMode::ZERO_PADDING), 0.0f);  // Outside
}

TEST_F(PaddingTest, ReplicationPadding) {
    // Input: 1x1x2x2
    Tensor4D input(1, 1, 2, 2);
    input.initialize({1.0f, 2.0f,
                     3.0f, 4.0f});
    
    // Expected: Replicated edges
    EXPECT_FLOAT_EQ(padded(input, {0,0,-1,-1}, PaddingMode::REPLICATION_PADDING), 1.0f);  // TL corner
    EXPECT_FLOAT_EQ(padded(input, {0,0,-1,0}, PaddingMode::REPLICATION_PADDING), 1.0f);  // Top edge
    EXPECT_FLOAT_EQ(padded(input, {0,0,-1,1}, PaddingMode::REPLICATION_PADDING), 2.0f);  // Top edge
    EXPECT_FLOAT_EQ(padded(input, {0,0,0,-1}, PaddingMode::REPLICATION_PADDING), 1.0f);  // Left edge
    EXPECT_FLOAT_EQ(padded(input, {0,0,2,2}, PaddingMode::REPLICATION_PADDING), 4.0f);  // BR corner
}

TEST_F(PaddingTest, ReflectionPadding) {
    // Input: 1x1x3x3
    Tensor4D input(1, 1, 3, 3);
    input.initialize({1.0f, 2.0f, 3.0f,
                     4.0f, 5.0f, 6.0f,
                     7.0f, 8.0f, 9.0f});
    
    // Expected: Reflected values
    /* 9,8,7,8,9
       6,5,4,5,6
       3,2,1,2,3
       6,5,4,5,6
       9,8,7,8,9 */
    EXPECT_FLOAT_EQ(padded(input, {0,0,-1,-1}, PaddingMode::REFLECTION_PADDING), 5.0f);  // Center of TL quadrant
    EXPECT_FLOAT_EQ(padded(input, {0,0,-1,0}, PaddingMode::REFLECTION_PADDING), 4.0f);  // Top edge
    EXPECT_FLOAT_EQ(padded(input, {0,0,0,-1}, PaddingMode::REFLECTION_PADDING), 2.0f);  // Left edge
    EXPECT_FLOAT_EQ(padded(input, {0,0,3,3}, PaddingMode::REFLECTION_PADDING), 5.0f);  // Center of BR quadrant
}

TEST_F(PaddingTest, CircularPadding) {
    // Input: 1x1x2x2
    Tensor4D input(1, 1, 2, 2);
    input.initialize({1.0f, 2.0f,
                     3.0f, 4.0f});
    
    // Expected: Circular repetition
    /* 4,3,4,3
       2,1,2,1
       4,3,4,3
       2,1,2,1 */
    EXPECT_FLOAT_EQ(padded(input, {0,0,-1,-1}, PaddingMode::CIRCULAR_PADDING), 4.0f);  // TL
    EXPECT_FLOAT_EQ(padded(input, {0,0,-1,0}, PaddingMode::CIRCULAR_PADDING), 3.0f);  // Top
    EXPECT_FLOAT_EQ(padded(input, {0,0,0,-1}, PaddingMode::CIRCULAR_PADDING), 2.0f);  // Left
    EXPECT_FLOAT_EQ(padded(input, {0,0,2,2}, PaddingMode::CIRCULAR_PADDING), 1.0f);  // Bottom-right
}