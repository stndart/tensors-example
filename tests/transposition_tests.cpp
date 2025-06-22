#include <gtest/gtest.h>

#include "tensor.h"
#include "convolution.h"

//---------------------------------------------------------------------------
// Fixtures
//---------------------------------------------------------------------------
class Tensor4DTest : public ::testing::Test {
  protected:
    // helper to compare two Index4
    void expectOrderEq(const Index4 &a, const Index4 &b) {
        for (int i = 0; i < 4; ++i) {
            EXPECT_EQ(a[i], b[i]) << "Mismatch at position " << i;
        }
    }
};

//---------------------------------------------------------------------------
// 1) Setting & Getting axes_order
//---------------------------------------------------------------------------
TEST_F(Tensor4DTest, SetAndGetAxesOrder) {
    Tensor4D t(2, 3, 4, 5);
    // default order should be {0,1,2,3}
    Index4 default_order = t.get_axes_order();
    expectOrderEq(default_order, Index4{0, 1, 2, 3});

    // set to a new custom permutation
    Index4 custom = {2, 0, 3, 1};
    t.set_axes_order(custom);
    auto got = t.get_axes_order();
    expectOrderEq(got, custom);
}

//---------------------------------------------------------------------------
// 2) real_index with an arbitrary custom order
//---------------------------------------------------------------------------
TEST_F(Tensor4DTest, RealIndexWithCustomOrder) {
    Tensor4D t(1, 1, 1, 1);
    // pick a non-trivial axes_order
    Index4 order = {3, 2, 1, 0};
    t.set_axes_order(order);

    // pick an index in logical (i,j,k,l) space
    Index4 idx = {5, 6, 7, 8};
    // compute real_index
    Index4 real = t.real_index(idx);

    // real[order[i]] should equal idx[i] for each i
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(real[order[i]], idx[i]) << "Failed mapping at i=" << i;
    }
}

//---------------------------------------------------------------------------
// 3) Transpose swaps the last two entries in axes_order
//---------------------------------------------------------------------------
TEST_F(Tensor4DTest, TransposeSwapsLastTwoAxes) {
    Tensor4D t(1, 1, 1, 1);
    // start with a known custom order
    Index4 before = {0, 2, 1, 3};
    t.set_axes_order(before);

    t.transpose(); // should swap positions [2] and [3]
    Index4 after = t.get_axes_order();

    // expected: {0,2,3,1}
    Index4 expect = {0, 2, 3, 1};
    expectOrderEq(after, expect);
}

//---------------------------------------------------------------------------
// 4) Element access via operator[] still works after transpose
//---------------------------------------------------------------------------
TEST_F(Tensor4DTest, AccessAfterTranspose) {
    // create a tiny tensor with unique values so we can track them
    // dims: N=1, C=1, H=2, W=3
    Tensor4D t(1, 1, 2, 3);
    // initialize so that t[{0,0,i,j}] = float(i*10 + j)
    t.initialize({
        0 * 10 + 0, 0 * 10 + 1, 0 * 10 + 2, // row 0: [0,1,2]
        1 * 10 + 0, 1 * 10 + 1, 1 * 10 + 2  // row 1: [10,11,12]
    });

    // transpose should swap H and W
    t.transpose();
    EXPECT_EQ(t.dimY(), 3);
    EXPECT_EQ(t.dimZ(), 2);

    // now t[{0,0,i,j}] should return original t[{0,0,j,i}]
    EXPECT_FLOAT_EQ((t[{0, 0, 0, 0}]), 0.0f);  // (0,0)->(0,0)
    EXPECT_FLOAT_EQ((t[{0, 0, 0, 1}]), 10.0f); // (0,1)->orig (1,0)
    EXPECT_FLOAT_EQ((t[{0, 0, 1, 0}]), 1.0f);  // (1,0)->orig (0,1)
    EXPECT_FLOAT_EQ((t[{0, 0, 2, 1}]),
                    12.0f); // (1,2)->orig (2,1)
    EXPECT_THROW((t[{0, 0, 1, 2}]), std::range_error);

    // A more robust check across all coords:
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 2; ++j) {
            float got = t[{0, 0, i, j}];
            float expect = (j * 10 + i);
            EXPECT_FLOAT_EQ(got, expect)
                << "at transposed(" << i << "," << j << ")";
        }
    }
}

//---------------------------------------------------------------------------
// 5) Convolution layer forward works correctly even after transposing axes
//---------------------------------------------------------------------------
TEST_F(Tensor4DTest, ConvolutionAfterTranspose) {
    // Input: 1×1×2×2 tensor
    // logical layout (H×W):
    // [1, 2]
    // [3, 4]
    Tensor4D input(2, 2, 1, 1);
    input.initialize({ 1.0f, 2.0f, 3.0f, 4.0f });
    // swap H and W axes
    input.set_axes_order({2, 3, 1, 0});

    // Kernel: 1×1×2×2, identity on corners
    // logical:
    // [1, 0]
    // [0, 1]
    Tensor4D kernel(1, 1, 2, 2);
    kernel.initialize({ 1.0f, 0.0f, 0.0f, 1.0f });
    kernel.transpose();

    // Convolution with no padding, stride=1
    Convolution conv(kernel, /*pad_h=*/0, /*pad_w=*/0, /*stride_h=*/1, /*stride_w=*/1);

    Tensor4D output(1, 1, 1, 1);
    conv.forward(input, output);

    // If we had not transposed, conv on [[1,2],[3,4]] with that kernel gives
    // 1*1 + 4*1 = 5
    EXPECT_FLOAT_EQ((output[{0, 0, 0, 0}]), 5.0f);
}
