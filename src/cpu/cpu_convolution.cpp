#include "cpu/cpu_convolution.h"
#include "convolution.h"

void cpu_convolution(const Tensor4D &input, const Tensor4D &kernel,
                     Tensor4D &output, const size_t H_pad, const size_t W_pad,
                     const size_t H_stride, const size_t W_stride,
                     const Matrix *flatten_kernel) {
    auto [O, C, kH, kW] = kernel.vsize().as_tuple();

    if (flatten_kernel == nullptr) {
        flatten_kernel = Convolution::get_flatten_kernel(kernel);
    }

    size_t H_out, W_out;
    calculate_HW_out(input.dimY(), input.dimZ(), kH, kW, H_pad, W_pad, H_stride,
                     W_stride, H_out, W_out);

    const size_t N_patches = input.dimW() * H_out * W_out;
    Matrix flatten_input(C * kH * kW, N_patches);
    Tensor4D::im2col(input, flatten_input, kH, kW, H_pad, W_pad, H_stride,
                     W_stride);

    Matrix output_m(O, N_patches);
    output_m.allocate_memory();
    Matrix::gemm(*flatten_kernel, flatten_input, output_m);

    matrix_to_tensor_reshape(output_m, output);
}

void cpu_convolution_simple(const Tensor4D &input, const Tensor4D &kernel,
                            Tensor4D &output, const size_t H_pad,
                            const size_t W_pad, const size_t H_stride,
                            const size_t W_stride) {

    auto [O, C, kH, kW] = kernel.vsize().as_tuple();
    auto [O2, B, H_in, W_in] = input.vsize().as_tuple();

    size_t H_out, W_out;
    calculate_HW_out(H_in, W_in, kH, kW, H_pad, W_pad, H_stride, W_stride,
                     H_out, W_out);

    output.fill(0);

    for (size_t bi = 0; bi < B; ++bi)
        for (size_t oi = 0; oi < O; ++oi)
            for (size_t ci = 0; ci < C; ++ci) {
                for (size_t oh = 0; oh < H_out; ++oh)
                    for (size_t ow = 0; ow < W_out; ++ow)
                        for (size_t khi = 0; khi < kH; ++khi)
                            for (size_t kwi = 0; kwi < kW; ++kwi) {
                                int h_in = (int)(oh * H_stride + khi) - H_pad;
                                int w_in = (int)(ow * W_stride + kwi) - W_pad;
                                if (h_in < 0 || w_in < 0 || h_in >= H_in ||
                                    w_in >= W_in)
                                    continue;

                                output[{bi, oi, oh, ow}] +=
                                    input[{bi, ci, (size_t)h_in,
                                           (size_t)w_in}] *
                                    kernel[{oi, ci, khi, kwi}];
                            }
            }
}
