#include "cpu/cpu_avg_pooling.h"

void cpu_avg_pooling(const Tensor4D &input, Tensor4D &output,
                     PaddingMode padding_mode, size_t H_pad, size_t W_pad,
                     size_t H_stride, size_t W_stride, size_t H_size,
                     size_t W_size) {
    auto [B, C, H, W] = input.vsize().as_tuple();
    output.fill(0);
    __half scale = 1.0 / H_size / W_size;

    for (size_t ohi = 0; ohi < output.dimY(); ++ohi)
        for (size_t owi = 0; owi < output.dimZ(); ++owi) {
            int oh = -H_pad + ohi * H_stride;
            int ow = -W_pad + owi * W_stride;
            for (size_t bi = 0; bi < B; ++bi)
                for (size_t ci = 0; ci < C; ++ci)
                    for (size_t hi = 0; hi < H; ++hi)
                        for (size_t wi = 0; wi < W; ++wi) {
                            int fhi = oh + hi;
                            int fwi = ow + wi;

                            Index4 idx = padded_access<Tensor4D>(
                                input, {bi, ci, (size_t)fhi, (size_t)fwi},
                                padding_mode);

                            __half candidate = 0;
                            if (idx >= 0 &&
                                idx < input.vsize()) // Non-zero padding branch
                                candidate = input[idx];

                            output[{bi, ci, ohi, owi}] += candidate * scale;
                        }
        }
}

void cpu_avg_pooling_backward(const Tensor4D &output_gradient,
                              Tensor4D &input_gradient,
                              PaddingMode padding_mode, size_t H_pad,
                              size_t W_pad, size_t H_stride, size_t W_stride,
                              size_t H_size, size_t W_size) {
    auto [B, C, H, W] = input_gradient.vsize().as_tuple();
    input_gradient.fill(0);
    __half scale = 1.0 / H_size / W_size;

    for (size_t ohi = 0; ohi < output_gradient.dimY(); ++ohi)
        for (size_t owi = 0; owi < output_gradient.dimZ(); ++owi) {
            int oh = -H_pad + ohi * H_stride;
            int ow = -W_pad + owi * W_stride;
            for (size_t bi = 0; bi < B; ++bi)
                for (size_t ci = 0; ci < C; ++ci)
                    for (size_t hi = 0; hi < H; ++hi)
                        for (size_t wi = 0; wi < W; ++wi) {
                            int fhi = oh + hi;
                            int fwi = ow + wi;

                            Index4 idx = padded_access<Tensor4D>(
                                input_gradient,
                                {bi, ci, (size_t)fhi, (size_t)fwi},
                                padding_mode);

                            if (idx >= 0 &&
                                idx < input_gradient
                                          .vsize()) // Zero padding branch
                                input_gradient[idx] +=
                                    output_gradient[{bi, ci, ohi, owi}] * scale;
                        }
        }
}