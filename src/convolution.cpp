#include "convolution.h"
#include "cuda_memory.cu"

Convolution::Convolution(Tensor4D &Kernel, std::optional<size_t> H_pad_,
                         std::optional<size_t> W_pad_,
                         std::optional<size_t> H_stride_,
                         std::optional<size_t> W_stride_)
    : kernel(&Kernel), flatten_kernel(nullptr) {
    H_pad = H_pad_.value_or(kernel->dimY() / 2);
    W_pad = W_pad_.value_or(kernel->dimZ() / 2);
    H_stride = H_stride_.value_or(1);
    W_stride = W_stride_.value_or(1);
}

Convolution::~Convolution() {
    if (flatten_kernel != nullptr)
        delete flatten_kernel;
}

void Convolution::get_flatten_kernel() {
    size_t O = kernel->dimW();
    size_t C = kernel->dimX();
    size_t kH = kernel->dimY();
    size_t kW = kernel->dimZ();

    if (flatten_kernel == nullptr) {
        flatten_kernel = new Matrix(O, C * kH * kW);
        for (size_t bi = 0; bi < O; ++bi)
            for (size_t ci = 0; ci < C; ++ci)
                for (size_t hi = 0; hi < kH; ++hi)
                    for (size_t wi = 0; wi < kW; ++wi) {
                        size_t input_idx =
                            bi * C * kH * kW + ci * kH * kW + hi * kW + wi;
                        size_t output_idx = ci * kH * kW + hi * kW + wi;
                        flatten_kernel
                            ->data()[bi * flatten_kernel->dimW() + output_idx] =
                            kernel->data()[input_idx];
                    }
    }
}

void Convolution::forward(const Tensor4D &input, Tensor4D &output) {
    if (input.dimX() != kernel->dimX()) {
        throw std::runtime_error("Tensor to kernel dimensions mismatch");
    }

    get_flatten_kernel();

    size_t O = kernel->dimW();
    size_t C = kernel->dimX();
    size_t kH = kernel->dimY();
    size_t kW = kernel->dimZ();

    size_t H_out, W_out;
    calculate_HW_out(input.dimY(), input.dimZ(), kH, kW, H_pad, W_pad, H_stride,
                     W_stride, H_out, W_out);

    const size_t N_patches = input.dimW() * H_out * W_out;
    Matrix flatten_input(C * kH * kW, N_patches);
    Tensor4D::im2col(input, flatten_input, kH, kW, H_pad, W_pad, H_stride,
                     W_stride);

    Matrix output_m(O, N_patches);
    Matrix::gemm(flatten_input, *flatten_kernel, output_m);

    Tensor4D::col2im(output_m, output, kH, kW, H_pad, W_pad, H_stride,
                     W_stride);
}

void Convolution::forward_simple(const Tensor4D &input, Tensor4D &output) {
    if (input.dimX() != kernel->dimX()) {
        throw std::runtime_error("Tensor to kernel dimensions mismatch");
    }

    size_t O = kernel->dimW();
    size_t C = kernel->dimX();
    size_t kH = kernel->dimY();
    size_t kW = kernel->dimZ();
    size_t B = input.dimW();
    size_t H_in = input.dimY();
    size_t W_in = input.dimZ();

    size_t H_out, W_out;
    calculate_HW_out(H_in, W_in, kH, kW, H_pad, W_pad, H_stride, W_stride,
                     H_out, W_out);

    if (output.dimW() != B || output.dimX() != O || output.dimY() != H_out ||
        output.dimZ() != W_out) {
        throw std::runtime_error("Output tensor dimensions mismatch");
    }

    for (size_t i = 0; i < output.size(); ++i)
        output.data()[i] = 0;

    for (size_t bi = 0; bi < B; ++bi)
        for (size_t oi = 0; oi < O; ++oi)
            for (size_t ci = 0; ci < C; ++ci) {
                size_t k_offset = oi * C * kH * kW + ci * kH * kW;
                for (size_t oh = 0; oh < H_out; ++oh)
                    for (size_t ow = 0; ow < W_out; ++ow)
                        for (size_t khi = 0; khi < kH; ++khi)
                            for (size_t kwi = 0; kwi < kW; ++kwi) {
                                int h_in = (int)(oh * H_stride + khi) - H_pad;
                                int w_in = (int)(ow * W_stride + kwi) - W_pad;
                                if (h_in < 0 || w_in < 0 || h_in >= H_in ||
                                    w_in >= W_in)
                                    continue;

                                size_t out_idx = bi * O * H_out * W_out +
                                                 oi * H_out * W_out +
                                                 oh * W_out + ow;
                                size_t in_idx = bi * C * H_in * W_in +
                                                ci * H_in * W_in + h_in * W_in +
                                                w_in;
                                output.data()[out_idx] +=
                                    input.data()[in_idx] *
                                    kernel->data()[k_offset + khi * kW + kwi];
                            }
            }
}