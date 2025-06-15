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
        flatten_kernel->allocate_memory();

        for (size_t bi = 0; bi < O; ++bi)
            for (size_t ci = 0; ci < C; ++ci)
                for (size_t hi = 0; hi < kH; ++hi)
                    for (size_t wi = 0; wi < kW; ++wi) {
                        size_t input_idx =
                            bi * C * kH * kW + ci * kH * kW + hi * kW + wi;
                        size_t output_idx = ci * kH * kW + hi * kW + wi;

                        // std::cout << bi << "x" << ci << "x" << hi << "x"
                        // << wi << " >> " << bi << "x" << output_idx << "\n";

                        (*flatten_kernel)[{bi, output_idx}] =
                            (*kernel)[{bi, ci, hi, wi}];
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
    output_m.allocate_memory();

    // flatten_input.print("finput");
    // (*flatten_kernel).print("fkernel");
    Matrix::gemm(*flatten_kernel, flatten_input, output_m);
    // output_m.print("output");

    matrix_to_tensor_reshape(output_m, output, true);
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
                                    (*kernel)[{oi, ci, khi, kwi}];
                            }
            }
}