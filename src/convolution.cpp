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

    flatten_kernel = Convolution::get_flatten_kernel(*kernel);
}

Convolution::~Convolution() {
    if (flatten_kernel != nullptr)
        delete flatten_kernel;
}

Matrix *Convolution::get_flatten_kernel(const Tensor4D &kernel) {
    auto [O, C, kH, kW] = kernel.vsize().as_tuple();

    Matrix *flatten_kernel = new Matrix(O, C * kH * kW);
    flatten_kernel->allocate_memory();
    tensor_to_matrix_reshape_const(kernel, *flatten_kernel);

    return flatten_kernel;
}

void cpu_convolution(const Tensor4D &input, const Tensor4D &kernel,
                     Tensor4D &output, const size_t H_pad, const size_t W_pad,
                     const size_t H_stride, const size_t W_stride,
                     const Matrix *flatten_kernel = nullptr) {
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

void Convolution::forward(const Tensor4D &input, Tensor4D &output) const {
    if (input.dimX() != kernel->dimX()) {
        throw std::runtime_error("Tensor to kernel dimensions mismatch");
    }

#ifdef USE_CUDA
    try {
        cuda_convolution(input, *kernel, output, H_pad, W_pad, H_stride,
                         W_stride, flatten_kernel);
        return;
    } catch (const std::exception &e) {
        std::cerr << "CUDA error: " << e.what();
        std::cerr << "Falling back to CPU\n";
    }
#endif

    // CPU fallback
    output.allocate_memory();
    cpu_convolution(input, *kernel, output, H_pad, W_pad, H_stride, W_stride,
                    flatten_kernel);
}

void Convolution::forward_simple(const Tensor4D &input,
                                 Tensor4D &output) const {
    auto [O, C, kH, kW] = kernel->vsize().as_tuple();
    auto [O2, B, H_in, W_in] = input.vsize().as_tuple();

    if (O != O2)
        throw std::runtime_error("Tensor to kernel dimensions mismatch");

    size_t H_out, W_out;
    calculate_HW_out(H_in, W_in, kH, kW, H_pad, W_pad, H_stride, W_stride,
                     H_out, W_out);

    if (output.dimW() != B || output.dimX() != O || output.dimY() != H_out ||
        output.dimZ() != W_out)
        throw std::runtime_error("Output tensor dimensions mismatch");

    // Actual data is always on GPU
    // doesn't work since D2H is non-const
    // #ifdef USE_CUDA
    //     input.D2H();
    //     kernel->D2H();
    // #endif

    // CPU always, since this is straitforward implementation, just for testing
    output.allocate_memory();
    cpu_convolution(input, *kernel, output, H_pad, W_pad, H_stride, W_stride,
                    flatten_kernel);

    // #ifdef USE_CUDA
    //     output.H2D();
    // #endif
}