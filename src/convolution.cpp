#include "convolution.h"
#include "cpu/cpu_convolution.h"
#include "cuda/cuda_memory.cu"

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

void Convolution::forward(const Tensor4D &input, Tensor4D &output) const {
    if (input.dimX() != kernel->dimX())
        throw std::runtime_error("Tensor to kernel dimensions mismatch");

    auto [O, C, kH, kW] = kernel->vsize().as_tuple();
    auto [B, C2, H_in, W_in] = input.vsize().as_tuple();

    size_t H_out, W_out;
    calculate_HW_out(input.dimY(), input.dimZ(), kH, kW, H_pad, W_pad, H_stride,
                     W_stride, H_out, W_out);

    if (output.vsize() !=
        Index4{B, O, static_cast<int32_t>(H_out), static_cast<int32_t>(W_out)})
        throw std::runtime_error("forward: output dimensions mismatch");

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

    if (output.vsize() !=
        Index4{B, O, static_cast<int32_t>(H_out), static_cast<int32_t>(W_out)})
        throw std::runtime_error("Output tensor dimensions mismatch");

    // Actual data is always on GPU
    // doesn't work since D2H is non-const
    // #ifdef USE_CUDA
    //     input.D2H();
    //     kernel->D2H();
    // #endif

    // CPU always, since this is straitforward implementation, just for testing
    output.allocate_memory();
    cpu_convolution_simple(input, *kernel, output, H_pad, W_pad, H_stride,
                           W_stride);

    // #ifdef USE_CUDA
    //     output.H2D();
    // #endif
}

void Convolution::backward(const Tensor4D &input,
                           const Tensor4D &output_gradient,
                           Tensor4D &kernel_gradient,
                           Tensor4D &input_gradient) const {
    if (kernel->vsize() != kernel_gradient.vsize())
        throw std::runtime_error(
            "backward: kernel gradient dimentions mismatch");

    if (input.vsize() != input_gradient.vsize())
        throw std::runtime_error(
            "backward: input gradient dimentions mismatch");

    throw std::runtime_error("Conv backward is not implemented");
}

void Convolution::backward_simple(const Tensor4D &input,
                                  const Tensor4D &output_gradient,
                                  Tensor4D &kernel_gradient,
                                  Tensor4D &input_gradient) const {
    if (kernel->vsize() != kernel_gradient.vsize())
        throw std::runtime_error(
            "backward: kernel gradient dimentions mismatch");

    if (input.vsize() != input_gradient.vsize())
        throw std::runtime_error(
            "backward: input gradient dimentions mismatch");

    auto [O, C, kH, kW] = kernel->vsize().as_tuple();
    auto [B, C2, H_in, W_in] = input.vsize().as_tuple();

    size_t H_out, W_out;
    calculate_HW_out(H_in, W_in, kH, kW, H_pad, W_pad, H_stride, W_stride,
                     H_out, W_out);

    kernel_gradient.allocate_memory();
    kernel_gradient.fill(0);

    for (int32_t oi = 0; oi < O; ++oi)
        for (int32_t ci = 0; ci < C; ++ci)
            for (int32_t khi = 0; khi < kH; ++khi)
                for (int32_t kwi = 0; kwi < kW; ++kwi)
                    for (int32_t bi = 0; bi < B; ++bi)
                        for (int32_t oh = 0; oh < H_out; ++oh)
                            for (int32_t ow = 0; ow < W_out; ++ow) {
                                int32_t h_in = oh * H_stride + khi - H_pad;
                                int32_t w_in = ow * W_stride + kwi - W_pad;
                                if (h_in < 0 || w_in < 0 || h_in >= H_in ||
                                    w_in >= W_in)
                                    continue;

                                kernel_gradient[{oi, ci, khi, kwi}] +=
                                    output_gradient[{bi, oi, oh, ow}] *
                                    input[{bi, ci, h_in, w_in}];
                            }

    Tensor4D rotated_kernel(C, O, kH, kW);
    rotated_kernel.allocate_memory();

    for (int32_t oi = 0; oi < O; ++oi)
        for (int32_t ci = 0; ci < C; ++ci)
            for (int32_t khi = 0; khi < kH; ++khi)
                for (int32_t kwi = 0; kwi < kW; ++kwi) {
                    rotated_kernel[{ci, oi, khi, kwi}] =
                        (*kernel)[{oi, ci, kH - khi - 1, kW - kwi - 1}];
                }

    input_gradient.allocate_memory();
    input_gradient.fill(0);

    cpu_convolution_simple(output_gradient, rotated_kernel, input_gradient,
                           kH - 1 - H_pad, kW - 1 - W_pad, 1, 1);
}

void Convolution::apply_gradient_step(const Tensor4D &kernel_gradient_step) {
    if (kernel->vsize() != kernel_gradient_step.vsize())
        throw std::runtime_error(
            "apply_gradient_step: kernel gradient dimentions "
            "mismatch");

    // inplace addition
    Tensor4D::add(*kernel, kernel_gradient_step, *kernel);
}