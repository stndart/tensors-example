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

void Convolution::backward(const Tensor4D &input,
                           const Tensor4D &output_gradient,
                           Tensor4D &kernel_gradient,
                           Tensor4D &input_gradient) const {}

void Convolution::apply_gradient_step(const Tensor4D &kernel_gradient_step) {}