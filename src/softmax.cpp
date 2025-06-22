#include "softmax.h"

#include <cmath>

void Softmax::forward(const Tensor4D &input, Tensor4D &output) const {
    if (output.vsize() != input.vsize())
        throw std::runtime_error("Forward: output dimensions mismatch");

#ifdef USE_CUDA
    try {
        cuda_softmax(input, output);
        return;
    } catch (const std::exception &e) {
        std::cerr << "CUDA error: " << e.what();
        std::cerr << "Falling back to CPU\n";
    }
#endif

    // CPU fallback
    output.allocate_memory();
    cpu_softmax(input, output);
}

void Softmax::backward(const Tensor4D &output_grad, const Tensor4D &output,
                       Tensor4D &input_grad) const {

    if (output_grad.vsize() != input_grad.vsize())
        throw std::runtime_error(
            "Backward: input gradient dimensions mismatch");
    if (output_grad.vsize() != output.vsize())
        throw std::runtime_error(
            "Backward: output gradient dimensions mismatch");

#ifdef USE_CUDA
    try {
        cuda_softmax_backward(output_grad, output, input_grad);
        return;
    } catch (const std::exception &e) {
        std::cerr << "CUDA error: " << e.what();
        std::cerr << "Falling back to CPU\n";
    }
#endif

    // CPU fallback
    input_grad.allocate_memory();
    cpu_softmax_backward(output_grad, output, input_grad);
}

void cpu_softmax(const Tensor4D &input, Tensor4D &output) {
    Index4 vsize = input.vsize();
    vsize[3] = 1; // one column
    Tensor4D safemax(vsize);
    safemax.allocate_memory();
    Tensor4D::max(input, 3, safemax);

    const auto [B, C, H, W] = input.vsize().as_tuple();
    for (int32_t bi = 0; bi < B; ++bi)
        for (int32_t ci = 0; ci < C; ++ci)
            for (int32_t hi = 0; hi < H; ++hi)
                for (int32_t wi = 0; wi < W; ++wi) {
                    output[{bi, ci, hi, wi}] =
                        exp(input[{bi, ci, hi, wi}] - safemax[{bi, ci, hi, 0}]);
                    // std::cout << output[{bi, ci, hi, wi}] << Index4{bi, ci,
                    // hi, wi} << " << exp(" << input[{bi, ci, hi, wi}] <<
                    // Index4{bi, ci, hi, wi} << ")\n";
                }

    // here safemax is used for normalization instead
    Tensor4D::sum(output, 3, safemax);

    for (int32_t bi = 0; bi < B; ++bi)
        for (int32_t ci = 0; ci < C; ++ci)
            for (int32_t hi = 0; hi < H; ++hi)
                for (int32_t wi = 0; wi < W; ++wi) {
                    output[{bi, ci, hi, wi}] =
                        output[{bi, ci, hi, wi}] / safemax[{bi, ci, hi, 0}];
                    // std::cout << output[{bi, ci, hi, wi}] << Index4{bi, ci,
                    // hi, wi} << " << " << safemax[{bi, ci, hi, 0}] <<
                    // Index4{bi, ci, 0, wi} << "\n";
                }
}

void cpu_softmax_backward(const Tensor4D &output_grad, const Tensor4D &output,
                          Tensor4D &input_grad) {

    const auto [B, C, H, W] = output_grad.vsize().as_tuple();

    input_grad.fill(0);
    for (int32_t bi = 0; bi < B; ++bi)
        for (int32_t ci = 0; ci < C; ++ci)
            for (int32_t hi = 0; hi < H; ++hi) {
                __half S = 0.0f;
                for (int32_t wi = 0; wi < W; ++wi)
                    S += output[{bi, ci, hi, wi}] *
                         output_grad[{bi, ci, hi, wi}];

                for (int32_t wi = 0; wi < W; ++wi) {
                    input_grad[{bi, ci, hi, wi}] =
                        output[{bi, ci, hi, wi}] *
                        (output_grad[{bi, ci, hi, wi}] - S);
                }
            }
}