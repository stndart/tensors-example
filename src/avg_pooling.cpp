#include "avg_pooling.h"
#include "cpu/cpu_avg_pooling.h"
#include "cuda/cuda_memory.cu"

AvgPooling::AvgPooling(Index4 input_shape, PaddingMode padding_mode,
                       std::optional<size_t> H_pad, std::optional<size_t> W_pad,
                       std::optional<size_t> H_stride,
                       std::optional<size_t> W_stride,
                       std::optional<size_t> H_size,
                       std::optional<size_t> W_size)
    : inputDims(input_shape), padding(padding_mode) {
    H_pad_ = H_pad.value_or(0);
    W_pad_ = W_pad.value_or(0);
    H_stride_ = H_stride.value_or(1);
    W_stride_ = W_stride.value_or(1);
    H_size_ = H_size.value_or(2);
    W_size_ = W_size.value_or(2);

    size_t output_H = (inputDims.y + 2 * H_pad_ - H_size_) / H_stride_ + 1;
    size_t output_W = (inputDims.z + 2 * W_pad_ - W_size_) / W_stride_ + 1;

    outputDims = Index4{inputDims.w, inputDims.x, output_H, output_W};
}

void AvgPooling::forward(const Tensor4D &input, Tensor4D &output) {
    if (output.vsize() != outputDims)
        throw std::runtime_error("Forward: output dimensions mismatch");
    if (input.vsize() != inputDims)
        throw std::runtime_error("Forward: input dimensions mismatch");

#ifdef USE_CUDA
    try {
        cuda_avg_pooling(input, output, H_pad_, W_pad_, H_stride_, W_stride_,
                         H_size_, W_size_);
        return;
    } catch (const std::exception &e) {
        std::cerr << "CUDA error: " << e.what();
        std::cerr << "Falling back to CPU\n";
    }
#endif

    // CPU fallback
    output.allocate_memory();
    cpu_avg_pooling(input, output, padding, H_pad_, W_pad_, H_stride_,
                    W_stride_, H_size_, W_size_);
}

void AvgPooling::backward(const Tensor4D &output_gradient,
                          Tensor4D &input_gradient) const {

    if (output_gradient.vsize() != outputDims)
        throw std::runtime_error(
            "Backward: output gradient dimensions mismatch");
    if (input_gradient.vsize() != inputDims)
        throw std::runtime_error(
            "Backward: input gradient dimensions mismatch");

#ifdef USE_CUDA
    try {
        cuda_avg_pooling_backward(output_gradient, input_gradient, padding,
                                  H_pad_, W_pad_, H_stride_, W_stride_, H_size_,
                                  W_size_);
        return;
    } catch (const std::exception &e) {
        std::cerr << "CUDA error: " << e.what();
        std::cerr << "Falling back to CPU\n";
    }
#endif

    // CPU fallback
    input_gradient.allocate_memory();
    cpu_avg_pooling_backward(output_gradient, input_gradient, padding, H_pad_,
                             W_pad_, H_stride_, W_stride_, H_size_, W_size_);
}