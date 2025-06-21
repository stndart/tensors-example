#pragma once

#include <optional>

#include "tensor.h"
#include "cpu/padding.h"

class AvgPooling {
  public:

  private:
    size_t H_pad_, W_pad_, H_stride_, W_stride_, H_size_, W_size_;
    Index4 inputDims, outputDims;
    PaddingMode padding;

  public:
    AvgPooling(Index4 input_shape,
               PaddingMode padding_mode = PaddingMode::REPLICATION_PADDING,
               std::optional<size_t> H_pad = std::nullopt,
               std::optional<size_t> W_pad = std::nullopt,
               std::optional<size_t> H_stride = std::nullopt,
               std::optional<size_t> W_stride = std::nullopt,
               std::optional<size_t> H_size = std::nullopt,
               std::optional<size_t> W_size = std::nullopt);
    ~AvgPooling() {}

    void forward(const Tensor4D &input, Tensor4D &output); // const is removed because of cache operations
    void backward(const Tensor4D &output_gradient,
                  Tensor4D &input_gradient) const;

    // Getters
    int H_pad() const { return H_pad_; }
    int W_pad() const { return W_pad_; }
    int H_stride() const { return H_stride_; }
    int W_stride() const { return W_stride_; }
    int H_size() const { return H_size_; }
    int W_size() const { return W_size_; }
};
