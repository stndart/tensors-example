#include "cpu/padding.h"

template <typename TensorType>
typename TensorType::Index padded_access(const TensorType &tensor,
                                         typename TensorType::Index idx,
                                         PaddingMode mode) {
    if (idx < tensor.vsize())
        return tensor[idx];

    TensorType::Index vidx(idx);
    switch (mode) {
    case PaddingMode::ZERO_PADDING:
        for (size_t i = 0; i < tensor.NDIMS; ++i)
            vidx[i] = -1;
        break;
    case PaddingMode::REPLICATION_PADDING:
        for (size_t i = 0; i < tensor.NDIMS; ++i) {
            if (vidx[i] >= tensor.vsize()[i])
                vidx[i] = tensor.vsize()[i] - 1;
        }
        break;
    case PaddingMode::REFLECTION_PADDING:
        for (size_t i = 0; i < tensor.NDIMS; ++i) {
            const auto size = tensor.vsize()[i];
            if (size == 0)
                continue; // Avoid division by zero

            if (size == 1) {
                vidx[i] = 0;
            } else {
                const auto period = 2 * (size - 1);
                // Fold index into [0, period-1] range
                vidx[i] %= period;
                // Reflect if index >= size
                if (vidx[i] >= size) {
                    vidx[i] = period - vidx[i];
                }
            }
        }
    case PaddingMode::CIRCULAR_PADDING:
        for (size_t i = 0; i < tensor.NDIMS; ++i) {
            const auto size = tensor.vsize()[i];
            if (size != 0) {
                vidx[i] %= size;
            }
        }
    }

    return vidx;
}

// Explicit instantiations
template Index4 padded_access<Tensor4D>(const Tensor4D &tensor, Index4 idx,
                                        PaddingMode mode);
template Index2 padded_access<Matrix>(const Matrix &tensor, Index2 idx,
                                      PaddingMode mode);