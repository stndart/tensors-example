#include "cpu/padding.h"

template <typename TensorType>
typename TensorType::Index padded_access(const TensorType &tensor,
                                         typename TensorType::Index idx,
                                         PaddingMode mode) {
    if (idx >= 0 && idx < tensor.vsize())
        return idx;

    typename TensorType::Index vidx(idx);
    switch (mode) {
    case PaddingMode::ZERO_PADDING:
        for (size_t i = 0; i < tensor.NDIMS; ++i)
            vidx[i] = -1;
        break;
    case PaddingMode::REPLICATION_PADDING:
        for (size_t i = 0; i < tensor.NDIMS; ++i) {
            if (vidx[i] < 0)
                vidx[i] = 0;
            else if (vidx[i] >= tensor.vsize()[i])
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
                if (vidx[i] < 0)
                    vidx[i] = -vidx[i];
                else if (vidx[i] >= size)
                    vidx[i] = 2 * (size - 1) - vidx[i];
            }
        }
    case PaddingMode::CIRCULAR_PADDING:
        for (size_t i = 0; i < tensor.NDIMS; ++i) {
            const auto size = tensor.vsize()[i];
            if (size != 0)
                vidx[i] = (vidx[i] % size + size) % size;
        }
    }

    return vidx;
}

// Explicit instantiations
template Index4 padded_access<Tensor4D>(const Tensor4D &tensor, Index4 idx,
                                        PaddingMode mode);
template Index2 padded_access<Matrix>(const Matrix &tensor, Index2 idx,
                                      PaddingMode mode);

__half padded(const Tensor4D &input, Index4 idx, PaddingMode padding) {
    Index4 vidx = padded_access(input, idx, padding);
    // std::cout << "Before padding: " << idx << "\n";
    // std::cout << "After padding: " << vidx << "\n";
    if (vidx >= 0 && vidx < input.vsize()) {
        for (size_t i = 0; i < 4; ++i)
            if (vidx[i] == -1)
                return 0;
        return input[vidx];
    }
    return 0;
}