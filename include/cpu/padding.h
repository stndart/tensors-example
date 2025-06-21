#pragma once

#include "tensor.h"

enum class PaddingMode {
    ZERO_PADDING,
    REPLICATION_PADDING,
    REFLECTION_PADDING,
    CIRCULAR_PADDING
};

template <typename TensorType>
typename TensorType::Index padded_access(const TensorType &tensor, typename TensorType::Index idx,
                     PaddingMode mode);

// Explicit instantiation declarations
extern template Index4 padded_access<Tensor4D>(const Tensor4D &tensor,
                                               Index4 idx, PaddingMode mode);
extern template Index2 padded_access<Matrix>(const Matrix &tensor, Index2 idx,
                                             PaddingMode mode);