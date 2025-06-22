#include "cpu/cpu_tensors.h"

#include <cstring>

void matrix_to_tensor_reshape(Matrix &TA, Tensor4D &TB) {
    if (TA.size() != TB.size()) {
        throw std::runtime_error("reshape: Tensor dimensions mismatch");
        return;
    }

    TB.clear();
    TB.data_ = TA.data_;
    TB.gpu_data_ = TA.gpu_data_;
    TA.data_ = nullptr;
    TA.gpu_data_ = nullptr;
    TB.axes_order = {0, 1, TA.axes_order[0] + 2, TA.axes_order[1] + 2};
}

void matrix_to_tensor_reshape_const(const Matrix &TA, Tensor4D &TB) {
    if (TA.size() != TB.size()) {
        throw std::runtime_error("reshape: Tensor dimensions mismatch");
        return;
    }

    if (TA.data_ != nullptr)
        memcpy(TB.data_, TA.data_, sizeof(__half) * TA.size());

#ifdef USE_CUDA
    if (TA.gpu_data_ != nullptr)
        cudaMemcpy(TB.data_, TA.data_, sizeof(__half) * TA.size(),
                   cudaMemcpyDeviceToDevice);
#endif

    TB.axes_order = {0, 1, TA.axes_order[0] + 2, TA.axes_order[1] + 2};
}

void tensor_to_matrix_reshape(Tensor4D &TA, Matrix &TB) {
    if (TA.size() != TB.size())
        throw std::runtime_error("reshape: Matrix dimensions mismatch");

    if (TA.dimW() > 1 || TA.dimX() > 1)
        throw std::runtime_error("reshape: Tensor is not matrix-like");

    TB.clear();
    TB.data_ = TA.data_;
    TB.gpu_data_ = TA.gpu_data_;
    TA.data_ = nullptr;
    TA.gpu_data_ = nullptr;

    if (TA.axes_order[0] < TA.axes_order[1])
        TB.axes_order = {0, 1};
    else
        TB.axes_order = {1, 0};
}

void tensor_to_matrix_reshape_const(const Tensor4D &TA, Matrix &TB) {
    if (TA.size() != TB.size()) {
        throw std::runtime_error("reshape: Matrix dimensions mismatch");
        return;
    }

    if (TA.data_ != nullptr)
        memcpy(TB.data_, TA.data_, sizeof(__half) * TA.size());

#ifdef USE_CUDA
    if (TA.gpu_data_ != nullptr)
        cudaMemcpy(TB.data_, TA.data_, sizeof(__half) * TA.size(),
                   cudaMemcpyDeviceToDevice);
#endif
}