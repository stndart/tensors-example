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
}

void tensor_to_matrix_reshape(Tensor4D &TA, Matrix &TB) {
    if (TA.size() != TB.size()) {
        throw std::runtime_error("reshape: Matrix dimensions mismatch");
        return;
    }

    TB.clear();
    TB.data_ = TA.data_;
    TB.gpu_data_ = TA.gpu_data_;
    TA.data_ = nullptr;
    TA.gpu_data_ = nullptr;
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