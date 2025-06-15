#pragma once

class Matrix;
class Tensor4D;

void tensor_to_matrix_reshape(Tensor4D &TA, Matrix &TB);
void tensor_to_matrix_reshape_const(const Tensor4D &TA, Matrix &TB);
void matrix_to_tensor_reshape(Matrix &TA, Tensor4D &TB);
void matrix_to_tensor_reshape_const(const Matrix &TA, Tensor4D &TB);

void cpu_matrix_gemm(const Matrix &A, const Matrix &B, Matrix &C);