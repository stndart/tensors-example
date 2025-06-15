#pragma once

#include "matrix.h"

#ifdef USE_CUDA
void cuda_matrix_gemm(const Matrix &A, const Matrix &B, Matrix &C);
#endif