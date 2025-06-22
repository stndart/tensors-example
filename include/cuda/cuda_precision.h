#ifndef USE_CUDA
#include <float.h>
using __half = float;
#define __half_max FLT_MAX
#else
#include <cuda_fp16.h>
#endif