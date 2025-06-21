#ifndef USE_CUDA
using __half = float;
#define __half_max FLT_MAX
#else
#include <cuda_fp16.h>
#endif