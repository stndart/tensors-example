#ifndef USE_CUDA
using __half = float;
#else
#include <cuda_fp16.h>
#endif