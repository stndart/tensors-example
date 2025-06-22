#ifndef USE_CUDA
#include <float.h>
using __half = float;
#define __half_max FLT_MAX
#else
#include <cuda_fp16.h>
#endif

__half max(__half one, __half two);
__half min(__half one, __half two);