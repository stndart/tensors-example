#include "cuda/cuda_precision.h"

__half max(__half one, __half two) {
    return (one < two) ? two : one;
}

__half min(__half one, __half two) {
    return (one < two) ? one : two;
}