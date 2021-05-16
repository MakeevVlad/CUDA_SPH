#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ float scalar_prod(float* a, float* b)
{
	return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}