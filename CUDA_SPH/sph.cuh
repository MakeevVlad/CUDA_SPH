#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_vector_math.cuh"


class Kernel
{
private:
	Kernel* d_this;
public:
	//Dimention
	size_t D;
	double C = 0;

	Kernel(size_t);
	~Kernel();
	Kernel* device()
	{
		return d_this;
	}

	//Returns the value of smothing kernel W(r_1, r_2, h)
	__device__ float W(vec3, vec3, float);
	//Returns the value of gradient of smothing kernel gradW(r_1, r_2, h)
	__device__ vec3 gradW(vec3, vec3, float);

	__device__ float lapW(vec3, vec3, float);
};