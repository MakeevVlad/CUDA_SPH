#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



class Kernel
{
public:
	//Dimention
	size_t D;
	double C = 0;

	Kernel(size_t);

	//Returns the value of smothing kernel W(r_1, r_2, h)
	__device__ float W(float*, float*, float);
	//Returns the value of gradient of smothing kernel gradW(r_1, r_2, h)
	__device__ float3 gradW(float*, float*, double);

};