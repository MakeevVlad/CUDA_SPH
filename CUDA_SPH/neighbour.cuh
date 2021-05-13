#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "particle.cuh"
class Neighbour
{
public:
	size_t* web;
	size_t size;

	Neighbour(size_t);

	

	__device__ void unary_init(Particle&);

	//Returns vector with neighbours' numbers
	__device__ int* operator()(size_t n);
	__device__ int* operator()(Particle&);

};

__global__ void init(Particle*, Kernel*);

__device__ bool nei(Particle*, Particle*);