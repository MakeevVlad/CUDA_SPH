#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "particle.cuh"
class Neighbour
{
private:
	Neighbour* d_this;
public:
	//linearized n*n matrix. Must not be used on host!
	size_t* neighbour;
	//Must not be used on host!
	size_t* nei_numbers;

	size_t n;


	//int* matrix;
	// 
	//__device__ Neighbour(size_t);
	Neighbour(size_t);
	Neighbour() {};

	Neighbour* device()
	{
		return d_this;
	}

	//Returns the number of neighbours for n-th particle
	__device__ size_t NeigboursNumber(size_t);

	//Returns ptr to array with neighbours' numbers
	__device__ size_t* getNeighbours(size_t);


};

///Should be called with <<<N, N>>>
__global__ void initNeighbour(Particle*, Neighbour*);

__device__ bool nei(Particle*, Particle*);