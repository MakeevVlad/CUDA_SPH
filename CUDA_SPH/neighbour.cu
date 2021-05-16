#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "neighbour.cuh"

/*
__device__ Neighbour::Neighbour(size_t n)
{
	nei_numbers = (size_t*)malloc(n * sizeof(size_t));
	for (size_t i = 0; i < n; ++i)
		nei_numbers[i] = 0;

}
*/
Neighbour::Neighbour(size_t _n)
{
	n = _n;

	cudaMalloc((void**)&d_this->nei_numbers, n * sizeof(size_t));
	cudaMalloc((void**)&d_this->neighbour, n * n * sizeof(size_t));

	cudaMalloc((void**)&d_this, sizeof(Neighbour));
	cudaMemcpy(d_this, this, sizeof(Neighbour), cudaMemcpyHostToDevice);
}

__device__ size_t Neighbour::NeigboursNumber(size_t i)
{
	return nei_numbers[i];
}

__device__ size_t* Neighbour::getNeighbours(size_t i)
{
	return neighbour + n*i;
}

__global__ void initNeighbour(Particle* pts, Neighbour* nei)
{
	return;
}
