#include <iostream>
#include "neighbour.cuh"
/*
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
	
	cudaMalloc((void**)&d_this, sizeof(Neighbour));
	cudaMemcpy(d_this, this, sizeof(Neighbour), cudaMemcpyHostToDevice);

	size_t* neighbour_tmp = new size_t[n * n];
	size_t* nei_numbers_tmp = new size_t[n];
	//
	//int* matrix_tmp = new int[n * n];
	//Must not be used on host!
	



	cudaMalloc((void**)&nei_numbers, n * sizeof(size_t));
	cudaMalloc((void**)&neighbour, n * n * sizeof(size_t));

	cudaMemcpy(nei_numbers, nei_numbers_tmp, n * sizeof(size_t), cudaMemcpyHostToDevice);
	cudaMemcpy(neighbour, neighbour_tmp, n * n * sizeof(size_t), cudaMemcpyHostToDevice);
	//
	//cudaMemcpy(matrix, matrix_tmp, n * n * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(&(d_this->neighbour), &neighbour, sizeof(size_t*), cudaMemcpyHostToDevice);
	cudaMemcpy(&(d_this->nei_numbers), &nei_numbers, sizeof(size_t*), cudaMemcpyHostToDevice);
	//
	//cudaMemcpy(&(d_this->matrix), &matrix, sizeof(int*), cudaMemcpyHostToDevice);

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
	int i = blockIdx.x;
	int j = threadIdx.x;

	__shared__ bool tmp[N];
	tmp[j] = 0;

	
	if (i < nei->n && j < nei->n)
	{
		
		if ((pts[i].pos - pts[j].pos).abssquared() < 4 * pts[i].h * pts[i].h)
		{
			//printf("%f %f \n", (pts[i].pos - pts[j].pos).abssquared(), pts[i].h);
			tmp[j] = 1;

		}


		__syncthreads();
		if (j == 0)
		{
			nei->nei_numbers[i] = 0;
			for (size_t c = 0; c < N; ++c)
			{
				
				if (tmp[c])
				{
					nei->neighbour[i * nei->n + nei->nei_numbers[i]] = c;
					++nei->nei_numbers[i];
				}
			}
		}
	}
}

