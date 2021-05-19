#include "sph.cuh"

Kernel::Kernel(size_t D_)
{
	D = D_;

	if (D == 1) C = 0.6667; // = 2/3
	else if (D == 2) C = 0.4547; // = 10/(7pi)
	else if (D == 3) C = 0.3183; // = 1/pi

	//cudaMalloc((void**)&d_this, sizeof(Kernel));
	//cudaMemcpy(d_this, this, sizeof(Kernel), cudaMemcpyHostToDevice);
}

__device__ float Kernel::W(vec3 r1, vec3 r2, float h)
{
	float k = abs(r1 - r2) / h;
	if (k <= 1)
	{
		return C * (1 - 1.5 * powf(k, 2) + 0.75 * powf(k, 3)) / powf(h, D);
	}
	else if (k > 1 && k <= 2)
	{
		return C * 0.25 * powf(2 - k, 3) / powf(h, D);
	}
	else return 0;
}

__device__ vec3 Kernel::gradW(vec3 r1, vec3 r2, float h)
{
	double k = abs(r1 - r2) / h;
	if (k <= 1)
	{
		return (r1 - r2) * C * (-3 * k + 2.25 * powf(k, 2)) / (powf(h, D) * abs(r1 - r2));
	}
	else if (k > 1 && k <= 2)
	{
		return (r1 - r2) * C * 0.75 * powf(2 - k, 2) / (powf(h, D) * abs(r1 - r2));
	}
	else return vec3(0, 0, 0);
}