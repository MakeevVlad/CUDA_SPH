#include "utils.cuh"

__device__ float scalar_prod(float* a, float* b)
{
	return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}

void axelerations(Particle* particle, Kernel* kernel, Neighbour* nei)
{
	size_t i = blockIdx.x;
	ax(i, particle, *kernel, *nei);
}
void step(Particle* particle, Kernel* kernel, Neighbour* nei, float dt)
{
	size_t i = blockIdx.x;
	size_t j = threadIdx.x;
	if (j < 3)
	{
		particle[i].vel[j] += particle[i].ax[j] * dt;
		particle[i].pos[j] += particle[i].vel[j] * dt;
	}
}

void dens(size_t n, Particle* particle, Kernel& kernel, Neighbour& neighbour)
{
	size_t i = threadIdx.x;
	if (i >= neighbour.nei_numbers[n])
		return;
	size_t j = neighbour.neighbour[i];
	atomicAdd(&particle[n].density, particle[j].get_mass() * kernel.W(particle[n].pos, particle[j].pos, particle[n].h));

}
float press(size_t n, Particle* particle, Kernel& kernel, Neighbour& neighbour)
{
	return particle[n].A * pow(particle[n].density, particle[n].gamma);
}

/*
float divVel(size_t n, Particle* particle, Kernel& kernel, Neighbour& nei)
{
	float divV = 0;
	for (size_t i : nei(n))
		divV += (particle[i].vel - particle[n].vel) *
		kernel.gradW(particle[n].pos, particle[i].pos, particle[n].h) *
		particle[i].get_mass();

	return divV / dens(n, particle, kernel);

}

vec3 rotVel(size_t n, float dt, Particle* particle, Kernel& kernel, Neighbour& nei)
{
	vec3 rotV = 0;
	for (size_t i : nei(n))
		rotV += (particle[i].vel - particle[n].vel) /
		kernel.gradW(particle[n].pos, particle[i].pos, particle[n].h) *
		particle[i].get_mass();

	return rotV / dens(n, particle, kernel);
}*/

void initParticles(Particle* particle, Kernel* kernel, Neighbour* neighbour)
{
	size_t i = blockIdx.x;
	if (blockIdx.x >= neighbour->n)
		return;
	 
	dens(i, particle, *kernel, *neighbour);
	if (threadIdx.x == 0)
	{
		particle[i].pressure = press(i, particle, *kernel, *neighbour);
		particle[i].ax.set(0, 0, 0);
	}
}





void ax(size_t n, Particle* particle, Kernel& kernel, Neighbour& neighbour)
{

	size_t j = blockIdx.x;
	vec3 ax = kernel.gradW(particle[n].pos, particle[j].pos, particle[n].h) * (particle[n].pressure / ( particle[n].density * particle[n].density) +
		particle[j].pressure / (particle[j].density * particle[j].density) * particle[j].get_mass());

	atomicAdd(particle[n].pos.r, ax[0]);
	atomicAdd(particle[n].pos.r + 1, ax[1]);
	atomicAdd(particle[n].pos.r + 2, ax[2]);
	
	__syncthreads();
	if (threadIdx.x == 0)
		particle[n].ax *= (-1) / particle[n].get_mass();

}

float U_lj(vec3 r1, vec3 r2)
{
	float s = 0.89;
	float e = 4;
	float r = abs(r1 - r2);
	float s_r = pow(s / r, 6);

	return 4 * e * (s_r * s_r - s_r);
}

