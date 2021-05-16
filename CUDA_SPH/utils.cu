#include "cuda_runtime.h"
#include "device_launch_parameters.h"Particle* particle
#include "utils.cuh"

__device__ float scalar_prod(float* a, float* b)
{
	return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}


float dens(size_t n, Particle* particle, Kernel& kernel)
{
	float dens = 0;
	for (Particle& p : particle)
		dens += p.get_mass() * kernel.W(particle[n].pos, p.pos, particle[n].h);

	return dens;
}
float dens(size_t n, Particle* particle, Kernel& kernel, Neighbour& neighbour)
{
	float dens = 0;
#pragma omp parallel for reduction(+:dens)
	for (size_t i : neighbour(n))
		dens += particle[i].get_mass() * kernel.W(particle[n].pos, particle[i].pos, particle[n].h);

	return dens;
}
float press(size_t n, Particle* particle, Kernel& kernel, Neighbour& neighbour)
{
	return particle[n].A * pow(particle[n].density, particle[n].gamma);
}

void adapt_h(size_t n, float dt, Particle* particle, Kernel& kernel)
{
	particle[n].h = particle[n].h * (1 + divVel(n, particle, kernel) / kernel.D);
}

float divVel(size_t n, Particle* particle, Kernel& kernel)
{
	float divV = 0;
	for (Particle& p : particle)
		if (nei(particle[n], p))
			divV += (p.vel - particle[n].vel) *
			kernel.gradW(particle[n].pos, p.pos, particle[n].h) *
			p.get_mass();

	return divV / dens(n, particle, kernel);

}
float divVel(size_t n, Particle* particle, Kernel& kernel, Neighbour& nei)
{
	float divV = 0;
	for (size_t i : nei(n))
		divV += (particle[i].vel - particle[n].vel) *
		kernel.gradW(particle[n].pos, particle[i].pos, particle[n].h) *
		particle[i].get_mass();

	return divV / dens(n, particle, kernel);

}

vec3 rotVel(size_t n, float dt, Particle* particle, Kernel& kernel)
{
	vec3 rotV = 0;
	for (Particle& p : particle)
		if (nei(particle[n], p))
			rotV += (p.vel - particle[n].vel) /
			kernel.gradW(particle[n].pos, p.pos, particle[n].h) *
			p.get_mass();

	return rotV / dens(n, particle, kernel);
}vec3 rotVel(size_t n, float dt, Particle* particle, Kernel& kernel, Neighbour& nei)
{
	vec3 rotV = 0;
	for (size_t i : nei(n))
		rotV += (particle[i].vel - particle[n].vel) /
		kernel.gradW(particle[n].pos, particle[i].pos, particle[n].h) *
		particle[i].get_mass();

	return rotV / dens(n, particle, kernel);
}

void refresh(Particle* particle, Kernel& kernel, Neighbour& neighbour)
{
	for (size_t i = 0; i < particle.size(); ++i)
	{
		particle[i].density = dens(i, particle, kernel, neighbour);
		particle[i].pressure = press(i, particle, kernel, neighbour);
	}
}



vec3 ax(size_t n, Particle* particle, Kernel& kernel, Neighbour& neighbour)
{


	vec3 ax(0, 0, 0);
#pragma omp parallel for reduction(+:ax)
	for (size_t i : neighbour(n))
	{
		if (i != n)
			ax += kernel.gradW(particle[n].pos, particle[i].pos, particle[n].h) *
			particle[i].get_mass() * U_lj(particle[n].pos, particle[i].pos) / particle[i].density;
	}
	return ax * (-1) / particle[n].get_mass();
}

vec3 ax_inv_Eu(size_t n, Particle* particle, Kernel& kernel, Neighbour& neighbour)
{
	vec3 ax(0, 0, 0);
	vec3 g(0, -5, 0);

#pragma omp parallel for reduction(+:ax)
	for (size_t i : neighbour(n))
	{
		if (i != n)
		{
			ax += kernel.gradW(particle[n].pos, particle[i].pos, particle[n].h) * (particle[n].pressure / particle[n].density / particle[n].density +
				particle[i].pressure / particle[i].density / particle[i].density) * particle[i].get_mass();
		}
	}
	return ax * (-1) / particle[n].get_mass() + g;
}

float U_lj(vec3 r1, vec3 r2)
{
	float s = 0.89;
	float e = 4;
	float r = abs(r1 - r2);
	float s_r = pow(s / r, 6);

	return 4 * e * (s_r * s_r - s_r);
}

