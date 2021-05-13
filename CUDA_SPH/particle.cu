#include "particle.cuh"


void Particle::set_mass(float _mass)
{
	mass = _mass;
}

void Particle::set_pos(float x, float y, float z)
{
	pos[0] = x;
	pos[1] = y;
	pos[2] = z;
}
void Particle::set_pos_i(float q, int i)
{
	pos[i] = q;
}

void Particle::set_vel(float x, float y, float z)
{
	vel[0] = x;
	vel[1] = y;
	vel[2] = z;
}
void Particle::set_vel_i(float q, int i)
{
	vel[i] = q;
}






ParticleAllocator::ParticleAllocator(Particle* p)
{
	cudaMalloc((void**)&d_this, sizeof(Particle));
	cudaMemcpy(d_this, &p, sizeof(Particle), cudaMemcpyHostToDevice);
}