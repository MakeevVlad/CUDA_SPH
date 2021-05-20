#include "particle.cuh"



Particle::Particle()
{
	mass = 1;
	density = 1;
	pressure = 1;
	h = 1;
	gamma = 1;
	A = 10;

	nu = 1000;
	rho0 = 0.5;
	k = 1;
	this->set_pos(0, 0, 0);
	this->set_vel(0, 0, 0);
	this->set_ax(0, 0, 0);
};

void Particle::allocate()
{
	cudaMalloc((void**)&d_this, sizeof(Particle));
	cudaMemcpy(d_this, this, sizeof(Particle), cudaMemcpyHostToDevice);
}


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

void Particle::set_ax(float x, float y, float z)
{
	ax[0] = x;
	ax[1] = y;
	ax[2] = z;
}

float Particle::get_pos_i(int i)
{
	Particle* tmp = new Particle;
	cudaError_t err = cudaMemcpy(tmp, d_this, sizeof(float), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
		printf(cudaGetErrorString(err));
	
	pos = tmp->pos;
	delete tmp;
	return pos[i];
}

void update_particles(Particle* ps, Particle* d_ps, size_t ps_num)
{

	cudaMemcpy(ps, d_ps, ps_num * sizeof(Particle), cudaMemcpyDeviceToHost);
}



Particle* Particle::device()
{
	return d_this;
}


Particle* device_particles_array(Particle* ps, size_t n)
{
	Particle* d_ps;

	cudaMalloc((void**)&d_ps, n * sizeof(Particle));
	cudaMemcpy(d_ps, ps, n * sizeof(Particle), cudaMemcpyHostToDevice);
	//for (size_t i = 0; i < n; ++i)
	//	cudaMemcpy(d_ps + i, ps + i, sizeof(Particle), cudaMemcpyHostToDevice);

	return d_ps;
}


