#include "utils.cuh"



__device__ float scalar_prod(float* a, float* b)
{
	return a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
}

__global__ void tester(Neighbour* nei)
{
	printf("Tester:\n");
	
	printf("%d\n", nei->n);
	for (size_t i = 0; i < nei->n; ++i)
	{
		for (size_t j = 0; j < nei->nei_numbers[i]; ++j)
			printf("%d ", nei->neighbour[nei->n * i + j]);
		printf("\n");
	}
	
}
void solver(Particle* particle, float dt, size_t iterations, size_t pts_number)
{
	Neighbour neighbour(pts_number);
	Kernel kernel(3);

	std::ofstream file("test.txt");

	Particle* d_particle = device_particles_array(particle, pts_number);
	

	for (size_t it = 0; it < iterations; ++it)
	{
		initNeighbour<<<pts_number, pts_number>>>(d_particle, neighbour.device());
		//tester<<<1, 1>>> (neighbour.device());
		//cudaDeviceSynchronize();

		initParticles<<<pts_number, pts_number>>>(d_particle, kernel.device(), neighbour.device());
		
		//axelerations<<<pts_number, 3>>>(d_particle, kernel.device(), neighbour.device());
		
		//step << <pts_number, pts_number>> > (d_particle, kernel.device(), neighbour.device(), dt); 
		
		
		if (it % 100 == 0)
		{
			
			system("cls");
			std::cout << "Progress: " << it * 100 / iterations << " % " << std::endl
				<< "interior time = " << it * dt << " s";
			
			for (size_t p = 0; p < pts_number; ++p)
				file << particle[p].get_pos_i(0) << " " << particle[p].get_pos_i(1) << " " << particle[p].get_pos_i(2) << " ";
			file << std::endl;
		}

	}



}


__global__ void initParticles(Particle* particle, Kernel* kernel, Neighbour* neighbour)
{
	int i = blockIdx.x;

	dens(i, particle, *kernel, *neighbour);

	__syncthreads();
	if (threadIdx.x == 0)
	{
		particle[i].pressure = press(i, particle, *kernel, *neighbour);
		particle[i].ax.set(0, 0, 0);
	}
}

__global__ void axelerations(Particle* particle, Kernel* kernel, Neighbour* nei)
{
	int i = blockIdx.x;
	ax(i, particle, *kernel, *nei);
}

__device__ bool bounds(vec3 pos)
{
	return ((pos[0] > 11) || (pos[0] < -1) || (pos[1] > 11) || (pos[1] < -1));
}



__global__ void step(Particle* particle, Kernel* kernel, Neighbour* nei, float dt)
{
	//Particle number
	size_t i = blockIdx.x;

	//Coordinate number
	size_t j = threadIdx.x;
	if (j < 3)
	{
		particle[i].vel[j] += particle[i].ax[j] * dt;
		particle[i].pos[j] += particle[i].vel[j] * dt;
	}


	//Fixme: Delete this after the addition of meshes:
	__syncthreads();
	size_t a = 0;
	while (bounds(particle[i].pos))
	{
		if (a > 5)
		{
			printf("Loop ");
			break;
		}

		if (particle[i].pos[0] > 11 || particle[i].pos[0] < -1)
		{
			particle[i].vel[0] *= -1;
			particle[i].pos[0] += particle[i].vel[0] * dt;
		}
		if (particle[i].pos[1] > 11 || particle[i].pos[1] < -1)
		{
			particle[i].vel[1] *= -1;
			particle[i].pos[1] += particle[i].vel[1] * dt;
		}
		a++;
	}

}



//Fixme: add usage of shared memoory
__device__ void dens(size_t n, Particle* particle, Kernel& kernel, Neighbour& neighbour)
{
	int i = threadIdx.x;


	__shared__ float res_dens[N];


	if (i < neighbour.nei_numbers[n])
	{

		i = neighbour.neighbour[n * neighbour.n + i];

		atomicAdd(&particle[n].density, particle[i].mass * kernel.W(particle[n].pos, particle[i].pos, particle[n].h));
	}
}
__device__ float press(size_t n, Particle* particle, Kernel& kernel, Neighbour& neighbour)
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



//Fixme: add usage of shared memoory
__device__ void ax(size_t n, Particle* particle, Kernel& kernel, Neighbour& neighbour)
{

	int i = threadIdx.x;
	if (i >= neighbour.NeigboursNumber(n))
		return;

	if (i == 0)
	{
		atomicExch(particle[n].ax.r, 0);
		atomicExch(particle[n].ax.r + 1, 0);
		atomicExch(particle[n].ax.r + 2, 0);
	}

	__syncthreads();
	
	int j = neighbour.neighbour[n * neighbour.n + i];
	
	vec3 ax = kernel.gradW(particle[n].pos, particle[j].pos, particle[n].h) * (particle[n].pressure / ( particle[n].density * particle[n].density) +
		particle[j].pressure / (particle[j].density * particle[j].density)) * particle[j].mass;


	atomicAdd(particle[n].ax.r, ax[0]);
	atomicAdd(particle[n].ax.r + 1, ax[1]);
	atomicAdd(particle[n].ax.r + 2, ax[2]);
	
	__syncthreads();
	if (i == 0)
	{
		particle[n].ax *= (-1) / particle[n].mass;
		printf("%f %f %f", particle[n].ax[0], particle[n].ax[1], particle[n].ax[2]);
	}

}



