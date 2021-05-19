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
__global__ void tester2( Particle* particle)
{
	printf("Tester:\n");

	for (size_t i = 0; i < 25; ++i)
	{
		printf("%f ", particle[i].density);
		
	}
	printf("\n");

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

		initParticles<<<pts_number, pts_number>>>(d_particle, kernel.device(), neighbour.device(), it);
		//tester2 << <1, 1 >> > (d_particle);
		
		axelerations<<<pts_number, pts_number >>>(d_particle, kernel.device(), neighbour.device());
		
		step << <pts_number, 3 >> > (d_particle, kernel.device(), neighbour.device(), dt);
		
		
		if (it % 10 == 0)
		{
			cudaDeviceSynchronize();
			update_particles(particle, d_particle, pts_number);
			if (it % 1000 == 0)
			{
				system("cls");
				std::cout << "Progress: " << it * 100 / iterations << " % " << std::endl
					<< "interior time = " << it * dt << " s";
			}
			for (size_t p = 0; p < pts_number; ++p)
				file << particle[p].pos[0] << " " << particle[p].pos[1] << " " << particle[p].pos[2] << " ";
			file << std::endl;
		}

	}

	file.close();


}


__global__ void initParticles(Particle* particle, Kernel* kernel, Neighbour* neighbour, size_t it)
{
	//printf("HERE %d \n", it);
	int n = blockIdx.x;

	dens(n, particle, *kernel, *neighbour);



	if (threadIdx.x == 0)
	{
		particle[n].pressure = press(n, particle, *kernel, *neighbour);
		particle[n].ax.set(0, 0, 0);
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
	/*
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
	}*/

}




__device__ void dens(size_t n, Particle* particle, Kernel& kernel, Neighbour& neighbour)
{
	int i = threadIdx.x;
	int j = 0;

	__shared__ float res_dens[N];
	res_dens[i] = 0;
	//printf("%d %d \n", i, neighbour.nei_numbers[n]);
	


	if (i < neighbour.nei_numbers[n])
	{

		j = neighbour.neighbour[n * neighbour.n + i];
		
		res_dens[i] = particle[j].mass* kernel.W(particle[n].pos, particle[j].pos, particle[n].h);
		
	}

	
	__syncthreads();
	
	if (i == 0)
	{
		particle[n].density = 0;
		for (size_t c = 0; c < neighbour.nei_numbers[n]; ++c)
			particle[n].density += res_dens[c];

		//printf("%f \n", particle[n].density);
	}
}
__device__ float press(size_t n, Particle* particle, Kernel& kernel, Neighbour& neighbour)
{
	return particle[n].A * powf(particle[n].density, particle[n].gamma);
}



//Fixme: add usage of shared memoory
__device__ void ax(size_t n, Particle* particle, Kernel& kernel, Neighbour& neighbour)
{
	//printf("HERE\n");

	int i = threadIdx.x;

	__shared__ vec3 res_ax[N];
	res_ax[i] = vec3(0, 0, 0);

	if (i < neighbour.nei_numbers[n] )
	{

		int j = neighbour.neighbour[n * neighbour.n + i];
		if (j != n)
			res_ax[i] = kernel.gradW(particle[n].pos, particle[j].pos, particle[n].h) *(particle[n].pressure / (particle[n].density * particle[n].density) +
			particle[j].pressure / (particle[j].density * particle[j].density)) * particle[j].mass;

	}

	__syncthreads();
	if (i == 0)
	{
		particle[n].ax = vec3(0, 0, 0);
		for (size_t c = 0; c < neighbour.nei_numbers[n]; ++c)
			particle[n].ax += res_ax[c];

		particle[n].ax *= (-1) / particle[n].mass;

		
	}
}



