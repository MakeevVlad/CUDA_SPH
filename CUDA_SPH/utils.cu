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
	int capture_rate = 10;
	int initNeighbours_rate = 5;
	int information_refresh_rate = 1000;


	Neighbour neighbour(pts_number);
	Kernel kernel(3);

	std::ofstream file("results/data/result3drop.txt");
	//std::ofstream fileax("testax.txt");
	Particle* d_particle = device_particles_array(particle, pts_number);
	
	//maximal quantity of neighbours
	size_t mxn = 0;
	size_t* d_mxn;
	cudaMalloc((void**)&d_mxn, sizeof(size_t));
	cudaMemcpy(d_mxn, &mxn, sizeof(size_t), cudaMemcpyHostToDevice);

	for (size_t it = 0; it < iterations; ++it)
	{

		if (it % initNeighbours_rate == 0)
		{
			initNeighbour << <pts_number, pts_number >> > (d_particle, neighbour.device(), d_mxn);
			//cudaMemcpy(&mxn, d_mxn, sizeof(size_t), cudaMemcpyDeviceToHost);
		}
		initParticles<<<pts_number, pts_number >>>(d_particle, kernel.device(), neighbour.device());
		//tester2 << <1, 1 >> > (d_particle);
		
		axelerations<<<pts_number, pts_number >>>(d_particle, kernel.device(), neighbour.device());
		
		step << <pts_number, 3 >> > (d_particle, kernel.device(), neighbour.device(), dt);
		//reflective_step <<<1, pts_number>>> (d_particle, kernel.device(), neighbour.device(), dt);
		
		if (it % capture_rate == 0)
		{
			cudaDeviceSynchronize();
			update_particles(particle, d_particle, pts_number);
			if (it % information_refresh_rate == 0)
			{
				system("cls");
				std::cout << "Progress: " << it * 100 / iterations << " % " << std::endl
					<< "interior time = " << it * dt << " s";
			}
			for (size_t p = 0; p < pts_number; ++p)
			{
				file << particle[p].pos[0] << " " << particle[p].pos[1] << " " << particle[p].pos[2] << " ";
				//fileax << particle[p].ax[0] << " " << particle[p].ax[1] << " " << particle[p].ax[2] << " ";
			}

			file << std::endl;
			//fileax << std::endl;
		}

	}

	file.close();
	//fileax.close();

	cudaFree(d_particle);

}


__global__ void initParticles(Particle* particle, Kernel* kernel, Neighbour* neighbour)
{
	//printf("HERE %d \n", it);
	int n = blockIdx.x;

	dens(n, particle, *kernel, *neighbour);


	__syncthreads();
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



		//Fixme: Delete this after the addition of meshes:
		
		//__syncthreads();
		


			if (particle[i].pos[j] > cubeSize || particle[i].pos[j] < -cubeSize)
			{
				particle[i].vel[j] *= -1 * 0.6;
				particle[i].pos[j] += particle[i].vel[j] * dt;
			}
			


	}

}

__global__ void reflective_step(Particle* particle, Kernel* kernel, Neighbour* nei, float dt)
{
	
	size_t i = threadIdx.x;

	particle[i].vel += particle[i].ax * dt;


	float tr[3][3] = { {-50, -20, -10}, {50, -20, -10}, {0, -20, 10} };
	point_t* trp = new point_t[3];
	for (size_t c = 0; c < 3; ++c)
	{
		trp[c] = new coord_t[3];
		for (size_t d = 0; d < 3; ++d)
		{
			trp[c][d] = tr[c][d];
		}
	}

	reflect(dt, particle[i].pos.r, particle[i].vel.r, trp);

	for (size_t c = 0; c < 3; ++c)
	{
		delete[] trp[c];
	}

	delete[] trp;
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
	//return particle[n].A * powf(particle[n].density, particle[n].gamma);


	return particle[n].k * (particle[n].density - particle[n].rho0);
}



//Fixme: add usage of shared memoory
__device__ void ax(size_t n, Particle* particle, Kernel& kernel, Neighbour& neighbour)
{
	//printf("HERE\n");
	vec3 g(0, -2, 0);
	int i = threadIdx.x;

	__shared__ vec3 res_ax[N];
	res_ax[i] = vec3(0, 0, 0);

	if (i < neighbour.nei_numbers[n] )
	{

		int j = neighbour.neighbour[n * neighbour.n + i];
		if (j != n)
		{
			res_ax[i] =  kernel.gradW(particle[n].pos, particle[j].pos, particle[n].h) * (-1) * (particle[n].pressure / (particle[n].density * particle[n].density) +
				particle[j].pressure / (particle[j].density * particle[j].density)) * particle[j].mass;
			res_ax[i] += (particle[j].vel - particle[n].vel) * particle[n].nu *
				kernel.lapW(particle[n].vel, particle[j].vel, particle[n].h) * particle[j].mass / particle[j].density;
			res_ax[i] += g;
		}
	}

	__syncthreads();
	if (i == 0)
	{
		
		for (size_t c = 0; c < neighbour.nei_numbers[n]; ++c)
			particle[n].ax += res_ax[c];
	}
}




