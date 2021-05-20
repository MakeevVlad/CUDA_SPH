
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#include "sph.cuh"
#include "particle.cuh"
#include "utils.cuh"

//#include "neighbour.cuh"

#include "mesh/primitives.h"


__global__ void test(Particle* p)
{
	printf("%f %f %f", p[0].pos[0], p[0].pos[1], p[0].pos[2]);
}

__global__ void tester1()
{
	printf("Tester:\n");
	/*
	printf("%f\n", pts[2].pos[0]);
	printf("%d\n", nei->n);
	for (size_t i = 0; i < nei->n; ++i)
	{
		for (size_t j = 0; j < nei->nei_numbers[i]; ++j)
			printf("%d ", nei->neighbour[nei->n * i + j]);
		printf("\n");
	}*/

};


int main()
{

	Particle* ps = new Particle[N];
	int p = sqrt(N);
	for (size_t i = 0; i <p; ++i)
		for (size_t j = 0; j < p; ++j)
		{
			ps[i * p + j].set_pos(i*1.1 - p/2 , j * 1.1 - p / 2, 0);
			ps[i * p + j].set_vel(0, 0, 0);
			ps[i * p + j].set_ax(0, 0, 0);
		}
	size_t pts_number = N;

	/* Generating from mesh */
	/* VolumeMesh mesh;
	mesh.construct_from_file("input.msh", "materials.mat");
	size_t pts_number = mesh.get_tetrahedra_number();

	// init masses and poses
	Particle* ps = new Particle[pts_number];
	for(size_t i = 0; i < pts_number; ++i) {
		mesh.initialize_particle(&(ps[i]), i);
	};
	*/


	float dt = 0.001;
	size_t iterations = 10000;
	solver(ps, dt, iterations, pts_number);


	return 0; 
}
