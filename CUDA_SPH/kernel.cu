#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#include "sph.cuh"
#include "particle.cuh"
#include "utils.cuh"
#include "neighbour.cuh"

__global__ void test(Particle* p)
{
	printf("%f %f %f", p[0].pos[0], p[1].pos[1], p[0].pos[2]);
}

int main()
{
	Particle p1, p2;

	p1.set_pos(1, 2, 3);
	p2.set_pos(9, 10, 11);
	Particle ps[2] = { p1, p2 };

	test<<<1, 1>>>(device_particles_array(ps, 2));
	return 0; 
}
