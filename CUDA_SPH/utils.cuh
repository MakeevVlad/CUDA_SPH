#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <fstream>
#include "particle.cuh"
#include "neighbour.cuh"
#include "sph.cuh"
#include "reflection.cuh"
void solver(Particle*, float, size_t, size_t);

__device__ void eiler_scheme(Particle*, Kernel&, double, size_t);

__global__ void reflective_step(Particle* particle, Kernel* kernel, Neighbour* nei, float dt);
__global__ void step(Particle*, Kernel*, Neighbour*,  float);
__global__ void axelerations(Particle*, Kernel*, Neighbour*);

// particle's axeleration, dv/dt 
__device__ void ax(size_t, Particle*, Kernel&, Neighbour&);

__device__ float energy(Particle*);

//density of smooth particle (particle number, vector with particles, kernel)
__device__ void dens(size_t n, Particle*, Kernel&, Neighbour&);

//pressure (-||-)
__device__ float press(size_t, Particle*, Kernel&, Neighbour&);
//adiabatic exponent (-||-)
__device__ float gamma(size_t, Particle*, Kernel&);




//rotor and divergence of velocity's field (-||-)
__device__ float  divVel(size_t, Particle*, Kernel&);
__device__ vec3	rotVel(size_t, Particle*, Kernel&);

__device__ vec3 lapVel(size_t, Particle*, Kernel&);

__device__ float scalar_prod(float*, float*);


//Should be called with <<<N, maxNeighbours>>>
__global__ void initParticles(Particle*, Kernel*, Neighbour*);