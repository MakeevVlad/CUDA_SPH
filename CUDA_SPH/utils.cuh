#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "particle.cuh"
#include "neighbour.cuh"
#include "sph.cuh"
__device__ void eiler_scheme(Particle*, Kernel&, double, size_t);

// particle's axeleration, dv/dt 
__device__ float3 ax(size_t, Particle*, Kernel&, Neighbour&);

__device__ float energy(Particle*);

//density of smooth particle (particle number, vector with particles, kernel)
__device__ float dens(size_t, Particle*, Kernel&);
__device__ float dens(size_t n, Particle*, Kernel&, Neighbour&);

//pressure (-||-)
__device__ float press(size_t, float, Particle*, Kernel&, Neighbour&);
//adiabatic exponent (-||-)
__device__ float gamma(size_t, float, Particle*, Kernel&);
//adaptive smooth radius  (-||-)
__device__ void adapt_h(size_t, float, Particle*, Kernel&);

__device__ void refresh(Particle*, Kernel&, Neighbour&);

//rotor and divergence of velocity's field (-||-)
__device__ float  divVel(size_t, Particle*, Kernel&);
__device__ float3	rotVel(size_t, Particle*, Kernel&);


__device__ float scalar_prod(float*, float*);