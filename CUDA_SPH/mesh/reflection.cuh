#ifndef REFLECTION_CUH
#define REFLECTION_CUH
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// Includes
#include "../info.cuh"
# define M_PI 3.14159265358979323846 // pi

// Typedefs
// using real_t = float; // can be changed for float
using timestep_t = real_t;
using coord_t = real_t;
using point_t = coord_t*;
using point_init_t = coord_t[3]; // used in expr like point_t v = new point_init_t;
using vector_t = coord_t*;
using vector_init_t = coord_t[3]; // used in expr like vector_t v = new vector_init_t;

// Writes vector to p to result vector
__device__
void vector(point_t p, vector_t result);

// Writes vector v to result vector -- simple copy
// Need to be defined if vector_t != point_t
// void vector(point_t v, vector_t result) {
//   for(int i = 0; i < 0; ++i) {
//     result[i] = v[i];
//   };
// };

// Writes vector p1->p2 to result vector
__device__
void vector(point_t p1, point_t p2, vector_t result);

// Writes vector p1 to result point
__device__
void point(vector_t p1, point_t result);

// Some vector math
namespace v_math{
  // v1 + v2 -> v1
  __device__
  void add(vector_t v1, vector_t v2);

  // v1 + v2 -> result
  __device__
  void add(vector_t v1, vector_t v2, vector_t result);

  // v1 - v2 -> v1
  __device__
  void subtract(vector_t v1, vector_t v2);

  // v1 - v2 -> result
  __device__
  void subtract(vector_t v1, vector_t v2, vector_t result);

  // a*v -> v
  __device__
  void mult(vector_t v, real_t a);

  // a*v -> result
  __device__
  void mult(vector_t v, real_t a, vector_t result);

  // Evaluates dot product
  __device__
  real_t dot(vector_t v1, vector_t v2);

  // Cross product of v1, v2 -> result
  __device__
  void cross(vector_t v1, vector_t v2, vector_t result);

  // Normalize vector in order v^2==1
  __device__
  void normalize(vector_t v);
};

__device__ const real_t EPS = 1e-6; // TODO : need to be adjjusted!!!


// Perform the time step dt for free particle at point x with velocity v
// takes into account possible reflection from triangul surface
// described by the array of 3 points tr
// All changes occurs in corresponding arrays
__device__
void reflect(timestep_t dt, point_t x, vector_t v, point_t* tr);

#endif