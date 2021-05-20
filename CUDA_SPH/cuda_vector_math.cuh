#ifndef CUDA_VECTOR_MATH_H
#define CUDA_VECTOR_MATH_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "info.cuh"
#include <iostream>
#include <vector>
#include <cmath>


class vec3
{
public:
	real_t r[3];
	
	__host__ __device__
	vec3();
	__host__ __device__
	vec3(real_t, real_t, real_t);
	__host__ __device__
  vec3(const vec3& v);
	__host__ __device__
	vec3(const real_t*);


	__host__ __device__
	vec3 operator+(const vec3&) const;
	__host__ __device__
	vec3& operator+=(const vec3&);
	__host__ __device__
	vec3 operator-(const vec3&) const;
	__host__ __device__
	vec3& operator-=(const vec3&);
	__host__ __device__
	vec3 operator*(real_t) const;
	__host__ __device__
	vec3& operator*=(real_t);
	__host__ __device__
	vec3 operator/=(real_t);
	__host__ __device__
	vec3 operator/(real_t) const;
	__host__ __device__
	real_t operator*(const vec3&) const;
	__host__ __device__
	friend vec3 cross(const vec3&, const vec3&); //vector multiplication
	__host__ __device__
	vec3& operator=(const vec3&);
	__host__ __device__
	vec3& operator=(const real_t*); //!!!
	__host__ __device__
	const real_t& operator[](int) const;
	__host__ __device__
	real_t& operator[](int);
	
	__host__ __device__
	void set(const real_t&, const real_t&, const real_t&);

	__host__ __device__
	real_t abs() const;
	__host__ __device__
	real_t abssquared() const;
	__host__ __device__
	void normalize();
	
	__host__ __device__
	real_t projection(const vec3&); //Not checked!!!
	__host__ __device__
		vec3 normalize();

};

	__host__ __device__
real_t vcos(const vec3&, const vec3&);
	__host__ __device__
real_t abs(const vec3&);

#endif