#ifndef CUDA_VECTOR_MATH_CU
#define CUDA_VECTOR_MATH_CU

#include "cuda_vector_math.cuh"
#include "info.cuh"


void vec3::set(const real_t& x, const real_t& y, const real_t& z) {
	r[0] = x;
	r[1] = y;
	r[2] = z;
};



vec3::vec3() { 

	set(0, 0, 0); 
};

vec3::vec3(real_t X, real_t Y, real_t Z) { 

	set(X, Y, Z); 
};

vec3::vec3(const vec3& v) {

	set(v[0], v[1], v[2]);
};

vec3::vec3(const real_t* other) 
{ 

	set(other[0], other[1], other[2]);
};

vec3 vec3::operator+(const vec3& other) const {
	vec3 res(*this);
	res += other;
	return res;
}

vec3& vec3::operator+=(const vec3& other) {
	r[0] += other[0];
	r[1] += other[1];
	r[2] += other[2];
	return *this;
}

vec3 vec3::operator-(const vec3& other) const {
	vec3 res(*this);
	res -= other;
	return res;
};

vec3& vec3::operator-=(const vec3& other) {
	r[0] -= other[0];
	r[1] -= other[1];
	r[2] -= other[2];
	return *this;
};

vec3& vec3::operator*=(real_t c) {
	r[0] *= c;
	r[1] *= c;
	r[2] *= c;
	return *this;
};

vec3 vec3::operator*(real_t c) const {
	vec3 res(*this);
	res *= c;
	return res;
};

vec3 vec3::operator/=(real_t c) {
	r[0] /= c;
	r[1] /= c;
	r[2] /= c;
	return *this;
};

vec3 vec3::operator/(real_t c) const {
	vec3 res(*this);
	res /= c;
	return res;
};

real_t vec3::operator*(const vec3& other) const {
	return r[0] * other[0] + r[1] * other[1] + r[2] * other[2];
};

vec3 cross(const vec3& v1, const vec3& v2) {
	return vec3( v1[1] * v2[2] - v1[2] * v2[1],
				 v1[2] * v2[0] - v1[0] * v2[2],
				 v1[0] * v2[1] - v1[1] * v2[0]);
};

vec3& vec3::operator=(const vec3& other) {
	set(other[0], other[1], other[2]);
	return *this;
};

vec3& vec3::operator=(const real_t* other) {
	vec3 res(other);
	return res;
};

const real_t& vec3::operator[](int i) const {
	return r[i];
};

real_t& vec3::operator[](int i) {
	return r[i];
};

real_t vec3::abs() const {
	return sqrtf(this->abssquared());
};

real_t vec3::abssquared() const {
	return r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
};

real_t vec3::projection(const vec3& other)
{
	return this->abs() * vcos(*this, other) ;
};

void vec3::normalize() {
	*this *= rsqrtf(abssquared);
};


real_t abs(const vec3& data) {
	return data.abs();
};

real_t vcos(const vec3& v1, const vec3& v2) {
	return v1 * v2 / (v1.abs() * v2.abs());
}


#endif