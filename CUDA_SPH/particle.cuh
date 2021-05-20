#pragma once
//#include "model.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_vector_math.cuh"

class Particle
{
private:
	Particle* d_this;
public:
	
	vec3 pos; //Position radius-vector

	vec3 vel; //Velosity vector
	vec3 ax; //Acceleration vector

	float mass; //Mass and radius
	//float energy;

	float k;
	float rho0;

	float density;
	float pressure;

	float h; //Smooth radius
	float gamma;
	float A; //Const dep on env
	float nu; //viscosity

	Particle();
	void allocate();


	//=========SET FUNCTIONS=========

	//Will set mass
	void set_mass(float);

	//Will set (x, y, z) coordinates
	void set_pos(float, float, float);
	//Will set q_i coordinate (q, i)
	void set_pos_i(float, int);

	//Will set (v_x, v_y, v_z) velocities
	void set_vel(float, float, float);
	//Will set v_i coordinate (v, i)
	void set_vel_i(float, int);

	void set_ax(float, float, float);


	//=========GET FUNCTIONS=========

	//Will fetch mass
	float get_mass();
	float get_dens();

	//Will fetch q_i coordinate (q, i)
	float get_pos_i(int);

	//Will fetch v_i coordinate (v, i)
	float get_vel_i(int);

	Particle* device();

};

void update_particles(Particle* ps, Particle* d_ps, size_t ps_num);

//Must be called after creatin an array of particles to send them to GPU
Particle* device_particles_array(Particle*, size_t);