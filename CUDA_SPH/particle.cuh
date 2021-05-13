#pragma once
//#include "model.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



class Particle
{
public:

	float* pos; //Position radius-vector

	float* vel; //Velosity vector
	float* ax; //Axeleration vector

	float* mass; //Mass and radius
	//float energy;

	float* density;
	float* pressure;

	float* h; //Smooth radius
	float* gamma;
	float* A; //Const dep on env

	Particle();
	__host__ void malloc();




	//=========SET FUNCTIONS=========

	//Will set mass
	__host__ void set_mass();

	//Will set (x, y, z) coordinates
	__host__ void set_pos(float, float, float);
	//Will set q_i coordinate (q, i)
	__host__ void set_pos_i(float, int);

	//Will set (v_x, v_y, v_z) velocities
	__host__ void set_vel(float, float, float);
	//Will set v_i coordinate (v, i)
	__host__ void set_vel_i(float, int);



	//=========GET FUNCTIONS=========

	//Will fetch mass
	__host__ void get_mass();

	//Will fetch q_i coordinate (q, i)
	__host__ float get_pos_i(int);

	//Will fetch v_i coordinate (v, i)
	__host__ float get_vel_i(int);

};

