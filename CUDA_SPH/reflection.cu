// Copyright - Evgeniy Dedkov, MIPT, 2021

// WARN! All vectors and points MUST be arrays of 3 coords
#pragma ones

// Includes
#include "info.cuh"
#include "mesh/reflection.cuh"

__device__
void vector(point_t p, vector_t result) {
  for(int i = 0; i < 3; ++i) {
    result[i] = p[i];
  };
};

// Writes vector v to result vector -- simple copy
// Need to be defined if vector_t != point_t
// void vector(point_t v, vector_t result) {
//   for(int i = 0; i < 0; ++i) {
//     result[i] = v[i];
//   };
// };

// Writes vector p1->p2 to result vector
__device__
void vector(point_t p1, point_t p2, vector_t result) {
  for(int i = 0; i < 3; ++i) {
    result[i] = p2[i] - p1[i];
  };
};

// Writes vector p1 to result point
__device__
void point(vector_t p1, point_t result) {
  for(int i = 0; i < 3; ++i) {
    result[i] = p1[i];
  };
};


// Some vector math
namespace v_math{
  // v1 + v2 -> v1
  __device__
  void add(vector_t v1, vector_t v2) {
    for(int i = 0; i < 3; ++i) {
      v1[i] += v2[i];
    };
  };

  // v1 + v2 -> result
  __device__
  void add(vector_t v1, vector_t v2, vector_t result) {
    for(int i = 0; i < 3; ++i) {
      result[i] = v1[i] + v2[i];
    };
  };

  // v1 - v2 -> v1
  __device__
  void subtract(vector_t v1, vector_t v2) {
    for(int i = 0; i < 3; ++i) {
      v1[i] -= v2[i];
    };
  };

  // v1 - v2 -> result
  __device__
  void subtract(vector_t v1, vector_t v2, vector_t result) {
    for(int i = 0; i < 3; ++i) {
      result[i] = v1[i] - v2[i];
    };
  };

  // a*v -> v
  __device__
  void mult(vector_t v, real_t a) {
    for(int i = 0; i < 3; ++i) {
      v[i] *= a;
    };
  };

  // a*v -> result
  __device__
  void mult(vector_t v, real_t a, vector_t result) {
    for(int i = 0; i < 3; ++i) {
      result[i] = v[i] * a;
    };
  };

  // Evaluates dot product
  __device__
  real_t dot(vector_t v1, vector_t v2) {
    real_t result = 0;
    for(int i = 0; i < 3; ++i) {
      result += v1[i]*v2[i];
    };
    return result;
  };

  // Cross product of v1, v2 -> result
  __device__
  void cross(vector_t v1, vector_t v2, vector_t result) {
    for(int i = 0; i < 3; ++i) {
      int c1 = (i+1)%3;
      int c2 = (i+2)%3;
      result[i] = v1[c1]*v2[c2] - v1[c2]*v2[c1];
    };
  };

  // Normalize vector in order v^2==1
  __device__
  void normalize(vector_t v) {
    real_t norm = rsqrtf(dot(v, v));
    for(int i = 0; i < 3; ++i) {
      v[i] = v[i] * norm;
    };
  };
};






// Perform the time step dt for free particle at point x with velocity v
// takes into account possible reflection from triangul surface
// described by the array of 3 points tr
// All changes occurs in corresponding arrays
__device__
void reflect(timestep_t dt, point_t x, vector_t v, point_t* tr) {
    using namespace v_math;
  
    // step 1 - finding intersection of (x, x+dx) ray and tr plane
    // eq of point on (x, x+dx)  r = p1 + mu * dp
    // eq of points on tr  (r,n) = -D
    vector_t p1 = new vector_init_t;
    vector(x, p1);
    vector_t dp = new vector_init_t;
    mult(v, dt, dp);
    vector_t r1 = new vector_init_t;
    vector_t r12 = new vector_init_t;
    vector_t r13 = new vector_init_t;
    vector_t n = new vector_init_t;
    vector(tr[0], r1);
    vector(tr[0], tr[1], r12);
    vector(tr[0], tr[2], r13);
    cross(r12, r13, n);
    normalize(n);
    real_t D = -dot(n, r1);
    real_t mu = -(D + dot(n, p1)) / dot(n, dp);

    if((mu > 0) && (mu < 1)) {
        // Reflection may occur => step 2 - check is p on triangle

        // Obtain p = p1 + mu * dp
        vector_t rp = new vector_init_t;
        mult(dp, mu, rp);
        add(rp, p1);
        point_t p = new point_init_t;
        point(rp, p);

        // Ckeck if sum of p->pi p->p(i+1) angles is 2pi
        vector_t* prs = new vector_t[3];
        for(int i = 0; i < 3; ++i) {
            prs[i] = new vector_init_t;
            vector(p, tr[i], prs[i]);
            normalize(prs[i]);
    };
    real_t angle_sum = 0;
    for(int i = 0; i < 3; ++i) {
        angle_sum += acosf(dot(prs[i], prs[(i+1)%3]));
    };
    if(fabsf(angle_sum - 2 * M_PI) < EPS) {
        // Reflection definetely takes place
        vector_t dx2 = new vector_init_t; // wrong dx movement after collision
        add(p1, dp, dx2);
        subtract(dx2, rp);
        // lets fix: dx2_true = dx2 - 2n (n,dx2)
        // and x_final = rp + dx2_true
        vector_t n_tmp = new vector_init_t;
        vector(n, n_tmp);
        mult(n_tmp, 2*dot(n, dx2));
        subtract(dx2, n_tmp);
        vector_t x_true = new vector_init_t;
        add(rp, dx2, x_true);
        point(x_true, x);
        // velocity: v_res = v - 2n(n,v)
        vector(n, n_tmp);
        mult(n_tmp, 2*dot(v, n));
        subtract(v, n_tmp);
        mult(v, 0.9);

        delete[] dx2;
        delete[] n_tmp;
        delete[] x_true;
    } else {
        // Still no reflections => no troubles 
        add(p1, dp);
        point(p1, x);
    };

    for(int i = 0; i < 3; ++i) {
        delete[] prs[i];
    };
    delete[] prs;
    delete[] rp;
    delete[] p;
    } else {
    // No reflection might be => enjoy, we have nothing to do
    add(p1, dp);
    point(p1, x);
    };

  delete[] p1;
  delete[] dp;
  delete[] n;
  delete[] r1;
  delete[] r12;
  delete[] r13;
};