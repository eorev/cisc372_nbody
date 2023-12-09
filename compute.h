#ifndef COMPUTE_H
#define COMPUTE_H

#include "vector.h" // Ensure this includes the definition of vector3

// Declare any global variables used in compute.cu
extern vector3 *d_values, **d_accels, *d_hPos, *d_hVel, *d_accel_sum;
extern double *d_mass;

// Function declarations

// compute: Updates the positions and velocities of objects based on
// gravitational forces.
void compute();

// CUDA kernel functions

// calcAccels: Calculates the accelerations based on positions and masses.
__global__ void calcAccels(vector3 **accels, vector3 *hPos, double *mass);

// sumColumns: Sums the acceleration components.
__global__ void sumCols(vector3 **accels, vector3 *accel_sum);

// updatePosAndVel: Updates the positions and velocities of the
// entities.
__global__ void updatePosAndVel(vector3 *accel_sum, vector3 *hPos,
                                vector3 *hVel);

#endif // COMPUTE_H