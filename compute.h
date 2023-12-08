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

// mapValuestoAccels: Maps values to acceleration vectors (if used in
// compute.cu).
__global__ void mapValuestoAccels(vector3 *values, vector3 **accels);

// fillAccelSum: Fills the acceleration sum array (if used in compute.cu).
__global__ void fillAccelSum(vector3 *accel_sum);

// calculateAccels: Calculates the accelerations based on positions and masses.
__global__ void calculateAccels(vector3 **accels, vector3 *hPos, double *mass);

// sumColumns: Sums the acceleration components.
__global__ void sumColumns(vector3 **accels, vector3 *accel_sum);

// updatePositionAndVelocity: Updates the positions and velocities of the
// entities.
__global__ void updatePositionAndVelocity(vector3 *accel_sum, vector3 *hPos,
                                          vector3 *hVel);

#endif // COMPUTE_H