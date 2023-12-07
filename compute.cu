#include "compute.h"
#include "config.h"
#include "vector.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// External declarations for global variables used in GPU computations
extern vector3 *d_values, **d_accels, *d_hPos, *d_hVel, *d_accel_sum;
extern double *d_mass;

/**
 * compute - Main function to update positions and velocities of objects in the
 * system. This function sets up the execution configuration and launches CUDA
 * kernels.
 */
void compute() {
  // Calculate grid dimensions for CUDA kernel execution
  int gridDimension = (NUMENTITIES + 7) / 8;
  dim3 gridThreads(8, 8, 3);
  dim3 gridBlocks(gridDimension, gridDimension);

  // Launch calculateAccels kernel to compute accelerations based on
  // gravitational forces
  calculateAccels<<<gridBlocks, gridThreads>>>(d_accels, d_hPos, d_mass);

  // Set up dimensions for sumColumns kernel
  int threadCount = 64;
  dim3 blockDimension(NUMENTITIES, 3);
  int sharedMemSize = 2 * threadCount * sizeof(double);

  // Launch sumColumns kernel to sum up acceleration components
  sumColumns<<<blockDimension, threadCount, sharedMemSize>>>(d_accels,
                                                             d_accel_sum);

  // Set up dimensions for updatePositionAndVelocity kernel
  int positionUpdateBlocks = (NUMENTITIES + 7) / 8;
  dim3 velocityUpdateThreads(8, 3);

  // Launch updatePositionAndVelocity kernel to update positions and velocities
  updatePositionAndVelocity<<<positionUpdateBlocks, velocityUpdateThreads>>>(
      d_accel_sum, d_hPos, d_hVel);
}

/**
 * CUDA Kernel Function - calculateAccels
 * Calculates the acceleration of each object due to gravitational forces from
 * all other objects.
 *
 * @param accels    2D array to store calculated accelerations.
 * @param positions Array of object positions.
 * @param masses    Array of object masses.
 */
__global__ void calculateAccels(vector3 **accels, vector3 *positions,
                                double *masses) {
  // Calculate unique index for each thread in the kernel
  int entityX = threadIdx.x + blockIdx.x * blockDim.x;
  int entityY = threadIdx.y + blockIdx.y * blockDim.y;
  int dimensionZ = threadIdx.z;

  // Check for out-of-bound indices
  if (entityX >= NUMENTITIES || entityY >= NUMENTITIES)
    return;

  // Shared memory for distance components
  __shared__ vector3 distComponents[8][8];

  // Calculate accelerations
  if (entityX == entityY) {
    accels[entityX][entityY][dimensionZ] = 0;
  } else {
    // Compute distance components between two entities
    distComponents[threadIdx.x][threadIdx.y][dimensionZ] =
        positions[entityX][dimensionZ] - positions[entityY][dimensionZ];
    __syncthreads();

    // Calculate squared distance and acceleration magnitude
    double distSquared = pow(distComponents[threadIdx.x][threadIdx.y][0], 2) +
                         pow(distComponents[threadIdx.x][threadIdx.y][1], 2) +
                         pow(distComponents[threadIdx.x][threadIdx.y][2], 2);
    double dist = sqrt(distSquared);
    double accelMagnitude = -GRAV_CONSTANT * masses[entityY] / distSquared;

    // Update acceleration vector
    accels[entityX][entityY][dimensionZ] =
        accelMagnitude * distComponents[threadIdx.x][threadIdx.y][dimensionZ] /
        dist;
  }
}

/**
 * CUDA Kernel Function - sumColumns
 * Sums up the acceleration components for each object.
 *
 * @param accels       2D array of acceleration vectors.
 * @param accelTotals  Array to store total acceleration for each object.
 */
__global__ void sumColumns(vector3 **accels, vector3 *accelTotals) {
  int rowIdx = threadIdx.x;
  int colIdx = blockIdx.x;
  int dimension = blockIdx.y;

  // Shared memory for partial sums
  __shared__ int offsetIndex;
  int blockSize = blockDim.x;
  int totalEntities = NUMENTITIES;
  extern __shared__ double sharedArray[];

  // Initialize shared array with acceleration values
  sharedArray[rowIdx] =
      (rowIdx < totalEntities) ? accels[colIdx][rowIdx][dimension] : 0;

  // Determine offset for partial sums
  if (rowIdx == 0)
    offsetIndex = blockSize;
  __syncthreads();

  // Sum acceleration components using shared memory
  while (offsetIndex < totalEntities) {
    sharedArray[rowIdx + blockSize] =
        (rowIdx + blockSize < totalEntities)
            ? accels[colIdx][rowIdx + offsetIndex][dimension]
            : 0;
    __syncthreads();

    if (rowIdx == 0)
      offsetIndex += blockSize;

    double sumVal = sharedArray[2 * rowIdx] + sharedArray[2 * rowIdx + 1];
    __syncthreads();
    sharedArray[rowIdx] = sumVal;
  }
  __syncthreads();

  // Final reduction to get the total sum
  for (int stride = 1; stride < blockSize; stride *= 2) {
    int arrayIndex = rowIdx * stride * 2;
    if (arrayIndex + stride < blockSize) {
      sharedArray[arrayIndex] += sharedArray[arrayIndex + stride];
    }
    __syncthreads();
  }

  // Write total sum to accelTotals
  if (rowIdx == 0) {
    accelTotals[colIdx][dimension] = sharedArray[0];
  }
}

/**
 * CUDA Kernel Function - updatePositionAndVelocity
 * Updates the positions and velocities of each object based on calculated
 * accelerations.
 *
 * @param totalAccel  Array of total accelerations for each object.
 * @param positions   Array of object positions.
 * @param velocities  Array of object velocities.
 */
__global__ void updatePositionAndVelocity(vector3 *totalAccel,
                                          vector3 *positions,
                                          vector3 *velocities) {
  // Calculate unique index for each thread
  int entityIdx = threadIdx.x + blockIdx.x * blockDim.x;
  int dimensionIdx = threadIdx.y;

  // Check for out-of-bound indices
  if (entityIdx >= NUMENTITIES)
    return;

  // Update velocities and positions
  velocities[entityIdx][dimensionIdx] +=
      totalAccel[entityIdx][dimensionIdx] * INTERVAL;
  positions[entityIdx][dimensionIdx] +=
      velocities[entityIdx][dimensionIdx] * INTERVAL;
}
