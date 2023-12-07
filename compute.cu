#include "compute.h"
#include "config.h"
#include "vector.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

extern vector3 *d_values, **d_accels, *d_hPos, *d_hVel, *d_accel_sum;
extern double *d_mass;

// compute: Updates the positions and velocities of objects in the system based
// on gravitational forces.
void compute() {
  int gridDimension = (NUMENTITIES + 7) / 8;
  dim3 gridThreads(8, 8, 3);
  dim3 gridBlocks(gridDimension, gridDimension);
  calculateAccels<<<gridBlocks, gridThreads>>>(d_accels, d_hPos, d_mass);
  int threadCount = 64;
  dim3 blockDimension(NUMENTITIES, 3);
  int sharedMemSize = 2 * threadCount * sizeof(double);
  sumAccelerationComponents<<<blockDimension, threadCount, sharedMemSize>>>(
      d_accels, d_accel_sum);
  int positionUpdateBlocks = (NUMENTITIES + 7) / 8;
  dim3 velocityUpdateThreads(8, 3);
  updatePosAndVel<<<positionUpdateBlocks, velocityUpdateThreads>>>(
      d_accel_sum, d_hPos, d_hVel);
}

__global__ void calculateAccels(vector3 **accels, vector3 *positions,
                                double *masses) {
  int entityX = threadIdx.x + blockIdx.x * blockDim.x;
  int entityY = threadIdx.y + blockIdx.y * blockDim.y;
  int dimensionZ = threadIdx.z;
  __shared__ vector3 distComponents[8][8];
  if (entityX >= NUMENTITIES || entityY >= NUMENTITIES)
    return;
  if (entityX == entityY) {
    accels[entityX][entityY][dimensionZ] = 0;
  } else {
    distComponents[threadIdx.x][threadIdx.y][dimensionZ] =
        positions[entityX][dimensionZ] - positions[entityY][dimensionZ];
    __syncthreads();
    double distSquared = distComponents[threadIdx.x][threadIdx.y][0] *
                             distComponents[threadIdx.x][threadIdx.y][0] +
                         distComponents[threadIdx.x][threadIdx.y][1] *
                             distComponents[threadIdx.x][threadIdx.y][1] +
                         distComponents[threadIdx.x][threadIdx.y][2] *
                             distComponents[threadIdx.x][threadIdx.y][2];
    double dist = sqrt(distSquared);
    double accelMagnitude = -GRAV_CONSTANT * masses[entityY] / distSquared;
    accels[entityX][entityY][dimensionZ] =
        accelMagnitude * distComponents[threadIdx.x][threadIdx.y][dimensionZ] /
        dist;
  }
}

__global__ void sumAccelerationComponents(vector3 **accels,
                                          vector3 *accelTotals) {
  int rowIdx = threadIdx.x;
  int colIdx = blockIdx.x;
  int dimension = blockIdx.y;
  __shared__ int offsetIndex;
  int blockSize = blockDim.x;
  int totalEntities = NUMENTITIES;
  extern __shared__ double sharedArray[];
  sharedArray[rowIdx] =
      rowIdx < totalEntities ? accels[colIdx][rowIdx][dimension] : 0;
  if (rowIdx == 0) {
    offsetIndex = blockSize;
  }
  __syncthreads();
  while (offsetIndex < totalEntities) {
    sharedArray[rowIdx + blockSize] =
        rowIdx + blockSize < totalEntities
            ? accels[colIdx][rowIdx + offsetIndex][dimension]
            : 0;
    __syncthreads();
    if (rowIdx == 0) {
      offsetIndex += blockSize;
    }
    double sumVal = sharedArray[2 * rowIdx] + sharedArray[2 * rowIdx + 1];
    __syncthreads();
    sharedArray[rowIdx] = sumVal;
  }
  __syncthreads();
  for (int stride = 1; stride < blockSize; stride *= 2) {
    int arrayIndex = rowIdx * stride * 2;
    if (arrayIndex + stride < blockSize) {
      sharedArray[arrayIndex] += sharedArray[arrayIndex + stride];
    }
    __syncthreads();
  }
  if (rowIdx == 0) {
    accelTotals[colIdx][dimension] = sharedArray[0];
  }
}

__global__ void updatePosAndVel(vector3 *totalAccel, vector3 *positions,
                                vector3 *velocities) {
  int entityIdx = threadIdx.x + blockIdx.x * blockDim.x;
  int dimensionIdx = threadIdx.y;
  if (entityIdx >= NUMENTITIES)
    return;

  velocities[entityIdx][dimensionIdx] +=
      totalAccel[entityIdx][dimensionIdx] * INTERVAL;
  positions[entityIdx][dimensionIdx] +=
      velocities[entityIdx][dimensionIdx] * INTERVAL;
}
