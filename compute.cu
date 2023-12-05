#include "config.h"
#include "vector.h"
#include <cuda_runtime.h>
#include <math.h>

#define NUMELEMENTS 1024
#define BLOCK_SIZE 16

// Function to compute dot product
__device__ double dot_product(vector3 a, vector3 b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Function to multiply vector by a scalar and store result in a third vector
__device__ void scalar_mult(vector3 v, double scalar, vector3 result) {
  result[0] = v[0] * scalar;
  result[1] = v[1] * scalar;
  result[2] = v[2] * scalar;
}

// Function to add two vectors and store result in a third vector
__device__ void vector_add(vector3 a, vector3 b, vector3 result) {
  result[0] = a[0] + b[0];
  result[1] = a[1] + b[1];
  result[2] = a[2] + b[2];
}

__global__ void computeAccelerationMatrix(vector3 *d_hPos, double *d_mass,
                                          vector3 *d_accels) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < NUMELEMENTS && j < NUMELEMENTS) {
    vector3 distance;
    vector3 accelResult;
    if (i == j) {
      FILL_VECTOR(d_accels[i * NUMELEMENTS + j], 0, 0, 0);
    } else {
      // Calculate distance
      for (int k = 0; k < 3; k++) {
        distance[k] = d_hPos[i][k] - d_hPos[j][k];
      }
      double magnitude_sq = dot_product(distance, distance);
      double magnitude = sqrt(magnitude_sq);
      double accelmag = -GRAV_CONSTANT * d_mass[j] / magnitude_sq;

      scalar_mult(distance, accelmag / magnitude, accelResult);
      for (int k = 0; k < 3; k++) {
        d_accels[i * NUMELEMENTS + j][k] = accelResult[k];
      }
    }
  }
}

__global__ void updateVelocityPosition(vector3 *d_hVel, vector3 *d_hPos,
                                       vector3 *d_accels) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < NUMELEMENTS) {
    vector3 totalAccel = {0, 0, 0};
    vector3 tempAccel;
    vector3 tempVel;
    for (int j = 0; j < NUMELEMENTS; j++) {
      vector_add(totalAccel, d_accels[i * NUMELEMENTS + j], tempAccel);
      for (int k = 0; k < 3; k++) {
        totalAccel[k] = tempAccel[k];
      }
    }

    scalar_mult(totalAccel, INTERVAL, tempAccel);
    vector_add(d_hVel[i], tempAccel, tempVel);
    for (int k = 0; k < 3; k++) {
      d_hVel[i][k] = tempVel[k];
    }
    scalar_mult(d_hVel[i], INTERVAL, tempVel);
    vector_add(d_hPos[i], tempVel, tempAccel);
    for (int k = 0; k < 3; k++) {
      d_hPos[i][k] = tempAccel[k];
    }
  }
}

void compute(vector3 *d_hPos, vector3 *d_hVel, double *d_mass) {
  vector3 *d_accels;
  cudaMalloc((void **)&d_accels, sizeof(vector3) * NUMELEMENTS * NUMELEMENTS);

  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 dimGrid((NUMELEMENTS + dimBlock.x - 1) / dimBlock.x,
               (NUMELEMENTS + dimBlock.y - 1) / dimBlock.y);

  computeAccelerationMatrix<<<dimGrid, dimBlock>>>(d_hPos, d_mass, d_accels);
  cudaDeviceSynchronize();

  updateVelocityPosition<<<(NUMELEMENTS + 255) / 256, 256>>>(d_hVel, d_hPos,
                                                             d_accels);

  cudaDeviceSynchronize();

  cudaFree(d_accels);
}
