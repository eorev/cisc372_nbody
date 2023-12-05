#include "config.h"
#include "vector.h"
#include <cuda_runtime.h>
#include <math.h>

#define NUMELEMENTS NUMENTITIES
#define BLOCK_SIZE 16

// Function to compute dot product
__device__ double dot_product(vector3 a, vector3 b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

// Function to multiply vector by a scalar and add to another vector
__device__ void vector_add_scaled(vector3 a, vector3 b, double scalar,
                                  vector3 result) {
  result[0] = a[0] + b[0] * scalar;
  result[1] = a[1] + b[1] * scalar;
  result[2] = a[2] + b[2] * scalar;
}

__global__ void computeAccelerationMatrix(vector3 *d_hPos, double *d_mass,
                                          vector3 *d_accels) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < NUMELEMENTS && j < NUMELEMENTS) {
    vector3 distance;
    if (i != j) {
      for (int k = 0; k < 3; k++) {
        distance[k] = d_hPos[j][k] - d_hPos[i][k];
      }
      double magnitude_sq = dot_product(distance, distance);
      double magnitude = sqrt(magnitude_sq);
      double accelmag = -GRAV_CONSTANT * d_mass[j] / magnitude_sq;
      for (int k = 0; k < 3; k++) {
        d_accels[i * NUMELEMENTS + j][k] = distance[k] * accelmag / magnitude;
      }
    } else {
      d_accels[i * NUMELEMENTS + j][0] = 0.0;
      d_accels[i * NUMELEMENTS + j][1] = 0.0;
      d_accels[i * NUMELEMENTS + j][2] = 0.0;
    }
  }
}

__global__ void updateVelocityPosition(vector3 *d_hVel, vector3 *d_hPos,
                                       vector3 *d_accels) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < NUMELEMENTS) {
    vector3 totalAccel = {0, 0, 0};
    for (int j = 0; j < NUMELEMENTS; j++) {
      vector_add_scaled(totalAccel, d_accels[i * NUMELEMENTS + j], 1.0,
                        totalAccel);
    }

    vector3 newVel;
    vector_add_scaled(d_hVel[i], totalAccel, INTERVAL, newVel);
    vector3 newPos;
    vector_add_scaled(d_hPos[i], newVel, INTERVAL, newPos);

    d_hVel[i][0] = newVel[0];
    d_hVel[i][1] = newVel[1];
    d_hVel[i][2] = newVel[2];
    d_hPos[i][0] = newPos[0];
    d_hPos[i][1] = newPos[1];
    d_hPos[i][2] = newPos[2];
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
