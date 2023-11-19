#include "config.h"
#include "vector.h"
#include <cuda_runtime.h>
#include <math.h>

#define NUMELEMENTS 1024
#define BLOCK_SIZE 16

__global__ void computeAccelerationMatrix(vector3 *accels, vector3 *d_hPos,
                                          double *d_mass) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  __shared__ vector3 sharedPos[BLOCK_SIZE][BLOCK_SIZE];
  __shared__ double sharedMass[BLOCK_SIZE];

  // Load a block of d_hPos and d_mass into shared memory
  if (threadIdx.y == 0 && threadIdx.x < BLOCK_SIZE && j < NUMELEMENTS) {
    sharedMass[threadIdx.x] = d_mass[j];
  }

  if (i < NUMELEMENTS && j < NUMELEMENTS && threadIdx.x < BLOCK_SIZE &&
      threadIdx.y < BLOCK_SIZE) {
    for (int k = 0; k < 3; k++) {
      sharedPos[threadIdx.y][threadIdx.x][k] = d_hPos[j][k];
    }
  }
  __syncthreads();

  if (i < NUMELEMENTS && j < NUMELEMENTS && i != j) {
    vector3 distance;
    if (i < j) { // Calculate only for i < j
      for (int k = 0; k < 3; k++) {
        distance[k] = sharedPos[threadIdx.y][threadIdx.x][k] - d_hPos[i][k];
      }

      double magnitude_sq = distance[0] * distance[0] +
                            distance[1] * distance[1] +
                            distance[2] * distance[2];
      double magnitude = sqrt(magnitude_sq);
      double accelmag = GRAV_CONSTANT * sharedMass[threadIdx.x] / magnitude_sq;

      // Compute acceleration vector
      for (int k = 0; k < 3; k++) {
        double accelComponent = accelmag * distance[k] / magnitude;
        accels[i * NUMELEMENTS + j][k] = accelComponent;
        accels[j * NUMELEMENTS + i][k] = -accelComponent; // Using symmetry instead of recomputation
      }
    }
  } else if (i < NUMELEMENTS) {
    for (int k = 0; k < 3; k++) {
      accels[i * NUMELEMENTS + j][k] = 0;
    }
  }
}

__global__ void updateVelocityPosition(vector3 *accels, vector3 *d_hPos,
                                       vector3 *d_hVel) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < NUMELEMENTS) {
    vector3 totalAccel = {0, 0, 0};
    int j;
    // Sum accelerations for object i
    for (j = 0; j < NUMELEMENTS; j++) {
      int k;
      #pragma unroll // trying out using a pragma function to increase speed
      for (k = 0; k < 3; k++)
        totalAccel[k] += accels[i * NUMELEMENTS + j][k];
    }

    // Update velocity and position
    int k;
    for (k = 0; k < 3; k++) {
      d_hVel[i][k] += totalAccel[k] * INTERVAL;
      d_hPos[i][k] += d_hVel[i][k] * INTERVAL;
    }
  }
}

void compute(vector3 *d_hPos, vector3 *d_hVel, double *d_mass) {
  vector3 *d_accels;
  cudaMalloc((void **)&d_accels, sizeof(vector3) * NUMELEMENTS * NUMELEMENTS);

  dim3 dimBlock(16, 16);
  dim3 dimGrid((NUMELEMENTS + dimBlock.x - 1) / dimBlock.x,
               (NUMELEMENTS + dimBlock.y - 1) / dimBlock.y);

  computeAccelerationMatrix<<<dimGrid, dimBlock>>>(d_accels, d_hPos, d_mass);
  updateVelocityPosition<<<(NUMELEMENTS + 255) / 256, 256>>>(d_accels, d_hPos,
                                                             d_hVel);

  cudaFree(d_accels);
}