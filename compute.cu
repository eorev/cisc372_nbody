#include "config.h"
#include "vector.h"
#include <cuda_runtime.h>
#include <math.h>

#define NUMELEMENTS NUMENTITIES
#define BLOCK_SIZE 16

__global__ void computeAccelerationMatrix(vector3 *d_hPos, double *d_mass,
                                          vector3 *d_accels) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < NUMELEMENTS && j < NUMELEMENTS) {
    if (i == j) {
      d_accels[i * NUMELEMENTS + j] = make_vector3(0, 0, 0);
    } else {
      vector3 distance =
          make_vector3(d_hPos[i].x - d_hPos[j].x, d_hPos[i].y - d_hPos[j].y,
                       d_hPos[i].z - d_hPos[j].z);
      double magnitude_sq = dot_product(distance, distance);
      double magnitude = sqrt(magnitude_sq);
      double accelmag = -GRAV_CONSTANT * d_mass[j] / magnitude_sq;
      d_accels[i * NUMELEMENTS + j] =
          scalar_mult(distance, accelmag / magnitude);
    }
  }
}

__global__ void updateVelocityPosition(vector3 *d_hVel, vector3 *d_hPos,
                                       vector3 *d_accels) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < NUMELEMENTS) {
    vector3 totalAccel = make_vector3(0, 0, 0);
    for (int j = 0; j < NUMELEMENTS; j++) {
      totalAccel = vector_add(totalAccel, d_accels[i * NUMELEMENTS + j]);
    }
    d_hVel[i] = vector_add(d_hVel[i], scalar_mult(totalAccel, INTERVAL));
    d_hPos[i] = vector_add(d_hPos[i], scalar_mult(d_hVel[i], INTERVAL));
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
