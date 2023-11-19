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
    if (threadIdx.y == 0 && j < NUMELEMENTS) {
        sharedMass[threadIdx.x] = (threadIdx.x < BLOCK_SIZE) ? d_mass[j] : 0;
    }

    if (i < NUMELEMENTS && j < NUMELEMENTS) {
        vector3 pos = d_hPos[j];
        for (int k = 0; k < 3; k++) {
            sharedPos[threadIdx.y][threadIdx.x][k] = pos[k];
        }
    }
    __syncthreads();

    if (i < NUMELEMENTS && j < NUMELEMENTS && i != j) {
        vector3 distance;
        double magnitude_sq = 0.0;
        for (int k = 0; k < 3; k++) {
            distance[k] = sharedPos[threadIdx.y][threadIdx.x][k] - d_hPos[i][k];
            magnitude_sq += distance[k] * distance[k];
        }

        double magnitude = sqrt(magnitude_sq);
        double accelmag = GRAV_CONSTANT * sharedMass[threadIdx.x] / magnitude_sq;

        for (int k = 0; k < 3; k++) {
            double accelComponent = accelmag * distance[k] / magnitude;
            atomicAdd(&accels[i * NUMELEMENTS + j][k], accelComponent);
        }
    }
}

__global__ void updateVelocityPosition(vector3 *accels, vector3 *d_hPos, vector3 *d_hVel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NUMELEMENTS) {
        vector3 totalAccel = {0, 0, 0};
        for (int j = 0; j < NUMELEMENTS; j++) {
            for (int k = 0; k < 3; k++) {
                totalAccel[k] += accels[i * NUMELEMENTS + j][k];
            }
        }

        for (int k = 0; k < 3; k++) {
            d_hVel[i][k] += totalAccel[k] * INTERVAL;
            d_hPos[i][k] += d_hVel[i][k] * INTERVAL;
        }
    }
}

void compute(vector3 *d_hPos, vector3 *d_hVel, double *d_mass) {
    vector3 *d_accels;
    cudaMalloc((void **)&d_accels, sizeof(vector3) * NUMELEMENTS * NUMELEMENTS);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((NUMELEMENTS + dimBlock.x - 1) / dimBlock.x,
                 (NUMELEMENTS + dimBlock.y - 1) / dimBlock.y);

    computeAccelerationMatrix<<<dimGrid, dimBlock>>>(d_accels, d_hPos, d_mass);
    updateVelocityPosition<<<(NUMELEMENTS + 255) / 256, 256>>>(d_accels, d_hPos, d_hVel);

    cudaFree(d_accels);
}
