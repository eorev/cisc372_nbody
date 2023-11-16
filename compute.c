#include <cuda_runtime.h>
#include "vector.h"
#include "config.h"
#include <math.h>

#define NUMELEMENTS 1024

__global__ void computeAccelerationMatrix(vector3* accels, vector3* hPos, double* mass) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k;

    if (i < NUMELEMENTS && j < NUMELEMENTS) {
        if (i != j) {
            vector3 distance;
            // Compute distance vector
            for (k = 0; k < 3; k++) 
                distance[k] = hPos[j][k] - hPos[i][k];

            double magnitude_sq = distance[0] * distance[0] + distance[1] * distance[1] + distance[2] * distance[2];
            double magnitude = sqrt(magnitude_sq);
            double accelmag = GRAV_CONSTANT * mass[j] / magnitude_sq;

            // Compute acceleration vector
            for (k = 0; k < 3; k++) 
                accels[i * NUMELEMENTS + j][k] = accelmag * distance[k] / magnitude;
        } else {
            for (k = 0; k < 3; k++) 
                accels[i * NUMELEMENTS + j][k] = 0;
        }
    }
}

__global__ void updateVelocityPosition(vector3* accels, vector3* hPos, vector3* hVel) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < NUMELEMENTS) {
        vector3 totalAccel = {0, 0, 0};
        // Sum accelerations for object i
        for (int j = 0; j < NUMELEMENTS; j++) {
			int k;
            for (k = 0; k < 3; k++) 
                totalAccel[k] += accels[i * NUMELEMENTS + j][k];
        }

        // Update velocity and position
		int k;
        for (k = 0; k < 3; k++) {
            hVel[i][k] += totalAccel[k] * INTERVAL;
            hPos[i][k] += hVel[i][k] * INTERVAL;
        }
    }
}

void compute(vector3* hPos, vector3* hVel, double* mass) {
    vector3 *d_hPos, *d_hVel, *d_accels;
    double *d_mass;

    // Allocating memory on GPU
    cudaMalloc((void **)&d_hPos, sizeof(vector3) * NUMELEMENTS);
    cudaMalloc((void **)&d_hVel, sizeof(vector3) * NUMELEMENTS);
    cudaMalloc((void **)&d_mass, sizeof(double) * NUMELEMENTS);
    cudaMalloc((void **)&d_accels, sizeof(vector3) * NUMELEMENTS * NUMELEMENTS);

    // Copying data from host to device
    cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMELEMENTS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMELEMENTS, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, sizeof(double) * NUMELEMENTS, cudaMemcpyHostToDevice);

    // Configuring kernel execution parameters
    dim3 dimBlock(16, 16);
    dim3 dimGrid((NUMELEMENTS + dimBlock.x - 1) / dimBlock.x, (NUMELEMENTS + dimBlock.y - 1) / dimBlock.y);

    // Launching kernels
    computeAccelerationMatrix<<<dimGrid, dimBlock>>>(d_accels, d_hPos, d_mass);
    updateVelocityPosition<<<(NUMELEMENTS + 255) / 256, 256>>>(d_accels, d_hPos, d_hVel);

    // Copying updated data back to host
    cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMELEMENTS, cudaMemcpyDeviceToHost);
    cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMELEMENTS, cudaMemcpyDeviceToHost);

    // Freeing allocated memory on GPU
    cudaFree(d_hPos);
    cudaFree(d_hVel);
    cudaFree(d_mass);
    cudaFree(d_accels);
}
