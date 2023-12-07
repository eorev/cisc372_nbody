#include "config.h"
#include "vector.h"
#include <math.h>
#include <stdlib.h>

extern vector3 *hPos;
extern vector3 *hVel;
extern double *mass;

// Kernel function to compute acceleration based on gravity
__global__ void computeAccelerationKernel(vector3 *accelerationVectors,
                                          vector3 *d_hPos, vector3 *d_hVel,
                                          double *d_mass) {
  int entityId =
      blockIdx.x * blockDim.x + threadIdx.x; // Unique ID for each entity
  int otherEntity, dimension;

  if (entityId < NUMENTITIES) {
    // Calculate acceleration for each entity
    for (otherEntity = 0; otherEntity < NUMENTITIES; otherEntity++) {
      if (entityId == otherEntity) {
        // No self-interaction, set acceleration to zero
        FILL_VECTOR(accelerationVectors[entityId * NUMENTITIES + otherEntity],
                    0, 0, 0);
      } else {
        vector3 distance;
        // Calculate distance vector between two entities
        for (dimension = 0; dimension < 3; dimension++)
          distance[dimension] =
              d_hPos[entityId][dimension] - d_hPos[otherEntity][dimension];

        double magnitude_sq = distance[0] * distance[0] +
                              distance[1] * distance[1] +
                              distance[2] * distance[2];
        double magnitude = sqrt(magnitude_sq);
        double accelerationMagnitude =
            -1 * GRAV_CONSTANT * d_mass[otherEntity] / magnitude_sq;
        // Compute gravitational acceleration
        FILL_VECTOR(accelerationVectors[entityId * NUMENTITIES + otherEntity],
                    accelerationMagnitude * distance[0] / magnitude,
                    accelerationMagnitude * distance[1] / magnitude,
                    accelerationMagnitude * distance[2] / magnitude);
      }
    }

    vector3 totalAcceleration = {0, 0, 0};
    // Sum all acceleration contributions
    for (otherEntity = 0; otherEntity < NUMENTITIES; otherEntity++) {
      for (dimension = 0; dimension < 3; dimension++)
        totalAcceleration[dimension] +=
            accelerationVectors[entityId * NUMENTITIES + otherEntity]
                               [dimension];
    }

    // Update velocity and position based on the calculated acceleration
    for (dimension = 0; dimension < 3; dimension++) {
      d_hVel[entityId][dimension] += totalAcceleration[dimension] * INTERVAL;
      d_hPos[entityId][dimension] = d_hVel[entityId][dimension] * INTERVAL;
    }
  }
}

// Main function to setup and execute the kernel
void compute() {
  // Allocate host memory for acceleration vectors
  vector3 *accelerationMatrix =
      (vector3 *)malloc(sizeof(vector3) * NUMENTITIES * NUMENTITIES);
  vector3 **accelerationVectors =
      (vector3 **)malloc(sizeof(vector3 *) * NUMENTITIES);
  for (int i = 0; i < NUMENTITIES; i++)
    accelerationVectors[i] = &accelerationMatrix[i * NUMENTITIES];

  // Allocate device memory
  vector3 *d_hPos;
  cudaMalloc((void **)&d_hPos, sizeof(vector3) * NUMENTITIES);
  vector3 *d_hVel;
  cudaMalloc((void **)&d_hVel, sizeof(vector3) * NUMENTITIES);
  double *d_mass;
  cudaMalloc((void **)&d_mass, sizeof(double) * NUMENTITIES);

  // Copy data from host to device
  cudaMemcpy(d_hPos, hPos, sizeof(vector3) * NUMENTITIES,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_hVel, hVel, sizeof(vector3) * NUMENTITIES,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES,
             cudaMemcpyHostToDevice);

  // Setup grid and block dimensions for the kernel
  int blockSize = 256;
  int gridSize = (NUMENTITIES + blockSize - 1) / blockSize;

  // Execute the kernel
  computeAccelerationKernel<<<gridSize, blockSize>>>(accelerationVectors,
                                                     d_hPos, d_hVel, d_mass);

  // Copy results back to host
  cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES,
             cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_hPos);
  cudaFree(d_hVel);
  cudaFree(d_mass);

  // Free host memory
  free(accelerationVectors);
  free(accelerationMatrix);
}
