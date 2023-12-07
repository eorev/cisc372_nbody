#include "config.h"
#include "vector.h"
#include <math.h>
#include <stdlib.h>

// Kernel function to compute acceleration based on gravity
__global__ void computeAccelerationKernel(vector3 *accelerationVectors,
                                          vector3 *positions,
                                          vector3 *velocities, double *masses) {
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
          distance[dimension] = positions[entityId][dimension] -
                                positions[otherEntity][dimension];

        double magnitude_sq = distance[0] * distance[0] +
                              distance[1] * distance[1] +
                              distance[2] * distance[2];
        double magnitude = sqrt(magnitude_sq);
        double accelerationMagnitude =
            -1 * GRAV_CONSTANT * masses[otherEntity] / magnitude_sq;
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
      velocities[entityId][dimension] +=
          totalAcceleration[dimension] * INTERVAL;
      positions[entityId][dimension] =
          velocities[entityId][dimension] * INTERVAL;
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
  vector3 *d_accelerationVectors;
  cudaMalloc((void **)&d_accelerationVectors,
             sizeof(vector3) * NUMENTITIES * NUMENTITIES);
  vector3 *d_positions;
  cudaMalloc((void **)&d_positions, sizeof(vector3) * NUMENTITIES);
  vector3 *d_velocities;
  cudaMalloc((void **)&d_velocities, sizeof(vector3) * NUMENTITIES);
  double *d_masses;
  cudaMalloc((void **)&d_masses, sizeof(double) * NUMENTITIES);

  // Copy data from host to device
  cudaMemcpy(d_positions, positions, sizeof(vector3) * NUMENTITIES,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_velocities, velocities, sizeof(vector3) * NUMENTITIES,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_masses, masses, sizeof(double) * NUMENTITIES,
             cudaMemcpyHostToDevice);

  // Setup grid and block dimensions for the kernel
  int blockSize = 256;
  int gridSize = (NUMENTITIES + blockSize - 1) / blockSize;

  // Execute the kernel
  computeAccelerationKernel<<<gridSize, blockSize>>>(
      d_accelerationVectors, d_positions, d_velocities, d_masses);

  // Copy results back to host
  cudaMemcpy(positions, d_positions, sizeof(vector3) * NUMENTITIES,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(velocities, d_velocities, sizeof(vector3) * NUMENTITIES,
             cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_accelerationVectors);
  cudaFree(d_positions);
  cudaFree(d_velocities);
  cudaFree(d_masses);

  // Free host memory
  free(accelerationVectors);
  free(accelerationMatrix);
}
