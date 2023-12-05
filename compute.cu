#include "config.h"
#include "cuda.h"
#include "vector.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

// Arrays for intermediate calculations
vector3 *intermediateValues;
vector3 **accelerationComponents;

// CUDA kernel for parallel computations
__global__ void computeKernel(vector3 *interValues, vector3 **accelComps,
                              vector3 *deviceVel, vector3 *devicePos,
                              double *deviceMass) {

  int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
  int idxI =
      threadIndex / NUMENTITIES; // Calculate i index based on current thread
  int idxJ =
      threadIndex % NUMENTITIES; // Calculate j index based on current thread

  if (threadIndex <
      NUMENTITIES * NUMENTITIES) { // Check if thread is within bounds
    if (idxI == idxJ) {
      FILL_VECTOR(interValues[threadIndex], 0, 0, 0);
    } else {
      vector3 distance;

      // Calculate distance between two entities
      distance[0] = devicePos[idxI][0] - devicePos[idxJ][0];
      distance[1] = devicePos[idxI][1] - devicePos[idxJ][1];
      distance[2] = devicePos[idxI][2] - devicePos[idxJ][2];
      double magnitude_sq = distance[0] * distance[0] +
                            distance[1] * distance[1] +
                            distance[2] * distance[2];
      double magnitude = sqrt(magnitude_sq);
      double accelmag = -GRAV_CONSTANT * deviceMass[idxJ] / magnitude_sq;

      // Calculate acceleration vector
      interValues[threadIndex][0] = accelmag * distance[0] / magnitude;
      interValues[threadIndex][1] = accelmag * distance[1] / magnitude;
      interValues[threadIndex][2] = accelmag * distance[2] / magnitude;
    }
  }

  // Sum accelerations for the current entity
  if (idxI < NUMENTITIES) {
    vector3 totalAccel = {0, 0, 0};
    for (int j = 0; j < NUMENTITIES; j++) {
      totalAccel[0] += interValues[idxI * NUMENTITIES + j][0];
      totalAccel[1] += interValues[idxI * NUMENTITIES + j][1];
      totalAccel[2] += interValues[idxI * NUMENTITIES + j][2];
    }

    // Update velocity and position based on total acceleration
    deviceVel[idxI][0] += totalAccel[0] * INTERVAL;
    devicePos[idxI][0] += deviceVel[idxI][0] * INTERVAL;

    deviceVel[idxI][1] += totalAccel[1] * INTERVAL;
    devicePos[idxI][1] += deviceVel[idxI][1] * INTERVAL;

    deviceVel[idxI][2] += totalAccel[2] * INTERVAL;
    devicePos[idxI][2] += deviceVel[idxI][2] * INTERVAL;
  }
}

// Function to execute the parallel compute operations
void compute() {
  vector3 *deviceVelocities, *devicePositions;
  double *deviceMasses;

  // Memory allocations on the GPU
  cudaMallocManaged(&deviceVelocities, sizeof(vector3) * NUMENTITIES);
  cudaMallocManaged(&devicePositions, sizeof(vector3) * NUMENTITIES);
  cudaMallocManaged(&deviceMasses, sizeof(double) * NUMENTITIES);

  // Copy data from host to device
  cudaMemcpy(deviceVelocities, hVel, sizeof(vector3) * NUMENTITIES,
             cudaMemcpyHostToDevice);
  cudaMemcpy(devicePositions, hPos, sizeof(vector3) * NUMENTITIES,
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceMasses, mass, sizeof(double) * NUMENTITIES,
             cudaMemcpyHostToDevice);

  // Allocate space for intermediate calculations
  cudaMallocManaged(&intermediateValues,
                    sizeof(vector3) * NUMENTITIES * NUMENTITIES);

  // Configuration for parallel execution
  int blockSize = 256;
  int numBlocks = (NUMENTITIES * NUMENTITIES + blockSize - 1) / blockSize;

  // Launching the CUDA kernel
  computeKernel<<<numBlocks, blockSize>>>(
      intermediateValues, accelerationComponents, deviceVelocities,
      devicePositions, deviceMasses);
  cudaDeviceSynchronize();

  // Copy results back to host memory
  cudaMemcpy(hVel, deviceVelocities, sizeof(vector3) * NUMENTITIES,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(hPos, devicePositions, sizeof(vector3) * NUMENTITIES,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(mass, deviceMasses, sizeof(double) * NUMENTITIES,
             cudaMemcpyDeviceToHost);

  // Freeing allocated memory on GPU
  cudaFree(intermediateValues);
}
