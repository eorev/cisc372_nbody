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

  accelComps[threadIndex] = &interValues[threadIndex * NUMENTITIES];

  if (threadIndex <
      NUMENTITIES * NUMENTITIES) { // Check if thread is within bounds
    if (idxI == idxJ) {
      FILL_VECTOR(accelComps[idxI][idxJ], 0, 0, 0);
    } else {
      vector3 distVec;

      // Calculate distance between two entities
      distVec[0] = devicePos[idxI][0] - devicePos[idxJ][0];
      distVec[1] = devicePos[idxI][1] - devicePos[idxJ][1];
      distVec[2] = devicePos[idxI][2] - devicePos[idxJ][2];
      double magSq = distVec[0] * distVec[0] + distVec[1] * distVec[1] +
                     distVec[2] * distVec[2];
      double magnitude = sqrt(magSq);
      double accelMagnitude = -GRAV_CONSTANT * deviceMass[idxJ] / magSq;

      // Calculate acceleration vector
      FILL_VECTOR(accelComps[idxI][idxJ],
                  accelMagnitude * distVec[0] / magnitude,
                  accelMagnitude * distVec[1] / magnitude,
                  accelMagnitude * distVec[2] / magnitude);
    }

    // Sum accelerations for the current entity
    vector3 totalAccel = {accelComps[threadIndex][0],
                          accelComps[threadIndex][1],
                          accelComps[threadIndex][2]};

    // Update velocity and position based on total acceleration
    deviceVel[idxI][0] += totalAccel[0] * INTERVAL;
    devicePos[idxI][0] = deviceVel[idxI][0] * INTERVAL;

    deviceVel[idxI][1] += totalAccel[1] * INTERVAL;
    devicePos[idxI][1] = deviceVel[idxI][1] * INTERVAL;

    deviceVel[idxI][2] += totalAccel[2] * INTERVAL;
    devicePos[idxI][2] = deviceVel[idxI][2] * INTERVAL;
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
  cudaMallocManaged(&accelerationComponents, sizeof(vector3 *) * NUMENTITIES);

  // Configuration for parallel execution
  int blockSize = 256;
  int numBlocks = (NUMENTITIES + blockSize - 1) / blockSize;

  // Launching the CUDA kernel
  computeKernel<<<numBlocks, blockSize>>>(
      intermediateValues, accelerationComponents, deviceVelocities,
      devicePositions, deviceMasses);
  cudaDeviceSynchronize();

  // Copy results back to host memory
  cudaMemcpy(hVel, deviceVelocities, sizeof(vector3) * NUMENTITIES,
             cudaMemcpyDefault);
  cudaMemcpy(hPos, devicePositions, sizeof(vector3) * NUMENTITIES,
             cudaMemcpyDefault);
  cudaMemcpy(mass, deviceMasses, sizeof(double) * NUMENTITIES,
             cudaMemcpyDefault);

  // Freeing allocated memory on GPU
  cudaFree(accelerationComponents);
  cudaFree(intermediateValues);
}
