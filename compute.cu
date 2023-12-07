#include "config.h"
#include "cuda.h"
#include "vector.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdlib.h>

// Values and Accels are Global Variables
vector3 *values;
vector3 **accels;

// The compute functions done parallel
__global__ void parallelCompute(vector3 *values, vector3 **accels,
                                vector3 *d_vel, vector3 *d_pos,
                                double *d_mass) {

  int currThreadId = blockIdx.x * blockDim.x + threadIdx.x;
  int i = currThreadId /
          NUMENTITIES; // define i in terms of what block/dimension/thread we
                       // are currently using
  int j = currThreadId %
          NUMENTITIES; // define j in terms of what block/dimension/thread we
                       // are currently using

  accels[currThreadId] = &values[currThreadId * NUMENTITIES];

  if (currThreadId <
      NUMENTITIES * NUMENTITIES) { // Ensure that the result of our block is
                                   // actually in bounds
    if (i == j) {
      FILL_VECTOR(accels[i][j], 0, 0, 0);
    } else {
      vector3 distance;

      distance[0] = d_pos[i][0] - d_pos[j][0];
      distance[1] = d_pos[i][1] - d_pos[j][1];
      distance[2] = d_pos[i][2] - d_pos[j][2];
      double magnitude_sq = distance[0] * distance[0] +
                            distance[1] * distance[1] +
                            distance[2] * distance[2];
      double magnitude = sqrt(magnitude_sq);
      double accelmag = -1 * GRAV_CONSTANT * d_mass[j] / magnitude_sq;
      FILL_VECTOR(accels[i][j], accelmag * distance[0] / magnitude,
                  accelmag * distance[1] / magnitude,
                  accelmag * distance[2] / magnitude);
    }

    vector3 accel_sum = {(double)*(accels[currThreadId])[0],
                         (double)*(accels[currThreadId])[1],
                         (double)*(accels[currThreadId])[2]};

    d_vel[i][0] += accel_sum[0] * INTERVAL;
    d_pos[i][0] = d_vel[i][0] * INTERVAL;

    d_vel[i][1] += accel_sum[1] * INTERVAL;
    d_pos[i][1] = d_vel[i][1] * INTERVAL;

    d_vel[i][2] += accel_sum[2] * INTERVAL;
    d_pos[i][2] = d_vel[i][2] * INTERVAL;
  }
}

// Run the three functions that implemenmt parallel design
void compute() {

  // d_hvel and d_hpos hold the hVel and hPos variables on the GPU
  vector3 *d_vel, *d_pos;
  double *d_mass;

  cudaMallocManaged((void **)&d_vel, (sizeof(vector3) * NUMENTITIES));
  cudaMallocManaged((void **)&d_pos, (sizeof(vector3) * NUMENTITIES));
  cudaMallocManaged((void **)&d_mass, (sizeof(double) * NUMENTITIES));

  // Copy memory from the host onto the GPU
  cudaMemcpy(d_vel, hVel, sizeof(vector3) * NUMENTITIES,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_pos, hPos, sizeof(vector3) * NUMENTITIES,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_mass, mass, sizeof(double) * NUMENTITIES,
             cudaMemcpyHostToDevice);

  // Allocate space on the GPU for these variables
  cudaMallocManaged((void **)&values,
                    sizeof(vector3) * NUMENTITIES * NUMENTITIES);
  cudaMallocManaged((void **)&accels, sizeof(vector3 *) * NUMENTITIES);

  // Determine number of blocks that we should be running
  int blockSize = 256;
  int numBlocks = (NUMENTITIES + blockSize - 1) / blockSize;

  parallelCompute<<<numBlocks, blockSize>>>(values, accels, d_vel, d_pos,
                                            d_mass);
  cudaDeviceSynchronize();

  // Copy the results back to the device
  cudaMemcpy(hVel, d_vel, sizeof(vector3) * NUMENTITIES, cudaMemcpyDefault);
  cudaMemcpy(hPos, d_pos, sizeof(vector3) * NUMENTITIES, cudaMemcpyDefault);
  cudaMemcpy(mass, d_mass, sizeof(double) * NUMENTITIES, cudaMemcpyDefault);

  cudaFree(accels);
  cudaFree(values);
}