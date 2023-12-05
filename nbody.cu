#include "compute.h"
#include "config.h"
#include "planets.h"
#include "vector.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Global variables for objects in the system
vector3 *hVel, *d_hVel;
vector3 *hPos, *d_hPos;
double *mass, *d_mass;

void initHostMemory(int numObjects) {
  hVel = (vector3 *)malloc(sizeof(vector3) * numObjects);
  hPos = (vector3 *)malloc(sizeof(vector3) * numObjects);
  mass = (double *)malloc(sizeof(double) * numObjects);

  // Allocate memory on the device
  cudaMalloc(&d_hVel, sizeof(vector3) * numObjects);
  cudaMalloc(&d_hPos, sizeof(vector3) * numObjects);
  cudaMalloc(&d_mass, sizeof(double) * numObjects);
}

void freeHostMemory() {
  free(hVel);
  free(hPos);
  free(mass);

  // Free device memory
  cudaFree(d_hVel);
  cudaFree(d_hPos);
  cudaFree(d_mass);
}

void planetFill() {
  double data[][7] = {SUN,     MERCURY, VENUS,  EARTH,  MARS,
                      JUPITER, SATURN,  URANUS, NEPTUNE};
  for (int i = 0; i <= NUMPLANETS; i++) {
    for (int j = 0; j < 3; j++) {
      hPos[i][j] = data[i][j];
      hVel[i][j] = data[i][j + 3];
    }
    mass[i] = data[i][6];
  }

  // Copy initialized data to the device
  cudaMemcpy(d_hPos, hPos, sizeof(vector3) * (NUMPLANETS + 1),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_hVel, hVel, sizeof(vector3) * (NUMPLANETS + 1),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_mass, mass, sizeof(double) * (NUMPLANETS + 1),
             cudaMemcpyHostToDevice);
}

void randomFill(int start, int count) {
  for (int i = start; i < start + count; i++) {
    for (int j = 0; j < 3; j++) {
      hVel[i][j] = (double)rand() / RAND_MAX * MAX_DISTANCE * 2 - MAX_DISTANCE;
      hPos[i][j] = (double)rand() / RAND_MAX * MAX_VELOCITY * 2 - MAX_VELOCITY;
      mass[i] = (double)rand() / RAND_MAX * MAX_MASS;
    }
  }

  // Copy random data to the device
  cudaMemcpy(&d_hPos[start], &hPos[start], sizeof(vector3) * count,
             cudaMemcpyHostToDevice);
  cudaMemcpy(&d_hVel[start], &hVel[start], sizeof(vector3) * count,
             cudaMemcpyHostToDevice);
  cudaMemcpy(&d_mass[start], &mass[start], sizeof(double) * count,
             cudaMemcpyHostToDevice);
}

void printSystem(FILE *handle) {
  // Ensure data is up-to-date from the device
  cudaMemcpy(hPos, d_hPos, sizeof(vector3) * NUMENTITIES,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(hVel, d_hVel, sizeof(vector3) * NUMENTITIES,
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < NUMENTITIES; i++) {
    fprintf(handle, "pos=(");
    for (int j = 0; j < 3; j++) {
      fprintf(handle, "%lf,", hPos[i][j]);
    }
    fprintf(handle, "),v=(");
    for (int j = 0; j < 3; j++) {
      fprintf(handle, "%lf,", hVel[i][j]);
    }
    fprintf(handle, "),m=%lf\n", mass[i]);
  }
}

int main(int argc, char **argv) {
  clock_t t0 = clock();
  srand(1234); // Fixed seed for reproducibility
  initHostMemory(NUMENTITIES);
  planetFill();
  randomFill(NUMPLANETS + 1, NUMASTEROIDS);

#ifdef DEBUG
  printSystem(stdout);
#endif

  for (int t_now = 0; t_now < DURATION; t_now += INTERVAL) {
    compute(d_hPos, d_hVel, d_mass);
  }

  clock_t t1 = clock() - t0;

#ifdef DEBUG
  printSystem(stdout);
#endif

  printf("This took a total time of %f seconds\n", (double)t1 / CLOCKS_PER_SEC);
  freeHostMemory();
}
