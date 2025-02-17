#include "compute.h"
#include "config.h"
#include "planets.h"
#include "vector.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

vector3 *hVel;
vector3 *hPos;
double *mass;

vector3 *d_values, **d_accels, *d_hPos, *d_hVel, *d_accel_sum;
double *d_mass;
vector3 *temp[NUMENTITIES];

void initHostMemory(int numObjects) {
  hVel = (vector3 *)malloc(sizeof(vector3) * numObjects);
  hPos = (vector3 *)malloc(sizeof(vector3) * numObjects);
  mass = (double *)malloc(sizeof(double) * numObjects);
}

void freeHostMemory() {
  free(hVel);
  free(hPos);
  free(mass);
}

void freeDeviceMemory() {
  for (int i = 0; i < NUMENTITIES; i++) {
    cudaFree(temp[i]);
  }
  cudaFree(d_accels);
  cudaFree(d_hPos);
  cudaFree(d_hVel);
  cudaFree(d_accel_sum);
  cudaFree(d_mass);
}

// planetFill: Fill the first NUMPLANETS+1 entries of the entity arrays with an
// estimation 				of our solar system (Sun+NUMPLANETS)
// Parameters: None Returns: None Fills the first 8 entries of our system with
// an estimation of the sun plus our 8 planets.
void planetFill() {
  int i, j;
  double data[][7] = {SUN,     MERCURY, VENUS,  EARTH,  MARS,
                      JUPITER, SATURN,  URANUS, NEPTUNE};
  for (i = 0; i <= NUMPLANETS; i++) {
    for (j = 0; j < 3; j++) {
      hPos[i][j] = data[i][j];
      hVel[i][j] = data[i][j + 3];
    }
    mass[i] = data[i][6];
  }
}

// randomFill: FIll the rest of the objects in the system randomly starting at
// some entry in the list Parameters: 	start: The index of the first open entry
// in our system (after planetFill). 				count: The
// number of random objects to put
// into our system Returns: None Side Effects: Fills count entries in our system
// starting at index start (0 based)
void randomFill(int start, int count) {
  int i, j = start;
  for (i = start; i < start + count; i++) {
    for (j = 0; j < 3; j++) {
      hVel[i][j] = (double)rand() / RAND_MAX * MAX_DISTANCE * 2 - MAX_DISTANCE;
      hPos[i][j] = (double)rand() / RAND_MAX * MAX_VELOCITY * 2 - MAX_VELOCITY;
      mass[i] = (double)rand() / RAND_MAX * MAX_MASS;
    }
  }
}

// printSystem: Prints out the entire system to the supplied file
// Parameters: 	handle: A handle to an open file with write access to prnt the
// data to Returns: 		none Side Effects: Modifies the file handle by
// writing to it.
void printSystem(FILE *handle) {
  int i, j;
  for (i = 0; i < NUMENTITIES; i++) {
    fprintf(handle, "pos=(");
    for (j = 0; j < 3; j++) {
      fprintf(handle, "%lf,", hPos[i][j]);
    }
    printf("),v=(");
    for (j = 0; j < 3; j++) {
      fprintf(handle, "%lf,", hVel[i][j]);
    }
    fprintf(handle, "),m=%lf\n", mass[i]);
  }
}

int main(int argc, char **argv) {
  clock_t t0 = clock();
  int t_now;
  // srand(time(NULL));
  srand(1234);
  initHostMemory(NUMENTITIES);
  planetFill();
  randomFill(NUMPLANETS + 1, NUMASTEROIDS);
// now we have a system.
#ifdef DEBUG
  printSystem(stdout);
#endif

  // Allocate memory on the GPU for the 2D array of acceleration vectors
  cudaMalloc((void ***)&d_accels, (NUMENTITIES) * sizeof(vector3 *));

  // Allocate memory for each row of the 2D acceleration vector array
  for (int i = 0; i < NUMENTITIES; i++) {
    cudaMalloc(&temp[i], sizeof(vector3) * NUMENTITIES);
  }

  // Copy the pointers of each row (stored in temp) to the GPU memory (d_accels)
  cudaMemcpy(d_accels, temp, sizeof(vector3 *) * NUMENTITIES,
             cudaMemcpyHostToDevice);

  // Allocate memory on the GPU for the positions (d_hPos) of each entity
  cudaMalloc((void **)&d_hPos, (NUMENTITIES) * sizeof(vector3));

  // Allocate memory on the GPU for the velocities (d_hVel) of each entity
  cudaMalloc((void **)&d_hVel, (NUMENTITIES) * sizeof(vector3));

  // Allocate memory on the GPU for the sum of accelerations (d_accel_sum) for
  // each entity
  cudaMalloc((void **)&d_accel_sum, (NUMENTITIES) * sizeof(vector3));

  // Allocate memory on the GPU for the mass (d_mass) of each entity
  cudaMalloc((void **)&d_mass, (NUMENTITIES) * sizeof(double));

  // Copy the positions from host memory (hPos) to device memory (d_hPos)
  cudaMemcpy(d_hPos, hPos, (NUMENTITIES) * sizeof(vector3),
             cudaMemcpyHostToDevice);

  // Copy the velocities from host memory (hVel) to device memory (d_hVel)
  cudaMemcpy(d_hVel, hVel, (NUMENTITIES) * sizeof(vector3),
             cudaMemcpyHostToDevice);

  // Copy the masses from host memory (mass) to device memory (d_mass)
  cudaMemcpy(d_mass, mass, (NUMENTITIES) * sizeof(double),
             cudaMemcpyHostToDevice);

  // Run the simulation for the specified duration
  for (t_now = 0; t_now < DURATION; t_now += INTERVAL) {
    compute();
  }

  // Copy the updated positions back to host memory from device memory
  cudaMemcpy(hPos, d_hPos, (NUMENTITIES) * sizeof(vector3),
             cudaMemcpyDeviceToHost);

  // Copy the updated velocities back to host memory from device memory
  cudaMemcpy(hVel, d_hVel, (NUMENTITIES) * sizeof(vector3),
             cudaMemcpyDeviceToHost);

  clock_t t1 = clock() - t0;
#ifdef DEBUG
  printSystem(stdout);
#endif
  printf("This took a total time of %f seconds\n", (double)t1 / CLOCKS_PER_SEC);

  freeHostMemory();
  freeDeviceMemory();
}