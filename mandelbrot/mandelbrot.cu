/*
 *  Ejercicio 4: Área del conjunto de Mandelbrot
 */

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

# define NPOINTS 2000
// # define NPOINTS 10
# define MAXITER 2000
# define THREADS_PER_BLOCK 256
# define NUM_BLOCKS 16

struct complex{
  double real;
  double imag;
};

const int ARR_BYTES = NPOINTS * NPOINTS * sizeof(struct complex);

__global__ void initComplex(struct complex *z, struct complex *c, int width) {
  int row = threadIdx.x + (blockIdx.x * blockDim.x);
  int col = threadIdx.y + (blockIdx.y* blockDim.y);

  if (row < width && col < width) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    // printf("(%d,%d)\n", i, j);
    c[row * width + col].real = -2.0 + 2.5 * (double)(i) / (double)(width) + 1.0e-7;
    c[row * width + col].imag = 1.125 * (double)(j) / (double)(width) + 1.0e-7;
    z[row * width + col] = c[row * width + col];
  }
}

__global__ void check(struct complex *z, struct complex *c, int *res, int width) {
  int row = threadIdx.x + (threadIdx.x + blockDim.x);
  int col = threadIdx.y + (threadIdx.y + blockDim.y);
  if (row < width && col < width) {
    int i = 0;
    double zReal, zImag, cReal, cImag;
    double zToCheck;
    zToCheck = z[row * width + col].real * z[row * width + col].real;
    zToCheck += z[row * width + col].imag * z[row * width + col].imag;
    while (i < MAXITER && zToCheck <= 4.0e0) {
      zReal = z[row * width + col].real;
      zImag = z[row * width + col].imag;
      cReal = c[row * width + col].real;
      cImag = c[row * width + col].imag;
      double ztemp = (zReal * zReal) - (zImag * zImag) + cReal;
      z[row * width + col].imag = zReal * zImag * 2 + cImag;;
      z[row * width + col].real = ztemp;
      zToCheck = z[row * width + col].real * z[row * width + col].real + z[row * width + col].imag * z[row * width + col].imag;
      if (zToCheck > 4.0e0) {
        *res++; 
      }
      i++;
    }
  }
}

int main() {
  double area, error;

  struct complex *h_z, *h_c;
  struct complex *d_z, *d_c;
  int *h_res, *d_res;
  

  h_z = (struct complex *)malloc(NPOINTS * NPOINTS * sizeof(struct complex));
  h_c = (struct complex *)malloc(NPOINTS * NPOINTS * sizeof(struct complex));
  h_res = (int *) malloc(sizeof(int));
  cudaMalloc((void **) &d_z, ARR_BYTES);
  cudaMalloc((void **) &d_c, ARR_BYTES);
  cudaMalloc((void **) &d_res, sizeof(int));

  h_res = 0;

  cudaMemcpy(d_z, h_z, ARR_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c, ARR_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_res, h_res, sizeof(int), cudaMemcpyHostToDevice);

  int threads = THREADS_PER_BLOCK / NUM_BLOCKS;
  int grids = (NPOINTS + NUM_BLOCKS - 1) / NUM_BLOCKS;
  printf("Blocks per Grids: %d\n", grids);
  printf("Threads per block: %d\n", threads);
  dim3 dimGrid(grids, grids);
  dim3 dimBlock(threads, threads);
  initComplex<<<dimGrid, dimBlock>>>(d_z, d_c, NPOINTS);
  check<<<dimGrid, dimBlock>>>(d_z, d_c, d_res, NPOINTS);

  cudaThreadSynchronize();

  cudaMemcpy(h_z, d_z, ARR_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_c, d_c, ARR_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_res, d_res, sizeof(int), cudaMemcpyDeviceToHost);


  cudaFree(d_z);
  cudaFree(d_c);
  free(h_z);
  free(h_c);

  int numoutside = *((int*)(&h_res));

  area = 2.0 * 2.5 * 1.125 * (double)( NPOINTS * NPOINTS - numoutside) / (double)(NPOINTS * NPOINTS);
  error = area / (double)NPOINTS;

  printf("Area of Mandlebrot set = %12.8f +/- %12.8f\n",area,error);

  return 0;
}
