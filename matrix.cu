/*
 * Ejercicio 3: Producto de matrices
 */

const int N = 8;
const int COLS = N;
const int ROWS = N;
const int ARR_BYTES = ROWS * COLS * sizeof(float);
const int NUM_BLOCKS = 16;

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

/* Utilidad para checar errores de CUDA */
void checkCUDAError(const char*);

__global__ void multiplication(float *a, float *b, float *c, int width) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  float sum = 0;

  if (row < width && col < width){
    for(int i = 0; i < width; i++)
      sum += a[row * width + i] * b[i * width + col];
  }

  c[row * width + col] = sum;
}

void fill(float* matrix, bool withZeros) {
  srand(time(0));
  int r = withZeros ? 0 : rand() % 1024;
  printf("Random number: %d\n", r);
  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLS; j++) {
      matrix[i * COLS + j] = (float)r;
    }
  }
}

int main() {
  float *h_a, *d_a;
  float *h_b, *d_b;
  float *h_c, *d_c;

  //Reserva espacio para arreglos 
	h_a = (float*)malloc(ROWS * COLS * sizeof(float));
	h_b = (float*)malloc(ROWS * COLS * sizeof(float));
	h_c = (float*)malloc(ROWS * COLS * sizeof(float));
	cudaMalloc((void**) &d_a, ARR_BYTES);
	cudaMalloc((void**) &d_b, ARR_BYTES);
	cudaMalloc((void**) &d_c, ARR_BYTES);

  fill(h_a, false);
  fill(h_b, false);
  fill(h_c, true);

  //Transfiere arreglo a device
  cudaMemcpy(d_a, h_a, ARR_BYTES, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, ARR_BYTES, cudaMemcpyHostToDevice);

  //Lanza el kernel
  // Si N < NUM_BLOCKS => Solo es necesario tener un grid (1, 1)
  // Si N > NUM_BLOCKS => encontrar cuantos Bloques requiero para abarcar
  //    toda la matriz
  unsigned int grid_rows = (N + NUM_BLOCKS - 1) / NUM_BLOCKS;
  unsigned int grid_cols = (N + NUM_BLOCKS - 1) / NUM_BLOCKS;
  dim3 dimGrid(grid_cols, grid_rows);
  dim3 dimBlock(NUM_BLOCKS, NUM_BLOCKS);
  multiplication<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, ROWS);

	/* Esperar a que todos los threads acaben y checar por errores */
  cudaThreadSynchronize();
  checkCUDAError("kernel invocation");

	//Toma el resultado
  cudaMemcpy(h_c, d_c, ARR_BYTES, cudaMemcpyDeviceToHost);
  checkCUDAError("memcpy");

  /* print out the result */
  printf("Results: \n");
  for (int i = 0; i < ROWS; i++) {
    for (int j = 0; j < COLS; j++) {
      printf("%.2f,\t", h_c[i + j]);
    }
    printf("\n");
  }
  printf("\n\n");

  /* Parte 1D: Liberar los arreglos */
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  free(h_a);
  free(h_b);
  free(h_c);

  return 0;
}

/* Utility function to check for and report CUDA errors */
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}
